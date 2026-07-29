import torch
import torch.nn as nn
from typing import Dict, List, Optional

from adni_pipeline.models.encoders import build_adni_encoders
from adni_pipeline.models.fusion import MultiheadAdditiveAttentionFusion
from adni_pipeline.models.classifier import MLP


class ADNIPmcmFLModel(nn.Module):
    def __init__(self, encoders: nn.ModuleDict, fusion: nn.Module,
                 classifier: nn.Module, modalities: List[str],
                 num_patches: int, enc_out_dim: int):
        super().__init__()
        self.encoders = encoders
        self.fusion = fusion
        self.classifier = classifier
        self.modalities = modalities
        self.num_patches = num_patches
        self.enc_out_dim = enc_out_dim

    def forward(self, X: Dict[str, torch.Tensor], observed: torch.Tensor,
                labels: Optional[torch.Tensor], prototypes: dict,
                return_embeds: bool = False, mode: str = "train"):
        """
        Args:
            X: dict mapping modality name -> (B, *input_shape) tensor.
            observed: (B, M) bool tensor.
            labels: (B,) class indices (None during inference / prototype update).
            prototypes: dict {mod_name: (C, enc_out_dim), "fusion": (C, fusion_dim)}.
            return_embeds: if True, return unimodal embeddings and fusion output.
            mode: 'train' or 'test'.

        Returns (depending on return_embeds):
            return_embeds=True:  (logits, unimodal_embeds, fusion_out, valid_mask)
            return_embeds=False: (logits, valid_mask)
        """
        device = observed.device
        B = observed.shape[0]
        num_mods = len(self.modalities)

        fusion_in = []
        mod_valid_lens = []
        unimodal_embeds = []

        for mod_idx, mod_name in enumerate(self.modalities):
            m_observed = observed[:, mod_idx]

            m_feats = torch.zeros((B, self.num_patches, self.enc_out_dim), device=device)
            if m_observed.sum() > 0:
                m_feats[m_observed] = self.encoders[mod_name](X[mod_name][m_observed])

            unimodal_embeds.append(m_feats.mean(dim=1))  # (B, enc_out_dim)

            valid_len = torch.zeros(B, dtype=torch.long, device=device)
            valid_len[m_observed] = self.num_patches
            mod_valid_lens.append(valid_len)

            fusion_in.append(m_feats)

        missing_mask = ~observed  # (B, M) True where modality is missing

        if labels is not None:
            self._replace_missing_train(
                fusion_in, mod_valid_lens, missing_mask, labels, prototypes,
            )
        elif mode == "test":
            closest_classes = self._get_closest_class_per_sample(
                unimodal_embeds, prototypes, observed,
            )
            self._replace_missing_train(
                fusion_in, mod_valid_lens, missing_mask, closest_classes, prototypes,
            )

        mod_mask = observed.float()  # (B, M)
        valid_mask = mod_mask.sum(dim=1) > 0  # (B,)

        fusion_in_valid = [f[valid_mask] for f in fusion_in]
        fused_input = torch.cat(fusion_in_valid, dim=1)
        mod_lens = [self.num_patches] * num_mods
        valid_lens_filtered = [vl[valid_mask] for vl in mod_valid_lens]
        fusion_out, _ = self.fusion(fused_input, modality_lens=mod_lens,
                                    valid_lens=valid_lens_filtered)
        logits = self.classifier(fusion_out)

        if return_embeds:
            unimodal_embeds_valid = [e[valid_mask] for e in unimodal_embeds]
            return logits, unimodal_embeds_valid, fusion_out, valid_mask
        return logits, valid_mask

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _replace_missing_train(self, fusion_in, mod_valid_lens, missing_mask,
                               class_indices, prototypes):
        """Replace zero patches for missing modalities with expanded prototypes."""
        device = missing_mask.device
        for mod_idx, mod_name in enumerate(self.modalities):
            missing_samples = missing_mask[:, mod_idx]
            if not missing_samples.any():
                continue
            if mod_name not in prototypes:
                continue

            proto = prototypes[mod_name].to(device)  # (C, enc_out_dim)
            sample_protos = proto[class_indices[missing_samples]]  # (n_miss, enc_out_dim)
            expanded = sample_protos.unsqueeze(1).expand(
                -1, self.num_patches, -1,
            )  # (n_miss, num_patches, enc_out_dim)
            fusion_in[mod_idx][missing_samples] = expanded
            mod_valid_lens[mod_idx][missing_samples] = self.num_patches

    def _get_closest_class_per_sample(self, unimodal_embeds, prototypes,
                                      observed):
        """Find nearest class for each sample using available modality embeddings."""
        B = observed.shape[0]
        device = observed.device
        closest_classes = torch.zeros(B, dtype=torch.long, device=device)

        for b in range(B):
            avail_mods = torch.where(observed[b])[0]
            if avail_mods.numel() == 0:
                continue

            avail_embs = torch.stack(
                [unimodal_embeds[m][b] for m in avail_mods.tolist()],
            )
            avg_emb = avail_embs.mean(dim=0)  # (enc_out_dim,)

            avail_protos = torch.stack(
                [prototypes[self.modalities[m]].to(device)
                 for m in avail_mods.tolist()],
            )
            avg_proto = avail_protos.mean(dim=0)  # (C, enc_out_dim)

            distances = torch.norm(avg_emb.unsqueeze(0) - avg_proto, dim=1)
            closest_classes[b] = distances.argmin()

        return closest_classes


def build_adni_pmcmfl_model(dataset_cfg: dict) -> ADNIPmcmFLModel:
    """Construct a full ADNIPmcmFLModel from a dataset config dict."""
    encoders = build_adni_encoders(dataset_cfg)

    enc_out_dim = dataset_cfg["enc_out_dim"]
    num_patches = dataset_cfg["num_patches"]
    fusion_cfg = dataset_cfg["fusion_configs"]
    cls_cfg = dataset_cfg["classifier_configs"]

    fusion = MultiheadAdditiveAttentionFusion(
        input_dim=enc_out_dim,
        hid_dim=fusion_cfg["hid_dim"],
        num_heads=fusion_cfg["num_heads"],
    )

    classifier_input_dim = fusion_cfg["num_heads"] * enc_out_dim
    classifier = MLP(
        input_dim=classifier_input_dim,
        hidden_dim=cls_cfg["hid_dim"],
        num_hid_layers=cls_cfg["num_hid_layers"],
        output_dim=dataset_cfg["num_classes"],
    )

    return ADNIPmcmFLModel(
        encoders=encoders,
        fusion=fusion,
        classifier=classifier,
        modalities=dataset_cfg["modalities"],
        num_patches=num_patches,
        enc_out_dim=enc_out_dim,
    )
