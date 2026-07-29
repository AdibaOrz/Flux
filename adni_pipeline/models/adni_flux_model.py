import torch
import torch.nn as nn
from typing import Dict, List

from adni_pipeline.models.classifier import MLP
from adni_pipeline.models.encoders import build_adni_encoders
from adni_pipeline.models.flux_components import ConfidenceHead
from adni_pipeline.models.fusion import MultiheadAdditiveAttentionFusion


class ADNIFluxModel(nn.Module):
    def __init__(self, encoders: nn.ModuleDict, fusion: nn.Module,
                 classifier: nn.Module,
                 unimodal_confidence_heads: nn.ModuleDict,
                 modalities: List[str], num_patches: int, enc_out_dim: int):
        super().__init__()
        self.encoders = encoders
        self.fusion = fusion
        self.classifier = classifier
        self.unimodal_confidence_heads = unimodal_confidence_heads
        self.modalities = modalities
        self.num_patches = num_patches
        self.enc_out_dim = enc_out_dim

    def forward(self, X: Dict[str, torch.Tensor], observed: torch.Tensor):
        device = observed.device
        B = observed.shape[0]
        num_mods = len(self.modalities)
        num_classes = self.classifier.net[-1].out_features

        fusion_in = []
        mod_valid_lens = []
        pooled_embeddings = []

        for mod_idx, mod_name in enumerate(self.modalities):
            m_observed = observed[:, mod_idx]

            m_feats = torch.zeros((B, self.num_patches, self.enc_out_dim), device=device)
            if m_observed.sum() > 0:
                m_feats[m_observed] = self.encoders[mod_name](X[mod_name][m_observed])

            fusion_in.append(m_feats)

            valid_len = torch.zeros(B, dtype=torch.long, device=device)
            valid_len[m_observed] = self.num_patches
            mod_valid_lens.append(valid_len)

            pooled_embeddings.append(m_feats.mean(dim=1))

        mod_mask = observed.float()
        valid_mask = mod_mask.sum(dim=1) > 0

        unimodal_logits_list = []
        unimodal_logvars_list = []
        B_valid = int(valid_mask.sum().item())

        for mod_idx, mod_name in enumerate(self.modalities):
            m_embed = pooled_embeddings[mod_idx][valid_mask]
            m_embed_detached = m_embed.detach()
            m_present = mod_mask[valid_mask][:, mod_idx]

            logits = torch.zeros(B_valid, num_classes, device=device)
            log_var = torch.zeros(B_valid, num_classes, device=device)
            if m_present.sum() > 0:
                keep = m_present == 1
                l, lv = self.unimodal_confidence_heads[mod_name](m_embed_detached[keep])
                logits[keep] = l
                log_var[keep] = lv

            unimodal_logits_list.append(logits)
            unimodal_logvars_list.append(log_var)

        unimodal_logits = torch.stack(unimodal_logits_list, dim=0)
        unimodal_logvars = torch.stack(unimodal_logvars_list, dim=0)

        fusion_in_valid = [f[valid_mask] for f in fusion_in]
        fused_input = torch.cat(fusion_in_valid, dim=1)
        mod_lens = [self.num_patches] * num_mods
        valid_lens_filtered = [vl[valid_mask] for vl in mod_valid_lens]
        fusion_out, _ = self.fusion(fused_input, modality_lens=mod_lens,
                                    valid_lens=valid_lens_filtered)
        mm_logits = self.classifier(fusion_out)
        mod_mask = mod_mask[valid_mask]

        return mm_logits, unimodal_logits, fusion_out, unimodal_logvars, mod_mask, valid_mask


def build_adni_flux_model(dataset_cfg: dict) -> ADNIFluxModel:
    encoders = build_adni_encoders(dataset_cfg)

    enc_out_dim = dataset_cfg["enc_out_dim"]
    num_patches = dataset_cfg["num_patches"]
    fusion_cfg = dataset_cfg["fusion_configs"]
    cls_cfg = dataset_cfg["classifier_configs"]
    modalities = dataset_cfg["modalities"]
    num_classes = dataset_cfg["num_classes"]
    flux_hidden_dim = dataset_cfg.get("flux_hidden_dim", 64)

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
        output_dim=num_classes,
    )

    unimodal_confidence_heads = nn.ModuleDict({
        mod: ConfidenceHead(
            input_dim=enc_out_dim,
            num_classes=num_classes,
            hidden_dim=flux_hidden_dim,
        )
        for mod in modalities
    })

    return ADNIFluxModel(
        encoders=encoders,
        fusion=fusion,
        classifier=classifier,
        unimodal_confidence_heads=unimodal_confidence_heads,
        modalities=modalities,
        num_patches=num_patches,
        enc_out_dim=enc_out_dim,
    )
