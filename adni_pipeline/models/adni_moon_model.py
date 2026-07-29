import torch
import torch.nn as nn
from typing import Dict, List

from adni_pipeline.models.encoders import build_adni_encoders
from adni_pipeline.models.fusion import MultiheadAdditiveAttentionFusion
from adni_pipeline.models.classifier import MLP
from adni_pipeline.models.moon_components import ProjectionHead


class ADNIMoonModel(nn.Module):
    """ADNI multimodal model with a MOON projection head on fusion output."""

    def __init__(self, encoders: nn.ModuleDict, fusion: nn.Module,
                 classifier: nn.Module, moon_prj_head: nn.Module,
                 modalities: List[str], num_patches: int, enc_out_dim: int):
        super().__init__()
        self.encoders = encoders
        self.fusion = fusion
        self.classifier = classifier
        self.moon_prj_head = moon_prj_head
        self.modalities = modalities
        self.num_patches = num_patches
        self.enc_out_dim = enc_out_dim

    def forward(self, X: Dict[str, torch.Tensor], observed: torch.Tensor,
                return_projection_output: bool = False):
        device = observed.device
        B = observed.shape[0]

        fusion_in = []
        mod_valid_lens = []

        for mod_idx, mod_name in enumerate(self.modalities):
            m_observed = observed[:, mod_idx]
            m_feats = torch.zeros((B, self.num_patches, self.enc_out_dim), device=device)
            if m_observed.sum() > 0:
                m_feats[m_observed] = self.encoders[mod_name](X[mod_name][m_observed])
            fusion_in.append(m_feats)

            valid_len = torch.zeros(B, dtype=torch.long, device=device)
            valid_len[m_observed] = self.num_patches
            mod_valid_lens.append(valid_len)

        fused_input = torch.cat(fusion_in, dim=1)
        mod_lens = [self.num_patches] * len(self.modalities)
        fusion_out, att_weights = self.fusion(fused_input, modality_lens=mod_lens, valid_lens=mod_valid_lens)
        logits = self.classifier(fusion_out)

        if return_projection_output:
            projected = self.moon_prj_head(fusion_out)
            return logits, projected, att_weights
        return logits, att_weights


def build_adni_moon_model(dataset_cfg: dict) -> ADNIMoonModel:
    """Construct an ADNIMoonModel from a dataset config dict."""
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
    moon_prj_head = ProjectionHead(input_dim=classifier_input_dim)

    return ADNIMoonModel(
        encoders=encoders,
        fusion=fusion,
        classifier=classifier,
        moon_prj_head=moon_prj_head,
        modalities=dataset_cfg["modalities"],
        num_patches=num_patches,
        enc_out_dim=enc_out_dim,
    )
