import torch
import torch.nn as nn
from typing import Dict, List, Optional

from adni_pipeline.models.encoders import build_adni_encoders
from adni_pipeline.models.pepsy_components import (
    SimilarityProjector, Pool, SpecificEmbedding, ContrastiveWeight,
    AggregationRebuild, GlobalFuser, PepsyClassifier,
)


class ADNIPepsyModel(nn.Module):

    def __init__(self, encoders: nn.ModuleDict, modalities: List[str],
                 num_patches: int, enc_out_dim: int, num_classes: int,
                 pool_size: int = 10, top_k: int = 3):
        super().__init__()
        self.encoders = encoders
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.num_patches = num_patches
        self.enc_out_dim = enc_out_dim

        self.sim_projectors = nn.ModuleDict({
            mod: SimilarityProjector(embed_dim=enc_out_dim)
            for mod in modalities
        })
        self.embeddings = SpecificEmbedding(self.num_modalities, hidden_dim=enc_out_dim)
        self.pool = Pool(embed_dim=enc_out_dim, pool_size=pool_size, top_k=top_k)
        self.classifier = PepsyClassifier(self.num_modalities, enc_out_dim, num_classes)
        self.contrastive_weight_module = ContrastiveWeight(self.num_modalities)
        self.rebuilder = AggregationRebuild()
        self.global_fuser = GlobalFuser()

        self.trained_prompts_checklist = torch.zeros(pool_size, dtype=torch.float32)

    def checking_trained_prompt(self):
        if self.pool.top_k_idx.device != self.trained_prompts_checklist.device:
            self.trained_prompts_checklist = self.trained_prompts_checklist.to(
                self.pool.top_k_idx.device)
        self.trained_prompts_checklist[self.pool.top_k_idx] += 1.0

    def reset_trained_prompts_checklist(self):
        self.trained_prompts_checklist = torch.zeros(
            self.pool.prompt.shape[0], dtype=torch.float32)

    def forward(self, X: Dict[str, torch.Tensor], observed: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                contrastive_weight: float = 0.1):
        B = observed.shape[0]
        device = observed.device
        D = self.enc_out_dim
        M = self.num_modalities

        features = torch.zeros(B, M, D, dtype=torch.float32, device=device)

        for mod_idx, mod_name in enumerate(self.modalities):
            m_observed = observed[:, mod_idx]
            if m_observed.sum() > 0:
                patch_feats = self.encoders[mod_name](X[mod_name][m_observed])
                pooled = patch_feats.mean(dim=1)
                features[m_observed, mod_idx] = pooled

        missing_mask = (1 - observed.float()).unsqueeze(-1)

        valid_samples = observed.sum(dim=1) > 0
        if torch.any(~valid_samples):
            raise ValueError("Some samples have all modalities missing!")

        denominator = observed.float().sum(dim=1, keepdim=True).unsqueeze(-1)
        mean_shared = features.sum(dim=1, keepdim=True) / denominator
        features = features * (1 - missing_mask) + mean_shared * missing_mask

        ori_features_intra = features.permute(1, 0, 2).reshape(-1, D)

        specific_emb = self.embeddings()
        specific_emb = specific_emb.expand(B, -1, -1).to(device)
        features = torch.cat([specific_emb, features], dim=-1)

        global_prompts = []
        reduced_sims = 0.0
        for mod_idx in range(M):
            query = features[:, mod_idx, :]
            reduce_sim, batched_prompts = self.pool(query)
            reduced_sims += reduce_sim
            global_prompts.append(batched_prompts.mean(dim=1, keepdim=True))

        global_prompts = torch.cat(global_prompts, dim=1)
        features = torch.cat([features, global_prompts], dim=-1)
        reduced_sims = reduced_sims / M

        sim_features_intra = torch.zeros(B, M, D, dtype=torch.float32, device=device)
        for mod_idx, mod_name in enumerate(self.modalities):
            dense = self.sim_projectors[mod_name](features[:, mod_idx:mod_idx + 1, :])
            sim_features_intra[:, mod_idx:mod_idx + 1, :] = dense

        sim_features_intra = sim_features_intra.permute(1, 0, 2).reshape(-1, D)

        intra_loss, intra_sim = self.contrastive_weight_module(sim_features_intra)
        ori_intra_loss, _ = self.contrastive_weight_module(ori_features_intra)
        contrastive_loss = intra_loss * 1.0 + ori_intra_loss * 0.5

        _, rebuilt_features = self.rebuilder(
            intra_sim[:B * M, :B * M],
            features.reshape(-1, D * 3),
        )

        rebuilt_features, _ = self.global_fuser(
            features.reshape(B, -1, D),
            rebuilt_features.reshape(B, -1, D),
        )

        outputs = self.classifier(rebuilt_features.reshape(B, -1))
        self.checking_trained_prompt()

        if labels is not None:
            cls_loss = nn.CrossEntropyLoss()(outputs, labels.long())
            total_loss = (cls_loss
                          + contrastive_weight * contrastive_loss
                          + 0.1 * reduced_sims)
            return outputs, total_loss

        return outputs


def build_adni_pepsy_model(cfg: dict) -> ADNIPepsyModel:
    encoders = build_adni_encoders(cfg)
    return ADNIPepsyModel(
        encoders=encoders,
        modalities=cfg["modalities"],
        num_patches=cfg["num_patches"],
        enc_out_dim=cfg["enc_out_dim"],
        num_classes=cfg["num_classes"],
        pool_size=cfg.get("pepsy_pool_size", 10),
        top_k=cfg.get("pepsy_top_k", 3),
    )
