import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveWeight(nn.Module):
    """KL-divergence-based contrastive loss across modality-expanded batch."""

    def __init__(self, num_modalities):
        super().__init__()
        self.temperature = 0.2
        self.num_modalities = num_modalities
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def _positive_negative_mask(self, sim_matrix, cur_bs):
        diag = np.eye(cur_bs)
        mask = torch.from_numpy(diag).to(sim_matrix.device).bool()

        oral_bs = cur_bs // self.num_modalities
        pos = np.zeros(sim_matrix.shape)
        for i in range(self.num_modalities):
            pos += np.eye(cur_bs, cur_bs, k=oral_bs * i)
            pos += np.eye(cur_bs, cur_bs, k=-oral_bs * i)
        pos = torch.from_numpy(pos).to(sim_matrix.device)
        pos[mask] = 0
        neg = 1 - pos
        neg[mask] = 0
        return pos.bool(), neg.bool()

    def forward(self, batch_emb):
        cur_bs = batch_emb.shape[0]
        norm = F.normalize(batch_emb, dim=1)
        sim = torch.matmul(norm, norm.T)

        pos_mask, neg_mask = self._positive_negative_mask(sim, cur_bs)
        positives = sim[pos_mask].view(cur_bs, -1)
        negatives = sim[neg_mask].view(cur_bs, -1)

        logits = torch.cat((positives, negatives), dim=-1)
        y_true = torch.cat(
            (torch.ones(cur_bs, positives.shape[-1]),
             torch.zeros(cur_bs, negatives.shape[-1])),
            dim=-1,
        ).to(batch_emb.device).float()

        predict = self.log_softmax(logits / self.temperature)
        loss = self.kl(predict, y_true)
        return loss, sim


class AggregationRebuild(nn.Module):
    """Softmax-weighted rebuild of features from similarity matrix."""

    def __init__(self):
        super().__init__()
        self.temperature = 0.2
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, similarity_matrix, batch_emb):
        n, d = batch_emb.shape
        sim = similarity_matrix / self.temperature
        sim = sim - torch.eye(n, device=sim.device).float() * 1e12
        weights = self.softmax(sim)
        rebuilt = torch.matmul(weights, batch_emb)
        return weights, rebuilt


class SimilarityProjector(nn.Module):
    """Projects concatenated [specific_emb, feature, prompt] to embed_dim."""

    def __init__(self, embed_dim=128, hid_dim=256):
        super().__init__()
        self.ln = nn.Sequential(
            nn.Linear(embed_dim * 3, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, embed_dim),
        )

    def forward(self, x):
        return self.ln(x)


class GlobalFuser(nn.Module):
    """Learned gate between original and rebuilt features via small convs."""

    def __init__(self):
        super().__init__()
        self.masking = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 8, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(8, 1, 3, 1, 1),
        )

    def forward(self, old, new):
        old = old.unsqueeze(1)
        new = new.unsqueeze(1)
        all_feat = torch.cat([old, new], dim=1)
        mask = torch.sigmoid(self.masking(all_feat))
        fused = old * mask + new * (1 - mask)
        return fused.squeeze(1), mask.squeeze(1)


class SpecificEmbedding(nn.Module):
    """Learnable per-modality embedding broadcast to each sample."""

    def __init__(self, num_modalities, hidden_dim=128):
        super().__init__()
        self.imputation = nn.Parameter(
            torch.randn(1, num_modalities, hidden_dim) * 0.01,
        )

    def forward(self):
        return self.imputation


class FLinear(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.ln = nn.Linear(emb_dim * 2, emb_dim, bias=False)
        self.weight = self.ln.weight

    def forward(self, x):
        return self.ln(x)


class Pool(nn.Module):
    """Prompt pool: retrieves top-k prompts per modality query."""

    def __init__(self, embed_dim=128, pool_size=10, top_k=3):
        super().__init__()
        self.top_k = top_k
        self.pool_size = pool_size
        self.embed_dim = embed_dim
        self.prompt = nn.Parameter(torch.zeros(pool_size, embed_dim))
        self.features_proj = FLinear(emb_dim=embed_dim)
        self.features_dropout = nn.Dropout(0.0)

        val = math.sqrt(6.0 / float(2 * embed_dim))
        nn.init.uniform_(self.prompt.data, -val, val)
        nn.init.uniform_(self.features_proj.weight, -1, 1)

    @staticmethod
    def _l2_normalize(x, dim=None, eps=1e-12):
        sq = torch.sum(x ** 2, dim=dim, keepdim=True)
        return x * torch.rsqrt(torch.maximum(sq, torch.tensor(eps, device=x.device)))

    def forward(self, x_embed, cls_features=None):
        cur_pool = self.prompt.shape[0]

        prompt_norm = self._l2_normalize(self.prompt, dim=1)
        x_proj = self.features_proj(self.features_dropout(x_embed))
        x_norm = self._l2_normalize(x_proj, dim=1)

        similarity = torch.matmul(x_norm, prompt_norm.t())
        _, idx = torch.topk(similarity, k=self.top_k, dim=1)

        prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
        if prompt_id.shape[0] < cur_pool:
            prompt_id = torch.cat([
                prompt_id,
                torch.full((cur_pool - prompt_id.shape[0],),
                            torch.min(idx.flatten()), device=prompt_id.device),
            ])
            id_counts = torch.cat([
                id_counts,
                torch.full((cur_pool - id_counts.shape[0],),
                            0, device=id_counts.device),
            ])

        _, major_idx = torch.topk(id_counts, k=self.top_k)
        major_prompt_id = prompt_id[major_idx]

        idx = major_prompt_id.expand(x_embed.shape[0], -1)
        self.top_k_idx = idx[0]

        batched_prompt = self.prompt[idx]
        batched_key_norm = prompt_norm[idx]

        x_norm_u = x_norm.unsqueeze(1)
        sim = batched_key_norm * x_norm_u
        reduce_sim = torch.sum(sim) / x_embed.shape[0]

        return reduce_sim, batched_prompt


class PepsyClassifier(nn.Module):
    """Classifier head for PEPSY: M * 3 * embed_dim -> hidden -> num_classes."""

    def __init__(self, num_modalities, embed_dim, num_classes, hid_dim=128):
        super().__init__()
        self.ln1 = nn.Linear(embed_dim * num_modalities * 3, hid_dim, True)
        self.ln2 = nn.Linear(hid_dim, num_classes, True)

    def forward(self, x):
        return self.ln2(F.relu(self.ln1(x)))
