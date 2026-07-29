# models/method_specific/pepsy_components.py
# PEPSY-specific model components

import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveWeight(nn.Module):
    """Contrastive loss component for PEPSY."""

    def __init__(self, num_modalities):
        super(ContrastiveWeight, self).__init__()
        self.temperature = 0.2
        self.num_modalities = num_modalities

        self.bce = torch.nn.BCELoss()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')

    def get_positive_and_negative_mask(self, similarity_matrix, cur_batch_size, batch_emb_mask=None):
        """Generate positive and negative masks for contrastive learning."""
        diag = np.eye(cur_batch_size)
        mask = torch.from_numpy(diag).to(similarity_matrix.device)
        mask = mask.type(torch.bool)

        oral_batch_size = cur_batch_size // self.num_modalities

        positives_mask = np.zeros(similarity_matrix.size())
        for i in range(self.num_modalities):
            ll = np.eye(cur_batch_size, cur_batch_size, k=oral_batch_size * i)
            lr = np.eye(cur_batch_size, cur_batch_size, k=-oral_batch_size * i)
            positives_mask += ll
            positives_mask += lr

        positives_mask = torch.from_numpy(positives_mask).to(similarity_matrix.device)
        if batch_emb_mask is not None:
            batch_emb_mask = 1 - batch_emb_mask
            positives_mask = (positives_mask * batch_emb_mask) * (batch_emb_mask.T)
        positives_mask[mask] = 0

        negatives_mask = 1 - positives_mask
        negatives_mask[mask] = 0

        return positives_mask.type(torch.bool), negatives_mask.type(torch.bool)

    def forward(self, batch_emb_om, batch_emb_mask=None):
        """Compute contrastive loss."""
        cur_batch_shape = batch_emb_om.shape

        # Get similarity matrix among mask samples
        norm_emb = F.normalize(batch_emb_om, dim=1)
        similarity_matrix = torch.matmul(norm_emb, norm_emb.transpose(0, 1))

        # Get positives and negatives similarity
        positives_mask, negatives_mask = self.get_positive_and_negative_mask(
            similarity_matrix, cur_batch_shape[0], batch_emb_mask)

        positives = similarity_matrix[positives_mask].view(cur_batch_shape[0], -1)
        negatives = similarity_matrix[negatives_mask].view(cur_batch_shape[0], -1)

        # Generate predict and target probability distributions matrix
        logits = torch.cat((positives, negatives), dim=-1)
        y_true = torch.cat(
            (torch.ones(cur_batch_shape[0], positives.shape[-1]),
             torch.zeros(cur_batch_shape[0], negatives.shape[-1])),
            dim=-1).to(batch_emb_om.device).float()

        # Multiple positives - KL divergence
        predict = self.log_softmax(logits / self.temperature)
        loss = self.kl(predict, y_true)

        return loss, similarity_matrix, logits, positives_mask


class AggregationRebuild(nn.Module):
    """Rebuild aggregation component for PEPSY."""

    def __init__(self):
        super(AggregationRebuild, self).__init__()
        self.temperature = 0.2
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mse = torch.nn.MSELoss()

    def forward(self, similarity_matrix, batch_emb_om):
        """Rebuild embeddings based on similarity matrix."""
        cur_batch_shape = batch_emb_om.shape

        # Get the weight among (oral, oral's masks, others, others' masks)
        similarity_matrix = similarity_matrix / self.temperature

        similarity_matrix = similarity_matrix - torch.eye(cur_batch_shape[0]).to(
            similarity_matrix.device).float() * 1e12
        rebuild_weight_matrix = self.softmax(similarity_matrix)  # (BxM)x(BxM)

        batch_emb_om = batch_emb_om.reshape(cur_batch_shape[0], -1)  # (BxM)xD

        # Generate the rebuilt batch embedding
        rebuild_batch_emb = torch.matmul(rebuild_weight_matrix, batch_emb_om)

        # Get oral' rebuilt batch embedding
        rebuild_oral_batch_emb = rebuild_batch_emb.reshape(cur_batch_shape[0], cur_batch_shape[1], -1)

        return rebuild_weight_matrix, rebuild_oral_batch_emb


class SimilarityProjector(nn.Module):
    """Similarity projector for PEPSY."""

    def __init__(self, embed_dim=128):
        super(SimilarityProjector, self).__init__()
        self.ln = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, x):
        return self.ln(x)


class GlobalFuser(nn.Module):
    """Global fuser for PEPSY."""

    def __init__(self):
        super(GlobalFuser, self).__init__()
        self.masking = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 8, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(8, 1, 3, 1, 1))

    def forward(self, old, new):
        old = old.unsqueeze(1)
        new = new.unsqueeze(1)
        all_feat = torch.cat([old, new], dim=1)
        mask = torch.sigmoid(self.masking(all_feat))
        all_feat = old * mask + new * (1 - mask)
        return all_feat.squeeze(1), mask.squeeze(1)


class SpecificEmbedding(nn.Module):
    """Specific embedding for missing modality imputation."""

    def __init__(self, num_modalities, hidden_dim=128):
        super(SpecificEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.imputation = nn.Parameter(torch.randn(1, self.num_modalities, self.hidden_dim) * 0.01)

    def forward(self, x):
        return self.imputation


class FLinear(nn.Module):
    """Linear layer for feature projection."""

    def __init__(self, emb_dim=128):
        super().__init__()
        self.ln = nn.Linear(emb_dim * 2, emb_dim, bias=False)
        self.weight = self.ln.weight

    def forward(self, x):
        return self.ln(x)


class Pool(nn.Module):
    """Prompt pool for PEPSY."""

    def __init__(self, embed_dim=128, pool_size=10, top_k=3, dropout_value=0.0):
        super(Pool, self).__init__()
        self.top_k = top_k
        self.pool_size = pool_size
        self.embed_dim = embed_dim
        self.prompt = nn.Parameter(torch.zeros(pool_size, embed_dim))
        self.features_proj = FLinear(emb_dim=embed_dim)
        self.features_dropout = nn.Dropout(dropout_value)

        # Prompt initialization (uniform distribution)
        val = math.sqrt(6. / float(2 * embed_dim))
        nn.init.uniform_(self.prompt.data, -val, val)
        nn.init.uniform_(self.features_proj.weight, -1, 1)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """L2 normalization."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, cls_features=None):
        """Forward pass for prompt pool."""
        current_pool_size = self.prompt.shape[0]

        x_embed_mean = x_embed

        prompt_norm = self.l2_normalize(self.prompt, dim=1)  # Pool_size, C
        x_embed_mean = self.features_proj(self.features_dropout(x_embed_mean))  # B, embed_dim
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # B, C

        similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B, Pool_size
        _, idx = torch.topk(similarity, k=self.top_k, dim=1)  # B, top_k

        prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)

        if prompt_id.shape[0] < current_pool_size:
            prompt_id = torch.cat([prompt_id, torch.full(
                (current_pool_size - prompt_id.shape[0],),
                torch.min(idx.flatten()), device=prompt_id.device)])
            id_counts = torch.cat([id_counts, torch.full(
                (current_pool_size - id_counts.shape[0],),
                0, device=id_counts.device)])

        _, major_idx = torch.topk(id_counts, k=self.top_k)
        major_prompt_id = prompt_id[major_idx]

        # Expand to batch
        idx = major_prompt_id.expand(x_embed.shape[0], -1)  # B, top_k
        self.top_k_idx = idx[0]

        batched_prompt = self.prompt[idx]  # B, top_k, C
        batched_key_norm = prompt_norm[idx]

        x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
        sim = batched_key_norm * x_embed_norm  # B, top_k, C
        reduce_sim = torch.sum(sim) / x_embed.shape[0]

        return reduce_sim, batched_prompt


class Classifier(nn.Module):
    """Classifier head for PEPSY."""

    def __init__(self, num_modalities, embed_dim, num_classes):
        super(Classifier, self).__init__()
        self.num_modalities = num_modalities
        self.ln1 = nn.Linear(embed_dim * num_modalities * 3, 128, True)  # x3 due to 3 different features
        self.ln2 = nn.Linear(128, num_classes, True)

    def forward(self, x):
        return self.ln2(F.relu(self.ln1(x)))
