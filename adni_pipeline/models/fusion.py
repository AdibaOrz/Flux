

import torch
import torch.nn as nn


class MultiheadAdditiveAttentionFusion(nn.Module):
    """Patch-level multi-context additive attention over concatenated modality sequences.

    Expects input x of shape (B, total_patches, feat_dim) where total_patches
    is the sum of per-modality patch counts. Missing modalities are masked out
    via valid_lens so their positions receive ~0 attention weight.
    """

    def __init__(self, input_dim: int = 128, hid_dim: int = 512, num_heads: int = 4):
        super().__init__()
        self.transform = nn.Linear(input_dim, hid_dim)
        self.activation = nn.Tanh()
        self.context_vectors = nn.Linear(hid_dim, num_heads)


    def forward(self, x, modality_lens=None, valid_lens=None):
        """
        Args:
            x: (B, seq_len, feat_dim) concatenated patch embeddings from all modalities.
            modality_lens: list of ints, per-modality patch count (e.g. [16, 16, 16, 16]).
            valid_lens: list of (B,) long tensors, per-modality valid length per sample
                        (num_patches if present, 0 if missing).
        Returns:
            output: (B, num_heads * input_dim)
        """
        batch_size, seq_len, _ = x.shape

        h = self.activation(self.transform(x))          # (B, seq_len, hid_dim)
        scores = self.context_vectors(h)                # (B, seq_len, num_heads)
        scores = scores.transpose(1, 2)                 # (B, num_heads, seq_len)

        if modality_lens is not None and valid_lens is not None:
            starts = []
            pos = 0
            for ml in modality_lens:
                starts.append(pos)
                pos += ml

            for mod_idx, (start, vl) in enumerate(zip(starts, valid_lens)):
                end = start + modality_lens[mod_idx]
                for b in range(batch_size):
                    mask_start = start + int(vl[b].item())
                    if mask_start < end:
                        scores[b, :, mask_start:end] = -1e9

        weights = torch.softmax(scores, dim=2)          # (B, num_heads, seq_len)
        context = torch.matmul(weights, x)              # (B, num_heads, feat_dim)
        output = context.reshape(batch_size, -1)        # (B, num_heads * feat_dim)
        return output, weights
