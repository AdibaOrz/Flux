# models/fusions.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiContextAdditiveAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_contexts: int = 4,
                 proj_dropout: float = 0.1, use_layer_norm: bool = True):
        """
        Multi-Context Additive Attention Fusion

        Args:
            input_dim: dimension of input modalities
            hidden_dim: dimension of projection space
            num_contexts: number of context vectors to use
            proj_dropout: dropout probability
            use_layer_norm: whether to use layer normalization
        """
        super().__init__()

        self.num_contexts = num_contexts
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # 1. Linear projection h -> u (shared across all contexts)
        self.proj = nn.Linear(input_dim, hidden_dim, bias=True)

        # 2. Multiple context vectors u_w
        self.context_vectors = nn.Parameter(torch.Tensor(num_contexts, hidden_dim))

        # 3. Optional normalization & dropout layers
        self.layer_norm = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        self.proj_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

        # 4. Fusion-specific layers
        self.fusion_proj = nn.Linear(input_dim * num_contexts, input_dim)
        self._init_weights()


    def _init_weights(self, mean=0.0, std=0.05):
        """ Initialize weights for context vectors """
        self.context_vectors.data.normal_(mean, std)

        # Initialize fusion-specific parameters
        if hasattr(self, 'fusion_proj'):
            nn.init.xavier_uniform_(self.fusion_proj.weight)
            nn.init.zeros_(self.fusion_proj.bias)


    def _masked_softmax(self, logits: torch.Tensor, mask: torch.Tensor, dim: int = -1,fill_value: float = -1e9):
        """
        logits: (B, M) or (B, K, M) for multi-context
        mask:   (B, M)  True = keep, False = missing
        returns: softmax over specified dimension, zeroing out masked positions
        """
        # Handle both 2D and 3D inputs
        if logits.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, 1, M)

        # 1) mask out logits for missing modalities
        mask = mask.to(torch.bool)
        masked_logits = logits.masked_fill(~mask, fill_value)

        # 2) standard softmax
        probs = F.softmax(masked_logits, dim=dim)

        # 3) zero out any residual on masked slots
        probs = probs * mask.to(probs.dtype)

        # 4) renormalize so surviving probs sum to 1
        probs /= probs.sum(dim=dim, keepdim=True) + 1e-13
        return probs


    def forward(self, modalities, modality_mask):
        """
        modalities: list[Tensor] with shapes (B, embed_dim)
        mask: (B, M) where M is the number of modalities (1: present, 0: missing)
        returns: - fused_representation (B, embed_dim)
                 - attention_weights   (B, K, M) where K is num_contexts
        """

        # 1. Stack & normalize
        h = torch.stack(modalities, dim=1)  # (B, M, E)
        h = self.layer_norm(h)  # (B, M, E)

        # 2. Project & dropout (shared projection for all contexts)
        u = torch.tanh(self.proj(h))  # (B, M, H)
        u = self.proj_dropout(u)  # (B, M, H)

        # 3. Compute attention for each context vector
        # Batch matrix multiplication: (B, M, H) x (K, H).T -> (B, K, M)
        logits = torch.einsum('bmh,kh->bkm', u, self.context_vectors)
        logits /= math.sqrt(u.size(-1))  # (B, K, M)

        # 4. Softmax attention weights for each context
        attn_weights = self._masked_softmax(logits, modality_mask, dim=2)  # (B, K, M)

        # 5. Apply attention for each context
        # (B, K, M) x (B, M, E) -> (B, K, E)
        context_outputs = torch.einsum('bkm,bme->bke', attn_weights, h)

        # 6. Fuse outputs from multiple contexts
        # Concatenate all context outputs
        fused = context_outputs.reshape(context_outputs.size(0), -1)  # (B, K*E)
        fused = self.fusion_proj(fused)  # (B, E)
        return fused, attn_weights


# Example usage and testing
if __name__ == "__main__":
    # Test the multi-context attention
    batch_size = 2
    num_modalities = 3
    embed_dim = 128
    hidden_dim = 64
    num_contexts = 4

    # Create dummy data
    modalities = [torch.randn(batch_size, embed_dim) for _ in range(num_modalities)]
    modality_mask = torch.ones(batch_size, num_modalities)
    modality_mask[0, 2] = 0  # Mask out last modality for first sample

    # Test different fusion methods
    model = MultiContextAdditiveAttention(
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_contexts=num_contexts,
    )

    output, attention_weights = model(modalities, modality_mask)
    print(f"Output shape: {output.shape}")  # Should be (B, embed_dim)
    print(f"Attention weights shape: {attention_weights.shape}")  # Should be (B, K, M)
    print(f"Attention weights sum per context: {attention_weights.sum(dim=2)}")  # Should be ~1 for each context