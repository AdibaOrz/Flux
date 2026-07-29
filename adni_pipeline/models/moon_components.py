"""MOON-specific components for the ADNI pipeline."""

import torch.nn as nn


class ProjectionHead(nn.Module):
    """Projects fusion features for MOON contrastive loss."""

    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        return self.mlp(x)
