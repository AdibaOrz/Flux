# models/method_specific/moon.py

import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        return self.mlp(x)
