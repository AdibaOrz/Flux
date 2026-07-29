# models/task_heads.py

import torch.nn as nn


# Head model
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # MLP Head
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)
