
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron classifier."""

    def __init__(self, input_dim, hidden_dim, num_hid_layers, output_dim, dropout=0.2):
        super().__init__()
        if num_hid_layers < 1:
            raise ValueError("MLP must have at least one hidden layer.")

        layers = []
        layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        for _ in range(num_hid_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
