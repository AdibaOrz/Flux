import torch
import torch.nn as nn



class CNNTimeSeries(nn.Module):
    def __init__(
            self,
            feat_dim: int,            # Input feature dimension (#channels)
            embed_dim: int,           # Size of final embedding
            conv_channels: int = 64,  # Number of filters per convolutional layer
            num_conv_layers: int = 3, # Number of stacked Conv1D layers
            kernel_size: int = 5,     # Kernel size for convolution
            dropout: float = 0.1      # Dropout rate for regularization
    ):
        super().__init__()
        conv_layers = []
        in_channels = feat_dim
        for _ in range(num_conv_layers):
            conv_layers.append(nn.Conv1d(in_channels, conv_channels, kernel_size=kernel_size))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.Dropout(dropout))
            in_channels = conv_channels

        self.conv_module = nn.Sequential(*conv_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Projection to embed_dim
        self.fc = nn.Linear(conv_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (torch.Tensor): Input tensor of shape (batch_size, feature_dim, seq_len)"""
        x = self.conv_module(x)      # -> (batch_size, conv_channels, seq_len)
        x = self.global_avg_pool(x)  # -> (batch_size, conv_channels, 1)
        x = x.squeeze(-1)            # -> (batch_size, conv_channels) Remove last dimension
        x = self.fc(x)               # -> (batch_size, embed_dim)
        return x


if __name__ == "__main__":
    batch_size, feat_dim, seq_len, embed_dim = 8, 3, 100, 128
    model = CNNTimeSeries(
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        conv_channels=32,
        num_conv_layers=3,
    )
    x = torch.randn(batch_size, feat_dim, seq_len)
    out = model(x)
    print("CNNTimeSeries output shape:", out.shape)

