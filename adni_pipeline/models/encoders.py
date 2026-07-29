
import math
import torch.nn as nn
import torch.nn.functional as F



class PatchEmbeddings(nn.Module):
    """Splits a 1-D feature vector into fixed-size patches and projects each."""

    def __init__(self, feature_size, num_patches, embed_dim):
        super().__init__()
        patch_size = math.ceil(feature_size / num_patches)
        self.pad_size = num_patches * patch_size - feature_size
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        # x: (B, feature_size) -> (B, num_patches, embed_dim)
        x = F.pad(x, (0, self.pad_size))
        x = x.view(x.shape[0], self.num_patches, self.patch_size)
        return self.projection(x)


class Custom3DCNN(nn.Module):
    """3D CNN for MRI volumes with input shape (B, 30, 90, 90)."""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout3d(0.2)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=3)
        self.dropout2 = nn.Dropout3d(0.2)

        self.conv5 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv3d(128, hidden_dim, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=5, stride=5)
        self.dropout3 = nn.Dropout3d(0.2)

        # Final feature map: (B, hidden_dim, 1, 3, 3)
        self.fc = nn.Linear(hidden_dim * 1 * 3 * 3, hidden_dim)

    def forward(self, x):
        # x: (B, 30, 90, 90) -> (B, hidden_dim)
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, 30, 90, 90)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(self.pool1(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout2(self.pool2(x))

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.dropout3(self.pool3(x))

        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_encoder(encoder_config: dict) -> nn.Module:
    """Build an encoder from a config dict.

    Args:
        encoder_config: must contain keys 'type', 'input_shape', 'hidden_dim', 'num_patches'.
            type='3dcnn_patch': Custom3DCNN -> PatchEmbeddings  (image)
            type='patch':       PatchEmbeddings only            (tabular)
    """
    encoder_type = encoder_config["type"]
    input_shape = encoder_config["input_shape"]
    hidden_dim = encoder_config["hidden_dim"]
    num_patches = encoder_config["num_patches"]

    if encoder_type == "3dcnn_patch":
        return nn.Sequential(
            Custom3DCNN(hidden_dim=hidden_dim),
            PatchEmbeddings(feature_size=hidden_dim, num_patches=num_patches, embed_dim=hidden_dim),
        )
    elif encoder_type == "patch":
        return PatchEmbeddings(feature_size=input_shape[0], num_patches=num_patches, embed_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def build_adni_encoders(dataset_cfg: dict) -> nn.ModuleDict:
    """Build all ADNI modality encoders from a dataset config dict.

    Args:
        dataset_cfg: must contain 'modalities' list and 'encoder_configs' dict.

    Returns:
        nn.ModuleDict mapping modality name -> encoder module.
        All encoders output (B, num_patches, hidden_dim).
    """
    encoders = nn.ModuleDict()
    for modality in dataset_cfg["modalities"]:
        enc_cfg = dataset_cfg["encoder_configs"][modality]
        encoders[modality] = get_encoder(enc_cfg)
    return encoders
