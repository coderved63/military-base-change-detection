"""Siamese CNN baseline for change detection.

Architecture: Shared-weight ResNet18 backbone extracts features from both
images. Feature difference is decoded via transposed convolutions to produce
a binary change mask.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SiameseCNN(nn.Module):
    """Siamese CNN with shared ResNet18 encoder and transposed-conv decoder.

    Args:
        backbone: Backbone architecture name (default: 'resnet18').
        pretrained: Whether to use ImageNet-pretrained weights.
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()

        # Shared encoder
        resnet = getattr(models, backbone)(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Remove avgpool and fc — keep feature extraction layers
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 64 channels
            resnet.layer2,  # 128 channels
            resnet.layer3,  # 256 channels
            resnet.layer4,  # 512 channels
        )

        # Decoder: upsample difference features back to input resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x1: Before image [B, 3, 256, 256].
            x2: After image [B, 3, 256, 256].

        Returns:
            Raw logits [B, 1, 256, 256].
        """
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)

        # Feature difference
        diff = torch.abs(f1 - f2)

        # Decode to change mask
        out = self.decoder(diff)
        return out


if __name__ == "__main__":
    # Quick sanity check
    model = SiameseCNN(pretrained=False)
    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)
    out = model(x1, x2)
    print(f"Input: {x1.shape}, Output: {out.shape}")
    assert out.shape == (2, 1, 256, 256), f"Unexpected shape: {out.shape}"
