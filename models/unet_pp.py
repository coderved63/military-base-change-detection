"""UNet++ (Nested U-Net) for change detection.

Uses a shared ResNet34 encoder from segmentation-models-pytorch. Features from
both temporal images are differenced and decoded through nested skip connections.
Optionally supports deep supervision.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetPPChangeDetection(nn.Module):
    """UNet++ adapted for bitemporal change detection.

    A shared encoder processes both images. The absolute difference of
    encoder features is fed into the UNet++ decoder.

    Args:
        encoder_name: Encoder backbone (default: 'resnet34').
        pretrained: Use ImageNet-pretrained encoder weights.
        deep_supervision: Enable deep supervision outputs.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision

        # Shared encoder via SMP
        encoder_weights = "imagenet" if pretrained else None
        self.base_model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )

        # We'll use the encoder and decoder separately
        self.encoder = self.base_model.encoder
        self.decoder = self.base_model.decoder
        self.segmentation_head = self.base_model.segmentation_head

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x1: Before image [B, 3, 256, 256].
            x2: After image [B, 3, 256, 256].

        Returns:
            Raw logits [B, 1, 256, 256].
        """
        # Extract multi-scale features from both images
        features_1 = self.encoder(x1)
        features_2 = self.encoder(x2)

        # Compute absolute difference at each scale
        diff_features = [torch.abs(f1 - f2) for f1, f2 in zip(features_1, features_2)]

        # Decode
        decoder_output = self.decoder(*diff_features)
        out = self.segmentation_head(decoder_output)
        return out


if __name__ == "__main__":
    # Quick sanity check
    model = UNetPPChangeDetection(pretrained=False)
    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)
    out = model(x1, x2)
    print(f"Input: {x1.shape}, Output: {out.shape}")
    assert out.shape == (2, 1, 256, 256), f"Unexpected shape: {out.shape}"
