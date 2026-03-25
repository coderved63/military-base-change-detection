"""ChangeFormer — Transformer-based change detection model.

Implements a hierarchical vision transformer (MiT-B1 style) with shared-weight
Siamese encoder and MLP decoder for change detection. Based on:
"A Transformer-Based Siamese Network for Change Detection" (arXiv:2201.01293).
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding for hierarchical feature extraction.

    Args:
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
        patch_size: Patch size for convolution.
        stride: Stride for convolution.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        patch_size: int = 7,
        stride: int = 4,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Tuple of (tokens [B, N, D], height, width).
        """
        x = self.proj(x)
        _, _, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        return x, h, w


class EfficientSelfAttention(nn.Module):
    """Efficient self-attention with spatial reduction.

    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        sr_ratio: Spatial reduction ratio.
    """

    def __init__(self, dim: int, num_heads: int = 1, sr_ratio: int = 8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        # Spatial reduction
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tokens [B, N, C].
            h: Feature map height.
            w: Feature map width.

        Returns:
            Output tokens [B, N, C].
        """
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            x_ = self.sr(x_)
            x_ = rearrange(x_, "b c h w -> b (h w) c")
            x_ = self.sr_norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return out


class MixFFN(nn.Module):
    """Mix Feed-Forward Network with depthwise convolution.

    Args:
        dim: Input/output dimension.
        mlp_ratio: Expansion ratio for hidden dimension.
    """

    def __init__(self, dim: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tokens [B, N, C].
            h: Feature map height.
            w: Feature map width.

        Returns:
            Output tokens [B, N, C].
        """
        x = self.fc1(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.act(self.dwconv(x))
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with efficient attention and MixFFN.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        sr_ratio: Spatial reduction ratio for attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        mlp_ratio: int = 4,
        sr_ratio: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, mlp_ratio)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tokens [B, N, C].
            h: Feature map height.
            w: Feature map width.

        Returns:
            Output tokens [B, N, C].
        """
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.ffn(self.norm2(x), h, w)
        return x


class MiTEncoder(nn.Module):
    """Mix Transformer (MiT) encoder — hierarchical vision transformer.

    Args:
        embed_dims: Embedding dimensions at each stage.
        num_heads: Number of attention heads at each stage.
        mlp_ratios: MLP expansion ratios at each stage.
        depths: Number of transformer blocks at each stage.
    """

    def __init__(
        self,
        embed_dims: List[int] = [64, 128, 320, 512],
        num_heads: List[int] = [1, 2, 5, 8],
        mlp_ratios: List[int] = [8, 8, 4, 4],
        depths: List[int] = [2, 2, 2, 2],
    ) -> None:
        super().__init__()
        self.num_stages = len(embed_dims)

        sr_ratios = [8, 4, 2, 1]
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]

        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(self.num_stages):
            in_ch = 3 if i == 0 else embed_dims[i - 1]
            self.patch_embeds.append(
                OverlapPatchEmbed(in_ch, embed_dims[i], patch_sizes[i], strides[i])
            )
            self.blocks.append(
                nn.ModuleList([
                    TransformerBlock(embed_dims[i], num_heads[i], mlp_ratios[i], sr_ratios[i])
                    for _ in range(depths[i])
                ])
            )
            self.norms.append(nn.LayerNorm(embed_dims[i]))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract hierarchical features.

        Args:
            x: Input image [B, 3, H, W].

        Returns:
            List of feature maps at each stage [B, C_i, H_i, W_i].
        """
        features = []
        for i in range(self.num_stages):
            x, h, w = self.patch_embeds[i](x)
            for blk in self.blocks[i]:
                x = blk(x, h, w)
            x = self.norms[i](x)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            features.append(x)
        return features


class MLPDecoder(nn.Module):
    """MLP-based decoder that fuses multi-scale difference features.

    Args:
        embed_dims: Embedding dimensions from each encoder stage.
        out_channels: Number of output channels (1 for binary change mask).
    """

    def __init__(
        self,
        embed_dims: List[int] = [64, 128, 320, 512],
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        unified_dim = embed_dims[0]

        self.linear_projections = nn.ModuleList([
            nn.Conv2d(dim, unified_dim, kernel_size=1)
            for dim in embed_dims
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(unified_dim * len(embed_dims), unified_dim, kernel_size=1),
            nn.BatchNorm2d(unified_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(unified_dim, out_channels, kernel_size=1)

    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        """Forward pass.

        Args:
            features: List of difference feature maps.
            target_size: (H, W) of the desired output.

        Returns:
            Logits [B, 1, H, W].
        """
        projected = []
        for i, (feat, proj) in enumerate(zip(features, self.linear_projections)):
            p = proj(feat)
            p = F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            projected.append(p)

        fused = self.fuse(torch.cat(projected, dim=1))
        out = self.head(fused)
        return out


class ChangeFormer(nn.Module):
    """ChangeFormer: Transformer-based Siamese network for change detection.

    Args:
        embed_dims: Embedding dims at each hierarchical stage.
        num_heads: Attention heads at each stage.
        mlp_ratios: MLP expansion ratios at each stage.
        depths: Transformer block counts at each stage.
        pretrained_backbone: Whether to load pretrained MiT weights.
    """

    def __init__(
        self,
        embed_dims: List[int] = [64, 128, 320, 512],
        num_heads: List[int] = [1, 2, 5, 8],
        mlp_ratios: List[int] = [8, 8, 4, 4],
        depths: List[int] = [2, 2, 2, 2],
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()

        # Shared Siamese encoder
        self.encoder = MiTEncoder(embed_dims, num_heads, mlp_ratios, depths)

        # MLP decoder
        self.decoder = MLPDecoder(embed_dims, out_channels=1)

        # TODO: Load pretrained MiT-B1 weights if pretrained_backbone is True

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x1: Before image [B, 3, 256, 256].
            x2: After image [B, 3, 256, 256].

        Returns:
            Raw logits [B, 1, 256, 256].
        """
        # Extract hierarchical features
        feats_1 = self.encoder(x1)
        feats_2 = self.encoder(x2)

        # Compute difference at each scale
        diff_feats = [torch.abs(f1 - f2) for f1, f2 in zip(feats_1, feats_2)]

        # Decode to change mask
        target_size = (x1.shape[2], x1.shape[3])
        out = self.decoder(diff_feats, target_size)
        return out


if __name__ == "__main__":
    # Quick sanity check
    model = ChangeFormer(pretrained_backbone=False)
    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 3, 256, 256)
    out = model(x1, x2)
    print(f"Input: {x1.shape}, Output: {out.shape}")
    assert out.shape == (1, 1, 256, 256), f"Unexpected shape: {out.shape}"
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
