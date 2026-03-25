"""Loss functions for binary change detection.

Provides BCEDiceLoss (default) and FocalLoss, both operating on raw logits.
A factory function ``get_loss`` reads the project config and returns the
selected loss module.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy and Dice Loss.

    Both components operate on raw logits — sigmoid is applied internally so
    the caller should **not** pre-apply it.

    Args:
        bce_weight: Scalar weight for the BCE component.
        dice_weight: Scalar weight for the Dice component.
        smooth: Smoothing constant for Dice to avoid division by zero.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the combined BCE + Dice loss.

        Args:
            logits: Raw model output of shape ``[B, 1, H, W]``.
            targets: Binary ground-truth masks of shape ``[B, 1, H, W]``
                with values in {0, 1}.

        Returns:
            Scalar loss tensor on the same device as the inputs.
        """
        # --- BCE component (numerically stable, operates on logits) ---
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        # --- Dice component ---
        probs = torch.sigmoid(logits)
        # Flatten spatial dims per sample for stable dice computation
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in change detection.

    Down-weights well-classified (easy) pixels so the model focuses on hard
    examples near the decision boundary.  Operates on raw logits.

    Args:
        alpha: Balancing factor for the positive class (1 − alpha for negative).
        gamma: Focusing exponent — higher values down-weight easy examples more.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model output of shape ``[B, 1, H, W]``.
            targets: Binary ground-truth masks of shape ``[B, 1, H, W]``
                with values in {0, 1}.

        Returns:
            Scalar loss tensor on the same device as the inputs.
        """
        # Per-pixel BCE (unreduced)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs = torch.sigmoid(logits)
        # p_t = probability of the true class
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        # alpha_t = alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def get_loss(config: Dict[str, Any]) -> nn.Module:
    """Factory function — instantiate a loss module from the project config.

    Reads ``config["loss"]["name"]`` to select the loss type and extracts
    the matching sub-key for constructor arguments.

    Args:
        config: Full project config dict (as loaded from ``config.yaml``).

    Returns:
        An ``nn.Module`` loss function ready for ``loss(logits, targets)``.

    Raises:
        ValueError: If the requested loss name is not recognised.
    """
    loss_cfg = config.get("loss", {})
    name = loss_cfg.get("name", "bce_dice")

    if name == "bce_dice":
        params = loss_cfg.get("bce_dice", {})
        return BCEDiceLoss(
            bce_weight=params.get("bce_weight", 0.5),
            dice_weight=params.get("dice_weight", 0.5),
        )
    elif name == "focal":
        params = loss_cfg.get("focal", {})
        return FocalLoss(
            alpha=params.get("alpha", 0.25),
            gamma=params.get("gamma", 2.0),
        )
    else:
        raise ValueError(
            f"Unknown loss '{name}'. Choose from: bce_dice, focal"
        )
