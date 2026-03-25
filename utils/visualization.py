"""Visualization utilities for change detection results.

Provides functions to plot predictions, overlay change maps, and track
training metrics over time.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def denormalize(
    img: np.ndarray,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Reverse ImageNet normalization for display.

    Args:
        img: Normalized image array [H, W, 3].
        mean: Channel means used for normalization.
        std: Channel stds used for normalization.

    Returns:
        Denormalized image clipped to [0, 1].
    """
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)


def plot_prediction(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    mask_gt: torch.Tensor,
    mask_pred: torch.Tensor,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot a single change detection prediction.

    Shows: Before | After | Ground Truth | Prediction in a 1x4 grid.

    Args:
        img_a: Before image tensor [3, H, W] (normalized).
        img_b: After image tensor [3, H, W] (normalized).
        mask_gt: Ground truth mask [1, H, W] (binary).
        mask_pred: Predicted mask [1, H, W] (binary or probability).
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Convert tensors to numpy
    a = denormalize(img_a.permute(1, 2, 0).cpu().numpy())
    b = denormalize(img_b.permute(1, 2, 0).cpu().numpy())
    gt = mask_gt.squeeze(0).cpu().numpy()
    pred = mask_pred.squeeze(0).cpu().numpy()

    titles = ["Before (A)", "After (B)", "Ground Truth", "Prediction"]
    images = [a, b, gt, pred]
    cmaps = [None, None, "gray", "gray"]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def overlay_changes(
    img_b: torch.Tensor,
    mask_pred: torch.Tensor,
    alpha: float = 0.4,
    color: tuple = (1.0, 0.0, 0.0),
) -> np.ndarray:
    """Overlay predicted change mask on the after image.

    Args:
        img_b: After image tensor [3, H, W] (normalized).
        mask_pred: Predicted binary mask [1, H, W].
        alpha: Overlay transparency.
        color: RGB color for the overlay (default: red).

    Returns:
        Overlaid image as numpy array [H, W, 3].
    """
    b = denormalize(img_b.permute(1, 2, 0).cpu().numpy())
    mask = mask_pred.squeeze(0).cpu().numpy()

    overlay = b.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask > 0.5,
            b[:, :, c] * (1 - alpha) + color[c] * alpha,
            b[:, :, c],
        )
    return overlay


def plot_metrics_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot training metric curves over epochs.

    Args:
        history: Dict mapping metric names to lists of per-epoch values.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    n_metrics = len(history)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))

    if n_metrics == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, history.items()):
        ax.plot(values, marker="o", markersize=2)
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
