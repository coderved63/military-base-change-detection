"""Visualization utilities for change detection results.

Provides helpers for:
- Plotting side-by-side predictions (Before | After | GT | Pred)
- Overlaying predicted change masks on satellite images
- Plotting metric curves across epochs
- Logging sample prediction grids to TensorBoard

All public functions accept **ImageNet-normalised** ``torch.Tensor`` inputs
with shape ``[C, H, W]`` and handle denormalisation internally.  The Agg
backend is set at import time so the module works in headless environments
(Google Colab, CI, remote servers).
"""

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

logger = logging.getLogger(__name__)

# ImageNet constants (duplicated here to avoid circular imports from data/)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy_hwc(tensor: torch.Tensor) -> np.ndarray:
    """Convert a ``[C, H, W]`` torch tensor to ``[H, W, C]`` numpy array.

    Args:
        tensor: Image tensor of shape ``[C, H, W]``.

    Returns:
        Numpy array of shape ``[H, W, C]`` (float32).
    """
    return tensor.detach().cpu().float().permute(1, 2, 0).numpy()


def _mask_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a ``[1, H, W]`` mask tensor to ``[H, W]`` numpy array.

    Args:
        tensor: Mask tensor of shape ``[1, H, W]``.

    Returns:
        Numpy array of shape ``[H, W]`` (float32).
    """
    return tensor.detach().cpu().float().squeeze(0).numpy()


def denormalize(
    img: np.ndarray,
    mean: np.ndarray = _IMAGENET_MEAN,
    std: np.ndarray = _IMAGENET_STD,
) -> np.ndarray:
    """Reverse ImageNet normalisation for display.

    Args:
        img: Normalised image of shape ``[H, W, 3]`` (float32).
        mean: Per-channel means used during normalisation.
        std: Per-channel standard deviations used during normalisation.

    Returns:
        Denormalised image clipped to ``[0, 1]``.
    """
    return np.clip(img * std + mean, 0.0, 1.0)


def _denorm_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Shortcut: ``[C, H, W]`` tensor → denormalised ``[H, W, C]`` numpy.

    Args:
        tensor: ImageNet-normalised image ``[C, H, W]``.

    Returns:
        Denormalised numpy array ``[H, W, C]`` in ``[0, 1]``.
    """
    return denormalize(_to_numpy_hwc(tensor))


# ---------------------------------------------------------------------------
# 1. plot_prediction
# ---------------------------------------------------------------------------

def plot_prediction(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    mask_true: torch.Tensor,
    mask_pred: torch.Tensor,
    filename: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot a single change-detection prediction as a 1×4 grid.

    Columns: **Before (A)** | **After (B)** | **Ground Truth** | **Prediction**.

    Images are denormalised from ImageNet stats before display.  Masks are
    rendered in binary black / white.

    Args:
        img_a: Before image ``[3, H, W]`` (ImageNet-normalised).
        img_b: After image ``[3, H, W]`` (ImageNet-normalised).
        mask_true: Ground-truth binary mask ``[1, H, W]`` (0 or 1).
        mask_pred: Predicted mask ``[1, H, W]`` (binary or probability).
        filename: If provided, save the figure to this path and close it.
            Otherwise the caller is responsible for ``plt.close(fig)``.

    Returns:
        The ``matplotlib.figure.Figure`` object.
    """
    a_np = _denorm_tensor(img_a)
    b_np = _denorm_tensor(img_b)
    gt_np = _mask_to_numpy(mask_true)
    pred_np = _mask_to_numpy(mask_pred)

    # Binarise prediction for clean display (handles probability maps)
    pred_np = (pred_np > 0.5).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["Before (A)", "After (B)", "Ground Truth", "Prediction"]
    images = [a_np, b_np, gt_np, pred_np]
    cmaps = [None, None, "gray", "gray"]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.tight_layout(pad=1.0)

    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Saved prediction plot: %s", path)

    return fig


# ---------------------------------------------------------------------------
# 2. overlay_changes
# ---------------------------------------------------------------------------

def overlay_changes(
    img_after: torch.Tensor,
    mask_pred: torch.Tensor,
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Overlay predicted change pixels on the *after* image.

    Changed pixels are tinted with ``color`` at the given ``alpha``
    transparency; unchanged pixels are left as-is.

    Args:
        img_after: After image ``[3, H, W]`` (ImageNet-normalised).
        mask_pred: Predicted binary mask ``[1, H, W]`` (0 or 1).
        alpha: Blending factor for the overlay colour (0 = transparent,
            1 = fully opaque).
        color: RGB overlay colour as **uint8** values in ``[0, 255]``
            (default red).

    Returns:
        Composited RGB image as a **uint8** numpy array ``[H, W, 3]``
        with values in ``[0, 255]``, ready for ``cv2.imwrite`` or display.
    """
    base = _denorm_tensor(img_after)  # [H, W, 3], float32 in [0, 1]
    mask = _mask_to_numpy(mask_pred)  # [H, W], float32

    # Normalise colour to [0, 1]
    color_f = np.array(color, dtype=np.float32) / 255.0

    overlay = base.copy()
    change_mask = mask > 0.5
    for c in range(3):
        overlay[:, :, c] = np.where(
            change_mask,
            base[:, :, c] * (1.0 - alpha) + color_f[c] * alpha,
            base[:, :, c],
        )

    return (overlay * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. plot_metrics_history
# ---------------------------------------------------------------------------

def plot_metrics_history(
    history_dict: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot training / validation metric curves across epochs.

    Creates one subplot per metric key.  Suitable for inclusion in reports
    or as a TensorBoard-compatible image.

    Args:
        history_dict: Mapping from metric name to a list of per-epoch
            values, e.g. ``{"f1": [0.5, 0.6, ...], "loss": [0.8, ...]}``.
        save_path: If provided, save the figure and close it.

    Returns:
        The ``matplotlib.figure.Figure`` object.
    """
    n_metrics = len(history_dict)
    if n_metrics == 0:
        fig, _ = plt.subplots()
        return fig

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, history_dict.items()):
        epochs = list(range(1, len(values) + 1))
        ax.plot(epochs, values, marker="o", markersize=3, linewidth=1.5)
        ax.set_title(name.upper(), fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=1.5)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Saved metrics plot: %s", path)

    return fig


# ---------------------------------------------------------------------------
# 4. log_predictions_to_tensorboard
# ---------------------------------------------------------------------------

def log_predictions_to_tensorboard(
    writer: SummaryWriter,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    mask_true: torch.Tensor,
    mask_pred: torch.Tensor,
    step: int,
    num_samples: int = 4,
) -> None:
    """Log a grid of sample predictions to TensorBoard.

    For each sample the grid contains four rows:
    *Before*, *After*, *Ground Truth*, *Prediction*.

    Images are denormalised; masks are expanded to 3-channel for consistent
    grid rendering.

    Args:
        writer: Active ``SummaryWriter`` instance.
        img_a: Before images ``[B, 3, H, W]`` (ImageNet-normalised).
        img_b: After images ``[B, 3, H, W]`` (ImageNet-normalised).
        mask_true: Ground-truth masks ``[B, 1, H, W]`` (binary).
        mask_pred: Predicted masks ``[B, 1, H, W]`` (binary or probability).
        step: Global training step (used as the x-axis in TensorBoard).
        num_samples: How many samples from the batch to include (taken
            from the front of the batch dimension).
    """
    n = min(num_samples, img_a.size(0))

    # Denormalise images on CPU (keep as tensors for vutils.make_grid)
    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)

    a = (img_a[:n].cpu().float() * std + mean).clamp(0.0, 1.0)
    b = (img_b[:n].cpu().float() * std + mean).clamp(0.0, 1.0)

    # Expand single-channel masks to 3-channel for the grid
    gt = mask_true[:n].cpu().float().expand(-1, 3, -1, -1)
    pred = (mask_pred[:n].cpu().float() > 0.5).float().expand(-1, 3, -1, -1)

    # Interleave: [a0, b0, gt0, pred0, a1, b1, gt1, pred1, ...]
    rows = []
    for i in range(n):
        rows.extend([a[i], b[i], gt[i], pred[i]])

    grid = vutils.make_grid(rows, nrow=4, padding=2, normalize=False)
    writer.add_image("Predictions/before_after_gt_pred", grid, global_step=step)
