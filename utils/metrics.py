"""Evaluation metrics for binary change detection.

Provides a ``ConfusionMatrix`` accumulator, standalone metric functions, and a
high-level ``MetricTracker`` that accepts raw logits and handles sigmoid +
thresholding internally.

All tensor operations stay on GPU until the final ``.item()`` call inside
``compute()`` so there is no unnecessary device transfer during the hot loop.
"""

from typing import Dict

import torch

# Small constant to prevent division-by-zero in metric formulas.
_EPS: float = 1e-7


# ---------------------------------------------------------------------------
# Low-level confusion-matrix accumulator
# ---------------------------------------------------------------------------

class ConfusionMatrix:
    """Accumulates TP / FP / FN / TN counts across batches.

    Counts are kept as plain Python ints (moved off GPU via a single
    ``.item()`` per update call) so that accumulated values never overflow
    a GPU scalar.

    Example::

        cm = ConfusionMatrix()
        for preds, targets in loader:
            cm.update(preds, targets)
        metrics = cm.compute()
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.tp: int = 0
        self.fp: int = 0
        self.fn: int = 0
        self.tn: int = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one batch of binary predictions.

        All boolean logic runs on whatever device the tensors live on; only
        the four resulting scalars are moved to CPU via ``.item()``.

        Args:
            preds: Binary predictions ``[B, 1, H, W]`` with values in {0, 1}.
            targets: Ground-truth masks ``[B, 1, H, W]`` with values in {0, 1}.
        """
        p = preds.bool().flatten()
        t = targets.bool().flatten()

        self.tp += (p & t).sum().item()
        self.fp += (p & ~t).sum().item()
        self.fn += (~p & t).sum().item()
        self.tn += (~p & ~t).sum().item()

    def compute(self) -> Dict[str, float]:
        """Derive all metrics from the accumulated counts.

        Returns:
            Dict with keys ``'f1'``, ``'iou'``, ``'precision'``, ``'recall'``,
            ``'oa'`` — each a plain Python float.
        """
        precision = self.tp / (self.tp + self.fp + _EPS)
        recall = self.tp / (self.tp + self.fn + _EPS)
        f1 = 2.0 * precision * recall / (precision + recall + _EPS)
        iou = self.tp / (self.tp + self.fp + self.fn + _EPS)
        oa = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + _EPS)

        return {
            "f1": f1,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "oa": oa,
        }


# ---------------------------------------------------------------------------
# Standalone convenience functions (single-batch, binary inputs)
# ---------------------------------------------------------------------------

def _quick_cm(preds: torch.Tensor, targets: torch.Tensor) -> ConfusionMatrix:
    """Create and populate a ConfusionMatrix from a single batch.

    Args:
        preds: Binary predictions ``[B, 1, H, W]``.
        targets: Ground-truth masks ``[B, 1, H, W]``.

    Returns:
        Populated ``ConfusionMatrix`` instance.
    """
    cm = ConfusionMatrix()
    cm.update(preds, targets)
    return cm


def compute_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute F1 score for a single batch.

    Args:
        preds: Binary predictions ``[B, 1, H, W]``.
        targets: Ground-truth masks ``[B, 1, H, W]``.

    Returns:
        F1 score as a float in [0, 1].
    """
    return _quick_cm(preds, targets).compute()["f1"]


def compute_iou(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute IoU (Jaccard index) for a single batch.

    Args:
        preds: Binary predictions ``[B, 1, H, W]``.
        targets: Ground-truth masks ``[B, 1, H, W]``.

    Returns:
        IoU score as a float in [0, 1].
    """
    return _quick_cm(preds, targets).compute()["iou"]


def compute_precision(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute precision for a single batch.

    Args:
        preds: Binary predictions ``[B, 1, H, W]``.
        targets: Ground-truth masks ``[B, 1, H, W]``.

    Returns:
        Precision score as a float in [0, 1].
    """
    return _quick_cm(preds, targets).compute()["precision"]


def compute_recall(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute recall for a single batch.

    Args:
        preds: Binary predictions ``[B, 1, H, W]``.
        targets: Ground-truth masks ``[B, 1, H, W]``.

    Returns:
        Recall score as a float in [0, 1].
    """
    return _quick_cm(preds, targets).compute()["recall"]


def compute_oa(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute overall accuracy for a single batch.

    Args:
        preds: Binary predictions ``[B, 1, H, W]``.
        targets: Ground-truth masks ``[B, 1, H, W]``.

    Returns:
        Overall accuracy as a float in [0, 1].
    """
    return _quick_cm(preds, targets).compute()["oa"]


# ---------------------------------------------------------------------------
# High-level tracker (accepts raw logits)
# ---------------------------------------------------------------------------

class MetricTracker:
    """End-to-end metric tracker for training / validation loops.

    Wraps a ``ConfusionMatrix`` and transparently applies sigmoid +
    thresholding to raw model logits before accumulating counts.

    Args:
        threshold: Decision threshold applied after sigmoid (default 0.5).

    Example::

        tracker = MetricTracker(threshold=0.5)
        for batch in val_loader:
            logits = model(batch["A"], batch["B"])
            tracker.update(logits, batch["mask"])
        results = tracker.compute()   # {"f1": ..., "iou": ..., ...}
        tracker.reset()
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.cm = ConfusionMatrix()

    def reset(self) -> None:
        """Reset the internal confusion matrix."""
        self.cm.reset()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Apply sigmoid + threshold and accumulate counts.

        This method is wrapped with ``@torch.no_grad()`` so it can be
        called safely inside a validation loop without affecting autograd.
        All operations run on the input tensor's device.

        Args:
            logits: Raw model output ``[B, 1, H, W]`` (pre-sigmoid).
            targets: Binary ground-truth masks ``[B, 1, H, W]`` with
                values in {0, 1}.
        """
        preds = (torch.sigmoid(logits) >= self.threshold).float()
        self.cm.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from accumulated counts.

        Returns:
            Dict with keys ``'f1'``, ``'iou'``, ``'precision'``, ``'recall'``,
            ``'oa'``.
        """
        return self.cm.compute()
