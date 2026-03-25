"""Evaluate a trained change-detection model on the test set.

Computes all metrics (F1, IoU, Precision, Recall, OA), saves a
``results.json``, generates a 20-sample prediction grid, and produces
overlay images for the top-10 predictions with the largest predicted
change area.

Usage:
    python evaluate.py --config configs/config.yaml \
        --checkpoint checkpoints/unet_pp_best.pth

    python evaluate.py --config configs/config.yaml \
        --checkpoint checkpoints/changeformer_best.pth \
        --model changeformer --output_dir ./my_outputs
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from data.dataset import ChangeDetectionDataset
from models import get_model
from utils.metrics import MetricTracker
from utils.visualization import overlay_changes, plot_prediction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU / batch-size helpers
# ---------------------------------------------------------------------------

def _detect_gpu_type() -> str:
    """Detect the current GPU type for batch-size selection.

    Returns:
        One of ``'T4'``, ``'V100'``, or ``'default'``.
    """
    if not torch.cuda.is_available():
        return "default"
    name = torch.cuda.get_device_name(0).upper()
    if "T4" in name:
        return "T4"
    elif "V100" in name:
        return "V100"
    return "default"


def get_train_batch_size(config: Dict[str, Any], model_name: str) -> int:
    """Look up the *training* batch size for the current GPU + model.

    Args:
        config: Full project config dict.
        model_name: Model identifier string.

    Returns:
        Training batch size as an integer.
    """
    gpu_type = _detect_gpu_type()
    model_sizes = config.get("batch_sizes", {}).get(model_name, {})
    return model_sizes.get(gpu_type, model_sizes.get("default", 4))


# ---------------------------------------------------------------------------
# Path resolution (same logic as train.py)
# ---------------------------------------------------------------------------

def resolve_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Build a path dict based on whether Colab mode is enabled.

    Args:
        config: Full project config dict.

    Returns:
        Dict with keys ``'data'``, ``'checkpoints'``, ``'logs'``,
        ``'outputs'``.
    """
    if config.get("colab", {}).get("enabled", False):
        c = config["colab"]
        return {
            "data": Path(c["data_dir"]),
            "checkpoints": Path(c["checkpoint_dir"]),
            "logs": Path(c["log_dir"]),
            "outputs": Path(c["output_dir"]),
        }

    p = config.get("paths", {})
    return {
        "data": Path(p.get("processed_data", "./processed_data")),
        "checkpoints": Path(p.get("checkpoint_dir", "./checkpoints")),
        "logs": Path(p.get("log_dir", "./logs")),
        "outputs": Path(p.get("output_dir", "./outputs")),
    }


# ---------------------------------------------------------------------------
# Evaluation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tracker: MetricTracker,
) -> Tuple[Dict[str, float], List[Dict[str, torch.Tensor]]]:
    """Run inference on the full test set and collect per-sample data.

    Args:
        model: Trained change-detection model (set to eval internally).
        loader: Test ``DataLoader``.
        device: Target device.
        tracker: ``MetricTracker`` (reset externally before this call).

    Returns:
        Tuple of ``(metrics_dict, samples_list)``.
        Each entry in ``samples_list`` is a dict with keys
        ``'A'``, ``'B'``, ``'mask'``, ``'pred'``, ``'change_area'``
        (all single-sample tensors on CPU).
    """
    model.eval()
    all_samples: List[Dict[str, Any]] = []

    for batch in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
        img_a = batch["A"].to(device, non_blocking=True)
        img_b = batch["B"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        logits = model(img_a, img_b)
        tracker.update(logits, mask)

        preds = (torch.sigmoid(logits) >= tracker.threshold).float()

        # Store each sample for later visualisation / ranking
        for i in range(img_a.size(0)):
            pred_i = preds[i].cpu()
            change_area = pred_i.sum().item()
            all_samples.append({
                "A": img_a[i].cpu(),
                "B": img_b[i].cpu(),
                "mask": mask[i].cpu(),
                "pred": pred_i,
                "change_area": change_area,
            })

    metrics = tracker.compute()
    return metrics, all_samples


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def save_prediction_grid(
    samples: List[Dict[str, torch.Tensor]],
    save_path: Path,
    num_rows: int = 5,
) -> None:
    """Save a grid of sample predictions (Before | After | GT | Pred).

    Args:
        samples: List of per-sample dicts from ``run_evaluation``.
        save_path: Destination image path.
        num_rows: Number of rows in the grid (4 columns each).
    """
    num_samples = min(num_rows, len(samples))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    from utils.visualization import _denorm_tensor, _mask_to_numpy

    col_titles = ["Before (A)", "After (B)", "Ground Truth", "Prediction"]

    for row in range(num_samples):
        s = samples[row]
        images = [
            _denorm_tensor(s["A"]),
            _denorm_tensor(s["B"]),
            _mask_to_numpy(s["mask"]),
            (_mask_to_numpy(s["pred"]) > 0.5).astype(np.float32),
        ]
        cmaps = [None, None, "gray", "gray"]

        for col in range(4):
            ax = axes[row, col]
            ax.imshow(images[col], cmap=cmaps[col], vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=12)

    fig.tight_layout(pad=1.0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved prediction grid (%d samples): %s", num_samples, save_path)


def save_top_overlays(
    samples: List[Dict[str, torch.Tensor]],
    output_dir: Path,
    top_k: int = 10,
) -> None:
    """Save overlay images for the top-K predictions by predicted change area.

    Args:
        samples: List of per-sample dicts from ``run_evaluation``.
        output_dir: Directory to save overlay PNGs.
        top_k: Number of overlays to save.
    """
    import cv2

    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Sort by predicted change area (descending) — most "interesting" first
    ranked = sorted(samples, key=lambda s: s["change_area"], reverse=True)
    num = min(top_k, len(ranked))

    for idx in range(num):
        s = ranked[idx]
        overlay_img = overlay_changes(
            img_after=s["B"],
            mask_pred=s["pred"],
            alpha=0.4,
            color=(255, 0, 0),
        )
        save_file = overlay_dir / f"top_{idx + 1:02d}_area_{s['change_area']:.0f}.png"
        cv2.imwrite(str(save_file), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

    logger.info("Saved %d overlay images: %s", num, overlay_dir)


# ---------------------------------------------------------------------------
# Console formatting
# ---------------------------------------------------------------------------

def print_metrics_table(
    metrics: Dict[str, float],
    model_name: str,
    checkpoint_path: Path,
    epoch: int,
) -> None:
    """Print a formatted metrics table to the console.

    Args:
        metrics: Dict of metric name to value.
        model_name: Model architecture name.
        checkpoint_path: Path to the loaded checkpoint.
        epoch: Training epoch the checkpoint was saved at.
    """
    border = "=" * 50
    logger.info(border)
    logger.info("  TEST SET RESULTS")
    logger.info(border)
    logger.info("  Model      : %s", model_name)
    logger.info("  Checkpoint : %s", checkpoint_path)
    logger.info("  Epoch      : %d", epoch)
    logger.info(border)
    logger.info("  %-12s  %s", "METRIC", "VALUE")
    logger.info("  " + "-" * 24)
    for name, value in metrics.items():
        logger.info("  %-12s  %.4f", name.upper(), value)
    logger.info(border)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point — parse CLI args, evaluate model, save outputs."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained change-detection model on the test set",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the model checkpoint (.pth).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override the model name from config.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Override the output directory (default: from config).",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override the binarisation threshold (default: from config).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---- Config -------------------------------------------------------
    with open(args.config, "r") as fh:
        config: Dict[str, Any] = yaml.safe_load(fh)

    model_name: str = args.model or config["model"]["name"]
    threshold: float = args.threshold or config.get("evaluation", {}).get("threshold", 0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Paths --------------------------------------------------------
    paths = resolve_paths(config)
    output_dir = args.output_dir or paths["outputs"]
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ---------------------------------------------------
    model = get_model(model_name, config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    ckpt_epoch = ckpt.get("epoch", -1)
    ckpt_f1 = ckpt.get("best_f1", -1.0)
    logger.info(
        "Loaded checkpoint: %s (epoch %d, best F1 %.4f)",
        args.checkpoint, ckpt_epoch, ckpt_f1,
    )

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model: %s (%.2fM parameters)", model_name, param_count)

    # ---- Test data ----------------------------------------------------
    # No gradients stored during eval → safe to use 2x training batch size
    train_bs = get_train_batch_size(config, model_name)
    eval_bs = train_bs * 2

    ds_cfg = config.get("dataset", {})
    test_ds = ChangeDetectionDataset(
        root=paths["data"] / "test", split="test", config=config,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=ds_cfg.get("pin_memory", True),
    )
    logger.info(
        "Test set: %d samples, %d batches (batch_size=%d, 2x train)",
        len(test_ds), len(test_loader), eval_bs,
    )

    # ---- Run evaluation -----------------------------------------------
    tracker = MetricTracker(threshold=threshold)
    wall_start = time.monotonic()

    metrics, all_samples = run_evaluation(model, test_loader, device, tracker)

    eval_time = time.monotonic() - wall_start
    logger.info("Evaluation completed in %.1fs", eval_time)

    # ---- Print formatted table ----------------------------------------
    print_metrics_table(metrics, model_name, args.checkpoint, ckpt_epoch)

    # ---- Save results.json --------------------------------------------
    results = {
        "model": model_name,
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt_epoch,
        "threshold": threshold,
        "num_test_samples": len(test_ds),
        "eval_time_seconds": round(eval_time, 2),
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results: %s", results_path)

    # ---- Prediction grid (20 samples, 5 rows x 4 cols) ----------------
    save_prediction_grid(
        samples=all_samples,
        save_path=output_dir / "prediction_grid.png",
        num_rows=min(5, len(all_samples)),
    )

    # ---- Individual sample plots (up to 20) ---------------------------
    vis_dir = output_dir / "predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)
    num_individual = min(20, len(all_samples))
    for idx in range(num_individual):
        s = all_samples[idx]
        plot_prediction(
            img_a=s["A"],
            img_b=s["B"],
            mask_true=s["mask"],
            mask_pred=s["pred"],
            filename=vis_dir / f"sample_{idx + 1:03d}.png",
        )
    logger.info("Saved %d individual prediction plots: %s", num_individual, vis_dir)

    # ---- Top-10 overlay images (by predicted change area) -------------
    save_top_overlays(
        samples=all_samples,
        output_dir=output_dir,
        top_k=10,
    )

    logger.info("All outputs saved to: %s", output_dir)


if __name__ == "__main__":
    main()
