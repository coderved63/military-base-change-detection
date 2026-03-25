"""Evaluation script for change detection models.

Runs a trained model on the test set, computes all metrics, and generates
visualization outputs.

Usage:
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/unet_pp_best.pth
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from data.dataset import ChangeDetectionDataset
from models import get_model
from utils.metrics import ConfusionMatrix
from utils.visualization import plot_prediction

logger = logging.getLogger(__name__)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    output_dir: Path = Path("./outputs"),
    max_vis: int = 20,
) -> Dict[str, float]:
    """Evaluate model on the full test set.

    Args:
        model: Trained change detection model.
        loader: Test DataLoader.
        device: Target device.
        threshold: Binarization threshold for predictions.
        output_dir: Directory to save visualization outputs.
        max_vis: Maximum number of sample predictions to save.

    Returns:
        Dict of metric name -> value.
    """
    model.eval()
    cm = ConfusionMatrix()
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            img_a = batch["A"].to(device)
            img_b = batch["B"].to(device)
            mask = batch["mask"].to(device)

            logits = model(img_a, img_b)
            preds = (torch.sigmoid(logits) > threshold).float()
            cm.update(preds, mask)

            # Save sample visualizations
            if vis_count < max_vis:
                for i in range(min(img_a.size(0), max_vis - vis_count)):
                    plot_prediction(
                        img_a[i], img_b[i], mask[i], preds[i],
                        save_path=vis_dir / f"sample_{vis_count:04d}.png",
                    )
                    vis_count += 1

    metrics = cm.compute()
    return metrics


def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate change detection model")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = args.model or config["model"]["name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = args.threshold or config.get("evaluation", {}).get("threshold", 0.5)

    # Resolve paths
    colab = config.get("colab", {})
    if colab.get("enabled", False):
        data_dir = Path(colab["data_dir"])
        output_dir = Path(colab["output_dir"])
    else:
        data_dir = Path(config["paths"]["processed_data"])
        output_dir = Path(config["paths"]["output_dir"])

    # Model
    model = get_model(model_name, config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded checkpoint: %s (epoch %d, F1 %.4f)",
                args.checkpoint, ckpt.get("epoch", -1), ckpt.get("best_f1", -1))

    # Test data
    ds_cfg = config.get("dataset", {})
    test_ds = ChangeDetectionDataset(data_dir / "test", split="test", config=config)
    test_loader = DataLoader(
        test_ds, batch_size=8, shuffle=False,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=ds_cfg.get("pin_memory", True),
    )

    # Evaluate
    metrics = evaluate(model, test_loader, device, threshold, output_dir)

    # Print results
    logger.info("=" * 50)
    logger.info("TEST SET RESULTS — %s", model_name)
    logger.info("=" * 50)
    for name, value in metrics.items():
        logger.info("  %-12s: %.4f", name.upper(), value)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
