"""Run inference on arbitrary before/after image pairs.

Loads a trained change detection model and produces binary change masks
for new satellite image pairs.

Usage:
    python inference.py --before path/to/before.png --after path/to/after.png \
        --model changeformer --checkpoint checkpoints/changeformer_best.pth
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from data.dataset import IMAGENET_MEAN, IMAGENET_STD
from models import get_model
from utils.visualization import overlay_changes, plot_prediction

logger = logging.getLogger(__name__)


def preprocess_image(
    image_path: Path,
    patch_size: int = 256,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load and preprocess a single image for inference.

    Reads the image, pads to a multiple of patch_size, and applies
    ImageNet normalization.

    Args:
        image_path: Path to the input image.
        patch_size: Patch size the model expects.

    Returns:
        Tuple of (preprocessed tensor [1, 3, H, W], original (H, W)).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    # Pad to multiple of patch_size
    pad_h = (patch_size - orig_h % patch_size) % patch_size
    pad_w = (patch_size - orig_w % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    # Normalize
    img = img.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = (img - mean) / std

    # HWC -> CHW, add batch dim
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, (orig_h, orig_w)


def sliding_window_inference(
    model: nn.Module,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    patch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Run inference using sliding window for large images.

    Splits images into non-overlapping patches, runs model on each,
    and stitches results back together.

    Args:
        model: Trained change detection model.
        img_a: Before image tensor [1, 3, H, W].
        img_b: After image tensor [1, 3, H, W].
        patch_size: Size of each patch.
        device: Inference device.

    Returns:
        Probability map [1, 1, H, W] (after sigmoid).
    """
    _, _, h, w = img_a.shape
    output = torch.zeros(1, 1, h, w, device="cpu")

    model.eval()
    with torch.no_grad():
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch_a = img_a[:, :, y:y + patch_size, x:x + patch_size].to(device)
                patch_b = img_b[:, :, y:y + patch_size, x:x + patch_size].to(device)

                logits = model(patch_a, patch_b)
                probs = torch.sigmoid(logits).cpu()
                output[:, :, y:y + patch_size, x:x + patch_size] = probs

    return output


def save_change_mask(
    mask: np.ndarray,
    save_path: Path,
    threshold: float = 0.5,
) -> None:
    """Save binary change mask as an image.

    Args:
        mask: Probability map [H, W] with values in [0, 1].
        save_path: Output file path.
        threshold: Binarization threshold.
    """
    binary = (mask > threshold).astype(np.uint8) * 255
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), binary)
    logger.info("Saved change mask: %s", save_path)


def main() -> None:
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description="Run change detection inference")
    parser.add_argument("--before", type=Path, required=True, help="Path to before image")
    parser.add_argument("--after", type=Path, required=True, help="Path to after image")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/inference"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = args.model or config["model"]["name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = config.get("dataset", {}).get("patch_size", 256)

    # Load model
    model = get_model(model_name, config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded model '%s' from %s", model_name, args.checkpoint)

    # Preprocess images
    img_a, (orig_h, orig_w) = preprocess_image(args.before, patch_size)
    img_b, _ = preprocess_image(args.after, patch_size)

    # Run inference
    prob_map = sliding_window_inference(model, img_a, img_b, patch_size, device)

    # Crop back to original size and save
    prob_map = prob_map[:, :, :orig_h, :orig_w]
    mask_np = prob_map.squeeze().numpy()

    args.output.mkdir(parents=True, exist_ok=True)
    save_change_mask(mask_np, args.output / "change_mask.png", args.threshold)

    # Save overlay visualization
    overlay = overlay_changes(img_b.squeeze()[:, :orig_h, :orig_w], prob_map.squeeze(0))
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    cv2.imwrite(str(args.output / "overlay.png"), cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR))
    logger.info("Saved overlay: %s", args.output / "overlay.png")


if __name__ == "__main__":
    main()
