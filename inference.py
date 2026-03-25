"""Run change-detection inference on arbitrary before/after image pairs.

Handles images of any resolution by tiling into 256x256 patches, running the
model on each patch, and stitching the probability map back together.  Outputs
a binary change mask PNG, an overlay visualisation, and prints the percentage
of area changed.

Usage:
    python inference.py --before path/to/before.png --after path/to/after.png \
        --model changeformer --checkpoint checkpoints/changeformer_best.pth

    python inference.py --before big_before.tif --after big_after.tif \
        --checkpoint checkpoints/unet_pp_best.pth --output results/
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

from data.dataset import IMAGENET_MEAN, IMAGENET_STD
from models import get_model
from utils.visualization import overlay_changes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(
    image_path: Path,
    patch_size: int = 256,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load an image from disk, pad to a patch-size multiple, and normalise.

    Args:
        image_path: Path to the input image (any format OpenCV supports).
        patch_size: Spatial size the model expects per patch.

    Returns:
        Tuple of ``(tensor, original_size)`` where tensor has shape
        ``[1, 3, H_padded, W_padded]`` and ``original_size`` is
        ``(orig_h, orig_w)`` before padding.

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = img.shape[:2]
    logger.info("Loaded %s (%d x %d)", image_path.name, orig_w, orig_h)

    # Pad to the nearest multiple of patch_size using reflection
    pad_h = (patch_size - orig_h % patch_size) % patch_size
    pad_w = (patch_size - orig_w % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    # uint8 → float32 [0,1] → ImageNet normalisation
    img = img.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = (img - mean) / std

    # HWC → CHW, add batch dim
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, (orig_h, orig_w)


# ---------------------------------------------------------------------------
# Tiled (sliding-window) inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def sliding_window_inference(
    model: nn.Module,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    patch_size: int = 256,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Run inference by tiling large images into non-overlapping patches.

    Each patch pair is fed through the model independently; the resulting
    probability maps are stitched back into a single full-resolution output.

    Args:
        model: Trained change-detection model (set to eval internally).
        img_a: Before image ``[1, 3, H, W]`` (padded to patch-size multiples).
        img_b: After image ``[1, 3, H, W]`` (same spatial size as ``img_a``).
        patch_size: Tile size in pixels.
        device: Inference device (CUDA or CPU).

    Returns:
        Probability map ``[1, 1, H, W]`` with values in ``[0, 1]`` (after
        sigmoid), on CPU.
    """
    model.eval()
    _, _, h, w = img_a.shape
    output = torch.zeros(1, 1, h, w)

    n_tiles = (h // patch_size) * (w // patch_size)
    tile_idx = 0

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch_a = img_a[:, :, y:y + patch_size, x:x + patch_size].to(device)
            patch_b = img_b[:, :, y:y + patch_size, x:x + patch_size].to(device)

            logits = model(patch_a, patch_b)
            probs = torch.sigmoid(logits).cpu()
            output[:, :, y:y + patch_size, x:x + patch_size] = probs

            tile_idx += 1

    logger.info("Inference complete: %d tiles processed", n_tiles)
    return output


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_binary_mask(
    prob_map: np.ndarray,
    save_path: Path,
    threshold: float = 0.5,
) -> None:
    """Binarise a probability map and save as a PNG.

    Args:
        prob_map: Probability values ``[H, W]`` in ``[0, 1]``.
        save_path: Destination file path.
        threshold: Decision threshold.
    """
    binary = (prob_map > threshold).astype(np.uint8) * 255
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), binary)
    logger.info("Saved binary mask: %s", save_path)


def save_overlay(
    img_b_tensor: torch.Tensor,
    pred_tensor: torch.Tensor,
    save_path: Path,
    threshold: float = 0.5,
) -> None:
    """Create and save an overlay visualisation.

    Args:
        img_b_tensor: After image ``[3, H, W]`` (ImageNet-normalised).
        pred_tensor: Prediction mask ``[1, H, W]`` (probability).
        save_path: Destination file path.
        threshold: Binarisation threshold applied before overlay.
    """
    binary_pred = (pred_tensor >= threshold).float()
    overlay_rgb = overlay_changes(
        img_after=img_b_tensor,
        mask_pred=binary_pred,
        alpha=0.4,
        color=(255, 0, 0),
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    logger.info("Saved overlay: %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point — parse CLI args, run inference, save outputs."""
    parser = argparse.ArgumentParser(
        description="Run change-detection inference on a before/after image pair",
    )
    parser.add_argument(
        "--before", type=Path, required=True,
        help="Path to the *before* image.",
    )
    parser.add_argument(
        "--after", type=Path, required=True,
        help="Path to the *after* image.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (overrides config). One of: siamese_cnn, unet_pp, changeformer.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the model checkpoint (.pth).",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/inference"),
        help="Output directory for results.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Binarisation threshold (default: from config).",
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
    patch_size: int = config.get("dataset", {}).get("patch_size", 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | Model: %s | Threshold: %.2f", device, model_name, threshold)

    # ---- Load model ---------------------------------------------------
    model = get_model(model_name, config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(
        "Loaded checkpoint: %s (epoch %d)",
        args.checkpoint, ckpt.get("epoch", -1),
    )

    # ---- Preprocess images --------------------------------------------
    img_a, (orig_h, orig_w) = load_and_preprocess(args.before, patch_size)
    img_b, (orig_h_b, orig_w_b) = load_and_preprocess(args.after, patch_size)

    if (orig_h, orig_w) != (orig_h_b, orig_w_b):
        logger.warning(
            "Image sizes differ: before=(%d,%d) after=(%d,%d). "
            "Using before dimensions for cropping.",
            orig_h, orig_w, orig_h_b, orig_w_b,
        )

    # ---- Run tiled inference ------------------------------------------
    prob_map = sliding_window_inference(model, img_a, img_b, patch_size, device)

    # Crop back to original resolution (remove padding)
    prob_map = prob_map[:, :, :orig_h, :orig_w]
    prob_np = prob_map.squeeze().numpy()  # [H, W]

    # ---- Compute change statistics ------------------------------------
    binary_np = (prob_np > threshold).astype(np.float32)
    total_pixels = orig_h * orig_w
    changed_pixels = int(binary_np.sum())
    pct_changed = (changed_pixels / total_pixels) * 100.0

    logger.info("=" * 50)
    logger.info("  CHANGE DETECTION RESULTS")
    logger.info("=" * 50)
    logger.info("  Image size     : %d x %d", orig_w, orig_h)
    logger.info("  Total pixels   : %d", total_pixels)
    logger.info("  Changed pixels : %d", changed_pixels)
    logger.info("  Area changed   : %.2f%%", pct_changed)
    logger.info("=" * 50)

    # ---- Save outputs -------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Binary change mask
    save_binary_mask(prob_np, output_dir / "change_mask.png", threshold)

    # Overlay visualisation
    img_b_cropped = img_b.squeeze()[:, :orig_h, :orig_w]  # [3, H, W]
    pred_cropped = prob_map.squeeze(0)[:, :orig_h, :orig_w]  # [1, H, W]
    save_overlay(img_b_cropped, pred_cropped, output_dir / "overlay.png", threshold)

    logger.info("All outputs saved to: %s", output_dir)


if __name__ == "__main__":
    main()
