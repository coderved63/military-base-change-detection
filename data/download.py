"""Download and preprocess change detection datasets.

Supports LEVIR-CD and WHU-CD datasets. Downloads raw data, crops 1024x1024
images into 256x256 non-overlapping patches, and organizes into train/val/test
splits.

Usage:
    python data/download.py --dataset levir-cd --raw_dir ./raw_data --out_dir ./processed_data
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def download_levir_cd(raw_dir: Path) -> None:
    """Download the LEVIR-CD dataset.

    Args:
        raw_dir: Directory to save the raw downloaded files.
    """
    # TODO: Implement download via gdown or direct URL
    raise NotImplementedError("LEVIR-CD download not yet implemented")


def download_whu_cd(raw_dir: Path) -> None:
    """Download the WHU-CD dataset.

    Args:
        raw_dir: Directory to save the raw downloaded files.
    """
    # TODO: Implement download
    raise NotImplementedError("WHU-CD download not yet implemented")


def crop_to_patches(
    image: np.ndarray,
    patch_size: int = 256,
) -> list[np.ndarray]:
    """Crop an image into non-overlapping patches.

    Args:
        image: Input image of shape (H, W) or (H, W, C).
        patch_size: Size of each square patch.

    Returns:
        List of cropped patches.
    """
    h, w = image.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = image[y : y + patch_size, x : x + patch_size]
            patches.append(patch)
    return patches


def process_split(
    raw_dir: Path,
    out_dir: Path,
    split: str,
    patch_size: int = 256,
) -> int:
    """Process a single dataset split (train/val/test).

    Reads image pairs and masks from raw_dir, crops into patches, and
    saves to out_dir.

    Args:
        raw_dir: Root directory of the raw dataset.
        out_dir: Output directory for processed patches.
        split: One of 'train', 'val', 'test'.
        patch_size: Size of each square patch.

    Returns:
        Number of patch triplets generated.
    """
    # TODO: Implement processing pipeline
    raise NotImplementedError("Split processing not yet implemented")


def preprocess_dataset(
    dataset: str,
    raw_dir: Path,
    out_dir: Path,
    patch_size: int = 256,
) -> None:
    """Run full preprocessing pipeline for a dataset.

    Args:
        dataset: Dataset name ('levir-cd' or 'whu-cd').
        raw_dir: Directory containing raw downloaded data.
        out_dir: Output directory for processed patches.
        patch_size: Size of each square patch.
    """
    logger.info("Preprocessing %s: %s -> %s", dataset, raw_dir, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        count = process_split(raw_dir, out_dir, split, patch_size)
        logger.info("  %s: %d patch triplets", split, count)


def main() -> None:
    """CLI entry point for dataset download and preprocessing."""
    parser = argparse.ArgumentParser(description="Download and preprocess change detection datasets")
    parser.add_argument("--dataset", type=str, default="levir-cd", choices=["levir-cd", "whu-cd"])
    parser.add_argument("--raw_dir", type=Path, default=Path("./raw_data"))
    parser.add_argument("--out_dir", type=Path, default=Path("./processed_data"))
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--skip_download", action="store_true", help="Skip download, only preprocess")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.skip_download:
        if args.dataset == "levir-cd":
            download_levir_cd(args.raw_dir)
        elif args.dataset == "whu-cd":
            download_whu_cd(args.raw_dir)

    preprocess_dataset(args.dataset, args.raw_dir, args.out_dir, args.patch_size)


if __name__ == "__main__":
    main()
