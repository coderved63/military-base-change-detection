"""Download and preprocess change detection datasets.

Supports LEVIR-CD (primary) and WHU-CD (secondary).  Downloads from Google
Drive via ``gdown``, extracts archives, crops 1024x1024 images into 256x256
non-overlapping patches, and organises into train/val/test splits.

LEVIR-CD expected raw structure after extraction::

    raw_dir/
    └── LEVIR-CD/
        ├── train/
        │   ├── A/          # before images  (1024x1024)
        │   ├── B/          # after images   (1024x1024)
        │   └── label/      # binary masks   (0/255)
        ├── val/
        │   ├── A/
        │   ├── B/
        │   └── label/
        └── test/
            ├── A/
            ├── B/
            └── label/

Usage:
    # Full pipeline: download + crop
    python data/download.py --dataset levir-cd --raw_dir ./raw_data --out_dir ./processed_data

    # Skip download (data already on disk), just crop
    python data/download.py --dataset levir-cd --raw_dir ./raw_data --out_dir ./processed_data --skip_download

    # On Colab — save processed patches to Drive
    python data/download.py --dataset levir-cd --raw_dir /content/raw_data \
        --out_dir /content/drive/MyDrive/change-detection/processed_data
"""

import argparse
import logging
import shutil
import zipfile
from pathlib import Path
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google Drive file IDs for LEVIR-CD
# These are publicly shared links from the dataset authors.
# If they break, download manually from:
#   https://github.com/justchenhao/LEVIR-CD
# ---------------------------------------------------------------------------
_LEVIR_CD_GDRIVE_IDS = {
    # The dataset is often shared as a single zip or split zips.
    # Update these IDs if the authors change the links.
    "full": "1RUFY9QDmVBfHuMRwYze7C5BlVsMr3Xm_",
}

_WHU_CD_GDRIVE_IDS = {
    "full": "1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_from_gdrive(file_id: str, output_path: Path) -> None:
    """Download a file from Google Drive using gdown.

    Args:
        file_id: Google Drive file ID.
        output_path: Local path to save the downloaded file.
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for downloading. Install with: pip install gdown"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info("Downloading from Google Drive (ID: %s) ...", file_id)
    gdown.download(url, str(output_path), quiet=False)
    logger.info("Downloaded: %s", output_path)


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip archive.

    Args:
        zip_path: Path to the zip file.
        extract_to: Directory to extract into.
    """
    logger.info("Extracting %s -> %s", zip_path.name, extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    logger.info("Extraction complete.")


def download_levir_cd(raw_dir: Path) -> Path:
    """Download the LEVIR-CD dataset from Google Drive.

    Downloads the zip, extracts it, and returns the path to the extracted
    dataset root.

    Args:
        raw_dir: Directory to save downloads and extracted data.

    Returns:
        Path to the extracted LEVIR-CD root directory.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "LEVIR-CD.zip"

    # Skip download if zip already exists
    if zip_path.exists():
        logger.info("LEVIR-CD zip already exists: %s", zip_path)
    else:
        _download_from_gdrive(_LEVIR_CD_GDRIVE_IDS["full"], zip_path)

    # Extract if not already extracted
    dataset_root = raw_dir / "LEVIR-CD"
    if dataset_root.exists() and any(dataset_root.iterdir()):
        logger.info("LEVIR-CD already extracted: %s", dataset_root)
    else:
        _extract_zip(zip_path, raw_dir)

    # Some zips have an extra nested folder — find the actual root
    dataset_root = _find_dataset_root(raw_dir, "LEVIR-CD")
    logger.info("LEVIR-CD root: %s", dataset_root)
    return dataset_root


def download_whu_cd(raw_dir: Path) -> Path:
    """Download the WHU-CD dataset from Google Drive.

    Args:
        raw_dir: Directory to save downloads and extracted data.

    Returns:
        Path to the extracted WHU-CD root directory.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "WHU-CD.zip"

    if zip_path.exists():
        logger.info("WHU-CD zip already exists: %s", zip_path)
    else:
        _download_from_gdrive(_WHU_CD_GDRIVE_IDS["full"], zip_path)

    dataset_root = raw_dir / "WHU-CD"
    if dataset_root.exists() and any(dataset_root.iterdir()):
        logger.info("WHU-CD already extracted: %s", dataset_root)
    else:
        _extract_zip(zip_path, raw_dir)

    dataset_root = _find_dataset_root(raw_dir, "WHU-CD")
    logger.info("WHU-CD root: %s", dataset_root)
    return dataset_root


def _find_dataset_root(parent: Path, name_hint: str) -> Path:
    """Locate the actual dataset root after extraction.

    Handles cases where the zip creates a nested folder like
    ``LEVIR-CD/LEVIR-CD/`` or the root is directly under ``parent``.

    Args:
        parent: Directory where the zip was extracted.
        name_hint: Expected folder name (e.g. ``'LEVIR-CD'``).

    Returns:
        Path to the directory containing ``train/``, ``val/``, ``test/``
        (or the closest match).
    """
    candidate = parent / name_hint
    if not candidate.exists():
        # Try to find it by scanning
        for d in parent.rglob(name_hint):
            if d.is_dir():
                candidate = d
                break

    # Check for nested structure
    nested = candidate / name_hint
    if nested.exists() and nested.is_dir():
        candidate = nested

    # Look for the split directories
    for d in [candidate] + list(candidate.iterdir()) if candidate.exists() else []:
        if isinstance(d, Path) and d.is_dir():
            if (d / "train").exists() or (d / "A").exists():
                return d

    return candidate


# ---------------------------------------------------------------------------
# Patch cropping
# ---------------------------------------------------------------------------

def crop_to_patches(
    image: np.ndarray,
    patch_size: int = 256,
) -> List[np.ndarray]:
    """Crop an image into non-overlapping square patches.

    Pixels that don't fit into a full patch at the right/bottom edges are
    discarded (e.g. a 1024x1024 image produces 16 patches of 256x256).

    Args:
        image: Input image of shape ``(H, W)`` or ``(H, W, C)``.
        patch_size: Side length of each square patch.

    Returns:
        List of cropped patches.
    """
    h, w = image.shape[:2]
    patches: List[np.ndarray] = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patches.append(image[y : y + patch_size, x : x + patch_size])
    return patches


def process_split(
    raw_dir: Path,
    out_dir: Path,
    split: str,
    patch_size: int = 256,
) -> int:
    """Process one dataset split: crop all images into patches.

    Reads 1024x1024 image triplets (A, B, label) from ``raw_dir/{split}/``,
    crops each into 256x256 patches, and saves to ``out_dir/{split}/``.

    Args:
        raw_dir: Root of the raw LEVIR-CD dataset (contains ``train/``,
            ``val/``, ``test/`` sub-folders).
        out_dir: Output root for processed patches.
        split: One of ``'train'``, ``'val'``, ``'test'``.
        patch_size: Patch size in pixels.

    Returns:
        Total number of patch triplets generated for this split.
    """
    split_in = raw_dir / split
    split_out = out_dir / split

    # Input directories
    dir_a_in = split_in / "A"
    dir_b_in = split_in / "B"
    dir_label_in = split_in / "label"

    if not dir_a_in.exists():
        logger.warning("Input directory missing: %s — skipping split '%s'", dir_a_in, split)
        return 0

    # Output directories
    dir_a_out = split_out / "A"
    dir_b_out = split_out / "B"
    dir_label_out = split_out / "label"
    for d in [dir_a_out, dir_b_out, dir_label_out]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect image filenames
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    filenames = sorted([
        f.name for f in dir_a_in.iterdir()
        if f.suffix.lower() in extensions
    ])
    logger.info("  %s: found %d images to crop", split, len(filenames))

    total_patches = 0

    for fname in filenames:
        # Read triplet
        img_a = cv2.imread(str(dir_a_in / fname), cv2.IMREAD_COLOR)
        img_b = cv2.imread(str(dir_b_in / fname), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(dir_label_in / fname), cv2.IMREAD_GRAYSCALE)

        if img_a is None or img_b is None or mask is None:
            logger.warning("  Skipping %s (could not read one or more files)", fname)
            continue

        # Crop into patches
        patches_a = crop_to_patches(img_a, patch_size)
        patches_b = crop_to_patches(img_b, patch_size)
        patches_m = crop_to_patches(mask, patch_size)

        stem = Path(fname).stem

        for idx, (pa, pb, pm) in enumerate(zip(patches_a, patches_b, patches_m)):
            patch_name = f"{stem}_{idx:04d}.png"
            cv2.imwrite(str(dir_a_out / patch_name), pa)
            cv2.imwrite(str(dir_b_out / patch_name), pb)
            cv2.imwrite(str(dir_label_out / patch_name), pm)

        total_patches += len(patches_a)

    logger.info("  %s: generated %d patch triplets", split, total_patches)
    return total_patches


# ---------------------------------------------------------------------------
# Check for pre-cropped dataset
# ---------------------------------------------------------------------------

def is_already_cropped(data_dir: Path) -> bool:
    """Check if a directory already contains processed (cropped) patches.

    A directory is considered processed if it has ``train/A/`` with at least
    one image file inside.

    Args:
        data_dir: Path to check.

    Returns:
        ``True`` if processed patches are present.
    """
    train_a = data_dir / "train" / "A"
    if not train_a.exists():
        return False
    extensions = {".png", ".jpg", ".tif"}
    return any(f.suffix.lower() in extensions for f in train_a.iterdir())


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_dataset(
    dataset: str,
    raw_dir: Path,
    out_dir: Path,
    patch_size: int = 256,
) -> None:
    """Run the full preprocessing pipeline for a dataset.

    Args:
        dataset: Dataset name (``'levir-cd'`` or ``'whu-cd'``).
        raw_dir: Directory containing the raw (extracted) dataset.
        out_dir: Output directory for processed patches.
        patch_size: Patch size in pixels.
    """
    # Check if output already exists
    if is_already_cropped(out_dir):
        logger.info("Processed data already exists at %s — skipping.", out_dir)
        logger.info("Delete the directory or use a different --out_dir to re-process.")
        return

    logger.info("Preprocessing %s: %s -> %s (patch_size=%d)", dataset, raw_dir, out_dir, patch_size)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in ["train", "val", "test"]:
        count = process_split(raw_dir, out_dir, split, patch_size)
        total += count

    logger.info("=" * 50)
    logger.info("Preprocessing complete: %d total patch triplets", total)
    logger.info("Output: %s", out_dir)
    logger.info("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for dataset download and preprocessing."""
    parser = argparse.ArgumentParser(
        description="Download and preprocess change detection datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (download + crop)
  python data/download.py --dataset levir-cd --raw_dir ./raw_data --out_dir ./processed_data

  # Already downloaded — just crop
  python data/download.py --dataset levir-cd --raw_dir ./raw_data --out_dir ./processed_data --skip_download

  # Colab: save to Drive
  python data/download.py --dataset levir-cd --raw_dir /content/raw_data \\
      --out_dir /content/drive/MyDrive/change-detection/processed_data
        """,
    )
    parser.add_argument(
        "--dataset", type=str, default="levir-cd",
        choices=["levir-cd", "whu-cd"],
        help="Dataset to download and preprocess (default: levir-cd).",
    )
    parser.add_argument(
        "--raw_dir", type=Path, default=Path("./raw_data"),
        help="Directory for raw downloads and extracted data.",
    )
    parser.add_argument(
        "--out_dir", type=Path, default=Path("./processed_data"),
        help="Output directory for processed 256x256 patches.",
    )
    parser.add_argument(
        "--patch_size", type=int, default=256,
        help="Patch size for cropping (default: 256).",
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip download step — only run preprocessing on existing data.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Step 1: Download (unless skipped)
    dataset_root = args.raw_dir
    if not args.skip_download:
        logger.info("Step 1: Downloading %s ...", args.dataset)
        if args.dataset == "levir-cd":
            dataset_root = download_levir_cd(args.raw_dir)
        elif args.dataset == "whu-cd":
            dataset_root = download_whu_cd(args.raw_dir)
    else:
        logger.info("Step 1: Download skipped (--skip_download)")
        # Try to find the dataset root in raw_dir
        if args.dataset == "levir-cd":
            dataset_root = _find_dataset_root(args.raw_dir, "LEVIR-CD")
        elif args.dataset == "whu-cd":
            dataset_root = _find_dataset_root(args.raw_dir, "WHU-CD")

    # Step 2: Preprocess (crop into patches)
    logger.info("Step 2: Cropping into %dx%d patches ...", args.patch_size, args.patch_size)
    preprocess_dataset(args.dataset, dataset_root, args.out_dir, args.patch_size)


if __name__ == "__main__":
    main()
