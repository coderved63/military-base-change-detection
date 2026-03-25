"""Google Colab setup script for the change-detection project.

Replaces the manual "Colab Setup Cell" with a single importable function.
Handles Drive mounting, GPU verification, dependency installation, directory
creation, and prints a status summary with checkmarks.

Usage (in a Colab notebook cell):
    !python setup_colab.py

    # Or import and call directly:
    from setup_colab import setup
    paths = setup()
"""

import importlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Packages that must be importable for the project to work.
_REQUIRED_PACKAGES: List[Tuple[str, str]] = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("segmentation_models_pytorch", "segmentation-models-pytorch"),
    ("timm", "timm"),
    ("einops", "einops"),
    ("albumentations", "albumentations"),
    ("cv2", "opencv-python-headless"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("yaml", "PyYAML"),
    ("tqdm", "tqdm"),
    ("tensorboard", "tensorboard"),
    ("gradio", "gradio"),
]


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------

def is_colab() -> bool:
    """Detect whether the code is running inside Google Colab.

    Returns:
        ``True`` if the ``google.colab`` package is importable.
    """
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive() -> bool:
    """Mount Google Drive at ``/content/drive``.

    Skips gracefully when not running in Colab or when already mounted.

    Returns:
        ``True`` if Drive is mounted (or was already), ``False`` otherwise.
    """
    if not is_colab():
        logger.info("  Not running in Colab — Drive mount skipped.")
        return False

    if Path("/content/drive/MyDrive").exists():
        logger.info("  Google Drive already mounted.")
        return True

    from google.colab import drive
    drive.mount("/content/drive")
    logger.info("  Google Drive mounted successfully.")
    return True


def check_gpu() -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """Check GPU availability and return device information.

    Returns:
        Tuple of ``(gpu_name, gpu_type, vram_gb)``.
        All ``None`` if no GPU is available.
    """
    import torch

    if not torch.cuda.is_available():
        return None, None, None

    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9

    name_upper = gpu_name.upper()
    if "T4" in name_upper:
        gpu_type = "T4"
    elif "V100" in name_upper:
        gpu_type = "V100"
    else:
        gpu_type = "other"

    return gpu_name, gpu_type, vram_gb


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def install_requirements() -> bool:
    """Install project dependencies from ``requirements.txt``.

    Returns:
        ``True`` if installation succeeded, ``False`` if the file is missing.
    """
    req_path = Path("requirements.txt")
    if not req_path.exists():
        logger.warning("  requirements.txt not found in %s", Path.cwd())
        return False

    logger.info("  Installing from requirements.txt ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_path)],
        stdout=subprocess.DEVNULL,
    )
    return True


def verify_packages() -> Dict[str, bool]:
    """Check that all required packages are importable.

    Returns:
        Dict mapping import name → ``True`` if importable.
    """
    results: Dict[str, bool] = {}
    for import_name, _ in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
            results[import_name] = True
        except ImportError:
            results[import_name] = False
    return results


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

def create_drive_dirs(
    drive_root: str = "/content/drive/MyDrive/change-detection",
) -> Dict[str, Path]:
    """Create all project directories on Google Drive.

    Args:
        drive_root: Root directory on Drive for this project.

    Returns:
        Dict mapping logical name → ``Path`` for each directory.
    """
    dirs = {
        "root": Path(drive_root),
        "checkpoints": Path(drive_root) / "checkpoints",
        "logs": Path(drive_root) / "logs",
        "outputs": Path(drive_root) / "outputs",
        "processed_data": Path(drive_root) / "processed_data",
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


# ---------------------------------------------------------------------------
# Status summary
# ---------------------------------------------------------------------------

def _tick(ok: bool) -> str:
    """Return a checkmark or cross for console display.

    Args:
        ok: Whether the check passed.

    Returns:
        ``'[OK]'`` or ``'[FAIL]'``.
    """
    return "[OK]" if ok else "[FAIL]"


def print_summary(
    colab: bool,
    drive_mounted: bool,
    gpu_name: Optional[str],
    gpu_type: Optional[str],
    vram_gb: Optional[float],
    deps_ok: bool,
    pkg_results: Dict[str, bool],
    dirs: Dict[str, Path],
) -> None:
    """Print a formatted status summary to the console.

    Args:
        colab: Whether running in Colab.
        drive_mounted: Whether Drive is mounted.
        gpu_name: GPU device name (or ``None``).
        gpu_type: Detected GPU type string.
        vram_gb: VRAM in gigabytes.
        deps_ok: Whether pip install succeeded.
        pkg_results: Per-package import check results.
        dirs: Created directory paths.
    """
    border = "=" * 60
    logger.info(border)
    logger.info("  SETUP STATUS SUMMARY")
    logger.info(border)

    # Environment
    logger.info("  %-30s %s", "Running in Colab", _tick(colab))
    logger.info("  %-30s %s", "Google Drive mounted", _tick(drive_mounted))

    # GPU
    if gpu_name is not None:
        logger.info("  %-30s %s", "GPU detected", _tick(True))
        logger.info("  %-30s %s", "  Device", gpu_name)
        logger.info("  %-30s %s", "  Type (for batch sizing)", gpu_type)
        logger.info("  %-30s %.1f GB", "  VRAM", vram_gb)
    else:
        logger.info("  %-30s %s", "GPU detected", _tick(False))
        logger.info("  %-30s", "  WARNING: CPU only — training will be slow")

    # Dependencies
    logger.info("  %-30s %s", "Dependencies installed", _tick(deps_ok))
    all_ok = all(pkg_results.values())
    logger.info("  %-30s %s", "All packages importable", _tick(all_ok))
    if not all_ok:
        for name, ok in pkg_results.items():
            if not ok:
                logger.info("    MISSING: %s", name)

    # Directories
    logger.info("  %-30s %s", "Drive directories created", _tick(bool(dirs)))
    for name, path in dirs.items():
        logger.info("    %-26s %s", name, path)

    logger.info(border)
    if all_ok and gpu_name is not None:
        logger.info("  All checks passed. Ready to train!")
    elif all_ok:
        logger.info("  Dependencies OK but no GPU. Consider changing runtime.")
    else:
        logger.info("  Some checks failed. Review the output above.")
    logger.info(border)


# ---------------------------------------------------------------------------
# Main setup function
# ---------------------------------------------------------------------------

def setup(
    drive_root: str = "/content/drive/MyDrive/change-detection",
    install_deps: bool = True,
) -> Dict[str, Path]:
    """Run the full Colab setup sequence.

    Steps:
        1. Mount Google Drive (if in Colab)
        2. Check GPU type and VRAM
        3. Install pip dependencies
        4. Verify all packages are importable
        5. Create Drive directories
        6. Print status summary

    Args:
        drive_root: Root directory on Google Drive for persistent storage.
        install_deps: Whether to run ``pip install -r requirements.txt``.

    Returns:
        Dict of project directory paths.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("  Military Base Change Detection — Colab Setup")
    logger.info("=" * 60)

    # 1. Google Drive
    logger.info("[1/5] Google Drive")
    colab = is_colab()
    drive_mounted = mount_drive()

    # 2. GPU
    logger.info("[2/5] GPU Check")
    gpu_name, gpu_type, vram_gb = check_gpu()
    if gpu_name:
        logger.info("  GPU: %s (%s) — %.1f GB VRAM", gpu_name, gpu_type, vram_gb)
    else:
        logger.warning("  No GPU detected.")

    # 3. Install dependencies
    logger.info("[3/5] Dependencies")
    deps_ok = False
    if install_deps:
        deps_ok = install_requirements()
        logger.info("  pip install: %s", _tick(deps_ok))
    else:
        logger.info("  Skipped (install_deps=False).")

    # 4. Verify imports
    logger.info("[4/5] Package Verification")
    pkg_results = verify_packages()
    ok_count = sum(pkg_results.values())
    logger.info("  %d/%d packages importable", ok_count, len(pkg_results))

    # 5. Create directories
    logger.info("[5/5] Drive Directories")
    dirs: Dict[str, Path] = {}
    if drive_mounted or not colab:
        dirs = create_drive_dirs(drive_root)
        logger.info("  Created %d directories under %s", len(dirs), drive_root)
    else:
        logger.warning("  Drive not mounted — skipping directory creation.")

    # Summary
    print_summary(
        colab=colab,
        drive_mounted=drive_mounted,
        gpu_name=gpu_name,
        gpu_type=gpu_type,
        vram_gb=vram_gb,
        deps_ok=deps_ok,
        pkg_results=pkg_results,
        dirs=dirs,
    )

    return dirs


if __name__ == "__main__":
    setup()
