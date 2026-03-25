"""Google Colab setup script.

Handles Drive mounting, GPU verification, dependency installation,
and path configuration. Run this at the start of every Colab session.

Usage (in Colab cell):
    !python setup_colab.py
    # Or import and call:
    from setup_colab import setup
    setup()
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def mount_drive() -> None:
    """Mount Google Drive at /content/drive.

    Skips if not running in Colab or already mounted.
    """
    if not is_colab():
        logger.info("Not running in Colab — skipping Drive mount.")
        return

    if Path("/content/drive/MyDrive").exists():
        logger.info("Google Drive already mounted.")
        return

    from google.colab import drive
    drive.mount("/content/drive")
    logger.info("Google Drive mounted successfully.")


def is_colab() -> bool:
    """Check if running inside Google Colab.

    Returns:
        True if running in Colab environment.
    """
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def check_gpu() -> Optional[str]:
    """Check GPU availability and print device info.

    Returns:
        GPU name string, or None if no GPU available.
    """
    import torch

    if not torch.cuda.is_available():
        logger.warning("No GPU detected! Training will be very slow.")
        return None

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    logger.info("GPU: %s (%.1f GB VRAM)", gpu_name, vram_gb)
    return gpu_name


def detect_gpu_type() -> str:
    """Detect GPU type for batch size selection.

    Returns:
        One of 'T4', 'V100', or 'default'.
    """
    import torch

    if not torch.cuda.is_available():
        return "default"

    name = torch.cuda.get_device_name(0).upper()
    if "T4" in name:
        return "T4"
    elif "V100" in name:
        return "V100"
    return "default"


def install_requirements() -> None:
    """Install project dependencies from requirements.txt."""
    req_path = Path("requirements.txt")
    if not req_path.exists():
        logger.warning("requirements.txt not found in %s", Path.cwd())
        return

    logger.info("Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "-r", str(req_path)
    ])
    logger.info("Dependencies installed.")


def create_drive_dirs(drive_root: str = "/content/drive/MyDrive/change-detection") -> Dict[str, Path]:
    """Create project directories on Google Drive.

    Args:
        drive_root: Root directory on Drive for this project.

    Returns:
        Dict mapping directory names to their paths.
    """
    dirs = {
        "root": Path(drive_root),
        "checkpoints": Path(drive_root) / "checkpoints",
        "logs": Path(drive_root) / "logs",
        "outputs": Path(drive_root) / "outputs",
        "data": Path(drive_root) / "processed_data",
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info("  %s: %s", name, path)

    return dirs


def setup(
    drive_root: str = "/content/drive/MyDrive/change-detection",
    install_deps: bool = True,
) -> Dict[str, Path]:
    """Run full Colab setup.

    Args:
        drive_root: Root directory on Google Drive.
        install_deps: Whether to install pip dependencies.

    Returns:
        Dict of project directory paths.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("=" * 60)
    logger.info("Military Base Change Detection — Colab Setup")
    logger.info("=" * 60)

    # 1. Mount Drive
    mount_drive()

    # 2. Check GPU
    gpu_name = check_gpu()
    gpu_type = detect_gpu_type()
    logger.info("GPU type for batch sizing: %s", gpu_type)

    # 3. Install dependencies
    if install_deps:
        install_requirements()

    # 4. Create Drive directories
    logger.info("Creating project directories on Drive...")
    dirs = create_drive_dirs(drive_root)

    logger.info("=" * 60)
    logger.info("Setup complete! Ready to train.")
    logger.info("=" * 60)

    return dirs


if __name__ == "__main__":
    setup()
