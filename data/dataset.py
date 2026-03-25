"""PyTorch Dataset for change detection tasks.

Loads pre-cropped 256x256 image patches (before/after) and binary change masks.
Supports synchronized augmentations via albumentations.ReplayCompose.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(config: Dict[str, Any]) -> A.ReplayCompose:
    """Build training augmentation pipeline with synchronized transforms.

    Args:
        config: Augmentation config dict from config.yaml.

    Returns:
        ReplayCompose that applies identical spatial transforms to A, B, and mask.
    """
    aug_cfg = config.get("augmentation", {})
    transforms = []

    if aug_cfg.get("horizontal_flip", 0) > 0:
        transforms.append(A.HorizontalFlip(p=aug_cfg["horizontal_flip"]))

    if aug_cfg.get("vertical_flip", 0) > 0:
        transforms.append(A.VerticalFlip(p=aug_cfg["vertical_flip"]))

    if aug_cfg.get("random_rotate_90", 0) > 0:
        transforms.append(A.RandomRotate90(p=aug_cfg["random_rotate_90"]))

    transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return A.ReplayCompose(
        transforms,
        additional_targets={"image_b": "image", "mask": "mask"},
    )


def get_val_transforms() -> A.Compose:
    """Build validation/test transform pipeline (normalize only).

    Returns:
        Compose with ImageNet normalization only.
    """
    return A.Compose(
        [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)],
        additional_targets={"image_b": "image"},
    )


class ChangeDetectionDataset(Dataset):
    """Dataset for loading change detection image pairs and masks.

    Expects directory structure:
        root/
        ├── A/        # before images
        ├── B/        # after images
        └── label/    # binary change masks (0=no change, 255=change)

    Args:
        root: Path to the split directory (e.g., processed_data/train).
        split: One of 'train', 'val', 'test'.
        config: Full config dict for augmentation settings.
        transform: Optional override for the transform pipeline.
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        config: Optional[Dict[str, Any]] = None,
        transform: Optional[Any] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split

        self.dir_a = self.root / "A"
        self.dir_b = self.root / "B"
        self.dir_label = self.root / "label"

        # Collect sorted file lists
        self.filenames = sorted([f.name for f in self.dir_a.iterdir() if f.suffix in (".png", ".jpg", ".tif")])
        logger.info("Loaded %d samples for split '%s' from %s", len(self.filenames), split, root)

        # Set up transforms
        if transform is not None:
            self.transform = transform
        elif split == "train" and config is not None:
            self.transform = get_train_transforms(config)
        else:
            self.transform = get_val_transforms()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dict with keys 'A', 'B', 'mask', 'filename'.
              - A: before image tensor [3, H, W]
              - B: after image tensor [3, H, W]
              - mask: binary change mask tensor [1, H, W] (float, 0 or 1)
              - filename: original filename string
        """
        fname = self.filenames[idx]

        # Lazy load — read from disk each time (no RAM caching)
        img_a = cv2.imread(str(self.dir_a / fname), cv2.IMREAD_COLOR)
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)

        img_b = cv2.imread(str(self.dir_b / fname), cv2.IMREAD_COLOR)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.dir_label / fname), cv2.IMREAD_GRAYSCALE)
        # Normalize 0/255 -> 0/1
        mask = (mask / 255.0).astype(np.float32)

        # Apply synchronized augmentations
        if isinstance(self.transform, A.ReplayCompose):
            transformed = self.transform(image=img_a, image_b=img_b, mask=mask)
            img_a = transformed["image"]
            img_b = transformed["image_b"]
            mask = transformed["mask"]
        else:
            transformed = self.transform(image=img_a, image_b=img_b)
            img_a = transformed["image"]
            img_b = transformed["image_b"]
            # Normalize only applied to images, mask stays as-is

        # HWC -> CHW for images, add channel dim for mask
        img_a = torch.from_numpy(img_a).permute(2, 0, 1).float()
        img_b = torch.from_numpy(img_b).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return {"A": img_a, "B": img_b, "mask": mask, "filename": fname}
