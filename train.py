"""Main training script for change detection models.

Supports AMP, gradient clipping, early stopping, checkpoint saving to Google
Drive, and resume from checkpoint after Colab disconnects.

Usage:
    python train.py --config configs/config.yaml --model unet_pp
    python train.py --config configs/config.yaml --model changeformer --resume checkpoints/changeformer_last.pth
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from data.dataset import ChangeDetectionDataset
from models import get_model
from utils.losses import get_loss
from utils.metrics import ConfusionMatrix
from utils.visualization import plot_prediction

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_gpu_type() -> str:
    """Detect the current GPU type for batch size selection.

    Returns:
        GPU type string ('T4', 'V100', or 'default').
    """
    if not torch.cuda.is_available():
        return "default"
    name = torch.cuda.get_device_name(0).upper()
    if "T4" in name:
        return "T4"
    elif "V100" in name:
        return "V100"
    return "default"


def get_batch_size(config: Dict[str, Any], model_name: str) -> int:
    """Get appropriate batch size based on GPU and model.

    Args:
        config: Full config dict.
        model_name: Model name string.

    Returns:
        Batch size integer.
    """
    gpu_type = detect_gpu_type()
    batch_sizes = config.get("batch_sizes", {}).get(model_name, {})
    return batch_sizes.get(gpu_type, batch_sizes.get("default", 4))


def get_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Resolve paths based on whether running on Colab or locally.

    Args:
        config: Full config dict.

    Returns:
        Dict with keys: 'data', 'checkpoints', 'logs', 'outputs'.
    """
    if config.get("colab", {}).get("enabled", False):
        colab = config["colab"]
        return {
            "data": Path(colab["data_dir"]),
            "checkpoints": Path(colab["checkpoint_dir"]),
            "logs": Path(colab["log_dir"]),
            "outputs": Path(colab["output_dir"]),
        }
    else:
        paths = config.get("paths", {})
        return {
            "data": Path(paths.get("processed_data", "./processed_data")),
            "checkpoints": Path(paths.get("checkpoint_dir", "./checkpoints")),
            "logs": Path(paths.get("log_dir", "./logs")),
            "outputs": Path(paths.get("output_dir", "./outputs")),
        }


def build_dataloaders(
    config: Dict[str, Any],
    data_dir: Path,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    Args:
        config: Full config dict.
        data_dir: Path to processed dataset root.
        batch_size: Batch size.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    ds_cfg = config.get("dataset", {})
    num_workers = ds_cfg.get("num_workers", 4)
    pin_memory = ds_cfg.get("pin_memory", True)

    train_ds = ChangeDetectionDataset(data_dir / "train", split="train", config=config)
    val_ds = ChangeDetectionDataset(data_dir / "val", split="val", config=config)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    config: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Run one training epoch.

    Args:
        model: The change detection model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: GradScaler for AMP.
        device: Target device.
        config: Full config dict.

    Returns:
        Tuple of (average loss, metrics dict).
    """
    model.train()
    running_loss = 0.0
    cm = ConfusionMatrix()
    train_cfg = config.get("training", {})
    accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    grad_clip = train_cfg.get("grad_clip_max_norm", 1.0)
    threshold = config.get("evaluation", {}).get("threshold", 0.5)

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        img_a = batch["A"].to(device)
        img_b = batch["B"].to(device)
        mask = batch["mask"].to(device)

        with autocast():
            logits = model(img_a, img_b)
            loss = criterion(logits, mask) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps

        # Metrics
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > threshold).float()
            cm.update(preds, mask)

    avg_loss = running_loss / len(loader)
    metrics = cm.compute()
    return avg_loss, metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """Run validation.

    Args:
        model: The change detection model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Target device.
        threshold: Binarization threshold.

    Returns:
        Tuple of (average loss, metrics dict).
    """
    model.eval()
    running_loss = 0.0
    cm = ConfusionMatrix()

    for batch in tqdm(loader, desc="Val", leave=False):
        img_a = batch["A"].to(device)
        img_b = batch["B"].to(device)
        mask = batch["mask"].to(device)

        logits = model(img_a, img_b)
        loss = criterion(logits, mask)
        running_loss += loss.item()

        preds = (torch.sigmoid(logits) > threshold).float()
        cm.update(preds, mask)

    avg_loss = running_loss / len(loader)
    metrics = cm.compute()
    return avg_loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    epoch: int,
    best_f1: float,
    save_path: Path,
) -> None:
    """Save a training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        scaler: GradScaler state.
        epoch: Current epoch number.
        best_f1: Best validation F1 so far.
        save_path: Path to save the checkpoint.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_f1": best_f1,
    }, save_path)
    logger.info("Saved checkpoint: %s", save_path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[int, float]:
    """Load a training checkpoint for resume.

    Args:
        path: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        scheduler: Scheduler to load state into.
        scaler: GradScaler to load state into.
        device: Target device.

    Returns:
        Tuple of (start_epoch, best_f1).
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    logger.info("Resumed from epoch %d (best F1: %.4f)", ckpt["epoch"], ckpt["best_f1"])
    return ckpt["epoch"], ckpt["best_f1"]


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train change detection model")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--model", type=str, default=None, help="Override model name from config")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint for resume")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = args.model or config["model"]["name"]
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Resolve paths
    paths = get_paths(config)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # Model
    model = get_model(model_name, config).to(device)
    logger.info("Model: %s (%.1fM params)", model_name,
                sum(p.numel() for p in model.parameters()) / 1e6)

    # Data
    batch_size = get_batch_size(config, model_name)
    train_loader, val_loader = build_dataloaders(config, paths["data"], batch_size)

    # Loss, optimizer, scheduler
    criterion = get_loss(config)
    lr = config.get("learning_rates", {}).get(model_name, config["training"]["learning_rate"])
    epochs = config.get("epoch_counts", {}).get(model_name, config["training"]["epochs"])

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    # TensorBoard
    writer = SummaryWriter(log_dir=str(paths["logs"] / model_name))

    # Resume
    start_epoch = 0
    best_f1 = 0.0
    if args.resume and args.resume.exists():
        start_epoch, best_f1 = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )

    # Early stopping state
    es_cfg = config["training"]["early_stopping"]
    patience = es_cfg.get("patience", 15)
    patience_counter = 0
    threshold = config.get("evaluation", {}).get("threshold", 0.5)

    # Training loop
    for epoch in range(start_epoch, epochs):
        logger.info("Epoch %d/%d", epoch + 1, epochs)

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device, threshold)
        scheduler.step()

        # Log
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"Val/{k}", v, epoch)

        logger.info(
            "  Train Loss: %.4f | Val Loss: %.4f | Val F1: %.4f | Val IoU: %.4f",
            train_loss, val_loss, val_metrics["f1"], val_metrics["iou"],
        )

        # Save last checkpoint (always)
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch + 1, best_f1,
            paths["checkpoints"] / f"{model_name}_last.pth",
        )

        # Save best checkpoint
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch + 1, best_f1,
                paths["checkpoints"] / f"{model_name}_best.pth",
            )
            logger.info("  New best F1: %.4f", best_f1)
        else:
            patience_counter += 1

        # Early stopping
        if es_cfg.get("enabled", True) and patience_counter >= patience:
            logger.info("Early stopping triggered at epoch %d", epoch + 1)
            break

    writer.close()
    logger.info("Training complete. Best F1: %.4f", best_f1)


if __name__ == "__main__":
    main()
