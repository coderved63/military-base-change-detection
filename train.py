"""Main training script for change detection models.

Supports mixed-precision training, gradient accumulation, gradient clipping,
early stopping on validation F1, checkpoint saving (best + last) to Google
Drive or local disk, and full resume from checkpoint after Colab disconnects.

Usage:
    python train.py --config configs/config.yaml --model unet_pp
    python train.py --config configs/config.yaml --model changeformer \
        --resume /content/drive/MyDrive/change-detection/checkpoints/changeformer_last.pth
"""

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from data.dataset import ChangeDetectionDataset
from models import get_model
from utils.losses import get_loss
from utils.metrics import MetricTracker
from utils.visualization import log_predictions_to_tensorboard

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Configures Python, NumPy, PyTorch (CPU + CUDA), and cuDNN for
    deterministic behaviour.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# GPU / config helpers
# ---------------------------------------------------------------------------

def detect_gpu_type() -> str:
    """Detect the current GPU type for automatic batch-size selection.

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


def get_batch_size(config: Dict[str, Any], model_name: str) -> int:
    """Look up the batch size for the current GPU + model combination.

    Args:
        config: Full project config dict.
        model_name: Model identifier string.

    Returns:
        Batch size as an integer.
    """
    gpu_type = detect_gpu_type()
    model_sizes = config.get("batch_sizes", {}).get(model_name, {})
    return model_sizes.get(gpu_type, model_sizes.get("default", 4))


def get_learning_rate(config: Dict[str, Any], model_name: str) -> float:
    """Look up the per-model learning rate, falling back to the global default.

    Args:
        config: Full project config dict.
        model_name: Model identifier string.

    Returns:
        Learning rate as a float.
    """
    return config.get("learning_rates", {}).get(
        model_name, config["training"]["learning_rate"]
    )


def get_num_epochs(config: Dict[str, Any], model_name: str) -> int:
    """Look up the per-model epoch count, falling back to the global default.

    Args:
        config: Full project config dict.
        model_name: Model identifier string.

    Returns:
        Number of epochs as an integer.
    """
    return config.get("epoch_counts", {}).get(
        model_name, config["training"]["epochs"]
    )


def resolve_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Build a path dict based on whether Colab mode is enabled.

    When ``config["colab"]["enabled"]`` is ``True`` all persistent artefacts
    point to Google Drive; otherwise they use the local ``paths`` section.

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
# Data
# ---------------------------------------------------------------------------

def build_dataloaders(
    config: Dict[str, Any],
    data_dir: Path,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation ``DataLoader`` instances.

    Args:
        config: Full project config dict.
        data_dir: Root of the processed dataset (contains ``train/``, ``val/``).
        batch_size: Mini-batch size.

    Returns:
        Tuple of ``(train_loader, val_loader)``.
    """
    ds_cfg = config.get("dataset", {})
    num_workers = ds_cfg.get("num_workers", 4)
    pin_memory = ds_cfg.get("pin_memory", True)

    train_ds = ChangeDetectionDataset(
        root=data_dir / "train", split="train", config=config,
    )
    val_ds = ChangeDetectionDataset(
        root=data_dir / "val", split="val", config=config,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Scheduler with linear warmup
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create a CosineAnnealingLR scheduler preceded by linear warmup.

    During the first ``warmup_epochs`` the LR ramps linearly from
    ``start_factor`` to the base LR, then cosine-decays for the remainder.

    Args:
        optimizer: Optimizer whose LR groups will be scheduled.
        total_epochs: Total number of training epochs.
        warmup_epochs: Number of warmup epochs (0 to disable).

    Returns:
        A learning-rate scheduler instance.
    """
    if warmup_epochs > 0 and warmup_epochs < total_epochs:
        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    return CosineAnnealingLR(optimizer, T_max=total_epochs)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    best_f1: float,
    best_epoch: int,
    save_path: Path,
) -> None:
    """Persist a full training checkpoint to disk.

    Args:
        model: Model whose weights to save.
        optimizer: Optimizer state to save.
        scheduler: LR scheduler state to save.
        scaler: ``GradScaler`` state to save.
        epoch: Epoch number just completed (1-indexed).
        best_f1: Best validation F1 achieved so far.
        best_epoch: Epoch that achieved ``best_f1``.
        save_path: Destination file path.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_f1": best_f1,
            "best_epoch": best_epoch,
        },
        save_path,
    )
    logger.info("Checkpoint saved → %s", save_path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[int, float, int]:
    """Restore training state from a checkpoint.

    Args:
        path: Checkpoint file to load.
        model: Model to receive saved weights.
        optimizer: Optimizer to receive saved state.
        scheduler: Scheduler to receive saved state.
        scaler: ``GradScaler`` to receive saved state.
        device: Target device for ``map_location``.

    Returns:
        Tuple of ``(start_epoch, best_f1, best_epoch)``.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    best_f1 = ckpt["best_f1"]
    best_epoch = ckpt.get("best_epoch", ckpt["epoch"])
    logger.info(
        "Resumed from epoch %d (best F1: %.4f @ epoch %d)",
        ckpt["epoch"], best_f1, best_epoch,
    )
    return ckpt["epoch"], best_f1, best_epoch


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    tracker: MetricTracker,
    accum_steps: int,
    grad_clip: float,
) -> Tuple[float, Dict[str, float]]:
    """Execute one full training epoch.

    Args:
        model: Change-detection model.
        loader: Training ``DataLoader``.
        criterion: Loss module (operates on raw logits).
        optimizer: Optimiser instance.
        scaler: ``GradScaler`` for mixed-precision training.
        device: Target CUDA / CPU device.
        tracker: ``MetricTracker`` (reset externally before this call).
        accum_steps: Number of gradient-accumulation micro-steps.
        grad_clip: Maximum gradient norm for clipping.

    Returns:
        Tuple of ``(average_loss, metrics_dict)``.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        img_a = batch["A"].to(device, non_blocking=True)
        img_b = batch["B"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with autocast():
            logits = model(img_a, img_b)
            loss = criterion(logits, mask) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Track loss (undo the accumulation scaling for logging)
        running_loss += loss.item() * accum_steps
        num_batches += 1

        # Track metrics (MetricTracker handles sigmoid + threshold internally)
        tracker.update(logits.detach(), mask)

        pbar.set_postfix(loss=f"{running_loss / num_batches:.4f}")

    avg_loss = running_loss / max(num_batches, 1)
    metrics = tracker.compute()
    return avg_loss, metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tracker: MetricTracker,
) -> Tuple[float, Dict[str, float], Optional[Dict[str, torch.Tensor]]]:
    """Run one full validation pass.

    Args:
        model: Change-detection model (set to eval internally).
        loader: Validation ``DataLoader``.
        criterion: Loss module (operates on raw logits).
        device: Target device.
        tracker: ``MetricTracker`` (reset externally before this call).

    Returns:
        Tuple of ``(average_loss, metrics_dict, last_batch)`` where
        ``last_batch`` is the final mini-batch dict (for visualisation).
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0
    last_batch: Optional[Dict[str, torch.Tensor]] = None

    pbar = tqdm(loader, desc="  Val  ", leave=False, dynamic_ncols=True)
    for batch in pbar:
        img_a = batch["A"].to(device, non_blocking=True)
        img_b = batch["B"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        logits = model(img_a, img_b)
        loss = criterion(logits, mask)

        running_loss += loss.item()
        num_batches += 1
        tracker.update(logits, mask)

        # Keep the last batch for TensorBoard visualisation
        last_batch = {
            "A": img_a,
            "B": img_b,
            "mask": mask,
            "logits": logits,
        }

        pbar.set_postfix(loss=f"{running_loss / num_batches:.4f}")

    avg_loss = running_loss / max(num_batches, 1)
    metrics = tracker.compute()
    return avg_loss, metrics, last_batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point — parse CLI args, build components, run training loop."""
    # ---- CLI ----------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Train a change-detection model",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override the model name from config (siamese_cnn | unet_pp | changeformer).",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to a checkpoint file to resume training from.",
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
    train_cfg = config["training"]
    seed: int = config.get("project", {}).get("seed", 42)

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_type = detect_gpu_type()
    logger.info("Device: %s | GPU type: %s", device, gpu_type)

    # ---- Paths --------------------------------------------------------
    paths = resolve_paths(config)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # ---- Hyperparams (auto from per-model tables) ---------------------
    batch_size = get_batch_size(config, model_name)
    lr = get_learning_rate(config, model_name)
    num_epochs = get_num_epochs(config, model_name)
    accum_steps: int = train_cfg.get("gradient_accumulation_steps", 1)
    grad_clip: float = train_cfg.get("grad_clip_max_norm", 1.0)
    warmup_epochs: int = train_cfg.get("warmup_epochs", 5)
    vis_interval: int = train_cfg.get("vis_interval", 5)
    threshold: float = config.get("evaluation", {}).get("threshold", 0.5)

    logger.info(
        "Hyperparams → model=%s  bs=%d  lr=%.1e  epochs=%d  accum=%d  warmup=%d",
        model_name, batch_size, lr, num_epochs, accum_steps, warmup_epochs,
    )

    # ---- Model --------------------------------------------------------
    model = get_model(model_name, config).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model: %s (%.2fM parameters)", model_name, param_count)

    # ---- Data ---------------------------------------------------------
    train_loader, val_loader = build_dataloaders(config, paths["data"], batch_size)
    logger.info(
        "Data: %d train batches, %d val batches (batch_size=%d)",
        len(train_loader), len(val_loader), batch_size,
    )

    # ---- Loss / optimiser / scheduler ---------------------------------
    criterion = get_loss(config).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg["weight_decay"],
    )

    scheduler = build_scheduler(optimizer, num_epochs, warmup_epochs)
    scaler = GradScaler(enabled=train_cfg.get("amp", True))

    # ---- TensorBoard --------------------------------------------------
    writer = SummaryWriter(log_dir=str(paths["logs"] / model_name))

    # ---- MetricTrackers -----------------------------------------------
    train_tracker = MetricTracker(threshold=threshold)
    val_tracker = MetricTracker(threshold=threshold)

    # ---- Resume -------------------------------------------------------
    start_epoch: int = 0
    best_f1: float = 0.0
    best_epoch: int = 0

    if args.resume is not None and args.resume.exists():
        start_epoch, best_f1, best_epoch = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device,
        )
    elif args.resume is not None:
        logger.warning("Resume path does not exist: %s — training from scratch", args.resume)

    # ---- Early stopping state -----------------------------------------
    es_cfg = train_cfg.get("early_stopping", {})
    es_enabled: bool = es_cfg.get("enabled", True)
    patience: int = es_cfg.get("patience", 15)
    patience_counter: int = 0

    # ---- Training loop ------------------------------------------------
    wall_start = time.monotonic()

    logger.info("=" * 60)
    logger.info("Starting training from epoch %d", start_epoch + 1)
    logger.info("=" * 60)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.monotonic()
        epoch_num = epoch + 1  # 1-indexed for display / checkpoints

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %d/%d  (lr=%.2e)", epoch_num, num_epochs, current_lr)

        # -- Train ------------------------------------------------------
        train_tracker.reset()
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            train_tracker, accum_steps, grad_clip,
        )

        # -- Validate ---------------------------------------------------
        val_tracker.reset()
        val_loss, val_metrics, last_val_batch = validate(
            model, val_loader, criterion, device, val_tracker,
        )

        # -- Step scheduler (after both train + val) --------------------
        scheduler.step()

        # -- TensorBoard scalars ----------------------------------------
        writer.add_scalar("Loss/train", train_loss, epoch_num)
        writer.add_scalar("Loss/val", val_loss, epoch_num)
        writer.add_scalar("LR", current_lr, epoch_num)

        for key, value in train_metrics.items():
            writer.add_scalar(f"Train/{key}", value, epoch_num)
        for key, value in val_metrics.items():
            writer.add_scalar(f"Val/{key}", value, epoch_num)

        # -- TensorBoard prediction images ------------------------------
        if last_val_batch is not None and epoch_num % vis_interval == 0:
            log_predictions_to_tensorboard(
                writer,
                img_a=last_val_batch["A"],
                img_b=last_val_batch["B"],
                mask_true=last_val_batch["mask"],
                mask_pred=last_val_batch["logits"],
                step=epoch_num,
                num_samples=4,
            )

        # -- Console log ------------------------------------------------
        epoch_time = time.monotonic() - epoch_start
        logger.info(
            "  Train — loss: %.4f | F1: %.4f | IoU: %.4f",
            train_loss, train_metrics["f1"], train_metrics["iou"],
        )
        logger.info(
            "  Val   — loss: %.4f | F1: %.4f | IoU: %.4f | Prec: %.4f | Rec: %.4f | OA: %.4f",
            val_loss,
            val_metrics["f1"],
            val_metrics["iou"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["oa"],
        )
        logger.info("  Epoch time: %.1fs", epoch_time)

        # -- Save last checkpoint (every epoch) -------------------------
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch=epoch_num,
            best_f1=best_f1,
            best_epoch=best_epoch,
            save_path=paths["checkpoints"] / f"{model_name}_last.pth",
        )

        # -- Save best checkpoint (if improved) -------------------------
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch_num
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch=epoch_num,
                best_f1=best_f1,
                best_epoch=best_epoch,
                save_path=paths["checkpoints"] / f"{model_name}_best.pth",
            )
            logger.info("  ★ New best F1: %.4f (epoch %d)", best_f1, best_epoch)
        else:
            patience_counter += 1
            logger.info(
                "  No improvement (%d/%d patience)", patience_counter, patience,
            )

        # -- Early stopping ---------------------------------------------
        if es_enabled and patience_counter >= patience:
            logger.info(
                "Early stopping triggered after %d epochs without improvement.",
                patience,
            )
            break

    # ---- Summary ------------------------------------------------------
    writer.close()
    total_time = time.monotonic() - wall_start
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("  Best val F1 : %.4f  (epoch %d)", best_f1, best_epoch)
    logger.info("  Total time  : %dh %dm %ds", int(hours), int(minutes), int(seconds))
    logger.info("  Checkpoints : %s", paths["checkpoints"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
