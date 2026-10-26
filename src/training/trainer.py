"""Modular training engine for remote sensing semantic segmentation models.

Supports U-Net, DeepLabV3+, and custom PyTorch segmentation models with:
- Combined loss (CrossEntropy + Dice) with ignore_index=0.
- Metric tracking (per-class IoU, mIoU, Overall Accuracy).
- Learning rate scheduling and gradient clipping.
- Automatic Mixed Precision (AMP) for GPU acceleration.
- Early stopping mechanism monitoring validation mIoU.
- Best checkpoint saving based on validation mIoU.
- Continuous history logging and loss curve visualization.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.losses import CombinedLoss
from src.models.metrics import SegmentationMetricTracker
from src.training.training_config import (
    CHECKPOINT_DIR,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    GRADIENT_CLIP_NORM,
    get_safe_device,
)
from src.utils.class_mapping import CLASS_NAMES, EVAL_CLASS_IDS, IGNORE_INDEX, NUM_CLASSES


class SemanticSegmentationTrainer:
    """Trainer class for semantic segmentation on LoveDA imagery."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str = "model",
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        learning_rate: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        use_amp: bool = True,
    ):
        self.model_name = model_name
        self.device = device if device is not None else get_safe_device()
        self.model = model.to(self.device)
        self.use_amp = use_amp and (self.device.type == "cuda")

        # Mixed Precision Scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Loss function
        if criterion is not None:
            self.criterion = criterion.to(self.device)
        else:
            self.criterion = CombinedLoss(
                ce_weight=1.0,
                dice_weight=1.0,
                ignore_index=IGNORE_INDEX,
                num_classes=NUM_CLASSES,
            ).to(self.device)

        # Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        # Scheduler
        self.scheduler = scheduler

        # Metric Tracker
        self.metric_tracker = SegmentationMetricTracker(
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
            eval_classes=EVAL_CLASS_IDS,
            class_names=CLASS_NAMES,
        )

        # Checkpoints
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # History
        self.history: List[Dict[str, Any]] = []
        self.best_val_miou: float = -1.0
        self.best_epoch: int = -1

    def train_epoch(
        self,
        train_loader: DataLoader,
        max_steps: Optional[int] = None,
    ) -> float:
        """Execute one training epoch."""
        self.model.train()
        total_loss = 0.0
        step_count = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            if max_steps is not None and batch_idx >= max_steps:
                break

            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at batch {batch_idx}. Skipping step.")
                continue

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRADIENT_CLIP_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRADIENT_CLIP_NORM)
                self.optimizer.step()

            total_loss += loss.item()
            step_count += 1

        return total_loss / max(step_count, 1)

    def validate_epoch(
        self,
        val_loader: DataLoader,
        max_steps: Optional[int] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Execute one validation epoch and compute segmentation metrics."""
        self.model.eval()
        self.metric_tracker.reset()
        total_loss = 0.0
        step_count = 0

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                if max_steps is not None and batch_idx >= max_steps:
                    break

                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                total_loss += loss.item()
                step_count += 1

                self.metric_tracker.update(logits, masks)

        avg_loss = total_loss / max(step_count, 1)
        metrics = self.metric_tracker.get_results()
        return avg_loss, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        start_epoch: int = 1,
        initial_patience_counter: int = 0,
        max_train_steps: Optional[int] = None,
        max_val_steps: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run complete model training and validation loop over specified epochs.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader
            Validation data loader.
        epochs : int
            Maximum number of epochs to train.
        start_epoch : int
            Starting epoch index (default: 1, or higher when resuming).
        initial_patience_counter : int
            Initial counter for early stopping (default: 0).
        max_train_steps : Optional[int]
            Optional step limit per epoch (None for full dataset).
        max_val_steps : Optional[int]
            Optional val step limit per epoch (None for full validation split).
        early_stopping_patience : Optional[int]
            Number of consecutive epochs without validation mIoU improvement before stopping.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        pd.DataFrame
            Complete training history.
        """
        if verbose:
            print("=" * 70)
            print(f"STARTING TRAINING: {self.model_name.upper()} ({start_epoch} -> {epochs} EPOCHS)")
            print(f"Device: {self.device} (AMP: {self.use_amp})")
            print(f"Optimizer: {self.optimizer.__class__.__name__} | Initial LR: {self.optimizer.param_groups[0]['lr']}")
            if early_stopping_patience:
                print(f"Early Stopping: Patience = {early_stopping_patience} epochs (monitoring val_mIoU)")
            print("=" * 70)

        patience_counter = initial_patience_counter

        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.perf_counter()

            # Train
            train_loss = self.train_epoch(train_loader, max_steps=max_train_steps)

            # Validate
            val_loss, val_metrics = self.validate_epoch(val_loader, max_steps=max_val_steps)

            # Current LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_time = time.perf_counter() - epoch_start
            val_miou = val_metrics["mIoU"]
            val_oa = val_metrics["Overall_Accuracy"]

            # Save checkpoints
            is_best = val_miou > self.best_val_miou
            if is_best:
                self.best_val_miou = val_miou
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            self._save_checkpoint(epoch, val_miou, is_best=is_best)

            # Record history
            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_mIoU": round(val_miou, 2),
                "val_OA": round(val_oa, 2),
                "lr": current_lr,
                "time_sec": round(epoch_time, 2),
                "is_best": is_best,
            }
            self.history.append(epoch_record)

            # Save ongoing history to disk
            history_df = pd.DataFrame(self.history)
            history_df.to_csv(self.checkpoint_dir / f"{self.model_name}_history.csv", index=False)

            if verbose:
                best_marker = " [*] BEST" if is_best else ""
                print(
                    f"Epoch {epoch:3d}/{epochs:3d} [{epoch_time:.1f}s] "
                    f"| Train Loss: {train_loss:.4f} "
                    f"| Val Loss: {val_loss:.4f} "
                    f"| Val mIoU: {val_miou:.2f}% "
                    f"| Val OA: {val_oa:.2f}%"
                    f"{best_marker}"
                )

            # Early Stopping Check
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                if verbose:
                    print("-" * 70)
                    print(
                        f"[Early Stopping Triggered] No improvement in validation mIoU for "
                        f"{early_stopping_patience} consecutive epochs."
                    )
                    print(f"Restoring best model checkpoint from Epoch {self.best_epoch} (Val mIoU: {self.best_val_miou:.2f}%)...")
                self.load_best_checkpoint()
                break

        # Save latest checkpoint at end
        if self.history:
            self._save_checkpoint(self.history[-1]["epoch"], self.history[-1]["val_mIoU"], is_best=False)

        if verbose:
            print("=" * 70)
            print(f"TRAINING COMPLETED: Best Val mIoU = {self.best_val_miou:.2f}% at Epoch {self.best_epoch}")
            print(f"Checkpoints saved to: {self.checkpoint_dir}")
            print("=" * 70)

        return pd.DataFrame(self.history)

    def _save_checkpoint(self, epoch: int, val_miou: float, is_best: bool = False):
        """Save model weights and metadata."""
        checkpoint = {
            "epoch": epoch,
            "model_name": self.model_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_miou": val_miou,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
            torch.save(checkpoint, best_path)

        latest_path = self.checkpoint_dir / f"{self.model_name}_latest.pth"
        torch.save(checkpoint, latest_path)

    def load_best_checkpoint(self) -> Dict[str, Any]:
        """Load weights from the best saved checkpoint."""
        best_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
        if not best_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {best_path}")

        checkpoint = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Load weights and optimizer state from an arbitrary checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    def plot_training_curves(
        self,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Plot and save Loss and mIoU curves over epochs."""
        if not self.history:
            raise RuntimeError("No training history available to plot. Call fit() first.")

        df = pd.DataFrame(self.history)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Subplot 1: Loss curves
        axes[0].plot(df["epoch"], df["train_loss"], marker="o", linewidth=2, label="Train Loss", color="#1f77b4")
        axes[0].plot(df["epoch"], df["val_loss"], marker="s", linewidth=2, label="Val Loss", color="#ff7f0e")
        axes[0].set_title(f"{self.model_name.upper()} — Loss Curve", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Loss", fontsize=11)
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend(fontsize=11)

        # Subplot 2: Validation mIoU & Overall Accuracy
        axes[1].plot(df["epoch"], df["val_mIoU"], marker="^", linewidth=2, label="Val mIoU (%)", color="#2ca02c")
        axes[1].plot(df["epoch"], df["val_OA"], marker="d", linewidth=2, label="Val OA (%)", color="#9467bd")
        if self.best_epoch > 0:
            axes[1].axvline(
                x=self.best_epoch,
                color="red",
                linestyle=":",
                label=f"Best mIoU ({self.best_val_miou:.1f}%)",
            )
        axes[1].set_title(f"{self.model_name.upper()} — Validation Metrics", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Percentage (%)", fontsize=11)
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend(fontsize=11)

        plt.tight_layout()

        if save_path:
            save_p = Path(save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_p, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig
