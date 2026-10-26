"""Full-scale training script for LoveDA semantic segmentation models.

Trains U-Net and DeepLabV3+ sequentially on the complete LoveDA dataset
with early stopping, CosineAnnealing scheduling, AMP acceleration,
and comprehensive checkpointing & visualization.
"""

import argparse
from pathlib import Path
import sys
import time

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.dataset.dataloader import get_dataloaders
from src.models.deeplabv3plus import DeepLabV3Plus
from src.models.losses import CombinedLoss
from src.models.unet import UNet
from src.preprocessing.preprocessing_config import TARGET_IMAGE_SIZE
from src.training.trainer import SemanticSegmentationTrainer
from src.training.training_config import (
    CHECKPOINT_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    OUTPUTS_DIR,
    REPORTS_DIR,
    get_safe_device,
)
from src.utils.class_mapping import IGNORE_INDEX, NUM_CLASSES, mask_to_rgb
from src.utils.visualization import denormalize_image_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Full Model Training for LoveDA Semantic Segmentation")
    parser.add_argument("--models", type=str, default="unet,deeplabv3p", help="Comma-separated model names to train")
    parser.add_argument("--epochs", type=int, default=150, help="Maximum epochs per model (default: 150)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience in epochs (0 to disable)")
    parser.add_argument("--no_early_stopping", action="store_true", help="Disable early stopping and train for all epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (default: 0 for Windows)")
    parser.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Optional step limit per epoch")
    parser.add_argument("--max_val_steps", type=int, default=None, help="Optional val step limit per epoch")
    return parser.parse_args()


def plot_cross_model_comparison(unet_history: pd.DataFrame, deeplab_history: pd.DataFrame, save_path: Path):
    """Plot comparative loss and mIoU curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Loss comparison
    axes[0].plot(unet_history["epoch"], unet_history["train_loss"], "b-o", label="U-Net Train Loss", linewidth=2, markersize=3)
    axes[0].plot(unet_history["epoch"], unet_history["val_loss"], "b--s", label="U-Net Val Loss", linewidth=2, markersize=3)
    axes[0].plot(deeplab_history["epoch"], deeplab_history["train_loss"], "r-o", label="DeepLabV3+ Train Loss", linewidth=2, markersize=3)
    axes[0].plot(deeplab_history["epoch"], deeplab_history["val_loss"], "r--s", label="DeepLabV3+ Val Loss", linewidth=2, markersize=3)
    axes[0].set_title("Loss Convergence Comparison", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Combined Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Validation mIoU and OA comparison
    axes[1].plot(unet_history["epoch"], unet_history["val_mIoU"], "b-^", label="U-Net Val mIoU (%)", linewidth=2.5, markersize=4)
    axes[1].plot(deeplab_history["epoch"], deeplab_history["val_mIoU"], "r-^", label="DeepLabV3+ Val mIoU (%)", linewidth=2.5, markersize=4)
    axes[1].plot(unet_history["epoch"], unet_history["val_OA"], "b:", label="U-Net Val OA (%)", linewidth=1.5)
    axes[1].plot(deeplab_history["epoch"], deeplab_history["val_OA"], "r:", label="DeepLabV3+ Val OA (%)", linewidth=1.5)
    axes[1].set_title("Validation Metric Trajectories", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Score (%)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_validation_predictions(unet_model, deeplab_model, val_loader, device, save_path: Path):
    """Generate visual qualitative comparison on validation samples."""
    val_images, val_masks = next(iter(val_loader))
    val_images = val_images.to(device)

    unet_model.eval()
    deeplab_model.eval()
    with torch.no_grad():
        u_logits = unet_model(val_images[:2])
        d_logits = deeplab_model(val_images[:2])
        u_preds = torch.argmax(u_logits, dim=1).cpu().numpy()
        d_preds = torch.argmax(d_logits, dim=1).cpu().numpy()

    gt_masks = val_masks[:2].cpu().numpy()
    rgb_imgs = [denormalize_image_tensor(val_images[i]) for i in range(2)]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i in range(2):
        axes[i, 0].imshow(rgb_imgs[i])
        axes[i, 0].set_title(f"Sample {i+1}: RGB Image", fontsize=12, fontweight="bold")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_to_rgb(gt_masks[i]))
        axes[i, 1].set_title(f"Sample {i+1}: Ground Truth", fontsize=12, fontweight="bold")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(mask_to_rgb(u_preds[i]))
        axes[i, 2].set_title(f"Sample {i+1}: U-Net Prediction", fontsize=12, fontweight="bold")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(mask_to_rgb(d_preds[i]))
        axes[i, 3].set_title(f"Sample {i+1}: DeepLabV3+ Prediction", fontsize=12, fontweight="bold")
        axes[i, 3].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = get_safe_device()
    use_amp = (not args.no_amp) and (device.type == "cuda")

    print("=" * 80)
    print("REMOTE SENSING SCENE UNDERSTANDING: FULL MODEL TRAINING")
    print("=" * 80)
    print(f"Device:               {device}")
    if torch.cuda.is_available():
        print(f"GPU Name:             {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM:           {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Automatic Mixed Prec: {use_amp}")
    print(f"Models to Train:      {args.models}")
    print(f"Max Epochs:           {args.epochs}")
    print(f"Batch Size:           {args.batch_size}")
    patience = None if (args.no_early_stopping or args.patience <= 0) else args.patience
    print(f"Early Stopping:       {'Disabled (training full epochs)' if patience is None else f'Patience = {patience} epochs'}")
    print(f"Target Resolution:    {TARGET_IMAGE_SIZE}")
    print("=" * 80)

    # Initialize DataLoaders
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=TARGET_IMAGE_SIZE,
    )
    print(f"Dataset Loaded: Train = {len(train_loader.dataset)} pairs, Val = {len(val_loader.dataset)} pairs")
    print(f"Batches per epoch: Train = {len(train_loader)}, Val = {len(val_loader)}")

    models_to_train = [m.strip().lower() for m in args.models.split(",")]
    trained_models = {}
    histories = {}

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Train U-Net
    if "unet" in models_to_train:
        print("\n" + "#" * 80)
        print("STAGE 1: TRAINING U-NET")
        print("#" * 80)

        unet_model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_features=32)
        unet_opt = torch.optim.AdamW(unet_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        unet_sched = torch.optim.lr_scheduler.CosineAnnealingLR(unet_opt, T_max=args.epochs, eta_min=1e-6)

        unet_trainer = SemanticSegmentationTrainer(
            model=unet_model,
            model_name="unet",
            optimizer=unet_opt,
            scheduler=unet_sched,
            device=device,
            checkpoint_dir=CHECKPOINT_DIR,
            use_amp=use_amp,
        )

        unet_history = unet_trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            max_train_steps=args.max_train_steps,
            max_val_steps=args.max_val_steps,
            early_stopping_patience=patience,
            verbose=True,
        )

        unet_curves = OUTPUTS_DIR / "unet_training_curves.png"
        unet_trainer.plot_training_curves(save_path=unet_curves)
        print(f"Saved U-Net training curves: {unet_curves}")

        trained_models["unet"] = unet_trainer.model
        histories["unet"] = unet_history

    # 2. Train DeepLabV3+
    if "deeplabv3p" in models_to_train or "deeplab" in models_to_train:
        print("\n" + "#" * 80)
        print("STAGE 2: TRAINING DEEPLABV3+")
        print("#" * 80)

        deeplab_model = DeepLabV3Plus(num_classes=NUM_CLASSES, backbone="resnet34", pretrained=False)
        deeplab_opt = torch.optim.AdamW(deeplab_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        deeplab_sched = torch.optim.lr_scheduler.CosineAnnealingLR(deeplab_opt, T_max=args.epochs, eta_min=1e-6)

        deeplab_trainer = SemanticSegmentationTrainer(
            model=deeplab_model,
            model_name="deeplabv3p",
            optimizer=deeplab_opt,
            scheduler=deeplab_sched,
            device=device,
            checkpoint_dir=CHECKPOINT_DIR,
            use_amp=use_amp,
        )

        deeplab_history = deeplab_trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            max_train_steps=args.max_train_steps,
            max_val_steps=args.max_val_steps,
            early_stopping_patience=patience,
            verbose=True,
        )

        deeplab_curves = OUTPUTS_DIR / "deeplabv3p_training_curves.png"
        deeplab_trainer.plot_training_curves(save_path=deeplab_curves)
        print(f"Saved DeepLabV3+ training curves: {deeplab_curves}")

        trained_models["deeplabv3p"] = deeplab_trainer.model
        histories["deeplabv3p"] = deeplab_history

    # 3. Cross-Model Visual Comparison & Consolidated History
    if "unet" in trained_models and "deeplabv3p" in trained_models:
        print("\n" + "=" * 80)
        print("GENERATING CROSS-MODEL COMPARISONS & VALIDATION INFERENCE")
        print("=" * 80)

        comp_curves_path = OUTPUTS_DIR / "models_training_comparison.png"
        plot_cross_model_comparison(histories["unet"], histories["deeplabv3p"], comp_curves_path)
        print(f"Saved comparative loss/mIoU curves: {comp_curves_path}")

        val_pred_path = OUTPUTS_DIR / "validation_inference_comparison.png"
        generate_validation_predictions(trained_models["unet"], trained_models["deeplabv3p"], val_loader, device, val_pred_path)
        print(f"Saved qualitative validation prediction comparisons: {val_pred_path}")

        # Consolidated history table
        u_hist = histories["unet"].copy()
        u_hist["model"] = "U-Net"
        d_hist = histories["deeplabv3p"].copy()
        d_hist["model"] = "DeepLabV3+"
        full_history = pd.concat([u_hist, d_hist], ignore_index=True)
        history_csv = REPORTS_DIR / "training_history.csv"
        full_history.to_csv(history_csv, index=False)
        print(f"Consolidated history written to: {history_csv}")

    # Summary Report
    report_txt = REPORTS_DIR / "training_report.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("PHASE 6 — FULL MODEL TRAINING REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Max Epochs: {args.epochs} | Early Stopping Patience: {args.patience}\n")
        f.write(f"Batch Size: {args.batch_size} | Learning Rate: {args.lr}\n\n")

        for m_name, hist in histories.items():
            best_row = hist[hist["is_best"]].iloc[-1] if not hist[hist["is_best"]].empty else hist.iloc[-1]
            f.write(f"Model: {m_name.upper()}\n")
            f.write(f"  - Total Epochs Completed: {len(hist)}\n")
            f.write(f"  - Best Epoch:             {best_row['epoch']}\n")
            f.write(f"  - Peak Validation mIoU:   {best_row['val_mIoU']}%\n")
            f.write(f"  - Peak Validation OA:     {best_row['val_OA']}%\n")
            f.write(f"  - Final Train Loss:       {hist['train_loss'].iloc[-1]}\n")
            f.write(f"  - Final Val Loss:         {hist['val_loss'].iloc[-1]}\n")
            f.write(f"  - Best Checkpoint:        {CHECKPOINT_DIR / f'{m_name}_best.pth'}\n\n")

        f.write("=" * 70 + "\n")
        f.write("TRAINING PIPELINE COMPLETE\n")
        f.write("Next Phase: Phase 7 — Comprehensive Model Evaluation & Quantitative Comparison\n")
        f.write("=" * 70 + "\n")

    print(f"\nFinal training report written to: {report_txt}")
    print("ALL TRAINING STAGES COMPLETE.")


if __name__ == "__main__":
    main()
