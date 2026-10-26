"""Balanced Training Script — Class-Imbalance Fix for LoveDA Segmentation.

KEY DIFFERENCES from the original train_full_models.py:
  1. CLASS-WEIGHTED LOSS: Median Frequency weights are applied to CrossEntropyLoss
     so rare classes (Barren=1.96x, Road=1.89x) get proportionally more gradient
     attention than dominant classes (Background=0.27x, Agriculture=0.40x).
  2. TRAIN/VAL SPLIT ONLY: Test split is completely excluded. No test_loader is
     created, referenced, or used at any point in this script.
  3. NEW CHECKPOINT DIRECTORY: Saves to changes/checkpoints/ to avoid overwriting
     the original Phase 6 checkpoints in checkpoints/.
  4. NEW REPORTS DIRECTORY: Saves to changes/reports/ for clean separation.

Data Flow:
    LoveDA/Train  (2,522 pairs) --> Train the model (backpropagation)
    LoveDA/Val    (1,669 pairs) --> Monitor validation mIoU per epoch, save best checkpoint
    LoveDA/Test   (1,796 imgs)  --> EXCLUDED completely (never loaded, never touched)
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
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    get_safe_device,
)
from src.utils.class_mapping import IGNORE_INDEX, NUM_CLASSES, mask_to_rgb
from src.utils.visualization import denormalize_image_tensor

# Import the class weights from the changes folder
CHANGES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHANGES_DIR))
from class_weights import get_lova_da_class_weights, print_class_weights_table

# Output directories — isolated from original outputs to avoid overwriting
CHANGES_CHECKPOINT_DIR = CHANGES_DIR / "checkpoints"
CHANGES_REPORTS_DIR = CHANGES_DIR / "reports"
CHANGES_OUTPUTS_DIR = CHANGES_DIR / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Balanced Training — Class Imbalance Fix for LoveDA Segmentation"
    )
    parser.add_argument("--models", type=str, default="unet,deeplabv3p",
                        help="Comma-separated model names: 'unet', 'deeplabv3p', or 'unet,deeplabv3p'")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Maximum training epochs per model (default: 150)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for DataLoaders (default: 8)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY,
                        help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=40,
                        help="Early stopping patience in epochs (0 to disable, default: 40)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (default: 0 for Windows safety)")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable Automatic Mixed Precision training")
    parser.add_argument("--focal", action="store_true",
                        help="Use Focal Loss instead of Weighted CrossEntropy + Dice")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Automatically resume from latest checkpoint if available (default: True)")
    return parser.parse_args()


def build_balanced_criterion(device: torch.device, use_focal: bool = False) -> CombinedLoss:
    """Build class-weighted CombinedLoss (CrossEntropy + Dice) with Median Frequency weights.

    This is the core fix for class imbalance:
    - Previously: CombinedLoss(class_weights=None)  → background dominates
    - Now:        CombinedLoss(class_weights=weights) → all 7 classes get fair gradients
    """
    class_weights = get_lova_da_class_weights().to(device)
    print_class_weights_table(class_weights)

    criterion = CombinedLoss(
        ce_weight=1.0,
        dice_weight=1.0,
        ignore_index=IGNORE_INDEX,
        num_classes=NUM_CLASSES,
        class_weights=class_weights,   # ← THE ONLY CHANGE vs original training
    )
    return criterion


def generate_val_predictions(unet_model, deeplab_model, val_loader, device, save_path: Path):
    """Generate side-by-side qualitative comparison on 2 validation samples."""
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
        axes[i, 2].set_title(f"Sample {i+1}: U-Net (Balanced)", fontsize=12, fontweight="bold")
        axes[i, 2].axis("off")
        axes[i, 3].imshow(mask_to_rgb(d_preds[i]))
        axes[i, 3].set_title(f"Sample {i+1}: DeepLabV3+ (Balanced)", fontsize=12, fontweight="bold")
        axes[i, 3].axis("off")

    plt.suptitle("Balanced Training — Validation Qualitative Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved qualitative comparison → {save_path}")


def main():
    args = parse_args()
    device = get_safe_device()
    use_amp = (not args.no_amp) and (device.type == "cuda")

    # Setup output directories
    CHANGES_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CHANGES_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHANGES_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  BALANCED TRAINING — CLASS IMBALANCE FIX")
    print("  Remote Sensing Semantic Segmentation (LoveDA)")
    print("=" * 80)
    print(f"  Device:               {device}")
    if torch.cuda.is_available():
        print(f"  GPU:                  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:                 {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"  AMP:                  {use_amp}")
    print(f"  Max Epochs:           {args.epochs}")
    print(f"  Batch Size:           {args.batch_size}")
    patience = None if args.patience <= 0 else args.patience
    print(f"  Early Stopping:       {'Disabled' if patience is None else f'Patience = {patience} epochs'}")
    print(f"  Checkpoint Dir:       {CHANGES_CHECKPOINT_DIR}")
    print(f"  Reports Dir:          {CHANGES_REPORTS_DIR}")
    print("=" * 80)
    print("\n  DATA SPLIT POLICY:")
    print("  Train split (2,522 pairs) -> used ONLY for training (gradient updates)")
    print("  Val   split (1,669 pairs) -> used ONLY for validation (checkpoint saving)")
    print("  Test  split               -> EXCLUDED COMPLETELY (never loaded)")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Step 1: Load ONLY Train and Val DataLoaders — Test excluded completely
    # -----------------------------------------------------------------------
    print("\n[Step 1/4] Loading Train and Val DataLoaders (Test excluded)...")
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=TARGET_IMAGE_SIZE,
        include_test=False,            # ← Test split is completely excluded
    )
    print(f"  Train: {len(train_loader.dataset):,} pairs | {len(train_loader)} batches/epoch")
    print(f"  Val:   {len(val_loader.dataset):,} pairs  | {len(val_loader)} batches/epoch")
    print("  Test:  NOT LOADED — excluded by design")

    # -----------------------------------------------------------------------
    # Step 2: Build Balanced Loss Function
    # -----------------------------------------------------------------------
    print("\n[Step 2/4] Building class-weighted balanced loss function...")
    criterion = build_balanced_criterion(device, use_focal=args.focal)
    print("  Loss: Weighted CrossEntropy (Median Frequency weights) + Dice Loss")

    # -----------------------------------------------------------------------
    # Step 3: Train Each Model
    # -----------------------------------------------------------------------
    models_to_train = [m.strip().lower() for m in args.models.split(",")]
    trained_models = {}
    all_histories = {}

    for model_name in models_to_train:
        print(f"\n{'=' * 80}")
        print(f"  TRAINING: {model_name.upper()} — Balanced (Class-Weighted Loss)")
        print(f"{'=' * 80}")

        # Instantiate model
        if model_name == "unet":
            model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_features=32)
            display_name = "U-Net"
        elif model_name in ("deeplabv3p", "deeplabv3plus"):
            model = DeepLabV3Plus(num_classes=NUM_CLASSES, backbone="resnet34", pretrained=False)
            display_name = "DeepLabV3+"
        else:
            print(f"  Unknown model '{model_name}'. Skipping.")
            continue

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {param_count:,} ({param_count / 1e6:.2f} M)")

        # Build optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6,
        )

        # Build trainer — passes the balanced loss criterion
        trainer = SemanticSegmentationTrainer(
            model=model,
            model_name=model_name,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=CHANGES_CHECKPOINT_DIR,
            use_amp=use_amp,
        )

        # Check for existing checkpoint to resume
        start_epoch = 1
        initial_patience = 0
        latest_ckpt = CHANGES_CHECKPOINT_DIR / f"{model_name}_latest.pth"
        best_ckpt = CHANGES_CHECKPOINT_DIR / f"{model_name}_best.pth"
        hist_csv = CHANGES_CHECKPOINT_DIR / f"{model_name}_history.csv"

        if args.resume and latest_ckpt.exists() and hist_csv.exists():
            print(f"  [Resume] Found existing checkpoint: {latest_ckpt}")
            ckpt_info = trainer.load_checkpoint(latest_ckpt)
            completed_epoch = ckpt_info.get("epoch", 0)
            start_epoch = completed_epoch + 1

            # Advance learning rate scheduler to match resumed epoch
            for _ in range(completed_epoch):
                scheduler.step()

            # Pre-load existing history records matching the checkpoint epoch
            hist_df = pd.read_csv(hist_csv)
            hist_df = hist_df[hist_df["epoch"] <= completed_epoch]
            trainer.history = hist_df.to_dict("records")

            # Restore best metrics and compute current patience count
            if best_ckpt.exists():
                best_info = torch.load(best_ckpt, map_location=device)
                trainer.best_val_miou = best_info.get("val_miou", -1.0)
                trainer.best_epoch = best_info.get("epoch", -1)
                initial_patience = max(0, completed_epoch - trainer.best_epoch)
            print(f"  [Resume] Resuming from Epoch {start_epoch}/{args.epochs}")
            print(f"  [Resume] Previous Best: Epoch {trainer.best_epoch} (mIoU: {trainer.best_val_miou:.2f}%)")
            print(f"  [Resume] Current Patience: {initial_patience}/{patience} epochs")

        # Train
        start_time = time.perf_counter()
        history_df = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            start_epoch=start_epoch,
            initial_patience_counter=initial_patience,
            early_stopping_patience=patience,
            verbose=True,
        )
        elapsed = time.perf_counter() - start_time

        best_row = history_df.loc[history_df["val_mIoU"].idxmax()]
        print(f"\n  {display_name} Training Complete:")
        print(f"    Total Time:      {elapsed / 3600:.2f} hours")
        print(f"    Best Epoch:      {int(best_row['epoch'])}")
        print(f"    Peak Val mIoU:   {best_row['val_mIoU']:.2f}%")
        print(f"    Peak Val OA:     {best_row['val_OA']:.2f}%")

        trained_models[model_name] = trainer.model
        all_histories[model_name] = history_df

    # -----------------------------------------------------------------------
    # Step 4: Save Reports and Qualitative Comparison
    # -----------------------------------------------------------------------
    print("\n[Step 4/4] Saving training histories and generating qualitative comparison...")

    for model_name, history_df in all_histories.items():
        csv_path = CHANGES_REPORTS_DIR / f"training_history_{model_name}_balanced.csv"
        history_df.to_csv(csv_path, index=False)
        print(f"  Saved training history → {csv_path}")

    if "unet" in trained_models and "deeplabv3p" in trained_models:
        qual_path = CHANGES_OUTPUTS_DIR / "val_qualitative_comparison_balanced.png"
        generate_val_predictions(
            trained_models["unet"],
            trained_models["deeplabv3p"],
            val_loader,
            device,
            qual_path,
        )

    print("\n" + "=" * 80)
    print("  BALANCED TRAINING COMPLETE")
    print(f"  Checkpoints saved in: {CHANGES_CHECKPOINT_DIR}")
    print(f"  Reports saved in:     {CHANGES_REPORTS_DIR}")
    print("  Run evaluate_on_val.py next to compute full metrics on Val split.")
    print("=" * 80)


if __name__ == "__main__":
    main()
