"""Interactive Quick-Test Script:
Loads the balanced-trained U-Net and DeepLabV3+ models with 4-View TTA,
runs inference on a validation sample (or custom image), displays the per-class IoU,
and saves a visual side-by-side comparison with color legend.

Usage:
    python changes/test_inference.py
    python changes/test_inference.py --sample_idx 42
"""

import argparse
from pathlib import Path
import sys
import time

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset.dataloader import get_dataloaders
from src.models.deeplabv3plus import DeepLabV3Plus
from src.models.unet import UNet
from src.preprocessing.preprocessing_config import TARGET_IMAGE_SIZE
from src.training.training_config import get_safe_device
from src.utils.class_mapping import (
    CLASS_COLORS,
    CLASS_NAMES,
    COLOR_PALETTE,
    EVAL_CLASS_IDS,
    NUM_CLASSES,
    mask_to_rgb,
)
from src.utils.visualization import denormalize_image_tensor

CHANGES_DIR = WORKSPACE_ROOT / "changes"
CHECKPOINTS_DIR = CHANGES_DIR / "checkpoints"
OUTPUT_DIR = CHANGES_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def predict_with_4view_tta(model, image_tensor, device):
    """Run 4-view TTA inference on a single (1, 3, H, W) image tensor."""
    model.eval()
    with torch.no_grad():
        img = image_tensor.to(device)

        # 4 orthogonal views
        l1 = model(img)
        p1 = F.softmax(l1, dim=1)

        l2 = model(torch.flip(img, dims=[3]))
        p2 = torch.flip(F.softmax(l2, dim=1), dims=[3])

        l3 = model(torch.flip(img, dims=[2]))
        p3 = torch.flip(F.softmax(l3, dim=1), dims=[2])

        l4 = model(torch.flip(img, dims=[2, 3]))
        p4 = torch.flip(F.softmax(l4, dim=1), dims=[2, 3])

        prob = (p1 + p2 + p3 + p4) / 4.0
        pred = torch.argmax(prob, dim=1).squeeze(0).cpu().numpy()
    return pred


def compute_sample_iou(pred_mask, gt_mask):
    """Compute per-class IoU for a single prediction and ground truth pair."""
    ious = {}
    for c in EVAL_CLASS_IDS:
        gt_c = (gt_mask == c)
        pred_c = (pred_mask == c)
        intersection = np.logical_and(gt_c, pred_c).sum()
        union = np.logical_or(gt_c, pred_c).sum()
        if union > 0:
            ious[CLASS_NAMES[c]] = (intersection / union) * 100.0
    return ious


def main():
    parser = argparse.ArgumentParser(description="Test balanced models with 4-view TTA")
    parser.add_argument("--sample_idx", type=int, default=15, help="Index of validation sample (0-1668)")
    args = parser.parse_args()

    device = get_safe_device()
    print("=" * 75)
    print("  QUICK INFERENCE TEST: BALANCED MODELS + 4-VIEW TTA")
    print("=" * 75)
    print(f"  Device:           {device}")
    if torch.cuda.is_available():
        print(f"  GPU:              {torch.cuda.get_device_name(0)}")

    # 1. Load checkpoints
    unet_ckpt_path = CHECKPOINTS_DIR / "unet_best.pth"
    deeplab_ckpt_path = CHECKPOINTS_DIR / "deeplabv3p_best.pth"

    if not unet_ckpt_path.exists() or not deeplab_ckpt_path.exists():
        print("[ERROR] Checkpoints not found in changes/checkpoints/")
        sys.exit(1)

    print("\n[1/3] Loading Best Model Checkpoints...")
    unet = UNet(in_channels=3, num_classes=NUM_CLASSES, base_features=32)
    deeplab = DeepLabV3Plus(num_classes=NUM_CLASSES, backbone="resnet34", pretrained=False)

    ckpt_u = torch.load(unet_ckpt_path, map_location=device)
    ckpt_d = torch.load(deeplab_ckpt_path, map_location=device)

    unet.load_state_dict(ckpt_u["model_state_dict"])
    deeplab.load_state_dict(ckpt_d["model_state_dict"])

    unet.to(device).eval()
    deeplab.to(device).eval()
    print(f"  U-Net loaded:      Best Epoch {ckpt_u.get('epoch', 'N/A')} (Peak mIoU: {ckpt_u.get('val_miou', 'N/A')}%)")
    print(f"  DeepLabV3+ loaded: Best Epoch {ckpt_d.get('epoch', 'N/A')} (Peak mIoU: {ckpt_d.get('val_miou', 'N/A')}%)")

    # 2. Fetch sample from validation set
    print(f"\n[2/3] Fetching Validation Sample #{args.sample_idx}...")
    _train_loader, val_loader = get_dataloaders(
        batch_size=1,
        num_workers=0,
        target_size=TARGET_IMAGE_SIZE,
        include_test=False,
    )
    val_dataset = val_loader.dataset

    if args.sample_idx < 0 or args.sample_idx >= len(val_dataset):
        print(f"Invalid index. Must be between 0 and {len(val_dataset)-1}")
        sys.exit(1)

    image_tensor, mask_tensor = val_dataset[args.sample_idx]
    image_tensor_batch = image_tensor.unsqueeze(0)
    gt_mask = mask_tensor.numpy()

    # 3. Run Inference with 4-View TTA
    print("\n[3/3] Running 4-View TTA Inference...")
    t0 = time.perf_counter()
    unet_pred = predict_with_4view_tta(unet, image_tensor_batch, device)
    t_u = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    deeplab_pred = predict_with_4view_tta(deeplab, image_tensor_batch, device)
    t_d = (time.perf_counter() - t0) * 1000

    print(f"  U-Net TTA Latency:      {t_u:.1f} ms")
    print(f"  DeepLabV3+ TTA Latency: {t_d:.1f} ms")

    # Metrics on this specific sample
    u_ious = compute_sample_iou(unet_pred, gt_mask)
    d_ious = compute_sample_iou(deeplab_pred, gt_mask)
    acc_u = (unet_pred == gt_mask).mean() * 100
    acc_d = (deeplab_pred == gt_mask).mean() * 100

    print("\n" + "-" * 75)
    print(f"  Sample #{args.sample_idx} Performance:")
    print(f"  U-Net Overall Pixel Accuracy:      {acc_u:.2f}% | Classes present: {list(u_ious.keys())}")
    print(f"  DeepLabV3+ Overall Pixel Accuracy: {acc_d:.2f}% | Classes present: {list(d_ious.keys())}")
    print("-" * 75)
    print(f"  {'Class Name':<15} {'U-Net IoU (%)':<18} {'DeepLabV3+ IoU (%)':<18}")
    print("-" * 75)
    all_present_classes = sorted(list(set(list(u_ious.keys()) + list(d_ious.keys()))))
    for c_name in all_present_classes:
        u_val = f"{u_ious.get(c_name, 0.0):.2f}%" if c_name in u_ious else "N/A"
        d_val = f"{d_ious.get(c_name, 0.0):.2f}%" if c_name in d_ious else "N/A"
        print(f"  {c_name:<15} {u_val:<18} {d_val:<18}")
    print("-" * 75)

    # 4. Save side-by-side visualization
    rgb_img = denormalize_image_tensor(image_tensor)
    gt_rgb = mask_to_rgb(gt_mask)
    u_rgb = mask_to_rgb(unet_pred)
    d_rgb = mask_to_rgb(deeplab_pred)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"Input Aerial Image (Sample #{args.sample_idx})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth Mask", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(u_rgb)
    axes[2].set_title(f"U-Net (Balanced + TTA)\nAccuracy: {acc_u:.2f}%", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(d_rgb)
    axes[3].set_title(f"DeepLabV3+ (Balanced + TTA)\nAccuracy: {acc_d:.2f}%", fontsize=12, fontweight="bold")
    axes[3].axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[c]) / 255.0, label=CLASS_NAMES[c])
        for c in EVAL_CLASS_IDS
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=7,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=11,
        frameon=True,
    )

    out_file = OUTPUT_DIR / f"test_prediction_sample_{args.sample_idx}.png"
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"\n[SUCCESS] Saved visual comparison to:\n  {out_file}")
    print("=" * 75)


if __name__ == "__main__":
    main()
