"""Aerial Image Semantic Segmentation Inference Pipeline.

Applies trained balanced remote sensing models (U-Net or DeepLabV3+) to arbitrary
aerial/drone images downloaded from the internet (.jpg, .png, .tiff, etc.).

Usage:
    python changes/prediction/predict_image.py --image path/to/drone_photo.jpg
    python changes/prediction/predict_image.py --image path/to/drone_photo.jpg --model deeplabv3p --tta
"""

import argparse
from pathlib import Path
import sys
import time

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.models.deeplabv3plus import DeepLabV3Plus
from src.models.unet import UNet
from src.preprocessing.preprocessing_config import IMAGE_MEAN, IMAGE_STD
from src.training.training_config import get_safe_device
from src.utils.class_mapping import (
    CLASS_COLORS,
    CLASS_NAMES,
    EVAL_CLASS_IDS,
    NUM_CLASSES,
    mask_to_rgb,
)

CHANGES_DIR = WORKSPACE_ROOT / "changes"
CHECKPOINTS_DIR = CHANGES_DIR / "checkpoints"
DEFAULT_OUTPUT_DIR = CHANGES_DIR / "prediction" / "outputs"


def load_model(model_name: str, device: torch.device):
    model_name = model_name.lower().strip()
    if model_name in ["unet", "u-net"]:
        model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_features=32)
        ckpt_path = CHECKPOINTS_DIR / "unet_best.pth"
    elif model_name in ["deeplab", "deeplabv3p", "deeplabv3plus"]:
        model = DeepLabV3Plus(num_classes=NUM_CLASSES, backbone="resnet34", pretrained=False)
        ckpt_path = CHECKPOINTS_DIR / "deeplabv3p_best.pth"
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose 'unet' or 'deeplabv3p'.")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"  Loaded {model_name.upper()}: Best Epoch {ckpt.get('epoch', 'N/A')}")
    return model


@torch.no_grad()
def predict_image(model, frame_bgr: np.ndarray, target_size: int, device: torch.device, use_tta: bool = False):
    h_orig, w_orig = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(frame_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    norm = resized.astype(np.float32) / 255.0
    norm = (norm - np.array(IMAGE_MEAN, dtype=np.float32)) / np.array(IMAGE_STD, dtype=np.float32)
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

    if not use_tta:
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    else:
        l1 = model(tensor)
        p1 = F.softmax(l1, dim=1)

        l2 = model(torch.flip(tensor, dims=[3]))
        p2 = torch.flip(F.softmax(l2, dim=1), dims=[3])

        l3 = model(torch.flip(tensor, dims=[2]))
        p3 = torch.flip(F.softmax(l3, dim=1), dims=[2])

        l4 = model(torch.flip(tensor, dims=[2, 3]))
        p4 = torch.flip(F.softmax(l4, dim=1), dims=[2, 3])

        prob = (p1 + p2 + p3 + p4) / 4.0
        pred = torch.argmax(prob, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_orig = cv2.resize(pred, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    return pred_orig, frame_rgb


def main():
    parser = argparse.ArgumentParser(description="Predict semantic segmentation on aerial drone image")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image file")
    parser.add_argument("--output", "-o", type=str, default="", help="Path for output image")
    parser.add_argument("--model", "-m", type=str, default="unet", choices=["unet", "deeplabv3p"], help="Model")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity")
    parser.add_argument("--tta", action="store_true", help="Enable 4-view TTA")
    parser.add_argument("--target_size", type=int, default=512, help="Model internal resolution")
    args = parser.parse_args()

    img_path = Path(args.image).resolve()
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    device = get_safe_device()
    model = load_model(args.model, device)

    frame_bgr = cv2.imread(str(img_path))
    if frame_bgr is None:
        print(f"[ERROR] Could not read image: {img_path}")
        sys.exit(1)

    t0 = time.perf_counter()
    pred_mask, frame_rgb = predict_image(model, frame_bgr, args.target_size, device, use_tta=args.tta)
    elapsed = (time.perf_counter() - t0) * 1000

    # Colorize mask
    colored_mask_rgb = mask_to_rgb(pred_mask)

    # Create overlay
    overlay = (frame_rgb * (1.0 - args.alpha) + colored_mask_rgb * args.alpha).astype(np.uint8)

    # Plot 3-panel figure: Original | Overlay | Mask
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(frame_rgb)
    axes[0].set_title(f"Original Aerial Image\n({frame_rgb.shape[1]}x{frame_rgb.shape[0]})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Segmentation Overlay ({args.model.upper()}{' + TTA' if args.tta else ''})\nLatency: {elapsed:.1f} ms", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(colored_mask_rgb)
    axes[2].set_title("Semantic Class Mask", fontsize=12, fontweight="bold")
    axes[2].axis("off")

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

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = DEFAULT_OUTPUT_DIR / f"{img_path.stem}_{args.model}_segmented.png"

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"\n[SUCCESS] Segmentation completed in {elapsed:.1f} ms!")
    print(f"Output saved to: {out_path}")


if __name__ == "__main__":
    main()
