"""Aerial Video Semantic Segmentation Inference Pipeline.

Applies trained balanced remote sensing models (U-Net or DeepLabV3+) to arbitrary
aerial/drone video files from the internet.

Features:
- Frame-by-frame GPU inference with mixed precision.
- Optional 4-View Test-Time Augmentation (TTA).
- Multiple visual modes: 'overlay' (blended mask), 'side_by_side', or 'mask_only'.
- Visual legend bar with official LoveDA class color swatches.
- Live processing speed (FPS) counter on frames and in terminal.

Usage:
    python changes/prediction/predict_video.py --video path/to/drone_video.mp4
    python changes/prediction/predict_video.py --video path/to/drone_video.mp4 --model deeplabv3p --mode side_by_side
    python changes/prediction/predict_video.py --video path/to/drone_video.mp4 --max_frames 150 --tta
"""

import argparse
from pathlib import Path
import sys
import time

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
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
    COLOR_PALETTE,
    EVAL_CLASS_IDS,
    NUM_CLASSES,
    mask_to_rgb,
)

CHANGES_DIR = WORKSPACE_ROOT / "changes"
CHECKPOINTS_DIR = CHANGES_DIR / "checkpoints"
DEFAULT_OUTPUT_DIR = CHANGES_DIR / "prediction" / "outputs"


def load_model(model_name: str, device: torch.device):
    """Load model with best balanced weights."""
    model_name = model_name.lower().strip()
    if model_name in ["unet", "u-net"]:
        model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_features=32)
        ckpt_path = CHECKPOINTS_DIR / "unet_best.pth"
    elif model_name in ["deeplab", "deeplabv3p", "deeplabv3plus"]:
        model = DeepLabV3Plus(num_classes=NUM_CLASSES, backbone="resnet34", pretrained=False)
        ckpt_path = CHECKPOINTS_DIR / "deeplabv3p_best.pth"
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose 'unet' or 'deeplabv3p'.")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"  Loaded {model_name.upper()} checkpoint: Epoch {ckpt.get('epoch', 'N/A')} (mIoU: {ckpt.get('val_miou', 'N/A')}%)")
    return model


def preprocess_frame(frame_bgr: np.ndarray, target_size: int, device: torch.device):
    """Convert BGR video frame to normalized PyTorch tensor [1, 3, H, W]."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = frame_rgb.shape[:2]

    # Resize to model input size
    resized = cv2.resize(frame_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Normalize using ImageNet statistics
    norm = resized.astype(np.float32) / 255.0
    norm = (norm - np.array(IMAGE_MEAN, dtype=np.float32)) / np.array(IMAGE_STD, dtype=np.float32)

    # Convert to tensor [1, 3, H, W]
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor, h_orig, w_orig


@torch.no_grad()
def predict_frame(model, tensor: torch.Tensor, use_tta: bool = False) -> np.ndarray:
    """Predict semantic mask with optional 4-view TTA."""
    if not use_tta:
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    else:
        # 4-view TTA
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

    return pred


def draw_legend_bar(frame: np.ndarray) -> np.ndarray:
    """Draw a professional color legend strip at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 36
    legend_bar = np.zeros((bar_h, w, 3), dtype=np.uint8) + 25  # Dark gray background

    # Evaluated classes: Background, Building, Road, Water, Barren, Forest, Agriculture
    classes_to_show = [
        (1, "Background"),
        (2, "Building"),
        (3, "Road"),
        (4, "Water"),
        (5, "Barren"),
        (6, "Forest"),
        (7, "Agriculture"),
    ]

    num_items = len(classes_to_show)
    col_width = w // num_items

    for idx, (cid, cname) in enumerate(classes_to_show):
        rgb_color = CLASS_COLORS[cid]
        bgr_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))

        x_start = idx * col_width + 8
        # Swatch rectangle
        cv2.rectangle(legend_bar, (x_start, 10), (x_start + 16, 26), bgr_color, -1)
        cv2.rectangle(legend_bar, (x_start, 10), (x_start + 16, 26), (200, 200, 200), 1)

        # Class text
        font_scale = 0.42 if w < 1280 else 0.52
        cv2.putText(
            legend_bar,
            cname,
            (x_start + 22, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    # Attach legend bar at the bottom
    combined = np.vstack([frame, legend_bar])
    return combined


def process_video(
    video_path: Path,
    output_path: Path,
    model_name: str = "unet",
    mode: str = "overlay",
    alpha: float = 0.45,
    target_size: int = 512,
    use_tta: bool = False,
    max_frames: int = 0,
):
    device = get_safe_device()
    use_amp = device.type == "cuda"

    print("=" * 80)
    print("  AERIAL VIDEO SEMANTIC SEGMENTATION")
    print("=" * 80)
    print(f"  Input Video:      {video_path}")
    print(f"  Output Video:     {output_path}")
    print(f"  Model:            {model_name.upper()} (Balanced Checkpoint)")
    print(f"  Device:           {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Display Mode:     {mode}")
    print(f"  4-View TTA:       {'ENABLED (Higher accuracy)' if use_tta else 'DISABLED (Real-time speed)'}")
    print(f"  Alpha Opacity:    {alpha}")
    print("=" * 80)

    # 1. Load model
    model = load_model(model_name, device)

    # 2. Open input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open input video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if (fps and fps > 0) else 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_frames = total_frames if max_frames <= 0 else min(total_frames, max_frames)
    print(f"  Video Metadata:   {orig_w}x{orig_h} @ {fps:.1f} FPS | Total frames: {target_frames:,}")

    # Determine output dimensions
    if mode == "side_by_side":
        out_w = orig_w * 2
        out_h = orig_h
    else:
        out_w = orig_w
        out_h = orig_h

    # Height with bottom legend bar
    out_h_with_legend = out_h + 36

    # 3. Setup video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h_with_legend))

    if not writer.isOpened():
        # Fallback to XVID / AVI if mp4v fails
        print("  Notice: mp4v codec failed, falling back to avc1/XVID...")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h_with_legend))

    frame_idx = 0
    start_total = time.perf_counter()
    fps_rolling = 0.0

    print("\nProcessing frames...")
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret or (0 < max_frames <= frame_idx):
            break

        t_frame_start = time.perf_counter()

        # Preprocess
        tensor, h_orig, w_orig = preprocess_frame(frame_bgr, target_size, device)

        # Inference
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            pred_mask_small = predict_frame(model, tensor, use_tta=use_tta)

        # Resize mask back to original frame dimensions
        pred_mask_orig = cv2.resize(pred_mask_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # Convert mask to RGB / BGR
        colored_mask_rgb = mask_to_rgb(pred_mask_orig)
        colored_mask_bgr = cv2.cvtColor(colored_mask_rgb, cv2.COLOR_RGB2BGR)

        # Construct visual frame based on mode
        if mode == "overlay":
            # Alpha blend colored mask with original aerial frame
            annotated_frame = cv2.addWeighted(frame_bgr, 1.0 - alpha, colored_mask_bgr, alpha, 0)
        elif mode == "side_by_side":
            annotated_frame = np.hstack([frame_bgr, colored_mask_bgr])
        elif mode == "mask":
            annotated_frame = colored_mask_bgr
        else:
            annotated_frame = frame_bgr

        # Calculate live FPS
        t_frame_end = time.perf_counter()
        instant_fps = 1.0 / max(1e-5, (t_frame_end - t_frame_start))
        fps_rolling = 0.9 * fps_rolling + 0.1 * instant_fps if fps_rolling > 0 else instant_fps

        # Top banner: Model info & live FPS badge
        cv2.rectangle(annotated_frame, (10, 10), (280, 52), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (10, 10), (280, 52), (0, 255, 128), 2)
        cv2.putText(
            annotated_frame,
            f"{model_name.upper()} {'+ TTA' if use_tta else ''} | {fps_rolling:.1f} FPS",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Add bottom class legend
        final_frame = draw_legend_bar(annotated_frame)

        # Write frame
        writer.write(final_frame)
        frame_idx += 1

        if frame_idx % 25 == 0 or frame_idx == target_frames:
            elapsed = time.perf_counter() - start_total
            avg_speed = frame_idx / elapsed if elapsed > 0 else 0
            pct = (frame_idx / target_frames) * 100 if target_frames > 0 else 0
            print(f"  Frame [{frame_idx:4d}/{target_frames:4d}] ({pct:5.1f}%) | Speed: {avg_speed:5.1f} FPS | Elapsed: {elapsed:.1f}s")

    cap.release()
    writer.release()

    total_time = time.perf_counter() - start_total
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0

    print("\n" + "=" * 80)
    print("  VIDEO PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"  Output Saved to:  {output_path}")
    print(f"  Frames Processed: {frame_idx:,}")
    print(f"  Total Time:       {total_time:.2f} seconds ({total_time/60:.2f} min)")
    print(f"  Average Speed:    {avg_fps:.1f} FPS")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Predict semantic segmentation on aerial drone video")
    parser.add_argument("--video", "-v", type=str, required=True, help="Path to input aerial video (.mp4, .avi, etc.)")
    parser.add_argument("--output", "-o", type=str, default="", help="Path for output video (default: changes/prediction/outputs/)")
    parser.add_argument("--model", "-m", type=str, default="unet", choices=["unet", "deeplabv3p"], help="Model architecture")
    parser.add_argument("--mode", type=str, default="overlay", choices=["overlay", "side_by_side", "mask"], help="Visual output mode")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay opacity (0.0 to 1.0, default: 0.45)")
    parser.add_argument("--target_size", type=int, default=512, help="Internal model resolution (default: 512)")
    parser.add_argument("--tta", action="store_true", help="Enable 4-View Test-Time Augmentation (higher accuracy, ~38-74 FPS)")
    parser.add_argument("--max_frames", type=int, default=0, help="Process up to N frames (0 = entire video)")

    args = parser.parse_args()

    v_path = Path(args.video).resolve()
    if not v_path.exists():
        print(f"[ERROR] Input video not found: {v_path}")
        sys.exit(1)

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_name = f"{v_path.stem}_{args.model}_{args.mode}{'_tta' if args.tta else ''}.mp4"
        out_path = DEFAULT_OUTPUT_DIR / out_name

    process_video(
        video_path=v_path,
        output_path=out_path,
        model_name=args.model,
        mode=args.mode,
        alpha=args.alpha,
        target_size=args.target_size,
        use_tta=args.tta,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
