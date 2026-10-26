"""Dual-Model Side-by-Side Aerial Video Semantic Segmentation.

Processes an input drone/aerial video simultaneously with BOTH trained balanced models:
- Left Panel:   Original Aerial Drone Video
- Center Panel: U-Net (Balanced, 4.32M params)
- Right Panel:  DeepLabV3+ (Balanced, 26.71M params)

Features:
- Real-time synchronous GPU processing on NVIDIA RTX 5060 Ti.
- Clear title banners on each panel.
- Live FPS throughput counter.
- Full LoveDA color legend strip along the bottom.
- Produces a high-quality playable .mp4 video file.

Usage:
    python changes/prediction/predict_video_dual.py --video "test viedos/viedo 1.mp4"
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
    NUM_CLASSES,
    mask_to_rgb,
)

CHANGES_DIR = WORKSPACE_ROOT / "changes"
CHECKPOINTS_DIR = CHANGES_DIR / "checkpoints"
DEFAULT_OUTPUT_DIR = CHANGES_DIR / "prediction" / "outputs"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_both_models(device: torch.device):
    """Load both U-Net and DeepLabV3+ with balanced weights."""
    unet_ckpt_path = CHECKPOINTS_DIR / "unet_best.pth"
    deeplab_ckpt_path = CHECKPOINTS_DIR / "deeplabv3p_best.pth"

    if not unet_ckpt_path.exists() or not deeplab_ckpt_path.exists():
        raise FileNotFoundError("Checkpoints not found in changes/checkpoints/")

    unet = UNet(in_channels=3, num_classes=NUM_CLASSES, base_features=32)
    deeplab = DeepLabV3Plus(num_classes=NUM_CLASSES, backbone="resnet34", pretrained=False)

    ckpt_u = torch.load(unet_ckpt_path, map_location=device)
    ckpt_d = torch.load(deeplab_ckpt_path, map_location=device)

    unet.load_state_dict(ckpt_u["model_state_dict"])
    deeplab.load_state_dict(ckpt_d["model_state_dict"])

    unet.to(device).eval()
    deeplab.to(device).eval()

    print(f"  [1/2] U-Net loaded:      Best Epoch {ckpt_u.get('epoch', 'N/A')} (Peak mIoU: {ckpt_u.get('val_miou', 'N/A')}%)")
    print(f"  [2/2] DeepLabV3+ loaded: Best Epoch {ckpt_d.get('epoch', 'N/A')} (Peak mIoU: {ckpt_d.get('val_miou', 'N/A')}%)")
    return unet, deeplab


def preprocess_frame(frame_bgr: np.ndarray, target_size: int, device: torch.device):
    """Convert BGR video frame to normalized PyTorch tensor [1, 3, H, W]."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = frame_rgb.shape[:2]

    resized = cv2.resize(frame_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    norm = resized.astype(np.float32) / 255.0
    norm = (norm - np.array(IMAGE_MEAN, dtype=np.float32)) / np.array(IMAGE_STD, dtype=np.float32)

    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor, h_orig, w_orig


def draw_panel_header(frame: np.ndarray, title: str, subtitle: str, color=(0, 255, 128)) -> np.ndarray:
    """Draw a clean top header card on each panel."""
    banner_h = 56
    w = frame.shape[1]
    overlay = frame.copy()

    # Dark translucent header bar
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    frame = cv2.addWeighted(frame, 0.25, overlay, 0.75, 0)

    # Accent top strip
    cv2.rectangle(frame, (0, 0), (w, 3), color, -1)

    # Title text
    cv2.putText(
        frame,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    # Subtitle text
    cv2.putText(
        frame,
        subtitle,
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return frame


def draw_bottom_legend(frame: np.ndarray) -> np.ndarray:
    """Attach a synchronized bottom legend bar across the full width."""
    h, w = frame.shape[:2]
    bar_h = 38
    legend_bar = np.zeros((bar_h, w, 3), dtype=np.uint8) + 22

    classes_to_show = [
        (1, "Background"),
        (2, "Building"),
        (3, "Road"),
        (4, "Water"),
        (5, "Barren"),
        (6, "Forest"),
        (7, "Agriculture"),
    ]

    col_width = w // len(classes_to_show)

    for idx, (cid, cname) in enumerate(classes_to_show):
        rgb_color = CLASS_COLORS[cid]
        bgr_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))

        x_start = idx * col_width + 16
        cv2.rectangle(legend_bar, (x_start, 10), (x_start + 18, 28), bgr_color, -1)
        cv2.rectangle(legend_bar, (x_start, 10), (x_start + 18, 28), (220, 220, 220), 1)

        cv2.putText(
            legend_bar,
            cname,
            (x_start + 24, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    return np.vstack([frame, legend_bar])


def process_dual_video(
    video_path: Path,
    output_path: Path,
    alpha: float = 0.50,
    target_size: int = 512,
    max_frames: int = 0,
):
    device = get_safe_device()
    use_amp = device.type == "cuda"

    print("=" * 85)
    print("  DUAL-MODEL SIDE-BY-SIDE VIDEO SEGMENTATION")
    print("=" * 85)
    print(f"  Input Video:      {video_path}")
    print(f"  Output Video:     {output_path}")
    print(f"  Models Evaluated: U-Net (Balanced) & DeepLabV3+ (Balanced)")
    print(f"  Device:           {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Layout:           [Original Video] | [U-Net Overlay] | [DeepLabV3+ Overlay]")
    print(f"  Alpha Opacity:    {alpha}")
    print("=" * 85)

    # 1. Load both models
    unet, deeplab = load_both_models(device)

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
    print(f"  Video Specs:      {orig_w}x{orig_h} @ {fps:.1f} FPS | {target_frames:,} frames")

    # Output dimension: 3 panels wide + legend
    out_w = orig_w * 3
    out_h = orig_h + 38

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        print("  Notice: mp4v codec failed, attempting avc1...")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    frame_idx = 0
    start_total = time.perf_counter()
    fps_rolling = 0.0

    print("\nProcessing synchronous frames...")
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret or (0 < max_frames <= frame_idx):
            break

        t_start = time.perf_counter()

        # Preprocess frame once for both models
        tensor, h_orig, w_orig = preprocess_frame(frame_bgr, target_size, device)

        # Simultaneous GPU inference with mixed precision
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                logits_u = unet(tensor)
                logits_d = deeplab(tensor)

            pred_u_small = torch.argmax(logits_u, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_d_small = torch.argmax(logits_d, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Resize masks back to native frame resolution
        pred_u_orig = cv2.resize(pred_u_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        pred_d_orig = cv2.resize(pred_d_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # Colorize masks
        mask_u_bgr = cv2.cvtColor(mask_to_rgb(pred_u_orig), cv2.COLOR_RGB2BGR)
        mask_d_bgr = cv2.cvtColor(mask_to_rgb(pred_d_orig), cv2.COLOR_RGB2BGR)

        # Alpha blend overlays
        overlay_u = cv2.addWeighted(frame_bgr, 1.0 - alpha, mask_u_bgr, alpha, 0)
        overlay_d = cv2.addWeighted(frame_bgr, 1.0 - alpha, mask_d_bgr, alpha, 0)

        # Compute speed
        t_elapsed = time.perf_counter() - t_start
        inst_fps = 1.0 / max(1e-5, t_elapsed)
        fps_rolling = 0.9 * fps_rolling + 0.1 * inst_fps if fps_rolling > 0 else inst_fps

        # Apply header cards on each panel
        panel_1 = draw_panel_header(frame_bgr.copy(), "1. Original Aerial Drone Video", f"Input Feed | {fps:.1f} FPS native", color=(100, 180, 255))
        panel_2 = draw_panel_header(overlay_u, "2. U-Net (Balanced)", "Skip-Conn Architecture | 4.32M params", color=(0, 255, 128))
        panel_3 = draw_panel_header(overlay_d, "3. DeepLabV3+ (Balanced)", f"ASPP ResNet-34 | {fps_rolling:.1f} Dual FPS", color=(255, 160, 50))

        # Horizontal stack [Original | U-Net | DeepLabV3+]
        tri_panel = np.hstack([panel_1, panel_2, panel_3])

        # Attach bottom legend bar
        final_frame = draw_bottom_legend(tri_panel)

        writer.write(final_frame)
        frame_idx += 1

        if frame_idx % 25 == 0 or frame_idx == target_frames:
            elapsed = time.perf_counter() - start_total
            avg_speed = frame_idx / elapsed if elapsed > 0 else 0
            pct = (frame_idx / target_frames) * 100 if target_frames > 0 else 0
            print(f"  Frame [{frame_idx:4d}/{target_frames:4d}] ({pct:5.1f}%) | Dual Speed: {avg_speed:5.1f} FPS | Elapsed: {elapsed:.1f}s")

    cap.release()
    writer.release()

    total_time = time.perf_counter() - start_total
    avg_fps = frame_idx / total_time if total_time > 0 else 0.0

    print("\n" + "=" * 85)
    print("  DUAL-MODEL VIDEO PROCESSING COMPLETE!")
    print("=" * 85)
    print(f"  Output Video:     {output_path}")
    print(f"  Frames Encoded:   {frame_idx:,} frames")
    print(f"  Total Run Time:   {total_time:.2f} seconds ({total_time/60:.2f} min)")
    print(f"  Average Speed:    {avg_fps:.1f} FPS (Both Models Concurrent)")
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="Generate 3-panel comparison video (Original | U-Net | DeepLabV3+)")
    parser.add_argument("--video", "-v", type=str, default="test viedos/viedo 1.mp4", help="Path to input video")
    parser.add_argument("--output", "-o", type=str, default="", help="Custom output video path")
    parser.add_argument("--alpha", type=float, default=0.50, help="Segmentation overlay opacity (0.0 to 1.0)")
    parser.add_argument("--target_size", type=int, default=512, help="Inference resolution")
    parser.add_argument("--max_frames", type=int, default=0, help="Limit frames (0 = all)")
    args = parser.parse_args()

    v_path = Path(args.video).resolve()
    if not v_path.exists():
        print(f"[ERROR] Input video not found: {v_path}")
        sys.exit(1)

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = DEFAULT_OUTPUT_DIR / f"{v_path.stem}_dual_models_comparison.mp4"

    process_dual_video(
        video_path=v_path,
        output_path=out_path,
        alpha=args.alpha,
        target_size=args.target_size,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
