"""High-Quality 3-Panel Side-by-Side Video Generator.

Layout:
- Top Header:
    [ Panel 1: Original Video ] | [ Panel 2: U-Net (Balanced) ] | [ Panel 3: DeepLabV3+ (Balanced) ]
- Middle:
    Three synchronized video panels (Original, U-Net Overlay, DeepLabV3+ Overlay)
- Bottom Legend:
    🔴 Building | 🟡 Road | 🔵 Water | 🟢 Forest | 🟠 Agriculture | 🟣 Barren | ⚪ Background

Encodes with Windows Media Foundation H.264 for universal playback compatibility.
"""

from pathlib import Path
import sys
import time

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
import numpy as np
import torch

from src.models.deeplabv3plus import DeepLabV3Plus
from src.models.unet import UNet
from src.preprocessing.preprocessing_config import IMAGE_MEAN, IMAGE_STD
from src.training.training_config import get_safe_device
from src.utils.class_mapping import CLASS_COLORS, NUM_CLASSES, mask_to_rgb

CHANGES_DIR = WORKSPACE_ROOT / "changes"
CHECKPOINTS_DIR = CHANGES_DIR / "checkpoints"
OUTPUT_DIR = CHANGES_DIR / "prediction" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_models(device: torch.device):
    unet = UNet(in_channels=3, num_classes=NUM_CLASSES, base_features=32)
    deeplab = DeepLabV3Plus(num_classes=NUM_CLASSES, backbone="resnet34", pretrained=False)

    ckpt_u = torch.load(CHECKPOINTS_DIR / "unet_best.pth", map_location=device)
    ckpt_d = torch.load(CHECKPOINTS_DIR / "deeplabv3p_best.pth", map_location=device)

    unet.load_state_dict(ckpt_u["model_state_dict"])
    deeplab.load_state_dict(ckpt_d["model_state_dict"])

    unet.to(device).eval()
    deeplab.to(device).eval()
    return unet, deeplab


def build_top_header(w_per_panel: int, header_h: int = 56) -> np.ndarray:
    """Build a crisp top header bar with 3 distinct panel titles."""
    total_w = w_per_panel * 3
    header = np.zeros((header_h, total_w, 3), dtype=np.uint8) + 20

    panels = [
        ("Original Video", (100, 200, 255)),
        ("U-Net (Balanced)", (0, 255, 128)),
        ("DeepLabV3+ (Balanced)", (255, 160, 50)),
    ]

    for i, (title, color) in enumerate(panels):
        x_start = i * w_per_panel
        x_end = (i + 1) * w_per_panel

        # Accent top line
        cv2.rectangle(header, (x_start + 4, 2), (x_end - 4, 6), color, -1)

        # Center text in panel
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.70
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, thickness)
        text_x = x_start + (w_per_panel - text_w) // 2
        text_y = (header_h + text_h) // 2 + 2

        cv2.putText(header, title, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Subtle vertical separator
        if i > 0:
            cv2.line(header, (x_start, 10), (x_start, header_h - 10), (60, 60, 60), 1)

    return header


def build_bottom_legend(total_w: int, legend_h: int = 56) -> np.ndarray:
    """Build a prominent, readable bottom legend showing all 7 classes."""
    legend = np.zeros((legend_h, total_w, 3), dtype=np.uint8) + 18

    # Classes: Building, Road, Water, Forest, Agriculture, Barren, Background
    items = [
        (2, "Building"),
        (3, "Road"),
        (4, "Water"),
        (6, "Forest"),
        (7, "Agriculture"),
        (5, "Barren"),
        (1, "Background"),
    ]

    col_w = total_w // len(items)

    for i, (cid, cname) in enumerate(items):
        rgb = CLASS_COLORS[cid]
        bgr = (int(rgb[2]), int(rgb_color_g := rgb[1]), int(rgb[0]))

        x_center = i * col_w + col_w // 2

        # Draw color circle / swatch
        swatch_x = x_center - 45
        swatch_y = legend_h // 2
        cv2.circle(legend, (swatch_x, swatch_y), 9, bgr, -1)
        cv2.circle(legend, (swatch_x, swatch_y), 9, (220, 220, 220), 1)

        # Draw label text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness = 1
        cv2.putText(
            legend,
            cname,
            (swatch_x + 16, swatch_y + 5),
            font,
            font_scale,
            (240, 240, 240),
            thickness,
            cv2.LINE_AA,
        )

        # Divider line
        if i > 0:
            cv2.line(legend, (i * col_w, 12), (i * col_w, legend_h - 12), (50, 50, 50), 1)

    # Top boundary line for legend
    cv2.line(legend, (0, 0), (total_w, 0), (70, 70, 70), 1)
    return legend


def main():
    input_video_path = WORKSPACE_ROOT / "test viedos" / "viedo 1.mp4"
    output_video_path = OUTPUT_DIR / "viedo 1_side_by_side_comparison.mp4"

    if not input_video_path.exists():
        print(f"[ERROR] Input video not found at: {input_video_path}")
        return

    device = get_safe_device()
    use_amp = device.type == "cuda"
    print("=" * 80)
    print("  GENERATING 3-PANEL SIDE-BY-SIDE VIDEO")
    print("=" * 80)
    print(f"  Input:    {input_video_path}")
    print(f"  Output:   {output_video_path}")
    print(f"  Device:   {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    unet, deeplab = load_models(device)

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print("[ERROR] Could not open input video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video:    {orig_w}x{orig_h} @ {fps:.1f} FPS | Total: {total_frames:,} frames")

    header_h = 56
    legend_h = 56
    total_w = orig_w * 3
    total_h = header_h + orig_h + legend_h

    # Ensure even dimensions for encoder
    total_w = total_w if total_w % 2 == 0 else total_w + 1
    total_h = total_h if total_h % 2 == 0 else total_h + 1

    header_bar = build_top_header(orig_w, header_h)
    legend_bar = build_bottom_legend(total_w, legend_h)

    # Setup VideoWriter - try H264 via MSMF first, fallback to mp4v
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.CAP_MSMF,
        cv2.VideoWriter_fourcc(*"H264"),
        fps,
        (total_w, total_h),
    )
    if not writer.isOpened():
        print("  Notice: MSMF H264 not available, using standard mp4v container...")
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (total_w, total_h),
        )

    alpha = 0.50
    target_size = 512
    frame_idx = 0
    t0_start = time.perf_counter()

    print("\nProcessing and encoding frames...")
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Preprocess
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        norm = (resized.astype(np.float32) / 255.0 - np.array(IMAGE_MEAN, dtype=np.float32)) / np.array(IMAGE_STD, dtype=np.float32)
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                pred_u = torch.argmax(unet(tensor), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                pred_d = torch.argmax(deeplab(tensor), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Resize masks back
        mask_u_orig = cv2.resize(pred_u, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask_d_orig = cv2.resize(pred_d, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Convert to BGR
        color_u = cv2.cvtColor(mask_to_rgb(mask_u_orig), cv2.COLOR_RGB2BGR)
        color_d = cv2.cvtColor(mask_to_rgb(mask_d_orig), cv2.COLOR_RGB2BGR)

        # Overlays
        overlay_u = cv2.addWeighted(frame_bgr, 1.0 - alpha, color_u, alpha, 0)
        overlay_d = cv2.addWeighted(frame_bgr, 1.0 - alpha, color_d, alpha, 0)

        # Stack middle 3 panels
        middle = np.hstack([frame_bgr, overlay_u, overlay_d])

        # Vertical stack: Header + Middle + Legend
        full_frame = np.vstack([header_bar, middle, legend_bar])

        writer.write(full_frame)
        frame_idx += 1

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            elapsed = time.perf_counter() - t0_start
            fps_cur = frame_idx / elapsed if elapsed > 0 else 0
            pct = (frame_idx / total_frames) * 100
            print(f"  Frame [{frame_idx:4d}/{total_frames:4d}] ({pct:5.1f}%) | Speed: {fps_cur:5.1f} FPS | Elapsed: {elapsed:.1f}s")

    cap.release()
    writer.release()

    total_time = time.perf_counter() - t0_start
    print("\n" + "=" * 80)
    print("  SIDE-BY-SIDE VIDEO READY!")
    print("=" * 80)
    print(f"  Saved to:   {output_video_path}")
    print(f"  Resolution: {total_w} x {total_h}")
    print(f"  Frames:     {frame_idx:,} frames")
    print(f"  Time:       {total_time:.2f} seconds ({frame_idx / total_time:.1f} FPS)")
    print("=" * 80)


if __name__ == "__main__":
    main()
