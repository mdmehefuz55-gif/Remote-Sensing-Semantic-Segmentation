"""Generates the clean cropped 3-panel aerial video:
- Extracts ONLY the sharp drone landscape footage (y: 292 to 562), discarding the top/bottom blurred letterboxes.
- Top Header:
    [ Panel 1: Original Video ] | [ Panel 2: U-Net (Balanced) ] | [ Panel 3: DeepLabV3+ (Balanced) ]
- Middle:
    Three widescreen panels: Original | U-Net Overlay | DeepLabV3+ Overlay
- Bottom Legend:
    🔴 Building | 🟡 Road | 🔵 Water | 🟢 Forest | 🟠 Agriculture | 🟣 Barren | ⚪ Background
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


def build_top_header(w_per_panel: int, header_h: int = 44) -> np.ndarray:
    total_w = w_per_panel * 3
    header = np.zeros((header_h, total_w, 3), dtype=np.uint8) + 18

    panels = [
        ("Original Video", (100, 200, 255)),
        ("U-Net (Balanced)", (0, 255, 128)),
        ("DeepLabV3+ (Balanced)", (255, 160, 50)),
    ]

    for i, (title, color) in enumerate(panels):
        x_start = i * w_per_panel
        x_end = (i + 1) * w_per_panel

        # Accent top bar
        cv2.rectangle(header, (x_start + 4, 1), (x_end - 4, 4), color, -1)

        # Title text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, thickness)
        text_x = x_start + (w_per_panel - text_w) // 2
        text_y = (header_h + text_h) // 2 + 1

        cv2.putText(header, title, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        if i > 0:
            cv2.line(header, (x_start, 6), (x_start, header_h - 6), (60, 60, 60), 1)

    return header


def build_bottom_legend(total_w: int, legend_h: int = 44) -> np.ndarray:
    legend = np.zeros((legend_h, total_w, 3), dtype=np.uint8) + 18

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
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))

        x_center = i * col_w + col_w // 2
        swatch_x = x_center - 40
        swatch_y = legend_h // 2

        cv2.circle(legend, (swatch_x, swatch_y), 8, bgr, -1)
        cv2.circle(legend, (swatch_x, swatch_y), 8, (220, 220, 220), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        thickness = 1
        cv2.putText(
            legend,
            cname,
            (swatch_x + 14, swatch_y + 5),
            font,
            font_scale,
            (240, 240, 240),
            thickness,
            cv2.LINE_AA,
        )

        if i > 0:
            cv2.line(legend, (i * col_w, 8), (i * col_w, legend_h - 8), (50, 50, 50), 1)

    cv2.line(legend, (0, 0), (total_w, 0), (60, 60, 60), 1)
    return legend


def main():
    input_video_path = WORKSPACE_ROOT / "test viedos" / "viedo 1.mp4"
    output_video_path = OUTPUT_DIR / "viedo 1_cropped_comparison.mp4"

    device = get_safe_device()
    use_amp = device.type == "cuda"
    unet, deeplab = load_models(device)

    cap = cv2.VideoCapture(str(input_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Crop bounds for the sharp 16:9 drone footage
    y_min, y_max = 292, 562
    crop_h = y_max - y_min  # 270
    crop_w = 480

    header_h = 44
    legend_h = 44
    total_w = crop_w * 3    # 1440
    total_h = header_h + crop_h + legend_h  # 358

    # Ensure even dimensions
    total_w = total_w if total_w % 2 == 0 else total_w + 1
    total_h = total_h if total_h % 2 == 0 else total_h + 1

    header_bar = build_top_header(crop_w, header_h)
    legend_bar = build_bottom_legend(total_w, legend_h)

    # Initialize video writer
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.CAP_MSMF,
        cv2.VideoWriter_fourcc(*"H264"),
        fps,
        (total_w, total_h),
    )
    if not writer.isOpened():
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

    print(f"Encoding {total_frames} cropped frames ({total_w}x{total_h})...")
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Crop to sharp drone camera region
        drone_crop = frame_bgr[y_min:y_max, :]

        # Preprocess drone crop for model
        crop_rgb = cv2.cvtColor(drone_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(crop_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        norm = (resized.astype(np.float32) / 255.0 - np.array(IMAGE_MEAN, dtype=np.float32)) / np.array(IMAGE_STD, dtype=np.float32)
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # GPU inference for both models
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                pred_u = torch.argmax(unet(tensor), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                pred_d = torch.argmax(deeplab(tensor), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Resize masks back to drone crop dimensions
        mask_u = cv2.resize(pred_u, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        mask_d = cv2.resize(pred_d, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

        color_u = cv2.cvtColor(mask_to_rgb(mask_u), cv2.COLOR_RGB2BGR)
        color_d = cv2.cvtColor(mask_to_rgb(mask_d), cv2.COLOR_RGB2BGR)

        overlay_u = cv2.addWeighted(drone_crop, 1.0 - alpha, color_u, alpha, 0)
        overlay_d = cv2.addWeighted(drone_crop, 1.0 - alpha, color_d, alpha, 0)

        middle = np.hstack([drone_crop, overlay_u, overlay_d])
        full_frame = np.vstack([header_bar, middle, legend_bar])

        # If odd pixel padding needed
        if full_frame.shape[0] != total_h or full_frame.shape[1] != total_w:
            full_frame = cv2.resize(full_frame, (total_w, total_h))

        writer.write(full_frame)
        frame_idx += 1

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            elapsed = time.perf_counter() - t0_start
            print(f"  Frame [{frame_idx:4d}/{total_frames:4d}] ({frame_idx/total_frames*100:5.1f}%) | Speed: {frame_idx/elapsed:5.1f} FPS")

    cap.release()
    writer.release()
    total_time = time.perf_counter() - t0_start
    print(f"\n[DONE] Cropped video saved to: {output_video_path}")
    print(f"Time: {total_time:.2f}s ({frame_idx/total_time:.1f} FPS)")


if __name__ == "__main__":
    main()
