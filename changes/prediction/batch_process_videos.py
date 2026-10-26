"""Batch processor to generate 3-panel comparison videos for videos 2, 3, 4, 5.

Outputs:
- changes/prediction/outputs/cropped 2.mp4
- changes/prediction/outputs/cropped 3.mp4
- changes/prediction/outputs/cropped 4.mp4
- changes/prediction/outputs/cropped 5.mp4

Layout per video:
- Top Header:  Original Video | U-Net (Balanced) | DeepLabV3+ (Balanced)
- Middle:      [Original Drone Video] | [U-Net Overlay] | [DeepLabV3+ Overlay]
- Bottom:      Full LoveDA Color Legend Strip
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
TEST_VIDEOS_DIR = WORKSPACE_ROOT / "test viedos"
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

    font_scale = 0.44 if w_per_panel < 350 else 0.65
    thickness = 1 if w_per_panel < 350 else 2

    for i, (title, color) in enumerate(panels):
        x_start = i * w_per_panel
        x_end = (i + 1) * w_per_panel

        cv2.rectangle(header, (x_start + 4, 1), (x_end - 4, 4), color, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
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
        (7, "Agriculture" if total_w >= 1000 else "Agri"),
        (5, "Barren"),
        (1, "Background" if total_w >= 1000 else "Backgrd"),
    ]

    col_w = total_w // len(items)
    is_compact = total_w < 1000
    font_scale = 0.36 if is_compact else 0.48
    radius = 6 if is_compact else 8
    offset_x = 28 if is_compact else 40

    for i, (cid, cname) in enumerate(items):
        rgb = CLASS_COLORS[cid]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))

        x_center = i * col_w + col_w // 2
        swatch_x = x_center - offset_x
        swatch_y = legend_h // 2

        cv2.circle(legend, (swatch_x, swatch_y), radius, bgr, -1)
        cv2.circle(legend, (swatch_x, swatch_y), radius, (220, 220, 220), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        cv2.putText(
            legend,
            cname,
            (swatch_x + (10 if is_compact else 14), swatch_y + 4),
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


def process_single_video(
    input_path: Path,
    output_path: Path,
    unet,
    deeplab,
    device: torch.device,
    use_amp: bool = True,
    panel_w: int = None,
    panel_h: int = None,
    target_size: int = 512,
    alpha: float = 0.50,
):
    print("\n" + "=" * 80)
    print(f"PROCESSING: {input_path.name} -> {output_path.name}")
    print("=" * 80)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Dynamically select panel dimensions if not specified
    if panel_w is None or panel_h is None:
        if orig_h > orig_w:
            panel_w, panel_h = 270, 480
        else:
            panel_w, panel_h = 480, 270

    print(f"  Input: {orig_w}x{orig_h} @ {fps:.1f} FPS | {total_frames:,} frames (Panel: {panel_w}x{panel_h})")

    header_h = 44
    legend_h = 44
    total_w = panel_w * 3
    total_h = header_h + panel_h + legend_h

    total_w = total_w if total_w % 2 == 0 else total_w + 1
    total_h = total_h if total_h % 2 == 0 else total_h + 1

    header_bar = build_top_header(panel_w, header_h)
    legend_bar = build_bottom_legend(total_w, legend_h)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.CAP_MSMF,
        cv2.VideoWriter_fourcc(*"H264"),
        fps,
        (total_w, total_h),
    )
    if not writer.isOpened():
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (total_w, total_h),
        )

    frame_idx = 0
    t0 = time.perf_counter()
    preview_saved = False

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Resize to standard panel dimensions (16:9)
        panel_bgr = cv2.resize(frame_bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

        # Preprocess for model (512x512)
        frame_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
        resized_512 = cv2.resize(frame_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        norm = (resized_512.astype(np.float32) / 255.0 - np.array(IMAGE_MEAN, dtype=np.float32)) / np.array(IMAGE_STD, dtype=np.float32)
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # Inference for both models
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                pred_u = torch.argmax(unet(tensor), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                pred_d = torch.argmax(deeplab(tensor), dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Resize masks to panel size
        mask_u = cv2.resize(pred_u, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
        mask_d = cv2.resize(pred_d, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)

        color_u = cv2.cvtColor(mask_to_rgb(mask_u), cv2.COLOR_RGB2BGR)
        color_d = cv2.cvtColor(mask_to_rgb(mask_d), cv2.COLOR_RGB2BGR)

        overlay_u = cv2.addWeighted(panel_bgr, 1.0 - alpha, color_u, alpha, 0)
        overlay_d = cv2.addWeighted(panel_bgr, 1.0 - alpha, color_d, alpha, 0)

        middle = np.hstack([panel_bgr, overlay_u, overlay_d])
        full_frame = np.vstack([header_bar, middle, legend_bar])

        if full_frame.shape[0] != total_h or full_frame.shape[1] != total_w:
            full_frame = cv2.resize(full_frame, (total_w, total_h))

        writer.write(full_frame)
        frame_idx += 1

        # Save middle snapshot preview
        if not preview_saved and frame_idx == min(150, total_frames // 2):
            preview_path = OUTPUT_DIR / f"preview_{output_path.stem}.png"
            cv2.imwrite(str(preview_path), full_frame)
            preview_saved = True

        if frame_idx % 100 == 0 or frame_idx == total_frames:
            elapsed = time.perf_counter() - t0
            pct = (frame_idx / total_frames) * 100
            fps_cur = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  Frame [{frame_idx:4d}/{total_frames:4d}] ({pct:5.1f}%) | Speed: {fps_cur:5.1f} FPS")

    cap.release()
    writer.release()
    total_t = time.perf_counter() - t0
    print(f"[COMPLETE] {output_path.name} finished in {total_t:.1f}s ({frame_idx/total_t:.1f} FPS)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch or single video processor for aerial footage.")
    parser.add_argument("--video", type=str, default=None, help="Specific video filename in 'test viedos' to process")
    parser.add_argument("--videos", type=str, default=None, help="Comma-separated video list (e.g. '7,8,9' or 'viedo 7.mp4,viedo 8.mp4')")
    parser.add_argument("--output", type=str, default=None, help="Output filename in outputs directory")
    args = parser.parse_args()

    device = get_safe_device()
    use_amp = device.type == "cuda"
    print("=" * 80)
    print("  AERIAL VIDEO SEGMENTATION PIPELINE (DUAL MODEL)")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    unet, deeplab = load_models(device)

    if args.videos:
        raw_list = [v.strip() for v in args.videos.split(",") if v.strip()]
        videos_to_process = []
        for v in raw_list:
            if not v.startswith("viedo "):
                in_name = f"viedo {v}.mp4"
                out_name = f"cropped {v}.mp4"
            else:
                in_name = v
                num = v.replace("viedo ", "").replace(".mp4", "").replace(".webm", "")
                out_name = f"cropped {num}.mp4"
            videos_to_process.append((in_name, out_name))
    elif args.video:
        out_name = args.output or f"cropped {args.video.replace('viedo ', '').replace('.mp4', '').replace('.webm', '')}.mp4"
        videos_to_process = [(args.video, out_name)]
    else:
        videos_to_process = [
            ("viedo 2.mp4", "cropped 2.mp4"),
            ("viedo 3.webm", "cropped 3.mp4"),
            ("viedo 4.webm", "cropped 4.mp4"),
            ("viedo 5.mp4", "cropped 5.mp4"),
            ("viedo 6.mp4", "cropped 6.mp4"),
            ("viedo 7.mp4", "cropped 7.mp4"),
            ("viedo 8.mp4", "cropped 8.mp4"),
            ("viedo 9.mp4", "cropped 9.mp4"),
        ]

    start_all = time.perf_counter()

    for in_name, out_name in videos_to_process:
        in_file = TEST_VIDEOS_DIR / in_name
        out_file = OUTPUT_DIR / out_name
        if in_file.exists():
            process_single_video(
                input_path=in_file,
                output_path=out_file,
                unet=unet,
                deeplab=deeplab,
                device=device,
                use_amp=use_amp,
            )
        else:
            print(f"[WARNING] File not found: {in_file}")

    total_time_all = time.perf_counter() - start_all
    print("\n" + "=" * 80)
    print(f"VIDEO PROCESSING COMPLETED IN {total_time_all/60:.2f} MINUTES!")
    print("=" * 80)


if __name__ == "__main__":
    main()
