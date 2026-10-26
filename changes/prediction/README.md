# Aerial Video & Image Inference Guide

This folder contains inference scripts to run semantic segmentation using your trained balanced models on **arbitrary aerial videos and photos downloaded from the internet**.

---

## 🚀 Quick Start for Videos

Place your downloaded aerial/drone video anywhere (for example, save it right in this folder as `changes/prediction/my_video.mp4`), then open PowerShell and run:

### 1. Standard Video Overlay (Real-Time Speed: ~140–230 FPS)
```powershell
python changes/prediction/predict_video.py --video changes/prediction/my_video.mp4
```
- Blends the segmentation mask semi-transparently on top of your video.
- Adds an on-screen live FPS badge and class color legend strip at the bottom.
- Saves the result to: `changes/prediction/outputs/my_video_unet_overlay.mp4`.

---

### 2. Side-by-Side Comparison Video
```powershell
python changes/prediction/predict_video.py --video changes/prediction/my_video.mp4 --mode side_by_side
```
- Creates a split video with the **Original Aerial Video on the Left** and the **Segmented Overlay on the Right**.

---

### 3. Using DeepLabV3+ with 4-View TTA (Maximum Accuracy)
```powershell
python changes/prediction/predict_video.py --video changes/prediction/my_video.mp4 --model deeplabv3p --tta
```

---

### 4. Fast Test on Just the First 10 Seconds (e.g. 150 Frames)
If you have a long video and just want to test a quick 5-second clip first:
```powershell
python changes/prediction/predict_video.py --video changes/prediction/my_video.mp4 --max_frames 150
```

---

## 🖼️ Quick Start for Photos (.jpg, .png)

If you download aerial still images:
```powershell
python changes/prediction/predict_image.py --image path/to/drone_photo.jpg --tta
```
- Generates a 3-panel figure: `[Original Photo] | [Overlay] | [Class Mask]` with color legend.
- Saves to: `changes/prediction/outputs/drone_photo_unet_segmented.png`.

---

## 🎨 LoveDA Class Color Palette Legend

| Class | Color | Description |
| :--- | :---: | :--- |
| **Background** | Light Grey | Other unclassified natural/urban structures |
| **Building** | 🔴 Red | Residential houses, commercial buildings, industrial roofs |
| **Road** | 🟡 Yellow | Highways, paved streets, rural dirt tracks |
| **Water** | 🔵 Blue | Rivers, lakes, ponds, canals |
| **Barren** | 🟣 Purple / Lilac | Exposed dry soil, sandbanks, unpaved cleared ground |
| **Forest** | 🟢 Green | Trees, dense canopies, woodlands |
| **Agriculture** | 🟠 Orange / Tan | Farmland crops, paddies, cultivated fields |

---

## ⚙️ Available Command-Line Flags

| Flag | Default | Description |
| :--- | :---: | :--- |
| `--video` / `-v` | *(Required)* | Path to your input video file (.mp4, .avi, .mov, etc.) |
| `--model` / `-m` | `unet` | Choose architecture: `unet` or `deeplabv3p` |
| `--mode` | `overlay` | Visual mode: `overlay`, `side_by_side`, or `mask` |
| `--alpha` | `0.45` | Overlay transparency (0.0 = original, 1.0 = solid mask) |
| `--tta` | `False` | Enable 4-View Test-Time Augmentation |
| `--max_frames` | `0` (all) | Limit processing to first N frames |
| `--output` / `-o` | *(Auto)* | Custom output path for the processed video |
