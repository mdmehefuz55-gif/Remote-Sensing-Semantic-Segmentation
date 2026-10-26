# 🛰️ Remote Sensing Scene Understanding using Semantic Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(CUDA%2012.8)-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![GPU](https://img.shields.io/badge/GPU-RTX%205060%20Ti%20(16GB)-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/)
[![Dataset](https://img.shields.io/badge/Dataset-LoveDA-green?logo=kaggle&logoColor=white)](https://github.com/Junjue-Wang/LoveDA)
[![TTA](https://img.shields.io/badge/Inference-4--View%20TTA-purple)](#-4-view-test-time-augmentation-tta)
[![Real-Time Video](https://img.shields.io/badge/Video%20Inference-25--50%20FPS-brightgreen)](#-real-time-aerial-video-inference-engine)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade deep learning framework for **high-resolution remote sensing scene understanding and land-cover semantic segmentation** evaluated on the **LoveDA** (Land-cover Dataset for Domain Adaptive Semantic Segmentation) benchmark and tested on real-world **4K/HD aerial drone videos**.

Features modular deep neural network architectures (**U-Net** and **DeepLabV3+** with Atrous Spatial Pyramid Pooling), **Median Frequency Class Weighting (MFW)** to overcome severe real-world class imbalance, **4-View Test-Time Augmentation (TTA)**, and a **real-time synchronous video inference engine** that runs dual-model segmentation faster than real-time on local GPU acceleration.

---

## 📑 Table of Contents

- [Key Results & The 2-Way Comparative Benchmark](#-key-results--the-2-way-comparative-benchmark)
- [Class Imbalance Solution: Median Frequency Weighting](#-class-imbalance-solution-median-frequency-weighting)
- [4-View Test-Time Augmentation (TTA)](#-4-view-test-time-augmentation-tta)
- [Real-Time Aerial Video Inference Engine](#-real-time-aerial-video-inference-engine)
- [LoveDA Dataset & Semantic Palette](#-loveda-dataset--semantic-palette)
- [Model Architectures](#-model-architectures)
- [Repository Structure](#-repository-structure)
- [Quickstart & Usage](#-quickstart--usage)
  - [1. Real-Time Drone Video Inference](#1-real-time-drone-video-inference)
  - [2. Quick Single-Sample Test](#2-quick-single-sample-test)
  - [3. Full Validation Set Evaluation](#3-full-validation-set-evaluation)
  - [4. Interactive Jupyter Notebooks](#4-interactive-jupyter-notebooks)
- [Hardware & Optimization](#-hardware--optimization)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🏆 Key Results & The 2-Way Comparative Benchmark

All models were evaluated on the **LoveDA Validation split (1,669 image-mask pairs)** with the test split strictly isolated. The table below presents the direct **2-Way Head-to-Head Benchmark** comparing the **Original Unbalanced Baseline (Without TTA)** against the **Balanced Models + 4-View TTA**:

### 1. Overall Model Summary

| Architecture | Training & Evaluation Paradigm | Mean IoU (%) | Overall Accuracy (%) | Macro F1-Score (%) | Macro Recall (%) | Throughput |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **U-Net** | Original (Unbalanced, No TTA) | 47.32% | 66.61% | 63.60% | 61.14% | 145.2 FPS |
| **U-Net** | **Balanced + 4-View TTA** | **48.16%** | **65.91%** | **64.72%** | **65.90%** | 37.9 FPS |
| *U-Net Delta* | *Net Improvement* | **+0.84%** | *-0.70%* | **+1.12%** | **+4.76%** | *-73.9% (4 views)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DeepLabV3+** | Original (Unbalanced, No TTA) | 43.98% | 64.19% | 60.15% | 57.96% | 235.0 FPS |
| **DeepLabV3+** | **Balanced + 4-View TTA** | **46.25%** | **64.51%** | **62.95%** | **66.08%** | 74.2 FPS |
| *DeepLab Delta*| *Net Improvement* | **+2.27%** | **+0.32%** | **+2.80%** | **+8.12%** | *-68.4% (4 views)* |

---

### 2. Per-Class IoU Breakdown & Net Deltas

| Class Name | Pixel Frequency | Loss Multiplier | Unbalanced U-Net IoU | Balanced+TTA U-Net IoU | U-Net Delta | Unbalanced DeepLab IoU | Balanced+TTA DeepLab IoU | DeepLab Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Background** | 34.72% | 0.2650× | 52.30% | **50.42%** | *-1.88%* | 49.82% | **48.21%** | *-1.61%* |
| **Building** | 9.20% | 1.0000× | 59.24% | **55.23%** | *-4.01%* | 52.00% | **50.03%** | *-1.97%* |
| **Road** | 4.86% | 1.8930× | 51.84% | **52.60%** | **+0.76%** | 48.87% | **51.99%** | **+3.12%** |
| **Water** | 8.24% | 1.1165× | 52.03% | **54.46%** | **+2.43%** | 54.60% | **53.14%** | *-1.46%* |
| **Barren** | 4.71% | 1.9568× | 27.06% | **36.10%** | **+9.04%** | 19.54% | **33.08%** | **+13.54%** |
| **Forest** | 12.17% | 0.7559× | 40.43% | **40.21%** | *-0.22%* | 38.40% | **40.38%** | **+1.98%** |
| **Agriculture** | 22.78% | 0.4039× | 48.34% | **48.11%** | *-0.23%* | 44.60% | **46.92%** | **+2.32%** |

---

### 3. Class Sensitivity & Recall Breakthrough

In the baseline unweighted model, minority classes were missed at catastrophic rates because the model maximized overall accuracy by guessing dominant categories (**Background: 34.72%**, **Agriculture: 22.78%**). Rebalancing eliminated this bias, dramatically increasing recall across all classes:

| Class Name | Unbalanced U-Net Recall | Balanced+TTA U-Net Recall | Recall Gain (U-Net) | Unbalanced DeepLab Recall | Balanced+TTA DeepLab Recall | Recall Gain (DeepLab) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Barren Land** | 32.61% | **55.58%** | **+22.97%** | 22.56% | **44.86%** | **+22.30%** |
| **Road Networks**| 62.27% | **65.05%** | **+2.78%** | 60.46% | **74.13%** | **+13.67%** |
| **Water Bodies** | 61.32% | **67.21%** | **+5.89%** | 68.11% | **81.47%** | **+13.36%** |
| **Building** | 80.11% | **80.20%** | **+0.09%** | 71.65% | **82.58%** | **+10.93%** |
| **Forest** | 55.11% | **66.35%** | **+11.24%** | 52.60% | **59.97%** | **+7.37%** |
| **Agriculture** | 53.60% | **52.03%** | *-1.57%* | 49.09% | **52.92%** | **+3.83%** |
| **Background** | 82.94% | **74.89%** | *-8.05%* | 81.28% | **66.63%** | *-14.65%* |

---

## ⚖️ Class Imbalance Solution: Median Frequency Weighting

To counteract the 57% pixel concentration in Background and Agriculture, we implemented **Median Frequency Weighting (MFW)** calculated across all 2,522 training images (661,127,168 total annotated pixels):

$$w_c = \frac{\text{median}(\{f_1, f_2, \dots, f_7\})}{f_c}, \quad \text{where } \text{median}(f) = f_{\text{Building}} = 9.20\%$$

```text
Class Weights Applied:
- Background:  0.2650× (dampens majority gradient saturation)
- Building:    1.0000× (median reference anchor)
- Road:        1.8930× (near-double penalty for fractured road ribbons)
- Water:       1.1165× (boosts water boundary detection)
- Barren:      1.9568× (maximum penalty forcing soil spectral separation)
- Forest:      0.7559× (calibrated vegetation balance)
- Agriculture: 0.4039× (prevents crop encroachment into barren soil)
```

The balanced compound loss combines class-weighted Cross-Entropy with multi-class Dice Loss:
$$L_{\text{total}} = -\sum_{c=1}^{7} w_c \cdot y_c \log(\hat{y}_c) + L_{\text{Dice}}, \quad \text{with strict } \text{ignore\_index}=0$$

---

## 🔄 4-View Test-Time Augmentation (TTA)

In drone and satellite remote sensing, orientation is non-canonical. **4-View TTA** cancels orientation bias at inference time by ensembling predictions from 4 orthogonal views without any retraining:

1. **Original View**: $P_1 = \text{Softmax}(M(X))$
2. **Horizontal Flip**: $P_2 = \text{Flip}_H(\text{Softmax}(M(\text{Flip}_H(X))))$
3. **Vertical Flip**: $P_3 = \text{Flip}_V(\text{Softmax}(M(\text{Flip}_V(X))))$
4. **180° Rotation**: $P_4 = \text{Flip}_{HV}(\text{Softmax}(M(\text{Flip}_{HV}(X))))$

$$\bar{P} = \frac{1}{4}(P_1 + P_2 + P_3 + P_4), \quad \hat{Y} = \arg\max_c(\bar{P})$$

- **Boost on U-Net**: mIoU rose from 47.21% $\rightarrow$ **48.16%** (**+0.95%**)
- **Boost on DeepLabV3+**: mIoU rose from 45.38% $\rightarrow$ **46.25%** (**+0.87%**)

---

## 🎥 Real-Time Aerial Video Inference Engine

We engineered a real-time aerial video inference engine located in [`changes/prediction/`](file:///c:/Users/CUTM/Desktop/Remote-Sensing-Semantic-Segmentation/changes/prediction/) capable of processing arbitrary 4K and HD drone footage from the internet.

### Synchronized 3-Panel Video Layout
```text
┌────────────────────────────────┬────────────────────────────────┬────────────────────────────────┐
│ 1. Original Aerial Drone Video │ 2. U-Net (Balanced)            │ 3. DeepLabV3+ (Balanced)       │
│ Raw Camera Feed                │ Skip Connections | 4.32M Param │ ASPP ResNet-34   | 26.71M Param│
├────────────────────────────────┼────────────────────────────────┼────────────────────────────────┤
│                                │                                │                                │
│       Original Footage         │   U-Net Segmentation Overlay   │ DeepLabV3+ Segmentation Overlay│
│                                │                                │                                │
├────────────────────────────────┴────────────────────────────────┴────────────────────────────────┤
│ [Background: Light Grey] [Building: Red] [Road: Yellow] [Water: Blue] [Barren: Purple] ...       │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Processed Test Videos (All Included in Repository)
All 11 raw test videos from `test viedos/` were processed into high-definition 3-panel comparison videos:

| Output Video File | Source Video | Native Resolution | Frames | GPU Speed | Output Video Link |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`cropped 1.mp4`** *(or `viedo 1_cropped_comparison.mp4`)* | `viedo 1.mp4` | 480×854 (widescreen cropped) | 971 frames | **43.2 FPS** | [cropped 1.mp4](file:///c:/Users/CUTM/Desktop/Remote-Sensing-Semantic-Segmentation/changes/prediction/outputs/cropped%201.mp4) |
| **`cropped 2.mp4`** | `viedo 2.mp4` | 4K UHD (3840×2160) | 602 frames | **26.8 FPS** | [cropped 2.mp4](file:///c:/Users/CUTM/Desktop/Remote-Sensing-Semantic-Segmentation/changes/prediction/outputs/cropped%202.mp4) |
| **`cropped 3.mp4`** | `viedo 3.webm` | HD (898×506) | 440 frames | **43.8 FPS** | [cropped 3.mp4](file:///c:/Users/CUTM/Desktop/Remote-Sensing-Semantic-Segmentation/changes/prediction/outputs/cropped%203.mp4) |
| **`cropped 4.mp4`** | `viedo 4.webm` | HD (898×506 @ 60 FPS) | 1,486 frames | **43.5 FPS** | [cropped 4.mp4](file:///c:/Users/CUTM/Desktop/Remote-Sensing-Semantic-Segmentation/changes/prediction/outputs/cropped%204.mp4) |
| **`cropped 5.mp4`** | `viedo 5.mp4` | 4K UHD (3840×2160) | 2,421 frames | **25.9 FPS** | [cropped 5.mp4](file:///c:/Users/CUTM/Desktop/Remote-Sensing-Semantic-Segmentation/changes/prediction/outputs/cropped%205.mp4) |
| **`cropped 6.mp4`** | `viedo 6.mp4` | 4K UHD (3840×2160 @ 60 FPS) | 937 frames | **26.4 FPS** | [cropped 6.mp4](file:///c:/Users/CUTM/Desktop/Remote-Sensing-Semantic-Segmentation/changes/prediction/outputs/cropped%206.mp4) |
| **`cropped 7.mp4`** | `viedo 7.mp4` | 4K UHD (3840×2160) | 1,483 frames | **25.3 FPS** | [cropped 7.mp4](file:///c:/Users/CUTM/Desktop/Remote-Sensing-Semantic-Segmentation/changes/prediction/outputs/cropped%207.mp4) |
| **`cropped 8.mp4`** | `viedo 8.mp4` | 4K Vertical (2160×3840) | 858 frames | **25.5 FPS** | [cropped 8.mp4](file:///c:/Users/CUTM/Desktop/Remote-Sensing-Semantic-Segmentation/changes/prediction/outputs/cropped%208.mp4) |
| **`cropped 9.mp4`** | `viedo 9.mp4` | 4K UHD (3840×2160 @ 60 FPS) | 625 frames | **26.7 FPS** | [cropped 9.mp4](file:///c:/Users/CUTM/Desktop/Remote-Sensing-Semantic-Segmentation/changes/prediction/outputs/cropped%209.mp4) |
| **`cropped 10.mp4`** | `viedo 10.mp4` | HD (1280×720) | 240 frames | **39.4 FPS** | [cropped 10.mp4](file:///c:/Users/CUTM/Desktop/Remote-Sensing-Semantic-Segmentation/changes/prediction/outputs/cropped%2010.mp4) |
| **`cropped 11.mp4`** | `viedo 11.mp4` | HD (1280×720) | 240 frames | **41.2 FPS** | [cropped 11.mp4](file:///c:/Users/CUTM/Desktop/Remote-Sensing-Semantic-Segmentation/changes/prediction/outputs/cropped%2011.mp4) |

*All videos run at **25.3–43.8 FPS** on an RTX 5060 Ti GPU, executing dual-model inference faster than real-time playback!*

---

## 🗺️ LoveDA Dataset & Semantic Palette

The **LoveDA** (Land-cover Dataset for Domain Adaptive Semantic Segmentation) benchmark contains high-resolution (0.3m GSD) aerial imagery collected across 18 distinct urban and rural regions in China.

| ID | Class Name | Color | RGB Value | Terrestrial Objects |
| :-: | :--- | :---: | :---: | :--- |
| **0** | *Ignore / No-Data* | Black | `(0, 0, 0)` | Tile borders, unlabeled pixels (excluded from loss) |
| **1** | **Background** | Light Grey | `(200, 200, 200)` | Miscellaneous natural and urban ground |
| **2** | **Building** | 🔴 Red | `(255, 0, 0)` | Houses, commercial roofs, industrial facilities |
| **3** | **Road** | 🟡 Yellow | `(255, 255, 0)` | Highways, streets, lanes, rural dirt tracks |
| **4** | **Water** | 🔵 Blue | `(0, 0, 255)` | Rivers, lakes, ponds, canals |
| **5** | **Barren** | 🟣 Purple | `(159, 129, 183)` | Dry uncultivated soil, sandbanks, excavation sites |
| **6** | **Forest** | 🟢 Green | `(0, 255, 0)` | Tree canopies, dense woodlands, orchards |
| **7** | **Agriculture** | 🟠 Orange | `(255, 195, 128)` | Crop fields, farmlands, cultivated paddies |

---

## 🏗️ Model Architectures

| Metric / Specification | U-Net | DeepLabV3+ |
| :--- | :---: | :---: |
| **Trainable Parameters** | **4,324,520** (~4.32 M) | **26,712,744** (~26.71 M) |
| **Backbone / Encoder** | 4-Stage Symmetric DoubleConv | ResNet-34 with Dilated Convolutions |
| **Spatial Feature Skips** | Identity skip connections across 4 scales | Low-level feature fusion (Stage 1) |
| **Context Extraction** | Multi-scale hierarchical pooling | Atrous Spatial Pyramid Pooling (ASPP: rates 6, 12, 18) |
| **Optimal Balanced Epoch** | **Epoch 44** (47.21% mIoU) | **Epoch 86** (45.38% mIoU) |
| **Evaluation with 4-View TTA** | **48.16% mIoU** | **46.25% mIoU** |
| **Inference Throughput** | ~141 FPS (Single) / 38 FPS (TTA) | ~231 FPS (Single) / 74 FPS (TTA) |
| **Primary Strength** | Thin ribbons & boundary corners (Roads: 52.60%) | Contextual multi-scale integration (+2.27% gain) |

---

## 📁 Repository Structure

```text
Remote-Sensing-Semantic-Segmentation/
├── src/                                  # Modular core library
│   ├── dataset/                          # PyTorch Dataset & DataLoaders
│   ├── preprocessing/                    # Geometric & photometric augmentations
│   ├── models/                           # U-Net, DeepLabV3+, Losses, Metric Tracker
│   ├── training/                         # Trainer, checkpointing, and scheduler
│   └── utils/                            # Class mapping, palettes, visualization
├── changes/                              # Class-balanced pipeline & production weights
│   ├── checkpoints/                      # unet_best.pth (47.21%), deeplabv3p_best.pth (45.38%)
│   ├── reports/                          # Formal 2-way comparison report & CSVs
│   ├── outputs/evaluation/               # Confusion matrices, error maps, visual predictions
│   ├── prediction/                       # Video & image real-time prediction engine
│   │   ├── predict_video.py              # Single-model video inference script
│   │   ├── predict_video_dual.py         # Dual-model 3-panel video script
│   │   ├── batch_process_videos.py       # Batch video processor for 4K/HD drone footage
│   │   ├── generate_clean_cropped_video.py # Crop-optimized video generator
│   │   ├── predict_image.py              # Single-image drone photo inference script
│   │   └── outputs/                      # Generated .mp4 comparison videos & snapshots
│   ├── 00_Fix_Class_Imbalance.ipynb      # MFW derivation & loss formulations
│   ├── 05_Balanced_Model_Training.ipynb  # Full training curves & convergence analysis
│   └── 06_Balanced_Model_Evaluation.ipynb# Benchmark tables & qualitative predictions
├── test viedos/                          # Uploaded raw drone/aerial test footage
├── explanation.md                        # Exhaustive end-to-end technical report
└── README.md                             # This repository guide
```

---

## 🚀 Quickstart & Usage

### 1. Real-Time Drone Video Inference

Run synchronous dual-model segmentation on any drone/aerial video file:
```powershell
python changes/prediction/predict_video_dual.py --video "path/to/drone_video.mp4"
```
- Creates a synchronized 3-panel split video: `[Original Video] | [U-Net Overlay] | [DeepLabV3+ Overlay]`.
- Includes on-screen live FPS counter and bottom color legend.
- Automatically saves to: `changes/prediction/outputs/`.

**Single-Model Video Overlay:**
```powershell
python changes/prediction/predict_video.py --video "path/to/drone_video.mp4" --model unet --mode overlay
```

---

### 2. Quick Single-Sample Test

Test the models on any sample from the LoveDA validation split in **5 seconds**:
```powershell
python changes/test_inference.py --sample_idx 15
```
*(Loads both models, runs 4-view TTA, prints per-class IoU in the terminal, and saves a 4-panel visual comparison to `changes/outputs/`)*.

---

### 3. Full Validation Set Evaluation

Re-evaluate both balanced checkpoints with 4-view TTA across all **1,669 validation images**:
```powershell
python changes/evaluate_on_val.py
```
*(Saves confusion matrices, error maps, and CSV reports to `changes/reports/` and `changes/outputs/evaluation/`)*.

---

### 4. Interactive Jupyter Notebooks

All notebooks are self-contained and fully pre-rendered:
- **`changes/00_Fix_Class_Imbalance.ipynb`**: Mathematical derivation of class weights and loss formulation.
- **`changes/05_Balanced_Model_Training.ipynb`**: Complete 150-epoch loss curves, metric dynamics, and convergence points.
- **`changes/06_Balanced_Model_Evaluation.ipynb`**: 2-way benchmark tables, confusion matrices, and qualitative visualizations.

---

## ⚡ Hardware & Optimization

- **GPU Acceleration**: Evaluated on NVIDIA GeForce RTX 5060 Ti GPU (16 GB GDDR7 VRAM).
- **Automatic Mixed Precision (AMP)**: FP16 autocasting via `torch.amp.autocast('cuda')` doubles throughput while cutting memory consumption in half.
- **Gradient Scaling**: `torch.cuda.amp.GradScaler` prevents numerical underflow in weighted minority loss gradients.
- **Encoder Codec**: Hardware H.264 video encoding via Windows Media Foundation (`cv2.CAP_MSMF`) ensuring universal playback compatibility across VLC, Windows Media Player, and web browsers.

---

## 📜 License & Acknowledgments

- **Dataset**: LoveDA benchmark created by Junjue Wang et al. (Wuhan University).
- **Codebase**: Built under the **MIT License**.
- For full mathematical formulations and in-depth scientific analysis, see **[`explanation.md`](file:///c:/Users/CUTM/Desktop/Remote-Sensing-Semantic-Segmentation/explanation.md)**.
