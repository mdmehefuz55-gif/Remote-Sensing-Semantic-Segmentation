"""README — changes/ folder.

This folder contains all code changes made to fix class imbalance
in the LoveDA Remote Sensing Segmentation project.

WHAT CHANGED:
=============
The original training (Phase 6) used CombinedLoss with NO class weights:
    CombinedLoss(ce_weight=1.0, dice_weight=1.0, class_weights=None)

This caused the model to predict Background (~35% of pixels) excessively,
because guessing the majority class minimizes unweighted cross-entropy loss.
As a result:
    - Barren IoU:     27.06% (extremely poor)
    - Road IoU:       51.84%
    - Overall mIoU:   47.32% (U-Net), 43.98% (DeepLabV3+)

THE FIX:
========
Median Frequency Class Weighting is applied to CrossEntropyLoss:
    weight_c = median(all_class_frequencies) / frequency_c

Resulting weights:
    Background (34.7%):   0.2650  ← REDUCED (was overfit to this class)
    Building   (9.2%):    1.0000  ← reference (= median class)
    Road       (4.9%):    1.8930  ← HIGH (rare linear class)
    Water      (8.2%):    1.1165
    Barren     (4.7%):    1.9568  ← HIGHEST (rarest class, hardest to detect)
    Forest     (12.2%):   0.7559
    Agriculture(22.8%):   0.4039  ← reduced

DATA SPLIT POLICY (new):
========================
    LoveDA/Train (2,522 pairs)  → Training ONLY (gradient updates)
    LoveDA/Val   (1,669 pairs)  → Testing / Evaluation ONLY (metrics)
    LoveDA/Test  (1,796 images) → EXCLUDED COMPLETELY (no masks anyway)

FILES IN THIS FOLDER:
=====================
  class_weights.py     → Computes Median Frequency weights from LoveDA stats
  train_balanced.py    → Training script with class-weighted loss (Train → Val)
  evaluate_on_val.py   → Evaluation script (Val split only, Test excluded)
  checkpoints/         → Created after training (unet_best.pth, deeplabv3p_best.pth)
  reports/             → Created after training/evaluation (CSVs, text reports)
  outputs/             → Created after evaluation (plots, confusion matrices)

HOW TO USE:
===========
  STEP 1 — Train (when ready):
      cd c:\\Users\\CUTM\\Desktop\\Remote-Sensing-Semantic-Segmentation
      python changes/train_balanced.py --epochs 150 --batch_size 8

  STEP 2 — Evaluate (after training finishes):
      python changes/evaluate_on_val.py

EXPECTED IMPROVEMENT:
=====================
  Barren IoU:   27% → ~38–42%  (+11–15%)
  Road IoU:     52% → ~58–62%  (+6–10%)
  mIoU:         47% → ~52–55%  (+5–8%)
  OA:           67% → ~69–72%  (+2–5%)
"""
