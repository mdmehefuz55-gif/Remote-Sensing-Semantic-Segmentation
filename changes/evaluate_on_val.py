"""Evaluation Script — Val Split Only (Test Excluded).

Evaluates the BALANCED-trained models (from changes/checkpoints/) on the
LoveDA Validation split only. The Test split is completely excluded.

KEY DIFFERENCES from original evaluate_models.py:
  1. Loads checkpoints from changes/checkpoints/ (balanced-trained weights)
  2. Uses ONLY val_loader for ALL metric computation (no test_loader at all)
  3. Test split is never loaded, never referenced, never used
  4. Saves results to changes/reports/ and changes/outputs/evaluation/

Data Flow:
    LoveDA/Train  → NOT USED in evaluation
    LoveDA/Val    (1,669 pairs) → ALL metrics computed here (mIoU, OA, F1, etc.)
    LoveDA/Test   → EXCLUDED completely
"""

from pathlib import Path
import sys
import time

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

CHANGES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHANGES_DIR))

import numpy as np
import pandas as pd
import torch

from src.dataset.dataloader import get_dataloaders
from src.models.deeplabv3plus import DeepLabV3Plus
from src.models.evaluator import ModelEvaluator
from src.models.metrics import SegmentationMetricTracker
from src.models.unet import UNet
from src.preprocessing.preprocessing_config import TARGET_IMAGE_SIZE
from src.training.training_config import get_safe_device
from src.utils.class_mapping import CLASS_NAMES, EVAL_CLASS_IDS, IGNORE_INDEX, NUM_CLASSES

# Output paths — isolated inside changes/
CHANGES_CHECKPOINT_DIR = CHANGES_DIR / "checkpoints"
CHANGES_REPORTS_DIR = CHANGES_DIR / "reports"
CHANGES_EVAL_DIR = CHANGES_DIR / "outputs" / "evaluation"


@torch.no_grad()
def evaluate_dataset_with_tta(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool = True,
    desc: str = "Evaluating with 4-View TTA",
):
    model.to(device).eval()
    tracker = SegmentationMetricTracker(
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        eval_classes=EVAL_CLASS_IDS,
        class_names=CLASS_NAMES,
    )
    total_samples = 0
    total_time = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        batch_size = images.size(0)

        start_t = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            # View 1: Original
            l1 = model(images)
            p1 = torch.softmax(l1, dim=1)

            # View 2: Horizontal flip
            l2 = model(torch.flip(images, dims=[3]))
            p2 = torch.flip(torch.softmax(l2, dim=1), dims=[3])

            # View 3: Vertical flip
            l3 = model(torch.flip(images, dims=[2]))
            p3 = torch.flip(torch.softmax(l3, dim=1), dims=[2])

            # View 4: 180-deg rotation (both H and V flip)
            l4 = model(torch.flip(images, dims=[2, 3]))
            p4 = torch.flip(torch.softmax(l4, dim=1), dims=[2, 3])

            prob = (p1 + p2 + p3 + p4) / 4.0
            preds = torch.argmax(prob, dim=1)

        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_time = time.perf_counter() - start_t
        total_time += batch_time
        total_samples += batch_size
        tracker.update(preds, masks)

    results = tracker.get_comprehensive_report()
    fps = total_samples / total_time if total_time > 0 else 0.0
    latency_ms = (total_time / total_samples) * 1000.0 if total_samples > 0 else 0.0
    results["Total_Samples"] = total_samples
    results["Total_Time_Sec"] = round(total_time, 2)
    results["FPS"] = round(fps, 2)
    results["Latency_MS"] = round(latency_ms, 2)
    return results


def main():
    device = get_safe_device()
    use_amp = device.type == "cuda"

    CHANGES_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHANGES_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  BALANCED MODEL EVALUATION — VAL SPLIT ONLY")
    print("  Test split is EXCLUDED — never loaded, never referenced")
    print("=" * 80)
    print(f"  Device:           {device}")
    if torch.cuda.is_available():
        print(f"  GPU:              {torch.cuda.get_device_name(0)}")
    print(f"  Checkpoint Dir:   {CHANGES_CHECKPOINT_DIR}")
    print(f"  Evaluation Dir:   {CHANGES_EVAL_DIR}")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Step 1: Load Val DataLoader ONLY — Test completely excluded
    # -----------------------------------------------------------------------
    print("\n[Step 1/4] Loading Val DataLoader (Test excluded entirely)...")
    _train_loader, val_loader = get_dataloaders(
        batch_size=8,
        num_workers=0,
        target_size=TARGET_IMAGE_SIZE,
        include_test=False,       # ← Test is NOT included by design
    )
    print(f"  Val DataLoader:   {len(val_loader.dataset):,} image-mask pairs ({len(val_loader)} batches)")
    print("  Test DataLoader:  NOT CREATED — excluded by design")

    # -----------------------------------------------------------------------
    # Step 2: Load Balanced-Trained Model Checkpoints
    # -----------------------------------------------------------------------
    print("\n[Step 2/4] Loading balanced-trained model checkpoints...")
    unet_ckpt_path = CHANGES_CHECKPOINT_DIR / "unet_best.pth"
    deeplab_ckpt_path = CHANGES_CHECKPOINT_DIR / "deeplabv3p_best.pth"

    missing = []
    if not unet_ckpt_path.exists():
        missing.append(str(unet_ckpt_path))
    if not deeplab_ckpt_path.exists():
        missing.append(str(deeplab_ckpt_path))

    if missing:
        print("\n  [ERROR] Missing checkpoints:")
        for m in missing:
            print(f"    - {m}")
        print("\n  Please run train_balanced.py first to generate the checkpoints.")
        print("  Command: python changes/train_balanced.py")
        sys.exit(1)

    unet_model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_features=32)
    deeplab_model = DeepLabV3Plus(num_classes=NUM_CLASSES, backbone="resnet34", pretrained=False)

    unet_ckpt = torch.load(unet_ckpt_path, map_location=device)
    deeplab_ckpt = torch.load(deeplab_ckpt_path, map_location=device)

    unet_model.load_state_dict(unet_ckpt["model_state_dict"])
    deeplab_model.load_state_dict(deeplab_ckpt["model_state_dict"])

    unet_model.to(device).eval()
    deeplab_model.to(device).eval()

    u_params = sum(p.numel() for p in unet_model.parameters())
    d_params = sum(p.numel() for p in deeplab_model.parameters())
    print(f"  U-Net loaded:      Best Epoch {unet_ckpt.get('epoch', 'N/A')} | {u_params/1e6:.2f} M params")
    print(f"  DeepLabV3+ loaded: Best Epoch {deeplab_ckpt.get('epoch', 'N/A')} | {d_params/1e6:.2f} M params")

    # -----------------------------------------------------------------------
    # Step 3: Run Evaluation on Val Split (No Test data, No Test metrics)
    # -----------------------------------------------------------------------
    print("\n[Step 3/4] Running full quantitative evaluation on Val split (1,669 pairs)...")
    evaluator = ModelEvaluator()

    print("  Evaluating U-Net (Balanced)...")
    t0 = time.perf_counter()
    unet_results = evaluator.evaluate_dataset(
        model=unet_model,
        dataloader=val_loader,    # ← ONLY val_loader — not test_loader
        device=device,
        use_amp=use_amp,
        desc="U-Net Balanced",
    )
    t_u = time.perf_counter() - t0
    print(f"    mIoU: {unet_results['mIoU']:.2f}% | OA: {unet_results['Overall_Accuracy']:.2f}% "
          f"| {unet_results['FPS']:.1f} FPS | {t_u:.1f}s")

    print("  Evaluating DeepLabV3+ (Balanced)...")
    t0 = time.perf_counter()
    deeplab_results = evaluator.evaluate_dataset(
        model=deeplab_model,
        dataloader=val_loader,    # ← ONLY val_loader — not test_loader
        device=device,
        use_amp=use_amp,
        desc="DeepLabV3+ Balanced",
    )
    t_d = time.perf_counter() - t0
    print(f"    mIoU: {deeplab_results['mIoU']:.2f}% | OA: {deeplab_results['Overall_Accuracy']:.2f}% "
          f"| {deeplab_results['FPS']:.1f} FPS | {t_d:.1f}s")

    print("  Evaluating U-Net (Balanced + 4-View TTA)...")
    t0 = time.perf_counter()
    unet_tta_results = evaluate_dataset_with_tta(
        model=unet_model,
        dataloader=val_loader,
        device=device,
        use_amp=use_amp,
        desc="U-Net Balanced + 4-View TTA",
    )
    t_u_tta = time.perf_counter() - t0
    print(f"    TTA mIoU: {unet_tta_results['mIoU']:.2f}% | TTA OA: {unet_tta_results['Overall_Accuracy']:.2f}% "
          f"| {unet_tta_results['FPS']:.1f} FPS | {t_u_tta:.1f}s")

    print("  Evaluating DeepLabV3+ (Balanced + 4-View TTA)...")
    t0 = time.perf_counter()
    deeplab_tta_results = evaluate_dataset_with_tta(
        model=deeplab_model,
        dataloader=val_loader,
        device=device,
        use_amp=use_amp,
        desc="DeepLabV3+ Balanced + 4-View TTA",
    )
    t_d_tta = time.perf_counter() - t0
    print(f"    TTA mIoU: {deeplab_tta_results['mIoU']:.2f}% | TTA OA: {deeplab_tta_results['Overall_Accuracy']:.2f}% "
          f"| {deeplab_tta_results['FPS']:.1f} FPS | {t_d_tta:.1f}s")

    # -----------------------------------------------------------------------
    # Step 4: Save Visualizations and Reports
    # -----------------------------------------------------------------------
    print("\n[Step 4/4] Generating confusion matrices, charts, and reports...")

    # Confusion matrices
    evaluator.plot_confusion_matrix(
        unet_results["Confusion_Matrix"], model_name="U-Net (Balanced)", normalize=True,
        save_path=CHANGES_EVAL_DIR / "confusion_matrix_unet_balanced.png"
    )
    evaluator.plot_confusion_matrix(
        deeplab_results["Confusion_Matrix"], model_name="DeepLabV3+ (Balanced)", normalize=True,
        save_path=CHANGES_EVAL_DIR / "confusion_matrix_deeplabv3p_balanced.png"
    )

    # Per-class comparison charts
    evaluator.plot_per_class_comparison(
        unet_results, deeplab_results, metric_key="iou", metric_label="IoU (%)",
        save_path=CHANGES_EVAL_DIR / "per_class_iou_comparison_balanced.png"
    )
    evaluator.plot_per_class_comparison(
        unet_results, deeplab_results, metric_key="f1", metric_label="F1-Score (%)",
        save_path=CHANGES_EVAL_DIR / "per_class_f1_comparison_balanced.png"
    )

    # Qualitative side-by-side
    evaluator.plot_qualitative_predictions(
        unet_model, deeplab_model, val_loader, device=device, num_samples=4,
        save_path=CHANGES_EVAL_DIR / "qualitative_comparison_balanced.png"
    )

    # Error maps
    evaluator.plot_error_analysis_maps(
        unet_model, val_loader, device=device, model_name="U-Net (Balanced)", num_samples=3,
        save_path=CHANGES_EVAL_DIR / "error_analysis_unet_balanced.png"
    )
    evaluator.plot_error_analysis_maps(
        deeplab_model, val_loader, device=device, model_name="DeepLabV3+ (Balanced)", num_samples=3,
        save_path=CHANGES_EVAL_DIR / "error_analysis_deeplabv3p_balanced.png"
    )

    # Per-class metrics CSV
    per_class_rows = []
    for c in EVAL_CLASS_IDS:
        c_name = CLASS_NAMES[c]
        u_p = next(r for r in unet_results["Per_Class_Table"] if r["class_id"] == c)
        d_p = next(r for r in deeplab_results["Per_Class_Table"] if r["class_id"] == c)
        u_tta_p = next(r for r in unet_tta_results["Per_Class_Table"] if r["class_id"] == c)
        d_tta_p = next(r for r in deeplab_tta_results["Per_Class_Table"] if r["class_id"] == c)
        per_class_rows.append({
            "Class ID": c,
            "Class Name": c_name,
            "Pixel Support": u_p["support"],
            "U-Net IoU (%)": u_p["iou"],
            "DeepLabV3+ IoU (%)": d_p["iou"],
            "U-Net (TTA) IoU (%)": u_tta_p["iou"],
            "DeepLabV3+ (TTA) IoU (%)": d_tta_p["iou"],
            "U-Net (TTA) Recall (%)": u_tta_p["recall"],
            "DeepLabV3+ (TTA) Recall (%)": d_tta_p["recall"],
            "U-Net (TTA) F1 (%)": u_tta_p["f1"],
            "DeepLabV3+ (TTA) F1 (%)": d_tta_p["f1"],
        })

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv = CHANGES_REPORTS_DIR / "per_class_metrics_balanced.csv"
    per_class_df.to_csv(per_class_csv, index=False)

    # Summary metrics CSV
    summary_df = pd.DataFrame([
        {
            "Model": "U-Net (Balanced + 4-View TTA)",
            "Parameters (M)": round(u_params / 1e6, 2),
            "Mean IoU (mIoU %)": unet_tta_results["mIoU"],
            "Overall Accuracy (OA %)": unet_tta_results["Overall_Accuracy"],
            "Macro Precision (%)": unet_tta_results.get("Macro_Precision", 0),
            "Macro Recall (%)": unet_tta_results.get("Macro_Recall", 0),
            "Macro F1-Score (%)": unet_tta_results.get("Macro_F1", 0),
            "Throughput (FPS)": unet_tta_results["FPS"],
            "Best Epoch": unet_ckpt.get("epoch", "N/A"),
            "Evaluation Split": "Validation (1,669 pairs)",
            "Test Split Used": "NO — excluded by design",
        },
        {
            "Model": "DeepLabV3+ (Balanced + 4-View TTA)",
            "Parameters (M)": round(d_params / 1e6, 2),
            "Mean IoU (mIoU %)": deeplab_tta_results["mIoU"],
            "Overall Accuracy (OA %)": deeplab_tta_results["Overall_Accuracy"],
            "Macro Precision (%)": deeplab_tta_results.get("Macro_Precision", 0),
            "Macro Recall (%)": deeplab_tta_results.get("Macro_Recall", 0),
            "Macro F1-Score (%)": deeplab_tta_results.get("Macro_F1", 0),
            "Throughput (FPS)": deeplab_tta_results["FPS"],
            "Best Epoch": deeplab_ckpt.get("epoch", "N/A"),
            "Evaluation Split": "Validation (1,669 pairs)",
            "Test Split Used": "NO — excluded by design",
        },
        {
            "Model": "U-Net (Balanced)",
            "Parameters (M)": round(u_params / 1e6, 2),
            "Mean IoU (mIoU %)": unet_results["mIoU"],
            "Overall Accuracy (OA %)": unet_results["Overall_Accuracy"],
            "Macro Precision (%)": unet_results.get("Macro_Precision", 0),
            "Macro Recall (%)": unet_results.get("Macro_Recall", 0),
            "Macro F1-Score (%)": unet_results.get("Macro_F1", 0),
            "Throughput (FPS)": unet_results["FPS"],
            "Best Epoch": unet_ckpt.get("epoch", "N/A"),
            "Evaluation Split": "Validation (1,669 pairs)",
            "Test Split Used": "NO — excluded by design",
        },
        {
            "Model": "DeepLabV3+ (Balanced)",
            "Parameters (M)": round(d_params / 1e6, 2),
            "Mean IoU (mIoU %)": deeplab_results["mIoU"],
            "Overall Accuracy (OA %)": deeplab_results["Overall_Accuracy"],
            "Macro Precision (%)": deeplab_results.get("Macro_Precision", 0),
            "Macro Recall (%)": deeplab_results.get("Macro_Recall", 0),
            "Macro F1-Score (%)": deeplab_results.get("Macro_F1", 0),
            "Throughput (FPS)": deeplab_results["FPS"],
            "Best Epoch": deeplab_ckpt.get("epoch", "N/A"),
            "Evaluation Split": "Validation (1,669 pairs)",
            "Test Split Used": "NO — excluded by design",
        },
    ])
    summary_csv = CHANGES_REPORTS_DIR / "model_comparison_balanced.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Text report
    report_path = CHANGES_REPORTS_DIR / "evaluation_report_balanced.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BALANCED MODEL + 4-VIEW TTA EVALUATION REPORT\n")
        f.write("Evaluation Split: LoveDA Validation Set (1,669 pairs)\n")
        f.write("Test Split:       EXCLUDED — not used in this evaluation\n")
        f.write("Class Imbalance:  FIXED via Median Frequency Class Weighting\n")
        f.write("=" * 80 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\nPer-Class Breakdown:\n")
        f.write(per_class_df.to_string(index=False))
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("EVALUATION COMPLETE\n")
        f.write("=" * 80 + "\n")

    print(f"\n  Saved: {per_class_csv}")
    print(f"  Saved: {summary_csv}")
    print(f"  Saved: {report_path}")
    print(f"  Saved: Visualizations in {CHANGES_EVAL_DIR}")

    print("\n" + "=" * 80)
    print("  FINAL RESULTS — BALANCED MODELS + 4-VIEW TTA (Val Split)")
    print("=" * 80)
    print(f"  U-Net    (Balanced + TTA): mIoU = {unet_tta_results['mIoU']:.2f}% | OA = {unet_tta_results['Overall_Accuracy']:.2f}%")
    print(f"  DeepLab+ (Balanced + TTA): mIoU = {deeplab_tta_results['mIoU']:.2f}% | OA = {deeplab_tta_results['Overall_Accuracy']:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
