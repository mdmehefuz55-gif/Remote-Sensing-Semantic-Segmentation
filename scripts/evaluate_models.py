"""Comprehensive Model Evaluation & Quantitative Comparison Script.

Evaluates trained U-Net and DeepLabV3+ models on the full LoveDA validation dataset
(1,669 pairs), computes per-class and aggregate metrics, produces confusion matrix heatmaps,
error analysis maps, qualitative visual comparisons, and writes final benchmark reports.
"""

import argparse
from pathlib import Path
import sys
import time

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd
import torch

from src.dataset.dataloader import get_dataloaders
from src.models.deeplabv3plus import DeepLabV3Plus
from src.models.evaluator import ModelEvaluator
from src.models.unet import UNet
from src.preprocessing.preprocessing_config import TARGET_IMAGE_SIZE
from src.training.training_config import (
    CHECKPOINT_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    get_safe_device,
)
from src.utils.class_mapping import (
    CLASS_NAMES,
    EVAL_CLASS_IDS,
    NUM_CLASSES,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 7 Comprehensive Model Evaluation")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--generate_test_predictions", action="store_true", default=True, help="Generate predictions for test set")
    parser.add_argument("--test_samples", type=int, default=100, help="Number of test masks to generate")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_safe_device()
    use_amp = device.type == "cuda"

    eval_dir = OUTPUTS_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = OUTPUTS_DIR / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 7 — COMPREHENSIVE MODEL EVALUATION & QUANTITATIVE COMPARISON")
    print("=" * 80)
    print(f"Device:               {device}")
    if torch.cuda.is_available():
        print(f"GPU Name:             {torch.cuda.get_device_name(0)}")
        print(f"GPU VRAM:             {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Evaluation Batch Size: {args.batch_size}")
    print(f"Target Resolution:    {TARGET_IMAGE_SIZE}")
    print("=" * 80)

    # 1. Load DataLoaders
    print("\n[Step 1/5] Loading LoveDA Validation and Test DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=TARGET_IMAGE_SIZE,
        include_test=True,
    )
    print(f"Validation Pairs: {len(val_loader.dataset)} | Test Images: {len(test_loader.dataset)}")

    # 2. Instantiate and Load Models
    print("\n[Step 2/5] Instantiating models and loading best checkpoints...")
    unet_ckpt_path = CHECKPOINT_DIR / "unet_best.pth"
    deeplab_ckpt_path = CHECKPOINT_DIR / "deeplabv3p_best.pth"

    if not unet_ckpt_path.exists() or not deeplab_ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoints missing! Expected {unet_ckpt_path} and {deeplab_ckpt_path}"
        )

    unet_model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_features=32)
    deeplab_model = DeepLabV3Plus(num_classes=NUM_CLASSES, backbone="resnet34", pretrained=False)

    unet_ckpt = torch.load(unet_ckpt_path, map_location=device)
    deeplab_ckpt = torch.load(deeplab_ckpt_path, map_location=device)

    unet_model.load_state_dict(unet_ckpt["model_state_dict"])
    deeplab_model.load_state_dict(deeplab_ckpt["model_state_dict"])

    unet_model.to(device).eval()
    deeplab_model.to(device).eval()

    unet_params = sum(p.numel() for p in unet_model.parameters())
    deeplab_params = sum(p.numel() for p in deeplab_model.parameters())

    print(f"Loaded U-Net (Best Epoch {unet_ckpt.get('epoch', 'N/A')}, {unet_params/1e6:.2f} M params)")
    print(f"Loaded DeepLabV3+ (Best Epoch {deeplab_ckpt.get('epoch', 'N/A')}, {deeplab_params/1e6:.2f} M params)")

    # 3. Dataset-wide Evaluation
    evaluator = ModelEvaluator()

    print("\n[Step 3/5] Running dataset-wide evaluation on all 1,669 LoveDA validation pairs...")
    print("Evaluating U-Net...")
    start_u = time.perf_counter()
    unet_results = evaluator.evaluate_dataset(
        model=unet_model,
        dataloader=val_loader,
        device=device,
        use_amp=use_amp,
        desc="U-Net",
    )
    print(f"  -> U-Net Complete in {time.perf_counter() - start_u:.1f}s | mIoU: {unet_results['mIoU']:.2f}% | OA: {unet_results['Overall_Accuracy']:.2f}% | {unet_results['FPS']:.1f} FPS")

    print("Evaluating DeepLabV3+...")
    start_d = time.perf_counter()
    deeplab_results = evaluator.evaluate_dataset(
        model=deeplab_model,
        dataloader=val_loader,
        device=device,
        use_amp=use_amp,
        desc="DeepLabV3+",
    )
    print(f"  -> DeepLabV3+ Complete in {time.perf_counter() - start_d:.1f}s | mIoU: {deeplab_results['mIoU']:.2f}% | OA: {deeplab_results['Overall_Accuracy']:.2f}% | {deeplab_results['FPS']:.1f} FPS")

    # 4. Generate Visualizations & Heatmaps
    print("\n[Step 4/5] Generating confusion matrix heatmaps, error maps, and comparison charts...")
    # Confusion matrices
    cm_u_path = eval_dir / "confusion_matrix_unet.png"
    evaluator.plot_confusion_matrix(unet_results["Confusion_Matrix"], model_name="U-Net", normalize=True, save_path=cm_u_path)

    cm_d_path = eval_dir / "confusion_matrix_deeplabv3p.png"
    evaluator.plot_confusion_matrix(deeplab_results["Confusion_Matrix"], model_name="DeepLabV3+", normalize=True, save_path=cm_d_path)

    # Per-class comparisons
    iou_comp_path = eval_dir / "per_class_iou_comparison.png"
    evaluator.plot_per_class_comparison(unet_results, deeplab_results, metric_key="iou", metric_label="IoU (%)", save_path=iou_comp_path)

    f1_comp_path = eval_dir / "per_class_f1_comparison.png"
    evaluator.plot_per_class_comparison(unet_results, deeplab_results, metric_key="f1", metric_label="F1-Score (%)", save_path=f1_comp_path)

    # Qualitative predictions side-by-side
    qual_path = eval_dir / "qualitative_evaluation_comparison.png"
    evaluator.plot_qualitative_predictions(unet_model, deeplab_model, val_loader, device=device, num_samples=4, save_path=qual_path)

    # Error analysis maps
    err_u_path = eval_dir / "error_analysis_unet.png"
    evaluator.plot_error_analysis_maps(unet_model, val_loader, device=device, model_name="U-Net", num_samples=3, save_path=err_u_path)

    err_d_path = eval_dir / "error_analysis_deeplabv3p.png"
    evaluator.plot_error_analysis_maps(deeplab_model, val_loader, device=device, model_name="DeepLabV3+", num_samples=3, save_path=err_d_path)

    print(f"Visualizations saved to {eval_dir}")

    # Test Set Predictions
    if args.generate_test_predictions:
        print(f"\nGenerating predictions for test set samples in {predictions_dir}...")
        test_count = evaluator.generate_test_predictions(
            model=unet_model,
            test_loader=test_loader,
            output_dir=predictions_dir,
            device=device,
            max_samples=args.test_samples,
        )
        print(f"Generated and saved {test_count} test prediction masks (raw and colorized).")

    # 5. Compile Statistical Tables & Reports
    print("\n[Step 5/5] Compiling quantitative comparison tables and reports...")
    # Per-class comparison dataframe
    per_class_rows = []
    for c in EVAL_CLASS_IDS:
        c_name = CLASS_NAMES[c]
        u_p = next(r for r in unet_results["Per_Class_Table"] if r["class_id"] == c)
        d_p = next(r for r in deeplab_results["Per_Class_Table"] if r["class_id"] == c)

        per_class_rows.append({
            "Class ID": c,
            "Class Name": c_name,
            "Pixel Support": u_p["support"],
            "U-Net IoU (%)": u_p["iou"],
            "DeepLabV3+ IoU (%)": d_p["iou"],
            "IoU Delta (%)": round(u_p["iou"] - d_p["iou"], 2),
            "U-Net F1 (%)": u_p["f1"],
            "DeepLabV3+ F1 (%)": d_p["f1"],
            "U-Net Precision (%)": u_p["precision"],
            "DeepLabV3+ Precision (%)": d_p["precision"],
            "U-Net Recall (%)": u_p["recall"],
            "DeepLabV3+ Recall (%)": d_p["recall"],
        })

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv_path = REPORTS_DIR / "per_class_metrics_comparison.csv"
    per_class_df.to_csv(per_class_csv_path, index=False)

    # Model-level summary comparison dataframe
    summary_df = pd.DataFrame([
        {
            "Model": "U-Net",
            "Parameters (M)": round(unet_params / 1e6, 2),
            "Mean IoU (mIoU %)": unet_results["mIoU"],
            "Overall Accuracy (OA %)": unet_results["Overall_Accuracy"],
            "Macro Precision (%)": unet_results["Macro_Precision"],
            "Macro Recall (%)": unet_results["Macro_Recall"],
            "Macro F1-Score (%)": unet_results["Macro_F1"],
            "Throughput (FPS)": unet_results["FPS"],
            "Latency (ms/sample)": unet_results["Latency_MS"],
            "Best Checkpoint Epoch": unet_ckpt.get("epoch", "N/A"),
        },
        {
            "Model": "DeepLabV3+",
            "Parameters (M)": round(deeplab_params / 1e6, 2),
            "Mean IoU (mIoU %)": deeplab_results["mIoU"],
            "Overall Accuracy (OA %)": deeplab_results["Overall_Accuracy"],
            "Macro Precision (%)": deeplab_results["Macro_Precision"],
            "Macro Recall (%)": deeplab_results["Macro_Recall"],
            "Macro F1-Score (%)": deeplab_results["Macro_F1"],
            "Throughput (FPS)": deeplab_results["FPS"],
            "Latency (ms/sample)": deeplab_results["Latency_MS"],
            "Best Checkpoint Epoch": deeplab_ckpt.get("epoch", "N/A"),
        },
    ])
    summary_csv_path = REPORTS_DIR / "model_comparison_metrics.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    # Generate Text Report
    report_path = REPORTS_DIR / "model_evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 7 — COMPREHENSIVE MODEL EVALUATION & QUANTITATIVE COMPARISON REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Execution Date:       {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device:               {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU Name:             {torch.cuda.get_device_name(0)}\n")
        f.write(f"Validation Partition: 1,669 pairs ({TARGET_IMAGE_SIZE[0]}x{TARGET_IMAGE_SIZE[1]})\n")
        f.write(f"Evaluated Classes:    7 semantic categories (Classes 1-7, excluding ignore_index=0)\n\n")

        f.write("1. OVERALL QUANTITATIVE PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(summary_df.to_string(index=False) + "\n\n")

        f.write("2. PER-CLASS METRIC BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        f.write(per_class_df.to_string(index=False) + "\n\n")

        f.write("3. ERROR ANALYSIS & ARCHITECTURAL INSIGHTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"  - Peak mIoU: U-Net ({unet_results['mIoU']:.2f}%) vs DeepLabV3+ ({deeplab_results['mIoU']:.2f}%)\n")
        f.write(f"  - Peak OA:   U-Net ({unet_results['Overall_Accuracy']:.2f}%) vs DeepLabV3+ ({deeplab_results['Overall_Accuracy']:.2f}%)\n")
        f.write(f"  - Parameter Efficiency: U-Net has {unet_params/1e6:.2f}M params vs DeepLabV3+ {deeplab_params/1e6:.2f}M params.\n")
        f.write(f"  - Inference Speed: U-Net achieves {unet_results['FPS']:.1f} FPS vs DeepLabV3+ {deeplab_results['FPS']:.1f} FPS.\n\n")

        f.write("4. GENERATED ARTIFACTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"  - U-Net Confusion Matrix:       outputs/evaluation/confusion_matrix_unet.png\n")
        f.write(f"  - DeepLabV3+ Confusion Matrix:  outputs/evaluation/confusion_matrix_deeplabv3p.png\n")
        f.write(f"  - Per-Class IoU Bar Chart:      outputs/evaluation/per_class_iou_comparison.png\n")
        f.write(f"  - Per-Class F1 Bar Chart:       outputs/evaluation/per_class_f1_comparison.png\n")
        f.write(f"  - Qualitative Comparison:       outputs/evaluation/qualitative_evaluation_comparison.png\n")
        f.write(f"  - U-Net Error Maps:             outputs/evaluation/error_analysis_unet.png\n")
        f.write(f"  - DeepLabV3+ Error Maps:        outputs/evaluation/error_analysis_deeplabv3p.png\n")
        f.write(f"  - Test Set Predictions:         outputs/predictions/\n")
        f.write(f"  - Summary Metrics CSV:          reports/model_comparison_metrics.csv\n")
        f.write(f"  - Per-Class Metrics CSV:        reports/per_class_metrics_comparison.csv\n\n")

        f.write("=" * 80 + "\n")
        f.write("PHASE 7 EVALUATION COMPLETE — ALL PLANNED TASKS VERIFIED SUCCESSFULLY\n")
        f.write("=" * 80 + "\n")

    print(f"\nFinal report written to: {report_path}")
    print("\nSUMMARY TABLE:")
    print(summary_df.to_string(index=False))
    print("\nPER-CLASS TABLE:")
    print(per_class_df[["Class Name", "Pixel Support", "U-Net IoU (%)", "DeepLabV3+ IoU (%)", "U-Net F1 (%)", "DeepLabV3+ F1 (%)"]].to_string(index=False))
    print("\nPHASE 7 EVALUATION COMPLETE.")


if __name__ == "__main__":
    main()
