"""Comprehensive Model Evaluation & Quantitative Analysis for Remote Sensing Segmentation.

Provides:
- Full-dataset evaluation across all 1,669 LoveDA validation pairs.
- Multi-metric computation: per-class IoU, mIoU, OA, Precision, Recall, and F1.
- Confusion matrix heatmaps (raw counts and row-normalized percentages).
- Comparative bar charts (U-Net vs DeepLabV3+ per-class performance).
- Qualitative visual comparison with side-by-side predictions.
- Pixel-level error analysis maps (TP vs FP / FN misclassifications).
- Test set inference pipeline generating raw and colorized prediction masks.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.metrics import (
    SegmentationMetricTracker,
    compute_macro_metrics,
    compute_precision_recall_f1,
)
from src.utils.class_mapping import (
    CLASS_COLORS,
    CLASS_NAMES,
    COLOR_PALETTE,
    EVAL_CLASS_IDS,
    EVAL_CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    mask_to_rgb,
)
from src.utils.visualization import denormalize_image_tensor


class ModelEvaluator:
    """Evaluates semantic segmentation models on LoveDA validation and test datasets."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        eval_classes: Optional[List[int]] = None,
        class_names: Optional[Dict[int, str]] = None,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eval_classes = eval_classes if eval_classes is not None else EVAL_CLASS_IDS
        self.class_names = class_names if class_names is not None else CLASS_NAMES

    @torch.no_grad()
    def evaluate_dataset(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        use_amp: bool = True,
        desc: str = "Evaluating",
    ) -> Dict[str, Any]:
        """Run dataset-wide evaluation and compute comprehensive metrics.

        Parameters
        ----------
        model : nn.Module
            Trained PyTorch segmentation model.
        dataloader : DataLoader
            Validation DataLoader.
        device : torch.device
            Evaluation device (CUDA or CPU).
        use_amp : bool
            Whether to use Automatic Mixed Precision.
        desc : str
            Description for logging.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing mIoU, OA, Macro_Precision, Macro_Recall, Macro_F1,
            Per_Class_Table, confusion matrix, and inference speed (FPS).
        """
        model.to(device).eval()
        tracker = SegmentationMetricTracker(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            eval_classes=self.eval_classes,
            class_names=self.class_names,
        )

        total_samples = 0
        total_time = 0.0

        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            batch_size = images.size(0)

            start_t = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                logits = model(images)
            preds = torch.argmax(logits, dim=1)
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

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        model_name: str = "Model",
        normalize: bool = True,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plot and save confusion matrix heatmap."""
        # Slice only evaluated classes (1 to 7)
        cm_sub = confusion_matrix[1:, 1:].astype(np.float64)
        labels = [self.class_names[c] for c in self.eval_classes]

        if normalize:
            row_sums = cm_sub.sum(axis=1, keepdims=True)
            cm_display = np.divide(cm_sub, row_sums, out=np.zeros_like(cm_sub), where=row_sums != 0) * 100.0
            fmt = ".1f"
            vmax = 100.0
            title = f"{model_name}: Normalized Confusion Matrix (%)"
            cbar_label = "Percentage (%)"
        else:
            cm_display = cm_sub
            fmt = ".0f"
            vmax = None
            title = f"{model_name}: Raw Pixel Confusion Matrix"
            cbar_label = "Pixel Count"

        fig, ax = plt.subplots(figsize=(9, 7.5))
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": cbar_label},
            ax=ax,
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Class", fontsize=11, labelpad=8)
        ax.set_ylabel("Ground Truth Class", fontsize=11, labelpad=8)
        plt.xticks(rotation=40, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_per_class_comparison(
        self,
        unet_results: Dict[str, Any],
        deeplab_results: Dict[str, Any],
        metric_key: str = "iou",
        metric_label: str = "IoU (%)",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plot comparative bar chart for per-class metrics between U-Net and DeepLabV3+."""
        u_table = {r["class_name"]: r[metric_key] for r in unet_results["Per_Class_Table"]}
        d_table = {r["class_name"]: r[metric_key] for r in deeplab_results["Per_Class_Table"]}

        class_names = [self.class_names[c] for c in self.eval_classes]
        u_scores = [u_table[c] for c in class_names]
        d_scores = [d_table[c] for c in class_names]

        x = np.arange(len(class_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width / 2, u_scores, width, label=f"U-Net (Mean: {unet_results['mIoU']:.2f}%)", color="#2563eb", alpha=0.9)
        rects2 = ax.bar(x + width / 2, d_scores, width, label=f"DeepLabV3+ (Mean: {deeplab_results['mIoU']:.2f}%)", color="#d97706", alpha=0.9)

        ax.set_title(f"Per-Class {metric_label} Comparison: U-Net vs DeepLabV3+", fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(fontsize=11)
        ax.set_ylim(0, max(max(u_scores), max(d_scores)) * 1.18)

        # Annotate bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    @torch.no_grad()
    def plot_qualitative_predictions(
        self,
        unet_model: nn.Module,
        deeplab_model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_samples: int = 4,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plot qualitative validation comparison between models and ground truth."""
        unet_model.to(device).eval()
        deeplab_model.to(device).eval()

        images, masks = next(iter(dataloader))
        images = images.to(device)
        num_samples = min(num_samples, images.size(0))

        with torch.no_grad():
            u_preds = torch.argmax(unet_model(images[:num_samples]), dim=1).cpu().numpy()
            d_preds = torch.argmax(deeplab_model(images[:num_samples]), dim=1).cpu().numpy()

        gt_masks = masks[:num_samples].cpu().numpy()
        rgb_imgs = [denormalize_image_tensor(images[i]) for i in range(num_samples)]

        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            axes[i, 0].imshow(rgb_imgs[i])
            axes[i, 0].set_title(f"Sample {i+1}: RGB Image", fontsize=11, fontweight="bold")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(mask_to_rgb(gt_masks[i]))
            axes[i, 1].set_title(f"Sample {i+1}: Ground Truth", fontsize=11, fontweight="bold")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(mask_to_rgb(u_preds[i]))
            axes[i, 2].set_title(f"Sample {i+1}: U-Net Prediction", fontsize=11, fontweight="bold")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(mask_to_rgb(d_preds[i]))
            axes[i, 3].set_title(f"Sample {i+1}: DeepLabV3+ Prediction", fontsize=11, fontweight="bold")
            axes[i, 3].axis("off")

        # Add legend
        legend_patches = [
            mpatches.Patch(color=np.array(CLASS_COLORS[c]) / 255.0, label=CLASS_NAMES[c])
            for c in EVAL_CLASS_IDS
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.02), fontsize=11)

        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    @torch.no_grad()
    def plot_error_analysis_maps(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        model_name: str = "U-Net",
        num_samples: int = 3,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plot visual error analysis maps highlighting correct predictions vs misclassified pixels."""
        model.to(device).eval()
        images, masks = next(iter(dataloader))
        images = images.to(device)
        num_samples = min(num_samples, images.size(0))

        with torch.no_grad():
            preds = torch.argmax(model(images[:num_samples]), dim=1).cpu().numpy()

        gt_masks = masks[:num_samples].cpu().numpy()
        rgb_imgs = [denormalize_image_tensor(images[i]) for i in range(num_samples)]

        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            gt = gt_masks[i]
            pr = preds[i]

            # Error map: green = correct, red = error, black = ignore
            error_map = np.zeros((*gt.shape, 3), dtype=np.uint8)
            valid_mask = (gt != self.ignore_index)
            correct_mask = valid_mask & (gt == pr)
            wrong_mask = valid_mask & (gt != pr)

            error_map[correct_mask] = [46, 204, 113]   # Green (Correct)
            error_map[wrong_mask] = [231, 76, 60]      # Red (Misclassified)
            error_map[~valid_mask] = [30, 30, 30]      # Dark Grey (Ignored)

            axes[i, 0].imshow(rgb_imgs[i])
            axes[i, 0].set_title(f"Sample {i+1}: RGB Image", fontsize=11, fontweight="bold")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(mask_to_rgb(gt))
            axes[i, 1].set_title(f"Sample {i+1}: Ground Truth", fontsize=11, fontweight="bold")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(mask_to_rgb(pr))
            axes[i, 2].set_title(f"Sample {i+1}: {model_name} Prediction", fontsize=11, fontweight="bold")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(error_map)
            acc = (np.sum(correct_mask) / np.sum(valid_mask) * 100.0) if np.sum(valid_mask) > 0 else 0.0
            axes[i, 3].set_title(f"Sample {i+1}: Error Map (Accuracy: {acc:.1f}%)", fontsize=11, fontweight="bold")
            axes[i, 3].axis("off")

        # Legend for Error Map
        legend_patches = [
            mpatches.Patch(color=[46 / 255, 204 / 255, 113 / 255], label="Correct Pixel (True Positive)"),
            mpatches.Patch(color=[231 / 255, 76 / 255, 60 / 255], label="Misclassified Pixel (Error)"),
            mpatches.Patch(color=[30 / 255, 30 / 255, 30 / 255], label="Ignore / No-Data"),
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=11)

        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    @torch.no_grad()
    def generate_test_predictions(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        output_dir: Path,
        device: torch.device,
        max_samples: Optional[int] = None,
    ) -> int:
        """Generate and save segmentation prediction masks for the unlabelled test set."""
        model.to(device).eval()
        raw_mask_dir = output_dir / "raw_masks"
        color_mask_dir = output_dir / "color_masks"
        raw_mask_dir.mkdir(parents=True, exist_ok=True)
        color_mask_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                images, stems = batch[0], batch[1]
            else:
                images, stems = batch, [f"test_{i}" for i in range(len(batch))]

            images = images.to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)

            for i in range(len(stems)):
                stem = stems[i]
                mask_arr = preds[i]

                # Save raw indexed mask
                raw_img = Image.fromarray(mask_arr, mode="L")
                raw_img.save(raw_mask_dir / f"{stem}.png")

                # Save colorized mask
                color_arr = mask_to_rgb(mask_arr)
                color_img = Image.fromarray(color_arr)
                color_img.save(color_mask_dir / f"{stem}.png")

                saved_count += 1
                if max_samples is not None and saved_count >= max_samples:
                    return saved_count

        return saved_count
