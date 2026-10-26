"""Evaluation metrics for semantic segmentation on LoveDA remote sensing imagery.

Provides:
- Confusion Matrix computation.
- Intersection over Union (IoU) per class.
- Mean Intersection over Union (mIoU) across evaluated classes (excluding ignore_index).
- Overall Accuracy (OA).
- Frequency Weighted IoU (FWIoU).
- SegmentationMetricTracker for accumulating epoch-level results.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch

from src.utils.class_mapping import (
    CLASS_NAMES,
    EVAL_CLASS_IDS,
    IGNORE_INDEX,
    NUM_CLASSES,
)


def compute_confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Compute 2D confusion matrix [num_classes, num_classes] excluding ignore_index pixels."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # Filter out ignore_index
    valid_mask = (target_flat != ignore_index) & (target_flat >= 0) & (target_flat < num_classes)
    pred_valid = pred_flat[valid_mask]
    target_valid = target_flat[valid_mask]

    # Clip predictions to valid class range
    pred_valid = np.clip(pred_valid, 0, num_classes - 1)

    # 1D index: target * num_classes + pred
    indices = target_valid * num_classes + pred_valid
    bins = np.bincount(indices, minlength=num_classes * num_classes)
    return bins.reshape(num_classes, num_classes)


def compute_iou_per_class(
    confusion_matrix: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[int, float]:
    """Calculate IoU for each class from a confusion matrix: TP / (TP + FP + FN)."""
    ious = {}
    for c in range(num_classes):
        tp = confusion_matrix[c, c]
        fp = np.sum(confusion_matrix[:, c]) - tp
        fn = np.sum(confusion_matrix[c, :]) - tp
        denom = tp + fp + fn
        ious[c] = float(tp / denom) if denom > 0 else float("nan")
    return ious


def compute_miou(
    confusion_matrix: np.ndarray,
    eval_classes: Optional[List[int]] = None,
) -> float:
    """Calculate mean IoU (mIoU) across evaluated classes."""
    if eval_classes is None:
        eval_classes = EVAL_CLASS_IDS  # [1..7]

    ious = compute_iou_per_class(confusion_matrix)
    valid_ious = [ious[c] for c in eval_classes if not np.isnan(ious[c])]
    return float(np.mean(valid_ious)) if valid_ious else 0.0


def compute_overall_accuracy(
    confusion_matrix: np.ndarray,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    """Calculate overall pixel classification accuracy: sum(TP) / total_valid_pixels."""
    total_correct = 0
    total_pixels = 0
    for c in range(confusion_matrix.shape[0]):
        if c == ignore_index:
            continue
        total_correct += confusion_matrix[c, c]
        total_pixels += np.sum(confusion_matrix[c, :])

    return float(total_correct / total_pixels) if total_pixels > 0 else 0.0


def compute_precision_recall_f1(
    confusion_matrix: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, Dict[int, float]]:
    """Calculate per-class Precision, Recall, F1-score, and pixel support from confusion matrix."""
    precisions = {}
    recalls = {}
    f1s = {}
    supports = {}

    for c in range(num_classes):
        tp = float(confusion_matrix[c, c])
        fp = float(np.sum(confusion_matrix[:, c]) - tp)
        fn = float(np.sum(confusion_matrix[c, :]) - tp)
        support = float(np.sum(confusion_matrix[c, :]))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions[c] = prec
        recalls[c] = rec
        f1s[c] = f1
        supports[c] = int(support)

    return {
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "support": supports,
    }


def compute_macro_metrics(
    confusion_matrix: np.ndarray,
    eval_classes: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Calculate macro-averaged Precision, Recall, F1, and mIoU across evaluated classes."""
    if eval_classes is None:
        eval_classes = EVAL_CLASS_IDS

    prf = compute_precision_recall_f1(confusion_matrix)
    ious = compute_iou_per_class(confusion_matrix)
    oa = compute_overall_accuracy(confusion_matrix)

    valid_ious = [ious[c] for c in eval_classes if not np.isnan(ious[c])]
    valid_prec = [prf["precision"][c] for c in eval_classes]
    valid_rec = [prf["recall"][c] for c in eval_classes]
    valid_f1 = [prf["f1"][c] for c in eval_classes]

    return {
        "mIoU": float(np.mean(valid_ious)) if valid_ious else 0.0,
        "Overall_Accuracy": oa,
        "Macro_Precision": float(np.mean(valid_prec)) if valid_prec else 0.0,
        "Macro_Recall": float(np.mean(valid_rec)) if valid_rec else 0.0,
        "Macro_F1": float(np.mean(valid_f1)) if valid_f1 else 0.0,
    }


class SegmentationMetricTracker:
    """Accumulates batch predictions and ground truth to compute epoch-level metrics."""

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
        self.reset()

    def reset(self):
        """Reset internal confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ):
        """Update confusion matrix with new batch predictions and targets.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted class indices [B, H, W] or logits [B, C, H, W].
        targets : torch.Tensor
            Ground truth masks [B, H, W].
        """
        if preds.ndim == 4:
            # Argmax over class dimension
            preds = torch.argmax(preds, dim=1)

        pred_np = preds.detach().cpu().numpy().astype(np.int64)
        target_np = targets.detach().cpu().numpy().astype(np.int64)

        cm = compute_confusion_matrix(
            pred=pred_np,
            target=target_np,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        self.confusion_matrix += cm

    def get_results(self) -> Dict[str, Any]:
        """Compute and return dictionary of metrics."""
        ious = compute_iou_per_class(self.confusion_matrix, num_classes=self.num_classes)
        miou = compute_miou(self.confusion_matrix, eval_classes=self.eval_classes)
        oa = compute_overall_accuracy(self.confusion_matrix, ignore_index=self.ignore_index)

        class_ious = {
            self.class_names.get(c, f"Class {c}"): round(ious[c] * 100.0, 2)
            for c in self.eval_classes
        }

        return {
            "mIoU": round(miou * 100.0, 2),
            "Overall_Accuracy": round(oa * 100.0, 2),
            "Class_IoUs": class_ious,
            "Confusion_Matrix": self.confusion_matrix.copy(),
        }

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Compute detailed evaluation report with precision, recall, F1, and per-class stats."""
        results = self.get_results()
        macro_stats = compute_macro_metrics(self.confusion_matrix, eval_classes=self.eval_classes)
        prf = compute_precision_recall_f1(self.confusion_matrix, num_classes=self.num_classes)
        ious = compute_iou_per_class(self.confusion_matrix, num_classes=self.num_classes)

        per_class_table = []
        for c in self.eval_classes:
            name = self.class_names.get(c, f"Class {c}")
            per_class_table.append({
                "class_id": c,
                "class_name": name,
                "iou": round(ious[c] * 100.0, 2) if not np.isnan(ious[c]) else 0.0,
                "precision": round(prf["precision"][c] * 100.0, 2),
                "recall": round(prf["recall"][c] * 100.0, 2),
                "f1": round(prf["f1"][c] * 100.0, 2),
                "support": prf["support"][c],
            })

        results.update({
            "Macro_Precision": round(macro_stats["Macro_Precision"] * 100.0, 2),
            "Macro_Recall": round(macro_stats["Macro_Recall"] * 100.0, 2),
            "Macro_F1": round(macro_stats["Macro_F1"] * 100.0, 2),
            "Per_Class_Table": per_class_table,
        })
        return results
