"""Loss functions for remote sensing semantic segmentation.

Includes:
- Cross-Entropy Loss with ignore_index handling for no-data pixels.
- Multiclass Dice Loss supporting ignore_index.
- Focal Loss for severe class imbalance.
- Combined Loss (Cross-Entropy + Dice Loss) standard for LoveDA.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.class_mapping import IGNORE_INDEX, NUM_CLASSES


class MulticlassDiceLoss(nn.Module):
    """Multiclass Dice Loss with ignore_index masking.

    Computes Dice loss across all valid classes (excluding ignore_index).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Multiclass Dice Loss.

        Parameters
        ----------
        logits : torch.Tensor
            Logits of shape [B, C, H, W].
        targets : torch.Tensor
            Integer ground truth masks of shape [B, H, W].
        """
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # Valid pixel mask excluding ignore_index
        valid_mask = (targets != self.ignore_index)  # [B, H, W]

        dice_scores = []
        for c in range(self.num_classes):
            if c == self.ignore_index:
                continue

            prob_c = probs[:, c, :, :] * valid_mask.float()
            target_c = (targets == c).float() * valid_mask.float()

            intersection = torch.sum(prob_c * target_c)
            cardinality = torch.sum(prob_c + target_c)

            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_scores.append(dice)

        if not dice_scores:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        mean_dice = torch.mean(torch.stack(dice_scores))
        return 1.0 - mean_dice


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in semantic segmentation."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = IGNORE_INDEX,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            valid_pixels = (targets != self.ignore_index).sum().float()
            return focal_loss.sum() / torch.clamp(valid_pixels, min=1.0)
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy and Multiclass Dice Loss.

    Loss = (ce_weight * CrossEntropy) + (dice_weight * DiceLoss)
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = IGNORE_INDEX,
        num_classes: int = NUM_CLASSES,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice_loss = MulticlassDiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.ce_weight > 0.0:
            loss = loss + self.ce_weight * self.ce_loss(logits, targets)
        if self.dice_weight > 0.0:
            loss = loss + self.dice_weight * self.dice_loss(logits, targets)
        return loss


def get_loss_function(
    loss_name: str = "combined",
    ignore_index: int = IGNORE_INDEX,
    num_classes: int = NUM_CLASSES,
    class_weights: Optional[torch.Tensor] = None,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> nn.Module:
    """Build and return configured loss module."""
    name = loss_name.lower()
    if name == "ce" or name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    elif name == "dice":
        return MulticlassDiceLoss(num_classes=num_classes, ignore_index=ignore_index)
    elif name == "focal":
        return FocalLoss(alpha=class_weights, ignore_index=ignore_index)
    elif name == "combined":
        return CombinedLoss(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            ignore_index=ignore_index,
            num_classes=num_classes,
            class_weights=class_weights,
        )
    else:
        raise ValueError(f"Unknown loss function name: {loss_name}")
