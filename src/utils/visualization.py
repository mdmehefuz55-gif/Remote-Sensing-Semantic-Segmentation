"""Visualization utilities for remote sensing images and semantic segmentation masks.

Supports plotting individual samples, image-mask-overlay comparisons,
data augmentation side-by-side verification, and batch visualization grids.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from src.utils.class_mapping import (
    CLASS_NAMES,
    CLASS_COLORS,
    mask_to_rgb,
    NUM_CLASSES,
)


def denormalize_image_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Convert a normalized [3, H, W] PyTorch tensor to [H, W, 3] uint8 numpy image.

    Parameters
    ----------
    tensor : torch.Tensor
        Image tensor of shape [3, H, W] or [H, W, 3].
    mean : Tuple[float, float, float]
        Per-channel normalization mean.
    std : Tuple[float, float, float]
        Per-channel normalization standard deviation.

    Returns
    -------
    np.ndarray
        RGB uint8 numpy array with values in [0, 255].
    """
    if isinstance(tensor, torch.Tensor):
        img = tensor.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            # [C, H, W] -> [H, W, C]
            img = np.transpose(img, (1, 2, 0))
    else:
        img = np.array(tensor)

    # Denormalize if float in range with negative values or scaled
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    img_denorm = img * std_arr + mean_arr
    img_denorm = np.clip(img_denorm * 255.0, 0, 255).astype(np.uint8)
    return img_denorm


def create_legend_handles(present_classes: Optional[List[int]] = None) -> List[mpatches.Patch]:
    """Create matplotlib legend patches for present classes."""
    classes = present_classes if present_classes is not None else list(range(NUM_CLASSES))
    handles = []
    for cid in classes:
        color = np.array(CLASS_COLORS.get(cid, (0, 0, 0))) / 255.0
        label = f"{cid}: {CLASS_NAMES.get(cid, 'Unknown')}"
        handles.append(mpatches.Patch(color=color, label=label))
    return handles


def plot_sample_and_mask(
    image: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
    title: Optional[str] = None,
    alpha: float = 0.45,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Display and optionally save a 3-panel figure: Original Image, Ground Truth Mask, Overlay.

    Parameters
    ----------
    image : Union[np.ndarray, torch.Tensor]
        RGB image array [H, W, 3] (uint8) or normalized tensor [3, H, W].
    mask : Union[np.ndarray, torch.Tensor]
        Integer mask array [H, W] or tensor [H, W].
    title : Optional[str]
        Overall figure title.
    alpha : float
        Transparency of mask overlay.
    save_path : Optional[Union[str, Path]]
        Destination path to save the generated figure.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    plt.Figure
        The rendered matplotlib Figure.
    """
    # Prepare image
    if isinstance(image, torch.Tensor):
        if image.ndim == 3 and image.shape[0] == 3:
            img_rgb = denormalize_image_tensor(image)
        else:
            img_rgb = image.detach().cpu().numpy()
    else:
        img_rgb = np.array(image)

    # Prepare mask
    if isinstance(mask, torch.Tensor):
        mask_arr = mask.detach().cpu().numpy()
    else:
        mask_arr = np.array(mask)
    if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
        mask_arr = mask_arr.squeeze(0)

    color_mask = mask_to_rgb(mask_arr)
    unique_ids = sorted(np.unique(mask_arr).tolist())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_rgb)
    axes[0].set_title("RGB Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(color_mask)
    axes[1].set_title(f"Ground Truth Mask (Classes: {unique_ids})", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(img_rgb)
    axes[2].imshow(color_mask, alpha=alpha)
    axes[2].set_title("RGB + Mask Overlay", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    # Legend
    legend_handles = create_legend_handles(unique_ids)
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(len(legend_handles), 4),
            bbox_to_anchor=(0.5, -0.05),
            fontsize=10,
            frameon=True,
        )

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    plt.tight_layout()

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_p, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_augmentation_comparison(
    orig_image: Union[np.ndarray, torch.Tensor],
    orig_mask: Union[np.ndarray, torch.Tensor],
    aug_image: Union[np.ndarray, torch.Tensor],
    aug_mask: Union[np.ndarray, torch.Tensor],
    title: str = "Data Augmentation Alignment Verification",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Display side-by-side comparison of original vs augmented image and mask pairs."""
    # Denormalize / prepare arrays
    if isinstance(orig_image, torch.Tensor):
        orig_img_arr = denormalize_image_tensor(orig_image)
    else:
        orig_img_arr = np.array(orig_image)

    if isinstance(aug_image, torch.Tensor):
        aug_img_arr = denormalize_image_tensor(aug_image)
    else:
        aug_img_arr = np.array(aug_image)

    orig_mask_arr = orig_mask.detach().cpu().numpy() if isinstance(orig_mask, torch.Tensor) else np.array(orig_mask)
    aug_mask_arr = aug_mask.detach().cpu().numpy() if isinstance(aug_mask, torch.Tensor) else np.array(aug_mask)

    if orig_mask_arr.ndim == 3:
        orig_mask_arr = orig_mask_arr.squeeze(0)
    if aug_mask_arr.ndim == 3:
        aug_mask_arr = aug_mask_arr.squeeze(0)

    orig_color = mask_to_rgb(orig_mask_arr)
    aug_color = mask_to_rgb(aug_mask_arr)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Row 0: Original
    axes[0, 0].imshow(orig_img_arr)
    axes[0, 0].set_title("Original RGB", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(orig_color)
    axes[0, 1].set_title(f"Original Mask (IDs: {sorted(np.unique(orig_mask_arr).tolist())})", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(orig_img_arr)
    axes[0, 2].imshow(orig_color, alpha=0.45)
    axes[0, 2].set_title("Original Overlay", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    # Row 1: Augmented
    axes[1, 0].imshow(aug_img_arr)
    axes[1, 0].set_title("Augmented RGB", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(aug_color)
    axes[1, 1].set_title(f"Augmented Mask (IDs: {sorted(np.unique(aug_mask_arr).tolist())})", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(aug_img_arr)
    axes[1, 2].imshow(aug_color, alpha=0.45)
    axes[1, 2].set_title("Augmented Overlay (Aligned)", fontsize=12, fontweight="bold")
    axes[1, 2].axis("off")

    all_unique = sorted(set(np.unique(orig_mask_arr).tolist() + np.unique(aug_mask_arr).tolist()))
    legend_handles = create_legend_handles(all_unique)
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(len(legend_handles), 4),
            bbox_to_anchor=(0.5, -0.03),
            fontsize=10,
            frameon=True,
        )

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_p, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_batch_samples(
    images: torch.Tensor,
    masks: torch.Tensor,
    num_samples: int = 4,
    title: str = "DataLoader Batch Samples Verification",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot a grid of batch samples: RGB Image, Ground Truth Mask, and Overlay side-by-side."""
    num_samples = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(num_samples, 3, figsize=(16, 4.5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_samples):
        img_arr = denormalize_image_tensor(images[i])
        mask_arr = masks[i].detach().cpu().numpy()
        if mask_arr.ndim == 3:
            mask_arr = mask_arr.squeeze(0)
        color_mask = mask_to_rgb(mask_arr)
        unique_classes = sorted(np.unique(mask_arr).tolist())

        # RGB Image
        axes[i, 0].imshow(img_arr)
        axes[i, 0].set_title(f"Sample {i + 1}: RGB Image", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # Ground Truth Mask
        axes[i, 1].imshow(color_mask)
        axes[i, 1].set_title(f"Sample {i + 1}: GT Mask (IDs: {unique_classes})", fontsize=11, fontweight="bold")
        axes[i, 1].axis("off")

        # Overlay
        axes[i, 2].imshow(img_arr)
        axes[i, 2].imshow(color_mask, alpha=0.45)
        axes[i, 2].set_title(f"Sample {i + 1}: Overlay", fontsize=11, fontweight="bold")
        axes[i, 2].axis("off")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_p, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
