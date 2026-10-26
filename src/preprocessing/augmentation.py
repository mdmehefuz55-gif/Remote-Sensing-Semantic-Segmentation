"""Data augmentation pipelines for training and validation in LoveDA semantic segmentation.

Ensures that:
1. Spatial transformations (flips, rotations, crops) are applied identically to both
   image and mask so pixel-level correspondence is perfectly preserved.
2. Color transformations (brightness, contrast, hue/saturation) are applied strictly
   to the image and NEVER to the integer mask.
3. Masks are NEVER normalized or bilinearly interpolated.
4. Validation pipeline uses strictly deterministic resize, normalization, and tensor
   conversion without any random augmentation.
"""

from typing import Any, Dict, Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch

from src.preprocessing.preprocessing_config import (
    AUGMENTATION_CONFIG,
    IMAGE_MEAN,
    IMAGE_STD,
    TARGET_IMAGE_SIZE,
)


class SegmentationTransformWrapper:
    """Wrapper around Albumentations Compose to ensure output masks are torch.LongTensor [H, W]."""

    def __init__(self, compose_transform: A.Compose):
        self.transform = compose_transform

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Apply transform and guarantee mask has torch.long dtype."""
        if mask is not None:
            # Ensure mask is 2D uint8 integer array
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = mask.squeeze(2)
            transformed = self.transform(image=image, mask=mask)
            # Ensure mask is torch.long [H, W]
            if isinstance(transformed["mask"], torch.Tensor):
                transformed["mask"] = transformed["mask"].long()
            else:
                transformed["mask"] = torch.from_numpy(transformed["mask"]).long()
        else:
            transformed = self.transform(image=image)

        # Ensure image is [3, H, W] float32
        if not isinstance(transformed["image"], torch.Tensor):
            transformed["image"] = torch.from_numpy(transformed["image"]).permute(2, 0, 1).float()

        return transformed


def get_training_augmentation(
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    mean: Tuple[float, float, float] = IMAGE_MEAN,
    std: Tuple[float, float, float] = IMAGE_STD,
    config: Optional[Dict[str, Any]] = None,
) -> SegmentationTransformWrapper:
    """Construct data augmentation pipeline for training samples.

    Applies identical spatial transformations to image and mask,
    color jitter to image only, normalization to image only,
    and converts to model-ready PyTorch tensors.

    Parameters
    ----------
    target_size : Tuple[int, int]
        (Height, Width) for resizing.
    mean : Tuple[float, float, float]
        Per-channel normalization mean for image.
    std : Tuple[float, float, float]
        Per-channel normalization std for image.
    config : Optional[Dict[str, Any]]
        Custom augmentation probabilities and ranges.

    Returns
    -------
    SegmentationTransformWrapper
        Callable transform taking (image=..., mask=...) and returning dict.
    """
    cfg = AUGMENTATION_CONFIG if config is None else {**AUGMENTATION_CONFIG, **config}
    target_h, target_w = target_size

    transform_list = [
        # Resize image with bilinear and mask with nearest neighbor
        A.Resize(
            height=target_h,
            width=target_w,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
        ),
        # Spatial Augmentations (applied identically to image and mask)
        A.HorizontalFlip(p=cfg.get("horizontal_flip_p", 0.5)),
        A.VerticalFlip(p=cfg.get("vertical_flip_p", 0.5)),
        A.RandomRotate90(p=cfg.get("random_rotate90_p", 0.5)),
        # Image-Only Color Augmentations (Albumentations automatically ignores mask for color transforms)
        A.ColorJitter(
            brightness=cfg.get("brightness_limit", 0.15),
            contrast=cfg.get("contrast_limit", 0.15),
            saturation=0.1,
            hue=0.05,
            p=cfg.get("color_jitter_p", 0.2),
        ),
        # Image Normalization (applied to image only, mask is untouched)
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        # Convert to PyTorch tensors
        ToTensorV2(),
    ]

    return SegmentationTransformWrapper(A.Compose(transform_list))


def get_validation_augmentation(
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    mean: Tuple[float, float, float] = IMAGE_MEAN,
    std: Tuple[float, float, float] = IMAGE_STD,
) -> SegmentationTransformWrapper:
    """Construct deterministic preprocessing pipeline for validation samples.

    Does NOT apply any random spatial or color augmentation.
    Performs deterministic resize, image normalization, and tensor conversion.

    Parameters
    ----------
    target_size : Tuple[int, int]
        (Height, Width) for resizing.
    mean : Tuple[float, float, float]
        Per-channel normalization mean.
    std : Tuple[float, float, float]
        Per-channel normalization std.

    Returns
    -------
    SegmentationTransformWrapper
        Callable transform taking (image=..., mask=...) and returning dict.
    """
    target_h, target_w = target_size

    transform_list = [
        # Deterministic resize: linear for image, nearest-neighbor for mask
        A.Resize(
            height=target_h,
            width=target_w,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
        ),
        # Image-only normalization
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        # Convert to PyTorch tensors
        ToTensorV2(),
    ]

    return SegmentationTransformWrapper(A.Compose(transform_list))


def get_test_augmentation(
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    mean: Tuple[float, float, float] = IMAGE_MEAN,
    std: Tuple[float, float, float] = IMAGE_STD,
) -> SegmentationTransformWrapper:
    """Construct deterministic preprocessing pipeline for unlabelled test samples."""
    target_h, target_w = target_size
    transform_list = [
        A.Resize(height=target_h, width=target_w, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]
    return SegmentationTransformWrapper(A.Compose(transform_list))
