"""Mask preprocessing pipeline for LoveDA remote sensing segmentation masks.

Strictly preserves integer class labels without color normalization,
uses nearest-neighbor interpolation for resizing, and outputs class-index
tensors of shape [H, W] with dtype torch.int64 suitable for segmentation losses.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch

from src.preprocessing.preprocessing_config import TARGET_IMAGE_SIZE
from src.utils.class_mapping import IGNORE_INDEX, RAW_CLASS_IDS


def load_mask(mask_path: Union[str, Path]) -> np.ndarray:
    """Read a semantic segmentation mask from disk as a 2D integer array.

    Parameters
    ----------
    mask_path : Union[str, Path]
        Path to mask file.

    Returns
    -------
    np.ndarray
        2D integer array of shape [H, W] and dtype uint8.
    """
    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask not found at {path.resolve()}")

    with Image.open(path) as img:
        arr = np.array(img, dtype=np.uint8)

    # In case mask was saved as RGB or has extra dimension
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr.squeeze(2)
        elif arr.shape[2] >= 3:
            # If multi-channel, check if channels are identical (grayscale saved as RGB)
            if np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 0], arr[..., 2]):
                arr = arr[..., 0]
            else:
                raise ValueError(
                    f"Expected single-channel integer mask, got multi-channel array with varying channels: shape {arr.shape}"
                )

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask [H, W], got shape {arr.shape} for {path}")

    return arr


def resize_mask(
    mask: np.ndarray,
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
) -> np.ndarray:
    """Resize mask to target dimensions strictly using NEAREST-NEIGHBOR interpolation.

    NEVER uses bilinear or bicubic interpolation to avoid creating non-existent class labels.

    Parameters
    ----------
    mask : np.ndarray
        Source 2D integer mask [H, W].
    target_size : Tuple[int, int]
        (Target Height, Target Width).

    Returns
    -------
    np.ndarray
        Resized 2D integer mask of shape [target_size[0], target_size[1]] and dtype uint8.
    """
    target_h, target_w = target_size
    if mask.shape[0] == target_h and mask.shape[1] == target_w:
        return mask.copy()

    # cv2.resize expects (width, height). INTER_NEAREST strictly preserves integer labels.
    resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.uint8)


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """Convert a 2D integer mask [H, W] to a PyTorch LongTensor [H, W].

    Parameters
    ----------
    mask : np.ndarray
        2D integer array of shape [H, W].

    Returns
    -------
    torch.Tensor
        Class-index tensor of shape [H, W] and dtype torch.int64 (Long).
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask array [H, W], got shape {mask.shape}")

    return torch.from_numpy(mask.astype(np.int64)).long()


def validate_mask_values(
    mask: Union[np.ndarray, torch.Tensor],
    allowed_ids: Optional[List[int]] = None,
) -> Tuple[bool, List[int]]:
    """Verify that a mask contains only valid class IDs.

    Parameters
    ----------
    mask : Union[np.ndarray, torch.Tensor]
        2D mask array or tensor.
    allowed_ids : Optional[List[int]]
        List of valid class IDs (default: RAW_CLASS_IDS [0..7]).

    Returns
    -------
    Tuple[bool, List[int]]
        (is_valid, list of unique class IDs found).
    """
    if allowed_ids is None:
        allowed_ids = RAW_CLASS_IDS

    if isinstance(mask, torch.Tensor):
        unique_vals = torch.unique(mask).cpu().tolist()
    else:
        unique_vals = np.unique(mask).tolist()

    valid_set = set(allowed_ids)
    is_valid = all(v in valid_set for v in unique_vals)
    return is_valid, sorted(unique_vals)


def preprocess_mask(
    mask_input: Union[str, Path, np.ndarray],
    target_size: Optional[Tuple[int, int]] = TARGET_IMAGE_SIZE,
    allowed_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """Complete mask preprocessing pipeline.

    Reads -> Resizes (Nearest Neighbor) -> Validates Class IDs -> Converts to LongTensor [H, W].

    Parameters
    ----------
    mask_input : Union[str, Path, np.ndarray]
        Path to mask or pre-loaded [H, W] numpy array.
    target_size : Optional[Tuple[int, int]]
        Target (Height, Width). If None, keeps original size.
    allowed_ids : Optional[List[int]]
        Valid class IDs.

    Returns
    -------
    torch.Tensor
        Model-ready integer mask tensor of shape [H, W] and dtype torch.long.
    """
    if isinstance(mask_input, (str, Path)):
        mask = load_mask(mask_input)
    else:
        mask = np.array(mask_input, dtype=np.uint8)

    if target_size is not None:
        mask = resize_mask(mask, target_size=target_size)

    is_valid, unique_vals = validate_mask_values(mask, allowed_ids=allowed_ids)
    if not is_valid:
        unexpected = [v for v in unique_vals if v not in (allowed_ids or RAW_CLASS_IDS)]
        raise ValueError(
            f"Preprocessed mask contains unexpected class IDs: {unexpected}. "
            f"Allowed IDs: {allowed_ids or RAW_CLASS_IDS}"
        )

    return mask_to_tensor(mask)
