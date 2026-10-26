"""Image preprocessing pipeline for LoveDA remote sensing images.

Implements reading RGB images, resizing, tensor conversion, and normalization
following standard deep learning conventions without modifying source files.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch

from src.preprocessing.preprocessing_config import (
    IMAGE_MEAN,
    IMAGE_STD,
    TARGET_IMAGE_SIZE,
)


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Read an RGB remote sensing image from disk.

    Parameters
    ----------
    image_path : Union[str, Path]
        Path to image file.

    Returns
    -------
    np.ndarray
        RGB image array of shape [H, W, 3] and dtype uint8.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found at {path.resolve()}")

    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb, dtype=np.uint8)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image, got shape {arr.shape} for {path}")

    return arr


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize RGB image to target dimensions using bilinear interpolation.

    Parameters
    ----------
    image : np.ndarray
        Source image [H, W, 3].
    target_size : Tuple[int, int]
        (Target Height, Target Width).
    interpolation : int
        OpenCV interpolation flag (default: cv2.INTER_LINEAR).

    Returns
    -------
    np.ndarray
        Resized image of shape [target_size[0], target_size[1], 3].
    """
    target_h, target_w = target_size
    if image.shape[0] == target_h and image.shape[1] == target_w:
        return image.copy()

    # cv2.resize expects (width, height)
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = IMAGE_MEAN,
    std: Tuple[float, float, float] = IMAGE_STD,
) -> np.ndarray:
    """Normalize image array to float32 using specified mean and standard deviation.

    Parameters
    ----------
    image : np.ndarray
        Image array with values in [0, 255] or [0.0, 1.0].
    mean : Tuple[float, float, float]
        Per-channel mean for RGB channels.
    std : Tuple[float, float, float]
        Per-channel std for RGB channels.

    Returns
    -------
    np.ndarray
        Normalized float32 image array of shape [H, W, 3].
    """
    img_float = image.astype(np.float32)
    if img_float.max() > 1.0:
        img_float = img_float / 255.0

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    return (img_float - mean_arr) / std_arr


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an image array [H, W, 3] to a PyTorch tensor [3, H, W].

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape [H, W, 3].

    Returns
    -------
    torch.Tensor
        PyTorch float32 tensor of shape [3, H, W].
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image with shape [H, W, 3], got {image.shape}")

    # Transpose [H, W, C] -> [C, H, W]
    tensor = torch.from_numpy(image.transpose(2, 0, 1))
    return tensor.float()


def preprocess_image(
    image_input: Union[str, Path, np.ndarray],
    target_size: Optional[Tuple[int, int]] = TARGET_IMAGE_SIZE,
    normalize: bool = True,
    mean: Tuple[float, float, float] = IMAGE_MEAN,
    std: Tuple[float, float, float] = IMAGE_STD,
) -> torch.Tensor:
    """Complete image preprocessing pipeline.

    Reads -> Resizes -> Normalizes -> Converts to Tensor [3, H, W].

    Parameters
    ----------
    image_input : Union[str, Path, np.ndarray]
        Path to image or pre-loaded [H, W, 3] numpy array.
    target_size : Optional[Tuple[int, int]]
        Target (Height, Width). If None, keeps original size.
    normalize : bool
        Whether to apply mean/std normalization.
    mean : Tuple[float, float, float]
        Normalization mean.
    std : Tuple[float, float, float]
        Normalization std.

    Returns
    -------
    torch.Tensor
        Model-ready tensor of shape [3, H, W] and dtype float32.
    """
    if isinstance(image_input, (str, Path)):
        img = load_image(image_input)
    else:
        img = np.array(image_input, dtype=np.uint8)

    if target_size is not None:
        img = resize_image(img, target_size=target_size)

    if normalize:
        img_norm = normalize_image(img, mean=mean, std=std)
    else:
        img_norm = (img.astype(np.float32) / 255.0) if img.max() > 1.0 else img.astype(np.float32)

    return image_to_tensor(img_norm)
