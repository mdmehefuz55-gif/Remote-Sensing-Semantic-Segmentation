"""Configuration for LoveDA image and mask preprocessing and augmentation.

All preprocessing and augmentation parameters are centralized here rather than
hardcoded in transforms or dataset code.
"""

from pathlib import Path
from typing import List, Tuple

# Candidate dataset root directories (checked in order)
CANDIDATE_DATASET_ROOTS: List[Path] = [
    Path("data/LoveDA"),
    Path("LoveDA"),
]


def get_dataset_root(custom_root: str = None) -> Path:
    """Resolve and return an existing dataset root directory.

    Parameters
    ----------
    custom_root : str, optional
        Custom directory path to check first.

    Returns
    -------
    Path
        Validated Path object to the existing LoveDA directory.
    """
    candidates = []
    if custom_root:
        candidates.append(Path(custom_root))
    candidates.extend(CANDIDATE_DATASET_ROOTS)

    for p in candidates:
        if p.exists() and (p / "Train").exists():
            return p.resolve()

    raise FileNotFoundError(
        f"LoveDA dataset not found in any of the candidate paths: {[str(c) for c in candidates]}. "
        "Please ensure the dataset is placed under 'LoveDA' or 'data/LoveDA'."
    )


# Target image dimensions for model training & evaluation
# LoveDA raw size is 1024x1024. 512x512 is standard for memory-efficient GPU training.
TARGET_IMAGE_SIZE: Tuple[int, int] = (512, 512)  # (Height, Width)
RAW_IMAGE_SIZE: Tuple[int, int] = (1024, 1024)

# Normalization constants (Standard ImageNet RGB statistics)
IMAGE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Augmentation hyperparameters
AUGMENTATION_CONFIG = {
    "horizontal_flip_p": 0.5,
    "vertical_flip_p": 0.5,
    "random_rotate90_p": 0.5,
    "shift_scale_rotate_p": 0.3,
    "shift_limit": 0.05,
    "scale_limit": 0.1,
    "rotate_limit": 15,
    "color_jitter_p": 0.2,
    "brightness_limit": 0.15,
    "contrast_limit": 0.15,
}

# DataLoader default parameters
DEFAULT_BATCH_SIZE: int = 4
DEFAULT_NUM_WORKERS: int = 0  # 0 is safe default on Windows to avoid process spawn overhead/deadlocks
DEFAULT_PIN_MEMORY: bool = True
RANDOM_SEED: int = 42
