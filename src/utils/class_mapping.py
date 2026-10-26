"""Centralized Class Mapping Configuration for LoveDA Dataset.

This module provides the definitive, single-source-of-truth class definitions,
ID mappings, human-readable labels, ignore index handling, and color palettes
for the LoveDA remote sensing dataset.
"""

from typing import Dict, List, Tuple
import numpy as np

# Raw class IDs present in LoveDA ground-truth masks
RAW_CLASS_IDS: List[int] = [0, 1, 2, 3, 4, 5, 6, 7]

# Designated ignore / no-data index in raw masks
IGNORE_INDEX: int = 0

# Human-readable class names for all raw IDs
CLASS_NAMES: Dict[int, str] = {
    0: "No-data / Ignore",
    1: "Background",
    2: "Building",
    3: "Road",
    4: "Water",
    5: "Barren",
    6: "Forest",
    7: "Agriculture",
}

# Evaluated semantic classes (excluding no-data / ignore)
EVAL_CLASS_IDS: List[int] = [1, 2, 3, 4, 5, 6, 7]
EVAL_CLASS_NAMES: Dict[int, str] = {k: CLASS_NAMES[k] for k in EVAL_CLASS_IDS}

# Total counts
NUM_CLASSES: int = len(RAW_CLASS_IDS)       # 8 (0-7)
NUM_EVAL_CLASSES: int = len(EVAL_CLASS_IDS) # 7 semantic categories

# Official LoveDA RGB Color Palette for visualization
# Format: (R, G, B) in range [0, 255]
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),          # No-data / Ignore -> Black
    1: (200, 200, 200),    # Background -> Light Grey
    2: (255, 0, 0),        # Building -> Red
    3: (255, 255, 0),      # Road -> Yellow
    4: (0, 0, 255),        # Water -> Blue
    5: (159, 129, 183),    # Barren -> Purple
    6: (0, 255, 0),        # Forest -> Green
    7: (255, 195, 128),    # Agriculture -> Orange / Tan
}

# Lookup array for vectorized conversion from mask [H, W] to RGB [H, W, 3]
COLOR_PALETTE: np.ndarray = np.array([CLASS_COLORS[i] for i in range(NUM_CLASSES)], dtype=np.uint8)


def get_class_name(class_id: int) -> str:
    """Return human-readable class name for a given class ID."""
    return CLASS_NAMES.get(class_id, f"Unknown ({class_id})")


def is_ignore(class_id: int) -> bool:
    """Check if class ID is designated as ignore / no-data."""
    return class_id == IGNORE_INDEX


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert a 2D integer class mask [H, W] to a 3D RGB colorized array [H, W, 3].

    Parameters
    ----------
    mask : np.ndarray
        2D integer array containing class IDs (0-7).

    Returns
    -------
    np.ndarray
        3D uint8 array with RGB color representation.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask [H, W], got shape {mask.shape}")
    
    # Clip to valid range to avoid indexing errors if invalid IDs exist
    safe_mask = np.clip(mask, 0, NUM_CLASSES - 1).astype(np.int64)
    return COLOR_PALETTE[safe_mask]
