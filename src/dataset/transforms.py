"""Transform factories for LoveDA dataset.

Provides clean access to training, validation, and test transform pipelines
while maintaining separation of concerns.
"""

from typing import Any, Dict, Optional, Tuple

from src.preprocessing.augmentation import (
    SegmentationTransformWrapper,
    get_test_augmentation,
    get_training_augmentation,
    get_validation_augmentation,
)
from src.preprocessing.preprocessing_config import (
    IMAGE_MEAN,
    IMAGE_STD,
    TARGET_IMAGE_SIZE,
)


def get_train_transforms(
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    mean: Tuple[float, float, float] = IMAGE_MEAN,
    std: Tuple[float, float, float] = IMAGE_STD,
    config: Optional[Dict[str, Any]] = None,
) -> SegmentationTransformWrapper:
    """Build and return training data transforms."""
    return get_training_augmentation(target_size=target_size, mean=mean, std=std, config=config)


def get_val_transforms(
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    mean: Tuple[float, float, float] = IMAGE_MEAN,
    std: Tuple[float, float, float] = IMAGE_STD,
) -> SegmentationTransformWrapper:
    """Build and return validation data transforms."""
    return get_validation_augmentation(target_size=target_size, mean=mean, std=std)


def get_test_transforms(
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    mean: Tuple[float, float, float] = IMAGE_MEAN,
    std: Tuple[float, float, float] = IMAGE_STD,
) -> SegmentationTransformWrapper:
    """Build and return test data transforms."""
    return get_test_augmentation(target_size=target_size, mean=mean, std=std)
