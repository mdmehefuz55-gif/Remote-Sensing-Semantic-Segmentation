"""Training package for LoveDA semantic segmentation models."""

from src.training.trainer import SemanticSegmentationTrainer
from src.training.training_config import (
    CHECKPOINT_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP_NORM,
    LOSS_CONFIG,
    OUTPUTS_DIR,
    REPORTS_DIR,
    SCHEDULER_CONFIG,
    get_safe_device,
)

__all__ = [
    "SemanticSegmentationTrainer",
    "get_safe_device",
    "CHECKPOINT_DIR",
    "REPORTS_DIR",
    "OUTPUTS_DIR",
    "DEFAULT_LR",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_NUM_EPOCHS",
    "EARLY_STOPPING_PATIENCE",
    "GRADIENT_CLIP_NORM",
    "LOSS_CONFIG",
    "SCHEDULER_CONFIG",
]
