"""Centralized Training Configuration for LoveDA Semantic Segmentation."""

from pathlib import Path
from typing import Dict, Any
import torch

# Base directories
CHECKPOINT_DIR: Path = Path("checkpoints")
REPORTS_DIR: Path = Path("reports")
OUTPUTS_DIR: Path = Path("outputs/training")

# Default Hyperparameters
DEFAULT_LR: float = 1e-4
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_BATCH_SIZE: int = 4
DEFAULT_NUM_EPOCHS: int = 5
EARLY_STOPPING_PATIENCE: int = 5
GRADIENT_CLIP_NORM: float = 1.0

# Loss Configuration
LOSS_CONFIG: Dict[str, Any] = {
    "type": "combined",
    "ce_weight": 1.0,
    "dice_weight": 1.0,
    "ignore_index": 0,
}

# Learning Rate Scheduler Configuration
SCHEDULER_CONFIG: Dict[str, Any] = {
    "type": "cosine",       # 'cosine' or 'plateau'
    "t_max": DEFAULT_NUM_EPOCHS,
    "eta_min": 1e-6,
}


def get_safe_device() -> torch.device:
    """Resolve and verify computation device with kernel sanity test.

    Returns CUDA if available and able to execute convolution kernels;
    otherwise falls back gracefully to CPU.
    """
    if torch.cuda.is_available():
        try:
            test_conv = torch.nn.Conv2d(1, 1, 1).cuda()
            test_x = torch.zeros(1, 1, 1, 1).cuda()
            test_conv(test_x)
            return torch.device("cuda")
        except Exception:
            # Benign fallback if current PyTorch installation lacks cubin for newer GPUs (e.g. sm_120)
            return torch.device("cpu")
    return torch.device("cpu")
