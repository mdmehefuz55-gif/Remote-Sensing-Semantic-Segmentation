"""DataLoader creation and configuration utilities for LoveDA semantic segmentation.

Provides configurable, robust DataLoaders for Train, Validation, and Test splits
with Windows-safe multiprocessing defaults.
"""

from pathlib import Path
from typing import Literal, Optional, Tuple, Union, overload
import torch
from torch.utils.data import DataLoader

from src.dataset.loveDA_dataset import LoveDADataset
from src.preprocessing.preprocessing_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PIN_MEMORY,
    TARGET_IMAGE_SIZE,
    get_dataset_root,
)


def create_dataloader(
    dataset: LoveDADataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
) -> DataLoader:
    """Create a configured PyTorch DataLoader for a given dataset.

    Parameters
    ----------
    dataset : LoveDADataset
        The dataset instance to load from.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle data each epoch.
    num_workers : int
        Number of subprocesses for data loading (0 on Windows avoids process spawn issues).
    pin_memory : Optional[bool]
        Whether to copy Tensors into CUDA pinned memory before returning them.
        If None, automatically enabled when CUDA is available.
    drop_last : bool
        Whether to drop the last incomplete batch.

    Returns
    -------
    DataLoader
        PyTorch DataLoader instance.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available() and DEFAULT_PIN_MEMORY

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


@overload
def get_dataloaders(
    root_dir: Optional[Union[str, Path]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    pin_memory: Optional[bool] = None,
    include_test: Literal[False] = False,
) -> Tuple[DataLoader, DataLoader]:
    ...


@overload
def get_dataloaders(
    root_dir: Optional[Union[str, Path]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    pin_memory: Optional[bool] = None,
    include_test: Literal[True] = ...,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ...


@overload
def get_dataloaders(
    root_dir: Optional[Union[str, Path]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    pin_memory: Optional[bool] = None,
    include_test: bool = False,
) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
    ...


def get_dataloaders(
    root_dir: Optional[Union[str, Path]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
    pin_memory: Optional[bool] = None,
    include_test: bool = False,
) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
    """Build and return train, validation (and optionally test) DataLoaders.

    Parameters
    ----------
    root_dir : Optional[Union[str, Path]]
        Dataset root directory. If None, resolved automatically.
    batch_size : int
        Batch size for all loaders.
    num_workers : int
        Number of workers for data loading.
    target_size : Tuple[int, int]
        Target image and mask dimensions (Height, Width).
    pin_memory : Optional[bool]
        Whether to pin memory in CUDA.
    include_test : bool
        Whether to also return test_loader.

    Returns
    -------
    Tuple[DataLoader, DataLoader] or Tuple[DataLoader, DataLoader, DataLoader]
        (train_loader, val_loader) or (train_loader, val_loader, test_loader).
    """
    resolved_root = get_dataset_root(root_dir)

    train_dataset = LoveDADataset(
        root_dir=resolved_root,
        split="Train",
        target_size=target_size,
    )

    val_dataset = LoveDADataset(
        root_dir=resolved_root,
        split="Val",
        target_size=target_size,
    )

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    if include_test:
        test_dataset = LoveDADataset(
            root_dir=resolved_root,
            split="Test",
            target_size=target_size,
        )
        test_loader = create_dataloader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        return train_loader, val_loader, test_loader

    return train_loader, val_loader
