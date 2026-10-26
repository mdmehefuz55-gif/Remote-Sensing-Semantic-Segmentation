"""Custom PyTorch Dataset implementation for the LoveDA remote sensing dataset.

Provides robust, stem-based image-mask pairing for supervised splits (Train, Val),
safe unlabelled handling for Test split, and composable transforms.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.preprocessing.augmentation import (
    get_test_augmentation,
    get_training_augmentation,
    get_validation_augmentation,
)
from src.preprocessing.preprocessing_config import (
    TARGET_IMAGE_SIZE,
    get_dataset_root,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class LoveDADataset(Dataset):
    """PyTorch Dataset for LoveDA Semantic Segmentation.

    Attributes
    ----------
    samples : List[Tuple[Path, Optional[Path], str]]
        List of (image_path, mask_path, sample_stem) tuples.
    split : str
        Dataset split: 'Train', 'Val', or 'Test'.
    transform : Optional[Callable]
        Transform pipeline applied to (image, mask).
    """

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        split: str = "Train",
        image_dir: Optional[Union[str, Path]] = None,
        mask_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
        strict_pairing: bool = True,
    ):
        """Initialize LoveDADataset.

        Parameters
        ----------
        root_dir : Optional[Union[str, Path]]
            Path to LoveDA root folder. If None and image_dir is not provided,
            searches standard candidate paths.
        split : str
            One of 'Train', 'Val', or 'Test' (case-insensitive).
        image_dir : Optional[Union[str, Path]]
            Direct path to image folder (overrides root_dir / split).
        mask_dir : Optional[Union[str, Path]]
            Direct path to mask folder (overrides root_dir / split).
        transform : Optional[Callable]
            Transform pipeline. If None, sensible default transforms are applied.
        target_size : Tuple[int, int]
            Target image dimensions (Height, Width) used if default transforms are created.
        strict_pairing : bool
            If True, raises error if any supervised image is missing its mask.
        """
        super().__init__()
        self.split = split.capitalize()
        self.target_size = target_size
        self.strict_pairing = strict_pairing

        # Resolve transform
        if transform is not None:
            self.transform = transform
        else:
            if self.split == "Train":
                self.transform = get_training_augmentation(target_size=target_size)
            elif self.split == "Val":
                self.transform = get_validation_augmentation(target_size=target_size)
            else:
                self.transform = get_test_augmentation(target_size=target_size)

        # Resolve and pair files
        self.samples = self._discover_and_pair(root_dir, image_dir, mask_dir)

    def _discover_files_under_dir(self, directory: Path, kind: str) -> List[Path]:
        """Discover image or mask files under a directory (supporting nested or flat structures)."""
        if not directory.exists():
            return []

        files = []
        for p in directory.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                path_str = str(p).lower()
                if kind == "mask":
                    if "masks_png" in path_str or "mask" in path_str:
                        files.append(p)
                elif kind == "image":
                    if "images_png" in path_str or "image" in path_str:
                        files.append(p)
                    elif "mask" not in path_str:
                        files.append(p)
        return sorted(files)

    def _discover_and_pair(
        self,
        root_dir: Optional[Union[str, Path]],
        image_dir: Optional[Union[str, Path]],
        mask_dir: Optional[Union[str, Path]],
    ) -> List[Tuple[Path, Optional[Path], str]]:
        """Pair images and masks by their filename stems."""
        # Case A: Explicit image_dir and mask_dir provided
        if image_dir is not None:
            img_path = Path(image_dir)
            if not img_path.exists():
                raise FileNotFoundError(f"Specified image_dir does not exist: {img_path}")
            images = [p for p in img_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            masks = []
            if mask_dir is not None:
                msk_path = Path(mask_dir)
                if msk_path.exists():
                    masks = [p for p in msk_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        else:
            # Case B: Resolve root_dir
            resolved_root = get_dataset_root(root_dir)
            split_dir = resolved_root / self.split
            if not split_dir.exists():
                # Check lowercase or capitalized
                alt_split = resolved_root / self.split.lower()
                if alt_split.exists():
                    split_dir = alt_split
                else:
                    raise FileNotFoundError(f"Split directory '{self.split}' not found under {resolved_root}")

            images = self._discover_files_under_dir(split_dir, kind="image")
            masks = self._discover_files_under_dir(split_dir, kind="mask") if self.split != "Test" else []

        if not images:
            raise FileNotFoundError(f"No image files found for split '{self.split}'")

        # Map by lowercase filename stem
        image_map: Dict[str, Path] = {p.stem.lower(): p for p in images}
        mask_map: Dict[str, Path] = {p.stem.lower(): p for p in masks}

        # Sorting key: integer if numeric stem, else alphabetical
        def sort_key(s: str):
            return int(s) if s.isdigit() else s

        samples: List[Tuple[Path, Optional[Path], str]] = []

        if self.split == "Test":
            # Test split has no ground-truth masks
            for stem in sorted(image_map.keys(), key=sort_key):
                samples.append((image_map[stem], None, stem))
        else:
            matched_stems = set(image_map.keys()) & set(mask_map.keys())
            images_without_masks = set(image_map.keys()) - set(mask_map.keys())
            masks_without_images = set(mask_map.keys()) - set(image_map.keys())

            if images_without_masks or masks_without_images:
                msg = (
                    f"[{self.split} Split] Mismatched image/mask pairs detected! "
                    f"Images without masks: {len(images_without_masks)}, "
                    f"Masks without images: {len(masks_without_images)}."
                )
                if self.strict_pairing:
                    raise ValueError(msg)
                else:
                    print(f"Warning: {msg}")

            for stem in sorted(matched_stems, key=sort_key):
                samples.append((image_map[stem], mask_map[stem], stem))

        return samples

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def get_raw_sample(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        """Return raw un-transformed numpy arrays for sample at idx."""
        img_path, mask_path, stem = self.samples[idx]

        with Image.open(img_path) as img:
            image_arr = np.array(img.convert("RGB"), dtype=np.uint8)

        mask_arr = None
        if mask_path is not None:
            with Image.open(mask_path) as msk:
                mask_arr = np.array(msk, dtype=np.uint8)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[..., 0]

        return image_arr, mask_arr, stem

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, str]]:
        """Retrieve and transform sample at index `idx`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor] (for Train and Val)
            image: FloatTensor of shape [3, H, W]
            mask: LongTensor of shape [H, W]
        Tuple[torch.Tensor, str] (for Test)
            image: FloatTensor of shape [3, H, W]
            stem: str (filename identifier)
        """
        img_path, mask_path, stem = self.samples[idx]

        # Read RGB Image
        with Image.open(img_path) as img:
            image_arr = np.array(img.convert("RGB"), dtype=np.uint8)

        # Read Mask if available
        if mask_path is not None:
            with Image.open(mask_path) as msk:
                mask_arr = np.array(msk, dtype=np.uint8)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[..., 0]

            if self.transform is not None:
                transformed = self.transform(image=image_arr, mask=mask_arr)
                image_tensor = transformed["image"]
                mask_tensor = transformed["mask"]
            else:
                image_tensor = torch.from_numpy(image_arr.transpose(2, 0, 1)).float() / 255.0
                mask_tensor = torch.from_numpy(mask_arr).long()

            # Ensure proper dtypes and shapes
            if not isinstance(image_tensor, torch.Tensor):
                image_tensor = torch.tensor(image_tensor, dtype=torch.float32)
            if not isinstance(mask_tensor, torch.Tensor):
                mask_tensor = torch.tensor(mask_tensor, dtype=torch.int64)

            return image_tensor.float(), mask_tensor.long()

        else:
            # Test sample without mask
            if self.transform is not None:
                transformed = self.transform(image=image_arr)
                image_tensor = transformed["image"]
            else:
                image_tensor = torch.from_numpy(image_arr.transpose(2, 0, 1)).float() / 255.0

            if not isinstance(image_tensor, torch.Tensor):
                image_tensor = torch.tensor(image_tensor, dtype=torch.float32)

            return image_tensor.float(), stem
