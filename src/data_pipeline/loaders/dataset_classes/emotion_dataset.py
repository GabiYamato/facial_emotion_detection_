"""
Facial Emotion Dataset for PyTorch.

Loads images from  data/raw/{class_folder}/{train|test}_{id}_aligned.jpg
Fully configurable via YAML config. Supports:
  - Grayscale / RGB loading
  - Configurable image sizes
  - Predefined train/test split by filename prefix
  - Custom class-to-folder mapping
  - Subset creation for k-fold splits
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class EmotionDataset(Dataset):
    """
    PyTorch Dataset for facial emotion detection.

    Directory structure expected:
        raw_dir/
            1/            # folder for class 0
                train_00001_aligned.jpg
                test_00001_aligned.jpg
            2/            # folder for class 1
            ...
            7/            # folder for class 6

    Args:
        root_dir:  Path to the raw data directory (e.g., ``data/raw``).
        split:     ``'train'``, ``'test'``, or ``'all'``.
        transform: Optional torchvision transform pipeline.
        config:    Data configuration dictionary from YAML.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        config: Optional[Dict] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.transform = transform
        self.config = config or {}

        # ---------- configuration ----------
        self.color_mode = self.config.get("color_mode", "grayscale")
        self.image_size = tuple(self.config.get("image_size", [48, 48]))
        self.num_classes = self.config.get("num_classes", 7)
        self.class_names = self.config.get(
            "class_names",
            ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
        )

        # class index -> folder name mapping
        default_mapping = {str(i): str(i + 1) for i in range(self.num_classes)}
        self.class_to_folder = self.config.get("class_to_folder", default_mapping)
        self.folder_to_class: Dict[str, int] = {
            v: int(k) for k, v in self.class_to_folder.items()
        }

        # ---------- load samples ----------
        self.samples: List[Tuple[str, int]] = self._load_samples()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Scan directory tree and collect ``(image_path, label)`` pairs."""
        samples: List[Tuple[str, int]] = []

        for folder_name in sorted(os.listdir(self.root_dir)):
            folder_path = self.root_dir / folder_name
            if not folder_path.is_dir():
                continue
            if folder_name not in self.folder_to_class:
                continue

            class_idx = self.folder_to_class[folder_name]

            for img_file in sorted(os.listdir(folder_path)):
                img_path = folder_path / img_file

                # filter by extension
                if img_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    continue

                # filter by split prefix
                if self.split == "train" and not img_file.startswith("train_"):
                    continue
                elif self.split == "test" and not img_file.startswith("test_"):
                    continue
                # split == 'all' → keep everything

                samples.append((str(img_path), class_idx))

        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # load image
        mode = "L" if self.color_mode == "grayscale" else "RGB"
        image = Image.open(img_path).convert(mode)

        # resize to configured size
        image = image.resize(
            (self.image_size[1], self.image_size[0]), Image.BILINEAR
        )

        # apply torchvision transforms (if provided)
        if self.transform is not None:
            image = self.transform(image)
        else:
            # fallback: manual tensor conversion
            arr = np.array(image, dtype=np.float32) / 255.0
            if self.color_mode == "grayscale":
                arr = arr[np.newaxis, ...]  # (1, H, W)
            else:
                arr = arr.transpose(2, 0, 1)  # (C, H, W)
            image = torch.from_numpy(arr)

        return image, label

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def get_labels(self) -> np.ndarray:
        """Return all labels as a numpy array."""
        return np.array([s[1] for s in self.samples])

    def get_class_distribution(self) -> Dict[int, int]:
        """Return ``{class_idx: count}`` dictionary."""
        from collections import Counter
        return dict(Counter(self.get_labels().tolist()))

    def get_sample_paths(self) -> List[str]:
        """Return list of all image file paths."""
        return [s[0] for s in self.samples]

    def subset(self, indices) -> "EmotionDataset":
        """Create a lightweight subset with the given indices."""
        new_ds = EmotionDataset.__new__(EmotionDataset)
        # copy attributes
        for attr in (
            "root_dir", "split", "transform", "config",
            "color_mode", "image_size", "num_classes", "class_names",
            "class_to_folder", "folder_to_class",
        ):
            setattr(new_ds, attr, getattr(self, attr))
        new_ds.samples = [self.samples[i] for i in indices]
        return new_ds

    def set_transform(self, transform: Optional[Callable]):
        """Swap the transform pipeline (e.g. train → val transforms)."""
        self.transform = transform

    def __repr__(self) -> str:
        dist = self.get_class_distribution()
        dist_str = ", ".join(
            f"{self.class_names[k]}={v}" for k, v in sorted(dist.items())
        )
        return (
            f"EmotionDataset(split={self.split}, n={len(self)}, "
            f"size={self.image_size}, mode={self.color_mode}, dist=[{dist_str}])"
        )
