"""
DataLoader factory for the Facial Emotion Detection project.

Provides helper functions to build:
  - Train / validation / test ``DataLoader`` objects
  - K-fold cross-validation splits (stratified)
  - Class-balanced sampling via ``WeightedRandomSampler``
  - Configurable torchvision transform pipelines
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Dict, List, Optional, Tuple

from src.data_pipeline.loaders.dataset_classes.emotion_dataset import EmotionDataset


# =========================================================================
# Transform builders
# =========================================================================

def build_transforms(config: Dict, split: str = "train") -> transforms.Compose:
    """
    Build a ``torchvision.transforms.Compose`` pipeline from the YAML config.

    Args:
        config: Full project config dictionary.
        split:  ``'train'`` or ``'test'`` / ``'val'``.

    Returns:
        A Compose transform pipeline.
    """
    aug_cfg = config.get("augmentation", {})
    data_cfg = config.get("data", {})
    color_mode = data_cfg.get("color_mode", "grayscale")
    is_train = split == "train" and aug_cfg.get("enabled", False)
    train_aug = aug_cfg.get("train", {})
    norm_cfg = aug_cfg.get("normalize", {})

    tf_list: list = []

    # --- Training augmentations (only for train split) ---
    if is_train:
        if train_aug.get("random_horizontal_flip", 0) > 0:
            tf_list.append(transforms.RandomHorizontalFlip(p=train_aug["random_horizontal_flip"]))

        if train_aug.get("random_rotation", 0) > 0:
            tf_list.append(transforms.RandomRotation(degrees=train_aug["random_rotation"]))

        affine = train_aug.get("random_affine", {})
        if affine.get("enabled", False):
            tf_list.append(transforms.RandomAffine(
                degrees=affine.get("degrees", 0),
                translate=tuple(affine.get("translate", [0, 0])),
                scale=tuple(affine.get("scale", [1, 1])),
            ))

        cj = train_aug.get("color_jitter", {})
        if cj.get("enabled", False):
            tf_list.append(transforms.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast", 0),
                saturation=cj.get("saturation", 0),
                hue=cj.get("hue", 0),
            ))

        gb = train_aug.get("gaussian_blur", {})
        if gb.get("enabled", False):
            tf_list.append(transforms.GaussianBlur(kernel_size=gb.get("kernel_size", 3)))

    # --- To tensor ---
    tf_list.append(transforms.ToTensor())

    # --- Random erasing (operates on tensors) ---
    if is_train:
        re = train_aug.get("random_erasing", {})
        if re.get("enabled", False):
            tf_list.append(transforms.RandomErasing(
                p=re.get("p", 0.2),
                scale=tuple(re.get("scale", [0.02, 0.15])),
            ))

    # --- Normalize ---
    mean = norm_cfg.get("mean", [0.5])
    std = norm_cfg.get("std", [0.5])
    tf_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(tf_list)


# =========================================================================
# Standard DataLoader creation
# =========================================================================

def create_dataloaders(
    config: Dict,
    verbose: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test ``DataLoader`` objects.

    If ``use_predefined_split`` is True (default), train/test are split by
    filename prefix. A fraction ``val_split`` of the training data is held
    out for validation.

    Returns:
        ``(train_loader, val_loader, test_loader)``
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    batch_size = train_cfg.get("batch_size", 64)
    num_workers = config["project"].get("num_workers", 4)
    pin_memory = config["project"].get("pin_memory", True)

    train_transform = build_transforms(config, split="train")
    eval_transform = build_transforms(config, split="test")

    # --- build datasets ---
    if data_cfg.get("use_predefined_split", True):
        train_ds = EmotionDataset(
            root_dir=data_cfg["raw_dir"], split="train",
            transform=train_transform, config=data_cfg,
        )
        test_ds = EmotionDataset(
            root_dir=data_cfg["raw_dir"], split="test",
            transform=eval_transform, config=data_cfg,
        )
        val_frac = data_cfg.get("val_split", 0.15)
        labels = train_ds.get_labels()
        indices = np.arange(len(train_ds))
        train_idx, val_idx = train_test_split(
            indices, test_size=val_frac, stratify=labels,
            random_state=config["project"].get("seed", 42),
        )
        val_ds = train_ds.subset(val_idx)
        train_ds_split = train_ds.subset(train_idx)
    else:
        full_ds = EmotionDataset(
            root_dir=data_cfg["raw_dir"], split="all",
            transform=train_transform, config=data_cfg,
        )
        labels = full_ds.get_labels()
        indices = np.arange(len(full_ds))
        train_val_idx, test_idx = train_test_split(
            indices, test_size=data_cfg.get("test_split", 0.15),
            stratify=labels, random_state=config["project"].get("seed", 42),
        )
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=data_cfg.get("val_split", 0.15),
            stratify=labels[train_val_idx],
            random_state=config["project"].get("seed", 42),
        )
        train_ds_split = full_ds.subset(train_idx)
        val_ds = full_ds.subset(val_idx)
        test_ds = full_ds.subset(test_idx)
    val_ds.set_transform(eval_transform)
    test_ds.set_transform(eval_transform)

    if verbose >= 1:
        print(f"  [Data] Train: {len(train_ds_split)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # --- class balancing ---
    bal_cfg = train_cfg.get("class_balancing", {})
    sampler = None
    shuffle_train = True

    if bal_cfg.get("enabled", False) and bal_cfg.get("method") == "oversampling":
        sampler = _build_weighted_sampler(train_ds_split)
        shuffle_train = False  # sampler handles shuffling

    # --- build loaders ---
    train_loader = DataLoader(
        train_ds_split,
        batch_size=batch_size,
        shuffle=shuffle_train if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


# =========================================================================
# K-Fold DataLoader creation
# =========================================================================

def create_kfold_splits(
    config: Dict,
    n_splits: int = 5,
    shuffle: bool = True,
    stratified: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return a list of ``(train_indices, val_indices)`` for k-fold CV.

    Operates on the *combined* (train+test) data unless
    ``use_predefined_split`` forces train-only.
    """
    data_cfg = config["data"]
    seed = config["project"].get("seed", 42)

    full_ds = EmotionDataset(
        root_dir=data_cfg["raw_dir"],
        split="all",
        config=data_cfg,
    )
    labels = full_ds.get_labels()

    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    splits = list(kf.split(np.zeros(len(labels)), labels))
    return splits


def create_kfold_dataloaders(
    config: Dict,
    fold_train_idx: np.ndarray,
    fold_val_idx: np.ndarray,
    verbose: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders for one fold of k-fold CV.

    Args:
        config:         Project config.
        fold_train_idx: Indices for training portion.
        fold_val_idx:   Indices for validation portion.

    Returns:
        ``(train_loader, val_loader)``
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    batch_size = train_cfg.get("batch_size", 64)
    num_workers = config["project"].get("num_workers", 4)
    pin_memory = config["project"].get("pin_memory", True)

    train_transform = build_transforms(config, split="train")
    eval_transform = build_transforms(config, split="test")

    full_ds = EmotionDataset(
        root_dir=data_cfg["raw_dir"],
        split="all",
        config=data_cfg,
    )

    train_ds = full_ds.subset(fold_train_idx)
    train_ds.set_transform(train_transform)
    val_ds = full_ds.subset(fold_val_idx)
    val_ds.set_transform(eval_transform)

    if verbose >= 1:
        print(f"  [Fold] Train: {len(train_ds)} | Val: {len(val_ds)}")

    bal_cfg = train_cfg.get("class_balancing", {})
    sampler = None
    shuffle_train = True
    if bal_cfg.get("enabled", False) and bal_cfg.get("method") == "oversampling":
        sampler = _build_weighted_sampler(train_ds)
        shuffle_train = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


# =========================================================================
# Class-balanced sampler
# =========================================================================

def _build_weighted_sampler(dataset: EmotionDataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples minority classes."""
    labels = dataset.get_labels()
    class_counts = np.bincount(labels, minlength=dataset.num_classes)
    class_weights = 1.0 / (class_counts.astype(np.float64) + 1e-9)
    sample_weights = class_weights[labels]
    sample_weights = torch.from_numpy(sample_weights).double()
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_class_weights(config: Dict) -> Optional[torch.Tensor]:
    """
    Compute inverse-frequency class weights for the loss function.

    Returns ``None`` if class balancing is disabled or method is not
    ``class_weights``.
    """
    train_cfg = config["training"]
    bal_cfg = train_cfg.get("class_balancing", {})

    if not bal_cfg.get("enabled", False):
        return None
    if bal_cfg.get("method") != "class_weights":
        return None

    data_cfg = config["data"]
    ds = EmotionDataset(root_dir=data_cfg["raw_dir"], split="train", config=data_cfg)
    labels = ds.get_labels()
    class_counts = np.bincount(labels, minlength=data_cfg.get("num_classes", 7)).astype(np.float64)
    total = class_counts.sum()
    weights = total / (len(class_counts) * class_counts + 1e-9)
    return torch.tensor(weights, dtype=torch.float32)
