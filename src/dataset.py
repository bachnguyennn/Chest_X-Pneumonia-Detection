"""
Custom Dataset and transforms for Chest X-Ray Pneumonia classification.

Uses the official Kaggle train/val/test splits.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# ImageNet normalization — standard for transfer learning on medical X-rays
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp"}


def get_transforms(
    split: str = "train",
    image_size: int = 224,
) -> transforms.Compose:
    """Return augmentation pipeline for train/val/test."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


class ChestXRayDataset(Dataset):
    """Load chest X-rays from Kaggle folder structure."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples: list[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                "Download the Kaggle dataset to data/raw/chest_xray/"
            )

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            label = CLASS_TO_IDX[class_name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(f"No images found in {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, str(img_path)

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.samples]


def compute_class_weights(labels: list[int], num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency weights for weighted cross-entropy."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_sampler(dataset: ChestXRayDataset) -> WeightedRandomSampler:
    """Oversample minority class during training."""
    labels = dataset.labels
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    sample_weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders(
    data_root: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
) -> dict[str, DataLoader]:
    """Create train, val, and test DataLoaders."""
    data_root = Path(data_root)
    loaders = {}

    for split in ("train", "val", "test"):
        transform = get_transforms(split, image_size)
        dataset = ChestXRayDataset(data_root, split=split, transform=transform)

        shuffle = split == "train" and not use_weighted_sampler
        sampler = None
        if split == "train" and use_weighted_sampler:
            sampler = get_weighted_sampler(dataset)
            shuffle = False

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor.cpu() * std + mean


def get_dataset_stats(data_root: str | Path) -> dict:
    """Return class counts per split for reporting."""
    data_root = Path(data_root)
    stats = {}
    for split in ("train", "val", "test"):
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        stats[split] = {}
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if class_dir.exists():
                count = sum(
                    1
                    for f in class_dir.iterdir()
                    if f.suffix.lower() in IMAGE_EXTENSIONS
                )
                stats[split][class_name] = count
    return stats
