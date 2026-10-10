"""
data/dataset.py
─────────────────────────────────────────────────────────────────────────────
CIFAR-10 with controllable class imbalance.

The majority class (class 0) retains all samples.
Every other class is sub-sampled so that:
    n_minority = floor(n_majority / imbalance_ratio)

Modes
-----
• "baseline"    — standard cross-entropy, no correction
• "weighted"    — weighted cross-entropy (inverse class frequency)
• "focal"       — focal loss (no special dataset changes)
• "oversampling"— minority classes are oversampled back to majority size

Usage
-----
    from data import get_dataloaders
    train_loader, val_loader, class_weights = get_dataloaders(cfg, seed=42)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.datasets import CIFAR10


# ── Augmentation pipelines ────────────────────────────────────────────────────

_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2023, 0.1994, 0.2010)


def _train_transform() -> T.Compose:
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])


def _val_transform() -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])


# ── Imbalanced dataset wrapper ────────────────────────────────────────────────

class ImbalancedCIFAR10(Dataset):
    """
    CIFAR-10 subset with long-tail class imbalance.

    Parameters
    ----------
    root            : str   — path to CIFAR-10 data directory
    train           : bool  — training split if True, else test split
    imbalance_ratio : int   — majority_count / minority_count  (1 = balanced)
    transform       : callable or None
    download        : bool
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        imbalance_ratio: int = 1,
        transform=None,
        download: bool = True,
    ) -> None:
        base = CIFAR10(root=root, train=train, transform=None, download=download)

        self.transform = transform
        self.data: List[np.ndarray] = []
        self.targets: List[int]     = []

        # ── Build per-class buckets ──────────────────────────────────────────
        num_classes = 10
        class_data: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
        for idx, label in enumerate(base.targets):
            class_data[label].append(idx)

        # Class 0 is the majority class
        n_majority = len(class_data[0])
        n_minority = max(1, math.floor(n_majority / imbalance_ratio))

        rng = np.random.default_rng(seed=0)   # fixed for reproducible splits

        for c in range(num_classes):
            indices = class_data[c]
            if c > 0 and imbalance_ratio > 1:
                indices = rng.choice(indices, size=n_minority, replace=False).tolist()
            for i in indices:
                self.data.append(base.data[i])
                self.targets.append(base.targets[i])

        self.data    = np.stack(self.data)
        self.targets = np.array(self.targets, dtype=np.int64)

        # ── Class statistics ─────────────────────────────────────────────────
        self.class_counts = np.bincount(self.targets, minlength=num_classes)
        total             = self.class_counts.sum()
        freq              = self.class_counts / total
        self.class_weights = torch.tensor(
            1.0 / (freq + 1e-8), dtype=torch.float32
        )
        self.class_weights /= self.class_weights.sum()   # normalise

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img  = self.data[index]
        label = int(self.targets[index])

        from PIL import Image
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Factory function ──────────────────────────────────────────────────────────

def get_dataloaders(
    cfg: dict,
    imbalance_ratio: int = 1,
    method: str = "baseline",
    seed: int = 42,
    device=None,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Build training and validation DataLoaders for CIFAR-10.

    Parameters
    ----------
    cfg             : dict  — project config (see config.yaml)
    imbalance_ratio : int   — majority:minority sample ratio
    method          : str   — 'baseline' | 'weighted' | 'focal' |
                               'oversampling' | 'etf'
    seed            : int   — random seed for sampler
    device          : torch.device — used to set pin_memory correctly (CUDA only)

    Returns
    -------
    train_loader  : DataLoader
    val_loader    : DataLoader
    class_weights : torch.Tensor  — per-class weights for loss functions
    """
    root       = cfg["dataset"]["root"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg.get("training", {}).get("num_workers", 4)
    # pin_memory only benefits CUDA — avoids UserWarning on CPU/MPS
    pin_mem = (device is not None and getattr(device, "type", "cpu") == "cuda")

    train_ds = ImbalancedCIFAR10(
        root=root,
        train=True,
        imbalance_ratio=imbalance_ratio,
        transform=_train_transform(),
    )
    val_ds = CIFAR10(root=root, train=False, transform=_val_transform(), download=True)

    # ── Sampler for oversampling ────────────────────────────────────────────────────
    sampler = None
    if method == "oversampling":
        sample_weights = train_ds.class_weights[train_ds.targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights.double(),
            num_samples=len(train_ds),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    return train_loader, val_loader, train_ds.class_weights
