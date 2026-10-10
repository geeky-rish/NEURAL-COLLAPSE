"""
data/imbalance_sampler.py
─────────────────────────────────────────────────────────────────────────────
Advanced sampling strategies for handling class imbalance.

Samplers
--------
ClassBalancedSampler   — equal number of samples per class per batch
SquareRootSampler      — sample proportional to √(class_count)
ProgressiveSampler     — linearly shifts from imbalanced → balanced over epochs

All samplers are compatible with PyTorch DataLoader's `sampler` argument.
"""
from __future__ import annotations

import math
from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Sampler


class ClassBalancedSampler(Sampler):
    """
    Samples each class equally every epoch (with replacement for minority classes).

    Parameters
    ----------
    targets          : list[int] — class label for each sample
    num_samples_each : int       — samples per class per epoch (default: max class count)
    generator        : torch.Generator — for reproducibility
    """

    def __init__(
        self,
        targets: List[int],
        num_samples_each: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.targets   = np.array(targets)
        self.generator = generator
        classes        = np.unique(self.targets)

        # Per-class indices
        self._class_indices = {c: np.where(self.targets == c)[0] for c in classes}

        # Default: match the majority class size
        if num_samples_each is None:
            num_samples_each = max(len(v) for v in self._class_indices.values())
        self.num_samples_each = num_samples_each
        self._num_samples = num_samples_each * len(classes)

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(
            seed=self.generator.initial_seed() if self.generator else None
        )
        indices = []
        for class_idx in self._class_indices.values():
            chosen = rng.choice(class_idx, size=self.num_samples_each, replace=True)
            indices.extend(chosen.tolist())
        rng.shuffle(indices)
        return iter(indices)


class SquareRootSampler(Sampler):
    """
    Samples proportional to √(class_count).
    This is a middle ground between uniform (no correction) and balanced.

    Parameters
    ----------
    targets      : list[int]
    num_samples  : int — total samples per epoch
    generator    : torch.Generator
    """

    def __init__(
        self,
        targets: List[int],
        num_samples: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.targets   = np.array(targets)
        self.generator = generator

        counts = np.bincount(self.targets)
        # Weight ∝ 1/√n_c  (so sampling prob ∝ √n_c relative normalised)
        sqrt_counts = np.sqrt(counts + 1e-8)
        self._weights = torch.tensor(
            sqrt_counts[self.targets] / sqrt_counts[self.targets].sum(),
            dtype=torch.double,
        )
        self._num_samples = num_samples or len(targets)

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(
            self._weights,
            num_samples=self._num_samples,
            replacement=True,
            generator=self.generator,
        )
        return iter(indices.tolist())


class ProgressiveSampler(Sampler):
    """
    Progressively transitions from the original imbalanced distribution to
    a class-balanced distribution over *warmup_epochs* epochs.

    At epoch=0   → natural imbalanced weights
    At epoch≥T   → class-balanced weights

    Call `set_epoch(e)` before each epoch to update the blend.

    Parameters
    ----------
    targets       : list[int]
    warmup_epochs : int
    num_samples   : int
    generator     : torch.Generator
    """

    def __init__(
        self,
        targets: List[int],
        warmup_epochs: int = 10,
        num_samples: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.targets       = np.array(targets)
        self.warmup_epochs = warmup_epochs
        self._num_samples  = num_samples or len(targets)
        self.generator     = generator
        self._epoch        = 0

        counts = np.bincount(self.targets).astype(float)
        # Natural weights (proportional to class frequency)
        self._natural  = torch.tensor(counts[self.targets] / counts.sum(), dtype=torch.double)
        # Balanced weights (uniform per class)
        balanced_per_class = 1.0 / (len(counts) * counts)
        self._balanced = torch.tensor(
            balanced_per_class[self.targets] / balanced_per_class[self.targets].sum(),
            dtype=torch.double,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        alpha  = min(1.0, self._epoch / max(1, self.warmup_epochs))
        w      = (1 - alpha) * self._natural + alpha * self._balanced
        w      = w / w.sum()
        indices = torch.multinomial(
            w, num_samples=self._num_samples,
            replacement=True, generator=self.generator,
        )
        return iter(indices.tolist())


def build_sampler(
    strategy: str,
    targets: List[int],
    cfg: dict,
    seed: int = 42,
) -> Optional[Sampler]:
    """
    Factory for sampling strategies.

    Parameters
    ----------
    strategy : 'balanced' | 'square_root' | 'progressive' | 'none'
    targets  : class labels for the training set
    cfg      : full config dict
    seed     : random seed

    Returns
    -------
    Sampler or None (None → use shuffle=True in DataLoader)
    """
    gen = torch.Generator().manual_seed(seed)

    if strategy == "balanced":
        return ClassBalancedSampler(targets, generator=gen)
    elif strategy == "square_root":
        return SquareRootSampler(targets, generator=gen)
    elif strategy == "progressive":
        warmup = cfg.get("sampling", {}).get("progressive_warmup", 10)
        return ProgressiveSampler(targets, warmup_epochs=warmup, generator=gen)
    elif strategy in ("none", "weighted"):
        return None   # caller handles WeightedRandomSampler or no sampler
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy!r}")
