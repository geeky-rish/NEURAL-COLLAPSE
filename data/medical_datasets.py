"""
data/medical_datasets.py
─────────────────────────────────────────────────────────────────────────────
Medical imaging dataset loaders for:
  • HAM10000  — Skin lesion classification (7 classes, severe imbalance)
  • ChestXRay — Pneumonia binary classification (2 classes)
  • RetinalOCT — Retinal OCT classification (4 classes)

Each dataset class follows the same protocol:
  __init__, __len__, __getitem__ → (image_tensor, label_int)
  .targets         → list[int]   (for sampler compatibility)
  .class_weights   → torch.Tensor (inverse-freq weights)
  .class_names     → list[str]

Factory: get_medical_dataloaders(cfg, seed) → (train_loader, val_loader, class_weights)

Download Instructions
─────────────────────
HAM10000   : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
ChestXRay  : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
RetinalOCT : https://www.kaggle.com/datasets/paultimothymooney/kermany2018
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .preprocessing import get_medical_transforms
from .imbalance_sampler import build_sampler


# ── HAM10000 ──────────────────────────────────────────────────────────────────

HAM_LABEL_MAP: Dict[str, int] = {
    "mel":   0,   # Melanoma
    "nv":    1,   # Melanocytic nevi
    "bcc":   2,   # Basal cell carcinoma
    "akiec": 3,   # Actinic keratoses
    "bkl":   4,   # Benign keratosis-like lesions
    "df":    5,   # Dermatofibroma
    "vasc":  6,   # Vascular lesions
}
HAM_CLASS_NAMES = ["Melanoma", "Nevi", "BCC", "AK", "BKL", "DF", "Vascular"]


def _apply_long_tail_subsampling(
    df: "pd.DataFrame",
    imbalance_ratio: float,
    seed: int,
) -> "pd.DataFrame":
    """
    Subsample *df* (training split only) to create a long-tail class distribution.

    The majority class (highest count) keeps ALL its samples.
    Every other class is reduced so that:
        n_minority = max(1, floor(n_majority / imbalance_ratio))

    Parameters
    ----------
    df              : DataFrame with a 'label' column
    imbalance_ratio : float — majority_count / minority_target_count
    seed            : int   — for reproducibility (same seed as dataset split)

    Returns
    -------
    Subsampled DataFrame, index reset.
    """
    import math

    rng           = np.random.default_rng(seed)
    counts        = df["label"].value_counts()
    majority_cls  = int(counts.idxmax())
    n_majority    = int(counts.max())
    n_minority    = max(1, math.floor(n_majority / imbalance_ratio))

    print(f"  [HAM10000] Applying imbalance_ratio={imbalance_ratio:.1f}")
    print(f"  [HAM10000] Majority class {majority_cls} ('{HAM_CLASS_NAMES[majority_cls]}'): "
          f"{n_majority} samples (unchanged)")
    print(f"  [HAM10000] Target minority size: {n_minority} samples/class")

    kept = []
    for cls, group in df.groupby("label"):
        if int(cls) == majority_cls or len(group) <= n_minority:
            kept.append(group)
        else:
            sampled = group.sample(n=n_minority,
                                   replace=False,
                                   random_state=int(rng.integers(0, 2**31)))
            kept.append(sampled)

    result = pd.concat(kept).reset_index(drop=True)

    # Log the resulting distribution
    final_counts = result["label"].value_counts().sort_index()
    print("  [HAM10000] Post-subsampling distribution:")
    for cls_idx, cnt in final_counts.items():
        print(f"    Class {cls_idx} ({HAM_CLASS_NAMES[int(cls_idx)]}): {cnt}")

    return result


class HAM10000Dataset(Dataset):
    """
    HAM10000 Skin Lesion Dataset.

    Parameters
    ----------
    csv_path        : str   — path to HAM10000_metadata.csv
    img_dir         : str   — directory containing JPEG images
    split           : 'train' | 'val' | 'test'
    transform       : callable
    val_frac        : float — fraction held out for validation (default 0.15)
    test_frac       : float — fraction held out for test       (default 0.10)
    seed            : int
    imbalance_ratio : float — majority:minority ratio for long-tail subsampling.
                              Only applied when split=='train' AND ratio > 1.0.
                              Default 1.0 = full natural distribution (no change).
    """

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        split: str = "train",
        transform=None,
        val_frac: float = 0.15,
        test_frac: float = 0.10,
        seed: int = 42,
        imbalance_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.img_dir   = Path(img_dir)
        self.transform = transform

        df = pd.read_csv(csv_path)
        df["label"] = df["dx"].map(HAM_LABEL_MAP)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        # ── Stratified split ──────────────────────────────────────────────────
        rng = np.random.default_rng(seed)
        df  = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        n   = len(df)
        n_test = int(n * test_frac)
        n_val  = int(n * val_frac)

        if split == "test":
            df = df.iloc[:n_test]
        elif split == "val":
            df = df.iloc[n_test: n_test + n_val]
        else:   # train
            df = df.iloc[n_test + n_val:]

        # ── Optional long-tail subsampling (train split only) ─────────────────
        if split == "train" and imbalance_ratio > 1.0:
            df = _apply_long_tail_subsampling(df, imbalance_ratio, seed)
        elif split != "train" and imbalance_ratio > 1.0:
            pass   # val/test never touched — intentional

        self.df          = df.reset_index(drop=True)
        self.targets     = self.df["label"].tolist()
        self.class_names = HAM_CLASS_NAMES

        # ── Recompute class weights from the (possibly subsampled) distribution
        counts = np.bincount(self.targets, minlength=len(HAM_LABEL_MAP)).astype(float)
        freq   = counts / counts.sum()
        self.class_weights = torch.tensor(1.0 / (freq + 1e-8), dtype=torch.float32)
        self.class_weights /= self.class_weights.sum()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row   = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['image_id']}.jpg"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


# ── Chest X-Ray (Pneumonia) ───────────────────────────────────────────────────

CHEST_CLASS_NAMES = ["Normal", "Pneumonia"]


class ChestXRayDataset(Dataset):
    """
    Chest X-Ray Pneumonia Dataset (Kaggle format).

    Expected directory layout:
        root/
          train/NORMAL/...
          train/PNEUMONIA/...
          val/NORMAL/...
          val/PNEUMONIA/...
          test/NORMAL/...
          test/PNEUMONIA/...

    Parameters
    ----------
    root      : str — dataset root directory
    split     : 'train' | 'val' | 'test'
    transform : callable
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
    ) -> None:
        super().__init__()
        self.transform = transform
        split_dir      = Path(root) / split

        self.samples: List[Tuple[Path, int]] = []
        for label, class_name in enumerate(CHEST_CLASS_NAMES):
            class_dir = split_dir / class_name.upper()
            if not class_dir.exists():
                # Try mixed case
                class_dir = split_dir / class_name
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for p in class_dir.glob(ext):
                    self.samples.append((p, label))

        self.targets     = [s[1] for s in self.samples]
        self.class_names = CHEST_CLASS_NAMES

        counts = np.bincount(self.targets, minlength=2).astype(float)
        freq   = counts / counts.sum()
        self.class_weights = torch.tensor(1.0 / (freq + 1e-8), dtype=torch.float32)
        self.class_weights /= self.class_weights.sum()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Retinal OCT ───────────────────────────────────────────────────────────────

OCT_CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


class RetinalOCTDataset(Dataset):
    """
    Retinal OCT Dataset (Kermany 2018 / Kaggle OCT2017 format).

    Expected layout:
        root/train/CNV/...   root/train/DME/...  etc.
        root/test/CNV/...    etc.

    Parameters
    ----------
    root      : str
    split     : 'train' | 'val' | 'test'
    transform : callable
    val_frac  : float — fraction carved from train for validation
    seed      : int
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        val_frac: float = 0.15,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.transform = transform
        root_path      = Path(root)

        # Collect all samples from the raw train / test folders
        raw_split = "train" if split in ("train", "val") else "test"
        all_samples: List[Tuple[Path, int]] = []
        for label, class_name in enumerate(OCT_CLASS_NAMES):
            class_dir = root_path / raw_split / class_name
            if not class_dir.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for p in class_dir.glob(ext):
                    all_samples.append((p, label))

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(all_samples))
        n_val   = int(len(all_samples) * val_frac)

        if split == "val":
            indices = indices[:n_val]
        elif split == "train":
            indices = indices[n_val:]
        # test → all samples from raw test folder

        self.samples     = [all_samples[i] for i in indices]
        self.targets     = [s[1] for s in self.samples]
        self.class_names = OCT_CLASS_NAMES

        counts = np.bincount(self.targets, minlength=4).astype(float)
        freq   = counts / counts.sum()
        self.class_weights = torch.tensor(1.0 / (freq + 1e-8), dtype=torch.float32)
        self.class_weights /= self.class_weights.sum()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Factory ───────────────────────────────────────────────────────────────────

def get_medical_dataloaders(
    cfg: dict,
    seed: int = 42,
    device=None,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Build train / val DataLoaders for the medical dataset specified in cfg.

    Parameters
    ----------
    cfg    : dict          — full project config
    seed   : int
    device : torch.device  — used to set pin_memory correctly (CUDA only)

    Returns
    -------
    train_loader, val_loader, class_weights
    """
    dataset_name = cfg["dataset"]["name"].lower()
    image_size   = cfg["dataset"].get("image_size", 224)
    batch_size   = cfg["training"]["batch_size"]
    strategy     = cfg.get("sampling", {}).get("strategy", "weighted") or "none"

    med_cfg      = cfg.get("medical", {})

    train_tf = get_medical_transforms("train", image_size, dataset_name)
    val_tf   = get_medical_transforms("val",   image_size, dataset_name)

    # ── Instantiate datasets ──────────────────────────────────────────────────
    if dataset_name == "ham10000":
        imb_ratio = float(cfg.get("dataset", {}).get("imbalance_ratio", 1.0))
        train_ds = HAM10000Dataset(
            csv_path=med_cfg["ham10000"]["csv_path"],
            img_dir=med_cfg["ham10000"]["img_dir"],
            split="train", transform=train_tf, seed=seed,
            imbalance_ratio=imb_ratio,          # only applied to train split
        )
        val_ds = HAM10000Dataset(
            csv_path=med_cfg["ham10000"]["csv_path"],
            img_dir=med_cfg["ham10000"]["img_dir"],
            split="val", transform=val_tf, seed=seed,
            # imbalance_ratio intentionally omitted — val always uses full distribution
        )

    elif dataset_name == "chestxray":
        train_ds = ChestXRayDataset(
            root=med_cfg["chestxray"]["root"], split="train", transform=train_tf,
        )
        val_ds = ChestXRayDataset(
            root=med_cfg["chestxray"]["root"], split="val", transform=val_tf,
        )

    elif dataset_name == "retinal_oct":
        train_ds = RetinalOCTDataset(
            root=med_cfg["retinal_oct"]["root"],
            split="train", transform=train_tf, seed=seed,
        )
        val_ds = RetinalOCTDataset(
            root=med_cfg["retinal_oct"]["root"],
            split="val", transform=val_tf, seed=seed,
        )

    else:
        raise ValueError(
            f"Unknown medical dataset: {dataset_name!r}. "
            "Choose from: ham10000 | chestxray | retinal_oct"
        )

    class_weights = train_ds.class_weights

    # ── Build sampler ─────────────────────────────────────────────────────────
    sampler = None
    shuffle = True

    if strategy == "weighted":
        sample_w = class_weights[train_ds.targets]
        sampler  = WeightedRandomSampler(
            weights=sample_w.double(),
            num_samples=len(train_ds),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        shuffle = False
    elif strategy != "none":
        sampler = build_sampler(strategy, train_ds.targets, cfg, seed)
        shuffle = False

    num_workers = cfg.get("training", {}).get("num_workers", 4)
    # pin_memory only benefits CUDA — avoids UserWarning on CPU/MPS
    pin_mem = (device is not None and getattr(device, "type", "cpu") == "cuda")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, shuffle=(sampler is None and shuffle),
        num_workers=num_workers, pin_memory=pin_mem, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2,
        shuffle=False, num_workers=num_workers, pin_memory=pin_mem,
    )

    return train_loader, val_loader, class_weights
