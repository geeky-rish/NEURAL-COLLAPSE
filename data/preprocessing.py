"""
data/preprocessing.py
─────────────────────────────────────────────────────────────────────────────
Medical-image-specific preprocessing and augmentation pipelines.

Provides:
  • get_medical_transforms(split, image_size, dataset_name)
    Returns a torchvision Compose transform appropriate for train/val/test.

Medical datasets need stronger augmentation than natural image benchmarks
because labelled examples are scarce, especially for minority pathologies.
"""
from __future__ import annotations

import torchvision.transforms as T


# ── Per-dataset normalisation constants ───────────────────────────────────────
_NORM = {
    "ham10000":    dict(mean=(0.7630, 0.5456, 0.5700), std=(0.1409, 0.1521, 0.1693)),
    "chestxray":   dict(mean=(0.4823, 0.4823, 0.4823), std=(0.2350, 0.2350, 0.2350)),
    "retinal_oct": dict(mean=(0.1914, 0.1914, 0.1914), std=(0.1965, 0.1965, 0.1965)),
    "cifar10":     dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    "default":     dict(mean=(0.485,  0.456,  0.406),  std=(0.229,  0.224,  0.225)),
}


def get_medical_transforms(
    split: str = "train",
    image_size: int = 224,
    dataset_name: str = "default",
) -> T.Compose:
    """
    Return an augmentation pipeline tailored to medical imaging.

    Parameters
    ----------
    split        : 'train' | 'val' | 'test'
    image_size   : target resolution after resize/crop
    dataset_name : key into normalisation table

    Returns
    -------
    torchvision.transforms.Compose
    """
    norm_params = _NORM.get(dataset_name.lower(), _NORM["default"])
    normalize   = T.Normalize(**norm_params)

    if split == "train":
        return T.Compose([
            T.Resize((image_size + 32, image_size + 32)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        # val / test — deterministic
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            normalize,
        ])


def get_cifar_transforms(split: str = "train") -> T.Compose:
    """Lightweight CIFAR-10 transforms (32×32, no resize needed)."""
    normalize = T.Normalize(**_NORM["cifar10"])
    if split == "train":
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    return T.Compose([T.ToTensor(), normalize])

def build_transforms(dataset_name: str, image_size: int = 224):
    """Compatibility wrapper for returning train and val transforms."""
    train_transform = get_medical_transforms(
        split="train",
        image_size=image_size,
        dataset_name=dataset_name,
    )
    val_transform = get_medical_transforms(
        split="val",
        image_size=image_size,
        dataset_name=dataset_name,
    )
    return train_transform, val_transform
