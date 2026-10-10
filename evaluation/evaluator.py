"""
evaluation/evaluator.py
─────────────────────────────────────────────────────────────────────────────
Post-training evaluation utilities.

Functions
---------
evaluate_checkpoint — load a saved .pt checkpoint and report accuracy + NC
extract_features    — run encoder forward-pass and return (features, labels)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from utils.metrics import compute_nc_metrics


def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the model encoder on every batch in *loader* and collect features.

    Parameters
    ----------
    model  : nn.Module  — must expose a `forward_features` method
    loader : DataLoader
    device : torch.device

    Returns
    -------
    features : torch.Tensor, shape (N, D)
    labels   : torch.Tensor, shape (N,)
    """
    model.eval()
    all_feats, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            feats  = model.forward_features(images)
            all_feats.append(feats.cpu())
            all_labels.append(labels)

    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


def evaluate_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Load weights from a checkpoint file and compute val accuracy + NC metrics.

    Parameters
    ----------
    checkpoint_path : str
    model           : nn.Module  — architecture must match the checkpoint
    val_loader      : DataLoader
    num_classes     : int
    device          : torch.device

    Returns
    -------
    dict with keys: val_acc, nc1, nc2
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)   # handle both raw and wrapped saves
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # ── Accuracy ──────────────────────────────────────────────────────────────
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds  = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    val_acc = 100.0 * correct / total if total > 0 else 0.0

    # ── NC metrics ────────────────────────────────────────────────────────────
    features, labels = extract_features(model, val_loader, device)
    nc = compute_nc_metrics(features, labels, num_classes=num_classes)

    return {"val_acc": val_acc, **nc}
