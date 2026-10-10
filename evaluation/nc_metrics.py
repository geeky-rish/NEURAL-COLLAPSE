"""
evaluation/nc_metrics.py
─────────────────────────────────────────────────────────────────────────────
Full Neural Collapse metric suite: NC1 through NC4.

NC1 — Within-class variability collapse
NC2 — Equiangular Tight Frame (ETF) alignment of class means
NC3 — Alignment between classifier weight vectors and class means
NC4 — Nearest-Class-Center (NCC) decision rule agreement with argmax

All four metrics approach 0 at perfect Neural Collapse.

Usage
-----
    from evaluation.nc_metrics import compute_all_nc_metrics, NCMetrics

    features, labels = extract_features(model, loader, device)
    metrics = compute_all_nc_metrics(features, labels, model, num_classes=7)
    print(metrics)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NCMetrics:
    """Container for one measurement of all four NC properties."""
    epoch:   int   = 0
    nc1:     float = float("nan")
    nc2:     float = float("nan")
    nc3:     float = float("nan")
    nc4:     float = float("nan")

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __repr__(self) -> str:
        return (f"NCMetrics(epoch={self.epoch}, "
                f"NC1={self.nc1:.4f}, NC2={self.nc2:.6f}, "
                f"NC3={self.nc3:.4f}, NC4={self.nc4:.4f})")


# ── Individual metric functions ───────────────────────────────────────────────

def _class_means(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-class means, global mean, and per-class counts.

    Returns: (class_means [C,D], global_mean [D], counts [C])
    """
    D = features.shape[1]
    class_means = torch.zeros(num_classes, D, device=features.device)
    counts      = torch.zeros(num_classes,    device=features.device)

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_means[c] = features[mask].mean(0)
            counts[c]      = mask.sum().float()

    global_mean = (class_means * counts.unsqueeze(1)).sum(0) / counts.sum().clamp(min=1)
    return class_means, global_mean, counts


def compute_nc1(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """
    NC1: trace(Σ_W) / trace(Σ_B)

    Lower → better within-class collapse.
    """
    D = features.shape[1]
    class_means, global_mean, counts = _class_means(features, labels, num_classes)

    # Within-class scatter
    Sw = torch.zeros(D, D, device=features.device)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        diff = features[mask] - class_means[c]
        Sw  += diff.T @ diff
    Sw /= features.shape[0]

    # Between-class scatter
    Sb = torch.zeros(D, D, device=features.device)
    for c in range(num_classes):
        if counts[c] == 0:
            continue
        d  = (class_means[c] - global_mean).unsqueeze(1)
        Sb += counts[c] * (d @ d.T)
    Sb /= features.shape[0]

    nc1 = (Sw.trace() / (Sb.trace() + 1e-8)).item()
    return nc1


def compute_nc2(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """
    NC2: mean squared deviation of pairwise cosine similarities from ETF ideal.

    ETF ideal cosine = -1/(C-1) for all pairs i≠j.
    Lower → better ETF alignment.
    """
    class_means, global_mean, counts = _class_means(features, labels, num_classes)
    active = [c for c in range(num_classes) if counts[c] > 0]
    C_eff  = len(active)
    if C_eff < 2:
        return float("nan")

    etf_target = -1.0 / (C_eff - 1)
    norms      = F.normalize(class_means[active], dim=1)   # (C_eff, D)
    cos        = norms @ norms.T                            # (C_eff, C_eff)

    terms = []
    for i in range(C_eff):
        for j in range(i + 1, C_eff):
            terms.append((cos[i, j].item() - etf_target) ** 2)

    return float(np.mean(terms))


def compute_nc3(
    features: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    num_classes: int,
) -> float:
    """
    NC3: mean cosine distance between classifier weight vectors and class means.

    Measures how well W_c aligns with μ_c (NC3 = 0 ↔ perfect alignment).
    Handles both standard nn.Linear (weight shape C×D) and ETFClassifier
    (weight shape D×C — prototype columns, not rows).
    """
    # Extract classifier weights
    fc = getattr(model, "fc", None)
    if fc is None or not hasattr(fc, "weight"):
        return float("nan")

    W = fc.weight.data.detach()   # nn.Linear → (C, D); ETFClassifier → (D, C)

    # Normalise to (C, D) regardless of head type
    if W.ndim != 2:
        return float("nan")
    rows, cols = W.shape
    if rows == num_classes and cols != num_classes:
        # Standard linear: (C, D) — use as-is
        W_cd = W
    elif cols == num_classes and rows != num_classes:
        # ETF buffer: (D, C) — transpose to (C, D)
        W_cd = W.T
    else:
        # Ambiguous or square — cannot determine orientation
        return float("nan")

    class_means, global_mean, counts = _class_means(features, labels, num_classes)
    active = [c for c in range(num_classes) if counts[c] > 0]
    if len(active) < 1:
        return float("nan")

    try:
        W_norm = F.normalize(W_cd[active], dim=1)         # (C_eff, D)
        M_norm = F.normalize(class_means[active], dim=1)  # (C_eff, D)

        if W_norm.shape != M_norm.shape:
            return float("nan")

        # Cosine similarity between each W_c and μ_c
        cos_diag = (W_norm * M_norm).sum(dim=1)           # (C_eff,)
        # NC3 = mean deviation from perfect alignment (cos=1)
        nc3 = (1.0 - cos_diag).mean().item()
        return nc3
    except Exception:
        return float("nan")


def compute_nc4(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    logits: Optional[torch.Tensor] = None,
) -> float:
    """
    NC4: agreement rate between NCC (nearest class center) and argmax classifier.

    NCC assigns label = argmin_c ‖h - μ_c‖ = argmax cosine(h, μ_c).
    NC4 = fraction of samples where NCC agrees with the model's argmax.

    If logits are None, NC4 cannot be computed and returns nan.
    """
    if logits is None:
        return float("nan")

    class_means, _, counts = _class_means(features, labels, num_classes)
    active = [c for c in range(num_classes) if counts[c] > 0]
    if not active:
        return float("nan")

    M_norm = F.normalize(class_means[active], dim=1)   # (C_eff, D)
    H_norm = F.normalize(features, dim=1)              # (N, D)

    ncc_pred   = (H_norm @ M_norm.T).argmax(dim=1)    # (N,) indices into active
    model_pred = logits.argmax(dim=1)                  # (N,)

    # Map NCC predictions back to global class indices
    active_t   = torch.tensor(active, device=features.device)
    ncc_global = active_t[ncc_pred]

    nc4 = (ncc_global == model_pred).float().mean().item()
    # Return as deviation from perfect agreement (0 = perfect)
    return 1.0 - nc4


# ── Unified entry-point ───────────────────────────────────────────────────────

def compute_all_nc_metrics(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    model: Optional[nn.Module] = None,
    logits: Optional[torch.Tensor] = None,
    epoch: int = 0,
) -> NCMetrics:
    """
    Compute NC1 through NC4 and return an NCMetrics dataclass.

    Parameters
    ----------
    features    : (N, D) float tensor
    labels      : (N,)   long tensor
    num_classes : int
    model       : nn.Module or None (required for NC3)
    logits      : (N, C) or None   (required for NC4)
    epoch       : int

    Returns
    -------
    NCMetrics
    """
    features = features.float()
    labels   = labels.long()

    nc1 = compute_nc1(features, labels, num_classes)
    nc2 = compute_nc2(features, labels, num_classes)
    nc3 = compute_nc3(features, labels, model, num_classes) if model else float("nan")
    nc4 = compute_nc4(features, labels, num_classes, logits)

    return NCMetrics(epoch=epoch, nc1=nc1, nc2=nc2, nc3=nc3, nc4=nc4)
