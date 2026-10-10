"""
utils/metrics.py
─────────────────────────────────────────────────────────────────────────────
Neural-Collapse metric computation.

NC1 — Within-Class Variability (Σ_W / Σ_B ratio).
      Measures how tightly feature vectors cluster around their class means.
      Lower is better → perfect collapse → NC1 = 0.

NC2 — Equiangular Tight Frame (ETF) alignment.
      Measures whether class-mean vectors form an ETF structure in feature
      space.  Perfect alignment → NC2 = 0.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def compute_nc_metrics(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """
    Compute NC1 (within-class scatter) and NC2 (ETF alignment) metrics.

    Parameters
    ----------
    features : torch.Tensor, shape (N, D)
        Penultimate-layer feature vectors (already detached from graph).
    labels   : torch.Tensor, shape (N,)
        Ground-truth class indices.
    num_classes : int
        Total number of classes in the dataset.

    Returns
    -------
    dict with keys:
        "nc1" : float  — average within-class covariance ratio
        "nc2" : float  — average cos-distance from ETF ideal
    """
    features = features.float()
    labels   = labels.long()

    D = features.shape[1]

    # ── Per-class means ──────────────────────────────────────────────────────
    class_means = torch.zeros(num_classes, D, device=features.device)
    counts      = torch.zeros(num_classes,    device=features.device)

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_means[c] = features[mask].mean(dim=0)
            counts[c]      = mask.sum().float()

    global_mean = (class_means * counts.unsqueeze(1)).sum(0) / counts.sum()

    # ── NC1: Within-class scatter ─────────────────────────────────────────────
    Sw = torch.zeros(D, D, device=features.device)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        diff = features[mask] - class_means[c]          # (n_c, D)
        Sw   = Sw + diff.T @ diff

    Sw = Sw / features.shape[0]

    # Between-class scatter
    Sb = torch.zeros(D, D, device=features.device)
    for c in range(num_classes):
        diff = (class_means[c] - global_mean).unsqueeze(1)  # (D, 1)
        Sb   = Sb + counts[c] * (diff @ diff.T)

    Sb = Sb / features.shape[0]

    # NC1: trace(Sw) / trace(Sb)
    trace_Sb = Sb.trace().item()
    nc1 = (Sw.trace() / (trace_Sb + 1e-8)).item()

    # ── NC2: ETF alignment ────────────────────────────────────────────────────
    # Ideal ETF: cosine similarity between any two distinct class means = -1/(C-1)
    active = [c for c in range(num_classes) if counts[c] > 0]
    C_eff  = len(active)
    etf_target = -1.0 / (C_eff - 1) if C_eff > 1 else 0.0

    # Normalise class means
    norms = class_means[active]
    norms = F.normalize(norms, dim=1)           # (C_eff, D)

    cos_matrix = norms @ norms.T               # (C_eff, C_eff)

    nc2_terms = []
    for i in range(C_eff):
        for j in range(i + 1, C_eff):
            nc2_terms.append((cos_matrix[i, j].item() - etf_target) ** 2)

    nc2 = float(np.mean(nc2_terms)) if nc2_terms else 0.0

    return {"nc1": nc1, "nc2": nc2}
