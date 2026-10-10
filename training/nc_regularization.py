"""
training/nc_regularization.py
─────────────────────────────────────────────────────────────────────────────
Neural Collapse-inspired auxiliary losses.

These losses are added to the primary classification loss to encourage
the feature geometry to develop NC properties during training — even under
class imbalance where NC may otherwise fail to emerge.

Losses
------
NCCollapseRegularizer — penalises within-class feature variance (encourages NC1)
ETFAlignmentLoss      — pushes class means toward an ETF arrangement (encourages NC2)
SupConLoss            — supervised contrastive loss for intra/inter-class structure

Usage in Trainer
----------------
    reg = NCCollapseRegularizer(weight=cfg["nc_regularization"]["collapse_weight"])
    total_loss = ce_loss + reg(features, labels)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NCCollapseRegularizer(nn.Module):
    """
    Encourages NC1: penalises within-class feature variance.

    For each class c, computes the mean squared distance from the class mean:
        L_collapse = (1/N) Σ_c Σ_{i∈c} ‖h_i - μ_c‖²

    This directly regularises within-class scatter (the numerator of NC1),
    pushing minority-class features to cluster tightly even under imbalance.

    Parameters
    ----------
    weight      : float — loss multiplier λ
    normalise   : bool  — normalise features to unit sphere before computing
    """

    def __init__(self, weight: float = 0.01, normalise: bool = True) -> None:
        super().__init__()
        self.weight    = weight
        self.normalise = normalise

    def forward(
        self,
        features: torch.Tensor,   # (B, D)
        labels: torch.Tensor,     # (B,)
    ) -> torch.Tensor:
        # Fast path: weight == 0 → this term is disabled; return a clean zero.
        # Using features.sum() * 0 keeps the zero on the same device/dtype
        # without building any expensive graph nodes.
        if self.weight == 0.0:
            return features.sum() * 0.0

        if self.normalise:
            features = F.normalize(features, dim=1)

        loss  = torch.tensor(0.0, device=features.device, requires_grad=True)
        count = 0

        for c in labels.unique():
            mask     = labels == c
            if mask.sum() < 2:
                continue
            feats_c  = features[mask]                       # (n_c, D)
            mean_c   = feats_c.mean(dim=0, keepdim=True)   # (1, D)
            variance = ((feats_c - mean_c) ** 2).sum(dim=1).mean()
            loss     = loss + variance
            count   += 1

        return self.weight * (loss / max(count, 1))


class ETFAlignmentLoss(nn.Module):
    """
    Encourages NC2: pushes class-mean vectors toward an ETF arrangement.

    The ideal ETF cosine similarity between any two distinct class means is
    -1 / (C - 1).  This loss penalises deviations from that ideal.

    Parameters
    ----------
    num_classes : int
    weight      : float — loss multiplier λ
    """

    def __init__(self, num_classes: int, weight: float = 0.01) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.weight      = weight
        self.etf_target  = -1.0 / (num_classes - 1) if num_classes > 1 else 0.0

    def forward(
        self,
        features: torch.Tensor,   # (B, D)
        labels: torch.Tensor,     # (B,)
    ) -> torch.Tensor:
        # Fast path: weight == 0 → this term is disabled; return a clean zero.
        if self.weight == 0.0:
            return features.sum() * 0.0

        # Compute per-class means
        D          = features.shape[1]
        class_means = torch.zeros(self.num_classes, D, device=features.device)
        class_seen  = torch.zeros(self.num_classes,    device=features.device, dtype=torch.bool)

        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_means[c] = F.normalize(features[mask].mean(0), dim=0)
                class_seen[c]  = True

        active = class_seen.nonzero(as_tuple=True)[0]
        if len(active) < 2:
            return torch.tensor(0.0, device=features.device)

        M    = class_means[active]                  # (C_eff, D)
        cos  = M @ M.T                              # (C_eff, C_eff)
        mask_off = ~torch.eye(len(active), dtype=torch.bool, device=features.device)
        deviations = (cos[mask_off] - self.etf_target) ** 2

        return self.weight * deviations.mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).

    Pulls same-class features together and pushes different-class features
    apart in the normalised embedding space.

    Parameters
    ----------
    temperature : float — τ controlling concentration
    weight      : float — loss multiplier λ
    """

    def __init__(self, temperature: float = 0.07, weight: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.weight      = weight

    def forward(
        self,
        features: torch.Tensor,   # (B, D)
        labels: torch.Tensor,     # (B,)
    ) -> torch.Tensor:
        B      = features.shape[0]
        device = features.device

        features = F.normalize(features, dim=1)   # unit sphere

        # Similarity matrix (B, B)
        sim = features @ features.T / self.temperature

        # Mask out self-similarities
        eye = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(eye, float("-inf"))

        # Positive mask: same class, excluding diagonal
        labels_col = labels.unsqueeze(1)             # (B, 1)
        pos_mask   = (labels_col == labels_col.T) & ~eye   # (B, B)

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Log-softmax over all other samples (negatives + positives)
        log_prob   = F.log_softmax(sim, dim=1)        # (B, B)

        # Mean over positive pairs
        mean_log_prob_pos = (log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)
        loss = -mean_log_prob_pos.mean()

        return self.weight * loss


class CombinedNCLoss(nn.Module):
    """
    Convenience wrapper: classification loss + NC regularization.

    Parameters
    ----------
    base_criterion    : nn.Module — primary loss (CE, focal, etc.)
    collapse_weight   : float
    etf_align_weight  : float
    num_classes       : int
    supcon_weight     : float — 0.0 to disable SupCon
    supcon_temperature: float
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        num_classes: int,
        collapse_weight: float   = 0.01,
        etf_align_weight: float  = 0.01,
        supcon_weight: float     = 0.0,
        supcon_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.base         = base_criterion
        self.nc_collapse  = NCCollapseRegularizer(weight=collapse_weight)
        self.etf_align    = ETFAlignmentLoss(num_classes=num_classes,
                                              weight=etf_align_weight)
        self.supcon       = SupConLoss(temperature=supcon_temperature,
                                        weight=supcon_weight) if supcon_weight > 0 else None

    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.base(logits, labels)
        loss = loss + self.nc_collapse(features, labels)
        loss = loss + self.etf_align(features, labels)
        if self.supcon is not None:
            loss = loss + self.supcon(features, labels)
        return loss
