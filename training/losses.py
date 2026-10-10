"""
training/losses.py
─────────────────────────────────────────────────────────────────────────────
Loss functions used across all experiment methods.

Supported criteria
------------------
• CrossEntropyLoss          — standard baseline
• WeightedCrossEntropyLoss  — class-frequency-weighted CE
• FocalLoss                 — Lin et al. (2017), gamma modulates hard-example
                              focus, optional per-class alpha weighting
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    L_focal = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma       : float        — focusing parameter (0 = standard CE)
    alpha       : Tensor|None  — per-class weight tensor of shape (C,);
                                 None → uniform weighting
    reduction   : str          — 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(
        self,
        logits: torch.Tensor,   # (B, C)
        targets: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)          # (B, C)
        probs     = log_probs.exp()

        # Gather per-sample probabilities for the true class
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        pt     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)      # (B,)

        # Focusing factor
        focal_weight = (1.0 - pt) ** self.gamma

        # Class weights (alpha)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
        else:
            alpha_t = torch.ones_like(pt)

        loss = -alpha_t * focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ── Factory ───────────────────────────────────────────────────────────────────

def get_criterion(
    method: str,
    class_weights: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> nn.Module:
    """
    Return the appropriate loss criterion for *method*.

    Parameters
    ----------
    method        : str           — 'baseline' | 'weighted' | 'focal' |
                                    'oversampling' | 'etf'
    class_weights : torch.Tensor  — per-class inverse-frequency weights (C,)
    cfg           : dict          — full config dict
    device        : torch.device

    Returns
    -------
    criterion : nn.Module
    """
    class_weights = class_weights.to(device)

    if method in ("baseline", "oversampling", "etf"):
        # Standard cross-entropy — the dataset / architecture handles balance
        return nn.CrossEntropyLoss()

    elif method == "weighted":
        return nn.CrossEntropyLoss(weight=class_weights)

    elif method == "focal":
        gamma = cfg.get("focal_loss", {}).get("gamma", 2.0)
        raw_alpha = cfg.get("focal_loss", {}).get("alpha", None)

        alpha: Optional[torch.Tensor] = None
        if raw_alpha is None:
            # Auto-compute from class frequencies (inverse frequency)
            alpha = class_weights.clone()
        elif isinstance(raw_alpha, (list, tuple)):
            alpha = torch.tensor(raw_alpha, dtype=torch.float32, device=device)

        return FocalLoss(gamma=gamma, alpha=alpha)

    else:
        raise ValueError(f"Unknown method for criterion: {method!r}")
