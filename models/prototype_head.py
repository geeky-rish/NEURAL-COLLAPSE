"""
models/prototype_head.py
─────────────────────────────────────────────────────────────────────────────
Learnable Class Prototype Head.

Instead of a standard linear layer W·h, this head maintains one learnable
prototype vector p_c per class and classifies via cosine similarity:

    logit_c = τ · cos(h, p_c)   where τ is a learnable temperature

Motivation in medical AI
─────────────────────────
• Prototype-based classification is interpretable: each class is represented
  by a single "ideal" feature vector in embedding space.
• Under class imbalance, prototypes decouple representation learning from
  the classifier bias — minority prototypes can still be placed optimally
  even with few training examples.
• Prototype geometry can be directly visualised (t-SNE/UMAP) and compared
  to the ETF ideal.

Reference: Snell et al. (2017) Prototypical Networks for Few-shot Learning.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeHead(nn.Module):
    """
    Learnable cosine-similarity prototype classifier.

    Parameters
    ----------
    feature_dim  : int   — input feature dimension D
    num_classes  : int   — number of classes C
    learnable    : bool  — if False, prototypes are fixed (ETF-like init)
    temperature  : float — initial temperature τ (learnable scalar)
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        learnable: bool   = True,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.learnable   = learnable

        # ── Prototype matrix P ∈ R^{C × D} ───────────────────────────────────
        P = self._init_prototypes(feature_dim, num_classes)
        if learnable:
            self.prototypes = nn.Parameter(P)
        else:
            self.register_buffer("prototypes", P)

        # Learnable log-temperature (starts at log(τ))
        self.log_tau = nn.Parameter(
            torch.tensor(math.log(temperature), dtype=torch.float32)
        )

    # ── Initialisation ────────────────────────────────────────────────────────

    @staticmethod
    def _init_prototypes(D: int, C: int) -> torch.Tensor:
        """
        Initialise prototypes to an approximate ETF arrangement so that
        training starts from a geometry-aware state.
        """
        P = torch.randn(C, D)
        if D >= C:
            Q, _ = torch.linalg.qr(P.T)   # (D, D)
            P    = Q[:, :C].T              # (C, D)
        P = F.normalize(P, dim=1)
        return P

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (B, D)  — L2-normalised or raw penultimate features

        Returns
        -------
        logits   : (B, C)
        """
        # Normalise both features and prototypes
        h = F.normalize(features,   dim=1)   # (B, D)
        P = F.normalize(self.prototypes, dim=1)   # (C, D)

        tau    = self.log_tau.exp().clamp(min=1e-4, max=100.0)
        logits = tau * h @ P.T               # (B, C)
        return logits

    @torch.no_grad()
    def get_prototype_matrix(self) -> torch.Tensor:
        """Return normalised prototype matrix (C, D) for visualisation."""
        return F.normalize(self.prototypes, dim=1)

    def extra_repr(self) -> str:
        tau = self.log_tau.exp().item()
        return (f"feature_dim={self.feature_dim}, "
                f"num_classes={self.num_classes}, "
                f"learnable={self.learnable}, "
                f"tau≈{tau:.4f}")
