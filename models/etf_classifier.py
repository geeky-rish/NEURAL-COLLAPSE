"""
models/etf_classifier.py
─────────────────────────────────────────────────────────────────────────────
Equiangular Tight Frame (ETF) Classifier head.

Replaces the standard linear layer with a fixed ETF weight matrix.
The class-prototype vectors are initialised to form a maximally equiangular
simplex ETF and are *frozen* throughout training (only the backbone is updated).

Theory
------
An ETF satisfies:
    W^T W = (C / (C-1)) * (I - (1/C) * 11^T)

where W ∈ R^{D×C} are the prototype vectors, C is the number of classes, and
D >= C-1 for the ETF to exist.

Reference: Zhu et al., "Geometric Analysis of Neural Collapse with Implicit
           Cross-Entropy Loss" (NeurIPS 2021).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ETFClassifier(nn.Module):
    """
    Fixed ETF linear head (no learnable parameters).

    Parameters
    ----------
    feature_dim  : int — input feature dimension D
    num_classes  : int — number of classes C  (must satisfy D >= C - 1)
    scale        : float — temperature scale on the cosine logits
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        scale: float = 1.0,
    ) -> None:
        super().__init__()

        if feature_dim < num_classes - 1:
            raise ValueError(
                f"ETF requires feature_dim ({feature_dim}) >= num_classes - 1 "
                f"({num_classes - 1})."
            )

        self.num_classes = num_classes
        self.scale       = scale

        # Build the ETF weight matrix and register as a *buffer* (not a param)
        W = self._init_etf(feature_dim, num_classes)
        self.register_buffer("weight", W)

    # ── Initialisation ────────────────────────────────────────────────────────

    @staticmethod
    def _init_etf(D: int, C: int) -> torch.Tensor:
        """
        Construct an ETF weight matrix W ∈ R^{D×C} via the standard recipe:
          1. Draw a random D×C Gaussian matrix.
          2. Orthogonalise via QR decomposition.
          3. Centre columns to have zero mean.
          4. Normalise each column to unit length.
        """
        # Random Gaussian initialisation
        W = torch.randn(D, C)

        # QR orthogonalisation (use only first C columns of Q)
        if D >= C:
            Q, _ = torch.linalg.qr(W)          # Q: (D, D)
            W    = Q[:, :C]                     # (D, C)
        else:
            # Under-determined; use thin QR on the transposed problem
            Q, _ = torch.linalg.qr(W.T)        # Q: (C, C)
            W    = Q[:D, :].T                   # (D, min(D,C))

        # Centre and normalise columns
        W = W - W.mean(dim=1, keepdim=True)
        W = F.normalize(W, dim=0)              # unit-norm columns
        return W                               # shape (D, C)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor, shape (B, D)
            L2-normalised penultimate-layer features.

        Returns
        -------
        logits : torch.Tensor, shape (B, C)
        """
        normed = F.normalize(features, dim=1)           # (B, D)
        logits = self.scale * normed @ self.weight       # (B, C)
        return logits

    def extra_repr(self) -> str:
        D, C = self.weight.shape
        return f"feature_dim={D}, num_classes={C}, scale={self.scale}"
