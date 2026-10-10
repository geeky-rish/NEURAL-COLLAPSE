"""
models/resnet.py
─────────────────────────────────────────────────────────────────────────────
ResNet-18 adapted for CIFAR-10 (32×32 images).

Key differences from the ImageNet variant
  • First conv: 3×3, stride 1 (instead of 7×7, stride 2)
  • No max-pool after the first conv
  • The feature layer (penultimate) is exposed for NC analysis.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNet18(nn.Module):
    """
    ResNet-18 backbone adapted for CIFAR-10 (small-image variant).

    Parameters
    ----------
    num_classes : int   — output dimension of the linear classifier head
    pretrained  : bool  — load ImageNet weights (True only makes sense for
                          fine-tuning experiments; default False for NC study)
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = False) -> None:
        super().__init__()

        # Load torchvision ResNet-18 (with or without pretrained weights)
        if pretrained:
            backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = tv_models.resnet18(weights=None)

        # ── Adapt for CIFAR (32×32) ──────────────────────────────────────────
        # Replace 7×7 conv (stride 2) with 3×3 conv (stride 1), remove max-pool
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        # Feature dimension before the classifier
        self.feature_dim: int = backbone.fc.in_features   # 512 for ResNet-18

        # Remove the original FC head; keep everything else as the encoder
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # ends at avgpool

        # Linear classifier head (standard or replaceable with ETF)
        self.fc = nn.Linear(self.feature_dim, num_classes)

    # ── Forward passes ────────────────────────────────────────────────────────

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate (feature) representation, shape (B, D)."""
        h = self.encoder(x)
        return h.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: features → logits, shape (B, C)."""
        h = self.forward_features(x)
        return self.fc(h)
