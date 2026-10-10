"""
models/mobilenet.py
─────────────────────────────────────────────────────────────────────────────
MobileNetV2 adapted for CIFAR-10 (32×32 images).

Used as a lightweight alternative to ResNet-18 in ablation experiments.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class MobileNetV2Custom(nn.Module):
    """
    MobileNetV2 backbone for CIFAR-10.

    Parameters
    ----------
    num_classes : int  — classifier output size
    pretrained  : bool — use ImageNet weights
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = False) -> None:
        super().__init__()

        if pretrained:
            backbone = tv_models.mobilenet_v2(
                weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1
            )
        else:
            backbone = tv_models.mobilenet_v2(weights=None)

        # Adapt first conv for 32×32 images (remove aggressive downsampling)
        backbone.features[0][0] = nn.Conv2d(
            3, 32, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.feature_dim: int = backbone.classifier[1].in_features  # 1280

        # Encoder = feature extractor without the classifier head
        self.encoder = backbone.features
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.pool(h)
        return h.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(x)
        return self.fc(h)
