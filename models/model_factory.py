"""
models/model_factory.py  (updated)
─────────────────────────────────────────────────────────────────────────────
Builds backbone + head from config.
Supports: linear | etf | prototype
"""
from __future__ import annotations

import torch.nn as nn

from .resnet import ResNet18
from .mobilenet import MobileNetV2Custom
from .etf_classifier import ETFClassifier
from .prototype_head import PrototypeHead


def build_model(cfg: dict, method: str = "baseline") -> nn.Module:
    """
    Instantiate backbone and attach the appropriate classifier head.

    Parameters
    ----------
    cfg    : dict — full project config
    method : str  — 'baseline'|'weighted_ce'|'focal'|'oversampling'|
                    'etf'|'prototype'

    Returns
    -------
    model : nn.Module  (with forward_features and forward methods)
    """
    backbone_name = cfg["model"]["backbone"].lower()
    num_classes   = cfg["dataset"]["num_classes"]
    pretrained    = cfg["model"].get("pretrained", False)
    head_type     = cfg["model"].get("head", "linear").lower()

    # ── method can also override head type ────────────────────────────────────
    if method == "etf":
        head_type = "etf"
    elif method == "prototype":
        head_type = "prototype"

    # ── Select backbone ───────────────────────────────────────────────────────
    if backbone_name == "resnet18":
        model = ResNet18(num_classes=num_classes, pretrained=pretrained)
    elif backbone_name == "mobilenetv2":
        model = MobileNetV2Custom(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown backbone: {backbone_name!r}. "
            "Choose from ['resnet18', 'mobilenetv2']."
        )

    feature_dim = model.feature_dim

    # ── Swap head ─────────────────────────────────────────────────────────────
    if head_type == "etf":
        scale = cfg.get("etf", {}).get("scale", 16.0)
        model.fc = ETFClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            scale=scale,
        )
        if cfg.get("etf", {}).get("fix_backbone", False):
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad_(False)

    elif head_type == "prototype":
        proto_cfg = cfg.get("prototype", {})
        model.fc  = PrototypeHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
            learnable=proto_cfg.get("learnable", True),
            temperature=proto_cfg.get("temperature", 0.07),
        )

    # linear head is already the default — no change needed

    return model
