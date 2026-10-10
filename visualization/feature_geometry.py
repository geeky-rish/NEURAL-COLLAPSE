"""
visualization/feature_geometry.py
─────────────────────────────────────────────────────────────────────────────
Geometric analysis plots for penultimate-layer feature space.

Functions
---------
plot_pca             — 2D PCA scatter (fast alternative to t-SNE)
plot_feature_norms   — per-class feature norm distributions (box + violin)
plot_cosine_heatmap  — class-mean cosine similarity matrix vs ETF ideal
plot_nc_evolution    — NC1/NC2/NC3/NC4 over training epochs (line plots)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]
_DPI = 150


def plot_pca(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: str = "./results/pca.png",
    title: str = "PCA Feature Visualisation",
    dpi: int = _DPI,
) -> None:
    """2D PCA scatter — fast geometry snapshot."""
    X = features.detach().cpu().numpy().astype(np.float32)
    y = labels.detach().cpu().numpy()
    num_classes = int(y.max()) + 1
    if class_names is None:
        class_names = [f"Class {c}" for c in range(num_classes)]

    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    var  = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7), dpi=dpi)
    counts  = np.bincount(y, minlength=num_classes)

    for c in range(num_classes):
        mask = y == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=_PALETTE[c % len(_PALETTE)], s=16, alpha=0.7,
                   label=f"{class_names[c]} (n={counts[c]})")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.8, ncol=2)
    ax.grid(alpha=0.2)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  [PCA] Saved → {save_path}")


def plot_feature_norms(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: str = "./results/feature_norms.png",
    title: str = "Per-Class Feature Norm Distribution",
    dpi: int = _DPI,
) -> None:
    """
    Box + strip plot of L2 feature norms per class.

    Large norm variance within a class suggests poor within-class collapse.
    Minority classes often have larger variance under imbalance.
    """
    X = features.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    num_classes = int(y.max()) + 1
    if class_names is None:
        class_names = [f"C{c}" for c in range(num_classes)]

    norms_per_class = [
        np.linalg.norm(X[y == c], axis=1) for c in range(num_classes)
    ]

    fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.9), 5), dpi=dpi)
    positions = list(range(num_classes))

    bp = ax.boxplot(norms_per_class, positions=positions, widths=0.5,
                    patch_artist=True, notch=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], _PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("L2 Feature Norm", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  [Feature Norms] Saved → {save_path}")


def plot_cosine_heatmap(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    save_path: str = "./results/cosine_heatmap.png",
    title: str = "Class-Mean Cosine Similarity",
    dpi: int = _DPI,
) -> None:
    """
    Heatmap of pairwise cosine similarities between class-mean vectors.

    Compares the learned geometry to the ETF ideal:
      • Diagonal should be 1.0
      • Off-diagonal should be -1/(C-1) for perfect ETF

    Color-codes deviations from the ETF ideal.
    """
    if class_names is None:
        class_names = [f"C{c}" for c in range(num_classes)]

    X = features.detach().cpu().float()
    y = labels.detach().cpu().long()

    class_means = torch.zeros(num_classes, X.shape[1])
    mask_active = torch.zeros(num_classes, dtype=torch.bool)

    for c in range(num_classes):
        mask = y == c
        if mask.sum() > 0:
            class_means[c] = F.normalize(X[mask].mean(0), dim=0)
            mask_active[c] = True

    M   = class_means[mask_active]
    cos = (M @ M.T).numpy()

    # ETF ideal for reference
    C_eff      = mask_active.sum().item()
    etf_target = -1.0 / (C_eff - 1) if C_eff > 1 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi)
    active_names = [class_names[i] for i in range(num_classes) if mask_active[i]]

    for ax, data, t in zip(axes, [cos, cos - etf_target],
                           ["Cosine Similarity", f"Deviation from ETF (ideal={etf_target:.3f})"]):
        vmax = max(abs(data.max()), abs(data.min()))
        im   = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(C_eff))
        ax.set_yticks(range(C_eff))
        ax.set_xticklabels(active_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(active_names, fontsize=8)
        ax.set_title(t, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046)

        for i in range(C_eff):
            for j in range(C_eff):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  [Cosine Heatmap] Saved → {save_path}")


def plot_nc_evolution(
    nc_history: List[Dict],
    save_path: str = "./results/nc_evolution.png",
    title: str = "Neural Collapse Metrics over Training",
    dpi: int = _DPI,
) -> None:
    """
    Line plots of NC1, NC2, NC3, NC4 across epochs.

    Parameters
    ----------
    nc_history : list of dicts with keys {epoch, nc1, nc2, nc3, nc4}
    """
    if not nc_history:
        return

    epochs = [d["epoch"] for d in nc_history]
    metrics = {
        "NC1 (within-class scatter)": [d.get("nc1", float("nan")) for d in nc_history],
        "NC2 (ETF deviation)":        [d.get("nc2", float("nan")) for d in nc_history],
        "NC3 (W–μ misalignment)":     [d.get("nc3", float("nan")) for d in nc_history],
        "NC4 (NCC disagreement)":     [d.get("nc4", float("nan")) for d in nc_history],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=dpi)
    axes = axes.flatten()

    for ax, (label, values), color in zip(axes, metrics.items(), _PALETTE):
        vals = np.array(values, dtype=float)
        valid = ~np.isnan(vals)
        if valid.any():
            ax.plot(np.array(epochs)[valid], vals[valid],
                    color=color, linewidth=2, marker="o", markersize=4)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Metric value (↓ better)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  [NC Evolution] Saved → {save_path}")
