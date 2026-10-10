"""
visualization/tsne_visualizer.py
─────────────────────────────────────────────────────────────────────────────
Publication-quality t-SNE visualisation of penultimate-layer features.

Visualises:
  • Class cluster separation (inter-class geometry)
  • Minority class separability
  • Collapse progression across training epochs
  • ETF vs linear head comparison

Uses PCA pre-reduction (to cfg pca_components) before t-SNE for speed and
stability when D >> 50.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Colour palette (up to 10 classes)
_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def plot_tsne(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: str = "./results/tsne.png",
    title: str = "t-SNE Feature Visualisation",
    perplexity: int = 30,
    n_iter: int = 1000,
    pca_components: int = 50,
    highlight_minority: bool = True,
    epoch: Optional[int] = None,
    dpi: int = 150,
) -> None:
    """
    Compute and plot t-SNE embedding of *features*.

    Parameters
    ----------
    features          : (N, D) float tensor — penultimate features
    labels            : (N,)   long tensor  — class indices
    class_names       : list[str] — if None, uses "Class 0", "Class 1", ...
    save_path         : output PNG file path
    title             : plot title
    perplexity        : t-SNE perplexity (recommended 5–50)
    n_iter            : t-SNE iterations
    pca_components    : PCA pre-reduction dimensionality (0 = skip PCA)
    highlight_minority: outline the smallest class in red
    epoch             : if provided, appended to title
    dpi               : figure DPI
    """
    X = features.detach().cpu().numpy().astype(np.float32)
    y = labels.detach().cpu().numpy()

    num_classes = int(y.max()) + 1
    if class_names is None:
        class_names = [f"Class {c}" for c in range(num_classes)]

    # ── PCA pre-reduction ─────────────────────────────────────────────────────
    if pca_components > 0 and X.shape[1] > pca_components:
        pca = PCA(n_components=min(pca_components, X.shape[0] - 1), random_state=42)
        X   = pca.fit_transform(X)

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    tsne    = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                   random_state=42, init="pca", learning_rate="auto")
    X_2d    = tsne.fit_transform(X)

    # ── Identify minority class ───────────────────────────────────────────────
    counts       = np.bincount(y, minlength=num_classes)
    minority_cls = int(counts.argmin()) if highlight_minority else -1

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    for c in range(num_classes):
        mask  = y == c
        color = _PALETTE[c % len(_PALETTE)]
        edge  = "red" if c == minority_cls else "none"
        lw    = 0.8  if c == minority_cls else 0.0
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=color, s=18, alpha=0.75,
            edgecolors=edge, linewidths=lw,
            label=f"{class_names[c]} (n={counts[c]})",
            zorder=3 if c == minority_cls else 2,
        )

    # Annotate class centroids
    for c in range(num_classes):
        mask = y == c
        if mask.sum() == 0:
            continue
        cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
        ax.text(cx, cy, class_names[c], fontsize=7, ha="center",
                fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.5, lw=0))

    title_str = title if epoch is None else f"{title} (epoch {epoch})"
    ax.set_title(title_str, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1", fontsize=11)
    ax.set_ylabel("t-SNE dim 2", fontsize=11)
    ax.legend(loc="best", fontsize=8, framealpha=0.8, ncol=2)
    ax.grid(alpha=0.2)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  [t-SNE] Saved → {save_path}")


def plot_tsne_evolution(
    features_by_epoch: dict,      # {epoch_int: (features_tensor, labels_tensor)}
    class_names: Optional[List[str]] = None,
    save_dir: str = "./results/tsne_evolution",
    perplexity: int = 30,
    pca_components: int = 50,
    dpi: int = 130,
) -> None:
    """
    Generate one t-SNE plot per epoch to show collapse progression.

    Parameters
    ----------
    features_by_epoch : dict mapping epoch → (features, labels)
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for epoch, (feats, lbls) in sorted(features_by_epoch.items()):
        save_path = str(Path(save_dir) / f"tsne_epoch{epoch:04d}.png")
        plot_tsne(
            feats, lbls,
            class_names=class_names,
            save_path=save_path,
            title="Feature Geometry Evolution",
            perplexity=perplexity,
            pca_components=pca_components,
            epoch=epoch,
            dpi=dpi,
        )
    print(f"  [t-SNE Evolution] {len(features_by_epoch)} frames saved to {save_dir}")
