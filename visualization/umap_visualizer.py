"""
visualization/umap_visualizer.py
─────────────────────────────────────────────────────────────────────────────
UMAP-based feature visualisation.

UMAP (McInnes et al., 2018) is faster than t-SNE on large datasets and
better preserves global structure — making it preferable for visualising
the overall ETF geometry and inter-class separability.

Requires: pip install umap-learn

Falls back gracefully (with a warning) if umap-learn is not installed.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

try:
    import umap as umap_lib
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False


def plot_umap(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: str = "./results/umap.png",
    title: str = "UMAP Feature Visualisation",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    highlight_minority: bool = True,
    epoch: Optional[int] = None,
    dpi: int = 150,
) -> None:
    """
    Compute and plot a 2-D UMAP embedding of *features*.

    Parameters
    ----------
    features          : (N, D) float tensor
    labels            : (N,)   long tensor
    class_names       : list[str]
    save_path         : output PNG path
    title             : plot title
    n_neighbors       : UMAP n_neighbors (controls local/global balance)
    min_dist          : UMAP min_dist (controls cluster compactness)
    highlight_minority: outline smallest class in red
    epoch             : if set, appended to title
    dpi               : figure DPI
    """
    if not _UMAP_AVAILABLE:
        warnings.warn(
            "umap-learn is not installed. Skipping UMAP plot.\n"
            "Install with: pip install umap-learn",
            UserWarning,
        )
        return

    X = features.detach().cpu().numpy().astype(np.float32)
    y = labels.detach().cpu().numpy()

    num_classes = int(y.max()) + 1
    if class_names is None:
        class_names = [f"Class {c}" for c in range(num_classes)]

    # ── UMAP reduction ────────────────────────────────────────────────────────
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        verbose=False,
    )
    X_2d = reducer.fit_transform(X)

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
    ax.set_xlabel("UMAP dim 1", fontsize=11)
    ax.set_ylabel("UMAP dim 2", fontsize=11)
    ax.legend(loc="best", fontsize=8, framealpha=0.8, ncol=2)
    ax.grid(alpha=0.2)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  [UMAP] Saved → {save_path}")
