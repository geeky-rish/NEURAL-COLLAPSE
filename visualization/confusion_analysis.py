"""
visualization/confusion_analysis.py
─────────────────────────────────────────────────────────────────────────────
Confusion matrix and per-class recall bar charts.

Medical AI focus: minority disease recall is the critical failure mode.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DPI = 150


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: str = "./results/confusion_matrix.png",
    title: str = "Confusion Matrix",
    normalise: bool = True,
    dpi: int = _DPI,
) -> None:
    """
    Annotated confusion matrix heatmap (raw counts + normalised).

    Parameters
    ----------
    cm          : (C, C) integer numpy array from sklearn.metrics.confusion_matrix
    class_names : list[str]
    save_path   : output PNG path
    title       : plot title
    normalise   : if True, also show row-normalised version side-by-side
    """
    C = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {c}" for c in range(C)]

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    ncols = 2 if normalise else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6), dpi=dpi)
    if ncols == 1:
        axes = [axes]

    datasets = [(cm, "Count", "Blues"), (cm_norm, "Row-Normalised", "RdYlGn")]

    for ax, (data, subtitle, cmap) in zip(axes, datasets[:ncols]):
        vmax = data.max()
        im   = ax.imshow(data, interpolation="nearest", cmap=cmap,
                         vmin=0, vmax=vmax)
        ax.figure.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(C))
        ax.set_yticks(range(C))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_ylabel("True label",      fontsize=10)
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_title(subtitle, fontsize=11)

        thresh = vmax / 2.0
        for i in range(C):
            for j in range(C):
                val  = data[i, j]
                fmt  = f"{val:.2f}" if subtitle == "Row-Normalised" else f"{int(val)}"
                color = "white" if val > thresh else "black"
                ax.text(j, i, fmt, ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  [Confusion Matrix] Saved → {save_path}")


def plot_per_class_recall(
    sensitivity_dict: Dict[str, float],
    save_path: str = "./results/per_class_recall.png",
    title: str = "Per-Class Sensitivity (Recall)",
    dpi: int = _DPI,
) -> None:
    """
    Horizontal bar chart of per-class sensitivity.

    Bars are coloured red if sensitivity < 0.5 (clinically dangerous for
    disease detection), yellow if < 0.7, green otherwise.

    Parameters
    ----------
    sensitivity_dict : {class_name: sensitivity_float}
    """
    names  = list(sensitivity_dict.keys())
    values = np.array(list(sensitivity_dict.values()))

    colors = []
    for v in values:
        if v < 0.5:
            colors.append("#C44E52")    # red — dangerous
        elif v < 0.7:
            colors.append("#DD8452")    # orange — concerning
        else:
            colors.append("#55A868")    # green — acceptable

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.55)), dpi=dpi)
    y_pos   = np.arange(len(names))

    bars = ax.barh(y_pos, values, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Sensitivity (Recall)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(0.5, color="red",    linestyle="--", linewidth=1.2,
               label="Clinical threshold (0.5)")
    ax.axvline(0.7, color="orange", linestyle="--", linewidth=1.2,
               label="Recommended threshold (0.7)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # Annotate values
    for bar, val in zip(bars, values):
        ax.text(min(val + 0.01, 1.02), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  [Per-Class Recall] Saved → {save_path}")
