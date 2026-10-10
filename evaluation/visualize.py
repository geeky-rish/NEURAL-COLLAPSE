"""
evaluation/visualize.py
─────────────────────────────────────────────────────────────────────────────
Publication-quality matplotlib plots for the Neural Collapse study.

Functions
---------
plot_imbalance_sweep  — Accuracy / NC1 / NC2 vs imbalance ratio (baseline)
plot_method_comparison — Bar chart: methods × metrics at fixed ratio
plot_nc_scatter        — Scatter: NC1 vs Accuracy coloured by method
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Shared style
_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
_DPI    = 150


# ── Imbalance sweep ───────────────────────────────────────────────────────────

def plot_imbalance_sweep(
    df: pd.DataFrame,
    save_dir: str = "./results",
    prefix: str   = "baseline",
) -> None:
    """
    Plot Accuracy, NC1 (log-scale), and NC2 vs imbalance ratio.

    Parameters
    ----------
    df       : DataFrame with columns [ratio, acc_mean, nc1_mean, nc2_mean]
               (and optionally acc_std, nc1_std, nc2_std)
    save_dir : output directory
    prefix   : filename prefix
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ratios = df["ratio"].values

    metrics = [
        ("acc_mean", "acc_std",  "Accuracy (%)",   False, "Accuracy vs Imbalance Ratio"),
        ("nc1_mean", "nc1_std",  "NC1 (log scale)", True,  "NC1 vs Imbalance Ratio"),
        ("nc2_mean", "nc2_std",  "NC2",             False, "NC2 vs Imbalance Ratio"),
    ]

    for col_mean, col_std, ylabel, log_y, title in metrics:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=_DPI)

        mean_vals = df[col_mean].values
        ax.plot(ratios, mean_vals, marker="o", color=_COLORS[0], linewidth=2)

        # Error band if std column exists
        if col_std in df.columns:
            std_vals = df[col_std].values
            ax.fill_between(ratios,
                            mean_vals - std_vals,
                            mean_vals + std_vals,
                            alpha=0.2, color=_COLORS[0])

        ax.invert_xaxis()
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Imbalance Ratio (majority / minority)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        fname = Path(save_dir) / f"{prefix}_{col_mean}.png"
        fig.savefig(fname, dpi=_DPI)
        plt.close(fig)
        print(f"  Saved → {fname}")


# ── Method comparison bar chart ───────────────────────────────────────────────

def plot_method_comparison(
    df: pd.DataFrame,
    save_dir: str  = "./results",
    prefix: str    = "comparison",
) -> None:
    """
    Side-by-side bar chart for Accuracy, NC1, NC2 across imbalance methods.

    Parameters
    ----------
    df : DataFrame with columns [method, acc_mean, nc1_mean, nc2_mean]
         (std columns optional)
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    methods = df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.6

    specs = [
        ("acc_mean", "acc_std",  "Accuracy (%)",   "Accuracy by Method",   False),
        ("nc1_mean", "nc1_std",  "NC1 (log scale)", "NC1 by Method",       True),
        ("nc2_mean", "nc2_std",  "NC2",             "NC2 by Method",        False),
    ]

    for col_mean, col_std, ylabel, title, log_y in specs:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=_DPI)

        vals = df[col_mean].values
        errs = df[col_std].values if col_std in df.columns else None

        bars = ax.bar(x, vals, width=width, color=_COLORS[:len(methods)],
                      yerr=errs, capsize=5, error_kw={"linewidth": 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        if log_y:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        fname = Path(save_dir) / f"{prefix}_{col_mean}.png"
        fig.savefig(fname, dpi=_DPI)
        plt.close(fig)
        print(f"  Saved → {fname}")


# ── NC1 vs Accuracy scatter ───────────────────────────────────────────────────

def plot_nc_scatter(
    df: pd.DataFrame,
    save_dir: str  = "./results",
    prefix: str    = "scatter",
) -> None:
    """
    Scatter plot of NC1 vs Accuracy, one point per (method, ratio) pair.

    Parameters
    ----------
    df : DataFrame with columns [method, imbalance_ratio, acc_mean, nc1_mean]
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    methods = df["method"].unique().tolist()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=_DPI)
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        ax.scatter(
            sub["nc1_mean"],
            sub["acc_mean"],
            label=method,
            color=_COLORS[i % len(_COLORS)],
            s=80, alpha=0.85, edgecolors="white", linewidths=0.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("NC1 (within-class scatter, log scale)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("NC1 vs Accuracy across Methods", fontsize=14)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fname = Path(save_dir) / f"{prefix}_nc1_vs_acc.png"
    fig.savefig(fname, dpi=_DPI)
    plt.close(fig)
    print(f"  Saved → {fname}")
