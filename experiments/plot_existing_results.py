"""
experiments/plot_existing_results.py
─────────────────────────────────────────────────────────────────────────────
Generate diagnostic plots from already-completed 30-epoch pilot runs:
  • ETF head   (etf_ham10000_s42)
  • Linear     (baseline_ham10000_s42)

Both used nc_regularization=ON (pre-fix bug), so these are labelled as
'Pilot (NC-reg active)' to be scientifically honest.

Plots produced
--------------
  results/pilot_plots/nc1_vs_epoch.png          NC1 over 30 epochs, both methods
  results/pilot_plots/nc4_vs_epoch.png          NC4 over 30 epochs, both methods
  results/pilot_plots/nc2_vs_epoch.png          NC2 (ETF deviation) over 30 epochs
  results/pilot_plots/per_class_recall.png      Side-by-side recall bar chart
  results/pilot_plots/val_acc_vs_epoch.png      Val accuracy curves
  results/pilot_plots/summary_table.png         Key metrics summary figure

Usage
-----
  python -m experiments.plot_existing_results
  python -m experiments.plot_existing_results --results-dir results
  python -m experiments.plot_existing_results --out-dir results/my_plots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "etf":      {"color": "#E63946", "marker": "o", "ls": "-",  "label": "ETF Head (pilot)"},
    "baseline": {"color": "#457B9D", "marker": "s", "ls": "--", "label": "Linear Baseline (pilot)"},
}
NOTE = "Note: pilot runs had NC-reg active (now fixed for Phase-1)"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _load_nc(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "nc_metrics.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p)


def _load_best(run_dir: Path) -> dict | None:
    p = run_dir / "best_results.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _load_class_metrics(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "class_metrics.csv"
    if not p.is_file():
        return None
    return pd.read_csv(p)


def _load_val_acc(run_dir: Path) -> list[float] | None:
    p = run_dir / "training_summary.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    return d.get("val_acc")


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_nc_metric_vs_epoch(
    data: dict[str, pd.DataFrame],
    col: str,
    ylabel: str,
    title: str,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, df in data.items():
        if col not in df.columns:
            continue
        s = PALETTE[key]
        ax.plot(df["epoch"], df[col],
                color=s["color"], marker=s["marker"], markersize=4,
                linestyle=s["ls"], linewidth=1.8, label=s["label"], alpha=0.9)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.text(0.5, -0.02, NOTE, ha="center", fontsize=8, color="gray", style="italic")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {out.relative_to(ROOT)}")


def plot_val_acc_vs_epoch(
    data: dict[str, list[float]],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, vals in data.items():
        if not vals:
            continue
        s = PALETTE[key]
        ax.plot(range(1, len(vals) + 1), vals,
                color=s["color"], marker=s["marker"], markersize=3,
                linestyle=s["ls"], linewidth=1.8, label=s["label"], alpha=0.9)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Validation Accuracy vs Epoch\n(HAM10000 natural distribution, seed=42)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.text(0.5, -0.02, NOTE, ha="center", fontsize=8, color="gray", style="italic")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {out.relative_to(ROOT)}")


def plot_per_class_recall(
    class_data: dict[str, pd.DataFrame],
    out: Path,
) -> None:
    """Grouped horizontal bar chart: recall per HAM class, ETF vs baseline."""
    all_dfs = {}
    for key, df in class_data.items():
        if df is not None and "recall" in df.columns:
            all_dfs[key] = df.set_index("class")["recall"]

    if not all_dfs:
        print("  [SKIP] No class metric data for per-class recall plot")
        return

    # Align on common classes
    combined = pd.DataFrame(all_dfs)
    classes = combined.index.tolist()
    n = len(classes)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (key, series) in enumerate(combined.items()):
        s = PALETTE.get(key, {"color": "#888", "label": key})
        offset = (i - (len(combined) - 1) / 2) * width
        bars = ax.bar(x + offset, series.values * 100, width,
                      label=s["label"], color=s["color"], alpha=0.82, edgecolor="white")
        # Label bars with value if > 1%
        for bar, val in zip(bars, series.values * 100):
            if val > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=7.5,
                        color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Recall / Sensitivity (%)", fontsize=12)
    ax.set_title("Per-Class Recall: ETF Head vs Linear Baseline\n(HAM10000, seed=42, natural distribution)",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.text(0.5, -0.02, NOTE, ha="center", fontsize=8, color="gray", style="italic")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {out.relative_to(ROOT)}")


def plot_summary_table(
    best_data: dict[str, dict],
    out: Path,
) -> None:
    """Render a summary metrics table as a matplotlib figure."""
    if not best_data:
        return

    rows   = []
    cols   = ["Method", "Val Acc (%)", "Macro F1", "ROC-AUC", "NC1", "NC2", "NC3", "NC4"]
    labels = {
        "etf":      "ETF Head",
        "baseline": "Linear Baseline",
    }
    for key, d in best_data.items():
        rows.append([
            labels.get(key, key),
            f"{d.get('best_val_acc', float('nan')):.1f}",
            f"{d.get('macro_f1', float('nan')):.4f}",
            f"{d.get('roc_auc', float('nan')):.4f}",
            f"{d.get('nc1', float('nan')):.3f}",
            f"{d.get('nc2', float('nan')):.4f}",
            f"{d.get('nc3', float('nan')):.4f}",
            f"{d.get('nc4', float('nan')):.4f}",
        ])

    fig, ax = plt.subplots(figsize=(10, 2 + 0.6 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 2.2)

    # Style header
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2B3A67")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Alternating row colours
    for i in range(1, len(rows) + 1):
        for j in range(len(cols)):
            tbl[i, j].set_facecolor("#EAF2FB" if i % 2 == 0 else "white")

    ax.set_title("Pilot Summary — HAM10000, seed=42, natural distribution\n" + NOTE,
                 fontsize=10, color="#333", pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {out.relative_to(ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Plot pilot run results (no training needed).")
    ap.add_argument("--results-dir", default=str(ROOT / "results"),
                    help="Path to results directory")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "pilot_plots"),
                    help="Where to save plots")
    args = ap.parse_args()

    res_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate runs
    RUN_DIRS = {
        "etf":      res_dir / "etf_ham10000_s42",
        "baseline": res_dir / "baseline_ham10000_s42",
    }
    missing = [k for k, d in RUN_DIRS.items() if not d.is_dir()]
    if missing:
        print(f"[WARNING] Missing run directories: {missing}")
        print("  These are the old 30-epoch pilot runs. Train them first.")

    present = {k: d for k, d in RUN_DIRS.items() if d.is_dir()}
    if not present:
        print("[ERROR] No run directories found. Exiting.")
        return

    # Load data
    nc_data   = {k: df for k, d in present.items() if (df := _load_nc(d)) is not None}
    best_data = {k: d for k, rd in present.items() if (d := _load_best(rd)) is not None}
    cls_data  = {k: df for k, d in present.items() if (df := _load_class_metrics(d)) is not None}
    acc_data  = {k: v for k, d in present.items() if (v := _load_val_acc(d)) is not None}

    print(f"\nGenerating plots -> {out_dir.relative_to(ROOT)}/")
    print(f"  NC data available: {list(nc_data.keys())}")
    print(f"  Class data available: {list(cls_data.keys())}\n")

    # NC1 vs epoch
    plot_nc_metric_vs_epoch(
        nc_data, col="nc1",
        ylabel="NC1 (Within-Class Scatter)",
        title="NC1 vs Epoch — ETF vs Linear Baseline\n(HAM10000, seed=42, natural distribution)",
        out=out_dir / "nc1_vs_epoch.png",
    )

    # NC4 vs epoch
    plot_nc_metric_vs_epoch(
        nc_data, col="nc4",
        ylabel="NC4 (NCC Disagreement Rate)",
        title="NC4 vs Epoch — ETF vs Linear Baseline\n(HAM10000, seed=42, natural distribution)",
        out=out_dir / "nc4_vs_epoch.png",
    )

    # NC2 vs epoch
    plot_nc_metric_vs_epoch(
        nc_data, col="nc2",
        ylabel="NC2 (ETF Cosine Deviation)",
        title="NC2 vs Epoch — ETF vs Linear Baseline\n(HAM10000, seed=42, natural distribution)",
        out=out_dir / "nc2_vs_epoch.png",
    )

    # NC3 vs epoch
    plot_nc_metric_vs_epoch(
        nc_data, col="nc3",
        ylabel="NC3 (W–μ Alignment)",
        title="NC3 vs Epoch — ETF vs Linear Baseline\n(HAM10000, seed=42, natural distribution)",
        out=out_dir / "nc3_vs_epoch.png",
    )

    # Validation accuracy curves
    plot_val_acc_vs_epoch(acc_data, out=out_dir / "val_acc_vs_epoch.png")

    # Per-class recall
    plot_per_class_recall(cls_data, out=out_dir / "per_class_recall_comparison.png")

    # Summary table
    plot_summary_table(best_data, out=out_dir / "summary_table.png")

    print(f"\n  All plots saved to {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
