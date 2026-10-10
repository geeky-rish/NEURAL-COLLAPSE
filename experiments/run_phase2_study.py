"""
experiments/run_phase2_study.py
─────────────────────────────────────────────────────────────────────────────
Phase-2 intervention sweep — "Can we fix it?"

Evaluates 5 interventions at imbalance_ratio=10 (where minority damage is
measurable but the model has not fully collapsed):

  1. baseline       — standard CE, no rebalancing  (control re-run at r=10)
  2. weighted_ce    — class-frequency weighted CE
  3. focal          — focal loss γ=2 + auto class-weight alpha
  4. oversampling   — WeightedRandomSampler (uniform class sampling)
  5. etf_nc_reg     — ETF head + NC collapse regularizer (geometry-aware)

Results are saved to ./results/phase2/  to avoid colliding with Phase-1.

Scientific Goal
---------------
  Identify which (if any) intervention yields:
  • Lower NC1 (healthier within-class geometry)
  • Higher Melanoma recall (clinically important minority class)
  • Better Macro F1 (balanced multi-class performance)

Protocol Tiers
--------------
  Tier-1 (default)  Single seed=42, Tier-1 fast_dev_batches.
                    6 runs, ~4-5 h on CPU.
  Tier-2            Same but full dataset (--fast-dev-batches 0).
  Tier-3            Multi-seed [42,7,123] for top-2 methods.

Usage
-----
  # Tier-1 (default, all 5 interventions at r=10):
  python -m experiments.run_phase2_study

  # Dry-run:
  python -m experiments.run_phase2_study --dry-run

  # Only specific methods:
  python -m experiments.run_phase2_study --methods weighted_ce focal

  # Different ratio:
  python -m experiments.run_phase2_study --ratio 50

  # Tier-3 multi-seed:
  python -m experiments.run_phase2_study --seeds 42 7 123
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ── Phase-2 method definitions ────────────────────────────────────────────────
# Each entry maps a short name → the train.py flags needed to invoke it.
#
# sampling_strategy field
# -----------------------
# Controls the DataLoader sampler passed to get_medical_dataloaders via the
# `sampling.strategy` config key.  Two valid values here:
#
#   "weighted"  — WeightedRandomSampler active (uniform class distribution)
#   "none"      — plain shuffle; no sampler rebalancing
#
# IMPORTANT: weighted_ce and focal MUST use "none" to avoid double-rebalancing.
# The loss function already applies inverse-frequency weights; activating the
# sampler simultaneously multiplies the minority gradient by ~ratio^2, which
# destroys representation geometry (observed in initial Phase-2 run: NC1=14.47,
# NC4=0.78, ROC-AUC=0.47 — worse than random initialization).
PHASE2_METHOD_CONFIGS: Dict[str, Dict] = {
    "baseline": {
        "method_flag":       "baseline",
        "head":              "linear",
        "nc_reg":            False,
        "sampling_strategy": "weighted",   # sampler only; CE loss is unweighted
        "description":       "Standard CE + WeightedRandomSampler (control re-run at r=10)",
    },
    "weighted_ce": {
        "method_flag":       "weighted_ce",
        "head":              "linear",
        "nc_reg":            False,
        "sampling_strategy": "none",       # MUST be none: loss already rebalances
        "description":       "Weighted CE only — sampler disabled to isolate loss effect",
    },
    "focal": {
        "method_flag":       "focal",
        "head":              "linear",
        "nc_reg":            False,
        "sampling_strategy": "none",       # MUST be none: focal+alpha already rebalances
        "description":       "Focal loss (γ=2, alpha=1/freq) only — sampler disabled",
    },
    "oversampling": {
        "method_flag":       "oversampling",
        "head":              "linear",
        "nc_reg":            False,
        "sampling_strategy": "weighted",   # sampler IS the intervention; plain CE loss
        "description":       "WeightedRandomSampler only — CE loss is unweighted",
    },
    "etf_nc_reg": {
        "method_flag":       "etf",
        "head":              "etf",
        "nc_reg":            True,
        "sampling_strategy": "weighted",   # ETF+NC-reg; sampler provides mild rebalancing
        "description":       "ETF head + NC collapse regularizer (geometry-aware)",
    },
    # ── Step-0 experiment: isolate batch-composition effect on NC-reg ─────────
    # Hypothesis: WeightedRandomSampler may not guarantee per-class batch presence
    # for very small tail classes (DF: 83, Vascular: 115 at r=10). ClassBalanced-
    # Sampler draws exactly num_samples_each from each class per epoch, ensuring
    # every class has guaranteed batch participation.
    # Change vs etf_nc_reg: ONLY sampling_strategy (balanced vs weighted).
    # Everything else held constant: ETF head, NC-reg on, CE loss, same ratio/seed.
    "etf_nc_reg_balanced": {
        "method_flag":       "etf",
        "head":              "etf",
        "nc_reg":            True,
        "sampling_strategy": "balanced",   # ClassBalancedSampler — guaranteed per-class representation
        "description":       "ETF+NC-reg + ClassBalancedSampler (Step-0: batch composition test)",
    },
}

DATASET  = "ham10000"
PHASE2_RESULTS_DIR = ROOT / "results" / "phase2"

_DEFAULT_RATIO          = 10
_DEFAULT_SEEDS          = [42]
_DEFAULT_EPOCHS         = 15
_DEFAULT_FAST_DEV_BATCHES = 75
_DEFAULT_BATCH_SIZE     = 16
_DEFAULT_NUM_WORKERS    = 0

LOG_DIR     = ROOT / "study_logs" / "phase2"
SUMMARY_CSV = PHASE2_RESULTS_DIR / "phase2_summary.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_tag(method_key: str, ratio: int, seed: int) -> str:
    """
    Unique run identifier.
    Uses the short method key (e.g. 'etf_nc_reg') not the train.py method flag,
    so Phase-2 results never collide with Phase-1 output directories.
    """
    return f"{method_key}_{DATASET}_r{ratio}_s{seed}"


def results_path(method_key: str, ratio: int, seed: int) -> Path:
    return PHASE2_RESULTS_DIR / run_tag(method_key, ratio, seed)


def is_complete(method_key: str, ratio: int, seed: int) -> bool:
    return (results_path(method_key, ratio, seed) / "best_results.json").is_file()


def log_path(method_key: str, ratio: int, seed: int) -> Path:
    return LOG_DIR / f"{method_key}_r{ratio}_s{seed}.log"


def _sep(char: str = "─", width: int = 64) -> str:
    return char * width


def _fmt_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _print_log_tail(lp: Path, n: int = 20) -> None:
    try:
        lines = lp.read_text(errors="replace").splitlines()
        tail  = lines[-n:] if len(lines) >= n else lines
        print(f"  {'─'*56}")
        print(f"  Last {len(tail)} lines of {lp.name}:")
        for line in tail:
            print(f"    {line}")
        print(f"  {'─'*56}")
    except Exception as exc:
        print(f"  [could not read log: {exc}]")


# ─────────────────────────────────────────────────────────────────────────────
# Run a single experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_single(
    method_key: str,
    ratio: int,
    seed: int,
    epochs: int,
    num_workers: int      = _DEFAULT_NUM_WORKERS,
    batch_size: int       = _DEFAULT_BATCH_SIZE,
    fast_dev_batches: int = _DEFAULT_FAST_DEV_BATCHES,
    dry_run: bool         = False,
) -> bool:
    cfg  = PHASE2_METHOD_CONFIGS[method_key]
    tag  = run_tag(method_key, ratio, seed)
    lp   = log_path(method_key, ratio, seed)
    lp.parent.mkdir(parents=True, exist_ok=True)

    nc_reg_flag       = "true" if cfg["nc_reg"] else "false"
    sampling_strategy = cfg["sampling_strategy"]

    cmd = [
        sys.executable, "-u",
        str(ROOT / "train.py"),
        "--method", cfg["method_flag"],
        "--override",
            f"dataset.name={DATASET}",
            f"dataset.imbalance_ratio={ratio}",
            f"seed={seed}",
            f"model.head={cfg['head']}",
            f"training.epochs={epochs}",
            f"training.num_workers={num_workers}",
            f"training.batch_size={batch_size}",
            "tracking.tensorboard=false",
            "visualization.enabled=false",
            "logging.log_every_n_epochs=1",
            f"debug.fast_dev_batches={fast_dev_batches}",
            f"nc_regularization.enabled={nc_reg_flag}",
            # Sampling strategy — per-method override to prevent double-rebalancing.
            # weighted_ce and focal use "none": their loss already rebalances via weights.
            # baseline, oversampling, etf_nc_reg use "weighted": sampler is their tool.
            f"sampling.strategy={sampling_strategy}",
            # Force the output directory to use the full study-level key, not just
            # the method flag. Without this, all ETF variants (etf_nc_reg,
            # etf_nc_reg_balanced, etc.) collide on etf_ham10000_r{ratio}_s{seed}/.
            f"logging.run_tag={tag}",
            # Route output to phase2 subdirectory — avoids Phase-1 collisions
            f"logging.results_dir={PHASE2_RESULTS_DIR}",
            f"logging.checkpoint_dir={ROOT / 'checkpoints' / 'phase2'}",
    ]



    print(f"\n  CMD: {' '.join(cmd)}")

    if dry_run:
        print(f"  [DRY RUN] → {results_path(method_key, ratio, seed)}/")
        print(f"             log → {lp}")
        return True

    t0 = time.time()
    child_env = {
        **__import__("os").environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
    }
    with open(lp, "w") as log_fh:
        proc = subprocess.run(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT,
            cwd=str(ROOT), env=child_env,
        )

    elapsed = time.time() - t0
    success = (proc.returncode == 0)

    if success:
        print(f"  [OK]  {tag}  ({_fmt_time(elapsed)})  → {results_path(method_key, ratio, seed)}/")
    else:
        print(f"  [FAIL] {tag}  exit={proc.returncode}  ({_fmt_time(elapsed)})  log → {lp}")
        _print_log_tail(lp)

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"  [cooldown] 5 s …")
    time.sleep(5)
    return success


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_metrics(method_key: str, ratio: int, seed: int) -> Optional[Dict]:
    rdir    = results_path(method_key, ratio, seed)
    br_path = rdir / "best_results.json"
    cp_path = rdir / "class_metrics.csv"

    if not br_path.is_file():
        return None

    br = json.loads(br_path.read_text())
    mel_recall = float("nan")
    if cp_path.is_file():
        try:
            df  = pd.read_csv(cp_path)
            row = df[df["class"].str.lower() == "melanoma"]
            if not row.empty:
                mel_recall = float(row["recall"].iloc[0])
        except Exception:
            pass

    return {
        "method_key":      method_key,
        "description":     PHASE2_METHOD_CONFIGS[method_key]["description"],
        "imbalance_ratio": ratio,
        "seed":            seed,
        "run_tag":         run_tag(method_key, ratio, seed),
        "best_val_acc":    br.get("best_val_acc",  float("nan")),
        "macro_f1":        br.get("macro_f1",      float("nan")),
        "roc_auc":         br.get("roc_auc",       float("nan")),
        "nc1":             br.get("nc1",            float("nan")),
        "nc2":             br.get("nc2",            float("nan")),
        "nc3":             br.get("nc3",            float("nan")),
        "nc4":             br.get("nc4",            float("nan")),
        "melanoma_recall": mel_recall,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary + Plots
# ─────────────────────────────────────────────────────────────────────────────

def generate_phase2_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart comparisons across interventions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARNING] matplotlib not available")
        return

    if df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                          "axes.spines.right": False})

    # Colour per method
    METHOD_COLORS = {
        "baseline":    "#888888",
        "weighted_ce": "#2196F3",
        "focal":       "#FF9800",
        "oversampling":"#4CAF50",
        "etf_nc_reg":  "#E63946",
    }

    metrics_to_plot = [
        ("macro_f1",        "Macro F1"),
        ("melanoma_recall", "Melanoma Recall (Sensitivity)"),
        ("nc1",             "NC1 (Within-Class Scatter)"),
        ("nc4",             "NC4 (NCC Disagreement)"),
    ]

    for col, label in metrics_to_plot:
        if col not in df.columns:
            continue
        # Aggregate mean across seeds
        grp = df.groupby("method_key")[col].mean().reset_index()
        grp = grp.sort_values(col, ascending=(col in ("nc1", "nc4")))

        fig, ax = plt.subplots(figsize=(8, 4))
        colors  = [METHOD_COLORS.get(m, "#666") for m in grp["method_key"]]
        bars    = ax.barh(grp["method_key"], grp[col], color=colors, height=0.55,
                          edgecolor="white", alpha=0.88)
        for bar, val in zip(bars, grp[col]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)
        ax.set_xlabel(label, fontsize=11)
        ax.set_title(f"Phase-2 Interventions — {label}\n(HAM10000, ratio={df['imbalance_ratio'].iloc[0]}, seed={df['seed'].iloc[0]})",
                     fontsize=11)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)
        fig.tight_layout()
        out_path = out_dir / f"phase2_{col}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  [Plot] {out_path.relative_to(ROOT)}")


def print_final_summary(
    n_done: int, n_failed: int, n_skipped: int, total: int,
    df: pd.DataFrame, wall_time: float,
) -> None:
    print("\n" + "=" * 64)
    print("  Phase-2 Study Complete")
    print("=" * 64)
    print(f"  Total runs : {total}")
    print(f"  Completed  : {n_done}")
    print(f"  Skipped    : {n_skipped}")
    print(f"  Failed     : {n_failed}")
    print(f"  Wall time  : {_fmt_time(wall_time)}")
    print(_sep())

    if not df.empty:
        cols = ["method_key", "macro_f1", "melanoma_recall", "nc1", "nc4", "roc_auc"]
        existing = [c for c in cols if c in df.columns]
        print("\n  Results summary:")
        print(df[existing].sort_values("macro_f1", ascending=False).to_string(index=False))

    print(_sep())
    print(f"  Summary CSV : {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"  Results dir : {PHASE2_RESULTS_DIR.relative_to(ROOT)}/")
    print("=" * 64 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-2 intervention sweep: which method best preserves "
                    "minority-class geometry under imbalance?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ratio",   type=int, default=_DEFAULT_RATIO,
                   help=f"Imbalance ratio to test (default: {_DEFAULT_RATIO})")
    p.add_argument("--seeds",   nargs="+", type=int, default=_DEFAULT_SEEDS,
                   help="Random seeds (default: [42])")
    p.add_argument("--methods", nargs="+", default=list(PHASE2_METHOD_CONFIGS.keys()),
                   choices=list(PHASE2_METHOD_CONFIGS.keys()),
                   help="Subset of interventions to run")
    p.add_argument("--epochs",           type=int, default=_DEFAULT_EPOCHS)
    p.add_argument("--fast-dev-batches", type=int, default=_DEFAULT_FAST_DEV_BATCHES)
    p.add_argument("--batch-size",       type=int, default=_DEFAULT_BATCH_SIZE)
    p.add_argument("--num-workers",      type=int, default=_DEFAULT_NUM_WORKERS)
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    methods = args.methods
    ratio   = args.ratio
    seeds   = args.seeds
    total   = len(methods) * len(seeds)

    print("═" * 64)
    print(f"  NC-MedAI  —  Phase-2 Intervention Study")
    print("═" * 64)
    print(f"  Methods         : {methods}")
    print(f"  Imbalance ratio : r={ratio}")
    print(f"  Seeds           : {seeds}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  fast_dev_batches: {args.fast_dev_batches}")
    print(f"  Total runs      : {total}")
    print(f"  Results dir     : {PHASE2_RESULTS_DIR.relative_to(ROOT)}/")
    print(_sep())
    print("  Intervention descriptions:")
    for m in methods:
        print(f"    {m:<15} : {PHASE2_METHOD_CONFIGS[m]['description']}")
    print(_sep())

    if args.dry_run:
        print("\n  [DRY RUN] Planned runs:\n")
        print(f"  {'Method':<15}  {'Tag':<40}  {'Sampler':<10}  Status")
        print(f"  {'─'*15}  {'─'*40}  {'─'*10}  {'─'*6}")
        for method_key in methods:
            cfg  = PHASE2_METHOD_CONFIGS[method_key]
            for seed in seeds:
                tag  = run_tag(method_key, ratio, seed)
                done = is_complete(method_key, ratio, seed)
                sampler_info = cfg["sampling_strategy"]
                status = "SKIP" if done else "RUN "
                print(f"  [{status}]  {method_key:<15}  {tag:<40}  "
                      f"sampler={sampler_info:<8}  "
                      f"loss_rebal={'yes' if method_key in ('weighted_ce','focal') else 'no '}")
        print()
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PHASE2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_done = n_failed = n_skipped = run_num = 0
    study_start = time.time()
    all_rows: List[Dict] = []

    for method_key in methods:
        for seed in seeds:
            run_num += 1
            tag = run_tag(method_key, ratio, seed)

            print(f"\n{_sep()}")
            print(f"  Run {run_num}/{total}  │  {tag}")
            print(_sep())

            if is_complete(method_key, ratio, seed):
                print(f"  [SKIP] already complete.")
                n_skipped += 1
                row = extract_metrics(method_key, ratio, seed)
                if row:
                    all_rows.append(row)
                continue

            elapsed_so_far = time.time() - study_start
            if run_num > 1:
                avg = elapsed_so_far / (n_done + n_failed + 1e-9)
                eta = avg * (total - run_num + 1)
                print(f"  Elapsed: {_fmt_time(elapsed_so_far)}  │  ETA: ~{_fmt_time(eta)}")

            try:
                ok = run_single(
                    method_key, ratio, seed,
                    epochs=args.epochs,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    fast_dev_batches=args.fast_dev_batches,
                )
                if ok:
                    n_done += 1
                    row = extract_metrics(method_key, ratio, seed)
                    if row:
                        all_rows.append(row)
                        print(f"  Metrics: F1={row['macro_f1']:.4f}  "
                              f"NC1={row['nc1']:.4f}  "
                              f"Melanoma={row['melanoma_recall']:.4f}")
                else:
                    n_failed += 1
            except Exception as exc:
                print(f"  [ERROR] {tag}: {exc}")
                n_failed += 1

    wall_time = time.time() - study_start

    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    if not df.empty:
        df.to_csv(SUMMARY_CSV, index=False)
        print(f"\n  [CSV] {SUMMARY_CSV.relative_to(ROOT)}  ({len(df)} rows)")

    if not args.skip_plots and not df.empty:
        plots_dir = PHASE2_RESULTS_DIR / "phase2_plots"
        print("  Generating Phase-2 comparison plots …")
        generate_phase2_plots(df, plots_dir)

    print_final_summary(n_done, n_failed, n_skipped, total, df, wall_time)


if __name__ == "__main__":
    main()
