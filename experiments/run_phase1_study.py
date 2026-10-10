"""
experiments/run_phase1_study.py
─────────────────────────────────────────────────────────────────────────────
Phase-1 automated experiment runner — Neural Collapse under Imbalance.

PROTOCOL TIERS
--------------
  Tier-1  CPU Exploratory Pilot  (default — runs tonight)
  ──────────────────────────────────────────────────────
  Methods          : etf, baseline (linear)
  Ratios           : 1, 10, 50
  Seeds            : 42
  Epochs           : 15
  fast_dev_batches : 75   (16 % of train per epoch; full val for NC metrics)
  batch_size       : 16
  num_workers      : 0
  Total runs       : 6
  Est. runtime     : ~4–5 hours on CPU
  Goal             : Directional NC geometry signal under imbalance.
                     Findings are pilot/exploratory, not fully converged.

  Tier-2  Weekend Full-batch Pilot  (--fast-dev-batches 0)
  ────────────────────────────────────────────────────────
  Same as Tier-1 but full dataset per epoch (~18 hours, 6 runs).

  Tier-3  Full Study  (--seeds 42 7 123)
  ──────────────────────────────────────
  Multi-seed, recommended on GPU hardware.

Usage
-----
  # Tier-1 exploratory (default):
  python -m experiments.run_phase1_study

  # Dry-run — show what would execute without running:
  python -m experiments.run_phase1_study --dry-run

  # Override any parameter:
  python -m experiments.run_phase1_study --epochs 10 --fast-dev-batches 0

  # Skip matplotlib (headless servers without display):
  python -m experiments.run_phase1_study --skip-plots
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
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── Project root (one level above this file) ─────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Study configuration ──────────────────────────────────────────────────────
# Tier-1 CPU exploratory defaults — change via CLI to run a larger study.
METHODS  = ["etf", "baseline"]
RATIOS   = [1, 10, 50]
SEEDS    = [42]           # single seed → 6 runs, ~4-5 h on CPU
DATASET  = "ham10000"

# Default Tier-1 hyper-parameters
_DEFAULT_EPOCHS          = 15
_DEFAULT_FAST_DEV_BATCHES = 75   # 75/469 = 16 % of train; full 24-batch val set intact
_DEFAULT_BATCH_SIZE      = 16
_DEFAULT_NUM_WORKERS     = 0

# ── Output locations ──────────────────────────────────────────────────────────
LOG_DIR      = ROOT / "study_logs" / "phase1"
RESULTS_DIR  = ROOT / "results"
PLOTS_DIR    = RESULTS_DIR / "phase1_plots"
SUMMARY_CSV  = RESULTS_DIR / "phase1_summary.csv"
GROUPED_CSV  = RESULTS_DIR / "phase1_grouped_summary.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_tag(method: str, ratio: int, seed: int) -> str:
    """Canonical run identifier — must match train.py's run_tag formula."""
    return f"{method}_{DATASET}_r{ratio}_s{seed}"


def results_path(method: str, ratio: int, seed: int) -> Path:
    return RESULTS_DIR / run_tag(method, ratio, seed)


def is_complete(method: str, ratio: int, seed: int) -> bool:
    """Return True iff best_results.json already exists for this run."""
    return (results_path(method, ratio, seed) / "best_results.json").is_file()


def log_path(method: str, ratio: int, seed: int) -> Path:
    return LOG_DIR / f"{method}_r{ratio}_s{seed}.log"


def _sep(char: str = "─", width: int = 64) -> str:
    return char * width


def _fmt_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _print_log_tail(log_path: Path, n: int = 20) -> None:
    """Print the last *n* lines of a log file to stdout (failure diagnostics)."""
    try:
        lines = log_path.read_text(errors="replace").splitlines()
        tail  = lines[-n:] if len(lines) >= n else lines
        print(f"  {'─'*56}")
        print(f"  Last {len(tail)} lines of {log_path.name}:")
        for line in tail:
            print(f"    {line}")
        print(f"  {'─'*56}")
    except Exception as exc:
        print(f"  [could not read log: {exc}]")


# ─────────────────────────────────────────────────────────────────────────────
# Run a single experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_single(
    method: str,
    ratio: int,
    seed: int,
    epochs: int,
    num_workers: int = _DEFAULT_NUM_WORKERS,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    fast_dev_batches: int = _DEFAULT_FAST_DEV_BATCHES,
    dry_run: bool = False,
) -> bool:
    """
    Launch one train.py call via subprocess.

    Returns True on success, False on failure.
    Automatically gc.collect()s and (if CUDA present) empties the cache
    after each run to reclaim memory for the next subprocess.

    num_workers=0 is the safe default for sequential overnight CPU runs:
    DataLoader worker processes each fork the parent's address space (~500 MB),
    causing OOM-kills when system RAM is under pressure.

    fast_dev_batches=75 limits training to 75/469 = 16 % of the HAM10000 train
    set per epoch while leaving the 24-batch val set fully intact for NC metrics.
    Set to 0 for Tier-2 full-dataset runs.
    """
    tag  = run_tag(method, ratio, seed)
    lp   = log_path(method, ratio, seed)
    lp.parent.mkdir(parents=True, exist_ok=True)

    # Map method → correct train.py flags
    # ETF  : --method etf   + model.head=etf
    # linear: --method baseline + model.head=linear (default)
    if method == "etf":
        method_flags = ["--method", "etf"]
        head_override = "model.head=etf"
    else:
        method_flags = ["--method", "baseline"]
        head_override = "model.head=linear"

    cmd = [
        sys.executable, "-u",           # -u = force unbuffered stdout/stderr
        str(ROOT / "train.py"),
        *method_flags,
        "--override",
            f"dataset.name={DATASET}",
            f"dataset.imbalance_ratio={ratio}",
            f"seed={seed}",
            head_override,
            f"training.epochs={epochs}",
            f"training.num_workers={num_workers}",
            f"training.batch_size={batch_size}",
            "tracking.tensorboard=false",
            "visualization.enabled=false",
            # Log every epoch so the file shows live progress (not every 5)
            "logging.log_every_n_epochs=1",
            # fast_dev_batches=0 means full dataset; >0 limits train batches
            # (val set has 24 batches, so fast_dev_batches=75 leaves it fully intact)
            f"debug.fast_dev_batches={fast_dev_batches}",
            # Phase-1 = clean comparison: no NC regularization on either method.
            # NC-reg is a Phase-2 intervention and must NOT be on by default.
            "nc_regularization.enabled=false",
    ]

    print(f"\n  CMD: {' '.join(cmd)}")

    if dry_run:
        print(f"  [DRY RUN] would write → {results_path(method, ratio, seed)}/")
        print(f"            log         → {lp}")
        return True

    t0 = time.time()
    # PYTHONUNBUFFERED=1 ensures every print() and logging call in the child
    # process is written to the log file immediately (no 8 KB block-buffer).
    # PYTHONFAULTHANDLER=1 dumps a traceback to stderr on SIGSEGV/SIGABRT
    # so OOM kills (exit=-9) leave a diagnostic in the log.
    child_env = {**__import__("os").environ, "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1"}
    with open(lp, "w") as log_fh:
        proc = subprocess.run(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            env=child_env,
        )

    elapsed = time.time() - t0
    success = (proc.returncode == 0)

    if success:
        print(f"  [OK]  {tag}  ({_fmt_time(elapsed)})  → {results_path(method, ratio, seed)}/")
    else:
        print(f"  [FAIL] {tag}  exit={proc.returncode}  ({_fmt_time(elapsed)})  log → {lp}")
        # Print last 20 log lines to help diagnose the failure
        _print_log_tail(lp, n=20)

    # ── Resource cleanup before next subprocess ─────────────────────────────
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Short cooldown to let the OS reclaim sockets / file descriptors
    print(f"  [cooldown] 5 s …")
    time.sleep(5)

    return success


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_metrics(method: str, ratio: int, seed: int) -> Optional[Dict]:
    """
    Read saved artifacts for one run and return a flat metric dict.
    Returns None if artifacts are missing.
    """
    rdir = results_path(method, ratio, seed)
    br_path  = rdir / "best_results.json"
    cls_path = rdir / "class_metrics.csv"

    if not br_path.is_file():
        return None

    br = json.loads(br_path.read_text())

    # Melanoma recall from class_metrics.csv
    mel_recall = float("nan")
    if cls_path.is_file():
        try:
            df_cls = pd.read_csv(cls_path)
            mel_row = df_cls[df_cls["class"].str.lower() == "melanoma"]
            if not mel_row.empty:
                mel_recall = float(mel_row["recall"].iloc[0])
        except Exception:
            pass

    return {
        "method":        method,
        "imbalance_ratio": ratio,
        "seed":          seed,
        "run_tag":       run_tag(method, ratio, seed),
        "best_val_acc":  br.get("best_val_acc",  float("nan")),
        "macro_f1":      br.get("macro_f1",       float("nan")),
        "roc_auc":       br.get("roc_auc",        float("nan")),
        "nc1":           br.get("nc1",            float("nan")),
        "nc2":           br.get("nc2",            float("nan")),
        "nc3":           br.get("nc3",            float("nan")),
        "nc4":           br.get("nc4",            float("nan")),
        "melanoma_recall": mel_recall,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["method", "imbalance_ratio", "seed"]).reset_index(drop=True)
    return df


def build_grouped_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std across seeds, grouped by method × imbalance_ratio."""
    if df.empty:
        return pd.DataFrame()

    metric_cols = ["best_val_acc", "macro_f1", "roc_auc",
                   "nc1", "nc2", "nc3", "nc4", "melanoma_recall"]
    existing = [c for c in metric_cols if c in df.columns]

    agg = df.groupby(["method", "imbalance_ratio"])[existing].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(grouped: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARNING] matplotlib not available — skipping plots")
        return

    if grouped.empty:
        print("  [WARNING] No data for plots — skipping")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("nc1_mean",           "nc1_std",           "NC1 (Within-class Scatter)",   "NC1"),
        ("nc4_mean",           "nc4_std",           "NC4 (NCC Disagreement Rate)",  "NC4"),
        ("macro_f1_mean",      "macro_f1_std",      "Macro F1",                     "macro_f1"),
        ("melanoma_recall_mean","melanoma_recall_std","Melanoma Recall (Sensitivity)","melanoma_recall"),
    ]

    METHOD_STYLE = {
        "etf":      {"color": "#E63946", "marker": "o", "linestyle": "-",  "label": "ETF Head"},
        "baseline": {"color": "#457B9D", "marker": "s", "linestyle": "--", "label": "Linear Baseline"},
    }

    ratios = sorted(grouped["imbalance_ratio"].unique())

    for mean_col, std_col, ylabel, fname in metrics:
        if mean_col not in grouped.columns:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))

        for method, style in METHOD_STYLE.items():
            sub = grouped[grouped["method"] == method].sort_values("imbalance_ratio")
            if sub.empty:
                continue
            x   = sub["imbalance_ratio"].values
            y   = sub[mean_col].values
            err = sub[std_col].values if std_col in sub.columns else None

            ax.plot(x, y,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=2, markersize=7,
                    label=style["label"])
            if err is not None:
                ax.fill_between(x, y - err, y + err,
                                alpha=0.15, color=style["color"])

        ax.set_xlabel("Imbalance Ratio (majority : minority)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel} vs Imbalance Ratio\n(HAM10000, ResNet-18, seeds 42/7/123)",
                     fontsize=11)
        ax.set_xticks(ratios)
        ax.set_xticklabels([f"r={r}" for r in ratios])
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        out = PLOTS_DIR / f"phase1_{fname}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  [Plot] Saved → {out.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary(
    n_done: int,
    n_failed: int,
    n_skipped: int,
    total: int,
    df: pd.DataFrame,
    grouped: pd.DataFrame,
    wall_time: float,
) -> None:
    print("\n" + "=" * 64)
    print("  Phase-1 Study Complete")
    print("=" * 64)
    print(f"  Total runs   : {total}")
    print(f"  Completed    : {n_done}")
    print(f"  Skipped      : {n_skipped}  (already had results)")
    print(f"  Failed       : {n_failed}")
    print(f"  Wall time    : {_fmt_time(wall_time)}")
    print(_sep())

    if not grouped.empty:
        for method in METHODS:
            g = grouped[grouped["method"] == method]
            if g.empty:
                continue
            print(f"\n  {method.upper()} — mean ± std across seeds")
            cols = ["imbalance_ratio", "macro_f1_mean", "macro_f1_std",
                    "melanoma_recall_mean", "melanoma_recall_std",
                    "nc1_mean", "nc1_std", "nc4_mean", "nc4_std"]
            existing = [c for c in cols if c in g.columns]
            print(g[existing].to_string(index=False))

        # Best configurations
        print(_sep())
        if "macro_f1_mean" in grouped.columns:
            for method in METHODS:
                g = grouped[grouped["method"] == method]
                if g.empty:
                    continue
                best = g.loc[g["macro_f1_mean"].idxmax()]
                print(f"  Best {method} (Macro F1):  "
                      f"ratio={int(best['imbalance_ratio'])}  "
                      f"F1={best['macro_f1_mean']:.4f}±{best.get('macro_f1_std', 0):.4f}  "
                      f"Melanoma_recall={best.get('melanoma_recall_mean', float('nan')):.4f}")

    print(_sep())
    print(f"  Summary CSV  : {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"  Grouped CSV  : {GROUPED_CSV.relative_to(ROOT)}")
    print(f"  Plots        : {PLOTS_DIR.relative_to(ROOT)}/")
    print(f"  Logs         : {LOG_DIR.relative_to(ROOT)}/")
    print("=" * 64 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase-1 NC-MedAI study: ETF vs Baseline × 3 imbalance ratios.\n"
            "Default: Tier-1 CPU exploratory pilot "
            "(6 runs, fast_dev_batches=75, ~4-5 h)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--epochs",           type=int, default=_DEFAULT_EPOCHS,
                   help=f"Training epochs per run (default: {_DEFAULT_EPOCHS})")
    p.add_argument("--fast-dev-batches", type=int, default=_DEFAULT_FAST_DEV_BATCHES,
                   help=(
                       f"Batches per training epoch (default: {_DEFAULT_FAST_DEV_BATCHES}). "
                       "0 = full dataset (Tier-2). "
                       "Val set (24 batches) is always fully used when this > 24."
                   ))
    p.add_argument("--num-workers", type=int, default=_DEFAULT_NUM_WORKERS,
                   help="DataLoader num_workers (default: 0). Use 0 for CPU stability — "
                        "forked workers each consume ~500 MB, causing OOM-kills.")
    p.add_argument("--batch-size",  type=int, default=_DEFAULT_BATCH_SIZE,
                   help=f"Training batch size (default: {_DEFAULT_BATCH_SIZE}). Lower = less peak RAM.")
    p.add_argument("--dry-run",    action="store_true",
                   help="Print plan without running anything")
    p.add_argument("--skip-plots", action="store_true",
                   help="Skip matplotlib plot generation")
    p.add_argument("--methods",    nargs="+", default=METHODS,
                   choices=METHODS, help="Subset of methods to run")
    p.add_argument("--ratios",     nargs="+", type=int, default=RATIOS,
                   help="Subset of imbalance ratios")
    p.add_argument("--seeds",      nargs="+", type=int, default=SEEDS,
                   help="Subset of seeds (default: [42] for Tier-1)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    methods = args.methods
    ratios  = args.ratios
    seeds   = args.seeds
    total   = len(methods) * len(ratios) * len(seeds)

    # ── Print study plan ──────────────────────────────────────────────────────
    fdb = args.fast_dev_batches
    protocol = "Tier-1 CPU Exploratory Pilot" if fdb > 0 else "Tier-2 Full-batch Pilot"
    if len(seeds) > 1:
        protocol = "Tier-3 Multi-seed Full Study"

    print(_sep("═"))
    print(f"  NC-MedAI  —  Phase-1 Study  [{protocol}]")
    print(_sep("═"))
    print(f"  Methods          : {methods}")
    print(f"  Ratios           : {ratios}")
    print(f"  Seeds            : {seeds}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  fast_dev_batches : {fdb}  "
          f"({'~16 % of train, full val intact' if 0 < fdb <= 75 else 'FULL dataset' if fdb == 0 else str(fdb) + ' batches'})")
    print(f"  batch_size       : {args.batch_size}")
    print(f"  num_workers      : {args.num_workers}")
    print(f"  Total runs       : {total}")
    print(f"  Logs             : {LOG_DIR.relative_to(ROOT)}/")

    # Runtime estimate for fast_dev_batches mode:
    # - Epoch 1: cold disk I/O (~6.4 s/batch measured)
    # - Epoch 2+: OS page cache warm (~1.5 s/batch)
    # - Val pass (24 batches, no grad, warm): ~72 s
    # Full-dataset fallback: ~90 s/epoch conservative.
    if fdb > 0:
        cold_epoch_s = fdb * 6.4 + 24 * 3.0    # epoch 1 cold
        warm_epoch_s = fdb * 1.5 + 24 * 3.0    # epoch 2+ warm
        sec_per_run  = cold_epoch_s + max(0, args.epochs - 1) * warm_epoch_s
        est_min_per_run = sec_per_run / 60
        est_h = total * sec_per_run / 3600
    else:
        sec_per_epoch = 90 if args.num_workers == 0 else 55
        est_min_per_run = args.epochs * sec_per_epoch / 60
        est_h = total * args.epochs * sec_per_epoch / 3600
    print(f"  Est. time        : ~{est_h:.1f} h total  "
          f"(~{est_min_per_run:.0f} min/run, cold ep1 then warm)")

    # ── Memory safety check ───────────────────────────────────────────────────
    try:
        import shutil
        mem = __import__("psutil").virtual_memory() if __import__("importlib").util.find_spec("psutil") else None
    except Exception:
        mem = None
    if mem is None:
        # Fallback: parse /proc/meminfo
        try:
            info = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                k, v = line.split(":", 1)
                info[k.strip()] = int(v.strip().split()[0])   # kB
            avail_gb = info.get("MemAvailable", 0) / 1024 / 1024
            swap_free_gb = info.get("SwapFree", 0) / 1024 / 1024
            print(f"  RAM avail   : {avail_gb:.1f} GB  |  Swap free: {swap_free_gb:.1f} GB")
            if avail_gb < 3.0:
                print("  [WARNING] Less than 3 GB RAM available. Consider:")
                print("    - killing other processes before running")
                print("    - using --num-workers 0 (already the default)")
                print("    - using --batch-size 8")
        except Exception:
            pass

    print(_sep())

    if args.dry_run:
        print("\n  [DRY RUN] Planned runs:\n")
        run_num = 0
        for method in methods:
            for ratio in ratios:
                for seed in seeds:
                    run_num += 1
                    tag = run_tag(method, ratio, seed)
                    done = is_complete(method, ratio, seed)
                    status = "SKIP" if done else "RUN "
                    print(f"  [{status}] {run_num:>2}/{total}  {tag}")
        print()
        return

    # ── Execution loop ────────────────────────────────────────────────────────
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_done    = 0
    n_failed  = 0
    n_skipped = 0
    run_num   = 0
    study_start = time.time()
    all_rows: List[Dict] = []

    for method in methods:
        for ratio in ratios:
            for seed in seeds:
                run_num += 1
                tag = run_tag(method, ratio, seed)

                print(f"\n{_sep()}")
                print(f"  Run {run_num}/{total}  │  {tag}")
                print(_sep())

                # ── Skip check ────────────────────────────────────────────────
                if is_complete(method, ratio, seed):
                    print(f"  [SKIP] best_results.json already exists — not overwriting.")
                    n_skipped += 1
                    row = extract_metrics(method, ratio, seed)
                    if row:
                        all_rows.append(row)
                    continue

                # ── Run ───────────────────────────────────────────────────────
                elapsed_so_far = time.time() - study_start
                if run_num > 1:
                    avg_per_run = elapsed_so_far / (n_done + n_failed + 1e-9)
                    remaining   = avg_per_run * (total - run_num + 1)
                    print(f"  Elapsed: {_fmt_time(elapsed_so_far)}  "
                          f"│  ETA: ~{_fmt_time(remaining)}")

                try:
                    ok = run_single(
                        method, ratio, seed,
                        epochs=args.epochs,
                        num_workers=args.num_workers,
                        batch_size=args.batch_size,
                        fast_dev_batches=args.fast_dev_batches,
                    )
                    if ok:
                        n_done += 1
                        row = extract_metrics(method, ratio, seed)
                        if row:
                            all_rows.append(row)
                            print(f"  Metrics: F1={row['macro_f1']:.4f}  "
                                  f"NC1={row['nc1']:.4f}  "
                                  f"Melanoma={row['melanoma_recall']:.4f}")
                    else:
                        n_failed += 1
                except Exception as exc:
                    print(f"  [ERROR] Unexpected exception for {tag}: {exc}")
                    n_failed += 1

    wall_time = time.time() - study_start

    # ── Aggregation ───────────────────────────────────────────────────────────
    print(f"\n{_sep()}")
    print("  Aggregating results …")

    df      = build_summary(all_rows)
    grouped = build_grouped_summary(df)

    if not df.empty:
        df.to_csv(SUMMARY_CSV, index=False)
        print(f"  [CSV] {SUMMARY_CSV.relative_to(ROOT)}  ({len(df)} rows)")

    if not grouped.empty:
        grouped.to_csv(GROUPED_CSV, index=False)
        print(f"  [CSV] {GROUPED_CSV.relative_to(ROOT)}  ({len(grouped)} rows)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.skip_plots:
        print("  Generating plots …")
        generate_plots(grouped)
    else:
        print("  [skip-plots] Skipping matplotlib generation.")

    # ── Final summary ─────────────────────────────────────────────────────────
    print_final_summary(
        n_done=n_done,
        n_failed=n_failed,
        n_skipped=n_skipped,
        total=total,
        df=df,
        grouped=grouped,
        wall_time=wall_time,
    )


if __name__ == "__main__":
    main()
