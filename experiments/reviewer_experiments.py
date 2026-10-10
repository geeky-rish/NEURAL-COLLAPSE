"""
experiments/reviewer_experiments.py
─────────────────────────────────────────────────────────────────────────────
Camera-ready revision: reviewer-requested GPU experiments for NC-MedAI.

Paper: "NC-MedAI: Geometry-Aware Training via Fixed Equiangular Tight Frame
        Heads for Imbalanced Dermoscopic Skin-Lesion Classification"

This script drives all three groups of reviewer experiments:

  GROUP A — Multi-seed reproducibility
    A1.  ETF+NC-Reg,       seed 7
    A2.  ETF+NC-Reg,       seed 123
    A3.  ETF+NC-Reg+Bal,   seed 7
    A4.  ETF+NC-Reg+Bal,   seed 123
    (seed 42 already exists in results/phase2/ — NOT re-run)

  GROUP B — Component-wise ablation (seed 42)
    B1.  ETF-only           — fixed ETF head,  no NC loss
    B2.  ETF + L_collapse   — fixed ETF head,  L_collapse only
    B3.  ETF + L_align      — fixed ETF head,  L_align only
    (B4  ETF + L_collapse + L_align = existing etf_nc_reg_ham10000_r10_s42)

  GROUP C — Lambda sensitivity (seed 42)
    C1.  lambda = 0.001
    C2.  lambda = 0.05
    (baseline lambda = 0.01 = existing etf_nc_reg_ham10000_r10_s42)

PROTOCOL
--------
All experiments reproduce the IDENTICAL setup as the original Phase-2 study:
  - Dataset  : HAM10000, 10:1 imbalance, same train/val/test split
  - Backbone : ResNet-18 (pretrained ImageNet)
  - Optimizer: SGD, lr=0.01, momentum=0.9, weight_decay=1e-4
  - Schedule : cosine annealing with 3-epoch warmup, 50 epochs
  - Batch size: 64

The ONLY intentional variables are seed / NC-reg weights / sampling strategy,
as stated in each experiment definition below.

LAMBDA DEFINITION
-----------------
The paper reports a single λ that multiplies BOTH NC regularization terms:
    L_total = L_CE + λ·L_collapse + λ·L_align

In the implementation this maps to:
    nc_regularization.collapse_weight = λ
    nc_regularization.etf_align_weight = λ

Both are set to the same value in all experiments.

OUTPUT DIRECTORIES
------------------
  results/
      reviewer_experiments/
          multiseed/   (Group A)
          ablations/   (Group B)
          lambda_sensitivity/  (Group C)

  checkpoints/
      reviewer_experiments/
          multiseed/
          ablations/
          lambda_sensitivity/

  study_logs/
      reviewer_experiments/
          multiseed/
          ablations/
          lambda_sensitivity/

USAGE
-----
  # Dry run (shows commands, no training):
  python -m experiments.reviewer_experiments --dry-run

  # Full run (GPU server):
  python -m experiments.reviewer_experiments

  # Run only one group:
  python -m experiments.reviewer_experiments --groups A
  python -m experiments.reviewer_experiments --groups B
  python -m experiments.reviewer_experiments --groups C
  python -m experiments.reviewer_experiments --groups A B

  # Run a single experiment by key:
  python -m experiments.reviewer_experiments --only A1 B2 C1

RESUME SAFETY
-------------
Each experiment is idempotent: if best_results.json already exists in the
output directory, the experiment is skipped. This allows safe re-invocation
after interruption on the GPU server.
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

# ─────────────────────────────────────────────────────────────────────────────
# Common hyperparameters (must match original Phase-2 study exactly)
# ─────────────────────────────────────────────────────────────────────────────
_DATASET          = "ham10000"
_IMB_RATIO        = 10
_EPOCHS           = 50
_BATCH_SIZE       = 64          # matches original config_snapshot.yaml
_LR               = 0.01        # matches original config_snapshot.yaml
_MOMENTUM         = 0.9
_WEIGHT_DECAY     = 1e-4
_WARMUP_EPOCHS    = 3
_LR_SCHEDULE      = "cosine"
_BACKBONE         = "resnet18"
_ETF_SCALE        = 16.0
_LAMBDA_PAPER     = 0.01        # original lambda from the paper

# Output roots (relative to repo root; resolved to absolute before use)
_RESULTS_ROOT     = ROOT / "results"  / "reviewer_experiments"
_CKPT_ROOT        = ROOT / "checkpoints" / "reviewer_experiments"
_LOG_ROOT         = ROOT / "study_logs" / "reviewer_experiments"

# ─────────────────────────────────────────────────────────────────────────────
# Experiment definitions
# ─────────────────────────────────────────────────────────────────────────────
# Each definition is a dict with the fields:
#   key               : unique identifier (e.g. "A1")
#   group             : "A" | "B" | "C"
#   run_tag           : unique directory/log name (no collisions with phase2/)
#   description       : human-readable description
#   method_flag       : value passed to --method in train.py
#   head              : model.head config value
#   nc_reg_enabled    : bool  — nc_regularization.enabled
#   collapse_weight   : float — L_collapse coefficient (λ_collapse)
#   align_weight      : float — L_align coefficient (λ_align)
#   sampling_strategy : "weighted" | "balanced" | "none"
#   seed              : int
#
# Keys to note:
#   - method_flag="etf" triggers get_criterion("etf", ...) → plain CE loss
#   - nc_reg_enabled=True  wraps base criterion in CombinedNCLoss
#   - nc_reg_enabled=False uses base criterion directly (no NC terms)
#   - collapse_weight=0.0  zeroes L_collapse; it is still added but contributes 0
#   - align_weight=0.0     zeroes L_align;    same as above
#   (For B1/ETF-only we use nc_reg_enabled=False to avoid any NC-reg overhead)

REVIEWER_EXPERIMENTS: Dict[str, Dict] = {

    # ── GROUP A: Multi-seed reproducibility ──────────────────────────────────
    "A1": {
        "key":               "A1",
        "group":             "A",
        "run_tag":           "etf_nc_reg_ham10000_r10_s7",
        "description":       "ETF+NC-Reg (λ=0.01, WeightedSampler), seed=7",
        "method_flag":       "etf",
        "head":              "etf",
        "nc_reg_enabled":    True,
        "collapse_weight":   _LAMBDA_PAPER,
        "align_weight":      _LAMBDA_PAPER,
        "sampling_strategy": "weighted",
        "seed":              7,
        "results_subdir":    "multiseed",
    },
    "A2": {
        "key":               "A2",
        "group":             "A",
        "run_tag":           "etf_nc_reg_ham10000_r10_s123",
        "description":       "ETF+NC-Reg (λ=0.01, WeightedSampler), seed=123",
        "method_flag":       "etf",
        "head":              "etf",
        "nc_reg_enabled":    True,
        "collapse_weight":   _LAMBDA_PAPER,
        "align_weight":      _LAMBDA_PAPER,
        "sampling_strategy": "weighted",
        "seed":              123,
        "results_subdir":    "multiseed",
    },
    "A3": {
        "key":               "A3",
        "group":             "A",
        "run_tag":           "etf_nc_reg_balanced_ham10000_r10_s7",
        "description":       "ETF+NC-Reg+Bal (λ=0.01, ClassBalancedSampler), seed=7",
        "method_flag":       "etf",
        "head":              "etf",
        "nc_reg_enabled":    True,
        "collapse_weight":   _LAMBDA_PAPER,
        "align_weight":      _LAMBDA_PAPER,
        "sampling_strategy": "balanced",
        "seed":              7,
        "results_subdir":    "multiseed",
    },
    "A4": {
        "key":               "A4",
        "group":             "A",
        "run_tag":           "etf_nc_reg_balanced_ham10000_r10_s123",
        "description":       "ETF+NC-Reg+Bal (λ=0.01, ClassBalancedSampler), seed=123",
        "method_flag":       "etf",
        "head":              "etf",
        "nc_reg_enabled":    True,
        "collapse_weight":   _LAMBDA_PAPER,
        "align_weight":      _LAMBDA_PAPER,
        "sampling_strategy": "balanced",
        "seed":              123,
        "results_subdir":    "multiseed",
    },

    # ── GROUP B: Component-wise ablation (seed 42) ───────────────────────────
    "B1": {
        "key":               "B1",
        "group":             "B",
        "run_tag":           "etf_only_ham10000_r10_s42",
        "description":       "ETF head only — no NC regularization (seed=42)",
        "method_flag":       "etf",
        "head":              "etf",
        "nc_reg_enabled":    False,   # trainer uses plain CE, no CombinedNCLoss
        "collapse_weight":   0.0,     # irrelevant when nc_reg_enabled=False
        "align_weight":      0.0,     # irrelevant when nc_reg_enabled=False
        "sampling_strategy": "weighted",
        "seed":              42,
        "results_subdir":    "ablations",
    },
    "B2": {
        "key":               "B2",
        "group":             "B",
        "run_tag":           "etf_lcollapse_ham10000_r10_s42",
        "description":       "ETF + L_collapse only (λ_collapse=0.01, λ_align=0.0, seed=42)",
        "method_flag":       "etf",
        "head":              "etf",
        "nc_reg_enabled":    True,
        "collapse_weight":   _LAMBDA_PAPER,   # L_collapse enabled
        "align_weight":      0.0,             # L_align disabled
        "sampling_strategy": "weighted",
        "seed":              42,
        "results_subdir":    "ablations",
    },
    "B3": {
        "key":               "B3",
        "group":             "B",
        "run_tag":           "etf_lalign_ham10000_r10_s42",
        "description":       "ETF + L_align only (λ_collapse=0.0, λ_align=0.01, seed=42)",
        "method_flag":       "etf",
        "head":              "etf",
        "nc_reg_enabled":    True,
        "collapse_weight":   0.0,             # L_collapse disabled
        "align_weight":      _LAMBDA_PAPER,   # L_align enabled
        "sampling_strategy": "weighted",
        "seed":              42,
        "results_subdir":    "ablations",
    },
    # B4 = etf_nc_reg_ham10000_r10_s42 (seed 42, both terms enabled) — EXISTS

    # ── GROUP C: Lambda sensitivity (seed 42) ─────────────────────────────────
    "C1": {
        "key":               "C1",
        "group":             "C",
        "run_tag":           "etf_nc_reg_lambda0001_ham10000_r10_s42",
        "description":       "ETF+NC-Reg, λ=0.001 (seed=42)",
        "method_flag":       "etf",
        "head":              "etf",
        "nc_reg_enabled":    True,
        "collapse_weight":   0.001,
        "align_weight":      0.001,
        "sampling_strategy": "weighted",
        "seed":              42,
        "results_subdir":    "lambda_sensitivity",
    },
    "C2": {
        "key":               "C2",
        "group":             "C",
        "run_tag":           "etf_nc_reg_lambda005_ham10000_r10_s42",
        "description":       "ETF+NC-Reg, λ=0.05 (seed=42)",
        "method_flag":       "etf",
        "head":              "etf",
        "nc_reg_enabled":    True,
        "collapse_weight":   0.05,
        "align_weight":      0.05,
        "sampling_strategy": "weighted",
        "seed":              42,
        "results_subdir":    "lambda_sensitivity",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def results_dir(exp: Dict) -> Path:
    return _RESULTS_ROOT / exp["results_subdir"] / exp["run_tag"]


def checkpoint_dir(exp: Dict) -> Path:
    return _CKPT_ROOT / exp["results_subdir"] / exp["run_tag"]


def log_path(exp: Dict) -> Path:
    return _LOG_ROOT / exp["results_subdir"] / f"{exp['run_tag']}.log"


def is_complete(exp: Dict) -> bool:
    """Return True if best_results.json already exists (experiment done)."""
    return (results_dir(exp) / "best_results.json").is_file()


# ─────────────────────────────────────────────────────────────────────────────
# Command builder
# ─────────────────────────────────────────────────────────────────────────────

def build_command(exp: Dict, num_workers: int = 4) -> List[str]:
    """
    Construct the exact train.py command for one reviewer experiment.

    All paths are relative or auto-resolved; no hard-coded absolute paths.
    The GPU server must run this from the repository root.
    """
    nc_reg_str    = "true" if exp["nc_reg_enabled"] else "false"
    results_path  = str(results_dir(exp))
    ckpt_path     = str(checkpoint_dir(exp))

    cmd = [
        sys.executable, "-u",
        str(ROOT / "train.py"),
        "--method",  exp["method_flag"],
        "--override",
            f"dataset.name={_DATASET}",
            f"dataset.imbalance_ratio={_IMB_RATIO}",
            f"seed={exp['seed']}",
            f"model.head={exp['head']}",
            f"model.backbone={_BACKBONE}",
            f"training.epochs={_EPOCHS}",
            f"training.batch_size={_BATCH_SIZE}",
            f"training.lr={_LR}",
            f"training.momentum={_MOMENTUM}",
            f"training.weight_decay={_WEIGHT_DECAY}",
            f"training.warmup_epochs={_WARMUP_EPOCHS}",
            f"training.lr_schedule={_LR_SCHEDULE}",
            f"training.num_workers={num_workers}",
            "tracking.tensorboard=false",
            "debug.fast_dev_batches=0",          # full dataset, no smoke-test shortcut
            "logging.log_every_n_epochs=1",
            "nc_tracking.enabled=true",
            "nc_tracking.every_n_epochs=1",
            f"nc_regularization.enabled={nc_reg_str}",
            f"nc_regularization.collapse_weight={exp['collapse_weight']}",
            f"nc_regularization.etf_align_weight={exp['align_weight']}",
            f"sampling.strategy={exp['sampling_strategy']}",
            f"etf.scale={_ETF_SCALE}",
            # Unique output directories — prevents any collision
            f"logging.run_tag={exp['run_tag']}",
            f"logging.results_dir={results_path}",
            f"logging.checkpoint_dir={ckpt_path}",
    ]
    return cmd


# ─────────────────────────────────────────────────────────────────────────────
# Single-experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def run_single(exp: Dict, num_workers: int = 4, dry_run: bool = False) -> bool:
    """
    Execute one reviewer experiment as a subprocess.

    Returns True on success, False on failure.
    """
    key     = exp["key"]
    tag     = exp["run_tag"]
    lp      = log_path(exp)
    rdir    = results_dir(exp)
    ckpt    = checkpoint_dir(exp)

    # Create directories
    lp.parent.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    cmd = build_command(exp, num_workers=num_workers)

    print(f"\n  [{key}] {exp['description']}")
    print(f"  CMD: {' '.join(cmd)}\n")

    if dry_run:
        print(f"  [DRY RUN]  results → {rdir}")
        print(f"             ckpt    → {ckpt}")
        print(f"             log     → {lp}")
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
        print(f"  [OK]   {tag}  ({_fmt_time(elapsed)})  → {rdir}")
    else:
        print(f"  [FAIL] {tag}  exit={proc.returncode}  ({_fmt_time(elapsed)})  log → {lp}")
        # Print tail of log on failure so the user sees the error immediately
        try:
            lines = lp.read_text(errors="replace").splitlines()
            tail  = lines[-25:] if len(lines) >= 25 else lines
            print(f"\n  Last {len(tail)} log lines:")
            for line in tail:
                print(f"    {line}")
        except Exception:
            pass

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Brief cooldown between runs to allow GPU memory to fully clear
    print("  [cooldown] 5 s …")
    time.sleep(5)
    return success


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_metrics(exp: Dict) -> Optional[Dict]:
    """Load best_results.json and class_metrics.csv for a completed run."""
    rdir    = results_dir(exp)
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
        "key":              exp["key"],
        "group":            exp["group"],
        "run_tag":          exp["run_tag"],
        "description":      exp["description"],
        "seed":             exp["seed"],
        "sampling":         exp["sampling_strategy"],
        "collapse_weight":  exp["collapse_weight"],
        "align_weight":     exp["align_weight"],
        "nc_reg_enabled":   exp["nc_reg_enabled"],
        "best_val_acc":     br.get("best_val_acc",     float("nan")),
        "macro_f1":         br.get("macro_f1",          float("nan")),
        "roc_auc":          br.get("roc_auc",           float("nan")),
        "nc1":              br.get("nc1",               float("nan")),
        "nc2":              br.get("nc2",               float("nan")),
        "nc4":              br.get("nc4",               float("nan")),
        "melanoma_recall":  mel_recall,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(rows: List[Dict], n_done: int, n_failed: int,
                  n_skipped: int, wall_time: float) -> None:
    print("\n" + "=" * 70)
    print("  NC-MedAI — Reviewer Experiments Complete")
    print("=" * 70)
    print(f"  Completed: {n_done}  |  Skipped: {n_skipped}  |  Failed: {n_failed}")
    print(f"  Wall time: {_fmt_time(wall_time)}")
    print("-" * 70)
    if rows:
        df = pd.DataFrame(rows)
        cols = ["key", "group", "seed", "macro_f1", "melanoma_recall",
                "nc1", "nc4", "roc_auc"]
        existing = [c for c in cols if c in df.columns]
        print(df[existing].to_string(index=False))
    print("=" * 70)
    print(f"  Results root : {_RESULTS_ROOT}/")
    print(f"  Ckpt root    : {_CKPT_ROOT}/")
    print(f"  Log root     : {_LOG_ROOT}/")

    # Save combined summary CSV
    if rows:
        summary_path = _RESULTS_ROOT / "reviewer_summary.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"  Summary CSV  : {summary_path}")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Camera-ready reviewer experiments for NC-MedAI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--groups", nargs="+", default=["A", "B", "C"],
        choices=["A", "B", "C"],
        help="Experiment groups to run (default: all three)"
    )
    p.add_argument(
        "--only", nargs="+", default=None,
        metavar="KEY",
        help="Run only specific experiment keys, e.g. --only A1 B2 C1"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands only; do not execute training"
    )
    p.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes (default: 4)"
    )
    p.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip experiments whose best_results.json already exists (default: True)"
    )
    p.add_argument(
        "--force", action="store_true", default=False,
        help="Re-run experiments even if results already exist"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Determine which experiments to run
    if args.only:
        invalid = [k for k in args.only if k not in REVIEWER_EXPERIMENTS]
        if invalid:
            print(f"[ERROR] Unknown experiment keys: {invalid}")
            print(f"  Valid keys: {list(REVIEWER_EXPERIMENTS.keys())}")
            sys.exit(1)
        selected = {k: REVIEWER_EXPERIMENTS[k] for k in args.only}
    else:
        selected = {
            k: v for k, v in REVIEWER_EXPERIMENTS.items()
            if v["group"] in args.groups
        }

    total = len(selected)

    print("=" * 70)
    print("  NC-MedAI — Camera-Ready Reviewer Experiments")
    print("=" * 70)
    print(f"  Groups      : {args.groups}")
    print(f"  Experiments : {list(selected.keys())}  ({total} total)")
    print(f"  Dry run     : {args.dry_run}")
    print(f"  Num workers : {args.num_workers}")
    print(f"  Results root: {_RESULTS_ROOT}/")
    print("-" * 70)
    for k, exp in selected.items():
        done = "✓ DONE" if is_complete(exp) else "  pending"
        print(f"  [{k}] {done}  {exp['description']}")
    print("=" * 70)

    # Create top-level output directories
    _RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    _CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    _LOG_ROOT.mkdir(parents=True, exist_ok=True)

    n_done = n_failed = n_skipped = run_num = 0
    study_start = time.time()
    all_rows: List[Dict] = []

    for k, exp in selected.items():
        run_num += 1
        print(f"\n{'─' * 70}")
        print(f"  Run {run_num}/{total}  [{k}]  {exp['run_tag']}")
        print(f"{'─' * 70}")

        # Skip logic
        if is_complete(exp) and not args.force:
            print(f"  [SKIP] best_results.json already exists.")
            n_skipped += 1
            row = extract_metrics(exp)
            if row:
                all_rows.append(row)
            continue

        ok = run_single(exp, num_workers=args.num_workers, dry_run=args.dry_run)

        if ok:
            n_done += 1
            if not args.dry_run:
                row = extract_metrics(exp)
                if row:
                    all_rows.append(row)
                    print(
                        f"  Metrics: F1={row['macro_f1']:.4f} "
                        f"NC1={row['nc1']:.4f} "
                        f"Melanoma={row['melanoma_recall']:.4f}"
                    )
        else:
            n_failed += 1

    wall_time = time.time() - study_start
    print_summary(all_rows, n_done, n_failed, n_skipped, wall_time)


if __name__ == "__main__":
    main()
