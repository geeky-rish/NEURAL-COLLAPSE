"""
experiments/sweep.py
─────────────────────────────────────────────────────────────────────────────
Two sweep functions:

run_imbalance_sweep — vary imbalance_ratio for a SINGLE method across seeds
run_method_sweep    — vary method at a FIXED imbalance_ratio across seeds

Both return a tidy pandas DataFrame and also save a CSV to results_dir.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .run_experiment import run_single_experiment
from utils.logging_utils import get_logger

_logger = get_logger("sweep")


def run_imbalance_sweep(
    cfg: dict,
    method: str             = "baseline",
    ratios: Optional[List[int]] = None,
    seeds: Optional[List[int]]  = None,
) -> pd.DataFrame:
    """
    Sweep over imbalance ratios for a single method.

    Parameters
    ----------
    cfg    : dict
    method : str
    ratios : list[int] — imbalance ratios to test; defaults to cfg value
    seeds  : list[int] — seeds for averaging; defaults to cfg value

    Returns
    -------
    DataFrame with aggregated per-ratio statistics
    """
    ratios = ratios or cfg["dataset"]["imbalance_ratios"]
    seeds  = seeds  or cfg["seeds"]

    results_dir = Path(cfg["logging"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ratio in ratios:
        seed_rows = []
        for seed in seeds:
            r = run_single_experiment(cfg, method=method,
                                      imbalance_ratio=ratio, seed=seed)
            seed_rows.append({
                "ratio": ratio,
                "seed":  seed,
                "acc":   r["best_val_acc"],
                "nc1":   r["nc1"],
                "nc2":   r["nc2"],
            })
            _logger.info(f"  ratio={ratio} seed={seed} → {r['best_val_acc']:.2f}%")

        # Aggregate across seeds
        arr_acc = np.array([s["acc"] for s in seed_rows])
        arr_nc1 = np.array([s["nc1"] for s in seed_rows])
        arr_nc2 = np.array([s["nc2"] for s in seed_rows])
        rows.append({
            "ratio":    ratio,
            "acc_mean": arr_acc.mean(), "acc_std": arr_acc.std(),
            "nc1_mean": arr_nc1.mean(), "nc1_std": arr_nc1.std(),
            "nc2_mean": arr_nc2.mean(), "nc2_std": arr_nc2.std(),
        })

    df = pd.DataFrame(rows)
    csv_path = results_dir / f"sweep_{method}_imbalance.csv"
    df.to_csv(csv_path, index=False)
    _logger.info(f"Saved imbalance-sweep CSV → {csv_path}")
    return df


def run_method_sweep(
    cfg: dict,
    imbalance_ratio: int        = 10,
    methods: Optional[List[str]] = None,
    seeds: Optional[List[int]]   = None,
) -> pd.DataFrame:
    """
    Sweep over methods at a fixed imbalance ratio.

    Parameters
    ----------
    cfg             : dict
    imbalance_ratio : int
    methods         : list[str] — defaults to cfg sweep.methods
    seeds           : list[int] — defaults to cfg seeds

    Returns
    -------
    DataFrame with per-method aggregated statistics
    """
    methods = methods or cfg["sweep"]["methods"]
    seeds   = seeds   or cfg["seeds"]

    results_dir = Path(cfg["logging"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for method in methods:
        seed_rows = []
        for seed in seeds:
            r = run_single_experiment(cfg, method=method,
                                      imbalance_ratio=imbalance_ratio, seed=seed)
            seed_rows.append({
                "method": method,
                "seed":   seed,
                "acc":    r["best_val_acc"],
                "nc1":    r["nc1"],
                "nc2":    r["nc2"],
            })
            _logger.info(f"  method={method} seed={seed} → {r['best_val_acc']:.2f}%")

        arr_acc = np.array([s["acc"] for s in seed_rows])
        arr_nc1 = np.array([s["nc1"] for s in seed_rows])
        arr_nc2 = np.array([s["nc2"] for s in seed_rows])
        rows.append({
            "method":   method,
            "acc_mean": arr_acc.mean(), "acc_std": arr_acc.std(),
            "nc1_mean": arr_nc1.mean(), "nc1_std": arr_nc1.std(),
            "nc2_mean": arr_nc2.mean(), "nc2_std": arr_nc2.std(),
        })

    df = pd.DataFrame(rows)
    csv_path = results_dir / f"sweep_methods_r{imbalance_ratio}.csv"
    df.to_csv(csv_path, index=False)
    _logger.info(f"Saved method-sweep CSV → {csv_path}")
    return df
