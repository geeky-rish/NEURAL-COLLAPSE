"""
experiments/run_experiment.py
─────────────────────────────────────────────────────────────────────────────
Runs a single (method, imbalance_ratio, seed) experiment cell.

This is the core atomic unit that the sweep scripts call in a loop.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
from utils.device import get_best_device

from config import load_config
from data import get_dataloaders
from models import build_model
from training import Trainer
from utils.seed import set_seed
from utils.logging_utils import get_logger

_logger = get_logger("run_experiment")


def run_single_experiment(
    cfg: dict,
    method: str         = "baseline",
    imbalance_ratio: int = 1,
    seed: int           = 42,
) -> Dict[str, Any]:
    """
    Execute one full training + evaluation run.

    Parameters
    ----------
    cfg             : dict — full project config
    method          : str  — 'baseline'|'weighted'|'focal'|'oversampling'|'etf'
    imbalance_ratio : int  — dataset imbalance ratio
    seed            : int  — random seed

    Returns
    -------
    results : dict
        val_acc, best_val_acc, nc1, nc2, train_loss/acc histories
    """
    set_seed(seed)

    device = get_best_device()[0]
    _logger.info(
        f"── Experiment | method={method} ratio={imbalance_ratio} "
        f"seed={seed} device={device} ──"
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, class_weights = get_dataloaders(
        cfg=cfg,
        imbalance_ratio=imbalance_ratio,
        method=method,
        seed=seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg, method=method)

    # ── Run tag for checkpoint naming ─────────────────────────────────────────
    run_tag = f"{method}_r{imbalance_ratio}_s{seed}"

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        class_weights= class_weights,
        cfg          = cfg,
        method       = method,
        seed         = seed,
        device       = device,
        run_tag      = run_tag,
    )

    results = trainer.run()

    _logger.info(
        f"  → val_acc={results['best_val_acc']:.2f}%  "
        f"NC1={results['nc1']:.4f}  NC2={results['nc2']:.6f}"
    )
    return results
