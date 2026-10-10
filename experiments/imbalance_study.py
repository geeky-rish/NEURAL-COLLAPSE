"""
experiments/imbalance_study.py
─────────────────────────────────────────────────────────────────────────────
Sweep over imbalance ratios (or class distributions) for a medical dataset.

Produces:
  • results/imbalance_study/imbalance_sweep_{method}.csv
  • Accuracy, NC1, NC2, F1, Sensitivity plots vs ratio
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from utils.device import get_best_device

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from data import get_dataloaders, get_medical_dataloaders
from models import build_model
from training import Trainer, get_criterion
from evaluation.nc_metrics import compute_all_nc_metrics
from evaluation.medical_metrics import compute_medical_metrics
from evaluation.evaluator import extract_features
from utils.seed import set_seed
from utils.logging_utils import get_logger

_logger = get_logger("imbalance_study")


def run_imbalance_study(
    cfg: dict,
    method: str = "focal",
    ratios: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Sweep imbalance ratios and record accuracy + NC + medical metrics.

    Parameters
    ----------
    cfg    : full project config
    method : training method to evaluate
    ratios : list of imbalance ratios (CIFAR only); for medical datasets,
             this controls sub-sampling of minority classes artificially
    seeds  : random seeds for averaging

    Returns
    -------
    pd.DataFrame  with per-ratio aggregated results
    """
    seeds  = seeds  or cfg.get("seeds", [42])
    ratios = ratios or cfg["dataset"].get("imbalance_ratios", [1, 5, 10, 20, 50, 100])
    device = get_best_device()[0]
    num_classes  = cfg["dataset"]["num_classes"]
    dataset_name = cfg["dataset"]["name"].lower()
    save_dir     = Path(cfg["logging"]["results_dir"]) / "imbalance_study"
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ratio in ratios:
        seed_rows = []

        for seed in seeds:
            set_seed(seed)
            _logger.info(f"  ratio={ratio}  seed={seed}  method={method}")

            # Data
            if dataset_name == "cifar10":
                train_loader, val_loader, cw = get_dataloaders(
                    cfg, imbalance_ratio=ratio, method=method, seed=seed)
            else:
                # Pass ratio through cfg so HAM10000Dataset._apply_long_tail_subsampling runs
                import copy as _copy
                cfg_copy = _copy.deepcopy(cfg)
                cfg_copy.setdefault("dataset", {})["imbalance_ratio"] = ratio
                train_loader, val_loader, cw = get_medical_dataloaders(cfg_copy, seed=seed)

            # Model + Trainer
            model   = build_model(cfg, method=method)
            run_tag = f"imbstudy_{method}_{dataset_name}_r{ratio}_s{seed}"
            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                class_weights=cw, cfg=cfg, method=method,
                seed=seed, device=device, run_tag=run_tag,
            )
            res = trainer.run()

            # Full eval
            model.eval()
            feats, lbls = _get_features(model, val_loader, device)
            lgts        = _get_logits(model, val_loader, device)
            y_true = lbls.numpy()
            y_pred = lgts.argmax(1).numpy()
            y_prob = torch.softmax(lgts, 1).numpy()

            cn  = getattr(val_loader.dataset, "class_names", None)
            med = compute_medical_metrics(y_true, y_pred, y_prob,
                                          class_names=cn, num_classes=num_classes)
            nc  = compute_all_nc_metrics(feats, lbls, num_classes, model, lgts)

            seed_rows.append({
                "ratio":    ratio,
                "seed":     seed,
                "acc":      res["best_val_acc"],
                "macro_f1": med["macro_f1"],
                "sens":     med["mean_sensitivity"],
                "spec":     med["mean_specificity"],
                "kappa":    med["kappa"],
                "nc1":      nc.nc1,
                "nc2":      nc.nc2,
                "nc3":      nc.nc3,
                "nc4":      nc.nc4,
            })

        # Aggregate
        row = {"ratio": ratio, "method": method}
        for key in ["acc", "macro_f1", "sens", "spec", "kappa", "nc1", "nc2", "nc3", "nc4"]:
            vals = np.array([r[key] for r in seed_rows], dtype=float)
            row[f"{key}_mean"] = float(vals.mean())
            row[f"{key}_std"]  = float(vals.std())
        rows.append(row)
        _logger.info(f"    ratio={ratio}: acc={row['acc_mean']:.2f}% "
                     f"F1={row['macro_f1_mean']:.3f} NC1={row['nc1_mean']:.4f}")

    df = pd.DataFrame(rows)
    out = save_dir / f"imbalance_sweep_{method}.csv"
    df.to_csv(out, index=False)
    _logger.info(f"Saved → {out}")
    print(df.to_string(index=False))
    return df


@torch.no_grad()
def _get_features(model, loader, device):
    model.eval()
    F_, L_ = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        F_.append(model.forward_features(imgs).cpu())
        L_.append(lbls)
    return torch.cat(F_), torch.cat(L_)


@torch.no_grad()
def _get_logits(model, loader, device):
    model.eval()
    G_ = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        G_.append(model(imgs).cpu())
    return torch.cat(G_)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--method",  default="focal")
    p.add_argument("--config",  default=None)
    p.add_argument("--override", nargs="*", default=None)
    args = p.parse_args()
    cfg = load_config(config_path=args.config, overrides=args.override)
    run_imbalance_study(cfg, method=args.method)
