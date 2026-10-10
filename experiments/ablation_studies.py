"""
experiments/ablation_studies.py
─────────────────────────────────────────────────────────────────────────────
Systematic ablation experiments to understand the contribution of each
design choice in the framework.

Ablation axes
-------------
1. backbone_size  — ResNet-18 vs MobileNetV2
2. nc_reg_weight  — collapse_weight: [0, 0.001, 0.01, 0.1]
3. etf_scale      — ETF temperature: [1, 4, 8, 16, 32]
4. sampling       — none / weighted / balanced / square_root
5. loss_type      — ce / weighted_ce / focal (gamma sweep)

Output: results/ablations/{axis}_ablation.csv + bar plots
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from utils.device import get_best_device

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from data import get_dataloaders, get_medical_dataloaders
from models import build_model
from training import Trainer
from evaluation.nc_metrics import compute_all_nc_metrics
from evaluation.medical_metrics import compute_medical_metrics
from utils.seed import set_seed
from utils.logging_utils import get_logger

_logger = get_logger("ablation")


def _single_run(
    cfg: dict,
    method: str,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    """Train one model and return a flat metric dict."""
    from data import get_dataloaders, get_medical_dataloaders

    dataset_name = cfg["dataset"]["name"].lower()
    if dataset_name == "cifar10":
        train_loader, val_loader, cw = get_dataloaders(
            cfg, imbalance_ratio=10, method=method, seed=seed)
    else:
        train_loader, val_loader, cw = get_medical_dataloaders(cfg, seed=seed)

    model   = build_model(cfg, method=method)
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        class_weights=cw, cfg=cfg, method=method,
        seed=seed, device=device, run_tag=f"ablation_{method}_s{seed}",
    )
    res = trainer.run()

    # NC eval
    model.eval()
    import copy
    all_f, all_l, all_g = [], [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            f    = model.forward_features(imgs)
            g    = model.fc(f)
            all_f.append(f.cpu()); all_l.append(lbls); all_g.append(g.cpu())
    feats = torch.cat(all_f); lbls = torch.cat(all_l); lgts = torch.cat(all_g)
    nc  = compute_all_nc_metrics(feats, lbls, cfg["dataset"]["num_classes"], model, lgts)
    cn  = getattr(val_loader.dataset, "class_names", None)
    med = compute_medical_metrics(
        lbls.numpy(), lgts.argmax(1).numpy(),
        torch.softmax(lgts, 1).numpy(), cn, cfg["dataset"]["num_classes"])

    return {
        "acc": res["best_val_acc"],
        "macro_f1": med["macro_f1"],
        "sensitivity": med["mean_sensitivity"],
        "nc1": nc.nc1, "nc2": nc.nc2,
    }


def _sweep_axis(
    cfg_base: dict,
    axis_name: str,
    overrides_list: List[Dict[str, Any]],
    label_key: str,
    method: str = "baseline",
    seeds: List[int] = [42],
    device: torch.device = None,
) -> pd.DataFrame:
    """Generic ablation sweep over a list of config override dicts."""
    device = device or get_best_device()[0]
    rows   = []

    for overrides in overrides_list:
        import copy, yaml
        cfg = copy.deepcopy(cfg_base)

        # Apply overrides
        for dotkey, val in overrides.items():
            keys = dotkey.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = val

        label = overrides.get(label_key, str(overrides))
        _logger.info(f"  [{axis_name}] {label_key}={label}")

        seed_results = []
        for seed in seeds:
            set_seed(seed)
            r = _single_run(cfg, method, seed, device)
            r["seed"]  = seed
            r[axis_name] = label
            seed_results.append(r)

        row = {axis_name: label}
        for key in ["acc", "macro_f1", "sensitivity", "nc1", "nc2"]:
            vals = np.array([r[key] for r in seed_results], dtype=float)
            row[f"{key}_mean"] = float(vals.mean())
            row[f"{key}_std"]  = float(vals.std())
        rows.append(row)
        _logger.info(f"    acc={row['acc_mean']:.2f}% F1={row['macro_f1_mean']:.3f}")

    return pd.DataFrame(rows)


def run_backbone_ablation(cfg, seeds=None, device=None) -> pd.DataFrame:
    seeds = seeds or cfg.get("seeds", [42])
    overrides_list = [
        {"model.backbone": "resnet18"},
        {"model.backbone": "mobilenetv2"},
    ]
    df = _sweep_axis(cfg, "backbone", overrides_list, "model.backbone",
                     method="focal", seeds=seeds, device=device)
    _save(df, cfg, "backbone_ablation.csv")
    return df


def run_nc_reg_ablation(cfg, seeds=None, device=None) -> pd.DataFrame:
    seeds = seeds or cfg.get("seeds", [42])
    weights = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
    overrides_list = [
        {"nc_regularization.enabled": True if w > 0 else False,
         "nc_regularization.collapse_weight": w,
         "nc_regularization.etf_align_weight": w}
        for w in weights
    ]
    for o, w in zip(overrides_list, weights):
        o["nc_regularization.collapse_weight"] = w   # label key
    df = _sweep_axis(cfg, "nc_reg_weight", overrides_list,
                     "nc_regularization.collapse_weight",
                     method="focal", seeds=seeds, device=device)
    _save(df, cfg, "nc_reg_ablation.csv")
    return df


def run_etf_scale_ablation(cfg, seeds=None, device=None) -> pd.DataFrame:
    seeds  = seeds or cfg.get("seeds", [42])
    scales = [1.0, 4.0, 8.0, 16.0, 32.0]
    overrides_list = [{"etf.scale": s} for s in scales]
    df = _sweep_axis(cfg, "etf_scale", overrides_list, "etf.scale",
                     method="etf", seeds=seeds, device=device)
    _save(df, cfg, "etf_scale_ablation.csv")
    return df


def run_sampling_ablation(cfg, seeds=None, device=None) -> pd.DataFrame:
    seeds      = seeds or cfg.get("seeds", [42])
    strategies = ["none", "weighted", "balanced", "square_root"]
    overrides_list = [{"sampling.strategy": s} for s in strategies]
    df = _sweep_axis(cfg, "sampling", overrides_list, "sampling.strategy",
                     method="focal", seeds=seeds, device=device)
    _save(df, cfg, "sampling_ablation.csv")
    return df


def _save(df: pd.DataFrame, cfg: dict, filename: str) -> None:
    save_dir = Path(cfg["logging"]["results_dir"]) / "ablations"
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / filename
    df.to_csv(out, index=False)
    _logger.info(f"Saved → {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--axis", default="backbone",
                   choices=["backbone", "nc_reg", "etf_scale", "sampling", "all"])
    p.add_argument("--config",   default=None)
    p.add_argument("--override", nargs="*", default=None)
    args = p.parse_args()
    cfg = load_config(config_path=args.config, overrides=args.override)
    dev = get_best_device()[0]

    if args.axis == "backbone" or args.axis == "all":
        run_backbone_ablation(cfg, device=dev)
    if args.axis == "nc_reg" or args.axis == "all":
        run_nc_reg_ablation(cfg, device=dev)
    if args.axis == "etf_scale" or args.axis == "all":
        run_etf_scale_ablation(cfg, device=dev)
    if args.axis == "sampling" or args.axis == "all":
        run_sampling_ablation(cfg, device=dev)
