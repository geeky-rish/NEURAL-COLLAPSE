"""
train.py — CLI entry point for a single training run.
Updated for the NC-MedAI Medical AI Framework.

Usage
-----
# HAM10000 with focal loss
python train.py --dataset ham10000 --method focal --seed 42

# CIFAR-10 baseline at imbalance ratio 10
python train.py --dataset cifar10 --method baseline --ratio 10

# ETF head on chest X-ray
python train.py --dataset chestxray --method etf --seed 42

# NC regularization enabled via override
python train.py --method focal --override nc_regularization.enabled=true

# Short test run
python train.py --method baseline --override training.epochs=3 dataset.name=cifar10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from data import get_dataloaders, get_medical_dataloaders
from models import build_model
from training import Trainer, get_criterion
from evaluation.nc_metrics import compute_all_nc_metrics
from evaluation.medical_metrics import compute_medical_metrics, print_medical_metrics
from visualization.tsne_visualizer import plot_tsne
from visualization.feature_geometry import plot_cosine_heatmap, plot_pca
from visualization.confusion_analysis import plot_confusion_matrix, plot_per_class_recall
from utils.seed import set_seed
from utils.logging_utils import get_logger
from utils.device import get_best_device
import torch
import numpy as np
import pandas as pd
import yaml
from utils.experiment_reporter import save_experiment_outputs

_logger = get_logger("train")

_start_time: float = 0.0

_METHOD_MAP = {
    "weighted_loss": "weighted",
    "weighted_ce":   "weighted",
    "focal_loss":    "focal",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NC-MedAI: Single experiment training run."
    )
    p.add_argument("--config",  default=None,
                   help="Path to YAML config (default: config/config.yaml)")
    p.add_argument("--dataset", default=None,
                   choices=["cifar10", "ham10000", "chestxray", "retinal_oct"],
                   help="Dataset override (sets dataset.name in config)")
    p.add_argument("--method",  default="baseline",
                   choices=["baseline", "weighted_ce", "focal", "oversampling",
                             "etf", "prototype", "balanced", "square_root"],
                   help="Training method / classifier head")
    p.add_argument("--ratio",   type=int, default=None,
                   help="Imbalance ratio (CIFAR-10 only)")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--profile", default=None,
                   choices=["apple_silicon", "cuda_gpu", "cpu_debug"],
                   help="Hardware profile to load before config overrides")
    p.add_argument("--visualize", action="store_true",
                   help="Generate t-SNE, PCA, cosine heatmap, confusion plots after training")
    p.add_argument("--override", nargs="*", default=None, metavar="KEY=VALUE",
                   help="Dot-notation config overrides")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from (e.g. latest.pth)")
    return p.parse_args()


def main() -> None:
    global _start_time
    _start_time = time.time()
    args = parse_args()

    # ── Build overrides list ─────────────────────────────────────────────────────
    overrides = list(args.override or [])
    if args.dataset:
        overrides.append(f"dataset.name={args.dataset}")
    if args.ratio is not None:
        overrides.append(f"dataset.imbalance_ratio={args.ratio}")

    cfg = load_config(config_path=args.config,
                      overrides=overrides or None,
                      profile=args.profile)
    
    # ── Method ↔ Model Head Routing ───────────────────────────────────────────
    head = cfg.get("model", {}).get("head", "linear")
    raw_method = args.method
    if raw_method == "baseline":
        if head == "etf": raw_method = "etf"
        elif head == "prototype": raw_method = "prototype"
        elif head == "linear": raw_method = "baseline"
    method = _METHOD_MAP.get(raw_method, raw_method)

    # ── Resolve effective seed ────────────────────────────────────────────────
    # Priority: --override seed=N (lands in cfg["seed"]) > --seed N (args.seed)
    # This ensures `python train.py --override seed=7` works identically to
    # `python train.py --seed 7`.
    seed = int(cfg.get("seed", args.seed))
    if seed != args.seed:
        # Keep cfg consistent so downstream modules that read cfg["seed"] agree
        cfg["seed"] = seed

    set_seed(seed)

    dataset_name = cfg["dataset"]["name"].lower()
    
    # ── Auto Num_Classes ──────────────────────────────────────────────────────
    _CLASS_MAP = {"cifar10": 10, "ham10000": 7, "chestxray": 2, "retinal_oct": 4}
    if dataset_name in _CLASS_MAP:
        cfg["dataset"]["num_classes"] = _CLASS_MAP[dataset_name]
    num_classes  = cfg["dataset"]["num_classes"]

    # ── Debug Mode Support ────────────────────────────────────────────────────
    if cfg.get("debug", {}).get("enabled", False):
        _logger.info("Debug mode enabled: using lightweight defaults.")
        cfg.setdefault("tracking", {})["tensorboard"] = False
        args.visualize = False
        cfg.setdefault("training", {})["num_workers"] = min(cfg.get("training", {}).get("num_workers", 4), 0)

    # ── Startup Validation ────────────────────────────────────────────────────
    if num_classes <= 0:
        raise ValueError(f"Invalid num_classes: {num_classes}")
    
    valid_heads = ["linear", "etf", "prototype"]
    if head not in valid_heads:
        raise ValueError(f"Invalid model.head: {head}. Must be one of {valid_heads}")
        
    schedule = cfg.get("training", {}).get("lr_schedule", "cosine")
    valid_schedules = ["cosine", "step", "none"]
    if schedule not in valid_schedules:
        raise ValueError(f"Invalid lr_schedule: {schedule}. Must be one of {valid_schedules}")

    # ── Device ───────────────────────────────────────────────────────────────────
    device, device_info = get_best_device(cfg)
    # Adapt num_workers to the active backend
    cfg.setdefault("training", {})["num_workers"] = device_info.recommended_workers(
        cfg.get("training", {}).get("num_workers", 4)
    )
    # Store device_info in cfg so reporter can access it
    cfg["_device_info"] = device_info.to_dict()

    # ── Resolve imbalance ratio ────────────────────────────────────────────────
    # Priority: --ratio N (args.ratio) > --override dataset.imbalance_ratio=N > default 1
    imb_ratio = int(
        args.ratio
        if args.ratio is not None
        else cfg.get("dataset", {}).get("imbalance_ratio", 1)
    )
    # Keep cfg["dataset"]["imbalance_ratio"] consistent for DataLoaders + reporter
    cfg.setdefault("dataset", {})["imbalance_ratio"] = imb_ratio
    cfg["_imbalance_ratio"] = imb_ratio   # quick-access for run_tag

    _logger.info(f"Dataset={dataset_name}  Method={method}  Seed={seed}  "
                 f"Ratio={imb_ratio}  Device={device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if dataset_name == "cifar10":
        train_loader, val_loader, class_weights = get_dataloaders(
            cfg, imbalance_ratio=imb_ratio, method=method, seed=seed, device=device)
        class_names = None
    else:
        train_loader, val_loader, class_weights = get_medical_dataloaders(
            cfg, seed=seed, device=device)
        class_names = getattr(train_loader.dataset, "class_names", None)

    # ── Model + Trainer ───────────────────────────────────────────────────────
    model   = build_model(cfg, method=method)
    # ── Run tag: always include ratio to prevent directory collisions ───────────
    # Default format: {method}_{dataset}_r{ratio}_s{seed}
    # Callers (e.g. run_phase2_study.py) can inject logging.run_tag as a config
    # override to use the full study-level key instead of the method-flag-based
    # name. This prevents variants that share the same method flag (e.g.
    # etf_nc_reg vs etf_nc_reg_balanced) from writing to the same directory.
    run_tag = f"{method}_{dataset_name}_r{imb_ratio}_s{seed}"
    run_tag = cfg.get("logging", {}).get("run_tag", run_tag) or run_tag

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        class_weights=class_weights, cfg=cfg, method=method,
        seed=seed, device=device, run_tag=run_tag,
    )
    if args.resume:
        trainer.resume(args.resume)
    results = trainer.run()

    # ── Full medical evaluation ───────────────────────────────────────────────
    model.eval()
    all_f, all_l, all_g = [], [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            f    = model.forward_features(imgs)
            g    = model.fc(f)
            all_f.append(f.cpu()); all_l.append(lbls); all_g.append(g.cpu())

    feats  = torch.cat(all_f)
    labels = torch.cat(all_l)
    logits = torch.cat(all_g)

    y_true = labels.numpy()
    y_pred = logits.argmax(1).numpy()
    y_prob = torch.softmax(logits, 1).numpy()

    nc  = compute_all_nc_metrics(feats, labels, num_classes, model, logits)
    med = compute_medical_metrics(y_true, y_pred, y_prob,
                                  class_names=class_names, num_classes=num_classes)

    # Inject y_true so reporter can do long-tail analysis
    results["_y_true"] = y_true.tolist()

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print(f"  NC-MedAI Training Complete")
    print(f"  Dataset : {dataset_name}   Method : {method}")
    print(f"  Seed    : {seed}      Device : {device}")
    print("─"*60)
    print(f"  Best Val Accuracy : {results['best_val_acc']:.2f}%")
    print(f"  Macro F1          : {med['macro_f1']:.4f}")
    print(f"  Mean Sensitivity  : {med['mean_sensitivity']:.4f}")
    print(f"  Mean Specificity  : {med['mean_specificity']:.4f}")
    print(f"  Cohen Kappa       : {med['kappa']:.4f}")
    print(f"  ROC-AUC           : {med['roc_auc']:.4f}")
    print("─"*60)
    print(f"  NC1 (scatter)     : {nc.nc1:.6f}")
    print(f"  NC2 (ETF dev)     : {nc.nc2:.8f}")
    print(f"  NC3 (W–μ align)   : {nc.nc3:.6f}")
    print(f"  NC4 (NCC disagree): {nc.nc4:.6f}")
    print("═"*60 + "\n")

    # ── Optional t-SNE / PCA / cosine heatmap ────────────────────────────────
    if args.visualize and not cfg.get("analysis", {}).get("lightweight", False):
        vis_dir = str(Path(cfg["logging"]["results_dir"]) / run_tag)
        _logger.info("Generating visualisations …")
        try:
            plot_tsne(feats, labels, class_names=class_names,
                      save_path=f"{vis_dir}/tsne.png",
                      title=f"t-SNE — {method} on {dataset_name}")
            plot_pca(feats, labels, class_names=class_names,
                     save_path=f"{vis_dir}/pca.png",
                     title=f"PCA — {method} on {dataset_name}")
            plot_cosine_heatmap(feats, labels, num_classes, class_names=class_names,
                                save_path=f"{vis_dir}/cosine_heatmap.png")
            _logger.info(f"  Visualisations saved to {vis_dir}/")
        except Exception as exc:
            _logger.warning(f"Visualisation failed (non-fatal): {exc}")

    # ── Centralized experiment output ─────────────────────────────────────────
    total_time_s = time.time() - _start_time
    out_dir = Path(cfg["logging"]["results_dir"]) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # config_snapshot.yaml
    try:
        with open(out_dir / "config_snapshot.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
    except Exception as exc:
        _logger.warning(f"config_snapshot.yaml save failed: {exc}")

    # metrics.csv (final summary row)
    try:
        summary = {
            "method": method, "dataset": dataset_name, "seed": seed,
            "best_val_acc": results["best_val_acc"],
            "macro_f1":     med["macro_f1"],
            "sensitivity":  med["mean_sensitivity"],
            "specificity":  med["mean_specificity"],
            "kappa":        med["kappa"],
            "roc_auc":      med["roc_auc"],
            "nc1": nc.nc1, "nc2": nc.nc2, "nc3": nc.nc3, "nc4": nc.nc4,
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        pd.DataFrame([summary]).to_csv(out_dir / "metrics.csv", index=False)
    except Exception as exc:
        _logger.warning(f"metrics.csv save failed: {exc}")

    # nc_metrics.csv
    try:
        if results.get("nc_history"):
            pd.DataFrame(results["nc_history"]).to_csv(
                out_dir / "nc_metrics.csv", index=False)
    except Exception as exc:
        _logger.warning(f"nc_metrics.csv save failed: {exc}")

    # All remaining rich artifacts via centralized reporter
    save_experiment_outputs(
        out_dir=out_dir,
        run_tag=run_tag,
        cfg=cfg,
        results=results,
        med=med,
        nc=nc,
        class_names=class_names,
        total_time_s=total_time_s,
        device=device,
    )

    print(f"  Results saved → {out_dir}")


if __name__ == "__main__":
    main()
