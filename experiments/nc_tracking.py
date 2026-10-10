"""
experiments/nc_tracking.py
─────────────────────────────────────────────────────────────────────────────
Train a model while logging NC1–NC4 metrics every N epochs.

Purpose
-------
Understand *when* during training Neural Collapse emerges (or fails to emerge)
under different methods and imbalance ratios. This gives insight into:

  • Does NC emerge earlier with ETF / prototype heads?
  • Does imbalance delay or prevent NC even in the terminal phase?
  • Which NC properties degrade first as imbalance increases?

Output
------
  results/<run_tag>/nc_evolution.csv   — per-epoch NC metrics
  results/<run_tag>/nc_evolution.png   — 4-panel NC metric plots
  results/<run_tag>/tsne_epochXXXX.png — (optional) feature evolution frames
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from utils.device import get_best_device
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from data import get_dataloaders, get_medical_dataloaders
from models import build_model
from training import get_criterion, build_scheduler
from training.nc_regularization import CombinedNCLoss
from evaluation.nc_metrics import compute_all_nc_metrics
from evaluation.evaluator import extract_features
from utils.seed import set_seed
from utils.logging_utils import get_logger
from visualization.feature_geometry import plot_nc_evolution
from visualization.tsne_visualizer import plot_tsne

_logger = get_logger("nc_tracking")


def run_nc_tracking(
    cfg: dict,
    method: str          = "baseline",
    imbalance_ratio: int = 10,
    seed: int            = 42,
    track_every: int     = 5,
    save_tsne: bool      = False,
) -> pd.DataFrame:
    """
    Train model, recording all NC metrics every *track_every* epochs.

    Parameters
    ----------
    cfg             : full project config
    method          : training method
    imbalance_ratio : dataset imbalance ratio (CIFAR only)
    seed            : random seed
    track_every     : log NC metrics every N epochs
    save_tsne       : if True, save t-SNE frames each tracking step

    Returns
    -------
    DataFrame with columns [epoch, nc1, nc2, nc3, nc4, val_acc]
    """
    set_seed(seed)
    device = get_best_device()[0]

    dataset_name = cfg["dataset"]["name"].lower()
    run_tag      = f"nc_track_{method}_r{imbalance_ratio}_s{seed}"
    save_dir     = Path(cfg["logging"]["results_dir"]) / run_tag
    save_dir.mkdir(parents=True, exist_ok=True)

    _logger.info(f"── NC Tracking | method={method} ratio={imbalance_ratio} seed={seed} ──")

    # ── Data ──────────────────────────────────────────────────────────────────
    if dataset_name == "cifar10":
        train_loader, val_loader, class_weights = get_dataloaders(
            cfg, imbalance_ratio=imbalance_ratio, method=method, seed=seed)
    else:
        train_loader, val_loader, class_weights = get_medical_dataloaders(cfg, seed=seed)

    # ── Model + criterion ─────────────────────────────────────────────────────
    model = build_model(cfg, method=method).to(device)

    from training.losses import get_criterion as _get_crit
    base_crit = _get_crit(method, class_weights, cfg, device)

    nc_cfg  = cfg.get("nc_regularization", {})
    use_reg = nc_cfg.get("enabled", False)
    if use_reg:
        criterion = CombinedNCLoss(
            base_criterion   = base_crit,
            num_classes      = cfg["dataset"]["num_classes"],
            collapse_weight  = nc_cfg.get("collapse_weight", 0.01),
            etf_align_weight = nc_cfg.get("etf_align_weight", 0.01),
        )
    else:
        criterion = base_crit

    train_cfg = cfg["training"]
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(train_cfg["lr"]),
        momentum=float(train_cfg.get("momentum", 0.9)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        nesterov=True,
    )
    scheduler = build_scheduler(optimizer, cfg)

    # ── Training loop with NC tracking ────────────────────────────────────────
    history: List[Dict] = []
    num_classes = cfg["dataset"]["num_classes"]

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        # Train
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            features = model.forward_features(images)
            logits   = model.fc(features)

            if use_reg and isinstance(criterion, CombinedNCLoss):
                loss = criterion(logits, features, labels)
            else:
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
        scheduler.step()

        # Track NC every N epochs
        if epoch % track_every == 0 or epoch == cfg["training"]["epochs"]:
            model.eval()
            with torch.no_grad():
                feats, lbls, lgts = _extract_all(model, val_loader, device)

            nc = compute_all_nc_metrics(feats, lbls, num_classes, model, lgts, epoch)
            val_acc = _quick_accuracy(lgts, lbls)

            row = {**nc.to_dict(), "val_acc": val_acc}
            history.append(row)

            _logger.info(
                f"  ep {epoch:3d} | NC1={nc.nc1:.4f} NC2={nc.nc2:.6f} "
                f"NC3={nc.nc3:.4f} NC4={nc.nc4:.4f} | acc={val_acc:.1f}%"
            )

            if save_tsne:
                tsne_path = str(save_dir / f"tsne_epoch{epoch:04d}.png")
                try:
                    from visualization.tsne_visualizer import plot_tsne as _pt
                    _pt(feats, lbls, save_path=tsne_path,
                        title=f"t-SNE ({method})", epoch=epoch, dpi=120)
                except Exception as e:
                    _logger.warning(f"t-SNE failed at epoch {epoch}: {e}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    df = pd.DataFrame(history)
    df.to_csv(save_dir / "nc_evolution.csv", index=False)
    _logger.info(f"  CSV saved → {save_dir / 'nc_evolution.csv'}")

    plot_nc_evolution(history, save_path=str(save_dir / "nc_evolution.png"))

    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def _extract_all(model, loader, device):
    model.eval()
    all_f, all_l, all_g = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        feats  = model.forward_features(images)
        logits = model.fc(feats)
        all_f.append(feats.cpu())
        all_l.append(labels)
        all_g.append(logits.cpu())
    return torch.cat(all_f), torch.cat(all_l), torch.cat(all_g)


def _quick_accuracy(logits, labels):
    preds = logits.argmax(1)
    return (preds == labels).float().mean().item() * 100.0


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--method",  default="baseline")
    p.add_argument("--ratio",   type=int, default=10)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--every",   type=int, default=5)
    p.add_argument("--tsne",    action="store_true")
    p.add_argument("--config",  default=None)
    p.add_argument("--override", nargs="*", default=None)
    args = p.parse_args()

    cfg = load_config(config_path=args.config, overrides=args.override)
    df  = run_nc_tracking(cfg, args.method, args.ratio, args.seed,
                          args.every, args.tsne)
    print(df.to_string(index=False))
