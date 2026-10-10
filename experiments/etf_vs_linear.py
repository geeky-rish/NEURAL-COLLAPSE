"""
experiments/etf_vs_linear.py
─────────────────────────────────────────────────────────────────────────────
Head-to-head comparison: Standard Linear Classifier vs ETF vs Prototype.

Evaluates each classifier head on the SAME backbone (same init seed, same
training schedule) to isolate the effect of the head geometry.

Metrics compared
----------------
  • Overall accuracy
  • Minority class sensitivity (recall)
  • Macro F1
  • NC1, NC2, NC3, NC4
  • Feature geometry (cosine heatmap, PCA)

Output
------
  results/etf_vs_linear/comparison_table.csv
  results/etf_vs_linear/comparison_bar.png
  results/etf_vs_linear/cosine_heatmap_{head}.png
  results/etf_vs_linear/pca_{head}.png
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from utils.device import get_best_device

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from data import get_dataloaders, get_medical_dataloaders
from models import build_model
from training import get_criterion, Trainer
from evaluation.nc_metrics import compute_all_nc_metrics
from evaluation.medical_metrics import compute_medical_metrics
from evaluation.evaluator import extract_features
from utils.seed import set_seed
from utils.logging_utils import get_logger
from evaluation.visualize import plot_method_comparison
from visualization.feature_geometry import plot_cosine_heatmap, plot_pca

_logger = get_logger("etf_vs_linear")

HEADS = ["linear", "etf", "prototype"]


def run_etf_vs_linear(
    cfg: dict,
    imbalance_ratio: int = 10,
    seeds: List[int] | None = None,
) -> pd.DataFrame:
    """
    Compare linear / ETF / prototype heads under identical conditions.

    Parameters
    ----------
    cfg             : full project config
    imbalance_ratio : dataset imbalance for CIFAR experiments
    seeds           : list of random seeds (averaged)

    Returns
    -------
    DataFrame with per-head aggregated metrics
    """
    seeds     = seeds or cfg.get("seeds", [42])
    device    = get_best_device()[0]
    save_dir  = Path(cfg["logging"]["results_dir"]) / "etf_vs_linear"
    save_dir.mkdir(parents=True, exist_ok=True)
    num_classes = cfg["dataset"]["num_classes"]
    dataset_name = cfg["dataset"]["name"].lower()

    rows = []

    for head in HEADS:
        seed_results = []

        for seed in seeds:
            set_seed(seed)
            _logger.info(f"── head={head} seed={seed} ──")

            # Build data
            if dataset_name == "cifar10":
                train_loader, val_loader, cw = get_dataloaders(
                    cfg, imbalance_ratio=imbalance_ratio, method=head, seed=seed)
            else:
                train_loader, val_loader, cw = get_medical_dataloaders(cfg, seed=seed)

            # Build model with specified head
            cfg["model"]["head"] = head
            model = build_model(cfg, method=head)

            # Train via Trainer
            run_tag  = f"etf_vs_linear_{head}_s{seed}"
            criterion = get_criterion("focal", cw, cfg, device)

            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                class_weights=cw, cfg=cfg, method=head,
                seed=seed, device=device, run_tag=run_tag,
            )
            results = trainer.run()

            # Full evaluation
            model.eval()
            feats, lbls, lgts = _extract_with_logits(model, val_loader, device)
            y_true = lbls.numpy()
            y_pred = lgts.argmax(1).numpy()
            y_prob = torch.softmax(lgts, dim=1).numpy()

            # Get class names if available
            cn = getattr(val_loader.dataset, "class_names", None)
            med = compute_medical_metrics(y_true, y_pred, y_prob,
                                          class_names=cn, num_classes=num_classes)
            nc  = compute_all_nc_metrics(feats, lbls, num_classes, model, lgts)

            seed_results.append({
                "head":      head,
                "seed":      seed,
                "acc":       results["best_val_acc"],
                "macro_f1":  med["macro_f1"],
                "sensitivity": med["mean_sensitivity"],
                "nc1": nc.nc1, "nc2": nc.nc2,
                "nc3": nc.nc3, "nc4": nc.nc4,
            })

            # Per-head visualizations (from last seed)
            plot_cosine_heatmap(
                feats, lbls, num_classes, class_names=cn,
                save_path=str(save_dir / f"cosine_heatmap_{head}.png"),
                title=f"Class-Mean Cosine Similarity — {head.upper()} head",
            )
            plot_pca(
                feats, lbls, class_names=cn,
                save_path=str(save_dir / f"pca_{head}.png"),
                title=f"PCA Feature Space — {head.upper()} head",
            )

        # Aggregate across seeds
        def _agg(key):
            vals = np.array([r[key] for r in seed_results])
            return vals.mean(), vals.std()

        row = {"method": head}
        for key in ["acc", "macro_f1", "sensitivity", "nc1", "nc2", "nc3", "nc4"]:
            m, s = _agg(key)
            row[f"{key}_mean"] = m
            row[f"{key}_std"]  = s
        rows.append(row)
        _logger.info(f"  {head}: acc={row['acc_mean']:.2f}% "
                     f"F1={row['macro_f1_mean']:.3f} NC1={row['nc1_mean']:.4f}")

    df = pd.DataFrame(rows)
    csv_path = save_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    _logger.info(f"Saved → {csv_path}")

    # Bar chart
    plot_method_comparison(
        df.rename(columns={"acc_mean": "acc_mean",
                            "nc1_mean": "nc1_mean",
                            "nc2_mean": "nc2_mean",
                            "acc_std":  "acc_std",
                            "nc1_std":  "nc1_std",
                            "nc2_std":  "nc2_std"}),
        save_dir=str(save_dir),
        prefix="comparison",
    )
    print("\n", df.to_string(index=False))
    return df


@torch.no_grad()
def _extract_with_logits(model, loader, device):
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


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ratio",   type=int, default=10)
    p.add_argument("--config",  default=None)
    p.add_argument("--override", nargs="*", default=None)
    args = p.parse_args()
    cfg = load_config(config_path=args.config, overrides=args.override)
    run_etf_vs_linear(cfg, imbalance_ratio=args.ratio)
