"""
run_sweep.py — CLI entry point for full experiment sweeps.
Updated for the NC-MedAI Medical AI Research Framework.

Sweep modes
-----------
imbalance   Phase 1 — vary imbalance ratio for one method
method      Phase 2 — compare all methods at fixed ratio
nc          NC tracking — log NC1–NC4 per epoch for one method
etf         ETF vs Linear vs Prototype head comparison
ablation    Ablation studies (backbone / nc_reg / etf_scale / sampling)

Usage
-----
# Phase 1 — imbalance sweep, baseline, CIFAR-10
python run_sweep.py --phase imbalance --method baseline --plot

# Phase 2 — method comparison at ratio 10
python run_sweep.py --phase method --ratio 10 --plot

# NC tracking (HAM10000, focal, every 5 epochs, with t-SNE)
python run_sweep.py --phase nc --method focal --tsne \
    --override dataset.name=ham10000 evaluation.track_nc_every_n_epochs=5

# ETF vs Linear vs Prototype
python run_sweep.py --phase etf

# All ablations
python run_sweep.py --phase ablation --axis all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from utils.logging_utils import get_logger

_logger = get_logger("run_sweep")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NC-MedAI: Run full experiment sweeps."
    )
    p.add_argument("--config", default=None)
    p.add_argument(
        "--phase", required=True,
        choices=["imbalance", "method", "nc", "etf", "ablation"],
        help="Sweep phase to run",
    )
    # Phase-specific
    p.add_argument("--method", default="baseline",
                   help="Training method (for imbalance / nc phases)")
    p.add_argument("--ratio",  type=int, default=10,
                   help="Imbalance ratio (for method phase)")
    p.add_argument("--plot",   action="store_true",
                   help="Generate plots after sweep")
    p.add_argument("--tsne",   action="store_true",
                   help="Save t-SNE frames (nc phase only)")
    p.add_argument("--every",  type=int, default=5,
                   help="NC tracking interval in epochs (nc phase)")
    p.add_argument(
        "--axis", default="all",
        choices=["all", "backbone", "nc_reg", "etf_scale", "sampling"],
        help="Ablation axis (ablation phase only)",
    )
    p.add_argument("--override", nargs="*", default=None, metavar="KEY=VALUE")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(config_path=args.config, overrides=args.override)

    # ── Phase: Imbalance sweep ────────────────────────────────────────────────
    if args.phase == "imbalance":
        from experiments.sweep import run_imbalance_sweep
        from evaluation.visualize import plot_imbalance_sweep

        _logger.info(f"── Phase: Imbalance sweep | method={args.method} ──")
        df = run_imbalance_sweep(cfg, method=args.method)
        print(df.to_string(index=False))

        if args.plot:
            from evaluation.visualize import plot_imbalance_sweep
            plot_imbalance_sweep(df,
                save_dir=cfg["logging"]["results_dir"],
                prefix=args.method)

    # ── Phase: Method comparison ──────────────────────────────────────────────
    elif args.phase == "method":
        from experiments.sweep import run_method_sweep
        from evaluation.visualize import plot_method_comparison, plot_nc_scatter

        _logger.info(f"── Phase: Method sweep | ratio={args.ratio} ──")
        df = run_method_sweep(cfg, imbalance_ratio=args.ratio)
        print(df.to_string(index=False))

        if args.plot:
            save_dir = cfg["logging"]["results_dir"]
            plot_method_comparison(df, save_dir=save_dir,
                                   prefix=f"comparison_r{args.ratio}")
            # NC scatter requires method column + nc1_mean + acc_mean
            if "method" in df.columns and "nc1_mean" in df.columns:
                df_scatter = df.rename(columns={
                    "nc1_mean": "nc1_mean",
                    "acc_mean": "acc_mean",
                })
                # Add dummy ratio column for scatter function
                df_scatter["imbalance_ratio"] = args.ratio
                plot_nc_scatter(df_scatter, save_dir=save_dir,
                                prefix=f"scatter_r{args.ratio}")

    # ── Phase: NC tracking ────────────────────────────────────────────────────
    elif args.phase == "nc":
        from experiments.nc_tracking import run_nc_tracking

        _logger.info(f"── Phase: NC Tracking | method={args.method} ──")
        df = run_nc_tracking(
            cfg,
            method=args.method,
            imbalance_ratio=args.ratio,
            seed=cfg.get("seeds", [42])[0],
            track_every=args.every,
            save_tsne=args.tsne,
        )
        print(df.to_string(index=False))

        if args.plot:
            from visualization.feature_geometry import plot_nc_evolution
            plot_nc_evolution(
                df.to_dict("records"),
                save_path=f"{cfg['logging']['results_dir']}/nc_evolution_{args.method}.png",
                title=f"NC Evolution — {args.method}",
            )

    # ── Phase: ETF vs Linear comparison ──────────────────────────────────────
    elif args.phase == "etf":
        from experiments.etf_vs_linear import run_etf_vs_linear

        _logger.info("── Phase: ETF vs Linear vs Prototype ──")
        df = run_etf_vs_linear(
            cfg,
            imbalance_ratio=args.ratio,
            seeds=cfg.get("seeds", [42]),
        )
        print(df.to_string(index=False))

    # ── Phase: Ablation studies ───────────────────────────────────────────────
    elif args.phase == "ablation":
        import torch
        from utils.device import get_best_device
        from experiments.ablation_studies import (
            run_backbone_ablation,
            run_nc_reg_ablation,
            run_etf_scale_ablation,
            run_sampling_ablation,
        )
        dev, _ = get_best_device(cfg)
        seeds = cfg.get("seeds", [42])

        _logger.info(f"── Phase: Ablation | axis={args.axis} ──")
        if args.axis in ("backbone", "all"):
            run_backbone_ablation(cfg, seeds=seeds, device=dev)
        if args.axis in ("nc_reg", "all"):
            run_nc_reg_ablation(cfg, seeds=seeds, device=dev)
        if args.axis in ("etf_scale", "all"):
            run_etf_scale_ablation(cfg, seeds=seeds, device=dev)
        if args.axis in ("sampling", "all"):
            run_sampling_ablation(cfg, seeds=seeds, device=dev)

    _logger.info("Sweep complete.")


if __name__ == "__main__":
    main()
