"""
training/scheduler.py
─────────────────────────────────────────────────────────────────────────────
Learning-rate scheduler factory.

Supported schedules (set via config.yaml → training.lr_schedule):
  • cosine  — CosineAnnealingLR with optional linear warm-up
  • step    — StepLR (decays by 0.1 at epochs 100 and 150)
  • none    — constant LR
"""
from __future__ import annotations

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    StepLR,
    LambdaLR,
)


def build_scheduler(
    optimizer: optim.Optimizer,
    cfg: dict,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build and return the LR scheduler specified in *cfg*.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    cfg       : dict — full project config

    Returns
    -------
    scheduler : LRScheduler instance
    """
    schedule      = cfg["training"].get("lr_schedule", "cosine").lower()
    total_epochs  = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"].get("warmup_epochs", 0)

    if schedule == "cosine":
        # Linear warm-up → cosine decay
        if warmup_epochs > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=1e-4,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            t_max = max(1, total_epochs - warmup_epochs)
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=1e-6,
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    elif schedule == "step":
        # Step decay: ×0.1 at epochs 100 and 150
        milestones = [100, 150]
        gamma = 0.1

        def lr_lambda(epoch: int) -> float:
            factor = 1.0
            for m in milestones:
                if epoch >= m:
                    factor *= gamma
            return factor

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif schedule == "none":
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    else:
        raise ValueError(
            f"Unknown lr_schedule: {schedule!r}. "
            "Choose from ['cosine', 'step', 'none']."
        )
