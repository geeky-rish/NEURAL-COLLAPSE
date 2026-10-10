"""
utils/logging_utils.py
─────────────────────────────────────────────────────────────────────────────
Provides:
  • get_logger  — returns a configured Python logger that writes to both
                  the console and a rotating log file.
  • AverageMeter — lightweight class for tracking running statistics
                   (loss, accuracy) during an epoch.
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


# ── Logger ────────────────────────────────────────────────────────────────────

def get_logger(
    name: str = "dl_proj",
    log_dir: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a logger that writes formatted messages to stdout and optionally
    to a rotating file in *log_dir*.

    Parameters
    ----------
    name    : str   — logger name (use module __name__ for per-module loggers)
    log_dir : str   — directory for the log file; None → no file handler
    level   : int   — logging level (default INFO)
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            filename=os.path.join(log_dir, f"{name}.log"),
            maxBytes=5 * 1024 * 1024,   # 5 MB
            backupCount=3,
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── AverageMeter ──────────────────────────────────────────────────────────────

class AverageMeter:
    """
    Tracks the running average and current value of a scalar metric.

    Usage
    -----
    meter = AverageMeter("Loss")
    for batch in loader:
        loss = criterion(...)
        meter.update(loss.item(), n=batch_size)
    print(meter)   # Loss: 0.3241 (avg 0.3412)
    """

    def __init__(self, name: str = "", fmt: str = ":.4f") -> None:
        self.name = name
        self.fmt  = fmt
        self.reset()

    def reset(self) -> None:
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count else 0.0

    def __repr__(self) -> str:
        fmtstr = "{name}: {val" + self.fmt + "} (avg {avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
