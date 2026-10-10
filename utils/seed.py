"""
utils/seed.py
─────────────────────────────────────────────────────────────────────────────
Centralised seed-setting for full reproducibility across random, numpy,
torch (CPU + GPU), and cuDNN.
"""
from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Fix all known random-number generators to *seed* for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # for multi-GPU setups

    # Ensure deterministic algorithms (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
