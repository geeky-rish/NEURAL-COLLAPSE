"""
utils/device.py
─────────────────────────────────────────────────────────────────────────────
Centralized hardware device selection and backend metadata.

Priority order: CUDA → MPS (Apple Silicon) → CPU

All device selection in the framework should route through get_best_device().
This keeps the rest of the codebase backend-agnostic.

Usage
-----
    from utils.device import get_best_device, DeviceInfo

    device, info = get_best_device(cfg)
    print(info.backend)      # "cuda" | "mps" | "cpu"
    print(info.to_dict())    # full metadata for run_info.json
"""
from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class DeviceInfo:
    """Backend metadata captured at device selection time."""
    device:           torch.device = field(default_factory=lambda: torch.device("cpu"))
    backend:          str  = "cpu"
    cuda_available:   bool = False
    mps_available:    bool = False
    cuda_version:     Optional[str] = None
    gpu_name:         Optional[str] = None
    gpu_memory_gb:    Optional[float] = None
    cpu_count:        int = 1
    platform:         str = ""
    pytorch_version:  str = ""
    python_version:   str = ""
    # Unified-memory estimate (MPS / CPU) in GB — None means not applicable
    estimated_ram_gb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["device"] = str(self.device)
        return d

    def supports_amp(self) -> bool:
        """True only when CUDA is active (AMP is stable on CUDA)."""
        return self.backend == "cuda"

    def pin_memory(self) -> bool:
        """True only for CUDA — avoids UserWarning on CPU/MPS."""
        return self.backend == "cuda"

    def recommended_workers(self, configured: int) -> int:
        """
        Return a safe num_workers value for the active backend.

        CUDA  → use configured value (can be high)
        MPS   → cap at 4 to avoid shared-memory issues
        CPU   → cap at 2 to avoid thrashing
        """
        if self.backend == "cuda":
            return configured
        elif self.backend == "mps":
            return min(configured, 4)
        else:
            return min(configured, 2)


def get_best_device(cfg: Optional[Dict[str, Any]] = None) -> tuple[torch.device, DeviceInfo]:
    """
    Detect and return the best available device with backend metadata.

    Priority: CUDA > MPS > CPU

    Parameters
    ----------
    cfg : optional config dict — reads system.force_device if present

    Returns
    -------
    (torch.device, DeviceInfo)
    """
    # Allow config/CLI to force a specific device
    forced = None
    if cfg is not None:
        forced = cfg.get("system", {}).get("force_device", None)

    cuda_ok = torch.cuda.is_available()
    mps_ok  = (
        hasattr(torch.backends, "mps") and
        torch.backends.mps.is_available()
    )

    if forced:
        device = torch.device(forced)
    elif cuda_ok:
        device = torch.device("cuda")
    elif mps_ok:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    backend = device.type  # "cuda" | "mps" | "cpu"

    info = DeviceInfo(
        device          = device,
        backend         = backend,
        cuda_available  = cuda_ok,
        mps_available   = mps_ok,
        cpu_count       = _cpu_count(),
        platform        = platform.platform(),
        pytorch_version = torch.__version__,
        python_version  = sys.version,
    )

    # Enrich with CUDA specifics
    if cuda_ok:
        try:
            info.cuda_version  = torch.version.cuda
            info.gpu_name      = torch.cuda.get_device_name(0)
            info.gpu_memory_gb = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        except Exception:
            pass

    # Estimate unified memory for MPS/CPU from psutil if available
    if backend in ("mps", "cpu"):
        try:
            import psutil
            info.estimated_ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
        except ImportError:
            pass

    return device, info


def _cpu_count() -> int:
    try:
        import os
        return os.cpu_count() or 1
    except Exception:
        return 1
