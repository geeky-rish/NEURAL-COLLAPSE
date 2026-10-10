"""
utils/experiment_reporter.py
─────────────────────────────────────────────────────────────────────────────
Stateless, post-hoc experiment output aggregator.

Responsibilities
----------------
- Aggregates already-computed metrics (med, nc, results, cfg) into files.
- NEVER recomputes metrics or runs forward passes.
- NEVER alters training, gradients, or checkpoints.
- Degrades gracefully when optional deps (matplotlib, sklearn) are missing.

Public API
----------
    from utils.experiment_reporter import save_experiment_outputs

    save_experiment_outputs(
        out_dir      = Path("results/etf_ham10000_s42"),
        run_tag      = "etf_ham10000_s42",
        cfg          = cfg,
        results      = trainer_results_dict,
        med          = medical_metrics_dict,
        nc           = nc_metrics_object,
        class_names  = ["Melanoma", "Nevi", ...],
        total_time_s = 142.3,
        device       = device,
    )

Outputs (all optional, fail-safe)
----------------------------------
- class_metrics.csv        per-class P/R/F1/sensitivity/specificity
- best_results.json        best epoch summary
- longtail_metrics.csv     head/mid/tail group analysis
- training_summary.json    epoch durations + throughput
- experiment_report.md     self-contained Markdown report
- run_info.json            env metadata (git, device, versions)
- confusion_matrix.png     auto-saved always (not just --visualize)
- per_class_recall.png     auto-saved always
- experiment_registry.csv  appended row in results/ root
"""
from __future__ import annotations

import csv
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger("experiment_reporter")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(fn, *args, label: str = "", **kwargs):
    """Call fn(*args, **kwargs) and log a warning on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        _log.warning(f"[reporter] {label or fn.__name__} failed (non-fatal): {exc}")
        return None


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Individual savers — each is self-contained and exception-safe
# ─────────────────────────────────────────────────────────────────────────────

def _save_class_metrics(out_dir: Path, med: Dict, class_names: List[str]):
    """Per-class precision, recall, F1, sensitivity, specificity → class_metrics.csv"""
    import pandas as pd

    rows = []
    for cls in class_names:
        rows.append({
            "class":       cls,
            "precision":   med.get("precision",    {}).get(cls, float("nan")),
            "recall":      med.get("recall",       {}).get(cls, float("nan")),
            "f1":          med.get("f1_per_class", {}).get(cls, float("nan")),
            "sensitivity": med.get("sensitivity",  {}).get(cls, float("nan")),
            "specificity": med.get("specificity",  {}).get(cls, float("nan")),
        })
    pd.DataFrame(rows).to_csv(out_dir / "class_metrics.csv", index=False)
    _log.info(f"  [reporter] class_metrics.csv → {out_dir}")


def _save_best_results(
    out_dir: Path,
    results: Dict,
    med: Dict,
    nc,
    total_time_s: float,
):
    """best_results.json — best epoch snapshot."""
    # Find best epoch by val_acc
    val_acc_list = results.get("val_acc", [])
    best_epoch = int(val_acc_list.index(max(val_acc_list))) + 1 if val_acc_list else -1

    payload = {
        "best_val_acc":      results.get("best_val_acc"),
        "best_epoch":        best_epoch,
        "macro_f1":          med.get("macro_f1"),
        "weighted_f1":       med.get("weighted_f1"),
        "mean_sensitivity":  med.get("mean_sensitivity"),
        "mean_specificity":  med.get("mean_specificity"),
        "kappa":             med.get("kappa"),
        "roc_auc":           med.get("roc_auc"),
        "nc1":               nc.nc1,
        "nc2":               nc.nc2,
        "nc3":               nc.nc3,
        "nc4":               nc.nc4,
        "total_training_time_s": total_time_s,
    }
    with open(out_dir / "best_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    _log.info(f"  [reporter] best_results.json → {out_dir}")


def _save_longtail_metrics(
    out_dir: Path,
    med: Dict,
    class_names: List[str],
    y_true,
):
    """longtail_metrics.csv — head/mid/tail recall by support-count terciles."""
    import numpy as np
    import pandas as pd

    C = len(class_names)
    counts = [int((y_true == c).sum()) for c in range(C)]
    sorted_idx = sorted(range(C), key=lambda i: counts[i], reverse=True)

    n = len(sorted_idx)
    head_end  = max(1, n // 3)
    tail_start = max(head_end + 1, n - n // 3)

    def group(idx):
        if idx < head_end:         return "head"
        if idx >= tail_start:      return "tail"
        return "mid"

    sensitivity = med.get("sensitivity", {})
    f1          = med.get("f1_per_class", {})
    rows = []
    for rank, cls_idx in enumerate(sorted_idx):
        cls = class_names[cls_idx]
        rows.append({
            "class":     cls,
            "support":   counts[cls_idx],
            "group":     group(rank),
            "recall":    sensitivity.get(cls, float("nan")),
            "f1":        f1.get(cls, float("nan")),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "longtail_metrics.csv", index=False)

    # Group summary
    summary = df.groupby("group")[["recall", "f1"]].mean().reset_index()
    _log.info(f"  [reporter] longtail_metrics.csv → {out_dir}")
    _log.info(f"  Long-tail group summary:\n{summary.to_string(index=False)}")


def _save_training_summary(
    out_dir: Path,
    results: Dict,
    run_tag: str,
    total_time_s: float,
):
    """training_summary.json — epoch durations + throughput."""
    epoch_times = results.get("epoch_times", [])
    payload = {
        "run_tag":          run_tag,
        "total_epochs":     len(results.get("train_loss", [])),
        "total_time_s":     total_time_s,
        "avg_epoch_time_s": float(sum(epoch_times) / len(epoch_times)) if epoch_times else None,
        "min_epoch_time_s": float(min(epoch_times)) if epoch_times else None,
        "max_epoch_time_s": float(max(epoch_times)) if epoch_times else None,
        "epoch_times_s":    [round(t, 2) for t in epoch_times],
        "train_loss":       results.get("train_loss", []),
        "train_acc":        results.get("train_acc", []),
        "val_acc":          results.get("val_acc", []),
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(payload, f, indent=2)
    _log.info(f"  [reporter] training_summary.json → {out_dir}")


def _save_run_info(
    out_dir: Path,
    cfg: Dict,
    run_tag: str,
    total_time_s: float,
    device=None,
):
    """run_info.json — environment metadata for reproducibility."""
    import torch

    # Prefer the pre-computed device_info injected by train.py via cfg["_device_info"]
    device_info_dict = cfg.get("_device_info", {})

    backend       = device_info_dict.get("backend", str(getattr(device, "type", "cpu")))
    cuda_available = device_info_dict.get("cuda_available", torch.cuda.is_available())
    mps_available  = device_info_dict.get("mps_available",
                         hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    payload = {
        "run_tag":          run_tag,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "hostname":         socket.gethostname(),
        "git_commit":       _git_hash(),
        "python_version":   device_info_dict.get("python_version",  sys.version),
        "pytorch_version":  device_info_dict.get("pytorch_version", torch.__version__),
        # Hardware backend
        "backend":          backend,
        "device":           device_info_dict.get("device", str(device)),
        "cuda_available":   cuda_available,
        "cuda_version":     device_info_dict.get("cuda_version", torch.version.cuda),
        "mps_available":    mps_available,
        "gpu_name":         device_info_dict.get("gpu_name"),
        "gpu_memory_gb":    device_info_dict.get("gpu_memory_gb"),
        "estimated_ram_gb": device_info_dict.get("estimated_ram_gb"),
        "cpu_count":        device_info_dict.get("cpu_count"),
        "hardware_profile": cfg.get("_profile"),
        # Experiment metadata
        "total_time_s":     total_time_s,
        "dataset":          cfg.get("dataset", {}).get("name"),
        "method":           cfg.get("model",   {}).get("head"),
        "backbone":         cfg.get("model",   {}).get("backbone"),
        "epochs":           cfg.get("training", {}).get("epochs"),
        "batch_size":       cfg.get("training", {}).get("batch_size"),
        "lr":               cfg.get("training", {}).get("lr"),
        "num_classes":      cfg.get("dataset", {}).get("num_classes"),
        "imbalance_ratio":  cfg.get("dataset", {}).get("imbalance_ratio", 1),
        "mixed_precision":  cfg.get("training", {}).get("mixed_precision", "auto"),
    }

    with open(out_dir / "run_info.json", "w") as f:
        json.dump(payload, f, indent=2)
    _log.info(f"  [reporter] run_info.json → {out_dir}")


def _save_experiment_report(
    out_dir: Path,
    run_tag: str,
    cfg: Dict,
    results: Dict,
    med: Dict,
    nc,
    class_names: List[str],
    total_time_s: float,
):
    """experiment_report.md — self-contained Markdown report."""
    val_acc_list = results.get("val_acc", [])
    best_epoch   = int(val_acc_list.index(max(val_acc_list))) + 1 if val_acc_list else "N/A"
    train_cfg    = cfg.get("training", {})
    dataset_cfg  = cfg.get("dataset", {})

    lines = [
        f"# Experiment Report: `{run_tag}`\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Git Commit:** {_git_hash()}\n",
        "---\n",
        "## Configuration\n",
        f"| Key | Value |",
        f"|-----|-------|",
        f"| Dataset | `{dataset_cfg.get('name')}` |",
        f"| Method / Head | `{cfg.get('model', {}).get('head')}` |",
        f"| Backbone | `{cfg.get('model', {}).get('backbone')}` |",
        f"| Epochs | {train_cfg.get('epochs')} |",
        f"| Batch Size | {train_cfg.get('batch_size')} |",
        f"| LR | {train_cfg.get('lr')} |",
        f"| LR Schedule | {train_cfg.get('lr_schedule')} |",
        f"| Num Classes | {dataset_cfg.get('num_classes')} |",
        "",
        "## Training Summary\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Time | {total_time_s:.1f}s ({total_time_s/60:.1f} min) |",
        f"| Epochs Completed | {len(results.get('train_loss', []))} |",
        f"| Best Val Accuracy | **{results.get('best_val_acc', 0):.2f}%** |",
        f"| Best Epoch | {best_epoch} |",
        "",
        "## Final Medical Metrics\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Macro F1 | {med.get('macro_f1', float('nan')):.4f} |",
        f"| Weighted F1 | {med.get('weighted_f1', float('nan')):.4f} |",
        f"| Mean Sensitivity | {med.get('mean_sensitivity', float('nan')):.4f} |",
        f"| Mean Specificity | {med.get('mean_specificity', float('nan')):.4f} |",
        f"| Cohen Kappa | {med.get('kappa', float('nan')):.4f} |",
        f"| ROC-AUC | {med.get('roc_auc', float('nan')):.4f} |",
        "",
        "## Per-Class Performance\n",
        "| Class | Precision | Recall | F1 |",
        "|-------|-----------|--------|----|",
    ]
    for cls in class_names:
        p = med.get("precision",    {}).get(cls, float("nan"))
        r = med.get("recall",       {}).get(cls, float("nan"))
        f = med.get("f1_per_class", {}).get(cls, float("nan"))
        lines.append(f"| {cls} | {p:.3f} | {r:.3f} | {f:.3f} |")

    lines += [
        "",
        "## Neural Collapse Metrics (Final)\n",
        f"| Metric | Value | Interpretation |",
        f"|--------|-------|----------------|",
        f"| NC1 (within-class scatter) | {nc.nc1:.6f} | lower = more collapsed |",
        f"| NC2 (ETF deviation) | {nc.nc2:.8f} | lower = closer to ETF |",
        f"| NC3 (W–μ alignment) | {nc.nc3:.6f} | lower = better aligned |",
        f"| NC4 (NCC disagreement) | {nc.nc4:.6f} | lower = NCC matches argmax |",
        "",
        "## Artifacts\n",
        "| File | Description |",
        "|------|-------------|",
        "| `config_snapshot.yaml` | Full resolved config |",
        "| `metrics.csv` | Final summary metrics |",
        "| `nc_metrics.csv` | NC1-NC4 per tracked epoch |",
        "| `class_metrics.csv` | Per-class P/R/F1 |",
        "| `best_results.json` | Best epoch summary |",
        "| `longtail_metrics.csv` | Head/mid/tail analysis |",
        "| `training_summary.json` | Epoch timing |",
        "| `run_info.json` | Environment metadata |",
        "| `confusion_matrix.png` | Confusion matrix plots |",
        "| `per_class_recall.png` | Per-class recall bars |",
        "",
        "---",
        f"*Auto-generated by `utils/experiment_reporter.py`*",
    ]

    (out_dir / "experiment_report.md").write_text("\n".join(lines))
    _log.info(f"  [reporter] experiment_report.md → {out_dir}")


def _save_confusion_plots(
    out_dir: Path,
    med: Dict,
    class_names: List[str],
    run_tag: str,
):
    """Confusion matrix and per-class recall — always saved (no --visualize required)."""
    import numpy as np
    from visualization.confusion_analysis import (
        plot_confusion_matrix, plot_per_class_recall
    )

    cm = np.array(med["confusion_matrix"])
    plot_confusion_matrix(
        cm, class_names=class_names,
        save_path=str(out_dir / "confusion_matrix.png"),
        title=f"Confusion Matrix — {run_tag}",
    )
    plot_per_class_recall(
        med["sensitivity"],
        save_path=str(out_dir / "per_class_recall.png"),
        title=f"Per-Class Recall — {run_tag}",
    )


def _update_registry(
    results_root: Path,
    run_tag: str,
    cfg: Dict,
    results: Dict,
    med: Dict,
    nc,
    total_time_s: float,
    ckpt_dir: Path,
):
    """Append one row to results/experiment_registry.csv (file-lock safe)."""
    registry_path = results_root / "experiment_registry.csv"

    fieldnames = [
        "experiment_id", "timestamp", "dataset", "method", "backbone",
        "imbalance_ratio", "epochs", "best_val_acc", "macro_f1", "kappa", "roc_auc",
        "nc1", "nc2", "nc3", "nc4", "total_time_s", "checkpoint_dir",
    ]
    row = {
        "experiment_id": run_tag,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":       cfg.get("dataset", {}).get("name"),
        "method":        cfg.get("model", {}).get("head"),
        "backbone":      cfg.get("model", {}).get("backbone"),
        "imbalance_ratio": cfg.get("dataset", {}).get("imbalance_ratio", 1),
        "epochs":        cfg.get("training", {}).get("epochs"),
        "best_val_acc":  round(results.get("best_val_acc", 0), 4),
        "macro_f1":      round(med.get("macro_f1", float("nan")), 4),
        "kappa":         round(med.get("kappa", float("nan")), 4),
        "roc_auc":       round(med.get("roc_auc", float("nan")), 4),
        "nc1":           round(nc.nc1, 6),
        "nc2":           round(nc.nc2, 8),
        "nc3":           round(nc.nc3, 6),
        "nc4":           round(nc.nc4, 6),
        "total_time_s":  round(total_time_s, 1),
        "checkpoint_dir": str(ckpt_dir),
    }

    write_header = not registry_path.exists()
    with open(registry_path, "a", newline="") as f:
        try:
            if HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_EX)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        finally:
            if HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_UN)
    _log.info(f"  [reporter] experiment_registry.csv updated → {registry_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry-point
# ─────────────────────────────────────────────────────────────────────────────

def save_experiment_outputs(
    out_dir: Path,
    run_tag: str,
    cfg: Dict[str, Any],
    results: Dict[str, Any],
    med: Dict[str, Any],
    nc,
    class_names: Optional[List[str]],
    total_time_s: float,
    device=None,
) -> None:
    """
    Write all experiment artifacts. Stateless and safe to call at any point
    after training completes.

    All individual savers are wrapped in try/except — failures are logged as
    warnings and never propagate to the caller.

    Parameters
    ----------
    out_dir       : experiment output directory (created if missing)
    run_tag       : unique experiment identifier string
    cfg           : full resolved config dict
    results       : dict returned by Trainer.run()
    med           : dict returned by compute_medical_metrics()
    nc            : NCMetrics dataclass from compute_all_nc_metrics()
    class_names   : list of string class names, or None
    total_time_s  : wall-clock training time in seconds
    device        : torch.device or None
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        num_classes = cfg.get("dataset", {}).get("num_classes", 0)
        class_names = [f"Class {c}" for c in range(num_classes)]

    import numpy as np
    y_true_list = results.get("_y_true", None)  # injected by train.py if available

    ckpt_dir = (
        Path(cfg.get("logging", {}).get("checkpoint_dir", "./checkpoints")) / run_tag
    )
    results_root = Path(cfg.get("logging", {}).get("results_dir", "./results"))

    _log.info(f"[reporter] Saving experiment artifacts → {out_dir}")

    _safe(_save_class_metrics,    out_dir, med, class_names,
          label="class_metrics.csv")
    _safe(_save_best_results,     out_dir, results, med, nc, total_time_s,
          label="best_results.json")
    _safe(_save_training_summary, out_dir, results, run_tag, total_time_s,
          label="training_summary.json")
    _safe(_save_run_info,         out_dir, cfg, run_tag, total_time_s, device,
          label="run_info.json")
    _safe(_save_experiment_report,out_dir, run_tag, cfg, results, med, nc,
          class_names, total_time_s,
          label="experiment_report.md")
    _safe(_save_confusion_plots,  out_dir, med, class_names, run_tag,
          label="confusion plots")

    # Long-tail requires y_true
    if y_true_list is not None:
        _safe(_save_longtail_metrics, out_dir, med, class_names,
              np.array(y_true_list), label="longtail_metrics.csv")
    else:
        _log.warning("[reporter] y_true not in results — skipping longtail_metrics.csv")

    _safe(_update_registry, results_root, run_tag, cfg, results, med, nc,
          total_time_s, ckpt_dir, label="experiment_registry.csv")

    _log.info(f"[reporter] All artifacts saved → {out_dir}")
