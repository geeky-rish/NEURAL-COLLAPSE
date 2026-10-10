"""
evaluation/__init__.py  (updated)
"""
from .evaluator import evaluate_checkpoint, extract_features
from .nc_metrics import compute_all_nc_metrics, NCMetrics
from .medical_metrics import compute_medical_metrics, print_medical_metrics
from .visualize import (
    plot_imbalance_sweep,
    plot_method_comparison,
    plot_nc_scatter,
)

__all__ = [
    "evaluate_checkpoint", "extract_features",
    "compute_all_nc_metrics", "NCMetrics",
    "compute_medical_metrics", "print_medical_metrics",
    "plot_imbalance_sweep", "plot_method_comparison", "plot_nc_scatter",
]
