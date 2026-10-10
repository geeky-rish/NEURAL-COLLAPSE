"""
training/__init__.py  (updated)
"""
from .losses import get_criterion
from .trainer import Trainer
from .scheduler import build_scheduler
from .nc_regularization import (
    NCCollapseRegularizer,
    ETFAlignmentLoss,
    SupConLoss,
    CombinedNCLoss,
)

__all__ = [
    "get_criterion", "Trainer", "build_scheduler",
    "NCCollapseRegularizer", "ETFAlignmentLoss",
    "SupConLoss", "CombinedNCLoss",
]
