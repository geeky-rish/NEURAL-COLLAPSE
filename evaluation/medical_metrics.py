"""
evaluation/medical_metrics.py
─────────────────────────────────────────────────────────────────────────────
Healthcare-oriented classification evaluation metrics.

Standard accuracy is insufficient for medical AI because:
  • A model predicting "Healthy" for every patient achieves 90%+ accuracy
    on datasets where only 10% are diseased — but has zero clinical value.
  • What matters is: can the model detect the rare, dangerous condition?

Provided metrics
----------------
sensitivity (recall)      — true positive rate per class
specificity               — true negative rate per class
precision (PPV)           — positive predictive value
F1 score                  — harmonic mean of precision and recall
ROC-AUC                   — area under receiver operating characteristic
Cohen's Kappa             — agreement beyond chance
Per-class report          — full breakdown for each class
Confusion matrix          — raw + normalised
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)


def compute_medical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, object]:
    """
    Compute a full suite of medical classification metrics.

    Parameters
    ----------
    y_true      : (N,)    — ground-truth integer labels
    y_pred      : (N,)    — predicted integer labels
    y_prob      : (N, C)  — softmax probabilities (optional, for AUC)
    class_names : list[str] — for readable output
    num_classes : int     — inferred from y_true if None

    Returns
    -------
    dict with keys:
        accuracy, macro_f1, weighted_f1, kappa,
        sensitivity (per-class), specificity (per-class),
        precision (per-class), recall (per-class), f1 (per-class),
        roc_auc (macro OvR), confusion_matrix, normalised_confusion_matrix,
        classification_report (str)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    C      = num_classes or int(y_true.max()) + 1

    if class_names is None:
        class_names = [f"Class {c}" for c in range(C)]

    results: Dict[str, object] = {}

    # ── Accuracy ──────────────────────────────────────────────────────────────
    results["accuracy"] = float((y_true == y_pred).mean() * 100.0)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(C)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    results["confusion_matrix"]            = cm.tolist()
    results["normalised_confusion_matrix"] = cm_norm.tolist()

    # ── Per-class sensitivity, specificity, precision ─────────────────────────
    sensitivity  = []
    specificity  = []
    precision_pc = []
    recall_pc    = []
    f1_pc        = []

    for c in range(C):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = sens
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        sensitivity.append(float(sens))
        specificity.append(float(spec))
        precision_pc.append(float(prec))
        recall_pc.append(float(rec))
        f1_pc.append(float(f1))

    results["sensitivity"]  = dict(zip(class_names, sensitivity))
    results["specificity"]  = dict(zip(class_names, specificity))
    results["precision"]    = dict(zip(class_names, precision_pc))
    results["recall"]       = dict(zip(class_names, recall_pc))
    results["f1_per_class"] = dict(zip(class_names, f1_pc))

    # ── Macro / Weighted averages ─────────────────────────────────────────────
    results["macro_f1"]    = float(np.mean(f1_pc))
    results["weighted_f1"] = float(np.average(f1_pc,
                                    weights=np.bincount(y_true, minlength=C)))

    results["mean_sensitivity"] = float(np.mean(sensitivity))
    results["mean_specificity"] = float(np.mean(specificity))

    # ── Cohen's Kappa ─────────────────────────────────────────────────────────
    results["kappa"] = float(cohen_kappa_score(y_true, y_pred))

    # ── ROC-AUC ───────────────────────────────────────────────────────────────
    if y_prob is not None:
        try:
            multi = "ovr" if C > 2 else None
            if C == 2:
                results["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob[:, 1])
                )
            else:
                results["roc_auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
        except Exception:
            results["roc_auc"] = float("nan")
    else:
        results["roc_auc"] = float("nan")

    # ── Full sklearn classification report ────────────────────────────────────
    results["classification_report"] = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    return results


def print_medical_metrics(metrics: Dict[str, object]) -> None:
    """Pretty-print a medical metrics dict to stdout."""
    print(f"\n{'─'*60}")
    print(f"  Accuracy       : {metrics['accuracy']:.2f}%")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1    : {metrics['weighted_f1']:.4f}")
    print(f"  Cohen Kappa    : {metrics['kappa']:.4f}")
    print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"  Mean Sensitivity: {metrics['mean_sensitivity']:.4f}")
    print(f"  Mean Specificity: {metrics['mean_specificity']:.4f}")
    print(f"\n{metrics['classification_report']}")
    print(f"{'─'*60}\n")
