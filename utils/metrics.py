"""
CyberFin Nexus — Metrics Utilities
AUC, precision, recall, F1, confusion matrix helpers.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, average_precision_score,
    accuracy_score
)


def compute_all_metrics(y_true, y_scores, threshold=0.5):
    """Compute comprehensive classification metrics."""
    y_true = np.asarray(y_true).flatten()
    y_scores = np.asarray(y_scores).flatten()

    # Ensure we have both classes
    if len(np.unique(y_true)) < 2:
        return {
            "auc": 0.0, "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "avg_precision": 0.0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0
        }

    y_pred = (y_scores >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_scores)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    avg_prec = average_precision_score(y_true, y_scores)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "avg_precision": float(avg_prec),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_pr_curve(y_true, y_scores):
    """Return precision-recall curve data."""
    y_true = np.asarray(y_true).flatten()
    y_scores = np.asarray(y_scores).flatten()
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return precision, recall, thresholds


def risk_category(score, thresholds=None):
    """Classify a risk score into a category string."""
    if thresholds is None:
        thresholds = {"low": 0.3, "medium": 0.6, "high": 0.85, "critical": 0.95}

    if score >= thresholds["critical"]:
        return "CRITICAL"
    elif score >= thresholds["high"]:
        return "HIGH"
    elif score >= thresholds["medium"]:
        return "MEDIUM"
    else:
        return "LOW"


def compute_roi(num_rings_detected, avg_loss_per_ring=500_000):
    """Estimate dollar savings from detected mule rings."""
    return num_rings_detected * avg_loss_per_ring
