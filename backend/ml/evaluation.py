"""Evaluation metrics (paper §IV-A): accuracy, precision, recall, F1, AUC, plus
confusion matrix, ROC and precision–recall curves."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _downsample_curve(x: np.ndarray, y: np.ndarray, max_points: int = 200) -> tuple[list[float], list[float]]:
    if len(x) <= max_points:
        return [float(v) for v in x], [float(v) for v in y]
    idx = np.unique(np.linspace(0, len(x) - 1, max_points).astype(int))
    return [float(v) for v in x[idx]], [float(v) for v in y[idx]]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int), labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "labels": ["HUMAN", "BOT"],
        "matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }


def roc_curve_data(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"fpr": [], "tpr": [], "thresholds": [], "auc": None}
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    fpr_s, tpr_s = _downsample_curve(fpr, tpr)
    return {"fpr": fpr_s, "tpr": tpr_s, "auc": float(auc(fpr, tpr))}


def pr_curve_data(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"precision": [], "recall": [], "average_precision": None}
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns arrays in decreasing recall order
    order = np.argsort(recall)
    rec_s, prec_s = _downsample_curve(recall[order], precision[order])
    ap = float(auc(recall[order], precision[order]))
    return {"precision": prec_s, "recall": rec_s, "average_precision": ap}


def probability_histogram(y_proba: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    counts, edges = np.histogram(np.clip(np.asarray(y_proba, dtype=float), 0, 1), bins=bins, range=(0.0, 1.0))
    return [
        {"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]), "count": int(counts[i])}
        for i in range(len(counts))
    ]


def full_evaluation(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    return {
        "metrics": classification_metrics(y_true, y_pred, y_proba),
        "confusion_matrix": confusion(y_true, y_pred),
        "roc_curve": roc_curve_data(y_true, y_proba),
        "pr_curve": pr_curve_data(y_true, y_proba),
        "probability_histogram": probability_histogram(y_proba),
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(np.asarray(y_true).astype(int) == 1)),
        "n_negative": int(np.sum(np.asarray(y_true).astype(int) == 0)),
    }
