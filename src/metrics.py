"""Classification metrics as small pure functions on integer label arrays."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Counts with rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred differ in shape")
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def row_normalize(cm: np.ndarray) -> np.ndarray:
    """P(pred = j | true = i). Rows with no samples become all zeros."""
    cm = np.asarray(cm, dtype=float)
    s = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(s > 0, cm / np.where(s > 0, s, 1), 0.0)
    return out


def accuracy(cm: np.ndarray) -> float:
    cm = np.asarray(cm, dtype=float)
    return float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan")


def precision_recall(cm: np.ndarray):
    """Per-class precision and recall. Undefined values are NaN."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    pred_tot = cm.sum(axis=0)
    true_tot = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(pred_tot > 0, tp / pred_tot, np.nan)
        recall = np.where(true_tot > 0, tp / true_tot, np.nan)
    return precision, recall


def f1(cm: np.ndarray) -> np.ndarray:
    p, r = precision_recall(cm)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)


def weighted_precision(cm: np.ndarray) -> float:
    """Precision averaged over classes, weighted by true class support (sklearn 'weighted')."""
    cm = np.asarray(cm, dtype=float)
    p, _ = precision_recall(cm)
    w = cm.sum(axis=1)
    p = np.nan_to_num(p, nan=0.0)
    return float((p * w).sum() / w.sum())


def macro_f1(cm: np.ndarray) -> float:
    return float(np.mean(f1(cm)))


def collapse(labels, mapping) -> np.ndarray:
    """Map class ids through a lookup list, e.g. genus -> conifer/broadleaf."""
    return np.asarray(mapping)[np.asarray(labels, dtype=int)]


def summary(y_true, y_pred, n_classes: int) -> dict:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    p, r = precision_recall(cm)
    return {
        "n": int(cm.sum()),
        "accuracy": accuracy(cm),
        "weighted_precision": weighted_precision(cm),
        "macro_f1": macro_f1(cm),
        "precision": p.tolist(),
        "recall": r.tolist(),
        "confusion": cm.tolist(),
    }
