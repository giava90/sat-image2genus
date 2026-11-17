"""From classifier behaviour to area and wood-volume estimates for a synthetic region.

Model of the region
-------------------
A region of stands with known true area per genus, a_i (hectares). Every stand
is classified once. A stand of true genus i is labelled j with probability
P(pred=j | true=i), taken from the row-normalised test confusion matrix. The
region is treated as large, so the expected predicted area is used and the
only source of uncertainty is the finite test set (handled by bootstrapping).

Two area estimators
-------------------
* count:    a_hat = C_test^T a                (area of each predicted label)
* adjusted: solve C_val^T x = a_hat for x     (confusion-matrix / inverse
            calibration with the matrix estimated on the validation set),
            then clip negatives to 0 and rescale so total area is unchanged.

Volume = area x growing stock per genus (m3/ha).
"""
from __future__ import annotations

import numpy as np

from metrics import confusion_matrix, row_normalize


def true_areas(shares, total_area: float) -> np.ndarray:
    shares = np.asarray(shares, dtype=float)
    if np.any(shares < 0) or not np.isclose(shares.sum(), 1.0):
        raise ValueError("shares must be non-negative and sum to 1")
    return shares * total_area


def count_estimate(areas_true, cond_test) -> np.ndarray:
    """Expected area labelled as each genus: a_hat_j = sum_i a_i P(j|i)."""
    return np.asarray(areas_true, float) @ np.asarray(cond_test, float)


def adjust(areas_counted, cond_val) -> np.ndarray:
    """Invert the validation confusion matrix. Falls back to least squares if singular."""
    areas_counted = np.asarray(areas_counted, float)
    A = np.asarray(cond_val, float).T
    try:
        x = np.linalg.solve(A, areas_counted)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A, areas_counted, rcond=None)[0]
    x = np.clip(x, 0.0, None)
    total = areas_counted.sum()
    if x.sum() > 0:
        x = x * (total / x.sum())
    return x


def volumes(areas, growing_stock) -> np.ndarray:
    return np.asarray(areas, float) * np.asarray(growing_stock, float)


def relative_error(est, true) -> np.ndarray:
    est = np.asarray(est, float)
    true = np.asarray(true, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(true != 0, (est - true) / true, np.nan)


def volume_errors(areas_true, cond_test, cond_val, growing_stock) -> dict:
    """Relative volume error per genus and in total, for both estimators."""
    gs = np.asarray(growing_stock, float)
    v_true = volumes(areas_true, gs)
    a_cnt = count_estimate(areas_true, cond_test)
    a_adj = adjust(a_cnt, cond_val)
    out = {}
    for name, a in (("count", a_cnt), ("adjusted", a_adj)):
        v = volumes(a, gs)
        out[name] = {
            "per_genus": relative_error(v, v_true),
            "total": float((v.sum() - v_true.sum()) / v_true.sum()),
        }
    return out


def bootstrap_volume_errors(
    y_true_test,
    y_pred_test,
    cond_val,
    shares,
    growing_stock,
    n_classes: int,
    n_boot: int = 1000,
    total_area: float = 10_000.0,
    seed: int = 0,
) -> dict:
    """Resample the test set with replacement n_boot times.

    Returns arrays of relative errors: {'count'|'adjusted': {'per_genus': (n_boot, k), 'total': (n_boot,)}}
    plus the point estimate on the full test set under key 'point'.
    """
    y_true_test = np.asarray(y_true_test, int)
    y_pred_test = np.asarray(y_pred_test, int)
    a_true = true_areas(shares, total_area)
    rng = np.random.default_rng(seed)
    n = len(y_true_test)
    res = {k: {"per_genus": np.empty((n_boot, n_classes)), "total": np.empty(n_boot)} for k in ("count", "adjusted")}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        cond = row_normalize(confusion_matrix(y_true_test[idx], y_pred_test[idx], n_classes))
        e = volume_errors(a_true, cond, cond_val, growing_stock)
        for k in ("count", "adjusted"):
            res[k]["per_genus"][b] = e[k]["per_genus"]
            res[k]["total"][b] = e[k]["total"]
    cond_full = row_normalize(confusion_matrix(y_true_test, y_pred_test, n_classes))
    res["point"] = volume_errors(a_true, cond_full, cond_val, growing_stock)
    return res
