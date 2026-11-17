"""Run 2, stage 1: class-share (quantification) estimators and synthetic-region sampling.

All estimators return estimated class shares (non-negative, summing to 1).
  cc   classify and count:             share of hard predictions per class
  ac   adjusted count:                 solve C_val^T x = cc, C_val[i, j] = P(pred j | true i) on validation
  pcc  probabilistic classify and count: mean predicted probability per class
  pac  probabilistic adjusted count:   solve M_val^T x = pcc, M_val[i, j] = mean p_j over validation patches of true class i
  em   EM prior-shift correction (Saerens, Latinne and Decaestecker 2002), started from the training shares
For ac and pac, negative solutions are clipped to 0 and the result renormalised (as in run 1).
"""
from __future__ import annotations

import numpy as np


def _clip_norm(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, float), 0.0, None)
    s = x.sum()
    return x / s if s > 0 else np.full(len(x), 1.0 / len(x))


def _solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def cc(y_pred, k: int) -> np.ndarray:
    return np.bincount(np.asarray(y_pred, int), minlength=k)[:k] / len(y_pred)


def pcc(probs) -> np.ndarray:
    return np.asarray(probs, float).mean(0)


def cond_matrix(y_true, y_pred, k: int) -> np.ndarray:
    """Row-normalised confusion matrix P(pred j | true i); empty rows become identity rows."""
    m = np.zeros((k, k))
    np.add.at(m, (np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    s = m.sum(1, keepdims=True)
    return np.where(s > 0, m / np.where(s > 0, s, 1), np.eye(k))


def soft_matrix(y_true, probs, k: int) -> np.ndarray:
    """M[i, j] = mean predicted probability of class j over patches of true class i."""
    probs = np.asarray(probs, float)
    y_true = np.asarray(y_true, int)
    m = np.eye(k)
    for i in range(k):
        if (y_true == i).any():
            m[i] = probs[y_true == i].mean(0)
    return m


def ac(cc_shares, cond_val) -> np.ndarray:
    return _clip_norm(_solve(np.asarray(cond_val, float).T, np.asarray(cc_shares, float)))


def pac(pcc_shares, soft_val) -> np.ndarray:
    return _clip_norm(_solve(np.asarray(soft_val, float).T, np.asarray(pcc_shares, float)))


def em(probs, train_prior, tol: float = 1e-8, max_iter: int = 1000) -> tuple[np.ndarray, int]:
    """Saerens et al. (2002): re-weight posteriors by new/old prior, re-estimate the prior, repeat."""
    p = np.asarray(probs, float)
    pt = np.asarray(train_prior, float)
    pt = pt / pt.sum()
    pi = pt.copy()
    for it in range(1, max_iter + 1):
        w = p * (pi / pt)
        w /= w.sum(1, keepdims=True)
        new = w.mean(0)
        if np.abs(new - pi).max() < tol:
            return new, it
        pi = new
    return pi, max_iter


def stand_counts(shares, n: int) -> np.ndarray:
    """Integer stands per class summing to n, by largest remainder."""
    shares = np.asarray(shares, float)
    if np.any(shares < 0) or not np.isclose(shares.sum(), 1.0):
        raise ValueError("shares must be non-negative and sum to 1")
    raw = shares * n
    c = np.floor(raw).astype(int)
    c[np.argsort(-(raw - c), kind="stable")[: n - c.sum()]] += 1
    return c


def sample_region(y_true, shares, n_stands: int, rng) -> np.ndarray:
    """Indices of a synthetic region: for each class, draw its stand count with replacement from that class's patches."""
    y_true = np.asarray(y_true, int)
    counts = stand_counts(shares, n_stands)
    out = []
    for i, c in enumerate(counts):
        if c == 0:
            continue
        pool = np.flatnonzero(y_true == i)
        if len(pool) == 0:
            raise ValueError(f"class {i} has share > 0 but no patches")
        out.append(rng.choice(pool, c, replace=True))
    return np.concatenate(out)


ESTIMATORS = ["cc", "ac", "pcc", "pac", "em"]


def all_estimates(y_pred, probs, k: int, cond_val, soft_val, train_prior) -> dict[str, np.ndarray]:
    c = cc(y_pred, k)
    p = pcc(probs)
    return {"cc": c, "ac": ac(c, cond_val), "pcc": p, "pac": pac(p, soft_val), "em": em(probs, train_prior)[0]}
