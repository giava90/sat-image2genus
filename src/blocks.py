"""Run 2, stage 3: spatial blocks, blocked train/val/test split, block bootstrap."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

BLOCK_M = 2000.0
BUFFER_M = 200.0


def block_ids(x, y, size: float = BLOCK_M) -> np.ndarray:
    """Grid-cell id for each point: cells of size x size metres anchored at (0, 0)."""
    bx = np.floor(np.asarray(x, float) / size).astype(np.int64)
    by = np.floor(np.asarray(y, float) / size).astype(np.int64)
    return bx * 10_000_000 + by


def blocked_split(x, y, test_frac: float = 0.2, val_frac: float = 0.1, seed: int = 0,
                  size: float = BLOCK_M, buffer: float = BUFFER_M) -> np.ndarray:
    """Assign whole blocks to 'test' (first ~test_frac of patches in a random block order),
    then 'val' (next ~val_frac), the rest to 'train'. Training patches whose centre lies
    within `buffer` metres (Chebyshev, i.e. overlapping windows of that size) of any
    held-out patch become 'dropped'. Returns an array of split labels.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    b = block_ids(x, y, size)
    ub, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(ub))
    cum = np.cumsum(cnt[order])
    lab_block = np.full(len(ub), "train", dtype=object)
    lab_block[order[cum - cnt[order] < test_frac * n]] = "test"
    is_val = (cum - cnt[order] >= test_frac * n) & (cum - cnt[order] < (test_frac + val_frac) * n)
    lab_block[order[is_val]] = "val"
    split = lab_block[inv].astype(object)
    held = split != "train"
    if held.any() and (~held).any() and buffer > 0:
        tree = cKDTree(np.column_stack([x[held], y[held]]))
        d, _ = tree.query(np.column_stack([x[~held], y[~held]]), k=1, p=np.inf)
        tr_idx = np.flatnonzero(~held)
        split[tr_idx[d < buffer]] = "dropped"
    return split.astype(str)


def block_bootstrap(groups, n_boot: int, seed: int = 0) -> list[np.ndarray]:
    """Resample whole groups with replacement. Returns n_boot index arrays into the items."""
    groups = np.asarray(groups)
    ug, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(ug))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        out.append(np.concatenate([members[k] for k in pick]))
    return out
