"""Run 2 feature variants for the LightGBM fairness check (stage 2).

The s2/60m patch equals pixels [7:13, 7:13] of the s2/200m patch (verified on
1,000 random patches, results_v2/s2_60m_vs_200m.json), so the 60 m patch is
taken from the 200 m cache rather than read again.
"""
from __future__ import annotations

import numpy as np

from features import BANDS, _stats, indices, summary_features

C0, C1 = 7, 13  # centre 6 x 6 window of the 20 x 20 patch


def centre_crop(x: np.ndarray) -> np.ndarray:
    """(N, C, 20, 20) -> (N, C, 6, 6), the 60 m label window."""
    return x[:, :, C0:C1, C0:C1]


def ring_mask(h: int = 20, w: int = 20) -> np.ndarray:
    """Boolean (h, w) mask, True outside the centre 6 x 6."""
    m = np.ones((h, w), bool)
    m[C0:C1, C0:C1] = False
    return m


def mask_centre(x: np.ndarray, value: float = 0.0) -> np.ndarray:
    """Copy of x with the centre 6 x 6 of every band set to value."""
    y = x.copy()
    y[:, :, C0:C1, C0:C1] = value
    return y


def _ring_pixels(x: np.ndarray) -> np.ndarray:
    """(N, C, 20, 20) -> (N, C, 1, 364): ring pixels as a 1-pixel-high image, so features._stats applies."""
    return x[:, :, ring_mask(*x.shape[2:])][:, :, None, :]


def centre_ring_features(x: np.ndarray):
    """Run 1's summary statistics computed separately on the centre 6 x 6 and on the ring around it."""
    fc, nc = summary_features(centre_crop(x))
    fr, nr = summary_features(_ring_pixels(x))
    return np.concatenate([fc, fr], 1), [f"centre_{n}" for n in nc] + [f"ring_{n}" for n in nr]


def raw_pixel_features(x: np.ndarray):
    """Every band value of every pixel as one feature."""
    n, c, h, w = x.shape
    names = [f"{BANDS[b]}_r{r}_c{q}" for b in range(c) for r in range(h) for q in range(w)]
    return x.reshape(n, -1).astype(np.float32), names


__all__ = ["centre_crop", "ring_mask", "mask_centre", "centre_ring_features", "raw_pixel_features", "indices", "_stats"]
