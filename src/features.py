"""Hand-made per-band summary features of Sentinel-2 patches (model 1)."""
from __future__ import annotations

import numpy as np

# Band order in the TreeSatAI S2 files (see DECISIONS.md D5).
BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
PERCENTILES = (10, 25, 50, 75, 90)


def _b(name: str) -> int:
    return BANDS.index(name)


def norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    den = a + b
    return np.where(den != 0, (a - b) / np.where(den != 0, den, 1), 0.0).astype(np.float32)


def indices(x: np.ndarray) -> dict[str, np.ndarray]:
    """x: (N, 12, H, W) reflectance. Returns per-pixel index maps (N, H, W)."""
    return {
        "NDVI": norm_diff(x[:, _b("B08")], x[:, _b("B04")]),
        "NDRE": norm_diff(x[:, _b("B8A")], x[:, _b("B05")]),
        "NDMI": norm_diff(x[:, _b("B08")], x[:, _b("B11")]),
    }


def _stats(m: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    """m: (N, H, W) -> mean, std and percentiles per sample."""
    flat = m.reshape(len(m), -1).astype(np.float32)
    out = {f"{prefix}_mean": flat.mean(1), f"{prefix}_std": flat.std(1)}
    pct = np.percentile(flat, PERCENTILES, axis=1)
    for p, v in zip(PERCENTILES, pct):
        out[f"{prefix}_p{p}"] = v
    return out


def summary_features(x: np.ndarray):
    """x: (N, 12, H, W). Returns (feature matrix (N, F) float32, feature names)."""
    feats: dict[str, np.ndarray] = {}
    for i, name in enumerate(BANDS):
        feats.update(_stats(x[:, i], name))
    for name, m in indices(x).items():
        feats.update(_stats(m, name))
    names = list(feats)
    return np.stack([feats[n] for n in names], axis=1).astype(np.float32), names
