"""Shared constants and paths."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
RESULTS = ROOT / "results"

# Target classes, fixed order used everywhere (index = class id).
GENERA = ["Picea", "Pinus", "Fagus", "Quercus"]
COMMON_NAMES = {"Picea": "spruce", "Pinus": "pine", "Fagus": "beech", "Quercus": "oak"}
CONIFERS = {"Picea", "Pinus"}
# Binary class ids: 0 = conifer, 1 = broadleaf.
GENUS_TO_BINARY = [0 if g in CONIFERS else 1 for g in GENERA]
BINARY_NAMES = ["conifer", "broadleaf"]

SEED = 0


def device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
