"""Run 2, stage 3: coordinates-only k-nearest-neighbour classifier (patch centres, Euclidean distance)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def knn_proba(xy_train, y_train, xy_query, k: int, n_classes: int) -> np.ndarray:
    """Share of each class among the k nearest training patches. Ties in argmax go to the lower class id."""
    xy_train = np.asarray(xy_train, float)
    y_train = np.asarray(y_train, int)
    _, j = cKDTree(xy_train).query(np.asarray(xy_query, float), k=k)
    j = j.reshape(len(xy_query), k)
    out = np.zeros((len(j), n_classes))
    for c in range(k):
        out[np.arange(len(j)), y_train[j[:, c]]] += 1
    return out / k
