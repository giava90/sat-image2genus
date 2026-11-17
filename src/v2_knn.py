"""Run 2, stage 3: coordinates-only 1-NN and 5-NN baselines on a given split.

Usage: python src/v2_knn.py --split orig|seed0|seed1
Writes results_v2/stage3_<split>/predictions_knn1.parquet and predictions_knn5.parquet.
"""
from __future__ import annotations

import argparse

import pandas as pd

from common import GENERA, RESULTS
from coord_baselines import knn_proba
from metrics import accuracy, confusion_matrix
from v2_splits import get_split
from v2common import save_predictions_v2


def main(split: str) -> None:
    df = get_split(split)
    c = pd.read_parquet(RESULTS / "patch_centres.parquet").set_index("file").loc[df.file]
    xy = c[["x", "y"]].to_numpy()
    tr = (df.split == "train").to_numpy()
    ev = df.split.isin(["val", "test"]).to_numpy()
    s = df[ev]
    for k in (1, 5):
        p = knn_proba(xy[tr], df.y.to_numpy()[tr], xy[ev], k, len(GENERA))
        save_predictions_v2(f"knn{k}", s.file, s.split, s.y, p, {"model": f"knn{k}", "split": split, "k": k,
                                                               "n_train": int(tr.sum())}, sub=f"stage3_{split}")
        te = (s.split == "test").to_numpy()
        print(split, f"knn{k}", round(accuracy(confusion_matrix(s.y[te], p[te].argmax(1), len(GENERA))), 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="orig")
    main(ap.parse_args().split)
