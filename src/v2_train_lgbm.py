"""Run 2: LightGBM with run 1's settings on alternative features and/or splits.

--features: summary200 (run 1 model 1), summary60, centre_ring, raw60, raw200, xy
--split:    orig (run 1 split) or a blocked split name, e.g. seed0
Writes results_v2/<sub>/predictions_<name>.parquet for val and test patches.
"""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import CACHE, GENERA, RESULTS
from features import summary_features
from metrics import summary
from train_m1_lgbm import PARAMS
from v2_features import centre_crop, centre_ring_features, raw_pixel_features
from v2_splits import get_split
from v2common import save_predictions_v2


def make_features(kind: str, X: np.ndarray, df: pd.DataFrame):
    if kind == "summary200":
        return summary_features(X)
    if kind == "summary60":
        return summary_features(centre_crop(X))
    if kind == "centre_ring":
        return centre_ring_features(X)
    if kind == "raw60":
        return raw_pixel_features(centre_crop(X))
    if kind == "raw200":
        return raw_pixel_features(X)
    if kind == "xy":
        c = pd.read_parquet(RESULTS / "patch_centres.parquet").set_index("file").loc[df.file]
        return c[["x", "y"]].to_numpy(np.float32), ["x", "y"]
    raise ValueError(kind)


def main(kind: str, split: str, name: str, sub: str, max_rounds: int) -> None:
    t0 = time.time()
    df = get_split(split)
    X = np.load(CACHE / "s2_200m.npy")  # run 1 cache, in index.parquet order
    assert len(X) == len(df)
    F, names = make_features(kind, X, df)
    t_feat = time.time() - t0
    tr = (df.split == "train").to_numpy()
    va = (df.split == "val").to_numpy()
    ev = df.split.isin(["val", "test"]).to_numpy()
    y = df.y.to_numpy()
    dtr = lgb.Dataset(F[tr], y[tr], feature_name=names, free_raw_data=False)
    dva = lgb.Dataset(F[va], y[va], reference=dtr)
    t1 = time.time()
    booster = lgb.train(PARAMS, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    t_train = time.time() - t1
    probs = booster.predict(F[ev], num_iteration=booster.best_iteration)
    s = df[ev]
    info = {"model": name, "features": kind, "split": split, "n_features": len(names), "n_train": int(tr.sum()),
            "best_iteration": int(booster.best_iteration), "feature_seconds": round(t_feat, 1),
            "train_seconds": round(t_train, 1), "device": "cpu", "params": PARAMS}
    save_predictions_v2(name, s.file, s.split, s.y, probs, info, sub=sub)
    te = (s.split == "test").to_numpy()
    r = summary(s.y[te], probs[te].argmax(1), len(GENERA))
    print(name, {k: r[k] for k in ("n", "accuracy")}, info["best_iteration"], f"{t_train:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", default="orig")
    ap.add_argument("--name", required=True)
    ap.add_argument("--sub", default="stage2")
    ap.add_argument("--max-rounds", type=int, default=3000)
    a = ap.parse_args()
    main(a.features, a.split, a.name, a.sub, a.max_rounds)
