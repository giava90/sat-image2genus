"""Model 1: LightGBM on per-band summary statistics of 200 m Sentinel-2 patches.

Trains on the train split, early-stops on the validation split, writes
predictions for val and test to results/predictions_m1_lgbm.parquet.
"""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import GENERA, RESULTS, SEED
from data import build_s2_cache
from features import summary_features
from metrics import summary
from predio import save_predictions

PARAMS = dict(
    objective="multiclass",
    num_class=len(GENERA),
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    seed=SEED,
    num_threads=8,
    verbose=-1,
)


def main(max_rounds: int, pilot: bool) -> None:
    t0 = time.time()
    df = pd.read_parquet(RESULTS / "index.parquet")
    X = build_s2_cache(df)
    F, names = summary_features(X)
    t_feat = time.time() - t0
    tr = (df.split == "train").to_numpy()
    va = (df.split == "val").to_numpy()
    ev = df.split.isin(["val", "test"]).to_numpy()
    if pilot:
        rng = np.random.default_rng(SEED)
        keep = np.zeros(len(df), bool)
        keep[rng.choice(np.flatnonzero(tr), 3000, replace=False)] = True
        tr = keep
    dtr = lgb.Dataset(F[tr], df.y[tr].to_numpy(), feature_name=names)
    dva = lgb.Dataset(F[va], df.y[va].to_numpy(), reference=dtr)
    t1 = time.time()
    booster = lgb.train(
        PARAMS,
        dtr,
        num_boost_round=max_rounds,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)],
    )
    t_train = time.time() - t1
    probs = booster.predict(F[ev], num_iteration=booster.best_iteration)
    sub = df[ev]
    info = {
        "model": "m1_lgbm",
        "n_features": len(names),
        "n_train": int(tr.sum()),
        "best_iteration": int(booster.best_iteration),
        "feature_seconds": round(t_feat, 1),
        "train_seconds": round(t_train, 1),
        "device": "cpu",
        "params": PARAMS,
    }
    if pilot:
        print(info)
        return
    save_predictions("m1_lgbm", sub.file, sub.split, sub.y, probs, info)
    te = (sub.split == "test").to_numpy()
    s = summary(sub.y[te], probs[te].argmax(1), len(GENERA))
    print({k: s[k] for k in ("n", "accuracy", "weighted_precision", "macro_f1")})
    imp = pd.Series(booster.feature_importance("gain"), index=names).sort_values(ascending=False)
    imp.to_csv(RESULTS / "m1_lgbm_feature_importance.csv")
    print(info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    main(a.max_rounds, a.pilot)
