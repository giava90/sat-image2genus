"""Run 2, stage 3 evaluation: spatial independence.

A. Original split: run 1 models, coordinate-only baselines and the best stage 2
   LightGBM variant, with patch- and block-bootstrap intervals (2 km blocks).
B. Blocked splits (results_v2/blocked_split_seed*.parquet): every model trained
   on them, scored on the blocked test set (all four-genus patches) and on its
   common subset (species with aerial data), block-bootstrap intervals.
Writes results_v2/stage3_accuracy.csv and stage3_paired.csv.
"""
from __future__ import annotations

import json

import pandas as pd

from common import RESULTS
from v2_eval import score
from v2common import RESULTS_V2

LABELS = {"M1": "LightGBM, run 1 features (200 m summary stats)", "M2": "small CNN, 200 m patch", "M3": "ResNet-18, aerial",
          "best_lgbm": "best stage 2 LightGBM variant", "knn1": "1-NN on patch centre", "knn5": "5-NN on patch centre",
          "lgbm_xy": "LightGBM on (x, y)"}


def main() -> None:
    idx = pd.read_parquet(RESULTS / "index.parquet")
    common_files = set(idx.loc[idx.aerial_available, "file"])
    best = json.loads((RESULTS_V2 / "stage2_best_variant.json").read_text())["best_lgbm_variant_by_val_accuracy"]
    acc_all, diff_all = [], []
    # A. original split
    for es in ("common", "full"):
        refs = {"M1": "run1:m1_lgbm", "M2": "run1:m2_cnn", "best_lgbm": f"stage2:{best}",
                "knn1": "stage3_orig:knn1", "knn5": "stage3_orig:knn5"}
        if (RESULTS_V2 / "stage3_orig" / "predictions_lgbm_xy.parquet").exists():
            refs["lgbm_xy"] = "stage3_orig:lgbm_xy"
        if es == "common":
            refs["M3"] = "run1:m3_resnet18"
        pairs = [("M1", "M2"), ("M1", "M3"), ("M2", "M3"), ("best_lgbm", "M2"), ("best_lgbm", "M3"), ("knn1", "M1"), ("knn1", "M2")]
        acc, diff = score(refs, common_files if es == "common" else None, pairs)
        acc_all.append(acc.assign(split="original", eval_set=es))
        diff_all.append(diff.assign(split="original", eval_set=es))
    # B. blocked splits
    for seed in ("seed0", "seed1"):
        d = RESULTS_V2 / f"stage3_{seed}"
        if not d.exists():
            continue
        cand = {"M1": "lgbm_summary200", "best_lgbm": best, "M2": "cnn_full", "knn1": "knn1", "knn5": "knn5", "lgbm_xy": "lgbm_xy"}
        refs = {k: f"stage3_{seed}:{v}" for k, v in cand.items() if (d / f"predictions_{v}.parquet").exists()}
        for es in ("common", "full"):
            pairs = [("M1", "M2"), ("best_lgbm", "M2"), ("M1", "best_lgbm"), ("knn1", "M1"), ("knn1", "M2")]
            acc, diff = score(refs, common_files if es == "common" else None, pairs)
            acc_all.append(acc.assign(split=f"blocked {seed}", eval_set=es))
            diff_all.append(diff.assign(split=f"blocked {seed}", eval_set=es))
    acc = pd.concat(acc_all)
    acc["description"] = acc.model.map(LABELS)
    acc.loc[acc.model == "best_lgbm", "description"] = f"best stage 2 LightGBM variant ({best})"
    acc.to_csv(RESULTS_V2 / "stage3_accuracy.csv", index=False, float_format="%.8f")
    diff = pd.concat(diff_all)
    diff.to_csv(RESULTS_V2 / "stage3_paired.csv", index=False, float_format="%.8f")
    print(acc.round(4).to_string(index=False))
    print(diff.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
