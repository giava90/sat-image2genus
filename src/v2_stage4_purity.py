"""Run 2, stage 4: test accuracy of run 1's saved models by label purity.

Purity = json_share_of_label, the area share of the file-name genus in the
dataset's multi-label JSON (index.parquet). Bins: < 0.6, 0.6-0.9, >= 0.9.
Also by whether the JSON's largest-share genus agrees with the label.
Writes results_v2/stage4_purity.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RESULTS
from predio import load_predictions
from v2common import RESULTS_V2

MODELS = ["m1_lgbm", "m2_cnn", "m3_resnet18"]
BINS = [0, 0.6, 0.9, np.inf]  # 19 of 27,656 patches have a JSON share > 1 (archive quirk); they go to the top bin
LABELS = ["< 0.6", "0.6 to < 0.9", ">= 0.9"]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> None:
    idx = pd.read_parquet(RESULTS / "index.parquet")
    idx["purity"] = pd.cut(idx.json_share_of_label, BINS, right=False, labels=LABELS)
    idx["json_top_agrees"] = np.where(idx.json_top_genus == idx.genus, "agrees", "disagrees")
    rows = []
    for es in ("common", "full"):
        for m in MODELS:
            if es == "full" and m == "m3_resnet18":
                continue
            p = load_predictions(m)
            p = p[p.split == "test"].merge(idx[["file", "purity", "json_top_agrees", "aerial_available"]], on="file")
            if es == "common":
                p = p[p.aerial_available]
            p["correct"] = p.y_pred == p.y_true
            for col in ("purity", "json_top_agrees"):
                for lev, g in p.groupby(col, observed=False):
                    lo, hi = wilson(int(g.correct.sum()), len(g))
                    rows.append({"eval_set": es, "model": m, "grouping": col, "level": str(lev), "n": len(g),
                                 "accuracy": g.correct.mean() if len(g) else np.nan, "wilson_lo": lo, "wilson_hi": hi})
            rows.append({"eval_set": es, "model": m, "grouping": "all", "level": "all", "n": len(p),
                         "accuracy": p.correct.mean(), "wilson_lo": np.nan, "wilson_hi": np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_V2 / "stage4_purity.csv", index=False, float_format="%.8f")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
