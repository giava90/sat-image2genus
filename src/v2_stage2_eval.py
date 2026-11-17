"""Run 2, stage 2 evaluation: variants vs run 1 models 1 and 2 on the same test patches.

The 'best fair baseline' among the LightGBM variants (a-d) is chosen by
validation accuracy, not test accuracy. Writes results_v2/stage2_accuracy.csv,
stage2_paired.csv, stage2_best_variant.json.
"""
from __future__ import annotations

import json

import pandas as pd

from common import RESULTS
from v2_eval import load, score
from v2common import RESULTS_V2

LGBM_VARIANTS = {"lgbm_summary60": "a. LightGBM, summary stats of 60 m patch",
                 "lgbm_centre_ring": "b. LightGBM, centre 6x6 + ring summary stats",
                 "lgbm_raw60": "c. LightGBM, raw pixels of 60 m patch",
                 "lgbm_raw200": "d. LightGBM, raw pixels of 200 m patch"}
CNN_VARIANTS = {"cnn_centre6": "e. CNN, centre 6x6 only", "cnn_masked_centre": "f. CNN, 200 m with centre 6x6 masked"}
OTHER = {"lgbm_summary200_repro": "LightGBM run 1 features, retrained (reproduction check)"}


def val_accuracy(ref: str) -> float:
    p = load(ref)
    p = p[p.split == "val"]
    return float((p.y_pred == p.y_true).mean())


def main() -> None:
    idx = pd.read_parquet(RESULTS / "index.parquet")
    avail = {k for k in {**LGBM_VARIANTS, **CNN_VARIANTS, **OTHER} if (RESULTS_V2 / "stage2" / f"predictions_{k}.parquet").exists()}
    val_acc = {k: val_accuracy(f"stage2:{k}") for k in avail}
    lg = [k for k in LGBM_VARIANTS if k in avail]
    best = max(lg, key=lambda k: val_acc[k])
    (RESULTS_V2 / "stage2_best_variant.json").write_text(json.dumps(
        {"best_lgbm_variant_by_val_accuracy": best, "val_accuracy": val_acc,
         "run1_val_accuracy": {m: val_accuracy(f"run1:{m}") for m in ("m1_lgbm", "m2_cnn", "m3_resnet18")}}, indent=2))
    acc_all, diff_all = [], []
    for es in ("common", "full"):
        files = set(idx.loc[idx.aerial_available, "file"]) if es == "common" else None
        refs = {"M1": "run1:m1_lgbm", "M2": "run1:m2_cnn"}
        if es == "common":
            refs["M3"] = "run1:m3_resnet18"
        refs.update({k: f"stage2:{k}" for k in sorted(avail)})
        pairs = [("M1", "M2")] + [("M1", k) for k in sorted(avail)] + [("M2", k) for k in sorted(avail)] + [(best, "M2")]
        if es == "common":
            pairs.append((best, "M3"))
        acc, diff = score(refs, files, pairs)
        acc_all.append(acc.assign(eval_set=es))
        diff_all.append(diff.assign(eval_set=es))
    acc = pd.concat(acc_all)
    acc["description"] = acc.model.map({**LGBM_VARIANTS, **CNN_VARIANTS, **OTHER}).fillna("run 1 model")
    acc["val_accuracy"] = acc.model.map(val_acc)
    acc.to_csv(RESULTS_V2 / "stage2_accuracy.csv", index=False, float_format="%.8f")
    diff = pd.concat(diff_all)
    diff.to_csv(RESULTS_V2 / "stage2_paired.csv", index=False, float_format="%.8f")
    print("best LightGBM variant by validation accuracy:", best)
    print(acc.round(4).to_string(index=False))
    print(diff.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
