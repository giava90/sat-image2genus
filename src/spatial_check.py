"""Diagnostic: does test accuracy depend on how close a test patch is to a training patch?

The archive's split lists carry no stand IDs, so neighbouring patches can sit on
both sides of the split, and 200 m Sentinel-2 windows of patches less than
200 m apart overlap. This script reads each patch's centre from the GeoTIFF
georeferencing (no extra download), finds the nearest training patch for every
test patch, and scores each model by distance band.

Writes results/patch_centres.parquet and results/spatial_check.csv.
"""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common import GENERA, RAW, RESULTS
from predio import load_predictions

MODELS = ["m1_lgbm", "m2_cnn", "m3_resnet18"]
BANDS_M = [0, 200, 600, 2000, np.inf]  # nearest-training-patch distance bands, metres (Chebyshev)


def patch_centres(index: pd.DataFrame) -> pd.DataFrame:
    out = RESULTS / "patch_centres.parquet"
    if out.exists():
        c = pd.read_parquet(out)
        if len(c) == len(index):
            return c
    from rasterio.io import MemoryFile

    rows = []
    with zipfile.ZipFile(RAW / "s2.zip") as z:
        for fn in index.file:
            with MemoryFile(z.read(f"s2/60m/{fn}")) as mf, mf.open() as ds:
                b = ds.bounds
                rows.append((fn, (b.left + b.right) / 2, (b.bottom + b.top) / 2, ds.crs.to_epsg()))
    c = pd.DataFrame(rows, columns=["file", "x", "y", "epsg"])
    c.to_parquet(out, index=False)
    return c


def main() -> None:
    index = pd.read_parquet(RESULTS / "index.parquet")
    c = patch_centres(index).rename(columns={"x": "cx", "y": "cy"}).set_index("file")
    df = index.join(c, on="file")
    assert df.epsg.nunique() == 1, "patches in several CRSs"
    tr = df[df.split == "train"]
    tree = cKDTree(tr[["cx", "cy"]].to_numpy())
    te = df[df.split == "test"].copy()
    d, j = tree.query(te[["cx", "cy"]].to_numpy(), k=1, p=np.inf)  # Chebyshev: windows overlap if < window size
    te["nn_train_dist_m"] = d
    te["nn_train_same_genus"] = tr.genus.to_numpy()[j] == te.genus.to_numpy()
    te["band"] = pd.cut(te.nn_train_dist_m, BANDS_M, right=False,
                        labels=[f"{int(a)}-{'inf' if np.isinf(b) else int(b)} m" for a, b in zip(BANDS_M[:-1], BANDS_M[1:])])
    rows = []
    for m in MODELS:
        p = RESULTS / f"predictions_{m}.parquet"
        if not p.exists():
            continue
        pr = load_predictions(m)
        pr = pr[pr.split == "test"][["file", "y_pred"]]
        t = te.merge(pr, on="file")
        t["correct"] = t.y_pred == t.y
        for es in ("common", "full"):
            tt = t[t.aerial_available] if es == "common" else t
            if es == "full" and m == "m3_resnet18":
                continue
            for band, g in tt.groupby("band", observed=False):
                rows.append({"eval_set": es, "model": m, "band": str(band), "n": len(g),
                             "accuracy": g.correct.mean() if len(g) else np.nan,
                             "share_nn_same_genus": g.nn_train_same_genus.mean() if len(g) else np.nan})
            rows.append({"eval_set": es, "model": m, "band": "all", "n": len(tt), "accuracy": tt.correct.mean(),
                         "share_nn_same_genus": tt.nn_train_same_genus.mean()})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "spatial_check.csv", index=False, float_format="%.8f")
    print("test patches by distance to nearest training patch:")
    print(te.band.value_counts().sort_index().to_string())
    print("median distance (m):", float(np.median(d)))
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
