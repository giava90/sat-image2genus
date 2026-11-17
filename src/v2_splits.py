"""Run 2: build blocked splits and give every training script the same split lookup.

Usage: python src/v2_splits.py [--seeds 0 1]
Writes results_v2/blocked_split_seed<k>.parquet (file, split) and a summary json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from blocks import BLOCK_M, BUFFER_M, block_ids, blocked_split
from common import GENERA, RESULTS
from v2common import RESULTS_V2


def index_with_centres() -> pd.DataFrame:
    idx = pd.read_parquet(RESULTS / "index.parquet")
    c = pd.read_parquet(RESULTS / "patch_centres.parquet").rename(columns={"x": "cx", "y": "cy"})
    df = idx.merge(c[["file", "cx", "cy"]], on="file", how="left")
    assert df.cx.notna().all() and len(df) == len(idx)
    df["block"] = block_ids(df.cx, df.cy)
    return df


def get_split(name: str) -> pd.DataFrame:
    """Index (run 1 order) with column 'split' replaced by the named split. 'orig' = run 1 split."""
    df = pd.read_parquet(RESULTS / "index.parquet")
    if name == "orig":
        return df
    s = pd.read_parquet(RESULTS_V2 / f"blocked_split_{name}.parquet")
    df = df.drop(columns="split").merge(s, on="file", how="left")
    assert df.split.notna().all()
    return df


def main(seeds) -> None:
    df = index_with_centres()
    summ = {"block_m": BLOCK_M, "buffer_m": BUFFER_M, "n_blocks": int(df.block.nunique()), "splits": {}}
    for k in seeds:
        s = blocked_split(df.cx, df.cy, 0.2, 0.1, seed=k)
        out = pd.DataFrame({"file": df.file, "split": s})
        out.to_parquet(RESULTS_V2 / f"blocked_split_seed{k}.parquet", index=False)
        t = df.assign(split=s)
        summ["splits"][f"seed{k}"] = {
            "counts": t.split.value_counts().to_dict(),
            "test_per_genus": t[t.split == "test"].genus.value_counts().reindex(GENERA).fillna(0).astype(int).to_dict(),
            "test_common_n": int((t.aerial_available & (t.split == "test")).sum()),
            "test_blocks": int(t[t.split == "test"].block.nunique()),
            "train_share_of_all": float((s == "train").mean()),
        }
    (RESULTS_V2 / "blocked_splits_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    main(ap.parse_args().seeds)
