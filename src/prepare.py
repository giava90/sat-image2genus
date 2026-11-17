"""Build the patch index (labels + split) and the Sentinel-2 cache.

Writes results/index.parquet: one row per target-genus patch with file, species,
genus, class id y, split (train/val/test), and whether aerial data was downloaded.
"""
from __future__ import annotations

from common import RESULTS
from data import build_index, build_s2_cache


def main() -> None:
    import zipfile

    from common import DATA, RAW

    lab = DATA / "labels" / "labels" / "TreeSatBA_v9_60m_multi_labels.json"
    if not lab.exists():
        with zipfile.ZipFile(RAW / "labels.zip") as z:
            z.extract("labels/TreeSatBA_v9_60m_multi_labels.json", DATA / "labels")
    df = build_index()
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS / "index.parquet", index=False)
    X = build_s2_cache(df)
    print(df.groupby(["genus", "split"]).size().unstack())
    print("aerial available:\n", df.groupby(["species", "aerial_available"]).size().unstack(fill_value=0))
    print("S2 cache", X.shape, X.dtype)


if __name__ == "__main__":
    main()
