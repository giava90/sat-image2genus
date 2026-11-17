"""Run 2, stage 2 check: how does the s2/60m patch sit inside the s2/200m patch?

For a random sample of four-genus patches, read both GeoTIFFs, compare bounds,
and search every (row, col) offset of a 6 x 6 window inside the 200 m patch for
an exact pixel match with the 60 m patch. Writes results_v2/s2_60m_vs_200m.json.
"""
from __future__ import annotations

import json
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
from rasterio.io import MemoryFile

from common import RAW, RESULTS, SEED
from v2common import RESULTS_V2


def read(z, name):
    with MemoryFile(z.read(name)) as mf, mf.open() as ds:
        return ds.read(), ds.bounds, ds.res


def main(n: int = 1000) -> None:
    idx = pd.read_parquet(RESULTS / "index.parquet")
    rng = np.random.default_rng(SEED)
    files = idx.file.to_numpy()[rng.choice(len(idx), n, replace=False)]
    offsets, shapes60, shapes200, bound_off, n_multi = Counter(), Counter(), Counter(), Counter(), 0
    with zipfile.ZipFile(RAW / "s2.zip") as z:
        for fn in files:
            a60, b60, r60 = read(z, f"s2/60m/{fn}")
            a200, b200, r200 = read(z, f"s2/200m/{fn}")
            shapes60[str(a60.shape)] += 1
            shapes200[str(a200.shape)] += 1
            # geographic offset of the 60 m patch's top-left corner in 200 m pixels
            bound_off[(round((b200.top - b60.top) / r200[1], 2), round((b60.left - b200.left) / r200[0], 2))] += 1
            h, w = a60.shape[1:]
            hits = [(r, c) for r in range(a200.shape[1] - h + 1) for c in range(a200.shape[2] - w + 1)
                    if np.array_equal(a200[:, r:r + h, c:c + w], a60)]
            n_multi += len(hits) > 1
            offsets[str(hits[0]) if hits else "no exact match"] += 1
    out = {"n_sampled": n, "shape_60m": dict(shapes60), "shape_200m": dict(shapes200),
           "pixel_offset_of_exact_match(row,col)": dict(offsets.most_common()),
           "patches_with_more_than_one_matching_offset": n_multi,
           "georef_offset_of_60m_top_left_in_200m_pixels(row,col)": {str(k): v for k, v in bound_off.most_common()}}
    RESULTS_V2.mkdir(exist_ok=True)
    (RESULTS_V2 / "s2_60m_vs_200m.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
