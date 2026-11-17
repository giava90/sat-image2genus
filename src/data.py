"""Index of target-genus patches, train/val/test split, and array caches.

Labels: the class of a patch is the genus of its file-name species
(see DECISIONS.md D3). The archive's fixed train/test lists are used as is;
a validation set is carved out of the training list, stratified by species.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import CACHE, GENERA, RAW, SEED

VAL_FRACTION = 0.10


def species_of(fn: str) -> str:
    """'Picea_abies_3_1234_WEFL_NLF.tif' -> 'Picea_abies'; 'Cleared_0_1_X.tif' -> 'Cleared'."""
    parts = Path(fn).name.split("_")
    for i, p in enumerate(parts):
        if p.isdigit():
            return "_".join(parts[:i])
    raise ValueError(f"no numeric token in {fn}")


def genus_of(fn: str) -> str:
    return species_of(fn).split("_")[0]


def aerial_zip_name(species: str) -> str:
    return f"aerial_60m_{species.lower()}.zip"


def read_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def build_index(raw: Path = RAW, val_fraction: float = VAL_FRACTION, seed: int = SEED) -> pd.DataFrame:
    rows = []
    for split, lst in (("train", "train_filenames.lst"), ("test", "test_filenames.lst")):
        for fn in read_list(raw / lst):
            g = genus_of(fn)
            if g in GENERA:
                rows.append((fn, species_of(fn), g, GENERA.index(g), split))
    df = pd.DataFrame(rows, columns=["file", "species", "genus", "y", "split"])
    # validation carve-out: stratified by species, fixed seed
    rng = np.random.default_rng(seed)
    is_train = df["split"] == "train"
    for sp, grp in df[is_train].groupby("species"):
        idx = grp.index.to_numpy()
        n_val = int(round(len(idx) * val_fraction))
        df.loc[rng.choice(idx, n_val, replace=False), "split"] = "val"
    df["aerial_zip"] = df["species"].map(aerial_zip_name)
    sizes = {}
    meta_path = raw.parent / "zenodo_meta" / "record_6780578.json"
    if meta_path.exists():
        sizes = {f["key"]: int(f["size"]) for f in json.loads(meta_path.read_text())["files"]}
    # available = zip present and complete (download.py deletes files that fail md5)
    df["aerial_available"] = df["aerial_zip"].map(
        lambda z: (raw / z).exists() and (raw / z).stat().st_size == sizes.get(z, -1)
    )
    # multi-label proportions from the JSON, kept for reference only
    lab_path = raw.parent / "labels" / "labels" / "TreeSatBA_v9_60m_multi_labels.json"
    if lab_path.exists():
        lab = json.loads(lab_path.read_text())
        df["json_top_genus"] = df["file"].map(lambda f: max(lab[f], key=lambda x: x[1])[0])
        df["json_share_of_label"] = df.apply(
            lambda r: dict((g, s) for g, s in lab[r["file"]]).get(r["genus"], 0.0), axis=1
        )
    return df.sort_values("file").reset_index(drop=True)


def _read_tif_bytes(buf: bytes) -> np.ndarray:
    from rasterio.io import MemoryFile

    with MemoryFile(buf) as mf, mf.open() as ds:
        return ds.read()


def build_s2_cache(df: pd.DataFrame, raw: Path = RAW, out: Path = CACHE / "s2_200m.npy") -> np.ndarray:
    """Read the 200 m Sentinel-2 patches for every row of df, in df order. Shape (N, 12, 20, 20)."""
    if out.exists():
        arr = np.load(out)
        if len(arr) == len(df):
            return arr
    out.parent.mkdir(parents=True, exist_ok=True)
    arrs = []
    with zipfile.ZipFile(raw / "s2.zip") as z:
        for fn in df["file"]:
            # 48 of 27,656 target patches are 21 px on one side: keep the first 20 rows/cols
            arrs.append(_read_tif_bytes(z.read(f"s2/200m/{fn}"))[:, :20, :20])
    arr = np.stack(arrs)
    np.save(out, arr)
    return arr


def build_aerial_cache(df: pd.DataFrame, size: int, raw: Path = RAW, out: Path | None = None) -> np.ndarray:
    """Read aerial patches for every row of df (all must have aerial_available), resize to size x size.

    Returns uint8 array (N, 4, size, size), band order as stored in the files.
    """
    import torch
    import torch.nn.functional as F

    out = out or CACHE / f"aerial_{size}.npy"
    if out.exists():
        arr = np.load(out)
        if len(arr) == len(df):
            return arr
    out.parent.mkdir(parents=True, exist_ok=True)
    result = np.zeros((len(df), 4, size, size), dtype=np.uint8)
    pos = {f: i for i, f in enumerate(df["file"])}
    for zname, grp in df.groupby("aerial_zip"):
        with zipfile.ZipFile(raw / zname) as z:
            names = {Path(n).name: n for n in z.namelist() if n.endswith(".tif")}
            for fn in grp["file"]:
                a = _read_tif_bytes(z.read(names[fn]))[:4]
                t = torch.from_numpy(a.astype(np.float32))[None]
                t = F.interpolate(t, size=(size, size), mode="area")
                result[pos[fn]] = t[0].round().clamp(0, 255).byte().numpy()
    np.save(out, result)
    return result
