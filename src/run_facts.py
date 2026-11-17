"""Save run facts cited in RESULTS.md to results/run_facts.json.

- label agreement between the file-name genus and the JSON's largest-share genus
- download total from data/download_ledger.csv (a copy goes to results/)
- class counts per split and evaluation set
- total training time of the three models
"""
from __future__ import annotations

import json
import shutil

import pandas as pd

from common import DATA, GENERA, RESULTS
from data import genus_of


def main() -> None:
    facts = {}
    lab = json.loads((DATA / "labels" / "labels" / "TreeSatBA_v9_60m_multi_labels.json").read_text())
    dis = sum(1 for f, v in lab.items() if max(v, key=lambda x: x[1])[0] != genus_of(f))
    facts["labels_all_patches"] = len(lab)
    facts["labels_all_filename_vs_json_top_disagree"] = dis
    facts["labels_all_disagree_share"] = dis / len(lab)
    idx = pd.read_parquet(RESULTS / "index.parquet")
    facts["labels_target_patches"] = len(idx)
    facts["labels_target_disagree_share"] = float((idx.json_top_genus != idx.genus).mean())
    facts["target_genus_counts"] = idx.genus.value_counts().reindex(GENERA).astype(int).to_dict()
    facts["split_counts"] = idx.split.value_counts().astype(int).to_dict()
    facts["common_counts"] = idx[idx.aerial_available].split.value_counts().astype(int).to_dict()
    facts["common_test_per_genus"] = idx[idx.aerial_available & (idx.split == "test")].genus.value_counts().reindex(GENERA).astype(int).to_dict()
    facts["full_test_per_genus"] = idx[idx.split == "test"].genus.value_counts().reindex(GENERA).astype(int).to_dict()
    # S2 200 m patches that are not 20 x 20 (cropped in data.build_s2_cache)
    import zipfile

    from rasterio.io import MemoryFile

    from common import RAW

    odd = 0
    with zipfile.ZipFile(RAW / "s2.zip") as z:
        for fn in idx.file:
            with MemoryFile(z.read(f"s2/200m/{fn}")) as mf, mf.open() as ds:
                odd += (ds.height, ds.width) != (20, 20)
    facts["s2_200m_patches_not_20x20"] = odd
    led = pd.read_csv(DATA / "download_ledger.csv")
    shutil.copy(DATA / "download_ledger.csv", RESULTS / "download_ledger.csv")
    facts["download_total_bytes"] = int(led.bytes.sum())
    facts["download_total_gb"] = led.bytes.sum() / 1e9
    tot = 0.0
    for m in ("m1_lgbm", "m2_cnn", "m3_resnet18"):
        p = RESULTS / f"run_{m}.json"
        if p.exists():
            r = json.loads(p.read_text())
            tot += r.get("train_seconds", 0) + r.get("feature_seconds", 0)
    facts["total_training_minutes_all_models"] = tot / 60
    (RESULTS / "run_facts.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
