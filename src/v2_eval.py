"""Run 2 evaluation helpers: accuracy with patch- and block-bootstrap intervals, paired differences.

Resamples are drawn once per evaluation set and shared by all models (paired).
Patch bootstrap: resample test patches. Block bootstrap: resample 2 km grid blocks
(blocks.block_ids on the patch centres) with all their test patches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from blocks import block_bootstrap, block_ids
from common import RESULTS, SEED
from predio import load_predictions
from v2common import load_predictions_v2

N_BOOT = 1000


def centres() -> pd.DataFrame:
    c = pd.read_parquet(RESULTS / "patch_centres.parquet")
    c["block"] = block_ids(c.x, c.y)
    return c.set_index("file")


def load(ref: str) -> pd.DataFrame:
    """'run1:m2_cnn' -> results/predictions_m2_cnn.parquet; 'stage2:cnn_centre6' -> results_v2/stage2/..."""
    src, name = ref.split(":", 1)
    return load_predictions(name) if src == "run1" else load_predictions_v2(name, sub=src)


def test_frame(ref: str, files: set | None) -> pd.DataFrame:
    p = load(ref)
    p = p[p.split == "test"]
    if files is not None:
        p = p[p.file.isin(files)]
    return p.sort_values("file").reset_index(drop=True)


def score(refs: dict[str, str], files: set | None, pairs: list[tuple[str, str]], n_boot: int = N_BOOT, seed: int = SEED):
    """refs: label -> prediction ref. pairs: (a, b) meaning accuracy(b) - accuracy(a).
    Returns (accuracy rows, paired-difference rows), each with patch and block intervals."""
    frames = {k: test_frame(v, files) for k, v in refs.items()}
    first = next(iter(frames.values()))
    for k, f in frames.items():
        assert f.file.tolist() == first.file.tolist(), f"{k}: test patches differ"
    n = len(first)
    blk = centres().loc[first.file, "block"].to_numpy()
    rng = np.random.default_rng(seed)
    patch_idx = rng.integers(0, n, (n_boot, n))
    block_idx = block_bootstrap(blk, n_boot, seed=seed + 1)
    correct = {k: (f.y_pred.to_numpy() == f.y_true.to_numpy()).astype(float) for k, f in frames.items()}
    acc_p = {k: c[patch_idx].mean(1) for k, c in correct.items()}
    acc_b = {k: np.array([c[i].mean() for i in block_idx]) for k, c in correct.items()}
    rows = []
    for k, c in correct.items():
        rows.append({"model": k, "ref": refs[k], "n": n, "n_blocks": int(len(np.unique(blk))), "accuracy": c.mean(),
                     "patch_ci_lo": np.percentile(acc_p[k], 2.5), "patch_ci_hi": np.percentile(acc_p[k], 97.5),
                     "block_ci_lo": np.percentile(acc_b[k], 2.5), "block_ci_hi": np.percentile(acc_b[k], 97.5)})
    drows = []
    for a, b in pairs:
        if a not in correct or b not in correct:
            continue
        dp, db = acc_p[b] - acc_p[a], acc_b[b] - acc_b[a]
        drows.append({"comparison": f"{b} - {a}", "a": a, "b": b, "n": n, "diff": correct[b].mean() - correct[a].mean(),
                      "patch_ci_lo": np.percentile(dp, 2.5), "patch_ci_hi": np.percentile(dp, 97.5),
                      "block_ci_lo": np.percentile(db, 2.5), "block_ci_hi": np.percentile(db, 97.5),
                      "patch_share_b_better": float((dp > 0).mean()), "block_share_b_better": float((db > 0).mean())})
    return pd.DataFrame(rows), pd.DataFrame(drows)
