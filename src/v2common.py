"""Run 2 shared helpers: output folder and prediction saving that never touches results/."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import GENERA, ROOT

RESULTS_V2 = ROOT / "results_v2"


def save_predictions_v2(name: str, files, split, y_true, probs, extra: dict | None = None, sub: str = "") -> pd.DataFrame:
    """Like predio.save_predictions, but under results_v2/[sub]/."""
    probs = np.asarray(probs, dtype=np.float32)
    df = pd.DataFrame({"file": list(files), "split": list(split), "y_true": np.asarray(y_true, int)})
    df["y_pred"] = probs.argmax(1)
    for i, g in enumerate(GENERA):
        df[f"p_{g}"] = probs[:, i]
    df["model"] = name
    out = RESULTS_V2 / sub if sub else RESULTS_V2
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"predictions_{name}.parquet", index=False)
    if extra:
        (out / f"run_{name}.json").write_text(json.dumps(extra, indent=2, default=str))
    return df


def load_predictions_v2(name: str, sub: str = "") -> pd.DataFrame:
    out = RESULTS_V2 / sub if sub else RESULTS_V2
    return pd.read_parquet(out / f"predictions_{name}.parquet")
