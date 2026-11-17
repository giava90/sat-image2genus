"""Save and load per-patch predictions as parquet, plus per-model metric summaries."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import GENERA, RESULTS


def save_predictions(model: str, files, split, y_true, probs, extra: dict | None = None) -> pd.DataFrame:
    probs = np.asarray(probs, dtype=np.float32)
    df = pd.DataFrame({"file": list(files), "split": list(split), "y_true": np.asarray(y_true, int)})
    df["y_pred"] = probs.argmax(1)
    for i, g in enumerate(GENERA):
        df[f"p_{g}"] = probs[:, i]
    df["model"] = model
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS / f"predictions_{model}.parquet", index=False)
    if extra:
        (RESULTS / f"run_{model}.json").write_text(json.dumps(extra, indent=2))
    return df


def load_predictions(model: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS / f"predictions_{model}.parquet")
