"""Model 2: compact CNN trained from scratch on 12 x 20 x 20 Sentinel-2 patches (200 m).

Per-band standardisation with training-set statistics, dihedral augmentation,
AdamW + cosine schedule, best epoch chosen on validation accuracy.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from common import GENERA, RESULTS, SEED, device
from data import build_s2_cache
from metrics import summary
from nets import SmallCNN
from predio import save_predictions
from torchtrain import fit, predict


def main(epochs: int, lr: float, bs: int, budget_min: float) -> None:
    torch.set_num_threads(8)
    dev = device()
    df = pd.read_parquet(RESULTS / "index.parquet")
    X = build_s2_cache(df).astype(np.float32)
    tr = (df.split == "train").to_numpy()
    va = (df.split == "val").to_numpy()
    ev = df.split.isin(["val", "test"]).to_numpy()
    mean = X[tr].mean(axis=(0, 2, 3), keepdims=True)
    std = X[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    Xn = torch.from_numpy((X - mean) / std)
    y = torch.from_numpy(df.y.to_numpy())
    prep = lambda xb: xb  # already standardised  # noqa: E731
    model = SmallCNN(in_ch=12, n_classes=len(GENERA))
    t0 = time.time()
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    hist = fit(
        model, Xn[tr], y[tr], Xn[va], y[va], prep, model.parameters(),
        epochs=epochs, lr=lr, bs=bs, device=dev, weight_decay=1e-4,
        time_budget_s=budget_min * 60, seed=SEED, log=log,
    )
    train_s = time.time() - t0
    probs = predict(model, Xn[ev], prep, 1024, dev)
    sub = df[ev]
    info = {
        "model": "m2_cnn",
        "device": str(dev),
        "params": sum(p.numel() for p in model.parameters()),
        "n_train": int(tr.sum()),
        "epochs": epochs, "lr": lr, "batch_size": bs,
        "train_seconds": round(train_s, 1),
        **{k: v for k, v in hist.items() if k != "history"},
        "history": hist["history"],
    }
    save_predictions("m2_cnn", sub.file, sub.split, sub.y, probs, info)
    torch.save(model.state_dict(), RESULTS.parent / "data" / "cache" / "m2_cnn.pt")
    te = (sub.split == "test").to_numpy()
    s = summary(sub.y[te], probs[te].argmax(1), len(GENERA))
    print(json.dumps({k: s[k] for k in ("n", "accuracy", "weighted_precision", "macro_f1")}))
    print(f"train seconds {train_s:.0f}, best epoch {hist['best_epoch']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--budget-min", type=float, default=20)
    a = ap.parse_args()
    main(a.epochs, a.lr, a.bs, a.budget_min)
