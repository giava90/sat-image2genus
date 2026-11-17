"""Run 2: run 1's small-CNN recipe (model 2) on input variants and/or splits.

--variant full   : 12 x 20 x 20 patch (run 1 model 2)
          centre : centre 12 x 6 x 6 only (the 60 m label window). The second
                   max-pool is replaced by identity, so 6 -> 3 px before the
                   last two conv layers and global average pooling.
          masked : 20 x 20 patch with the centre 6 x 6 set to 0 after per-band
                   standardisation (= the training mean), in every split.
--split  orig or a blocked split name (seed0, seed1)
Same recipe as run 1: 40 epochs, AdamW lr 2e-3, batch 256, cosine schedule,
dihedral augmentation, best epoch on validation accuracy, seed 0.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from common import CACHE, GENERA, SEED, device
from metrics import summary
from nets import SmallCNN
from torchtrain import fit, predict
from v2_features import centre_crop, mask_centre
from v2_splits import get_split
from v2common import save_predictions_v2


def build_model(variant: str) -> nn.Module:
    m = SmallCNN(in_ch=12, n_classes=len(GENERA))
    if variant == "centre":
        assert isinstance(m.features[5], nn.MaxPool2d)
        m.features[5] = nn.Identity()
    return m


def main(variant: str, split: str, name: str, sub: str, epochs: int, lr: float, bs: int, budget_min: float) -> None:
    torch.set_num_threads(8)
    dev = device()
    df = get_split(split)
    X = np.load(CACHE / "s2_200m.npy").astype(np.float32)
    assert len(X) == len(df)
    tr = (df.split == "train").to_numpy()
    va = (df.split == "val").to_numpy()
    ev = df.split.isin(["val", "test"]).to_numpy()
    if variant == "centre":
        X = np.ascontiguousarray(centre_crop(X))
    mean = X[tr].mean(axis=(0, 2, 3), keepdims=True)
    std = X[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    Xn = (X - mean) / std
    if variant == "masked":
        Xn = mask_centre(Xn, 0.0)
    Xn = torch.from_numpy(Xn)
    y = torch.from_numpy(df.y.to_numpy())
    model = build_model(variant)
    t0 = time.time()
    hist = fit(model, Xn[tr], y[tr], Xn[va], y[va], lambda b: b, model.parameters(), epochs=epochs, lr=lr, bs=bs,
               device=dev, weight_decay=1e-4, time_budget_s=budget_min * 60, seed=SEED,
               log=lambda s: print(f"[{name}] {s}", flush=True))
    train_s = time.time() - t0
    probs = predict(model, Xn[ev], lambda b: b, 1024, dev)
    s = df[ev]
    info = {"model": name, "variant": variant, "split": split, "device": str(dev),
            "params": sum(p.numel() for p in model.parameters()), "n_train": int(tr.sum()),
            "epochs": epochs, "lr": lr, "batch_size": bs, "train_seconds": round(train_s, 1),
            **{k: v for k, v in hist.items() if k != "history"}, "history": hist["history"]}
    save_predictions_v2(name, s.file, s.split, s.y, probs, info, sub=sub)
    te = (s.split == "test").to_numpy()
    r = summary(s.y[te], probs[te].argmax(1), len(GENERA))
    print(name, json.dumps({k: r[k] for k in ("n", "accuracy")}), f"train {train_s:.0f}s best epoch {hist['best_epoch']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=["full", "centre", "masked"])
    ap.add_argument("--split", default="orig")
    ap.add_argument("--name", required=True)
    ap.add_argument("--sub", default="stage2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--budget-min", type=float, default=15)
    a = ap.parse_args()
    main(a.variant, a.split, a.name, a.sub, a.epochs, a.lr, a.bs, a.budget_min)
