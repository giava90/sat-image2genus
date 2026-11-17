"""Model 3: ImageNet-pretrained ResNet-18 fine-tuned on 4-band aerial patches.

Uses only patches whose species has aerial data (DECISIONS D6/D7). Training
patches are capped per genus; images are area-downscaled from 304 px to --size.
The first convolution is widened to 4 bands (NIR, G, B, R), see nets.py.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from common import CACHE, GENERA, RESULTS, SEED, device
from data import build_aerial_cache
from metrics import summary
from nets import freeze_resnet_except, resnet18_4band
from predio import save_predictions
from torchtrain import fit, predict


def select_rows(df: pd.DataFrame, cap: int, seed: int = SEED) -> pd.DataFrame:
    """Aerial-available rows: all val/test, training capped at `cap` per genus (random, fixed seed)."""
    a = df[df.aerial_available]
    rng = np.random.default_rng(seed)
    keep = []
    for g, grp in a[a.split == "train"].groupby("genus"):
        idx = grp.index.to_numpy()
        keep.extend(rng.choice(idx, min(cap, len(idx)), replace=False).tolist())
    keep.extend(a[a.split != "train"].index.tolist())
    return df.loc[sorted(keep)].reset_index(drop=True)


def main(size: int, cap: int, epochs: int, lr: float, bs: int, budget_min: float, mode: str, task: str) -> None:
    torch.set_num_threads(8)
    dev = device()
    df = select_rows(pd.read_parquet(RESULTS / "index.parquet"), cap)
    t0 = time.time()
    files_path = CACHE / f"aerial_{size}_files.txt"
    cache_path = CACHE / f"aerial_{size}.npy"
    if files_path.exists() and files_path.read_text().split("\n") != df.file.tolist():
        cache_path.unlink(missing_ok=True)  # different selection: rebuild
    X = build_aerial_cache(df, size, out=cache_path)
    files_path.write_text("\n".join(df.file))
    cache_s = time.time() - t0
    print(f"aerial cache {X.shape} built/loaded in {cache_s:.0f}s", flush=True)

    tr = (df.split == "train").to_numpy()
    va = (df.split == "val").to_numpy()
    ev = df.split.isin(["val", "test"]).to_numpy()
    # per-band means on training patches, as a check of the band order (NIR brightest over vegetation)
    band_means = {g: X[tr & (df.genus == g).to_numpy()].mean(axis=(0, 2, 3)).round(1).tolist() for g in GENERA}
    print("band means (NIR,G,B,R) per genus:", band_means, flush=True)
    Xt = torch.from_numpy(X)
    xf = Xt[torch.from_numpy(tr)].float() / 255.0
    mean = xf.mean(dim=(0, 2, 3)).view(1, 4, 1, 1)
    std = xf.std(dim=(0, 2, 3)).view(1, 4, 1, 1)
    del xf

    def prep(xb):
        return (xb.float() / 255.0 - mean) / std

    if task == "binary":
        from common import GENUS_TO_BINARY

        y_all = torch.tensor([GENUS_TO_BINARY[v] for v in df.y])
        n_classes = 2
    else:
        y_all = torch.from_numpy(df.y.to_numpy())
        n_classes = len(GENERA)

    model = resnet18_4band(n_classes=n_classes, pretrained=True)
    if mode == "full":
        params = list(model.parameters())
        train_mode_fn = None
    else:  # "last": layer4 + fc only, frozen BN statistics elsewhere
        params = freeze_resnet_except(model, ("layer4", "fc"))

        def train_mode_fn(m):
            m.train()
            for name, ch in m.named_children():
                if name not in ("layer4", "fc"):
                    ch.eval()

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    t1 = time.time()
    hist = fit(
        model, Xt[torch.from_numpy(tr)], y_all[torch.from_numpy(tr)],
        Xt[torch.from_numpy(va)], y_all[torch.from_numpy(va)], prep, params,
        epochs=epochs, lr=lr, bs=bs, device=dev, weight_decay=1e-4,
        time_budget_s=budget_min * 60, seed=SEED, log=log, train_mode_fn=train_mode_fn,
    )
    train_s = time.time() - t1
    probs = predict(model, Xt[torch.from_numpy(ev)], prep, 128, dev)
    sub = df[ev]
    info = {
        "model": "m3_resnet18",
        "device": str(dev),
        "task": task,
        "image_size": size,
        "cap_per_genus": cap,
        "mode": mode,
        "first_conv": "widened to 4 bands (NIR,G,B,R); RGB filters copied, NIR = mean of RGB filters; x0.75",
        "n_train": int(tr.sum()),
        "n_train_per_genus": df[tr].genus.value_counts().to_dict(),
        "epochs": epochs, "lr": lr, "batch_size": bs,
        "cache_seconds": round(cache_s, 1),
        "train_seconds": round(train_s, 1),
        "band_means_train": band_means,
        **{k: v for k, v in hist.items() if k != "history"},
        "history": hist["history"],
    }
    if task == "binary":
        # store binary probabilities in the first two prob columns; genus columns unused
        p4 = np.zeros((len(probs), 4), np.float32)
        p4[:, :2] = probs
        probs = p4
    save_predictions("m3_resnet18", sub.file, sub.split, sub.y, probs, info)
    torch.save(model.state_dict(), CACHE / "m3_resnet18.pt")
    te = (sub.split == "test").to_numpy()
    if task != "binary":
        s = summary(sub.y[te], probs[te].argmax(1), len(GENERA))
        print(json.dumps({k: s[k] for k in ("n", "accuracy", "weighted_precision", "macro_f1")}))
    print(f"train seconds {train_s:.0f}, best epoch {hist['best_epoch']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--budget-min", type=float, default=80)
    ap.add_argument("--mode", choices=["full", "last"], default="full")
    ap.add_argument("--task", choices=["genus", "binary"], default="genus")
    a = ap.parse_args()
    main(a.size, a.cap, a.epochs, a.lr, a.bs, a.budget_min, a.mode, a.task)
