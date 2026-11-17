"""Minimal in-memory training loop for small image tensors on CPU or GPU."""
from __future__ import annotations

import copy
import math
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def dihedral_augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random flips and transposes per sample (all 8 symmetries of a square). x: (B, C, H, W), H == W."""
    b = x.shape[0]
    m = torch.rand(b, 3, generator=gen) < 0.5
    m = m.to(x.device)
    x = torch.where(m[:, 0, None, None, None], x.flip(3), x)
    x = torch.where(m[:, 1, None, None, None], x.flip(2), x)
    x = torch.where(m[:, 2, None, None, None], x.transpose(2, 3), x)
    return x


def batches(n: int, bs: int, shuffle: bool, gen: torch.Generator | None = None):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield idx[i : i + bs]


@torch.no_grad()
def predict(model: nn.Module, X, prep: Callable, bs: int, device) -> np.ndarray:
    model.eval()
    out = []
    for idx in batches(len(X), bs, shuffle=False):
        xb = prep(X[idx]).to(device)
        out.append(F.softmax(model(xb), 1).cpu())
    return torch.cat(out).numpy()


def fit(
    model: nn.Module,
    Xtr,
    ytr: torch.Tensor,
    Xva,
    yva: torch.Tensor,
    prep: Callable,
    params,
    epochs: int,
    lr: float,
    bs: int,
    device,
    weight_decay: float = 1e-4,
    time_budget_s: float | None = None,
    seed: int = 0,
    log: Callable = print,
    train_mode_fn: Callable | None = None,
    class_weight: torch.Tensor | None = None,
):
    """Train with AdamW + cosine schedule (1 warm-up epoch). Keeps the epoch with best val accuracy.

    prep(x_raw_batch) -> float tensor ready for the model. train_mode_fn(model) sets train/eval
    on submodules (for frozen-BN fine-tuning); default model.train().
    """
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(Xtr) / bs)
    total = epochs * steps_per_epoch
    warm = steps_per_epoch

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    cw = class_weight.to(device) if class_weight is not None else None
    best = {"acc": -1.0}
    history = []
    t_start = time.time()
    for ep in range(epochs):
        t0 = time.time()
        (train_mode_fn or (lambda m: m.train()))(model)
        tot_loss, n_seen = 0.0, 0
        for idx in batches(len(Xtr), bs, shuffle=True, gen=gen):
            xb = dihedral_augment(prep(Xtr[idx]), gen).to(device)
            yb = ytr[idx].to(device)
            loss = F.cross_entropy(model(xb), yb, weight=cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(idx)
            n_seen += len(idx)
        pva = predict(model, Xva, prep, bs * 2, device)
        va_loss = float(F.nll_loss(torch.log(torch.from_numpy(pva).clamp_min(1e-8)), yva).item())
        va_acc = float((pva.argmax(1) == yva.numpy()).mean())
        rec = {
            "epoch": ep + 1,
            "train_loss": tot_loss / n_seen,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "seconds": time.time() - t0,
        }
        history.append(rec)
        log(f"epoch {ep + 1}/{epochs} train_loss {rec['train_loss']:.4f} val_loss {va_loss:.4f} val_acc {va_acc:.4f} ({rec['seconds']:.0f}s)")
        if va_acc > best["acc"]:
            best = {"acc": va_acc, "epoch": ep + 1, "state": copy.deepcopy(model.state_dict())}
        if time_budget_s is not None and time.time() - t_start > time_budget_s and ep + 1 < epochs:
            log(f"time budget reached after epoch {ep + 1}; stopping")
            break
    model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"], "best_val_acc": best["acc"], "history": history, "epochs_run": len(history)}
