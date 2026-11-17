"""Timed pilot: training throughput (images/s) for models 2 and 3 on this machine.

Writes results/pilot_timing.json. Used to size the real runs to the compute budget.
"""
from __future__ import annotations

import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import RESULTS, device
from nets import SmallCNN, freeze_resnet_except, resnet18_4band


def throughput(model: nn.Module, params, shape, bs: int, iters: int, dev, frozen_bn: bool = False) -> float:
    opt = torch.optim.AdamW(params, lr=1e-4)
    x = torch.randn(bs, *shape, device=dev)
    y = torch.randint(0, 4, (bs,), device=dev)
    model.to(dev)
    model.train()
    if frozen_bn:
        for name, ch in model.named_children():
            if not any(p.requires_grad for p in ch.parameters()):
                ch.eval()
    for i in range(iters + 2):
        if i == 2:
            t0 = time.time()
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return bs * iters / (time.time() - t0)


@torch.no_grad()
def infer_throughput(model, shape, bs, iters, dev) -> float:
    model.eval()
    x = torch.randn(bs, *shape, device=dev)
    model(x)
    t0 = time.time()
    for _ in range(iters):
        model(x)
    return bs * iters / (time.time() - t0)


def main() -> None:
    dev = device()
    torch.set_num_threads(8)
    out = {"device": str(dev), "threads": torch.get_num_threads(), "m2": {}, "m3": {}}
    m = SmallCNN()
    out["m2"]["params"] = sum(p.numel() for p in m.parameters())
    out["m2"]["train_img_per_s"] = throughput(m, m.parameters(), (12, 20, 20), 256, 20, dev)
    print("m2", out["m2"], flush=True)
    for size in (112, 128, 160, 224):
        r = {}
        m = resnet18_4band()
        r["full_train_img_per_s"] = throughput(m, m.parameters(), (4, size, size), 64, 4, dev)
        m = resnet18_4band()
        ps = freeze_resnet_except(m, ("layer4", "fc"))
        r["layer4_fc_train_img_per_s"] = throughput(m, ps, (4, size, size), 64, 4, dev, frozen_bn=True)
        m = resnet18_4band()
        ps = freeze_resnet_except(m, ("layer3", "layer4", "fc"))
        r["layer34_fc_train_img_per_s"] = throughput(m, ps, (4, size, size), 64, 4, dev, frozen_bn=True)
        r["infer_img_per_s"] = infer_throughput(m, (4, size, size), 128, 3, dev)
        out["m3"][size] = r
        print("m3", size, r, flush=True)
    (RESULTS / "pilot_timing.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
