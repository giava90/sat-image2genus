"""Diagram of the small CNN (model 2) with tensor shapes and parameter counts taken from the model itself.

Writes results_v2/fig_cnn_architecture.png and results_v2/cnn_architecture.json.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, Rectangle

import plotstyle as ps
from nets import SmallCNN
from v2common import RESULTS_V2


def n_params(*mods) -> int:
    return int(sum(p.numel() for m in mods for p in m.parameters()))


def main() -> None:
    ps.apply()
    m = SmallCNN(in_ch=12, n_classes=4).eval()
    f = m.features
    x = torch.zeros(1, 12, 20, 20)
    shapes = [tuple(x.shape[1:])]
    with torch.no_grad():
        for i, layer in enumerate(f):
            x = layer(x)
            if i in (1, 2, 4, 5, 7, 8):
                shapes.append(tuple(x.shape[1:]))
        out = m.head(f[9](x))
    shapes.append(tuple(out.shape[1:]))
    # nodes: input, conv pair 1, pool, conv pair 2, pool, conv pair 3, global pool, output
    ops = [
        ("2 × (3×3 convolution,\nbatch norm, ReLU)", n_params(f[0], f[1])),
        ("max pool\n2×2", 0),
        ("2 × (3×3 convolution,\nbatch norm, ReLU)", n_params(f[3], f[4])),
        ("max pool\n2×2", 0),
        ("2 × (3×3 convolution,\nbatch norm, ReLU)", n_params(f[6], f[7])),
        ("global average\npool", 0),
        ("dropout 0.3,\nlinear layer", n_params(m.head)),
    ]
    total = n_params(m)
    assert total == sum(p for _, p in ops)
    (RESULTS_V2 / "cnn_architecture.json").write_text(json.dumps(
        {"shapes": [list(s) for s in shapes], "ops": [{"op": o.replace("\n", " "), "params": p} for o, p in ops],
         "total_params": total, "conv_layers": 6, "linear_layers": 1}, indent=2))

    def label(s):
        return f"{s[0]} × {s[1]} × {s[2]}" if len(s) == 3 else f"{s[0]}"

    fig, ax = plt.subplots(figsize=(15, 5.2))
    gap = 2.45
    xpos, x0 = [], 0.0
    for s in shapes:
        w = 0.11 * np.sqrt(s[0])
        xpos.append((x0, w))
        x0 += w + gap
    cnn = ps.MODEL_COLORS["m2_cnn"]
    for k, (s, (xl, w)) in enumerate(zip(shapes, xpos)):
        h = 0.17 * (s[1] if len(s) == 3 else 1)
        face = ps.SURFACE if k in (0, len(shapes) - 1) else cnn
        alpha = 1.0 if k in (0, len(shapes) - 1) else 0.35 + 0.65 * (k / len(shapes))
        ax.add_patch(Rectangle((xl, -h / 2), w, h, facecolor=face, alpha=alpha if face == cnn else 1,
                               edgecolor=cnn if face == cnn else ps.TEXT_2, lw=1.4))
        ax.text(xl + w / 2, -1.95, label(s), ha="center", va="top", fontsize=9.5, color=ps.TEXT_2)
    ax.text(xpos[0][0] + xpos[0][1] / 2, 1.95, "input:\n12 bands\n× 20 × 20 px", ha="center", va="bottom", fontsize=9.5)
    xl, w = xpos[-1]
    ax.text(xl + w / 2, 1.95, "output:\nprobabilities of\nspruce, pine,\nbeech, oak", ha="center", va="bottom", fontsize=9.5)
    for k, (op, p) in enumerate(ops):
        a = xpos[k][0] + xpos[k][1] + 0.08
        b = xpos[k + 1][0] - 0.08
        ax.add_patch(FancyArrowPatch((a, 0), (b, 0), arrowstyle="-|>", mutation_scale=12, color=ps.TEXT_2, lw=1.2))
        ax.text((a + b) / 2, 0.22, op, ha="center", va="bottom", fontsize=8.4)
        ax.text((a + b) / 2, -0.22, f"{p:,} parameters" if p else "no parameters", ha="center", va="top",
                fontsize=8.2, color=ps.TEXT_2)
    ax.text(xpos[0][0], -2.55, "Box height = image size in pixels; box width grows with the number of channels. "
            "Labels below the boxes: channels × height × width.", fontsize=9, color=ps.TEXT_2)
    ax.set_xlim(-0.3, x0 - gap + 0.3)
    ax.set_ylim(-2.8, 3.2)
    ax.axis("off")
    ax.set_title(f"The small CNN (model 2): 6 convolution layers and 1 linear layer, {total:,} trainable parameters\n"
                 "Input: one 200 m Sentinel-2 patch with 10 m pixels", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_V2 / "fig_cnn_architecture.png")
    plt.close(fig)
    print(json.dumps({"shapes": shapes, "ops": ops, "total": total}, default=str))


if __name__ == "__main__":
    main()
