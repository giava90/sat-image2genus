"""Run 2 figure: CNN minus LightGBM accuracy, original vs fair baseline vs blocked split (full test set).

Reads results_v2/stage2_paired.csv and stage3_paired.csv; writes results_v2/fig_v2_cnn_lead.png.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

import plotstyle as ps
from v2common import RESULTS_V2


def main() -> None:
    ps.apply()
    s2 = pd.read_csv(RESULTS_V2 / "stage2_paired.csv")
    s3 = pd.read_csv(RESULTS_V2 / "stage3_paired.csv")
    best = json.loads((RESULTS_V2 / "stage2_best_variant.json").read_text())["best_lgbm_variant_by_val_accuracy"]
    es = "full"
    rows = [
        ("original split\nvs run 1 LightGBM (200 m summary statistics)", s3[(s3.split == "original") & (s3.eval_set == es) & (s3.comparison == "M2 - M1")]),
        ("original split\nvs fair LightGBM (centre + ring statistics)", s2[(s2.eval_set == es) & (s2.comparison == f"M2 - {best}")]),
        ("blocked split, all retrained\nvs run 1 LightGBM features", s3[(s3.split == "blocked seed0") & (s3.eval_set == es) & (s3.comparison == "M2 - M1")]),
        ("blocked split, all retrained\nvs fair LightGBM (centre + ring statistics)", s3[(s3.split == "blocked seed0") & (s3.eval_set == es) & (s3.comparison == "M2 - best_lgbm")]),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    c = ps.MODEL_COLORS["m2_cnn"]
    for i, (lab, r) in enumerate(rows):
        r = r.iloc[0]
        y = len(rows) - 1 - i
        ax.plot([100 * r.block_ci_lo, 100 * r.block_ci_hi], [y, y], color=c, lw=2.5, solid_capstyle="round")
        ax.plot(100 * r["diff"], y, "o", ms=8, color=c)
        ax.text(100 * r.block_ci_hi + 0.4, y, f"{100 * r['diff']:+.1f}", va="center", fontsize=10, color=ps.TEXT_2)
    ax.axvline(0, color=ps.TEXT_2, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=9.5)
    ax.set_xlim(-0.5, 16)
    ax.set_xlabel("small CNN minus LightGBM, accuracy points (95% block-bootstrap interval)")
    ax.set_title("How much of the CNN's lead survives a fair baseline and a blocked split\n"
                 "full four-genus test sets (original split n = 2,738; blocked split n = 5,536)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_V2 / "fig_v2_cnn_lead.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
