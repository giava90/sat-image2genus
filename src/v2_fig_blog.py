"""Web version of the oak-sweep figure for the blog post (plain labels, no project shorthand).

Reads results_v2/stage1_oak_sweep.csv (same data as fig_v2_oak_sweep.png) and writes
results_v2/fig_v2_oak_sweep_blog.png. The placeholder status of volumes and regions is
stated in the blog text instead of the figure title.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

import plotstyle as ps
from v2common import RESULTS_V2

PANELS = [("cc", "Counting labels"), ("ac", "Adjusted count"), ("pcc", "Counting probabilities"),
          ("pac", "Adjusted probabilities"), ("em", "EM correction")]
MODELS = [("m1_lgbm", "LightGBM, Sentinel-2 statistics"), ("m2_cnn", "Small CNN, Sentinel-2 patches"),
          ("m3_resnet18", "ResNet-18, aerial photos")]


def main() -> None:
    ps.apply()
    sw = pd.read_csv(RESULTS_V2 / "stage1_oak_sweep.csv")
    fig, axes = plt.subplots(1, len(PANELS), figsize=(14, 4.4), sharey=True)
    for ax, (e, title) in zip(axes, PANELS):
        ax.axhline(0, color=ps.TEXT_2, lw=1)
        for m, lab in MODELS:
            d = sw[(sw.model == m) & (sw.estimator == e)]
            c = ps.MODEL_COLORS[m]
            ax.fill_between(d.oak_share * 100, d.p2_5 * 100, d.p97_5 * 100, color=c, alpha=0.15, lw=0)
            ax.plot(d.oak_share * 100, d.bias * 100, color=c, lw=2, label=lab)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("true oak share, %")
    axes[0].set_ylabel("error in oak volume, %")
    axes[0].set_ylim(-60, 150)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle("How wrong is the estimated oak volume when oak is rare?\n"
                 "Line: average error over 1,000 simulated regions. Band: range of 95% of the regions. "
                 "The axis is cut at -60% and +150%.", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    fig.savefig(RESULTS_V2 / "fig_v2_oak_sweep_blog.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
