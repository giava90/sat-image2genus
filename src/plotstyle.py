"""Shared matplotlib style: validated 3-slot categorical palette, recessive axes."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
# categorical slots 1-3 (validated all-pairs, light mode): fixed per model
MODEL_COLORS = {"m1_lgbm": "#2a78d6", "m2_cnn": "#eb6834", "m3_resnet18": "#1baf7a"}
MODEL_LABELS = {
    "m1_lgbm": "M1 LightGBM, S2 statistics",
    "m2_cnn": "M2 small CNN, S2 patches",
    "m3_resnet18": "M3 ResNet-18, aerial",
}
MODEL_SHORT = {"m1_lgbm": "M1 LightGBM", "m2_cnn": "M2 CNN", "m3_resnet18": "M3 ResNet-18"}
# sequential blue ramp (light -> dark) for confusion matrices
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 9.5,
            "text.color": TEXT,
            "savefig.dpi": 160,
        }
    )
