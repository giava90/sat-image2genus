"""Volume step: from saved predictions to per-genus wood-volume error. Reruns in seconds.

Reads results/predictions_*.parquet, results/index.parquet and volumes.yaml.
For each model and scenario, estimates area per genus by counting predictions
and with a confusion-matrix adjustment (validation matrix applied to test
predictions), converts to volume, and bootstraps the test set.

Writes results/volume_summary.csv, results/volume_bootstrap.parquet,
results/fig3_volume_error.png.

Usage: python src/volume_step.py [--eval-set common|full] [--n-boot 1000]
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import plotstyle as ps
from common import GENERA, RESULTS, ROOT, SEED
from metrics import accuracy, confusion_matrix, row_normalize
from predio import load_predictions
from volume import bootstrap_volume_errors

MODELS = ["m1_lgbm", "m2_cnn", "m3_resnet18"]
TOTAL_AREA_HA = 10_000.0
TARGETS = GENERA + ["Total"]


def load_volumes(path=ROOT / "volumes.yaml"):
    cfg = yaml.safe_load(path.read_text())
    gs = np.array([float(cfg["growing_stock"][g]) for g in GENERA])
    return gs, bool(cfg.get("placeholder", True)), cfg.get("unit", "m3_per_ha")


def scenarios(index: pd.DataFrame) -> dict[str, np.ndarray]:
    counts = index.genus.value_counts().reindex(GENERA).to_numpy(float)
    return {
        "S1 dataset shares": counts / counts.sum(),
        "S2 equal shares": np.full(len(GENERA), 1 / len(GENERA)),
    }


def frames(model: str, index: pd.DataFrame, eval_set: str):
    p = load_predictions(model)
    if eval_set == "common":
        p = p[p.file.isin(set(index.loc[index.aerial_available, "file"]))]
    return p[p.split == "val"], p[p.split == "test"]


def main(eval_set: str, n_boot: int) -> None:
    ps.apply()
    index = pd.read_parquet(RESULTS / "index.parquet")
    gs, placeholder, unit = load_volumes()
    tag = " (placeholder volumes)" if placeholder else ""
    models = [m for m in MODELS if (RESULTS / f"predictions_{m}.parquet").exists()]
    if eval_set == "full":
        models = [m for m in models if m != "m3_resnet18"]
    scen = scenarios(index)
    rows, boot_rows = [], []
    meta = {"eval_set": eval_set, "n_boot": n_boot, "total_area_ha": TOTAL_AREA_HA, "placeholder_volumes": placeholder,
            "growing_stock": dict(zip(GENERA, gs.tolist())), "unit": unit,
            "scenario_shares": {k: dict(zip(GENERA, v.round(4).tolist())) for k, v in scen.items()}, "models": {}}
    for m in models:
        val, test = frames(m, index, eval_set)
        cm_val = confusion_matrix(val.y_true, val.y_pred, len(GENERA))
        cond_val = row_normalize(cm_val)
        cm_test = confusion_matrix(test.y_true, test.y_pred, len(GENERA))
        meta["models"][m] = {"n_val": len(val), "n_test": len(test),
                             "test_accuracy": accuracy(cm_test),
                             "val_confusion": cm_val.tolist(),
                             "test_confusion": cm_test.tolist(),
                             "val_rownorm": np.round(cond_val, 4).tolist(),
                             "test_rownorm": np.round(row_normalize(cm_test), 4).tolist()}
        for sname, shares in scen.items():
            r = bootstrap_volume_errors(test.y_true, test.y_pred, cond_val, shares, gs, len(GENERA),
                                        n_boot=n_boot, total_area=TOTAL_AREA_HA, seed=SEED)
            for est in ("count", "adjusted"):
                mat = np.column_stack([r[est]["per_genus"], r[est]["total"]])  # (n_boot, 5)
                pt = list(r["point"][est]["per_genus"]) + [r["point"][est]["total"]]
                mae = np.abs(r[est]["per_genus"]).mean(1)
                for t, target in enumerate(TARGETS):
                    col = mat[:, t]
                    rows.append({"model": m, "scenario": sname, "estimator": est, "target": target,
                                 "point": pt[t], "median": np.median(col), "p2_5": np.percentile(col, 2.5),
                                 "p97_5": np.percentile(col, 97.5), "boot_sd": col.std()})
                rows.append({"model": m, "scenario": sname, "estimator": est, "target": "MeanAbsGenus",
                             "point": float(np.mean(np.abs(pt[:4]))), "median": np.median(mae),
                             "p2_5": np.percentile(mae, 2.5), "p97_5": np.percentile(mae, 97.5), "boot_sd": mae.std()})
                boot_rows.append(pd.DataFrame(mat, columns=TARGETS).assign(model=m, scenario=sname, estimator=est, boot=np.arange(n_boot)))
    summ = pd.DataFrame(rows)
    suffix = "" if eval_set == "common" else "_full"
    summ.to_csv(RESULTS / f"volume_summary{suffix}.csv", index=False, float_format="%.8f")
    pd.concat(boot_rows).to_parquet(RESULTS / f"volume_bootstrap{suffix}.parquet", index=False)
    (RESULTS / f"volume_meta{suffix}.json").write_text(json.dumps(meta, indent=2))
    fig3(summ, models, tag, eval_set, meta, suffix)
    show = summ[summ.target.isin(["Picea", "Total", "MeanAbsGenus"])].copy()
    for c in ("point", "median", "p2_5", "p97_5"):
        show[c] = (show[c] * 100).round(2)
    print(f"volume errors in % ({eval_set} test set){tag}")
    print(show[["model", "scenario", "estimator", "target", "point", "p2_5", "p97_5"]].to_string(index=False))


def fig3(summ: pd.DataFrame, models, tag: str, eval_set: str, meta: dict, suffix: str) -> None:
    scen_names = list(meta["scenario_shares"])
    fig, axes = plt.subplots(len(scen_names), 1, figsize=(12.5, 4.0 * len(scen_names)), sharex=True)
    axes = np.atleast_1d(axes)
    k = len(models)
    slot = 0.84 / (2 * k)
    for ax, sname in zip(axes, scen_names):
        ax.axhline(0, color=ps.TEXT_2, lw=1, zorder=1)
        for t_i, target in enumerate(TARGETS):
            for j, m in enumerate(models):
                for e_i, est in enumerate(("count", "adjusted")):
                    r = summ[(summ.model == m) & (summ.scenario == sname) & (summ.estimator == est) & (summ.target == target)].iloc[0]
                    x = t_i + (2 * j + e_i - (2 * k - 1) / 2) * slot
                    c = ps.MODEL_COLORS[m]
                    ax.plot([x, x], [r.p2_5 * 100, r.p97_5 * 100], color=c, lw=2, solid_capstyle="round", zorder=2)
                    ax.plot(x, r.point * 100, marker="o", ms=7, zorder=3,
                            mfc=c if est == "count" else ps.SURFACE, mec=c, mew=1.8)
        ax.set_xticks(range(len(TARGETS)), [f"{t}" for t in TARGETS[:-1]] + ["Total volume"])
        ax.axvline(len(GENERA) - 0.5, color=ps.GRID, lw=1)
        ax.set_ylabel("volume error, % of true")
        shares = ", ".join(f"{g} {v:.0%}" for g, v in meta["scenario_shares"][sname].items())
        ax.set_title(f"{sname}: {shares}", loc="left", fontsize=10.5)
    # legend: model colour + estimator fill (secondary encoding)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=ps.MODEL_COLORS[m], lw=2, marker="o", mfc=ps.MODEL_COLORS[m], mec=ps.MODEL_COLORS[m], label=ps.MODEL_LABELS[m]) for m in models]
    handles += [Line2D([], [], color=ps.TEXT_2, lw=0, marker="o", mfc=ps.TEXT_2, mec=ps.TEXT_2, label="counting predictions"),
                Line2D([], [], color=ps.TEXT_2, lw=0, marker="o", mfc=ps.SURFACE, mec=ps.TEXT_2, mew=1.8, label="confusion-matrix adjusted")]
    axes[-1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5)
    fig.suptitle(f"Wood-volume error per genus for a synthetic region{tag}\n"
                 f"point estimate on the {eval_set} test set with 95% bootstrap interval ({meta['n_boot']} resamples)",
                 x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.005, "Per-genus errors equal the area errors and do not depend on the growing-stock values; only the total does."
             + (" Growing stock values are PLACEHOLDERS from volumes.yaml." if tag else ""),
             fontsize=8.5, color=ps.TEXT_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(RESULTS / f"fig3_volume_error{suffix}.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", choices=["common", "full"], default="common")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    main(a.eval_set, a.n_boot)
