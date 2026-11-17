"""Score all models on the same test patches; write metrics and figures 1 and 2.

Evaluation sets (DECISIONS D7):
  common - test patches of species with aerial data; all three models.
  full   - all four-genus test patches; models 1 and 2 only.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import plotstyle as ps
from common import BINARY_NAMES, GENERA, GENUS_TO_BINARY, RESULTS, SEED
from metrics import collapse, confusion_matrix, summary
from predio import load_predictions

MODELS = ["m1_lgbm", "m2_cnn", "m3_resnet18"]
N_BOOT = 1000


def available_models() -> list[str]:
    return [m for m in MODELS if (RESULTS / f"predictions_{m}.parquet").exists()]


def eval_frame(model: str, split: str, eval_set: str, index: pd.DataFrame) -> pd.DataFrame:
    p = load_predictions(model)
    p = p[p.split == split]
    if eval_set == "common":
        keep = set(index.loc[index.aerial_available, "file"])
        p = p[p.file.isin(keep)]
    return p.sort_values("file").reset_index(drop=True)


def boot_ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> None:
    ps.apply()
    index = pd.read_parquet(RESULTS / "index.parquet")
    models = available_models()
    out = {"n_boot": N_BOOT, "classes": GENERA, "sets": {}}
    rng = np.random.default_rng(SEED)
    for eval_set in ("common", "full"):
        ms = models if eval_set == "common" else [m for m in models if m != "m3_resnet18"]
        frames = {m: eval_frame(m, "test", eval_set, index) for m in ms}
        if not frames:
            continue
        ref = frames[ms[0]].file.tolist()
        for m in ms:
            assert frames[m].file.tolist() == ref, f"{m}: test patches differ from {ms[0]}"
        n = len(ref)
        y = frames[ms[0]].y_true.to_numpy()
        boots = rng.integers(0, n, (N_BOOT, n))  # same resamples for every model (paired)
        res = {"n": n, "models": {}, "paired_accuracy_diff": {}}
        acc_b = {}
        for m in ms:
            yp = frames[m].y_pred.to_numpy()
            s = summary(y, yp, len(GENERA))
            yb, pb = collapse(y, GENUS_TO_BINARY), collapse(yp, GENUS_TO_BINARY)
            sb = summary(yb, pb, 2)
            correct = (yp == y).astype(float)
            correct_b = (pb == yb).astype(float)
            acc_b[m] = correct[boots].mean(1)
            s["accuracy_ci95"] = boot_ci(acc_b[m])
            sb["accuracy_ci95"] = boot_ci(correct_b[boots].mean(1))
            s["binary"] = sb
            res["models"][m] = s
        for i, a in enumerate(ms):
            for b in ms[i + 1 :]:
                d = acc_b[b] - acc_b[a]
                res["paired_accuracy_diff"][f"{b} - {a}"] = {
                    "point": res["models"][b]["accuracy"] - res["models"][a]["accuracy"],
                    "ci95": boot_ci(d),
                    "share_of_resamples_b_better": float((d > 0).mean()),
                }
        out["sets"][eval_set] = res
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))

    # flat table
    rows = []
    for eval_set, res in out["sets"].items():
        for m, s in res["models"].items():
            r = {"eval_set": eval_set, "model": m, "n": s["n"], "accuracy": s["accuracy"],
                 "acc_ci_lo": s["accuracy_ci95"][0], "acc_ci_hi": s["accuracy_ci95"][1],
                 "weighted_precision": s["weighted_precision"], "macro_f1": s["macro_f1"],
                 "binary_accuracy": s["binary"]["accuracy"],
                 "bin_ci_lo": s["binary"]["accuracy_ci95"][0], "bin_ci_hi": s["binary"]["accuracy_ci95"][1]}
            for i, g in enumerate(GENERA):
                r[f"precision_{g}"] = s["precision"][i]
                r[f"recall_{g}"] = s["recall"][i]
            for i, g in enumerate(BINARY_NAMES):
                r[f"precision_{g}"] = s["binary"]["precision"][i]
                r[f"recall_{g}"] = s["binary"]["recall"][i]
            rows.append(r)
    pd.DataFrame(rows).to_csv(RESULTS / "metrics_table.csv", index=False, float_format="%.8f")

    common = out["sets"]["common"]
    fig1(common, models)
    fig2(common, models)
    print(pd.DataFrame(rows)[["eval_set", "model", "n", "accuracy", "acc_ci_lo", "acc_ci_hi", "weighted_precision", "binary_accuracy"]].round(4).to_string())
    print(json.dumps({k: v["paired_accuracy_diff"] for k, v in out["sets"].items()}, indent=1))


def fig1(res: dict, models: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), gridspec_kw={"width_ratios": [1.0, 1.35, 1.35]})
    k = len(models)
    w = 0.8 / k
    # (a) overall accuracy, 4 genera and conifer/broadleaf, with bootstrap 95% CI
    ax = axes[0]
    for j, m in enumerate(models):
        s = res["models"][m]
        vals = [s["accuracy"], s["binary"]["accuracy"]]
        cis = [s["accuracy_ci95"], s["binary"]["accuracy_ci95"]]
        xs = np.arange(2) + (j - (k - 1) / 2) * w
        ax.bar(xs, vals, width=w * 0.92, color=ps.MODEL_COLORS[m], label=ps.MODEL_LABELS[m], zorder=2)
        err = np.array([[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]])
        ax.errorbar(xs, vals, yerr=err, fmt="none", ecolor=ps.TEXT_2, elinewidth=1, capsize=2, zorder=3)
        for x, v, c in zip(xs, vals, cis):
            ax.text(x, c[1] + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=ps.TEXT)
    ax.set_xticks(range(2), ["4 genera", "conifer vs\nbroadleaf"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy (95% bootstrap CI)")
    ax.set_title("(a) Overall accuracy", loc="left")
    # (b, c) per-genus precision and recall
    for ax, key, title in ((axes[1], "precision", "(b) Precision per genus"), (axes[2], "recall", "(c) Recall per genus")):
        for j, m in enumerate(models):
            vals = res["models"][m][key]
            xs = np.arange(len(GENERA)) + (j - (k - 1) / 2) * w
            ax.bar(xs, vals, width=w * 0.92, color=ps.MODEL_COLORS[m], zorder=2)
        ax.set_xticks(range(len(GENERA)), [f"{g}" for g in GENERA])
        ax.set_ylim(0, 1.0)
        ax.set_title(title, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Test accuracy on the common test set (n = {res['n']}, same patches for all models)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(RESULTS / "fig1_accuracy.png")
    plt.close(fig)


def fig2(res: dict, models: list[str]) -> None:
    cmap = LinearSegmentedColormap.from_list("seqblue", ps.SEQ_BLUE)
    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 4.8))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        cm = np.array(res["models"][m]["confusion"], float)
        rn = cm / cm.sum(1, keepdims=True)
        ax.imshow(rn, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(GENERA)):
            for j in range(len(GENERA)):
                col = "#ffffff" if rn[i, j] > 0.55 else ps.TEXT
                ax.text(j, i, f"{rn[i, j]:.0%}\n{int(cm[i, j])}", ha="center", va="center", fontsize=8.5, color=col)
        ax.set_xticks(range(len(GENERA)), GENERA)
        ax.set_yticks(range(len(GENERA)), GENERA)
        ax.set_xlabel("predicted genus")
        ax.set_ylabel("true genus")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{ps.MODEL_LABELS[m]}\naccuracy {res['models'][m]['accuracy']:.3f}", loc="left", fontsize=10)
    fig.suptitle("Confusion matrices, common test set (row %: share of each true genus; count below)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(RESULTS / "fig2_confusion.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
