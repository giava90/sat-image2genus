"""Run 2, stage 1: volume errors under realistic genus shares, five share estimators.

Reads run 1's saved predictions (results/predictions_*.parquet, read only),
regions.yaml and volumes.yaml. For every model, evaluation set and scenario:
  * build a region of 5,000 stands by resampling test patches of each genus to
    the target shares (stratified, with replacement), 1,000 replicates;
  * in every replicate also bootstrap the validation set, so ac and pac carry
    validation noise;
  * estimate genus shares with cc, ac, pcc, pac, em (src/quantify.py);
  * per-genus volume error = area error (growing stock cancels); total volume
    error uses the growing stock in volumes.yaml (PLACEHOLDER values).
Plus an oak sweep (true oak share 1-30 %, other genera in S1 proportions).

Writes results_v2/stage1_*.csv/.parquet/.json and fig_v2_rmse_by_estimator*.png,
fig_v2_oak_sweep.png.
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import plotstyle as ps
from common import GENERA, RESULTS, ROOT, SEED
from predio import load_predictions
from quantify import ESTIMATORS, all_estimates, cond_matrix, sample_region, soft_matrix, stand_counts
from v2common import RESULTS_V2

MODELS = ["m1_lgbm", "m2_cnn", "m3_resnet18"]
N_STANDS = 5000
K = len(GENERA)
PCOLS = [f"p_{g}" for g in GENERA]
TARGETS = GENERA + ["Total"]
OAK_SHARES = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]
EST_LABELS = {"cc": "classify & count", "ac": "adjusted count", "pcc": "prob. classify & count",
              "pac": "prob. adjusted count", "em": "EM prior shift"}


def load_cfg():
    v = yaml.safe_load((ROOT / "volumes.yaml").read_text())
    r = yaml.safe_load((ROOT / "regions.yaml").read_text())
    gs = np.array([float(v["growing_stock"][g]) for g in GENERA])
    return gs, bool(v.get("placeholder", True)), r


def train_prior(model: str, index: pd.DataFrame) -> np.ndarray:
    if model == "m3_resnet18":
        n = json.loads((RESULTS / "run_m3_resnet18.json").read_text())["n_train_per_genus"]
        c = np.array([n[g] for g in GENERA], float)
    else:
        c = index[index.split == "train"].genus.value_counts().reindex(GENERA).to_numpy(float)
    return c / c.sum()


def frames(model: str, index: pd.DataFrame, eval_set: str):
    p = load_predictions(model)
    if eval_set == "common":
        p = p[p.file.isin(set(index.loc[index.aerial_available, "file"]))]
    val, test = p[p.split == "val"], p[p.split == "test"]
    return ({"y": val.y_true.to_numpy(), "yp": val.y_pred.to_numpy(), "P": val[PCOLS].to_numpy(float)},
            {"y": test.y_true.to_numpy(), "yp": test.y_pred.to_numpy(), "P": test[PCOLS].to_numpy(float)})


def simulate(val, test, shares, prior, gs, n_rep, seed):
    """Returns errors array (n_rep, n_estimators, 5): relative error per genus and total volume."""
    rng = np.random.default_rng(seed)
    true = stand_counts(shares, N_STANDS) / N_STANDS
    v_true = (true * gs).sum()
    err = np.full((n_rep, len(ESTIMATORS), K + 1), np.nan)
    nv = len(val["y"])
    for r in range(n_rep):
        idx = sample_region(test["y"], shares, N_STANDS, rng)
        vi = rng.integers(0, nv, nv)
        cv = cond_matrix(val["y"][vi], val["yp"][vi], K)
        sv = soft_matrix(val["y"][vi], val["P"][vi], K)
        est = all_estimates(test["yp"][idx], test["P"][idx], K, cv, sv, prior)
        for e_i, e in enumerate(ESTIMATORS):
            x = est[e]
            with np.errstate(divide="ignore", invalid="ignore"):
                err[r, e_i, :K] = np.where(true > 0, (x - true) / true, np.nan)
            err[r, e_i, K] = (x * gs).sum() / v_true - 1
    return err


def summarise(err, keys: dict) -> list[dict]:
    rows = []
    for e_i, e in enumerate(ESTIMATORS):
        for t_i, t in enumerate(TARGETS):
            col = err[:, e_i, t_i]
            rows.append({**keys, "estimator": e, "target": t, "bias": col.mean(), "rmse": np.sqrt((col ** 2).mean()),
                         "sd": col.std(), "p2_5": np.percentile(col, 2.5), "p97_5": np.percentile(col, 97.5),
                         "median_abs": np.median(np.abs(col))})
    return rows


def main(n_rep: int) -> None:
    t0 = time.time()
    ps.apply()
    index = pd.read_parquet(RESULTS / "index.parquet")
    gs, placeholder, rcfg = load_cfg()
    common_counts = index[index.aerial_available].genus.value_counts().reindex(GENERA).to_numpy(float)
    full_counts = index.genus.value_counts().reindex(GENERA).to_numpy(float)
    scen_by_set = {}
    for es, cnt in (("common", common_counts), ("full", full_counts)):
        s = {"S1 dataset shares": cnt / cnt.sum(), "S2 equal shares": np.full(K, 1 / K)}
        for name, v in rcfg["scenarios"].items():
            s[f"{name}"] = np.asarray(v, float)
        scen_by_set[es] = s
    meta = {"n_rep": n_rep, "n_stands": N_STANDS, "placeholder_volumes": placeholder,
            "regions_placeholder": bool(rcfg.get("placeholder", True)), "growing_stock": dict(zip(GENERA, gs.tolist())),
            "scenarios": {es: {k: dict(zip(GENERA, np.round(v, 4).tolist())) for k, v in s.items()} for es, s in scen_by_set.items()},
            "train_prior": {}, "oak_shares": OAK_SHARES}
    rows, raw = [], []
    for es in ("common", "full"):
        models = MODELS if es == "common" else MODELS[:2]
        for s_i, (sname, shares) in enumerate(scen_by_set[es].items()):
            for m in models:
                val, test = frames(m, index, es)
                prior = train_prior(m, index)
                meta["train_prior"][m] = dict(zip(GENERA, np.round(prior, 4).tolist()))
                # same seed for every model within (set, scenario): identical region draws (paired)
                err = simulate(val, test, shares, prior, gs, n_rep, seed=SEED + 100 * s_i + (0 if es == "common" else 1000))
                keys = {"eval_set": es, "scenario": sname, "model": m}
                rows += summarise(err, keys)
                raw.append(pd.DataFrame(err.reshape(n_rep, -1), columns=[f"{e}|{t}" for e in ESTIMATORS for t in TARGETS]).assign(**keys, rep=np.arange(n_rep)))
                print(f"{es} {sname} {m} done ({time.time() - t0:.0f}s)", flush=True)
    summ = pd.DataFrame(rows)
    summ.to_csv(RESULTS_V2 / "stage1_volume_summary.csv", index=False, float_format="%.8f")
    pd.concat(raw).to_parquet(RESULTS_V2 / "stage1_volume_replicates.parquet", index=False)
    # oak sweep, common set
    s1 = scen_by_set["common"]["S1 dataset shares"]
    sweep_rows = []
    for o_i, oak in enumerate(OAK_SHARES):
        shares = np.append(s1[:3] / s1[:3].sum() * (1 - oak), oak)
        for m in MODELS:
            val, test = frames(m, index, "common")
            err = simulate(val, test, shares, train_prior(m, index), gs, n_rep, seed=SEED + 5000 + o_i)
            for e_i, e in enumerate(ESTIMATORS):
                col = err[:, e_i, 3]
                sweep_rows.append({"oak_share": oak, "model": m, "estimator": e, "bias": col.mean(),
                                   "rmse": np.sqrt((col ** 2).mean()), "p2_5": np.percentile(col, 2.5),
                                   "p97_5": np.percentile(col, 97.5)})
        print(f"oak sweep {oak} done ({time.time() - t0:.0f}s)", flush=True)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(RESULTS_V2 / "stage1_oak_sweep.csv", index=False, float_format="%.8f")
    meta["seconds"] = round(time.time() - t0, 1)
    (RESULTS_V2 / "stage1_meta.json").write_text(json.dumps(meta, indent=2))
    tag = " (placeholder volumes, placeholder regions)" if placeholder else " (placeholder regions)"
    for es in ("common", "full"):
        fig_rmse(summ[summ.eval_set == es], es, tag)
    fig_sweep(sweep, tag)
    print(f"stage 1 done in {time.time() - t0:.0f}s")


def fig_rmse(summ: pd.DataFrame, es: str, tag: str) -> None:
    scen = list(dict.fromkeys(summ.scenario))
    models = [m for m in MODELS if m in set(summ.model)]
    fig, axes = plt.subplots(len(scen), K, figsize=(13, 2.5 * len(scen) + 1.2), sharex=True)
    x = np.arange(len(ESTIMATORS))
    for r, sname in enumerate(scen):
        for c, g in enumerate(GENERA):
            ax = axes[r, c]
            for j, m in enumerate(models):
                d = summ[(summ.scenario == sname) & (summ.model == m) & (summ.target == g)].set_index("estimator").loc[ESTIMATORS]
                ax.plot(x + (j - (len(models) - 1) / 2) * 0.18, d.rmse * 100, "o", ms=6, color=ps.MODEL_COLORS[m],
                        label=ps.MODEL_LABELS[m] if (r == 0 and c == 0) else None)
            ax.set_ylim(bottom=0)
            if r == 0:
                ax.set_title(g, fontsize=10.5)
            if c == 0:
                share = summ[summ.scenario == sname].scenario.iloc[0]
                ax.set_ylabel(f"{sname}\nRMSE, % of true", fontsize=9)
            ax.set_xticks(x, ["CC", "AC", "PCC", "PAC", "EM"])
    fig.legend(loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Per-genus volume RMSE by share estimator, {es} test set{tag}\n"
                 f"5,000-stand regions, 1,000 replicates with test and validation resampling", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(RESULTS_V2 / f"fig_v2_rmse_by_estimator_{es}.png")
    plt.close(fig)


def fig_sweep(sweep: pd.DataFrame, tag: str) -> None:
    fig, axes = plt.subplots(1, len(ESTIMATORS), figsize=(14, 4.2), sharey=True)
    for ax, e in zip(axes, ESTIMATORS):
        ax.axhline(0, color=ps.TEXT_2, lw=1)
        for m in MODELS:
            d = sweep[(sweep.model == m) & (sweep.estimator == e)]
            c = ps.MODEL_COLORS[m]
            ax.fill_between(d.oak_share * 100, d.p2_5 * 100, d.p97_5 * 100, color=c, alpha=0.15, lw=0)
            ax.plot(d.oak_share * 100, d.bias * 100, color=c, lw=2, label=ps.MODEL_LABELS[m])
        ax.set_title(EST_LABELS[e], fontsize=10.5)
        ax.set_xlabel("true oak share, %")
    axes[0].set_ylabel("oak volume error, % of true")
    axes[0].set_ylim(-60, 150)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Oak volume error against true oak share{tag}\nmean error (line) and 95% interval (band); "
                 f"other genera in S1 proportions; common test set; y axis cut at +150% (CC and PCC exceed it below about 5% oak, see stage1_oak_sweep.csv)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(RESULTS_V2 / "fig_v2_oak_sweep.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=1000)
    main(ap.parse_args().n_rep)
