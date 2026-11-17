"""Render run 2 results as Markdown tables (results_v2/tables_v2.md). Every number is read from results_v2/."""
from __future__ import annotations

import json

import pandas as pd

from common import GENERA
from v2common import RESULTS_V2

EST = ["cc", "ac", "pcc", "pac", "em"]
EST_NAME = {"cc": "CC", "ac": "AC", "pcc": "PCC", "pac": "PAC", "em": "EM"}
MOD = {"m1_lgbm": "M1 LightGBM", "m2_cnn": "M2 CNN", "m3_resnet18": "M3 ResNet-18"}


def pct(x, d=1):
    return "-" if pd.isna(x) else f"{100 * x:.{d}f}%"


def spct(x, d=1):
    return "-" if pd.isna(x) else f"{100 * x:+.{d}f}"


def acc_cell(r, kind):
    return f"{pct(r.accuracy)} ({pct(r[f'{kind}_ci_lo'])} to {pct(r[f'{kind}_ci_hi'])})"


def diff_cell(r, kind):
    return f"{spct(r['diff'])} ({spct(r[f'{kind}_ci_lo'])} to {spct(r[f'{kind}_ci_hi'])})"


def main() -> None:
    out = []
    # ---------------- gap table
    s2p = pd.read_csv(RESULTS_V2 / "stage2_paired.csv")
    s3p = pd.read_csv(RESULTS_V2 / "stage3_paired.csv")
    best = json.loads((RESULTS_V2 / "stage2_best_variant.json").read_text())["best_lgbm_variant_by_val_accuracy"]
    out.append("### T1. The CNN's lead over LightGBM, three ways (accuracy points; 95% intervals: patch bootstrap / 2 km block bootstrap)\n")
    out.append(f"Best fair baseline = `{best}`, chosen among stage 2 variants a-d by validation accuracy.\n")
    out.append("| comparison | test set | n test | CNN minus LightGBM | patch-bootstrap 95% CI | block-bootstrap 95% CI |")
    out.append("|---|---|---|---|---|---|")

    def gap_row(label, es, r):
        out.append(f"| {label} | {es} | {int(r.n)} | {spct(r['diff'])} | {spct(r.patch_ci_lo)} to {spct(r.patch_ci_hi)} | {spct(r.block_ci_lo)} to {spct(r.block_ci_hi)} |")

    for es in ("full", "common"):
        r = s3p[(s3p.split == "original") & (s3p.eval_set == es) & (s3p.comparison == "M2 - M1")]
        if len(r):
            gap_row("original split, run 1 LightGBM (200 m summary stats)", es, r.iloc[0])
        r = s2p[(s2p.eval_set == es) & (s2p.comparison == f"M2 - {best}")]
        if len(r):
            gap_row("original split, best fair LightGBM baseline", es, r.iloc[0])
        for sp in sorted(s3p.split.unique()):
            if not sp.startswith("blocked"):
                continue
            for comp, lab in (("M2 - M1", "run 1 LightGBM features"), ("M2 - best_lgbm", "best fair LightGBM baseline")):
                r = s3p[(s3p.split == sp) & (s3p.eval_set == es) & (s3p.comparison == comp)]
                if len(r):
                    gap_row(f"{sp} split (all retrained), {lab}", es, r.iloc[0])
    # ---------------- stage 2
    a2 = pd.read_csv(RESULTS_V2 / "stage2_accuracy.csv")
    out.append("\n### T2. Stage 2: is the LightGBM baseline fair? Test accuracy on the original split\n")
    out.append("Differences against run 1's M1 and M2 are paired (same test patches, same 1,000 patch resamples).\n")
    for es in ("common", "full"):
        out.append(f"\n{es} test set:\n")
        out.append("| model | description | validation acc. | test accuracy (patch 95% CI) | minus M1 (95% CI) | minus M2 (95% CI) |")
        out.append("|---|---|---|---|---|---|")
        for _, r in a2[a2.eval_set == es].iterrows():
            d1 = s2p[(s2p.eval_set == es) & (s2p.comparison == f"{r.model} - M1")]
            d2 = s2p[(s2p.eval_set == es) & (s2p.comparison == f"{r.model} - M2")]
            out.append(f"| {r.model} | {r.description} | {pct(r.val_accuracy)} | {acc_cell(r, 'patch')} | "
                       f"{diff_cell(d1.iloc[0], 'patch') if len(d1) else '-'} | {diff_cell(d2.iloc[0], 'patch') if len(d2) else '-'} |")
    # ---------------- stage 3
    a3 = pd.read_csv(RESULTS_V2 / "stage3_accuracy.csv")
    out.append("\n### T3. Stage 3: accuracy on the original split and on blocked splits\n")
    out.append("Blocked splits regroup all four-genus patches into 2 km blocks (20% of patches test, 10% validation) and drop training patches within 200 m of a held-out patch. Model 3 was not retrained on blocked splits.\n")
    for es in ("full", "common"):
        out.append(f"\n{es} test set:\n")
        out.append("| model | split | n test | blocks | accuracy | patch-bootstrap 95% CI | block-bootstrap 95% CI |")
        out.append("|---|---|---|---|---|---|---|")
        for _, r in a3[a3.eval_set == es].iterrows():
            out.append(f"| {r.description} | {r.split} | {int(r.n)} | {int(r.n_blocks)} | {pct(r.accuracy)} | "
                       f"{pct(r.patch_ci_lo)} to {pct(r.patch_ci_hi)} | {pct(r.block_ci_lo)} to {pct(r.block_ci_hi)} |")
    out.append("\n### T4. Stage 3: paired differences, patch vs block bootstrap\n")
    out.append("| split | test set | comparison | difference (points) | patch-bootstrap 95% CI | block-bootstrap 95% CI |")
    out.append("|---|---|---|---|---|---|")
    for _, r in s3p.iterrows():
        out.append(f"| {r.split} | {r.eval_set} | {r.comparison} | {spct(r['diff'])} | {spct(r.patch_ci_lo)} to {spct(r.patch_ci_hi)} | {spct(r.block_ci_lo)} to {spct(r.block_ci_hi)} |")
    # ---------------- stage 1
    v = pd.read_csv(RESULTS_V2 / "stage1_volume_summary.csv")
    meta = json.loads((RESULTS_V2 / "stage1_meta.json").read_text())
    tag = " (placeholder volumes, placeholder regions)" if meta["placeholder_volumes"] else " (placeholder regions)"
    for es in ("common", "full"):
        sc = meta["scenarios"][es]
        out.append(f"\n### T5-{es}. Stage 1: per-genus volume error by estimator, {es} test set{tag}\n")
        out.append(f"{meta['n_rep']} replicates of {meta['n_stands']:,} stands; test patches and validation set resampled in every replicate. "
                   "Cell = bias / RMSE, in % of the true genus volume. Scenario shares (spruce/pine/beech/oak): "
                   + "; ".join(f"{k} " + "/".join(f"{100 * x:.0f}" for x in s.values()) for k, s in sc.items()) + ".\n")
        out.append("| scenario | model | estimator | " + " | ".join(GENERA) + " | total volume | mean RMSE over genera |")
        out.append("|---|---|---|" + "---|" * (len(GENERA) + 2))
        vv = v[v.eval_set == es]
        for s in sc:
            for m in [x for x in MOD if x in set(vv.model)]:
                for e in EST:
                    sub = vv[(vv.scenario == s) & (vv.model == m) & (vv.estimator == e)].set_index("target")
                    cells = [f"{spct(sub.loc[g, 'bias'])} / {pct(sub.loc[g, 'rmse'])}" for g in GENERA + ["Total"]]
                    out.append(f"| {s} | {MOD[m]} | {EST_NAME[e]} | " + " | ".join(cells) + f" | {pct(sub.loc[GENERA, 'rmse'].mean())} |")
    # best estimator per scenario/model by mean RMSE
    out.append(f"\n### T6. Stage 1: estimator with the lowest mean per-genus RMSE, common test set{tag}\n")
    out.append("| scenario | " + " | ".join(MOD.values()) + " |")
    out.append("|---|---|---|---|")
    vc = v[(v.eval_set == "common") & (v.target.isin(GENERA))].groupby(["scenario", "model", "estimator"]).rmse.mean().reset_index()
    for s in meta["scenarios"]["common"]:
        cells = []
        for m in MOD:
            d = vc[(vc.scenario == s) & (vc.model == m)].sort_values("rmse")
            cc = d[d.estimator == "cc"].rmse.iloc[0]
            cells.append(f"{EST_NAME[d.estimator.iloc[0]]} {pct(d.rmse.iloc[0])} (CC {pct(cc)})")
        out.append(f"| {s} | " + " | ".join(cells) + " |")
    # oak sweep
    sw = pd.read_csv(RESULTS_V2 / "stage1_oak_sweep.csv")
    out.append(f"\n### T7. Stage 1: oak volume error against true oak share, common test set{tag}\n")
    out.append("Other genera in S1 (common-set) proportions. Cell = bias / RMSE in % of true oak volume.\n")
    out.append("| true oak share | model | " + " | ".join(EST_NAME[e] for e in EST) + " |")
    out.append("|---|---|" + "---|" * len(EST))
    for o in sorted(sw.oak_share.unique()):
        for m in MOD:
            d = sw[(sw.oak_share == o) & (sw.model == m)].set_index("estimator")
            out.append(f"| {100 * o:g}% | {MOD[m]} | " + " | ".join(f"{spct(d.loc[e, 'bias'])} / {pct(d.loc[e, 'rmse'])}" for e in EST) + " |")
    # ---------------- stage 4
    p4 = pd.read_csv(RESULTS_V2 / "stage4_purity.csv")
    out.append("\n### T8. Stage 4: test accuracy by label purity (run 1 models; Wilson 95% intervals)\n")
    out.append("Purity = area share of the labelled genus in the archive's multi-label JSON.\n")
    out.append("| test set | grouping | level | n | " + " | ".join(MOD.values()) + " |")
    out.append("|---|---|---|---|---|---|---|")
    for es in ("common", "full"):
        q = p4[p4.eval_set == es]
        for (gr, lev), g in q.groupby(["grouping", "level"], sort=False):
            cells = []
            for m in MOD:
                r = g[g.model == m]
                cells.append("-" if not len(r) else (f"{pct(r.accuracy.iloc[0])}" + ("" if pd.isna(r.wilson_lo.iloc[0]) else f" ({pct(r.wilson_lo.iloc[0])} to {pct(r.wilson_hi.iloc[0])})")))
            out.append(f"| {es} | {gr} | {lev} | {int(g.n.iloc[0])} | " + " | ".join(cells) + " |")
    (RESULTS_V2 / "tables_v2.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
