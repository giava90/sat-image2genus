"""Render the saved results as Markdown tables (results/tables.md) for RESULTS.md.

Every number comes from files under results/; nothing is typed by hand.
"""
from __future__ import annotations

import json

import pandas as pd

from common import GENERA, RESULTS
from plotstyle import MODEL_LABELS

MODELS = ["m1_lgbm", "m2_cnn", "m3_resnet18"]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def spct(x, d=1):
    return f"{100 * x:+.{d}f}%"


def main() -> None:
    met = json.loads((RESULTS / "metrics.json").read_text())
    out = []
    # A. overall
    out.append("### A. Overall test scores\n")
    out.append("| test set | model | n | accuracy (95% CI) | weighted precision | macro F1 | conifer vs broadleaf accuracy (95% CI) |")
    out.append("|---|---|---|---|---|---|---|")
    for es, res in met["sets"].items():
        for m, s in res["models"].items():
            b = s["binary"]
            out.append(
                f"| {es} | {MODEL_LABELS[m]} | {s['n']} | {pct(s['accuracy'])} ({pct(s['accuracy_ci95'][0])} to {pct(s['accuracy_ci95'][1])}) "
                f"| {pct(s['weighted_precision'])} | {s['macro_f1']:.3f} | {pct(b['accuracy'])} ({pct(b['accuracy_ci95'][0])} to {pct(b['accuracy_ci95'][1])}) |"
            )
    # B. paired differences
    out.append("\n### B. Paired accuracy differences (same test patches, 1,000 paired bootstrap resamples)\n")
    out.append("| test set | comparison | difference in accuracy | 95% CI | share of resamples where the first model is better |")
    out.append("|---|---|---|---|---|")
    for es, res in met["sets"].items():
        for k, d in res["paired_accuracy_diff"].items():
            a, b = k.split(" - ")
            out.append(f"| {es} | {MODEL_LABELS[a]} minus {MODEL_LABELS[b]} | {spct(d['point'])} | {spct(d['ci95'][0])} to {spct(d['ci95'][1])} | {d['share_of_resamples_b_better']:.3f} |")
    # C. per-genus precision / recall
    out.append("\n### C. Per-genus precision and recall, common test set\n")
    out.append("| model | " + " | ".join(f"{g} P | {g} R" for g in GENERA) + " | conifer P | conifer R | broadleaf P | broadleaf R |")
    out.append("|---|" + "---|" * (2 * len(GENERA) + 4))
    for m, s in met["sets"]["common"]["models"].items():
        cells = []
        for i in range(len(GENERA)):
            cells += [f"{s['precision'][i]:.3f}", f"{s['recall'][i]:.3f}"]
        b = s["binary"]
        cells += [f"{b['precision'][0]:.3f}", f"{b['recall'][0]:.3f}", f"{b['precision'][1]:.3f}", f"{b['recall'][1]:.3f}"]
        out.append(f"| {MODEL_LABELS[m]} | " + " | ".join(cells) + " |")
    # D. training runs
    out.append("\n### D. Training runs (device, time, selected epoch)\n")
    out.append("| model | device | training patches | training time | epochs run | selected epoch / rounds | best validation accuracy |")
    out.append("|---|---|---|---|---|---|---|")
    for m in MODELS:
        p = RESULTS / f"run_{m}.json"
        if not p.exists():
            out.append(f"| {MODEL_LABELS[m]} | not run | | | | | |")
            continue
        r = json.loads(p.read_text())
        sel = r.get("best_epoch", r.get("best_iteration"))
        out.append(f"| {MODEL_LABELS[m]} | {r.get('device')} | {r.get('n_train')} | {r.get('train_seconds', 0) / 60:.1f} min | {r.get('epochs_run', '-')} | {sel} | {pct(r['best_val_acc']) if 'best_val_acc' in r else '-'} |")
    # E/F. volume errors
    for suffix, label in (("", "common test set, all three models"), ("_full", "full test set, models 1 and 2")):
        p = RESULTS / f"volume_summary{suffix}.csv"
        if not p.exists():
            continue
        v = pd.read_csv(p)
        meta = json.loads((RESULTS / f"volume_meta{suffix}.json").read_text())
        tag = " (placeholder volumes)" if meta["placeholder_volumes"] else ""
        out.append(f"\n### E{suffix}. Volume error in % of true volume, {label}{tag}\n")
        out.append("Point estimate on the test set, with the 95% bootstrap interval in brackets. 'count' = area by counting predictions; 'adjusted' = after confusion-matrix adjustment. "
                   "The validation matrix used for the adjustment is held fixed, so adjusted intervals leave out validation-set noise and are too narrow. "
                   "The last column averages |error| over the four genera. Bootstrapping an absolute value biases it upward, so its interval can exclude a point estimate near zero; read it as the spread of the error size, not as a confidence interval.\n")
        out.append("| scenario | model | estimator | " + " | ".join(GENERA) + " | total volume | mean abs. genus error |")
        out.append("|---|---|---|" + "---|" * (len(GENERA) + 2))
        for sc in v.scenario.unique():
            for m in [x for x in MODELS if x in set(v.model)]:
                for est in ("count", "adjusted"):
                    sub = v[(v.scenario == sc) & (v.model == m) & (v.estimator == est)].set_index("target")
                    cells = [f"{spct(sub.loc[t, 'point'])} [{spct(sub.loc[t, 'p2_5'])}, {spct(sub.loc[t, 'p97_5'])}]" for t in GENERA + ["Total"]]
                    mae = sub.loc["MeanAbsGenus"]
                    cells.append(f"{pct(mae['point'])} [{pct(mae['p2_5'])}, {pct(mae['p97_5'])}]")
                    out.append(f"| {sc} | {MODEL_LABELS[m]} | {est} | " + " | ".join(cells) + " |")
        # headline sentences
        out.append(f"\n**Spruce sentences ({label}, scenario S1, counting predictions{tag}):**\n")
        for m in [x for x in MODELS if x in set(v.model)]:
            acc = meta["models"][m]["test_accuracy"]
            r = v[(v.scenario == "S1 dataset shares") & (v.model == m) & (v.estimator == "count") & (v.target == "Picea")].iloc[0]
            out.append(f"- A classifier with {pct(acc)} accuracy ({MODEL_LABELS[m]}) misstates spruce volume by {spct(r['point'])} (95% bootstrap interval {spct(r['p2_5'])} to {spct(r['p97_5'])}){tag}.")
    # G. spatial proximity diagnostic
    p = RESULTS / "spatial_check.csv"
    if p.exists():
        sc = pd.read_csv(p)
        out.append("\n### G. Accuracy by distance from a test patch to the nearest training patch (Chebyshev distance between patch centres)\n")
        out.append("200 m Sentinel-2 windows overlap when the distance is under 200 m.\n")
        for es in sc.eval_set.unique():
            s = sc[sc.eval_set == es]
            bands = list(dict.fromkeys(s.band))
            ms = list(dict.fromkeys(s.model))
            out.append(f"\n{es} test set:\n")
            out.append("| distance band | n | " + " | ".join(MODEL_LABELS[m] for m in ms) + " |")
            out.append("|---|---|" + "---|" * len(ms))
            for b in bands:
                row = s[s.band == b]
                n = int(row.n.iloc[0])
                accs = [row[row.model == m].accuracy.iloc[0] for m in ms]
                out.append(f"| {b} | {n} | " + " | ".join("-" if pd.isna(a) else pct(a) for a in accs) + " |")
    (RESULTS / "tables.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
