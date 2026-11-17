"""Run 2 training queue, part 2 (after part 1 in logs/run2_train_queue.sh).

1. Stage 2 evaluation, which picks the best LightGBM variant by validation accuracy.
2. That variant on blocked split seed0.
3. Blocked split seed1 (LightGBM run 1 features, best variant, LightGBM on x/y, CNN)
   only if the training time used so far leaves room for it in the 45-minute budget.
Training time is the sum of train_seconds (+ feature_seconds) in results_v2/**/run_*.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
R2 = ROOT / "results_v2"
BUDGET_S = 45 * 60
CNN_SEED1_EST_S = 12 * 60  # measured CNN epoch time x 40 epochs on the seed0 blocked split, rounded up


def run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run([PY, *args], check=True, cwd=ROOT)


def used_s() -> float:
    t = 0.0
    for p in R2.rglob("run_*.json"):
        r = json.loads(p.read_text())
        t += r.get("train_seconds", 0) + r.get("feature_seconds", 0)
    return t


def main() -> None:
    run("src/v2_stage2_eval.py")
    best = json.loads((R2 / "stage2_best_variant.json").read_text())["best_lgbm_variant_by_val_accuracy"]
    feat = best.replace("lgbm_", "")
    run("src/v2_train_lgbm.py", "--features", feat, "--split", "seed0", "--name", best, "--sub", "stage3_seed0")
    u = used_s()
    print(f"training seconds used so far: {u:.0f}", flush=True)
    if u + CNN_SEED1_EST_S + 120 <= BUDGET_S:
        run("src/v2_train_lgbm.py", "--features", "summary200", "--split", "seed1", "--name", "lgbm_summary200", "--sub", "stage3_seed1")
        run("src/v2_train_lgbm.py", "--features", feat, "--split", "seed1", "--name", best, "--sub", "stage3_seed1")
        run("src/v2_train_lgbm.py", "--features", "xy", "--split", "seed1", "--name", "lgbm_xy", "--sub", "stage3_seed1")
        run("src/v2_train_cnn.py", "--variant", "full", "--split", "seed1", "--name", "cnn_full", "--sub", "stage3_seed1")
    else:
        print("SKIP blocked split seed1: not enough training budget left", flush=True)
    print(f"QUEUE2_DONE training seconds used: {used_s():.0f}", flush=True)


if __name__ == "__main__":
    main()
