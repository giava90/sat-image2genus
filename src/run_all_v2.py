"""Run 2 end to end (needs run 1's data/cache/s2_200m.npy and results/ in place).

python src/run_all_v2.py
Stage order: tests, 60 m check, blocked splits, stage 1 volume, stage 2 training, kNN,
stage 3 training (queue 2 also runs stage 2 evaluation), stage 3 evaluation, stage 4, tables.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS = [
    ["-m", "pytest", "-q", "tests"],
    ["src/v2_check_60m.py"],
    ["src/v2_splits.py", "--seeds", "0", "1"],
    ["src/v2_stage1_volume.py"],
    ["src/v2_train_lgbm.py", "--features", "summary200", "--name", "lgbm_summary200_repro"],
    ["src/v2_train_lgbm.py", "--features", "summary60", "--name", "lgbm_summary60"],
    ["src/v2_train_lgbm.py", "--features", "centre_ring", "--name", "lgbm_centre_ring"],
    ["src/v2_train_lgbm.py", "--features", "raw60", "--name", "lgbm_raw60"],
    ["src/v2_train_cnn.py", "--variant", "centre", "--name", "cnn_centre6"],
    ["src/v2_train_cnn.py", "--variant", "masked", "--name", "cnn_masked_centre"],
    ["src/v2_train_lgbm.py", "--features", "raw200", "--name", "lgbm_raw200"],
    ["src/v2_knn.py", "--split", "orig"],
    ["src/v2_knn.py", "--split", "seed0"],
    ["src/v2_knn.py", "--split", "seed1"],
    ["src/v2_train_lgbm.py", "--features", "summary200", "--split", "seed0", "--name", "lgbm_summary200", "--sub", "stage3_seed0"],
    ["src/v2_train_lgbm.py", "--features", "xy", "--split", "orig", "--name", "lgbm_xy", "--sub", "stage3_orig"],
    ["src/v2_train_lgbm.py", "--features", "xy", "--split", "seed0", "--name", "lgbm_xy", "--sub", "stage3_seed0"],
    ["src/v2_train_cnn.py", "--variant", "full", "--split", "seed0", "--name", "cnn_full", "--sub", "stage3_seed0"],
    ["src/v2_queue2.py"],
    ["src/v2_stage3_eval.py"],
    ["src/v2_stage4_purity.py"],
    ["src/v2_fig_gap.py"],
    ["src/v2_run_facts.py"],
    ["src/v2_make_tables.py"],
]

if __name__ == "__main__":
    for s in STEPS:
        print("+", " ".join(s), flush=True)
        subprocess.run([PY, *s], check=True, cwd=ROOT)
