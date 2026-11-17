"""Run the whole pipeline after download: tests, prepare, three models, evaluation, volume step.

Usage: python src/run_all.py [--skip-m3]
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent


def run(args: list[str], cwd: Path = SRC) -> None:
    t0 = time.time()
    print(f"\n=== {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=cwd, check=True)
    print(f"=== done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    run(["-m", "pytest", "-q", "tests"], cwd=ROOT)
    run(["prepare.py"])
    run(["train_m1_lgbm.py"])
    run(["train_m2_cnn.py"])
    if "--skip-m3" not in sys.argv:
        run(["train_m3_resnet.py"])
    run(["evaluate.py"])
    run(["volume_step.py"])
    run(["volume_step.py", "--eval-set", "full"])
    run(["spatial_check.py"])
    run(["run_facts.py"])
    run(["make_tables.py"])


if __name__ == "__main__":
    main()
