"""Run 2 facts cited in RESULTS_V2.md: training time per run and in total. Writes results_v2/run2_facts.json."""
from __future__ import annotations

import json

from v2common import RESULTS_V2


def main() -> None:
    runs = {}
    for p in sorted(RESULTS_V2.rglob("run_*.json")):
        r = json.loads(p.read_text())
        runs[str(p.relative_to(RESULTS_V2)).replace("\\", "/")] = round(r.get("train_seconds", 0) + r.get("feature_seconds", 0), 1)
    tot = sum(runs.values())
    out = {"training_seconds_per_run": runs, "training_seconds_total": round(tot, 1), "training_minutes_total": round(tot / 60, 1)}
    (RESULTS_V2 / "run2_facts.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
