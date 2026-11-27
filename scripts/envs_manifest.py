"""Write the release manifest of the data-out tasks: what a benchmark version freezes.

    uv run python scripts/envs_manifest.py            # writes data/envs_hard/manifest.json
    uv run python scripts/envs_manifest.py --check    # exit 1 if the tasks differ from it

A release (docs/release-policy.md) is the benchmark version in envs/config.json plus, per task,
the sha256 of every file that decides a score (TASK.md, evaluate.py, eval/*.py, checks/
expect.json) and the rubric version. Two runs with the same manifest graded the same task; a
report from a run whose manifest differs is not comparable and is not mixed into a leaderboard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from qec_drift.agent import envs_grade as eg
from qec_drift.agent.envs_kit import ENVS, TASKS
from qec_drift.scoring.envs_rubric import load_rubric

MANIFEST = eg.OUT / "manifest.json"


def task_files(task: str) -> list[Path]:
    root = ENVS / task
    files = [root / "TASK.md", root / "evaluate.py", root / "checks" / "expect.json"]
    files += sorted((root / "eval").glob("*.py"))
    return [f for f in files if f.exists()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest() -> dict:
    tasks = {}
    for task in sorted(TASKS):
        root = ENVS / task
        tasks[task] = dict(
            rubric=load_rubric(task).VERSION,
            files={str(f.relative_to(root)): sha256(f) for f in task_files(task)},
        )
    return dict(version=eg.CONFIG["version"], config=sha256(ENVS / "config.json"), tasks=tasks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with the stored manifest")
    a = ap.parse_args()
    m = manifest()
    if a.check:
        old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        if old != m:
            print(f"tasks differ from {MANIFEST}; a change to a scored file is a new version")
            return 1
        print(f"{m['version']}: tasks match the manifest")
        return 0
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
