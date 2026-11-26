"""Build the data-out leaderboard: one page with every task, every model and every baseline.

    uv run python scripts/envs_leaderboard.py

Reads data/envs_hard/references.json (trivial and reference), selection.json (each model's
pass rate and its 95% interval, scripts/select_tasks.py) and acceptance.json (the gates,
scripts/accept_task.py), and grades each task's ceilings (envs/<task>/ceilings/) from the
cached reports. Writes docs/leaderboard/index.html, a static page with the data inline, so it
opens from disk or GitHub Pages with no server.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from qec_drift import paths
from qec_drift.agent import envs_grade as eg
from qec_drift.agent.envs_kit import ENVS, TASKS

DOC = paths.REPO_ROOT / "docs" / "leaderboard" / "index.html"
TEMPLATE = Path(__file__).with_name("envs_leaderboard_template.html")


def _load(name: str) -> dict:
    f = eg.OUT / name
    return json.loads(f.read_text()) if f.exists() else {}


def ceilings(task: str) -> dict[str, dict]:
    out = {}
    for p in sorted((ENVS / task / "ceilings").glob("*.py")):
        if not p.name.startswith("_"):
            out[p.stem] = eg.pass_rate(eg.instance_rows(task, p, p.stem))
    return out


def board() -> dict:
    refs = _load("references.json").get("tasks", {})
    sel = {t["task"]: t for t in _load("selection.json").get("tasks", [])}
    acc = {t["task"]: t for t in _load("acceptance.json").get("tasks", [])}
    tasks = []
    for task in sorted(TASKS):
        s = sel.get(task, {})
        base = {k: refs.get(task, {}).get(k, {}) for k in ("trivial", "reference")}
        base.update(ceilings(task))
        a = acc.get(task, {})
        tasks.append(
            dict(
                task=task,
                status=s.get("status", "pending"),
                close_call=s.get("close_call", False),
                models=s.get("models", {}),
                baselines={k: dict(pass_rate=v.get("pass_rate"), score=v.get("score"))
                           for k, v in base.items()},
                verdict=a.get("verdict"),
                gates={k: g["status"] for k, g in a.get("gates", {}).items()},
            )
        )  # fmt: skip
    return dict(
        threshold=eg.CONFIG["threshold"],
        config=eg.CONFIG["version"],
        built=date.today().isoformat(),
        tasks=tasks,
    )


def main() -> None:
    data = json.dumps(board(), indent=1).replace("</", "<\\/")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(TEMPLATE.read_text().replace("/*DATA*/null", data))
    print(DOC)


if __name__ == "__main__":
    main()
