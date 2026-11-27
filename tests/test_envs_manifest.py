"""The release manifest covers every task's scored files and changes when one of them does."""

from __future__ import annotations

import importlib.util

from qec_drift import paths
from qec_drift.agent.envs_kit import TASKS

spec = importlib.util.spec_from_file_location(
    "envs_manifest", paths.REPO_ROOT / "scripts" / "envs_manifest.py")
man = importlib.util.module_from_spec(spec)
spec.loader.exec_module(man)


def test_manifest_lists_every_task_and_its_grader():
    m = man.manifest()
    assert set(m["tasks"]) == set(TASKS)
    for task, t in m["tasks"].items():
        assert t["rubric"].startswith(f"{task}/rubric-")
        assert {"TASK.md", "evaluate.py", "eval/harness.py", "eval/rubric.py"} <= set(t["files"])


def test_manifest_is_deterministic_and_tracks_content(tmp_path, monkeypatch):
    assert man.manifest() == man.manifest()
    task = sorted(TASKS)[0]
    real = man.sha256
    target = paths.REPO_ROOT / "envs" / task / "eval" / "harness.py"
    monkeypatch.setattr(man, "sha256", lambda p: "x" if p == target else real(p))
    assert man.manifest()["tasks"][task]["files"]["eval/harness.py"] == "x"
