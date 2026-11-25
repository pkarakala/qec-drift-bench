"""The mismatch track's runner: a fake CLI edits solution.py; the harness saves and grades it."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys

import pandas as pd
import pytest

from qec_drift.agent import decoder_run_cc as cc
from qec_drift.agent import decoder_run_codex as codex
from qec_drift.agent import isolation
from qec_drift.mismatch import run_cc as mm

ENV = "readout_drift"  # the quickest kit to build (seconds); QEC's takes minutes
SOLUTION = (
    "import numpy as np\n"
    "def solve(data):\n"
    "    x, y = data['calib_iq'], data['calib_labels']\n"
    "    m0, m1 = x[y == 0].mean(0), x[y == 1].mean(0)\n"
    "    cov = np.cov(np.vstack([x[y == 0] - m0, x[y == 1] - m1]).T)\n"
    "    return {'discriminator': [{'mean': m0, 'cov': cov, 'weight': 1.0, 'label': 0},\n"
    "                              {'mean': m1, 'cov': cov, 'weight': 1.0, 'label': 1}],\n"
    "            'diagnosis': {'changed': False, 'kinds': {}}}  # edited by the agent\n"
)


@pytest.fixture(scope="module")
def kits(tmp_path_factory):
    return mm.build_kits(tmp_path_factory.mktemp("mismatch"), [ENV])


def fake_backend(tmp_path, seen: dict) -> cc.Backend:
    def run(workdir, model, effort, max_usd, config_dir, task_prompt=None):
        seen["prompt"], seen["files"] = task_prompt, sorted(p.name for p in workdir.iterdir())
        (workdir / "solution.py").write_text(SOLUTION)
        (workdir / "NOTES_AGENT.md").write_text("nothing changed")
        return [{"type": "result", "terminal_reason": "completed"}], 0

    def prepare(name):
        d = tmp_path / "state" / name
        d.mkdir(parents=True)
        return d

    return cc.Backend(
        "fake",
        "fake_stream",
        lambda: None,
        prepare,
        run,
        lambda d: None,
        lambda ev: 2,
        lambda ev, code: dict(
            stop="completed",
            tokens_in=1,
            tokens_out=1,
            cost_usd=0.0,
            throttled=False,
            login_lost=False,
        ),
    )


def test_kit_holds_hard_instances_and_no_truth(kits):
    kit = kits[ENV]
    assert {p.name for p in kit.iterdir()} == {
        "TASK.md",
        "solution.py",
        "dev_data",
        "check.py",
        "PROMPT.md",
    }
    data = pickle.loads(sorted((kit / "dev_data").glob("*.pkl"))[0].read_bytes())
    assert data["tier"] == "hard"
    assert not {"truth", "answer", "seed", "rule"} & set(data)
    text = mm.prompt(kit)
    assert "hard tier" in text and f"at most {isolation.TOOL_CALLS} tool calls" in text


def test_episode_saves_the_solution_and_a_row(tmp_path, monkeypatch, kits):
    monkeypatch.setattr(mm, "agents_dir", lambda: tmp_path / "agents")
    monkeypatch.setattr(mm, "build_kits", lambda root, envs: kits)
    seen: dict = {}
    args = mm.parse_args(
        [
            "--model",
            "fake-model",
            "--arm",
            "fake_arm",
            "--rollouts",
            "1",
            "--envs",
            ENV,
            "--out",
            str(tmp_path / "out"),
        ]
    )
    df = mm.run_arm(args, backend=fake_backend(tmp_path, seen))
    assert "hard tier" in seen["prompt"] and "dev_data" in seen["files"]
    row = df.iloc[0]
    assert (row.task, row.outcome, row.turns) == (f"mismatch-{ENV}-hard", "submitted", 2)
    saved = tmp_path / "agents" / "fake-model" / ENV / "r0"
    assert (saved / "solution.py").read_text() == SOLUTION
    assert json.loads((saved / "meta.json").read_text())["tier"] == "hard"
    assert pd.read_parquet(tmp_path / "out" / "fake_arm.parquet").attempt_id.notna().all()
    assert (tmp_path / "out" / "work" / "fake_arm" / f"{ENV}-r0").is_dir()
    again = mm.run_arm(args, backend=fake_backend(tmp_path, seen := {}))
    assert seen == {} and len(again) == 1  # a submitted rollout is not run twice


def test_saved_solution_meets_the_environments_own_grader(tmp_path):
    sol = tmp_path / "solution.py"
    sol.write_text(SOLUTION)
    out = subprocess.run(
        [sys.executable, "evaluate.py", "--solution", str(sol), "--quick", "--tier", "hard"],
        cwd=mm.TRACK / ENV,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    assert 0.0 <= float(out.stdout.split()[-1]) <= 1.0


def test_both_scaffolds_take_the_tracks_prompt(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(isolation, "stream", lambda cmd, *a, **k: (calls.append(cmd), ([], 0))[1])
    cc.run_cli(tmp_path, "m", None, 1.0, tmp_path / "cfg", task_prompt="TRACK PROMPT")
    assert calls[-1][calls[-1].index("-p", 3) + 1] == "TRACK PROMPT"
    cmd = codex.command(tmp_path, tmp_path / "cfg", "m", None, task_prompt="TRACK PROMPT")
    assert cmd[-1] == "TRACK PROMPT"
    assert codex.command(tmp_path, tmp_path / "cfg", "m", None)[-1] == cc.prompt()


def test_label_rejects_run_py_separators():
    args = mm.parse_args(["--model", "a__b", "--arm", "x"])
    with pytest.raises(ValueError):
        mm.label_of(args)
