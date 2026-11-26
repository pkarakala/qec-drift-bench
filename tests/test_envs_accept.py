"""The acceptance gates and the rubric re-grader for the data-out tasks, on made-up inputs."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from qec_drift import paths
from qec_drift.scoring import envs_rubric


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, paths.REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


acc = _script("accept_task")
sel = _script("select_tasks")


def test_a_section_with_a_unit_needs_a_source_or_an_assumption():
    text = (
        "## Story\n\nThe qubit is at 5 GHz.\n\n"
        "## Device\n\nT1 of 30 us (Carroll 2022).\n\n"
        "## Limits\n\nA 20 us wait (assumption).\n\n"
        "## Scoring\n\nNo units here.\n"
    )
    assert acc.uncited_sections(text) == ["Story"]


def test_one_failed_gate_rejects_and_any_pending_gate_leaves_it_open():
    g = {k: dict(status="pass") for k in acc.GATES}
    assert acc.verdict(g) == "accepted"
    g["H1"]["status"] = "pending"
    assert acc.verdict(g) == "open"
    g["A3"]["status"] = "fail"
    assert acc.verdict(g) == "rejected"


@pytest.mark.parametrize("task", ["readout_drift", "pulse_wrong_model"])
def test_every_shipped_task_has_its_files_and_cites_its_numbers(task):
    t = acc.Task(task, dry=True)
    assert t.a1()["status"] == "pass"
    assert t.a13()["status"] == "pass"
    assert t.ceilings, "a ceiling solution in envs/<task>/ceilings/"


def test_the_readout_rubric_is_rebuilt_from_a_report_row():
    row = dict(rule="fix", changed=True, truth={"t1_drop": 1.0}, fix=0.5, kind=1.0, magnitude=0.0)
    R = envs_rubric.load_rubric("readout_drift")
    row["score"] = R.total(envs_rubric.instance_checks("readout_drift", R, row))
    stats = envs_rubric.check_stats("readout_drift", [dict(instances=[row])])
    assert stats["fix_gap"] == (0.5, 1)
    assert stats["change_call"] == (1.0, 1)
    with pytest.raises(ValueError):
        envs_rubric.check_stats("readout_drift", [dict(instances=[dict(row, score=0.99)])])


def test_an_errored_instance_fails_the_valid_answer_gate():
    R = envs_rubric.load_rubric("pulse_wrong_model")
    c = envs_rubric.instance_checks("pulse_wrong_model", R, dict(error="timeout", score=0.0))
    assert R.total(c) == 0.0


def test_the_interval_covers_the_point_and_widens_with_spread_between_seeds():
    rng = np.random.default_rng(0)
    even = [np.array([1, 0] * 10, float) for _ in range(5)]
    split = [np.ones(20), np.zeros(20), np.ones(20), np.zeros(20), np.full(20, 0.5)]
    lo, hi = sel.cluster_bootstrap(even, rng)
    lo2, hi2 = sel.cluster_bootstrap(split, rng)
    assert lo <= 0.5 <= hi and lo2 <= 0.5 <= hi2
    assert hi2 - lo2 > hi - lo


def test_a_close_call_is_marked_when_the_best_interval_spans_the_threshold():
    rates = pd.DataFrame(dict(env=["t"], model=["m"], pass_rate=[0.7], score=[0.7], rollouts=[3]))
    refs = {"t": {"reference": {"pass_rate": 0.6, "score": 0.7}}}
    d = sel.decide("t", rates, refs, 0.75, {("t", "m"): (0.62, 0.79)})
    assert d["status"] == "included" and d["close_call"]
    text = sel.markdown([d], rates, refs, 0.75, {("t", "m"): (0.62, 0.79)})
    assert "| t | included (close call) | 0.60 / 0.70 | 0.70 [0.62, 0.79] / 0.70 |" in text


def test_the_leaderboard_page_takes_its_data_inline(tmp_path, monkeypatch):
    lb = _script("envs_leaderboard")
    data = dict(threshold=0.75, config="c", built="2026-11-26", tasks=[])
    monkeypatch.setattr(lb, "board", lambda: data)
    monkeypatch.setattr(lb, "DOC", tmp_path / "index.html")
    lb.main()
    html = (tmp_path / "index.html").read_text()
    assert "/*DATA*/null" not in html and '"threshold": 0.75' in html
