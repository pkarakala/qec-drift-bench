"""Re-grade data-out task reports check by check, with each task's own rubric.

Every data-out task scores an instance as named, weighted checks (envs/<task>/eval/rubric.py:
gates, weighted checks, multipliers and parts). The grader's per-instance report holds the
fields those checks are computed from. This module rebuilds the checks from a report, confirms
that their total equals the stored score, and summarises each check over a set of reports. The
acceptance gate A10 (identifiable, docs/task-standard.md) reads the summary.

    rubric = load_rubric("readout_drift")
    stats = check_stats("readout_drift", [report, ...])   # {check: (mean value, instances)}
"""

from __future__ import annotations

import importlib.util
from collections import defaultdict
from collections.abc import Callable, Iterable
from types import ModuleType

import numpy as np

from qec_drift.agent.envs_kit import ENVS

TOL = 1e-9


def load_rubric(task: str) -> ModuleType:
    """envs/<task>/eval/rubric.py, imported under a name of its own."""
    path = ENVS / task / "eval" / "rubric.py"
    spec = importlib.util.spec_from_file_location(f"_rubric_{task}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _readout(R: ModuleType, row: dict) -> dict:
    return R.instance_checks(
        row["rule"], row["changed"] == bool(row["truth"]), row["fix"], row["kind"], row["magnitude"]
    )


def _coherence(R: ModuleType, row: dict) -> dict:
    changed = row["rule"] == "fix"
    return R.instance_checks(
        row["rule"], row["changed"] == changed, row["fix"], row["kind"], row["params"]
    )


# task -> (rubric module, one report instance) -> {check: {"v", "w", ...}}
READERS: dict[str, Callable[[ModuleType, dict], dict]] = {
    "coherence_tls": _coherence,
    "readout_drift": _readout,
}


def instance_checks(task: str, R: ModuleType, row: dict) -> dict:
    """The rubric checks of one report instance; an instance that errored fails the gate."""
    return R.invalid() if "error" in row else READERS[task](R, row)


def check_stats(task: str, reports: Iterable[dict]) -> dict[str, tuple[float, int]]:
    """Per check: mean value and number of instances where it applies.

    A check named in the rubric's REQUIRES counts only on instances where the check it requires
    passed (a fault's location only under the right kind), so one miss is not counted twice.
    Raises ValueError if a rebuilt total differs from the stored score.
    """
    R = load_rubric(task)
    req = getattr(R, "REQUIRES", {})
    vals: dict[str, list[float]] = defaultdict(list)
    for rep in reports:
        for row in rep["instances"]:
            c = instance_checks(task, R, row)
            if abs(R.total(c) - row["score"]) > TOL:
                raise ValueError(f"{task}: rubric total {R.total(c)} != stored {row['score']}")
            for k, x in c.items():
                if k not in req or c.get(req[k], {}).get("v", 0.0) >= 1.0:
                    vals[k].append(x["v"])
    return {k: (float(np.mean(v)), len(v)) for k, v in vals.items()}
