"""One results table for QEC-Drift and the mismatch environments, one row per graded instance.

usage:
  python experiments/shared_table.py                       # mismatch rows, paper seeds
  python experiments/shared_table.py --qec-drift data/agent_launch2/gpt_6_astra_max_codex.parquet \
      --qec-model gpt-6-astra --qec-effort max --qec-scaffold codex
  (--qec-drift may repeat; the other --qec-* flags then repeat in the same order)

Writes experiments/results/shared_table.csv and prints a Markdown summary. Each benchmark keeps
its own grader and its own native score; the shared columns are the ones both can answer:

  changed_correct  the agent's "did the system change?" answer is right (QEC-Drift: not
                   report_adequate vs not adequate_truth; here: `changed` vs the truth)
  fix              share of the gap closed between the stale model and the best-known model
                   (QEC-Drift gap_closed; here the env's fix credit), clipped to [0, 1]
  diag_correct     the diagnosis is right (QEC-Drift diagnosis_correct: scope plus
                   localization; here: changed_correct and, on changed instances, at least half
                   the env's diagnosis-detail credit)
  passed           QEC-Drift's rule applied to both: diag_correct, fix >= 0.5 where a fix is
                   graded, and (QEC-Drift drift tasks) gap_drift >= 0.5
  score            native [0, 1] score (mismatch envs only; QEC-Drift has none)

Intervals: cluster bootstrap with the seed (mismatch) or the task (QEC-Drift) as the cluster,
the same routine and settings as analyze.py. Licence: Apache-2.0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze import BOOT, CONFIG, OUT, load
from run import agents_dir
from stats import cluster_bootstrap

AGENTS_DIR = OUT / "agent_solutions"
PASS_GAP = 0.5  # QEC-Drift awareness.PASS_GAP
FIELDS = ["benchmark", "task", "cluster", "instance", "seed_set", "solution", "kind", "model",
          "effort", "scaffold", "provider", "rollout", "truth_changed", "said",
          "changed_correct", "fix", "fix_drift", "diag", "diag_correct", "passed", "score",
          "error"]


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _agent_meta(env: str, solution: str) -> dict:
    """model, effort, scaffold of an agent solution from its meta.json (tool-loop runs) or,
    for the Claude Code pilot, from its folder name."""
    _, label, rollout = solution.split("__")
    found = [d / label / env / rollout / "meta.json" for d in (AGENTS_DIR, agents_dir("hard"))]
    found = [f for f in found if f.exists()]
    m = json.loads(found[0].read_text()) if found else {}
    return dict(model=m.get("model", label), effort=m.get("effort"),
                scaffold=m.get("scaffold", "claude_code_headless"),
                provider=m.get("provider", "anthropic" if label.startswith("claude") else ""),
                rollout=rollout)


def mismatch_rows(recs=None) -> list[dict]:
    paper = {str(s) for s in CONFIG["paper_seeds"]}
    out = []
    for r in recs if recs is not None else load():
        fix, diag = _num(r["fix"]), _num(r["diag"])
        right = r["said"] is not None and bool(r["said"]) == bool(r["truth_changed"])
        diag_ok = right and (not r["truth_changed"] or diag is None or diag >= 0.5)
        meta = (_agent_meta(r["env"], r["solution"]) if r["kind"] == "agent" else
                dict(model="", effort=None, scaffold="scripted", provider="", rollout=""))
        out.append(dict(
            benchmark="mismatch-envs", task=r["env"], cluster=r["seed"],
            instance=f"{r['seed']}:{r['index']}",
            seed_set="public" if r["seed"] == "public" else "paper" if r["seed"] in paper
            else "secret",
            solution=r["solution"], kind=r["kind"], **meta, truth_changed=bool(r["truth_changed"]),
            said=r["said"], changed_correct=right, fix=fix, fix_drift=None, diag=diag,
            diag_correct=diag_ok, passed=diag_ok and (fix is None or fix >= PASS_GAP),
            score=_num(r["score"]), error=bool(r["error"])))
    return out


def qec_drift_rows(path: Path, model: str, effort: str | None, scaffold: str,
                   provider: str = "") -> list[dict]:
    """Rows of a QEC-Drift score table (decoder_run*.py --awareness parquet) in this schema."""
    import pandas as pd  # noqa: PLC0415  (only needed for QEC-Drift tables)
    df = pd.read_parquet(path)
    out = []
    for r in df.to_dict("records"):
        truth = not bool(r["adequate_truth"])
        said = None if r.get("report_adequate") is None else not bool(r["report_adequate"])
        gap, drift = _num(r.get("gap_closed")), _num(r.get("gap_drift"))
        out.append(dict(
            benchmark="qec-drift", task=r["task"], cluster=r["task"],
            instance=f"{r['task']}:{r.get('seed', '')}", seed_set=str(r.get("set", "")),
            solution=Path(path).stem, kind="agent", model=model, effort=effort,
            scaffold=scaffold, provider=provider, rollout=str(r.get("attempt_id", "")),
            truth_changed=truth, said=said, changed_correct=said == truth,
            fix=None if gap is None else min(max(gap, 0.0), 1.0),
            fix_drift=None if drift is None else min(max(drift, 0.0), 1.0),
            diag=None, diag_correct=bool(r.get("diagnosis_correct", False)),
            passed=bool(r.get("passed", False)), score=None,
            error=r.get("outcome") == "error"))
    return out


def summarize(rows: list[dict], seed_sets=("paper", "")) -> list[dict]:
    """Per (benchmark, task, solution), QEC-Drift tasks pooled as "all" (one row per task and
    arm there, so the task is the cluster): pass rate, changed-answer accuracy, false alarm, miss,
    mean fix and mean native score, with cluster-bootstrap intervals on the rates."""
    rng = np.random.default_rng(BOOT["rng_seed"] + 2)
    groups = defaultdict(list)
    for r in rows:
        if r["seed_set"] in seed_sets or r["benchmark"] != "mismatch-envs":
            task = r["task"] if r["benchmark"] == "mismatch-envs" else "all"
            groups[(r["benchmark"], task, r["solution"])].append(r)
    out = []
    for (bench, task, sol), rs in sorted(groups.items()):
        def rate(field, sub=rs):
            by = defaultdict(list)
            for r in sub:
                by[r["cluster"]].append(float(bool(r[field])))
            return cluster_bootstrap(by, BOOT["resamples"], rng, BOOT["level"]) if by else None
        ctrl = [dict(r, fa=r["said"] is not False) for r in rs if not r["truth_changed"]]
        chg = [dict(r, ms=r["said"] is not True) for r in rs if r["truth_changed"]]
        fixes = [r["fix"] for r in rs if r["fix"] is not None]
        scores = [r["score"] for r in rs if r["score"] is not None]
        out.append(dict(benchmark=bench, task=task, solution=sol, kind=rs[0]["kind"],
                        model=rs[0]["model"], scaffold=rs[0]["scaffold"], n=len(rs),
                        passed=rate("passed"), changed_correct=rate("changed_correct"),
                        false_alarm=rate("fa", ctrl), miss=rate("ms", chg),
                        fix=float(np.mean(fixes)) if fixes else None,
                        score=float(np.mean(scores)) if scores else None))
    return out


def _ci(t):
    return "-" if t is None else f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"


def _f(x):
    return "-" if x is None else f"{x:.2f}"


def markdown(S: list[dict]) -> str:
    lines = ["| benchmark | task | solution | n | pass | changed right | false alarm | miss "
             "| mean fix | score |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        cells = [s["benchmark"], s["task"], s["solution"], str(s["n"]), _ci(s["passed"]),
                 _ci(s["changed_correct"]), _ci(s["false_alarm"]), _ci(s["miss"]),
                 _f(s["fix"]), _f(s["score"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--qec-drift", action="append", default=[], type=Path)
    ap.add_argument("--qec-model", action="append", default=[])
    ap.add_argument("--qec-effort", action="append", default=[])
    ap.add_argument("--qec-scaffold", action="append", default=[])
    ap.add_argument("--kinds", default="floor,reference,agent",
                    help="mismatch solution kinds in the printed summary (CSV keeps all)")
    ap.add_argument("--out", type=Path, default=OUT / "shared_table.csv")
    a = ap.parse_args()
    rows = mismatch_rows()
    for i, p in enumerate(a.qec_drift):
        pick = lambda xs, d=None: xs[i] if i < len(xs) else d  # noqa: E731
        rows += qec_drift_rows(p, pick(a.qec_model, p.stem), pick(a.qec_effort),
                               pick(a.qec_scaffold, "api"))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    kinds = set(a.kinds.split(","))
    S = [s for s in summarize(rows) if s["benchmark"] != "mismatch-envs" or s["kind"] in kinds]
    print(markdown(S))


if __name__ == "__main__":
    main()
