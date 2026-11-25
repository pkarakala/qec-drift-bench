"""Per-check rubric breakdown of every scored run, and the rubric reference page.

usage: python experiments/rubric_report.py [--tier hard]

Each environment's score is a fixed rubric of named, weighted checks (<env>/eval/rubric.py).
This script re-grades every stored report under results/raw/ (or raw_hard/) with that rubric,
from the fields the report already holds, checks that the rubric total equals the stored score
on every instance (runs older than the rubric included), and writes
  experiments/results/rubric[_hard].csv   one row per (env, solution, check)
  experiments/results/rubric[_hard].md    the same, readable
  docs/rubrics.md                         every check, what failing it means, and its weights
Per check: `pass` is the mean value over the instances where it applies; `lost` is the mean
score it costs per instance. For each instance the losses add up to 1 - score: a failed gate
costs everything, a multiplier m costs 1 - m, and a weighted check costs m x weight x (1 - value).
Headline rows use the paper seeds only. Agent rollouts are pooled per model. Licence: Apache-2.0.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from run import raw_dir

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "experiments" / "config.json").read_text())
OUT = ROOT / "experiments" / "results"
SHARED = ("harness", "rubric", "_isolate", "exact", "evidence")
TOL = 1e-9


def load_env(env: str):
    """Import <env>/eval/harness.py and rubric.py (same module names in every env)."""
    for m in SHARED:
        sys.modules.pop(m, None)
    path = str(ROOT / env / "eval")
    sys.path.insert(0, path)
    try:
        return importlib.import_module("harness"), importlib.import_module("rubric")
    finally:
        sys.path.remove(path)


def checks_qec(H, R, row):
    if "error" in row:
        return R.invalid()
    t = H.Truth(**row["truth"])
    return R.instance_checks(row["rule"], row["changed"] == (row["rule"] != "control"),
                             row["fix"], row["harm"], H.detail_parts(row["reported"], t))


def checks_readout(H, R, row):
    if "error" in row:
        return R.invalid()
    return R.instance_checks(row["rule"], row["changed"] == bool(row["truth"]), row["fix"],
                             row["kind"], row["magnitude"])


def checks_pulse(H, R, row):
    if "error" in row:
        return R.invalid()
    fix = row["noharm"]
    if row["rule"] == "fix":
        fix = float(np.clip((row["L_stale"] - row["L_sol"]) / (row["L_stale"] - row["L_oracle"]),
                            0, 1))
    parts = {p: row.get(f"{p}_credit", 1.0) for p in R.PARTS}  # other: hard tier only
    return R.instance_checks(row["rule"], row["changed"] == (row["kind"] != "control"), fix,
                             row["noharm"], parts)


READERS = {"qec_drift_d3": checks_qec, "readout_drift": checks_readout,
           "pulse_wrong_model": checks_pulse}


def losses(c: dict) -> dict:
    """Score each check costs this instance; they add up to 1 - score."""
    for k, x in c.items():
        if x.get("gate") and x["v"] < 1:
            return {k: 1.0}
    m = float(np.prod([x["v"] for x in c.values() if x.get("mult")]))
    out = {k: (1.0 - x["v"]) if x.get("mult") else m * x["w"] * (1.0 - x["v"])
           for k, x in c.items() if not x.get("part")}
    return out


def group(name: str) -> str:
    return "agent " + name.split("__")[1] if name.startswith("agent__") else name


def grade(tier: str):
    paper = {str(s) for s in CONFIG["paper_seeds"]}
    stats = defaultdict(lambda: {"n": 0, "score": 0.0, "pass": defaultdict(list),
                                 "lost": defaultdict(float)})
    bad, n_rows, rubrics = [], 0, {}
    for env in CONFIG["envs"]:
        H, R = load_env(env)
        rubrics[env] = R
        for f in sorted((raw_dir(tier) / env).glob("*/*.json")):
            if f.stem not in paper:
                continue
            rep = json.loads(f.read_text())
            st = stats[(env, group(f.parent.name))]
            for row in rep["instances"]:
                c = READERS[env](H, R, row)
                n_rows += 1
                if abs(R.total(c) - row["score"]) > TOL:
                    bad.append(f"{f.relative_to(ROOT)} #{row.get('index', row.get('i'))}: "
                               f"rubric {R.total(c):.6f} vs stored {row['score']:.6f}")
                st["n"] += 1
                st["score"] += row["score"]
                for k, x in c.items():
                    st["pass"][k].append(x["v"])
                for k, v in losses(c).items():
                    st["lost"][k] += v
    return stats, bad, n_rows, rubrics


def write_tables(stats, rubrics, tier):
    sfx = "" if tier == "standard" else f"_{tier}"
    rows, md = [], [f"# Rubric breakdown ({tier} tier, paper seeds)", "",
                    "pass = mean check value where it applies; lost = mean score the check "
                    "costs per instance (the losses add up to 1 - score). Generated by "
                    "`experiments/rubric_report.py`; rubric definitions in "
                    "[docs/rubrics.md](../../docs/rubrics.md).", ""]
    for env in CONFIG["envs"]:
        R = rubrics[env]
        sols = sorted(s for e, s in stats if e == env)
        if not sols:
            continue
        ids = list(R.CHECKS)
        md += [f"## {env} ({R.VERSION})", "",
               "| solution | n | score | " + " | ".join(ids) + " |",
               "|---|---|---|" + "---|" * len(ids)]
        for s in sols:
            st = stats[(env, s)]
            cells = []
            for k in ids:
                p = st["pass"].get(k)
                lost = st["lost"].get(k, 0.0) / st["n"]
                cells.append("" if not p else f"{np.mean(p):.2f} / -{lost:.3f}")
                rows.append(dict(env=env, solution=s, check=k, n=st["n"],
                                 applies=len(p or []), passed=np.mean(p) if p else "",
                                 lost=lost, score=st["score"] / st["n"]))
            md.append(f"| {s} | {st['n']} | {st['score'] / st['n']:.3f} | " + " | ".join(cells)
                      + " |")
        md.append("")
    with open(OUT / f"rubric{sfx}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / f"rubric{sfx}.md").write_text("\n".join(md))


def write_docs(rubrics):
    md = ["# Grading rubrics", "",
          "Each environment's score is one fixed, versioned rubric of named checks, the same "
          "for every instance, solver, seed and tier. Every check is computed by script from "
          "the hidden truth and the answer; there is no LLM judge. A **gate** must pass or the "
          "instance scores 0. A **weighted** check adds weight x value (value in [0, 1]); on "
          "every instance the weights that apply sum to 1. A **multiplier** scales the "
          "weighted sum. A **part** is reported only (pulse: the diagnosis is the product of "
          "its parts). Source: `<env>/eval/rubric.py`; this page is generated from it by "
          "`experiments/rubric_report.py`. Per-check results: "
          "`experiments/results/rubric.md` and `rubric_hard.md`.", ""]
    for env, R in rubrics.items():
        md += [f"## {env} (`{R.VERSION}`)", "",
               "| check | fails when | " + " | ".join(R.RULES) + " |",
               "|---|---|" + "---|" * len(R.RULES)]
        for k, (why, how) in R.CHECKS.items():
            cells = [str(how.get(r, "")) for r in R.RULES]
            md.append(f"| `{k}` | {why} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["QEC diagnosis shares for the true kind (they sum to 1): hot qubit kind 0.3, qubit 0.4, factor 0.3 (with an onset "
           "inside the batch: qubit 0.26, factor 0.19, onset 0.25); coupler kind 0.3, pair 0.4, "
           "factor 0.3; burst kind 0.3, onset 0.35, decay 0.35; switching TLS kind 0.3, qubit "
           "0.4, factor 0.15, dwell 0.15; unlisted change (hard tier) kind 1.", ""]
    (ROOT / "docs" / "rubrics.md").write_text("\n".join(md))


def main() -> int:
    tier = sys.argv[sys.argv.index("--tier") + 1] if "--tier" in sys.argv else "hard"
    stats, bad, n_rows, rubrics = grade(tier)
    write_tables(stats, rubrics, tier)
    write_docs(rubrics)
    print(f"{n_rows} instances re-graded with the rubric; {len(bad)} differ from the stored score")
    for b in bad[:20]:
        print("  ", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
