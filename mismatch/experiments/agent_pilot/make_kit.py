"""Build the agent-pilot kit: what an agent sees for each environment, and nothing else.

usage: python experiments/agent_pilot/make_kit.py OUT_DIR [--tier hard]

For each environment, OUT_DIR/<env>/ gets TASK.md, solution.py (the starter), dev_data/*.pkl
(the `data` dict of every public instance, no truth), check.py (runs solve on dev_data and
checks the return shape and time, no scoring) and PROMPT.md. eval/ and the grader are not
copied: the agent works on another machine and its solution.py is scored back here on the
frozen paper seeds with the environment's own evaluate.py. With --tier hard, dev_data holds the
hard tier's public instances and PROMPT.md says so; TASK.md is the same file (it has a hard-tier
section). Licence: Apache-2.0.
"""

import argparse
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIMIT_S = {"qec_drift_d3": 600, "readout_drift": 120, "pulse_wrong_model": 120}


def public_data(env: str, seed: int | None = None, tier: str = "standard"):
    """The `data` dict of every instance of `seed` (None = the public set) of `tier`, in the order
    the env's evaluate.py builds them, without truth."""
    sys.path.insert(0, str(ROOT / env / "eval"))
    import harness as h  # noqa: E402  (each env's own harness, imported in turn)
    try:
        if env == "qec_drift_d3":
            s, mix = h.instance_list(seed, False, tier)
            return [h.build_instance(s, i, fam, tier)[0] for i, fam in enumerate(mix)]
        if env == "readout_drift":
            return [h.make_instance(s, spec, tier=tier)[0]
                    for s, spec in h.instance_list(seed, tier=tier)]
        h.use_tier(tier)
        return [inst["data"] for inst in h.build(h.PUBLIC_SEED if seed is None else int(seed),
                                                 h.N_INSTANCES)]
    finally:
        sys.path.pop(0)
        for m in ("harness", "rubric", "_isolate"):  # same names in every env
            sys.modules.pop(m, None)


CHECK = '''"""Run solve() from solution.py on every dev_data instance; report shape errors and time.
This does not score: the answers are hidden. Usage: python check.py [N]"""
import pickle, sys, time
from pathlib import Path
import solution
files = sorted(Path("dev_data").glob("*.pkl"))[: int(sys.argv[1]) if len(sys.argv) > 1 else None]
for f in files:
    data = pickle.loads(f.read_bytes())
    t = time.time()
    out = solution.solve(data)
    dt = time.time() - t
    ok = isinstance(out, dict) and isinstance(out.get("diagnosis", {}).get("changed"), bool)
    print(f"{f.name}: {'ok' if ok else 'BAD RETURN'}  changed={out.get('diagnosis', {}).get('changed')}"
          f"  {dt:.1f} s{'  OVER LIMIT' if dt > LIMIT else ''}")
'''

PROMPT = """You are solving a research task. Read TASK.md in this folder first.

Rules for this run:
- Work only inside this folder. Do not use the web. Do not look for the task's grader or its
  source anywhere on this machine: it is not here, and looking for it ends the run.
- `python evaluate.py` from TASK.md is not available. Instead, dev_data/ holds the `data`
  dict of each public instance (pickled), without answers. `python check.py` runs your
  solve() on them and checks the return shape and the time limit ({limit} s per instance).
- Edit solution.py. When you are done, it must define solve(data) -> dict as TASK.md says.
  It will be scored afterwards on fresh secret instances you have not seen.
- Python packages available: numpy, scipy, stim, pymatching. Do not install others.
- When you are finished, write a short NOTES_AGENT.md: what you think changes in this task, how
  your solution decides "changed", and what it does about it.
"""

HARD_NOTE = """- Every instance in this run is from the hard tier (`data["tier"] == "hard"`). Read the
  "Hard tier" section of TASK.md: it changes what can happen and what you may report.
"""


def write_env(env: str, d: Path, tier: str = "standard") -> int:
    """Write one environment's kit into the empty or new folder `d`; return the instance count."""
    limit = LIMIT_S[env]
    (d / "dev_data").mkdir(parents=True)
    shutil.copy(ROOT / env / "TASK.md", d / "TASK.md")
    shutil.copy(ROOT / env / "sandbox" / "solution.py", d / "solution.py")
    for i, data in enumerate(public_data(env, tier=tier)):
        (d / "dev_data" / f"{i:02d}.pkl").write_bytes(pickle.dumps(data))
    (d / "check.py").write_text(CHECK.replace("LIMIT", str(limit)).replace(
        "if dt > " + str(limit), f"if dt > {limit}"))
    prompt = PROMPT.format(limit=limit)
    if tier == "hard":
        prompt = prompt.replace("- Python packages", HARD_NOTE + "- Python packages")
    (d / "PROMPT.md").write_text(prompt)
    return len(list((d / "dev_data").glob("*.pkl")))


def main(out: Path, tier: str = "standard"):
    for env in LIMIT_S:
        d = out / env
        if d.exists():
            shutil.rmtree(d)
        print(env, write_env(env, d, tier), "instances")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--tier", choices=("standard", "hard"), default="hard")
    a = ap.parse_args()
    main(a.out.resolve(), a.tier)
