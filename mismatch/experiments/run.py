"""Run every solution in experiments/config.json on the public seed and the frozen paper seeds.

usage: python experiments/run.py [--jobs N] [--env NAME] [--solution NAME] [--seeds K]
                                 [--tier hard]

Each (env, solution, seed) run calls the environment's own `evaluate.py --report`, so the
numbers come from the same grader a submission meets. Reports land in
experiments/results/raw/<env>/<solution>/<seed>.json and are reused when present, so an
interrupted run resumes where it stopped. Delete a file to recompute it.

`reference_swapped_data` is the reference scored with MISMATCH_ABLATE=1: each instance gets
the previous instance's data (eval/_isolate.py). `--tier hard` runs config.json's "hard"
solutions on the same seeds with `evaluate.py --tier hard` and writes to results/raw_hard/.
Licence: Apache-2.0.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "experiments" / "config.json").read_text())
RAW = ROOT / "experiments" / "results" / "raw"
THREAD_PINS = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}


AGENTS = ROOT / "experiments" / "results" / "agent_solutions"


def raw_dir(tier: str = "standard") -> Path:
    return RAW if tier == "standard" else RAW.with_name(f"raw_{tier}")


def agents_dir(tier: str = "standard") -> Path:
    return AGENTS if tier == "standard" else AGENTS.with_name(f"agent_solutions_{tier}")


def solutions(env: str, tier: str = "standard") -> dict:
    """Configured solutions plus every agent solution found under results/agent_solutions/
    <model>/<env>/r<k>/solution.py (agent_solutions_hard/ for the hard tier; kind "agent",
    name agent__<model>__r<k>)."""
    conf = CONFIG["solutions"] if tier == "standard" else CONFIG["tiers"][tier]["solutions"]
    sols = {k: tuple(v) for k, v in conf[env].items()}
    for f in sorted(agents_dir(tier).glob(f"*/{env}/r*/solution.py")):
        model, rollout = f.parts[-4], f.parts[-2]
        sols[f"agent__{model}__{rollout}"] = (os.path.relpath(f, ROOT / env), "agent")
    return sols


def seed_tag(seed: int | None) -> str:
    return "public" if seed is None else str(seed)


def jobs(env_filter=None, sol_filter=None, n_seeds=None, tier="standard"):
    seeds = [None] + CONFIG["paper_seeds"][:n_seeds]
    for env in CONFIG["envs"]:
        if env_filter and env != env_filter:
            continue
        for name, (rel, kind) in solutions(env, tier).items():
            if sol_filter and name != sol_filter:
                continue
            k = CONFIG["check_seeds"] + 1 if kind in ("lazy", "leak") else len(seeds)
            only = covered_seeds(env, rel)
            for seed in seeds[:k]:
                if only is None or seed_tag(seed) in only:
                    yield env, name, rel, seed, tier


def covered_seeds(env: str, rel: str) -> set[str] | None:
    """Seed tags an agent solution answers, from its seeds.json (instance-shape runs of
    tool_loop answer only the seeds they played), or None for every seed."""
    f = (ROOT / env / rel).resolve().parent / "seeds.json"
    return set(map(str, json.loads(f.read_text()))) if f.exists() else None


def run_one(env: str, name: str, rel: str, seed: int | None,
            tier: str = "standard") -> tuple[str, float, float]:
    out = raw_dir(tier) / env / name / f"{seed_tag(seed)}.json"
    if out.exists():
        return f"{env}/{name}/{seed_tag(seed)}", json.loads(out.read_text())["score"], 0.0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    cmd = [sys.executable, "evaluate.py", "--solution", rel, "--report", str(tmp), "--tier", tier]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    environ = dict(os.environ, **THREAD_PINS,
                   MISMATCH_ABLATE="1" if name == "reference_swapped_data" else "0")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=ROOT / env, capture_output=True, text=True, env=environ)
    if res.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"{env}/{name}/{seed_tag(seed)} failed:\n{res.stderr[-800:]}")
    rep = json.loads(tmp.read_text())
    rep["_meta"] = {"env": env, "solution": name, "path": rel, "seed": seed_tag(seed),
                    "wall_s": round(time.time() - t0, 1), "config": CONFIG["version"],
                    "tier": tier}
    out.write_text(json.dumps(rep, default=float))
    tmp.unlink()
    return f"{env}/{name}/{seed_tag(seed)}", rep["score"], time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--env")
    ap.add_argument("--solution")
    ap.add_argument("--seeds", type=int, help="use only the first K paper seeds")
    ap.add_argument("--tier", choices=("standard", "hard"), default="hard")
    a = ap.parse_args()
    todo = list(jobs(a.env, a.solution, a.seeds, a.tier))
    # Longest first: the QEC references dominate the wall time.
    todo.sort(key=lambda j: (j[0] != "qec_drift_d3", "reference" not in j[1]))
    print(f"{len(todo)} runs on {a.jobs} workers", flush=True)
    failed = 0
    with ThreadPoolExecutor(a.jobs) as pool:
        futs = [pool.submit(run_one, *j) for j in todo]
        for f in as_completed(futs):
            try:
                tag, score, secs = f.result()
                print(f"{tag:55s} {score:.4f} {secs:7.1f} s", flush=True)
            except RuntimeError as e:
                failed += 1
                print(e, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
