"""Score trivial, reference, starter and every grader check in every environment.

usage: python run_all.py [--seeds N] [--env NAME] [--ablate] [--tier hard]

Runs on the public seeds, then on N secret seeds (default 5) drawn fresh, and prints the mean
and standard error per solution. A grader check is a lazy or exploit solution in <env>/checks/
listed in checks/expect.json with kind "lazy", "leak" or "heuristic". Heuristics are honest
simple methods and are only listed. The run fails (exit 1) if any lazy or leak check scores more than
trivial + 0.05 on the same seeds: that means the scorer rewards something other than the skill
the task claims to measure.

--ablate adds a data-ablation run: the reference is scored with every instance's data swapped
for the previous instance's (MISMATCH_ABLATE=1, see eval/_isolate.py). If the reference still
beats trivial + 0.05, the score does not depend on the data and the task fails the check.
"""

import argparse
import json
import os
import secrets
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MARGIN = 0.05
BASE = {"trivial": "eval/_trivial.py", "reference": "eval/_reference.py",
        "starter": "sandbox/solution.py"}


def score(env: Path, rel: str, seed: int | None, ablate: bool = False,
          tier: str = "standard") -> float:
    cmd = [sys.executable, "evaluate.py", "--solution", rel, "--tier", tier]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    environ = dict(os.environ, MISMATCH_ABLATE="1" if ablate else "0")
    out = subprocess.run(cmd, cwd=env, capture_output=True, text=True, env=environ)
    lines = out.stdout.strip().splitlines()
    try:
        return float(lines[-1])
    except (IndexError, ValueError):
        sys.exit(f"{env.name} {rel}: no score\n{out.stderr[-500:]}")


def mean_se(xs: list[float]) -> str:
    if len(xs) < 2:
        return f"{xs[0]:.3f}" if xs else "-"
    return f"{statistics.mean(xs):.3f} +/- {statistics.stdev(xs) / len(xs) ** 0.5:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="number of secret seeds")
    ap.add_argument("--env", help="run one environment only")
    ap.add_argument("--ablate", action="store_true", help="also score the reference on "
                    "swapped data (data-ablation check)")
    ap.add_argument("--tier", choices=("standard", "hard"), default="hard")
    a = ap.parse_args()
    seeds = [None] + [secrets.randbelow(2**31) for _ in range(a.seeds)]
    failed = []
    envs = sorted(p.parent for p in ROOT.glob("*/evaluate.py"))
    for env in envs:
        if a.env and env.name != a.env:
            continue
        checks = json.loads((env / "checks" / "expect.json").read_text())
        sols = dict(BASE) | {c["file"]: f"checks/{c['file']}" for c in checks}
        kinds = {c["file"]: c["kind"] for c in checks}
        if a.ablate:
            sols["reference_ablated"] = BASE["reference"]
            kinds["reference_ablated"] = "lazy"
        res: dict[str, list[float]] = {k: [] for k in sols}
        for seed in seeds:
            for name, rel in sols.items():
                t0 = time.time()
                res[name].append(score(env, rel, seed, ablate=name == "reference_ablated",
                                       tier=a.tier))
                tag = "public" if seed is None else "secret"
                print(f"{env.name:18s} {tag:6s} {name:26s} {res[name][-1]:.4f}"
                      f"  {time.time() - t0:6.1f} s", flush=True)
        print(f"\n{env.name} ({a.tier}): public, then mean +/- se over {a.seeds} secret seeds")
        for name in sols:
            print(f"  {name:26s} {res[name][0]:.3f}   {mean_se(res[name][1:])}")
            if name in BASE or kinds.get(name) not in ("lazy", "leak"):
                continue
            over = [s - t for s, t in zip(res[name], res["trivial"]) if s - t > MARGIN]
            if over:
                failed.append(f"{env.name}/{name}: beats trivial by {max(over):.3f}")
        print()
    for f in failed:
        print("GRADER CHECK FAILED:", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
