"""Run the mismatch track (hard tier) with Claude Code or Codex as the agent, on a subscription.

The three mismatch environments (mismatch/README.md) ask the agent for data, not for code that
runs next to hidden shots: `solve(data) -> dict` returns a detector error model, Gaussian
readout components or pulse samples, and each environment's own grader applies that data to
hidden test data. One episode per environment and rollout, in the program shape of the mismatch
pilot: the work folder holds the pilot kit (TASK.md, the starter solution.py, dev_data/ with the
public instances' data and no answers, check.py); the agent edits solution.py. The harness then
copies solution.py out to mismatch/experiments/results/agent_solutions_hard/<label>/<env>/r<k>/,
where mismatch/experiments/run.py grades it on the public and paper seeds exactly as it grades
every baseline.

The episode runs on the decoder tasks' scaffold: decoder_run_cc.CLAUDE_CODE or
decoder_run_codex.BACKEND, with agent/isolation.py's sandbox profile, login copy without a
refresh token, tool-call and time limits, and the throttle and login-loss handling of
decoder_run_cc. The work folder has a random name, as there. Only the prompt is this track's.

Example:
    uv run python -m qec_drift.mismatch.run_cc --backend claude-code --model claude-opus-5-5 \\
        --arm opus_mm_hard --rollouts 3
    uv run python mismatch/experiments/run.py --tier hard --seeds 5
    uv run python mismatch/experiments/analyze.py --tier hard
    uv run python mismatch/experiments/selection.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import secrets
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pandas as pd

from qec_drift import paths
from qec_drift.agent import attempts, isolation
from qec_drift.agent import decoder_run as dr
from qec_drift.agent import decoder_run_cc as cc
from qec_drift.agent.provider_limits import interrupted as provider_interrupted

TRACK = paths.REPO_ROOT / "mismatch"
EXPERIMENTS = TRACK / "experiments"
TIER = "hard"  # the only tier in this track: the standard tier is saturated (README)
BUDGET_NOTE = (
    "- Budget: at most {calls} tool calls and about 50 minutes of wall-clock time. The run is "
    "stopped at either limit, and the solution.py that exists then is scored.\n"
)
PROVIDER = {"claude-code": "anthropic", "codex": "openai"}


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def kit_module() -> ModuleType:
    """mismatch/experiments/agent_pilot/make_kit.py: what an agent may see, and nothing else."""
    return _load("mismatch_make_kit", EXPERIMENTS / "agent_pilot" / "make_kit.py")


def env_names() -> list[str]:
    return list(kit_module().LIMIT_S)


def agents_dir() -> Path:
    """Where run.py --tier hard finds agent solutions (run.agents_dir("hard"))."""
    return EXPERIMENTS / "results" / f"agent_solutions_{TIER}"


def build_kits(root: Path, envs: list[str]) -> dict[str, Path]:
    """One kit per environment under root/kit/, built once and reused (the QEC kit takes a few
    minutes: its weak faults are redrawn until the data carries enough evidence)."""
    kit = kit_module()
    out = {}
    for env in envs:
        d = root / "kit" / env
        if not (d / "PROMPT.md").is_file():  # written last, so a half-built kit is rebuilt
            if d.exists():
                isolation.clear(d)
            kit.write_env(env, d, TIER)
        out[env] = d
    return out


def prompt(kit_dir: Path) -> str:
    """The pilot's prompt for this kit, with this scaffold's budget."""
    note = BUDGET_NOTE.format(calls=isolation.TOOL_CALLS)
    text = (kit_dir / "PROMPT.md").read_text()
    return text.replace("- Python packages", note + "- Python packages")


def label_of(args: argparse.Namespace) -> str:
    """The solution folder's model label: run.py names the rollout agent__<label>__r<k>."""
    label = args.label or (args.model + (f"-{args.effort}" if args.effort else ""))
    if "__" in label or "/" in label:
        raise ValueError(f"label {label!r} may not hold '__' or '/'")
    return label


def task_name(env: str) -> str:
    return f"mismatch-{env}-{TIER}"


def save_solution(workdir: Path, dest: Path, meta: dict) -> str:
    """Copy the agent's solution.py and notes (regular files only, size-capped, as the decoder
    scaffold reads decoder.py) to `dest`, with meta.json. A missing solution is saved empty: it
    scores 0 on every instance, so a failed rollout counts as a failure, not as a gap."""
    src = dr.read_regular_file(workdir / "solution.py")
    if dest.exists():
        isolation.clear(dest)
    dest.mkdir(parents=True)
    (dest / "solution.py").write_text(src)
    notes = dr.read_regular_file(workdir / "NOTES_AGENT.md", limit=200_000)
    if notes:
        (dest / "NOTES_AGENT.md").write_text(notes)
    (dest / "meta.json").write_text(json.dumps(meta, indent=1))
    return src


def run_episode(
    env: str, rollout: int, kit_dir: Path, args: argparse.Namespace, backend: cc.Backend
) -> tuple[dict, dict]:
    """One episode in a folder with a random name, moved to work/<arm>/<env>-r<k> afterwards."""
    root = args.out / "work" / args.arm
    root.mkdir(parents=True, exist_ok=True)
    workdir = (root / f"ep-{secrets.token_hex(6)}").resolve()
    final = (root / f"{env}-r{rollout}").resolve()
    if final.exists():
        isolation.clear(final)
    try:
        return _episode(env, rollout, kit_dir, workdir, args, backend)
    finally:
        if workdir.exists():
            workdir.rename(final)


def _episode(
    env: str,
    rollout: int,
    kit_dir: Path,
    workdir: Path,
    args: argparse.Namespace,
    backend: cc.Backend,
) -> tuple[dict, dict]:
    shutil.copytree(kit_dir, workdir)
    text = prompt(kit_dir)
    config_dir = backend.prepare(workdir.name)
    started = time.monotonic()
    try:
        events, code = backend.run(
            workdir, args.model, args.effort, args.max_usd, config_dir, task_prompt=text
        )
    finally:
        backend.release(config_dir)
    seconds = time.monotonic() - started
    turns = backend.turns(events)
    ended = backend.summary(events, code)
    if ended.get("login_lost"):
        raise cc.LoginLost(ended["stop"])
    if ended["throttled"] and turns == 0:
        raise cc.Throttled(ended["stop"])
    interrupted = provider_interrupted(events)
    label = label_of(args)
    dest = agents_dir() / label / env / f"r{rollout}"
    meta = dict(
        model=args.model,
        effort=args.effort,
        scaffold=backend.name,
        provider=PROVIDER.get(backend.name, ""),
        tier=TIER,
        arm=args.arm,
        finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    src = save_solution(workdir, dest, meta)
    edited = bool(src) and src != (kit_dir / "solution.py").read_text()
    row = dict(
        task=task_name(env),
        env=env,
        tier=TIER,
        rollout=rollout,
        model=args.model,
        effort=args.effort,
        label=label,
        scaffold=backend.name,
        outcome="error" if interrupted else ("submitted" if edited else "no_solution"),
        provider_interrupted=interrupted,
        stop=ended["stop"],
        turns=turns,
        tokens_in=ended["tokens_in"],
        tokens_out=ended["tokens_out"],
        cost_usd=ended["cost_usd"],
        seconds=round(seconds),
        solution_bytes=len(src),
        solution_path=os.path.relpath(dest, paths.REPO_ROOT),
    )
    trajectory = dict(
        task=row["task"],
        rollout=rollout,
        model=args.model,
        effort=args.effort,
        scaffold=backend.name,
        trace_kind=backend.trace_kind,
        prompt=text,
        events=events,
        solution_src=src,
        workdir=str(workdir),
    )
    return row, trajectory


def safe_episode(
    env: str, rollout: int, kit_dir: Path, args: argparse.Namespace, backend: cc.Backend
) -> tuple[dict, dict]:
    """One episode, tried again after a wait while the subscription is at its limit; any other
    failure lands in the table as an error row (decoder_run_cc.safe_episode's policy)."""
    reason = "Throttled: gave up"
    for _ in range(cc.THROTTLE_TRIES):
        try:
            return run_episode(env, rollout, kit_dir, args, backend)
        except cc.Throttled as exc:
            print(f"{env} r{rollout}: {exc}; next try in {cc.THROTTLE_WAIT} s", flush=True)
            time.sleep(cc.THROTTLE_WAIT)
        except cc.LoginLost:
            raise
        except Exception as exc:  # noqa: BLE001 - every failure must land in the table
            where = traceback.format_exc(limit=-3).strip().splitlines()[-6:]
            reason = (f"{type(exc).__name__}: {exc} | " + " / ".join(where))[:1500]
            break
    row = dict(
        task=task_name(env),
        env=env,
        tier=TIER,
        rollout=rollout,
        model=args.model,
        effort=args.effort,
        label=label_of(args),
        scaffold=backend.name,
        outcome="error",
        provider_interrupted=False,
        stop="error",
        turns=0,
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        seconds=0,
        solution_bytes=0,
        solution_path="",
        error=reason,
    )
    return row, dict(
        task=row["task"], rollout=rollout, error=reason, trace_kind=backend.trace_kind, events=[]
    )


def backend_for(name: str) -> cc.Backend:
    if name == "claude-code":
        return cc.CLAUDE_CODE
    from qec_drift.agent import decoder_run_codex  # noqa: PLC0415 (only for Codex arms)

    return decoder_run_codex.BACKEND


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--backend", choices=sorted(PROVIDER), default="claude-code")
    ap.add_argument("--model", required=True, help="a full model id, e.g. claude-opus-5-5")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--label", default=None, help="solution folder label (default: model)")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--envs", default="", help="comma list (default: all three)")
    ap.add_argument("--max-usd", type=float, default=cc.EPISODE_USD, help="per episode")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--out", type=Path, default=paths.DATA / "mismatch_hard")
    return ap.parse_args(argv)


def run_arm(args: argparse.Namespace, backend: cc.Backend | None = None) -> pd.DataFrame:
    """Every (env, rollout) episode of one arm. Rollouts already submitted in the arm's table
    are skipped, so an arm resumes after a throttle exit or a lost login."""
    backend = backend or backend_for(args.backend)
    cc.EPISODE_TAG = args.arm  # the state directories' prefix: two arms never share one
    backend.check()
    envs = [e for e in args.envs.split(",") if e] or env_names()
    args.out.mkdir(parents=True, exist_ok=True)
    kits = build_kits(args.out, envs)
    table = args.out / f"{args.arm}.parquet"
    rows = pd.read_parquet(table).to_dict("records") if table.exists() else []
    done = {(r["env"], r["rollout"]) for r in rows if r["outcome"] == "submitted"}
    jobs = [(e, k) for k in range(args.rollouts) for e in envs if (e, k) not in done]
    traj_path = paths.trajectories_dir() / args.out.name / f"mismatch_{args.arm}.jsonl.gz"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    with traj_path.open("ab") as trace, ThreadPoolExecutor(args.concurrency) as pool:
        for start in range(0, len(jobs), args.concurrency):
            batch = jobs[start : start + args.concurrency]
            try:
                results = list(
                    pool.map(lambda j: safe_episode(j[0], j[1], kits[j[0]], args, backend), batch)
                )
            except cc.LoginLost as exc:
                print(f"login lost: {exc}; the arm stops here", flush=True)
                if rows:
                    attempts.write_table(table, rows)
                sys.exit(cc.EXIT_LOGIN_LOST)
            for row, trajectory in results:
                saved = attempts.append_attempt(trace, row, trajectory, args.arm)
                key = (row["env"], row["rollout"])
                rows = [r for r in rows if (r["env"], r["rollout"]) != key] + [saved]
            attempts.write_table(table, rows)
            if any(row.get("provider_interrupted") for row, _ in results):
                sys.exit(cc.EXIT_THROTTLED)
    df = pd.DataFrame(rows)
    if not df.empty:
        cols = ["env", "rollout", "outcome", "stop", "turns", "seconds", "solution_bytes"]
        print(df[cols].to_string(index=False))
    print(
        f"solutions in {os.path.relpath(agents_dir(), paths.REPO_ROOT)}/{label_of(args)}/; grade "
        "them with: uv run python mismatch/experiments/run.py --tier hard --seeds 5"
    )
    return df


def main() -> None:
    run_arm(parse_args())


if __name__ == "__main__":
    main()
