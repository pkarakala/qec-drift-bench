"""Agent pilot driver (runs on the operator's machine; contains no grader).

usage: python3 run_pilot.py [--agents claude:claude-opus-5-5,agy:<slug>@high,codex:default]
                            [--parallel 3] [--timeout-min 45] [--rollouts 3] [--tier hard]

For each agent, environment and rollout, copies <env>/ into runs/<cli>-<model>/<env>/r<k>/ and
runs that CLI headless there with PROMPT.md. Keeps the transcript, solution.py and
NOTES_AGENT.md. Skips runs whose done.json shows a finished or timed-out episode, so it can resume; an
episode that exited with an error (expired login, usage limit) runs again. Delete a done.json
to rerun any episode. With --tier hard the kit comes from hard/<env>/ (make_kit.py hard --tier
hard) and the runs go to runs_hard/. Permissions per CLI:
  claude  acceptEdits, file tools plus `python`, `ls`, `head`, `wc` in Bash; no web, no subagents
  codex   `codex exec --sandbox workspace-write` (writes only in the run folder, no network)
  gemini  auto_edit approval mode; no web tools
  agy     Antigravity CLI headless (`agy -p`). It has no per-run permission flag: headless runs
          soft-deny anything not pre-approved, so allow only python/python3/ls/head/wc and writes in
          the run folder through permissions.allow in ~/.gemini/antigravity-cli/settings.json, and
          deny web tools there. Model as `agy:<slug>` or `agy:<slug>@<effort>`; list slugs with
          `agy models`. https://antigravity.google/docs/cli/headless
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENVS = ["readout_drift", "pulse_wrong_model", "qec_drift_d3"]
DEFAULT = "claude:claude-opus-5-5,claude:claude-sonnet-5-5,claude:claude-haiku-5-5"
CLAUDE_TOOLS = ("Read,Write,Edit,Glob,Grep,Bash(python:*),Bash(python3:*),Bash(ls:*),"
                "Bash(head:*),Bash(wc:*)")


def command(cli: str, model: str, prompt: str) -> list[str]:
    if cli == "claude":
        return ["claude", "-p", prompt, "--model", model, "--output-format", "stream-json",
                "--verbose", "--permission-mode", "acceptEdits", "--allowedTools", CLAUDE_TOOLS,
                "--disallowedTools", "WebFetch,WebSearch,Task"]
    if cli == "codex":
        cmd = ["codex", "exec", "--sandbox", "workspace-write", "--skip-git-repo-check", "--json"]
        return cmd + ([] if model == "default" else ["-m", model]) + [prompt]
    if cli == "gemini":
        cmd = ["gemini", "--approval-mode", "auto_edit", "--output-format", "stream-json",
               "--exclude-tools", "web_fetch,google_web_search"]
        return cmd + ([] if model == "default" else ["-m", model]) + ["-p", prompt]
    if cli == "agy":
        slug, _, effort = model.partition("@")
        cmd = ["agy", "--model", slug, "--output-format", "stream-json"]
        return cmd + (["--effort", effort] if effort else []) + ["-p", prompt]
    raise ValueError(cli)


def episode(cli, model, env, k, timeout_s, tier="standard"):
    kit, runs = (HERE, HERE / "runs") if tier == "standard" else (HERE / tier, HERE / f"runs_{tier}")
    d = runs / f"{cli}-{model.replace('@', '-')}" / env / f"r{k}"
    done = d / "done.json"
    if done.exists() and json.loads(done.read_text()).get("returncode") in (0, "timeout"):
        return  # an episode that failed to start (expired login, usage limit) runs again
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(kit / env, d)
    env_vars = dict(os.environ, PATH=f"{HERE / 'venv' / 'bin'}:{os.environ['PATH']}")
    t0 = time.time()
    with open(d / "transcript.jsonl", "w") as out, open(d / "stderr.txt", "w") as err:
        p = subprocess.Popen(command(cli, model, (d / "PROMPT.md").read_text()), cwd=d,
                             stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env_vars)
        try:
            rc = p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = "timeout"
    (d / "done.json").write_text(json.dumps({
        "cli": cli, "model": model, "env": env, "rollout": k, "tier": tier, "returncode": rc,
        "wall_s": round(time.time() - t0),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    print(cli, model, env, k, rc, round(time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default=DEFAULT, help="comma list of cli:model")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--timeout-min", type=int, default=45)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--tier", choices=("standard", "hard"), default="hard")
    a = ap.parse_args()
    agents = [tuple(x.split(":", 1)) for x in a.agents.split(",")]
    jobs = [(c, m, e, k) for k in range(a.rollouts) for c, m in agents for e in ENVS]
    with ThreadPoolExecutor(a.parallel) as pool:
        list(pool.map(lambda j: episode(*j, a.timeout_min * 60, a.tier), jobs))


if __name__ == "__main__":
    main()
