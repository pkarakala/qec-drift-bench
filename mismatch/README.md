# Mismatch track (hard tier)

Three environments ask one question in three parts of a quantum device. Is the model of the
device still right? If not, what changed, and what is the fix? The agent returns data, not code.
The data is a detector error model (QEC), Gaussian readout components (readout) or 24 pulse
samples (pulse). Each environment's own grader applies that data to hidden test data. The
grader never runs agent code next to the hidden data. This is why the track avoids the
isolation blocker in [docs/kaggle-benchmarks-port.md](../docs/kaggle-benchmarks-port.md).

The environments were built and audited in the private repository `pkarakala/mismatch-envs`
(Pranav Reddy). Only the hard tier is in this track. The standard tier fails the selection rule
below.

| environment | what the agent gets | listed changes | unlisted change (report it as `other`) | instances per seed |
|---|---|---|---|---|
| `qec_drift_d3` | d = 3 surface-code circuit, pre-change and post-change syndromes | hot qubit, onset, coupler, burst, switching TLS (all weak, kept only with at least 20 nats of evidence) | correlated error on two data qubits | 20 (10 controls) |
| `readout_drift` | labeled calibration shots, unlabeled IQ shots in 8 batches | six readout drift families | IQ-plane stretch of the readout chain | 30 (15 controls) |
| `pulse_wrong_model` | Rabi, Ramsey and DRAG-check data, a 6 ns X-gate design | drive scale, frequency, compression | drive strength on the 1-2 and 2-3 transitions | 32 (16 controls) |

Each folder follows [CONVENTIONS.md](CONVENTIONS.md): `TASK.md` (what the solver reads),
`evaluate.py`, `eval/` (hidden device, truth, grader, rubric), `checks/` (cheat checks),
`sandbox/solution.py` (the starter), `NOTES.md` (predictions, then measurements).

## Selection: keep a task only if the best agent passes under 75%

A task is one environment at one tier. The pass rule is QEC-Drift's rule
(`agent/awareness.passed`) applied to each instance: the "changed" answer is right; on a changed
instance, at least half of the diagnosis credit; and where a fix is graded, at least half of the
gap closed (`experiments/shared_table.py`). The pass rate is the mean over instances, averaged
over rollouts, on the paper seeds. A task is kept when the best model's pass rate is below 0.75.
`experiments/selection.py` applies the rule and writes
[experiments/results/selection.md](experiments/results/selection.md). The 0.75 here is a
selection rule. It is not the 0.75 in `scoring/failure_modes_decoder.RELATIVE`, which is a
per-run pass rule of the decoder tasks.

Each cell is the pass rate / native score (trivial floor 0.50).

| task | status | reference (best known method) | Opus 5.5 | Sonnet 5.5 | Haiku 5.5 |
|---|---|---|---|---|---|
| qec_drift_d3, standard | excluded | 0.96 / 0.96 | 0.99 / 0.98 | 0.92 / 0.92 | 0.86 / 0.86 |
| readout_drift, standard | excluded | 0.93 / 0.93 | 0.95 / 0.95 | 0.74 / 0.73 | 0.85 / 0.85 |
| pulse_wrong_model, standard | excluded | 1.00 / 0.99 | 0.99 / 0.99 | 0.98 / 0.97 | 0.99 / 0.99 |
| qec_drift_d3, hard | included (pending agent runs) | 0.67 / 0.73 | not run | not run | not run |
| readout_drift, hard | included (pending agent runs) | 0.65 / 0.74 | not run | not run | not run |
| pulse_wrong_model, hard | included (pending agent runs) | 0.61 / 0.63 | not run | not run | not run |

Standard tier: Claude Code headless, 3 rollouts per model, 10 paper seeds (mismatch-envs, 2026-11-18).
Hard tier: public seed and 5 paper seeds. Until the hard-tier agent runs are in, a hard task is
kept on the evidence that the strongest known method passes under 0.75. A told-the-form
ceiling (the reference told the form of the unlisted change, never its size) reaches
0.87 / 0.92 / 0.93 native, so the tasks are solvable. A hard task where the best agent reaches
0.75 leaves the track.

## Rubric and cheat checks

- One fixed, versioned rubric per environment (`<env>/eval/rubric.py`, listed in
  [docs/rubrics.md](docs/rubrics.md)). Every check is a script over the hidden truth and the
  answer. There is no LLM judge. `experiments/rubric_report.py` re-grades every stored run and
  checks that the rubric total equals the stored score on every instance.
- Lazy and leak checks (`<env>/checks/`, listed in `checks/expect.json`): constant answers,
  "always changed", "always other", reading frames or files the solver should not see, and
  importing the grader. `python run_all.py --tier hard` fails when any of them scores above
  trivial + 0.05. On the hard tier every one scores at or below trivial.
- A data ablation: the reference on another instance's data must fall to or below trivial.
- The solver runs through `eval/_isolate.py`: a fresh `python -I` process in an empty temp
  folder, with a timeout and JSON out. The seed reaches the grader on a pipe, never in argv or
  the environment.

## Run it on this harness

```bash
# Episodes: the decoder tasks' Claude Code or Codex scaffold, same sandbox and limits
uv run python -m qec_drift.mismatch.run_cc --backend claude-code --model claude-opus-5-5 \
    --arm opus_mm_hard --rollouts 3
uv run python -m qec_drift.mismatch.run_cc --backend codex --model gpt-6-astra --effort max \
    --arm astra_mm_hard --rollouts 3
# Grading: every solution, the public seed and 5 paper seeds, through each env's evaluate.py
uv run python mismatch/experiments/run.py --tier hard --seeds 5 --jobs 4
uv run python mismatch/experiments/analyze.py --tier hard
uv run python mismatch/experiments/selection.py
# One table with the decoder tasks (QEC-Drift columns and pass rule)
uv run python mismatch/experiments/shared_table.py --qec-drift data/agent_launch2/<arm>.parquet \
    --qec-model <model> --qec-effort <effort> --qec-scaffold codex
```

The runner (`src/qec_drift/mismatch/run_cc.py`) uses one episode per environment and rollout.
The work folder holds the pilot kit (`experiments/agent_pilot/make_kit.py`): TASK.md, the
starter, the public instances' data without answers, and a shape checker. Only the prompt
differs from the decoder tasks. The harness copies `solution.py` to
`experiments/results/agent_solutions_hard/<model>/<env>/r<k>/`, and `run.py` grades it from
there. Grading runs agent code through `eval/_isolate.py`, not under `sandbox-exec`. Grade
agent solutions in a container or a throwaway VM, as the mismatch pilot did.

The scripts here default to the hard tier. The code still has a `--tier standard` path,
because the hard tier reuses its families, but the standard tier has no results here and is
not part of the track.

## Seeds

`experiments/config.json` holds the paper seeds, drawn once on 2026-11-15. They are public in
this repository, so they are a reproduction set, not a held-out split. A leaderboard split must
be drawn by the host and never committed.

## Provenance

Copied from `pkarakala/mismatch-envs` at branch `claude/project-thread-pb92h0`, commit
`0dcf2ef`. The environment folders are byte-identical. The changes are the hard-tier defaults
of the scripts above. Regenerating `summary_hard.md`, `rubric_hard.md`, `instances_hard.csv` and
`docs/rubrics.md` here gives the same bytes as in mismatch-envs. AI use is recorded in
[AI_CONTRIBUTIONS.md](AI_CONTRIBUTIONS.md).
