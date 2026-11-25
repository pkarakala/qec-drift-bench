# Conventions shared by every environment

These environments follow the Agent's Infinite Game in Quantum task format
(agents-infinite-game.org/how-to-contribute.html). Each one is a self-contained folder.

## Layout

```
<env>/
  TASK.md              what the solver reads: story, data, interface, scoring, limits, sources
  evaluate.py          `python evaluate.py` scores sandbox/solution.py and prints one float
  eval/harness.py      hidden device, ground truth, baselines (stale, oracle), scoring
  eval/rubric.py       the scoring rubric: named, weighted checks (see docs/rubrics.md)
  eval/_trivial.py     no-effort baseline (trusts the nominal model, says "no change")
  eval/_reference.py   known-method baseline (the bar an agent should reach)
  sandbox/solution.py  starter file the solver edits (behaves like trivial, with TODOs)
  NOTES.md             predictions written before measuring, then measured scores and runtimes
```

## Interface

- The solution module defines `solve(data: dict) -> dict`. `data` holds only what a lab
  engineer would have (nominal model, calibration data, new data). No hidden truth.
- The grader runs `solve` through `eval/_isolate.py` (identical in every environment): a fresh
  `python -I` process in an empty temp folder, with only the solution's folder and site-packages
  on its path, a timeout, and JSON out. The solver returns data, never code (a DEM, Gaussian
  components, pulse samples); the grader applies it to hidden test data itself.
- `eval/harness.py` defines `evaluate(solution_path: str, quick: bool = False) -> dict` that returns
  `{"score": float, "instances": [per-instance breakdown]}` and a `__main__` that prints the score.
- `evaluate.py` takes `--solution PATH` (default `sandbox/solution.py`), `--quick` (smoke test),
  `--seed N` or env `MISMATCH_SEED` (secret instance set; without it, the fixed public set) and
  `--report PATH` (per-instance breakdown with the truth, for maintainers only). It prints one
  number, the score. The seed is passed on a pipe so the solver's process cannot see it.
- A solver exception, a timeout, or a malformed return scores 0 for that instance.

## Instances and scoring

- Instances are drawn in order from one master seed; every hidden parameter comes from
  `numpy.random.default_rng`, so a score is reproducible for a given seed. The public seed is a
  smoke test; scored runs use a secret seed and a sandbox that hides `eval/`.
- About half of the instances are controls where nothing changed. Each instance's scoring rule
  (fix, diagnosis only, control) is fixed at build time from the true parameters, never from the
  test sample.
- Every answer includes `changed: bool`. Fix credit is paid only when it is right; a false alarm on
  a control scores 0. A solver that always gives the same answer cannot beat doing nothing.
- Only diagnosis fields that the provided data can identify are graded, with tolerances tied to
  the achievable precision. Nothing the TASK.md states is graded.
- Per-instance score in [0, 1] = w_fix * fix + w_diag * diagnosis, weights stated in TASK.md.
- fix = gap closed: clip((m_stale - m_sol) / (m_stale - m_oracle), 0, 1), where m is the hidden
  error metric (lower is better), stale = what the nominal model gives, oracle = what the true
  model gives, all on the same hidden test data. The upper anchor is named for what it is and
  TASK.md says whether it is a ceiling. On a control instance, fix = 1 if the solution is no worse
  than stale within a stated tolerance, falling linearly to 0.
- The reference is the strongest method we know. Never weaken it or retune a task toward a target
  reference score; if known methods solve the task, say so.

## Grader checks

- `checks/` holds lazy and exploit solutions and `checks/expect.json` lists each with a kind
  (`lazy`, `leak` or `heuristic`) and why. `run_all.py` fails if any lazy or leak check beats
  trivial + 0.05 on the public seed or on fresh secret seeds. Add a check for every new shortcut found.
- Data ablation: `run_all.py --ablate` scores the reference on swapped data (MISMATCH_ABLATE=1,
  handled in `eval/_isolate.py`, invisible to the solver) and fails if it still beats trivial +
  0.05. A task must need its data.
- Total score = mean over instances. Direction: maximize.
- The scoring rules are a fixed, versioned rubric in `eval/rubric.py`: named checks (gates,
  weighted checks, multipliers) that the harness scores with, each computed by script from the
  truth and the answer (no LLM judge). Every report lists each instance's checks and the rubric
  version. Changing a check or a weight is a new version. `docs/rubrics.md` is generated from the
  rubric files by `experiments/rubric_report.py`, which also re-grades stored runs and fails if a
  stored score differs from the rubric's.

## Hard tiers

- `evaluate.py --tier hard` (or env `MISMATCH_TIER=hard`) scores a harder tier: same data, return
  format and scoring rules, `data["tier"] == "hard"`, its own public seed, and secret seeds salted
  so a tier never shares draws with another.
- A hard tier is open-world: the device can also change in a way TASK.md does not describe. The
  answer for it is "other". Only noticing it (and not blaming a listed kind) is graded, never its
  form, which TASK.md does not state.
- The reference is unchanged: the strongest method that trusts the stated model. A "ceiling"
  solution in `experiments/solutions/` is the reference told the form of the unlisted change
  (never its size). It is not a valid solver; it shows that the tier's room above the reference
  can be reached from the data.
- Each hard tier has a `lazy_always_other` check: always "changed" with "other" must still lose
  to trivial.

## Limits

- At most 28 qubits, simulator only, one CPU workstation, whole job well under 3 hours.
  Target: reference solution's full evaluation under 10 minutes on one core.
- Dependencies: numpy, scipy; stim and pymatching where a QEC circuit is needed. Python >= 3.10.

## Physics and citations

- Every physical number and model choice in TASK.md cites a source with an arXiv id or DOI.
- Mismatches must be realistic in size and timescale (see the realism review: stationary faults
  and fabrication-level parameter errors are realistic; ms-scale wandering is not).

## Style

- Files under 600 lines. No em dashes. Plain, short TASK.md (a solver must read it fast).
- Licence: task text CC BY 4.0, code Apache-2.0.
