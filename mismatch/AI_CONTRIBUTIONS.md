# AI contributions log

Many venues now ask authors to disclose how generative AI was used (for example the NeurIPS,
ICML and ICLR LLM policies and the ACL Responsible NLP checklist). This file is the record we
draw that disclosure from. This copy covers the mismatch track (`mismatch/` and
`src/qec_drift/mismatch/`); it starts in the private repository `pkarakala/mismatch-envs`.
Pranav Reddy directed every piece of work below, chose what to build,
reviewed and merged every pull request, and is responsible for the content. Git commits carry
human authors only; this file is where AI work is attributed.

Each entry gives the date (UTC), the model and its reasoning-effort setting as reported by the
session, the tool, the role, and what was produced. "Claude Code session" means Anthropic's
Claude Code running in a cloud container with access to this repository. No paid model
evaluation of the benchmark itself has been run: the agent pilot used the author's own
subscriptions (Claude Max, Google AI Pro via the Antigravity CLI), with no pay-per-token API.

| date (UTC) | model (effort) | tool | role | contribution |
|---|---|---|---|---|
| 2026-11-11 | Claude Opus 5.5 (medium) | Claude Code session | code and text generation | Built the three environments (qec_drift_d3, readout_drift, pulse_wrong_model): harnesses, trivial and reference solutions, TASK.md, NOTES.md. Thread "Build Infinite Game style mismatch environments". |
| 2026-11-11 | Claude Opus 5.5 (medium) | Claude Code session | literature search and drafting | Related-work map, paper outline and an adversarial review of the QEC-Drift realism notes. Thread "Map related work and draft paper outline". |
| 2026-11-13 | Claude Fable 5.2 (max) | Claude Code session | independent review | Review of the three environments before release: 18 findings (grader leaks, lazy solutions that beat trivial, weak references), with exploit scripts and probes. Thread "Review mismatch environments before sharing". |
| 2026-11-14 | Claude Opus 5.5 (medium) | Claude Code session | literature search and analysis | State-of-the-art survey of 2025 to 2026 papers; identifiability probes and a syndrome-only DEM estimator prototype for qec_drift_d3. Thread "Survey state-of-the-art drift and agent-eval research". |
| 2026-11-14 | Claude Opus 5.5 (high) | Claude Code session | code generation | PRs #1 to #5: grader hardening and the lazy and leak check battery, data-ablation check, readout label floor, pulse drive compression family, QEC switching-TLS family. Thread "Evaluate and extend QEC-Drift benchmark". |
| 2026-11-15 | Claude Opus 5.5 (high) | Claude Code session | code generation, analysis, drafting | Reproducible experiment pipeline (`experiments/`), component ablations, statistics, tables and figures, the Kaggle v1 scope note, and the workshop paper draft (kept outside this repo). Thread "Make the drift-benchmark research paper-ready". |
| 2026-11-15 | Claude Opus 5.5, Sonnet 5.5, Haiku 5.5 (default effort) | Claude Code headless (`claude -p`) on the author's laptop | benchmark subject | Agent pilot: 3 models x 3 tasks x 3 rollouts, 45-minute limit, file tools and `python` only, no web. Their solutions are graded in `experiments/`; they wrote no code in this repository. |
| 2026-11-15 | Gemini 3.8 Flash (high) | Antigravity CLI headless (`agy -p`) on the author's laptop | benchmark subject | Agent pilot, stopped by the plan's quota after 1 complete episode (pulse r0) and 1 partial one; web search could not be disabled in this tool, and a scan found no web calls. Not reported in the paper's tables. |
| 2026-11-16 to 2026-11-17 | Claude Opus 5.5 (high) | Claude Code session (same thread) and Claude Code on the author's laptop via Remote Control | code generation, analysis, review | Agent aggregation, scoring runs, re-audit of the 2026-11-13 review findings, results filled into the paper. |
| 2026-11-17 | Claude Opus 5.5 (high) | Claude Code session | code generation | OpenAI-style tool-loop adapter (`experiments/tool_loop/`) in QEC-Drift's harness shape, with a one-episode-per-instance mode graded through a replay solution, the shared results table (`experiments/shared_table.py`) and their tests; no model was run. Thread "Prepare drift envs for the Logan sync". |
| 2026-11-18 | Claude Opus 5.5 (high) | Claude Code session | code generation, analysis | Hard tiers (`--tier hard`) for all three tasks: an unlisted change per task (readout IQ stretch, pulse drive strength on the higher transitions with a 6 ns gate, QEC correlated data-qubit error), weak listed QEC faults with an exact evidence guard, told-the-form ceiling solutions, `lazy_always_other` checks, the hard-tier runs on the public and 5 paper seeds, and a fixed, versioned scoring rubric per environment (`eval/rubric.py`, `docs/rubrics.md`, per-check breakdowns from `experiments/rubric_report.py`); no model was run. |
| 2026-11-25 | Claude Opus 5.5 (high) | Claude Code session | code generation, analysis | Moved the hard tier into this repository as the mismatch track: the 75% selection rule (`experiments/selection.py`), the hard-tier pilot kit, the runner on QEC-Drift's Claude Code and Codex scaffolds (`src/qec_drift/mismatch/run_cc.py`, with a `task_prompt` argument added to both scaffolds) and its tests; no model was run by this session. Thread "Move hard drift tasks into Logan's repo". |

## What the AI did not do

- It did not run any pay-per-token model. Baseline scores come from scripted solutions
  (trivial, reference, heuristics, ablations, lazy and leak checks); agent scores come from the
  pilot rows above, whose solutions were written by the models under test, not edited by us.
- It did not contact anyone or submit anything.
- Physics numbers in TASK.md files cite papers. Many sources were read through a text-fetch tool;
  numbers quoted in the paper must be checked against the PDFs by a human author before
  submission (see the checklist in the paper draft's README).

## How to add an entry

Add one row per working session that changes this repository or the paper. Use the model name
and effort the tool reports, not a guess. Keep the role to one of: code generation, text
drafting, literature search, analysis, review.
