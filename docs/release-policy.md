# Release policy for the data-out tasks (release-policy-v1)

A release is a frozen set of tasks that every leaderboard number refers to. This file says what
a release freezes, how secret seeds are drawn and retired, how contamination is limited, and
what the sealed harness that gate A6 needs must do. It complements
[task-standard.md](task-standard.md), which decides whether a task may enter a release.

## What a release freezes

- **Version.** `version` in `envs/config.json` (for example `envs-v1-2026-11-26`). Every report,
  leaderboard row and paper table names it.
- **Scored files.** Per task, the sha256 of TASK.md, evaluate.py, eval/*.py and
  checks/expect.json, and the rubric version. `uv run python scripts/envs_manifest.py` writes
  them to `data/envs_hard/manifest.json`; `--check` fails if any scored file has changed.
- **Seeds.** The public smoke-test seed, the paper seeds (public, a reproduction set) and the
  host's secret leaderboard seeds (below).
- **Selection.** The S1 decisions (`data/envs_hard/selection.json`) and the acceptance cards
  (`acceptance.json`) the release was built from.

A change to any scored file, to a seed list or to the standard is a new version. A score from one
version is never mixed with a score from another; the leaderboard shows the version it was built
from. Fixes to documentation that is not scored (NOTES.md, this file) do not change the version.

## Changelog

`docs/envs-changelog.md` gets one row per version, newest first: date, version, what changed, and
which tasks must be re-scored. A task's NOTES.md keeps the reasoning; the changelog keeps the
fact.

## Seeds

| set | who holds it | use | lifetime |
|---|---|---|---|
| public | the repository | smoke tests, dev data for agent runs (held-out families left out) | forever |
| paper | the repository | reproducing published tables | forever; never used for a leaderboard |
| leaderboard | the host, never committed | the leaderboard | one version |

Rotation: each version's leaderboard seeds are drawn fresh by the host with
`secrets.randbelow(2**31)`, before any model is run on that version. When a version is
superseded, its leaderboard seeds are published next to its results, so every past number can be
reproduced; they are then never used again for scoring. A model scored on a version keeps that
version's number; it is re-run, not carried over, on the next version.

## Contamination controls

- **Canary.** Every TASK.md ends with the canary line of task-standard.md, so the task text can be
  filtered out of training corpora. Each release also asks every scored model to complete the
  canary GUID from its prefix; a model that completes it is reported as possibly trained on the
  tasks, next to its score.
- **Hidden truth.** The truth is generated from a seed at grading time; no answer file exists to
  memorise. Agent runs see only the data of the public seed, without the held-out fault families
  (task-standard.md, S1).
- **Held-out families.** Each hard tier keeps at least one fault family out of the dev data, so a
  solution must notice a break in the model it has not seen.
- **Rotation.** New leaderboard seeds per version (above), so a solution tuned on published
  instances meets new ones.

## Sealed harness (gate A6)

In local mode the grader and the solution run on one machine, and `agent/envs_probe.py` reports
which channels are open. A release needs sealed mode, in which every channel the probe checks is
closed. The design:

1. **Two containers.** The grader container holds the task, the seeds and the truth. The solver
   container holds only the submitted solution folder, the Python runtime and the task's pinned
   dependencies.
2. **Data in, answer out.** For each instance the grader sends the instance's data as JSON on the
   solver's stdin and reads the answer as JSON from its stdout, as `eval/_isolate.py` does today
   inside one machine. Nothing else crosses: no shared volume, no environment variables, no
   argv beyond the solution's path.
3. **No network.** The solver container has no network interface other than loopback.
4. **Fresh per instance.** Each instance starts from the same read-only image, so no state carries
   from one instance to the next and no instance sees another's data or answer.
5. **Limits.** CPU, memory and the task's solver timeout are enforced by the container runtime;
   a solver that exceeds them fails the instance's valid-answer gate.
6. **Proof.** The probe runs as a solution in sealed mode and must report every channel closed;
   its report is saved with the release.

The gate stays pending until a host runs this; no release is published while it is pending.
