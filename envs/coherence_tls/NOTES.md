# coherence_tls: notes (version 1, 2026-11-26)

The fifth environment, built through the acceptance standard (docs/task-standard.md) and designed
from what the hard-tier pilots of 2026-11-26 showed. Its acceptance card is in
docs/envs-acceptance.md.

## Why this task

A tunable transmon relaxes fast near the frequency of any two-level-system defect, and those
defects move over hours (Klimov 2018, arXiv:1809.01043; Carroll 2022, arXiv:2105.15201). Labs
scan T1 against frequency and re-choose idle frequencies; it is a daily calibration step the
other environments do not cover (they work at a fixed frequency).

## Design choices

- Device model: a linear background rate plus one Lorentzian per TLS in the qubit frequency,
  with the coupling and decoherence ranges of Klimov 2018. One T1 point per frequency per scan
  (one delay), binomial shot noise, fixed readout error.
- The change call is defined by effect size: the rate a better allowed idle frequency saves,
  against a floor of 10% of the nominal rate at the old idle point; draw rules leave a gap
  (changed at 2 floors or more, controls at 0.5 floors or less). Every scan has wander (1 MHz
  of TLS frequency, 5% of coupling), so "a TLS moved a little" is not "changed".
- The solver returns data (one frequency and a diagnosis). The grader evaluates the true rate
  there, so no solver code touches the truth.
- Diagnosis tolerances are 2 to 4 Cramer-Rao sigma from the latest scan with every TLS
  parameter and the background free.
- Hard tier: one nominal TLS switches between two frequencies (a thermal fluctuator coupled to
  it; Klimov 2018 sees telegraphic jumps up to ~60 MHz at rates of 50 uHz to 5 mHz; Schlor 2019,
  arXiv:1901.05352). The rate that counts is the long-run mixture. The switch is visible in the
  history: the calibration scan (hour -1) shows one state and an earlier scan (hour -3 or -2)
  the other, which the stated model forbids (it allows changes only after the calibration).
  So the change is identifiable from the data, and a solution must notice it without an
  example (the held-out pilot protocol).

## Lessons from the four earlier hard tiers, applied here

- The selection rule (QEC-Drift pass rule) credits a combined instance (unlisted change plus a
  listed one) as diagnosed when the listed kind is right: diagnosis = (kinds + params) / 2 is
  0.75 with "other" missing. So combined instances do not test noticing. This tier keeps 8 pure
  unlisted instances, 4 combined and 4 listed among its 16 changed ones (flux had 6, 6 and 4).
- A generic refit can half-fix an unlisted change (flux): here the fix needs the mixture, and
  the diagnosis needs the word "other", which only a solution that noticed the break reports.

## Measured

Hard tier, 5 paper seeds (docs/envs-acceptance.md). Pass rate is the QEC-Drift rule per
instance; native score in brackets.

- trivial 0.50 (0.50); reference 0.73 (0.77); ceiling genie_switch 0.95 native.
- The reference passes every listed change (tls_moved, tls_new, both: 1.00) and most combined
  ones (0.80 to 0.90), and fails every pure switching instance (0.00): it never says "other".
- Every automatic gate passes except A6 (local mode; needs a sealed harness).

Prediction, written before the agent run (AGENTS.md rule 4): with half the instances controls
that agents get right, the pass rate is 0.5 + 0.5 x (changed pass rate). An agent that matches
the reference on listed and combined changes reaches 0.73, so this task clears the 75% bar only
if the best agent notices the switch in fewer than about 1 in 8 pure switching instances. We
expect Opus 5.5 at high effort to notice it more often than that (agents flagged the unlisted
flux echo in 42% of instances), so we predict the task fails S1 at about 0.78. If it does, the
next version makes the listed changes harder (smaller moves near the wander, overlapping
defects) rather than moving the bar.

Result (held-out pilot, Claude Code at xhigh effort, 2 rollouts each): Fable 5.2 0.74
[0.69, 0.79], Opus 5.5 0.74 [0.69, 0.79]. The prediction was wrong. Neither model noticed the
switching defect on any pure switching instance (0 of 80 each), so both land at the reference.
The task is kept, as a close call: its interval spans the bar. Both models pass every listed
change, so the next version should make the listed changes harder, not the unlisted one.
