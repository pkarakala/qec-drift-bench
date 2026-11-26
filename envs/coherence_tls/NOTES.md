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

(filled in from experiments/results after the paper-seed runs)
