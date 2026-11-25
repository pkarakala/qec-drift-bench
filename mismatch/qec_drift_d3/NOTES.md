# qec_drift_d3 notes

Rewritten 2026-11-14 after the independent review (findings numbered as in the review).

## 2026-11-14: switching two-level systems (tls_switching)

The rebuilt task was near-solved (reference 0.95) and every fault was a single step: on, or on
at one onset. Real TLS defects also switch: Berritta et al. 2026 (arXiv 2608.02086) track a
transmon's TLS telegraph with switching times of about 2.2 s. A decoder recalibrated on "the
qubit is hot now" can be wrong half the time.

- New family `tls_switching` (2 of 20; hot and onset drop from 3 to 2 each). One qubit's
  channels jump between background and a hot factor (log-uniform 3 to 10) as a symmetric random
  telegraph with exponential dwell times, mean 3,000 to 15,000 shots (0.3 to 1.5 s,
  log-uniform; faster than Berritta's 2.2 s so a 5 s batch holds several switches;
  assumption).
- Supported labels: draws are kept only with at least 3 switches in the batch and 20 to 80
  percent on-time, so the data can tell switching from one onset.
- Grading: the shipped decoder faces the switching device, so the exact table is the 50/50
  mixture of the hot and background tables (exact, no sampling). informed = matching on the
  time-averaged model (factor (1 + F) / 2). As with hot qubits, most draws open no 5 percent
  decoding gap at d = 3 and are graded on diagnosis. Detail: qubit 0.4, hot factor within 1.25x
  0.15, dwell within 2x 0.15.
- Reference: after the p_ij test names a hot qubit, a two-state Viterbi path over 250-shot
  windows of per-shot log-likelihood ratios (switch prior 0.001 per window); two or more
  transitions means telegraph. The factor is refitted on the "on" shots, dwell = shots /
  transitions, decoder = the time-averaged model. Spot test on 7 draws: 6 detected with every
  detail right; it missed a 3.25x draw (called a 2.0x stationary hot qubit). Onset, hot,
  coupler, burst and control instances were not misread as switching.
- New heuristic check `heuristic_no_tls.py` = the previous reference. On a TLS instance it
  reports a hot qubit: `changed` right, kind wrong, so 0.3 of 1.

Measured (public seed; run_all.py on 1 secret seed, exit 0):

| solution | public | secret |
|---|---|---|
| trivial | 0.500 | 0.500 |
| reference | 0.950 | 0.940 |
| heuristic_no_tls | 0.880 | 0.935 (seed 4242; reference 0.988 there) |
| lazy_always_changed | 0.475 | 0.455 |
| lazy_never_changed | 0.500 | 0.500 |
| qec_cheap (heuristic) | 0.705 | 0.677 |
| other checks | at most 0.19 | at most 0.19 |

The previous reference keeps full marks everywhere except the TLS instances: 0.3 and 0.3 on the
public set, 0.65 and 0.3 on seed 4242 (the 0.65 is a fix-graded 9.8x draw where its stationary
model still recovers some of the gap). The reference scores 1.0 on all four, including a 3.1x
draw. The reference's one public loss is unchanged (a weak
hot qubit it does not detect). Runtime: the reference takes about 5.5 min with a warm cache;
building the two extra TLS tables cold adds about 1 min per TLS instance.

## What changed and why

- Isolation (1). `solve` runs in a child process (`eval/_isolate.py`, shared by all three
  environments): empty working folder, `python -I`, only the solution's folder plus site-packages
  on its path, environment stripped, only `data` passed, 10 minute timeout. `evaluate.py` re-execs
  itself so a secret seed never appears in `/proc/<pid>/cmdline` or `environ`. The solver returns
  data only (a detector error model and a diagnosis); a `decode` callable is no longer accepted.
  stdout carries only the score; `--report PATH` writes the breakdown for maintainers. Scored runs
  take `--seed N` or `MISMATCH_SEED`; without one the fixed public set is the smoke test.
- Scoring rule (2). Fix credit only when `changed` is right, 0 for a false alarm on a control,
  10 controls in 20 instances. A solver that always gives the same yes/no answer can at most tie
  the trivial baseline. False-alarm and miss rates are in the report. Factor tolerance is 1.25x,
  the burst decay is drawn per instance (no longer printed), onset credit is given only where the
  onset is inside the batch, so a fixed guess earns nothing for free (6).
- Anchors (3, quick route). The grader decodes the returned model with PyMatching. stale =
  PyMatching on the nominal model, informed = PyMatching on the true model, renamed from "oracle"
  and documented as not a ceiling. The model is validated (parses, 24 detectors, 1 observable,
  every probability in (0, 0.5]); invalid scores 0.
- Exact scoring (5). Every logical error rate is computed from the exact joint distribution of
  (syndrome, logical flip) of the after-change device (`eval/exact.py`: two Walsh-Hadamard
  transforms over 2^25 outcomes, 8 s), summed over the likeliest syndromes until the left-out mass
  is under 1e-6. No sampling noise, no rule switch: the rule (control, fix, diagnosis only) is
  decided from the true model before the answer is seen. Tables are cached on disk between runs,
  encrypted with a key derived from the instance seed.
- No gap gate (10). Faults are drawn at their stated sizes (hot qubit and coupler log-uniform 2 to
  10x) and kept. `eval/survey.py` measures how many open a decoding gap of 5% (true-model matching
  against before-change-model matching): 12 of 40 hot-qubit draws (median gap 2.4%), 1 of 30
  coupler draws (median 0.0%). The rest are graded on diagnosis only, with a do-no-harm factor.
  This is the finding the old gate hid: realistic drift at d = 3 is mostly a diagnosis problem.
- Burst (8). One regime, McEwen 2022: about one event per 10 s, decay about 25 ms (drawn per
  instance, 15 to 40 ms = 150 to 400 shots). Size (2 to 10x) and a Gaussian spatial profile are
  marked as assumptions. The device after the burst is the background, so bursts are diagnosis
  only.
- Reference (6). Rewritten as the strongest method we know: per-qubit background by Gauss-Newton
  on all pairwise moments jointly (rms error 8% in log factor, against 30% for the old coordinate
  descent and 30% for a uniform background), a two-sample p_ij search over all 17 hot-qubit and
  all gate-pair hypotheses with a permutation-test threshold (100 relabelings, no tuned constant),
  CUSUM onset with a refit, and a spatial matched filter with a decay fit for bursts. Uniform
  background gives fix about 0 on every fault (a uniform scale does not change matching); the
  joint per-qubit fit helps, so it is kept. A syndrome-only DEM fit (Takou et al. 2026) was tried
  in another thread and needs 200k+ shots; not used.
- Physics and citations (10, 14, 16, 17). Thorbeck rate given as our own estimate from their
  counts, Burnett cited only for "more than a factor 2", CX not CZ, the uniform hot-qubit
  multiplier marked as an assumption, Bhardwaj 2025, Sivak 2024, McEwen 2024 and Takou 2026 added.
  All four new arXiv ids were checked to resolve to the papers named.
- Identifiability. Coupler and hot-qubit labels stay graded: the reference never confused them on
  48 dev faults (and a separate probe found every stationary label recoverable at 20 sigma or
  more). Morales-Hashemi et al. 2026 (arXiv 2610.03545) shows indistinguishable models exist for
  coherent noise; none of the Pauli families here is affected.
- Lazy battery (`checks/`, `checks/expect.json`). Ported from the review: `qec_frames.py`
  (stack leak), `qec_const.py`, `qec_cheap.py`; added `qec_import.py` (import leak by module
  name), `lazy_blind_scale.py` (all nominal probabilities x1.6, never reads shots, fixed
  "changed"), `lazy_always_changed.py`, `lazy_never_changed.py` (reference refit, never reports),
  `lazy_malformed.py`. Two of them load `eval/_reference.py` by path, a shortcut for the probe.

## Measured

Public set, then mean +/- standard error over 5 secret seeds (20 instances each). Wall time is
the median per full run on a shared 4-core machine, with the per-instance exact tables already
cached; the first run of a seed also builds them (about 9 more minutes, see trivial).

| solution | public | 5 secret seeds | wall time |
|---|---|---|---|
| eval/_trivial.py | 0.500 | 0.500 +/- 0.000 (n=5) | 582.5 s |
| eval/_reference.py | 0.950 | 0.947 +/- 0.019 (n=5) | 480.5 s |
| sandbox/solution.py | 0.500 | 0.500 +/- 0.000 (n=5) | 19.5 s |
| checks/qec_frames | 0.000 | 0.000 +/- 0.000 (n=5) | 18.0 s |
| checks/qec_import | 0.000 | 0.000 +/- 0.000 (n=5) | 20.0 s |
| checks/qec_const | 0.208 | 0.206 +/- 0.006 (n=5) | 19.0 s |
| checks/qec_cheap | 0.759 | 0.667 +/- 0.025 (n=5) | 18.0 s |
| checks/lazy_blind_scale | 0.224 | 0.201 +/- 0.004 (n=5) | 170.0 s |
| checks/lazy_always_changed | 0.475 | 0.465 +/- 0.011 (n=5) | 478.5 s |
| checks/lazy_never_changed | 0.500 | 0.500 +/- 0.000 (n=5) | 496.5 s |
| checks/lazy_malformed | 0.000 | 0.000 +/- 0.000 (n=5) | 21.0 s |

Reference over the 6 sets (120 instances): false alarms 1 of 60 controls; misses 3 of 60 changes
(a 2.3x hot qubit on qubit 1, a 2.6x onset on qubit 19, a weak 3.1x burst); one 2.1x onset called a
coupler next to it. 14 of 60 changes were graded on fix (6 hot, 7 onset, 1 coupler); the reference
reached fix 0.62 and 0.79 on two of them and 0.98 or more on the rest. Burst decay estimates run low on weak
bursts (0.5 to 0.65 of the truth on 4 of 12). Mean score by family: control 0.98, hot 0.94,
onset 0.90, coupler 0.95, burst 0.86.

Checks: both leaks score 0 (the child process has neither the grader's frames nor eval/ on its
path). Every lazy check is at or below trivial (rule: <= trivial + 0.05). `qec_cheap.py` scores
0.67 on secret seeds: it really detects changes from raw detector rates with no false alarms and
localises hot qubits, so it is listed as "heuristic", not lazy; the scoring pays for noticing.

Runtime: reference full evaluation 17 min cold (tables built), about 8 min warm; solve is
13 to 15 s per instance. Exact tables: 8 s per instance, plus 6 s per decoder over the about 2M
syndromes kept. Above the 10 minute target when cold.

## Remaining limits

- A solver that reads `eval/` from disk can regenerate the public smoke-test set; scored runs must
  use a secret seed and a sandbox that hides `eval/`. Same user, so `/proc` and the cache folder
  are readable; the cache is useless without the seed.
- The task is close to solved by known methods (reference 0.95 public). Its misses are 2x faults on
  edge qubits, weak bursts, and decay estimates of weak bursts. The honest next step for headroom
  is physics (continuous drift as in Bhardwaj 2025, d = 5), not a weaker reference.
- At d = 3 the fix part matters on few instances (about 30% of hot qubits, almost no couplers).
- The hot-qubit model (every channel times one factor) and all burst shape parameters are
  assumptions. Background spread sigma 0.3 is stated in TASK.md and used as a prior by the reference.
- The upper anchor is not a ceiling: an exact decoder of the true model is 13 to 19% better than
  matching (review finding 3); this only matters through the fix clip at 1.

## 2026-11-18: hard tier (`--tier hard`)

Why. The standard tier is close to solved (reference 0.95). A first hard tier with only weak
listed faults (factors 1.2 to 2.5, kept only when the new batch carries at least 20 nats of
evidence) still left the reference at 0.95 on the public set: the evidence guard makes every
weak fault findable, and the reference finds them. Shrinking the faults further would only make
the task unfair. What a stated-model method cannot do is notice a change the model has no term
for, so the hard tier adds one.

Design. The unlisted change is a correlated two-qubit depolarization (`DEPOLARIZE2`) on one pair
of data qubits within 2.9 lattice units (data qubits never share a gate), applied in every round
after the data-qubit idle noise: unmodelled crosstalk (Sarovar et al. 2020, arXiv 1908.09855).
Its probability is log-uniform in 2e-3 to 1e-2 (one to five times the nominal error; an
assumption). It is 5 of the 10 changed instances per set ("common", as TASK.md says), the listed
kinds are drawn weak, and the same 20-nat evidence guard applies to every changed draw.
Diagnosis of the unlisted change is kind only ("other"): TASK.md does not state its form, so
nothing about its form is graded. Secret seeds are salted so the hard tier shares no draws with
the standard tier. No prediction was written down before these runs; the design was iterated on
the public set and on early held-out runs, and the numbers below come from one run of the frozen
design.

Measured (public set and the 5 paper seeds, 95% cluster-bootstrap intervals,
`experiments/results/summary_hard.md`): trivial 0.50, reference 0.73 [0.65, 0.81], ceiling
(`experiments/solutions/qec_drift_d3/genie_corr.py`, the reference told that a pair of data
qubits may have a correlated error, never its size) 0.87 [0.80, 0.94]. Per family, reference
then ceiling: correlated 0.24 then 0.84, onset 0.64 then 0.50, TLS 0.38 then 0.38, burst 0.75,
hot 0.80, coupler 0.86, controls 1.00 for both. The reference misses some correlated changes and
blames a listed kind for most of the rest (rubric: `diag_kind` costs it 0.148 per instance).
Weak TLS and onset faults remain hard for both; that is room for an agent too. Lazy and leak
checks: `lazy_always_other` 0.33, `lazy_always_changed` 0.27, `qec_const` 0.17, all below
trivial; reference on swapped data 0.37.

Honest limits. The ceiling is not an upper bound: it is one method that knows the form, and an
agent that infers the form from the pairwise detector statistics could do better. The
correlated-error size range is an assumption; the 20-nat guard is what keeps every draw findable
in principle.
