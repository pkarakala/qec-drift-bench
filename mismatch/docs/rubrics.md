# Grading rubrics

Each environment's score is one fixed, versioned rubric of named checks, the same for every instance, solver, seed and tier. Every check is computed by script from the hidden truth and the answer; there is no LLM judge. A **gate** must pass or the instance scores 0. A **weighted** check adds weight x value (value in [0, 1]); on every instance the weights that apply sum to 1. A **multiplier** scales the weighted sum. A **part** is reported only (pulse: the diagnosis is the product of its parts). Source: `<env>/eval/rubric.py`; this page is generated from it by `experiments/rubric_report.py`. Per-check results: `experiments/results/rubric.md` and `rubric_hard.md`.

## qec_drift_d3 (`qec_drift_d3/rubric-v1`)

| check | fails when | control | fix | diagnosis_only |
|---|---|---|---|---|
| `valid_answer` | crashed, timed out, missing or ill-typed fields, or a DEM that fails validation | gate | gate | gate |
| `change_call` | false alarm on a control, or a missed change (gate) | 1.0 | 0.3 | 0.3 |
| `fix_gap` | the returned decoder model does not close the decoding gap the change opened |  | 0.35 |  |
| `diag_kind` | wrong kind of change (for an unlisted change: a listed kind named instead of other) |  | 0.35 x share | 0.7 x share |
| `diag_location` | wrong qubit or gate pair |  | 0.35 x share | 0.7 x share |
| `diag_size` | factor outside the tolerance |  | 0.35 x share | 0.7 x share |
| `diag_timing` | onset, decay or dwell time outside the tolerance |  | 0.35 x share | 0.7 x share |
| `no_harm` | the shipped decoder is worse than both anchors (1% free, none at 10%) | multiplier |  | multiplier |

## readout_drift (`readout_drift/rubric-v1`)

| check | fails when | control | fix | diagnosis_only |
|---|---|---|---|---|
| `valid_answer` | crashed, timed out, too many or invalid components, or ill-typed diagnosis | gate | gate | gate |
| `change_call` | false alarm on a control, or a missed change (gate) | 0.15 | 0.15 | 0.15 |
| `fix_gap` | the discriminator does not close the error gap the change opened |  | 0.6 |  |
| `no_harm` | the discriminator is worse than the stale one (0.2 percentage points free, none at 1.2) | 0.6 |  | 0.6 |
| `kinds` | wrong set of changed families (Jaccard overlap; an unlisted change is "other") | 0.15 | 0.15 | 0.15 |
| `magnitudes` | a graded family's size outside its tolerance | 0.1 | 0.1 | 0.1 |

## pulse_wrong_model (`pulse_wrong_model/rubric-v1`)

| check | fails when | control | fix | diagnosis |
|---|---|---|---|---|
| `valid_answer` | crashed, timed out, wrong pulse length, over the amplitude cap, or ill-typed diagnosis | gate | gate | gate |
| `change_call` | false alarm on a control, or a missed change (gate) | 0.4 | gate | gate |
| `fix_gap` | the pulse does not close the infidelity gap (in decades) the change opened |  | 0.6 |  |
| `no_harm` | the pulse is worse than the design pulse (0.1 decade free, none at 1) | 0.6 |  | multiplier |
| `diagnosis` | product of the four parts below |  | 0.4 | 1.0 |
| `diag_scale` | drive scale off by more than 2 Cramer-Rao sigma (none at 4) |  | part | part |
| `diag_f01` | qubit frequency off by more than 2 sigma (none at 4) |  | part | part |
| `diag_compression` | compression off by more than 2 sigma (none at 4) |  | part | part |
| `diag_other` | wrong `other` flag (hard tier: an effect outside the model's form; always right in the standard tier) |  | part | part |

QEC diagnosis shares for the true kind (they sum to 1): hot qubit kind 0.3, qubit 0.4, factor 0.3 (with an onset inside the batch: qubit 0.26, factor 0.19, onset 0.25); coupler kind 0.3, pair 0.4, factor 0.3; burst kind 0.3, onset 0.35, decay 0.35; switching TLS kind 0.3, qubit 0.4, factor 0.15, dwell 0.15; unlisted change (hard tier) kind 1.
