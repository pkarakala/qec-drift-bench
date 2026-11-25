# qec_drift_d3: is the decoder's noise model still right?

You run a distance-3 rotated surface code memory experiment (Z basis, 3 rounds, 17 physical
qubits, CX gates). Your decoder was built from a nominal noise model. Earlier you took labeled
shots; now you have a new batch of unlabeled shots. Between the two, or during the new batch, the
device may have changed. Decide whether it did, say what changed, and ship a noise model (a
detector error model) for the device as it is at the end of the new batch. The grader decodes
your model with PyMatching, so the job is to repair the model for a fixed decoder.

Open question. Can an agent tell from syndrome statistics alone when its decoder's model has gone
stale, and say what changed, when most changes of realistic size barely move a matching decoder's
error rate at this distance? Estimating detector error models from syndromes under drift is an
active topic (Bhardwaj et al. 2025, Takou et al. 2026); this task asks for the decision and the
diagnosis, not only the refit.

Implement in `sandbox/solution.py`:

```python
def solve(data: dict) -> dict: ...
```

Run `python evaluate.py` to score it (`--quick` for a 4-instance smoke test). It prints one number.
`solve` runs in a separate process that sees only `data`.

## What `data` holds

| key | type | meaning |
|---|---|---|
| `circuit` | str | Stim circuit of the nominal model (flattened), uniform noise `p_nominal` on every channel |
| `p_nominal` | float | 2e-3 |
| `distance`, `rounds` | int | 3, 3 |
| `shot_period_us` | float | 100 us between shots (so a batch lasts 5 s) |
| `qubit_coords` | dict[int, tuple] | Stim qubit index to (x, y) |
| `data_qubits`, `measure_qubits` | list[int] | 9 data, 8 measure qubits |
| `gate_pairs` | list[tuple[int, int]] | every pair that shares a CX gate |
| `detector_coords` | dict | Stim detector coordinates (x, y, round) |
| `pre_detectors` | bool array (50000, 24) | labeled shots taken before any change |
| `pre_observables` | bool array (50000,) | logical observable flips of those shots |
| `post_detectors` | bool array (50000, 24) | new shots, unlabeled, in time order |

The nominal model is never exactly right: each qubit has its own error factor (log-normal around
1, sigma 0.3; an assumption) that the nominal model leaves out, the same before and after any
change. The pre shots show it.

## What `solve` returns

```python
{
  "dem": str,             # Stim detector error model: 24 detectors, 1 observable, every p in (0, 0.5]
  "diagnosis": {
    "changed": bool,      # did the device differ from the pre batch at any time in the new batch?
    "kind": "none" | "hot_qubit" | "hot_coupler" | "burst" | "tls_switching",
    "qubit": int,         # hot_qubit, tls_switching
    "pair": [int, int],   # hot_coupler
    "factor": float,      # hot_qubit / hot_coupler / tls_switching (its hot state): multiplier
    "onset_shot": int,    # index in post_detectors where the change began (0: before the batch)
    "decay_shots": float, # burst: exponential decay constant, in shots
    "dwell_shots": float  # tls_switching: mean time between switches, in shots
  }
}
```

## What can happen (hidden, one per instance)

Each scored set has 20 instances, shuffled: 10 controls (`none`), 2 `hot_qubit` from before the
batch, 2 `hot_qubit` that switch on inside it, 2 `hot_coupler`, 2 `burst`, 2 `tls_switching`.

- `hot_qubit`: every noise channel touching one qubit (uniform choice) is multiplied by a factor,
  log-uniform in 2 to 10. Sources: T1 of single qubits fluctuates by more than a factor 2
  (Burnett et al. 2019) and can change by up to an order of magnitude when a two-level-system
  defect drifts into resonance (Klimov et al. 2018). Mapping that to one multiplier on every
  channel is an assumption: a T1 defect raises idle and measurement error most.
- `hot_qubit` with an onset inside the batch: onset uniform in shots 10,000 to 40,000. Defects
  can be rearranged by radiation impacts; from the counts in Thorbeck et al. 2023 (34 of 656 jumps
  coincided with impacts, over about 11 h) we estimate about 3 such events per hour (our own
  estimate). The task conditions on the change falling inside the batch.
- `hot_coupler`: the CX gate noise on one pair (uniform choice) is multiplied by a factor,
  log-uniform in 2 to 10 (assumption).
- `burst`: a correlated event in the regime of McEwen et al. 2022: events about every 10 s,
  error rates recovering over about 25 ms. Here the decay time is drawn per instance,
  log-uniform in 15 to 40 ms (150 to 400 shots; the spread is an assumption). The peak multiplier
  (2 to 10 at the centre) and the Gaussian spatial profile (centre anywhere on the patch, radius 2
  to 6 lattice units) are assumptions. Onset uniform in shots 5,000 to 35,000. Newer chips with
  gap engineering resist such events (McEwen et al. 2024; on Willow the analogous events come
  about once an hour and decay in 369 us, Google Quantum AI 2024). After a burst the device is
  back to its background.
- `tls_switching`: one qubit (uniform choice) is coupled to a two-level system that switches as a
  symmetric random telegraph: the qubit's channels jump between their background value and a
  hot factor (log-uniform 3 to 10) and back, with exponential dwell times of mean 0.3 to 1.5 s
  (3,000 to 15,000 shots, log-uniform). Berritta et al. 2026 (arXiv 2608.02086) measure TLS
  switching times of about 2.2 s on a transmon; our faster range keeps several switches inside
  one batch (assumption). Every scored draw shows at least 3 switches in the new batch and is
  hot 20 to 80 percent of the time. The shipped decoder faces the switching device, so its rate
  is the average over the hot and background states, half each.
- `none`: nothing changed.

Faults are drawn at these sizes with no selection. Most do not open a decoding gap at d = 3: on
exact computation, 30% of hot-qubit draws (40 draws) and 3% of coupler draws (30 draws) make
matching on the true model beat matching on the before-change model by 5% or more; bursts never
do, because the device has recovered. Those are graded on diagnosis only (below).

## Scoring (maximize, in [0, 1], mean over the 20 instances)

These rules are this environment's fixed rubric (named, weighted checks, each computed by
script; no LLM judge; listed in `docs/rubrics.md` of the repository).

Logical error rates (LER) are exact: the grader computes the full distribution of syndromes and
logical flips of the device at the end of the batch (no sampling noise; left-out tail below 1e-6).
Anchors: stale = PyMatching on the nominal model; informed = PyMatching on the true model (for
tls_switching, its time average). The informed anchor is not a ceiling: a different model can
beat it; fix is clipped at 1.

- Control: `harm` if `changed` is false, else 0.
- Change, `changed` false: 0.
- Change graded on fix: `0.3 + 0.35 * fix + 0.35 * detail`, with
  `fix = clip((stale - yours) / (stale - informed), 0, 1)`.
- Change graded on diagnosis only: `harm * (0.3 + 0.7 * detail)`.
- `harm` = 1 if your LER is within 1% of the worse anchor, falling linearly to 0 at 10% worse.
- Which rule applies is fixed by the true model before your answer is seen: fix if matching on the
  true model beats matching on the before-change model by at least 5%.
- `detail` in [0, 1] counts only if `kind` is right: kind 0.3; hot qubit: qubit 0.4, factor within
  1.25 times 0.3 (factor needs the right qubit); if the onset is inside the batch, qubit 0.26,
  factor 0.19, onset within 1,000 shots 0.25; coupler: pair 0.4, factor within 1.25 times 0.3;
  burst: onset within 25 shots 0.35, decay within 1.25 times 0.35; tls_switching: qubit 0.4,
  factor of the hot state within 1.25 times 0.15, dwell within 2 times 0.15.
- An exception, a timeout (10 minutes), an invalid model or a malformed return scores 0.

Baselines on the public set: trivial (nominal model, "no change") 0.50; reference (per-qubit
background fit, two-sample p_ij hypothesis search over hot qubits and gate pairs with a
permutation threshold, CUSUM onset, spatial matched filter for bursts) 0.95;
mean over 5 secret seeds 0.947 +/- 0.019.

## Hard tier (`python evaluate.py --tier hard`)

The same device, data, return format and scoring, with `data["tier"] == "hard"` and its own
public seed. Two things change.

- The device may now change in a way the kinds above do not describe. If it does, `changed` is
  true and `kind` is `"other"`; no other field is graded for it, and `detail` is 1 when the kind
  is right. Such changes are common in this tier. The rule for each instance (fix, diagnosis
  only) is fixed as above, from the true device.
- The listed kinds are drawn weak: hot-qubit and hot-coupler factors log-uniform in 1.2 to 2.5,
  the TLS hot state 1.3 to 3, the burst peak 1.3 to 4 (assumptions, on the low side of the cited
  ranges: Burnett et al. 2019 see T1 move by a factor 2 and more; gap-engineered chips see weaker
  bursts, McEwen et al. 2024).

A change is drawn again until the new batch carries at least 20 nats of evidence: the expected
log-likelihood ratio of `post_detectors` under the changed device against the before-change
device, computed exactly over all 2^24 syndromes (the shots after the change times the
Kullback-Leibler divergence of their syndrome distributions; summed over the time profile for a
burst and over the hot shots for a TLS). 20 nats is what a likelihood-ratio test needs to find a
change with 90 percent power at a 1 percent false-alarm rate after a Bonferroni correction over
2,000 hypotheses (kind, location, timing): log(2000 / 0.01) = 12.2 and 12.2 + 1.28 sqrt(2 x 20)
is about 20. So an ideal test could find every change. Every other rule above is unchanged.

## Limits

17 qubits, CPU only, Python with numpy, scipy, stim, pymatching. The reference takes about 13 s
per instance.

## Sources

Stim: Gidney 2021, arXiv 2103.02202. PyMatching: Higgott and Gidney 2023, arXiv 2303.15933.
Surface code: Fowler et al. 2012, arXiv 1208.0928. Pairwise detector statistics: Chen et al.
2021, arXiv 2102.06132; Spitz et al. 2018, arXiv 1712.02360. Drifting-noise estimation from
syndromes at d = 3: Bhardwaj et al. 2025, arXiv 2511.09491. Syndrome-only DEM fits: Takou et al.
2026, arXiv 2606.11496. Decoder-prior tuning gains of 3 to 11% at d = 5: Sivak et al. 2024, arXiv
2406.02700. T1 fluctuations: Burnett et al. 2019, arXiv 1901.04417; Klimov et al. 2018, arXiv
1809.01043. Impacts and defects: Thorbeck et al. 2023, arXiv 2210.04780. Bursts: McEwen et al.
2022, arXiv 2104.05219; McEwen et al. 2024, arXiv 2402.15644; Google Quantum AI 2024, arXiv
2408.13687. Assumptions (no source): p_nominal = 2e-3, background sigma 0.3, coupler factors,
burst size and shape, the decay spread, the 5% gap rule and every scoring weight.

Licence: this text CC BY 4.0; code Apache-2.0.
