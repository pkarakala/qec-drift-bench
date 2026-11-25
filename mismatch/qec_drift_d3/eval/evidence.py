"""How much evidence a new batch carries about a change, computed exactly. Licence: Apache-2.0.

`expected_llr` is the hard tier's draw rule: the expected log-likelihood ratio (nats) of the new
batch under the changed device against the before-change device, i.e. the number of affected
shots times the Kullback-Leibler divergence of their syndrome distributions (exact over all 2^24
syndromes, eval/exact.py). A change is drawn again until this is at least EVIDENCE_MIN, so an
ideal test that knew what to look for would find it (see NOTES.md, hard tier).

`realized_llr` is the same ratio on the batch actually sampled: the statistic of a genie test
that knows the alternative (fault, size, timing). It is for maintainers (experiments/genie.py),
never used in scoring.
"""

from __future__ import annotations

import math

import numpy as np
import stim

from exact import ND, joint_table

BURST_GRID = (1.0, 0.5, 0.25, 0.125, 0.0625)  # time-profile levels g at which tables are built


def syndrome_probs(c: stim.Circuit) -> np.ndarray:
    """P(s) for every syndrome s (bit d = detector d), observable summed out."""
    return joint_table(c.detector_error_model()).sum(1)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / np.maximum(q[m], 1e-300))))


def _burst_circuit(h, base, t, g):
    coords = base.get_final_qubit_coordinates()
    qf = {q: b * (1 + (t.amplitude - 1) * g * math.exp(
        -((x - t.centre[0]) ** 2 + (y - t.centre[1]) ** 2) / (2 * t.radius**2)))
        for q, b in t.background.items() for (x, y) in [coords[q]]}
    return h.scaled_circuit(base, qf)


def _burst_profile(h, t) -> np.ndarray:
    shots = np.arange(t.onset_shot, h.N_POST)
    return np.exp(-(shots - t.onset_shot) / t.decay_shots)


def expected_llr(h, base: stim.Circuit, t) -> float:
    """Expected log-likelihood ratio of the new batch, changed vs before-change device. `h` is
    the harness module (scaled_circuit, after_circuit, N_POST)."""
    if t.family == "none":
        return 0.0
    p_bg = syndrome_probs(h.scaled_circuit(base, t.background))
    if t.family == "burst":
        ks = [kl(syndrome_probs(_burst_circuit(h, base, t, g)), p_bg) for g in BURST_GRID]
        lg, lk = np.log(BURST_GRID[::-1]), np.log(np.maximum(ks[::-1], 1e-30))
        g = _burst_profile(h, t)
        x = np.log(g)
        # log-log interpolation inside the grid; below it KL ~ g^2 (second order in the change)
        logk = np.where(x >= lg[0], np.interp(x, lg, lk), lk[0] + 2.0 * (x - lg[0]))
        return float(np.exp(logk).sum())
    d = kl(syndrome_probs(h.after_circuit(base, t)), p_bg)
    if t.family == "tls_switching":
        edges = [0, *t.switches, h.N_POST]
        on = sum(edges[i + 1] - edges[i] for i in range(len(edges) - 1)
                 if (i % 2 == 0) == t.start_on)
        return float(on * d)
    return float((h.N_POST - t.onset_shot) * d)


def syndrome_index(det: np.ndarray) -> np.ndarray:
    return (np.asarray(det, dtype=np.int64) << np.arange(ND)).sum(1)


def realized_llr(h, base: stim.Circuit, t, post: np.ndarray) -> float:
    """log P(post | changed device, known timing) - log P(post | before-change device)."""
    if t.family == "none":
        return 0.0
    idx = syndrome_index(post)
    lb = np.log(np.maximum(syndrome_probs(h.scaled_circuit(base, t.background))[idx], 1e-300))
    if t.family == "burst":
        g = _burst_profile(h, t)
        grid = np.array((0.0, *BURST_GRID[::-1]))
        tab = np.stack([lb[t.onset_shot:]] + [
            np.log(np.maximum(syndrome_probs(_burst_circuit(h, base, t, gg))[idx[t.onset_shot:]],
                              1e-300)) for gg in BURST_GRID[::-1]])
        j = np.clip(np.searchsorted(grid, g) - 1, 0, len(grid) - 2)
        w = (g - grid[j]) / (grid[j + 1] - grid[j])
        cols = np.arange(len(g))
        la = (1 - w) * tab[j, cols] + w * tab[j + 1, cols]
        return float((la - lb[t.onset_shot:]).sum())
    la = np.log(np.maximum(syndrome_probs(h.after_circuit(base, t))[idx], 1e-300))
    if t.family == "tls_switching":
        # two-state hidden Markov model with the true switching rate, started 50/50
        r = 1.0 / t.dwell_shots
        stay, move = math.log(1 - r), math.log(r)
        a = np.array([math.log(0.5) + lb[0], math.log(0.5) + la[0]])
        for k in range(1, len(idx)):
            a = np.array([np.logaddexp(a[0] + stay, a[1] + move) + lb[k],
                          np.logaddexp(a[0] + move, a[1] + stay) + la[k]])
        return float(np.logaddexp(a[0], a[1]) - lb.sum())
    return float((la - lb)[t.onset_shot:].sum())
