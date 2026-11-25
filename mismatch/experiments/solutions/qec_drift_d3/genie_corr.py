"""Ceiling for the QEC hard tier: the reference, told the form of the unlisted change.

The hard tier adds a change TASK.md does not describe: a correlated two-qubit depolarizing error
on two nearby data qubits, once per round after the data qubits' own depolarization. This
solution is the reference with one more hypothesis family of that form, for every data pair
within 2.9 lattice units, at the reference's factor grid (probability = factor x 0.8 x
p_nominal; it is told the form, never the size). It competes with the listed hypotheses in the
same pairwise-moment test, and is reported as kind "other". Everything else is the reference's.
Its score shows how much of the tier a solver that discovers the form can reach.
"""
import math

import numpy as np
import stim

from _load import load

R = load("reference")
MAX_DIST = 2.9


class Hypotheses(R.Hypotheses):
    """The reference's hypotheses plus ("other", data pair); same moments and factor grid."""

    def __init__(self, base, mom, bg, data):
        c, dq = data["qubit_coords"], sorted(data["data_qubits"])
        self.base, self.mom, self.bg = base, mom, bg
        self.data_set, self.p0 = set(dq), 0.8 * data["p_nominal"]
        self.items = [("hot_qubit", q) for q in sorted(c)]
        self.items += [("hot_coupler", tuple(p)) for p in data["gate_pairs"]]
        self.items += [("other", (a, b)) for i, a in enumerate(dq) for b in dq[i + 1:]
                       if math.dist(c[a][:2], c[b][:2]) <= MAX_DIST]
        self.m0 = mom.model(self.circuit(None, 1.0))
        self.dm = np.array([[mom.model(self.circuit(h, f)) - self.m0 for f in R.FACTORS]
                            for h in self.items])

    def circuit(self, h, f) -> stim.Circuit:
        if h is None or h[0] != "other":
            return super().circuit(h, f)
        out = stim.Circuit()
        for ins in self.base:  # scale one instruction at a time, then add the pair error
            one = stim.Circuit()
            one.append(ins)
            out += R.scaled(one, self.bg)
            if ins.name == "DEPOLARIZE1" and {t.value for t in ins.targets_copy()} == self.data_set:
                out.append("DEPOLARIZE2", list(h[1]), [min(f * self.p0, 0.5)])
        return out


R.Hypotheses = Hypotheses


def solve(data):
    return R.solve(data)
