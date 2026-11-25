"""Maintainer tool: how often does a fault drawn at the stated sizes open a decoding gap?

usage: python eval/survey.py FAMILY N [SEED0]   (FAMILY: hot or coupler)
Draws N faults exactly as the instance generator does and prints, per draw, the factor, the
location and the exact relative gap (background-matching vs true-model matching), then the share
that is graded on fix (gap >= GAP_GATE). Uses a looser tail (1e-5) for speed. Apache-2.0.
"""
import sys

import numpy as np

import exact
import harness as H

exact.TAIL = 1e-5
fam, n = sys.argv[1], int(sys.argv[2])
seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 777
base = H.nominal_circuit()
gaps = []
for k in range(n):
    t = H.draw_truth(np.random.default_rng([seed0, k]), fam, base)
    tab = exact.Table.from_dem(H.after_circuit(base, t).detector_error_model())
    e_bg = tab.ler_of(H.matching_for(H.scaled_circuit(base, t.background)))
    e_tr = tab.ler_of(H.matching_for(H.after_circuit(base, t)))
    gaps.append((e_bg - e_tr) / e_bg)
    print(f"{fam} {t.qubit if t.qubit is not None else t.pair} factor {t.factor:.2f} "
          f"gap {gaps[-1]:+.3f}", flush=True)
g = np.array(gaps)
print(f"{fam}: {np.mean(g >= H.GAP_GATE):.2f} of {n} draws open a gap >= {H.GAP_GATE}; "
      f"median gap {np.median(g):+.3f}")
