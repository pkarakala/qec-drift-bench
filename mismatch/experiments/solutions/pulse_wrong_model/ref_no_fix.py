"""Ablation: the reference's decision and diagnosis, with the design pulse."""
import numpy as np
from _load import load


def solve(data):
    out = load("reference").solve(data)
    out["pulse"] = np.asarray(data["nominal_pulse"])
    return out
