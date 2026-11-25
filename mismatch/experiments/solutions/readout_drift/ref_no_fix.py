"""Ablation: the reference's decision and diagnosis, with the calibration discriminator."""
from _load import load


def solve(data):
    out = load("reference").solve(data)
    out["discriminator"] = load("trivial").solve(data)["discriminator"]
    return out
