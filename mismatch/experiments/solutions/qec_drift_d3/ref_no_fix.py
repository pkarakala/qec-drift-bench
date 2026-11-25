"""Ablation: the reference's decision and diagnosis, but the nominal model as the decoder."""
from _load import load


def solve(data):
    out = load("reference").solve(data)
    out["dem"] = load("trivial").solve(data)["dem"]
    return out
