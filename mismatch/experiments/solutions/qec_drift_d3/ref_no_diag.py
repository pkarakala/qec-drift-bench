"""Ablation: the reference's decoder model and changed flag, with no diagnosis detail."""
from _load import load


def solve(data):
    out = load("reference").solve(data)
    out["diagnosis"] = {"changed": out["diagnosis"]["changed"], "kind": "none"}
    return out
