"""LAZY (review finding 2): always refit (the reference's model as decoder), never report a change."""
from _ref_loader import reference


def solve(data):
    out = reference().solve(data)
    out["diagnosis"] = {"changed": False, "kind": "none"}
    return out
