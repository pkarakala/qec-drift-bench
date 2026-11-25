"""LAZY (review finding 2): the reference's model and diagnosis, but always answers "changed"."""
from _ref_loader import reference


def solve(data):
    out = reference().solve(data)
    out["diagnosis"]["changed"] = True
    if out["diagnosis"]["kind"] == "none":
        out["diagnosis"].update(kind="hot_qubit", qubit=9, factor=4.0, onset_shot=0)
    return out
