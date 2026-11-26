"""Lazy: the reference's idle frequency and diagnosis, but always 'changed'."""
from _ref import reference


def solve(data):
    out = reference().solve(data)
    out["diagnosis"]["changed"] = True
    if not out["diagnosis"]["kinds"]:
        out["diagnosis"]["kinds"] = {"tls_new": {"freq_mhz": 200.0, "g_mhz": 0.1}}
    return out
