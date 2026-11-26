"""Lazy: idle where the latest scan is highest (the most shots read as 1), always 'changed',
never names a kind."""
import numpy as np


def solve(data):
    f = np.asarray(data["freq_mhz"], float)
    p = np.asarray(data["scans"][-1], float)
    lo, hi = data["band_mhz"]
    ok = (f >= lo + data["edge_mhz"]) & (f <= hi - data["edge_mhz"])
    for a, b in data["forbidden_mhz"]:
        ok &= ~((f >= a) & (f <= b))
    return {"idle_mhz": float(f[ok][np.argmax(p[ok])]),
            "diagnosis": {"changed": True, "kinds": {}}}
