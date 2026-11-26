"""Ceiling for the coherence_tls hard tier: the reference told the form (never the size) of the
unlisted change, a nominal TLS that switches between two frequencies.

Method. Fit the reference's hypotheses to each of the two oldest scans as well as the latest.
Under the stated model those scans equal the nominal list (changes happen only after the
calibration scan at hour -1), so a TLS that a history scan finds moved is switching; its second
frequency is the one that scan finds. Its share of time in that state is estimated from all
four scans (the latest one checked against both positions), with a Laplace prior. Today's
listed changes come from the latest scan's hypotheses with the switching TLS left out. The
idle frequency minimizes the long-run mixture rate. Licence: Apache-2.0.
"""
import numpy as np

from _load import load

R = load("reference")
SAME_MHZ = 3.0


def moved_in(S):
    cands, npar = R.hypotheses(S)
    name = R.pick(cands, npar)
    if name in ("move", "both"):
        j = cands[name][3]["tls"]
        return j, float(cands[name][2][j][0])
    return None


def solve(data):
    hist = [moved_in(R.Scan(data, s)) for s in (0, 1)]
    hist = [h for h in hist if h is not None]
    S = R.Scan(data)
    cands, npar = R.hypotheses(S)
    name = R.pick(cands, npar)
    if not hist:
        return R.answer(S, name, cands[name])
    j, f_b = hist[0]
    f_a = S.nom[j][0]
    # States of the four scans: hour -1 is f_a; the history scans as found; the latest one as
    # its own fit places TLS j.
    n_b = sum(1 for h in hist if h[0] == j) + 0
    _, bg, tls, info = cands[name]
    today_b = abs(tls[j][0] - f_b) < SAME_MHZ
    n_b += today_b
    p_b = (n_b + 1) / (4 + 2)
    kinds = {"other": {"tls": j, "freq_a_mhz": f_a, "freq_b_mhz": f_b, "p_b": p_b}}
    listed = name
    if name in ("move", "both") and info.get("tls") == j:
        listed = {"move": "none", "both": "new"}[name]
    if listed in ("move", "both"):
        kinds["tls_moved"] = {"tls": info["tls"], "freq_mhz": float(tls[info["tls"]][0])}
    if listed in ("new", "both"):
        kinds["tls_new"] = {"freq_mhz": float(tls[-1][0]), "g_mhz": float(tls[-1][1])}
    a = [list(t) for t in tls]
    b = [list(t) for t in tls]
    a[j][0], b[j][0] = f_a, f_b
    fine = np.arange(0.0, S.band + 0.025, 0.05)
    ok = R.allowed(data, fine)
    g = (1 - p_b) * S.rate(fine, bg, a) + p_b * S.rate(fine, bg, b)
    best = float(fine[ok][np.argmin(g[ok])])
    g_idle = float(np.interp(data["idle_mhz"], fine, g))
    changed = g_idle - float(g[ok].min()) > data["floor_rate_per_us"]
    return {"idle_mhz": best if changed else data["idle_mhz"],
            "diagnosis": {"changed": bool(changed), "kinds": kinds if changed else {}}}
