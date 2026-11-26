"""Reference for coherence_tls: model comparison over the listed changes on the latest scan,
then the idle frequency with the lowest fitted rate.

Method. The latest scan is modelled exactly as TASK.md states it: readout error, one delay,
a linear background rate and one Lorentzian per TLS. Every nominal TLS is free inside its
stated wander (Gaussian priors on frequency and coupling; widths free in log with a weak prior),
and the fit is weighted least squares on the populations with binomial variances
(scipy.optimize.least_squares). Hypotheses: no change; one TLS moved (each TLS in turn, its
frequency prior removed, started at every dip the no-change fit leaves unexplained); a new TLS
(started at each such dip); and the best move plus a new TLS. A hypothesis must beat the
simpler one by PENALTY per extra free parameter (a BIC-like rule, Schwarz 1978,
doi:10.1214/aos/1176344136). The idle frequency returned is the allowed one with the lowest
fitted rate; the change call is made on effect size, the fitted rate at the old idle point
against that, compared with one floor (the midpoint of the draw rules' gap in log units). Uses
only the latest scan and the nominal list; trusts the stated model; never reports "other".
Licence: Apache-2.0.
"""
import numpy as np
from scipy.optimize import least_squares

WANDER_MHZ, WANDER_G = 1.0, 0.05      # stated in TASK.md
W_PRIOR_LOG = 0.3                     # weak prior on a nominal TLS's log width (design choice)
GAP_MHZ = 10.0
PENALTY = 12.0                        # chi^2 units per extra free parameter
N_CAND = 6                            # dips tried as starts
NEW_G, NEW_W = 0.15, 2.0              # start values for a new TLS (MHz)


class Scan:
    def __init__(self, data, scan=-1):
        self.f = np.asarray(data["freq_mhz"], float)
        self.band = float(data["band_mhz"][1])
        self.p = np.asarray(data["scans"][scan], float)
        self.n = float(data["shots"])
        self.tau = float(data["delay_us"])
        self.e0, self.e1 = data["readout_error"]["e0"], data["readout_error"]["e1"]
        nom = data["nominal"]
        self.nom = [(t["freq_mhz"], t["g_mhz"], t["width_mhz"]) for t in nom["tls"]]
        self.bg0 = list(nom["bg_rate_per_us"])
        pc = np.clip(self.p, 2.0 / self.n, 1 - 2.0 / self.n)
        self.sd = np.sqrt(pc * (1 - pc) / self.n)
        self.data = data

    @staticmethod
    def tls_rate(f, tls):
        out = np.zeros_like(np.asarray(f, float))
        for f0, g, w in tls:
            out = out + 4 * np.pi * g ** 2 * w / (w ** 2 + (f - f0) ** 2)
        return out

    def rate(self, f, bg, tls):
        return bg[0] + bg[1] * f / self.band + self.tls_rate(f, tls)

    def model(self, bg, tls):
        return self.e0 + (1 - self.e0 - self.e1) * np.exp(-self.tau * self.rate(self.f, bg, tls))

    def fit(self, free_f=(), new=None):
        """Fit with nominal TLS (those in free_f without a frequency prior) and optionally a new
        TLS started at new=(f, g, w). Returns (chi2, bg, tls)."""
        k = len(self.nom)
        x0 = [*self.bg0]
        for f0, g, w in self.nom:
            x0 += [f0, np.log(g), np.log(w)]
        if new is not None:
            x0 += [new[0], np.log(new[1]), np.log(new[2])]
        x0 = np.array(x0, float)
        for j, f0 in free_f:
            x0[2 + 3 * j] = f0

        def unpack(x):
            tls = [(x[2 + 3 * i], np.exp(x[3 + 3 * i]), np.exp(x[4 + 3 * i]))
                   for i in range(k + (new is not None))]
            return x[:2], tls

        free = {j for j, _ in free_f}

        def resid(x):
            bg, tls = unpack(x)
            r = [(self.model(bg, tls) - self.p) / self.sd]
            pri = []
            for i, (f0, g, w) in enumerate(self.nom):
                if i not in free:
                    pri.append((x[2 + 3 * i] - f0) / WANDER_MHZ)
                pri.append((x[3 + 3 * i] - np.log(g)) / (WANDER_G / np.sqrt(3)))
                pri.append((x[4 + 3 * i] - np.log(w)) / W_PRIOR_LOG)
            return np.concatenate(r + [np.array(pri)])

        lo = np.full(x0.size, -np.inf)
        hi = np.full(x0.size, np.inf)
        lo[0], hi[0] = 0.0, 1.0
        for i in range(k + (new is not None)):
            lo[2 + 3 * i], hi[2 + 3 * i] = -20.0, self.band + 20.0
            lo[3 + 3 * i], hi[3 + 3 * i] = np.log(0.005), np.log(3.0)
            lo[4 + 3 * i], hi[4 + 3 * i] = np.log(0.05), np.log(60.0)
        x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
        sol = least_squares(resid, x0, bounds=(lo, hi), method="trf", x_scale="jac",
                            max_nfev=400)
        bg, tls = unpack(sol.x)
        return float(np.sum(sol.fun ** 2)), bg, tls

    def dips(self, bg, tls, n=N_CAND):
        """Frequencies where the scan relaxes faster than the fit: unexplained dips."""
        p = np.clip((self.p - self.e0) / (1 - self.e0 - self.e1), 1e-3, 1)
        g_meas = -np.log(p) / self.tau
        ex = g_meas - self.rate(self.f, bg, tls)
        ex = np.convolve(ex, np.ones(3) / 3, mode="same")
        out = []
        for i in np.argsort(ex)[::-1]:
            if ex[i] <= 0 or len(out) >= n:
                break
            if all(abs(self.f[i] - o) >= 3.0 for o in out):
                out.append(float(self.f[i]))
        return out


def allowed(data, f):
    lo, hi = data["band_mhz"]
    ok = (f >= lo + data["edge_mhz"]) & (f <= hi - data["edge_mhz"])
    for a, b in data["forbidden_mhz"]:
        ok &= ~((f >= a) & (f <= b))
    return ok


def hypotheses(S):
    """{name: (chi2, bg, tls, info)} and the number of extra parameters of each."""
    c0, bg0, t0 = S.fit()
    cands = {"none": (c0, bg0, t0, {})}
    starts = S.dips(bg0, t0)
    best_move = None
    for j in range(len(S.nom)):
        for f0 in starts:
            if abs(f0 - S.nom[j][0]) < 3 * WANDER_MHZ:
                continue
            c, bg, t = S.fit(free_f=[(j, f0)])
            if best_move is None or c < best_move[0]:
                best_move = (c, bg, t, {"tls": j})
    if best_move is not None:
        cands["move"] = best_move
    best_new = None
    for f0 in starts:
        c, bg, t = S.fit(new=(f0, NEW_G, NEW_W))
        if best_new is None or c < best_new[0]:
            best_new = (c, bg, t, {})
    if best_new is not None:
        cands["new"] = best_new
    if best_move is not None and best_new is not None:
        j = best_move[3]["tls"]
        _, bgm, tm = best_move[:3]
        best = None
        for f0 in S.dips(bgm, tm):
            c, bg, t = S.fit(free_f=[(j, tm[j][0])], new=(f0, NEW_G, NEW_W))
            if best is None or c < best[0]:
                best = (c, bg, t, {"tls": j})
        if best is not None:
            cands["both"] = best
    return cands, {"none": 0, "move": 1, "new": 3, "both": 4}


def pick(cands, npar):
    best = "none"
    for name in ("move", "new", "both"):
        if name in cands and cands[name][0] + PENALTY * npar[name] < \
                cands[best][0] + PENALTY * npar[best]:
            best = name
    return best


def answer(S, name, cand, changed=None):
    data = S.data
    _, bg, tls, info = cand
    fine = np.arange(0.0, S.band + 0.025, 0.05)
    ok = allowed(data, fine)
    g = S.rate(fine, bg, tls)
    best = float(fine[ok][np.argmin(g[ok])])
    gap = float(S.rate(np.array([data["idle_mhz"]]), bg, tls)[0] - g[ok].min())
    if changed is None:
        changed = gap > data["floor_rate_per_us"]
    kinds = {}
    if changed and name in ("move", "both"):
        j = info["tls"]
        kinds["tls_moved"] = {"tls": j, "freq_mhz": float(tls[j][0])}
    if changed and name in ("new", "both"):
        f0, g0, _ = tls[-1]
        kinds["tls_new"] = {"freq_mhz": float(f0), "g_mhz": float(g0)}
    return {"idle_mhz": best if changed else data["idle_mhz"],
            "diagnosis": {"changed": bool(changed), "kinds": kinds}}


def solve(data):
    S = Scan(data)
    cands, npar = hypotheses(S)
    name = pick(cands, npar)
    return answer(S, name, cands[name])
