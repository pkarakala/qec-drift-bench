"""Ceiling for the readout hard tier: the reference, told the form of the unlisted change.

The hard tier adds a change TASK.md does not describe (an anisotropic stretch of the IQ plane).
This solution is the reference with one more candidate family whose form it is given, never its
size: centres T mu and noise covariance s^2 (f g^2 T T^T + (1 - f) I), T = R(psi) diag(sqrt(rho),
1/sqrt(rho)) R(psi)^T. Everything else (fits, EM over populations, penalty, floors) is the
reference's. Its score shows how much of the tier a solver that discovers the form can reach.
"""
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from _load import load

R = load("reference")
STRETCH_STARTS = [(np.log(r), a) for r in (1.6, 2.5) for a in np.arange(4) * np.pi / 4]
MIN_RHO = 1.2  # report "other" above this (the genie knows the form, not the tier's range)


def stretch(lr, psi):
    c, s = np.cos(psi), np.sin(psi)
    Rp = np.array([[c, -s], [s, c]])
    return Rp @ np.diag([np.exp(lr / 2), np.exp(-lr / 2)]) @ Rp.T


class Genie(R.Model):
    def device_full(self, lr=0.0, psi=0.0, **par):
        m0, m1, sig, r = self.device(**par)
        T = stretch(lr, psi)
        g = np.exp(par.get("lg", 0.0))
        scale = sig / np.sqrt(self.f * g * g + 1 - self.f)  # = s_cal * exp(ls)
        C = scale**2 * (self.f * g * g * T @ T.T + (1 - self.f) * np.eye(2))
        return m0 @ T.T, m1 @ T.T, C, r

    def loglik(self, **par):
        m0, m1, C, r = self.device_full(**par)
        W = np.linalg.inv(np.linalg.cholesky(C))
        xw, a, b = self.x @ W.T, m0 @ W.T, m1 @ W.T
        f0, f1 = R.dens(xw, a[self.b], b[self.b], 1.0, r)
        A, B = (1 - self.pt) * f0 + self.pt * f1, (1 - self.pt) * f1 + self.pt * f0
        w = self.w
        for _ in range(15):
            wb = w[self.b]
            resp = wb * B / (A + wb * (B - A) + 1e-300)
            w = np.clip(np.bincount(self.b, resp, self.nb) / np.bincount(self.b, minlength=self.nb),
                        1e-3, 1 - 1e-3)
        self.w = w
        wb = w[self.b]
        return float(np.log(A + wb * (B - A) + 1e-300).sum() + len(self.x) * np.log(abs(np.linalg.det(W))))

    def fit_any(self, fams):
        names = [R.PAR[f] for f in fams if f != "iq_stretch"]
        st = "iq_stretch" in fams
        if not st:
            return self.fit(fams)
        listed = [self.single.get(n, R.NEUTRAL[n]) for n in names]
        best = None
        for s0 in STRETCH_STARTS:
            x0 = [*listed, *s0]
            fn = lambda v: -self.loglik(**dict(zip(names, v[:len(names)])),  # noqa: E731
                                        lr=v[-2], psi=v[-1])
            r = minimize(fn, x0, method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 800})
            if best is None or r.fun < best.fun:
                best = r
        par = dict(zip(names, best.x[:len(names)]))
        par.update(lr=float(best.x[-2]), psi=float(best.x[-1]))
        return -best.fun, par


def components(m0, m1, C, r):
    cov = C.tolist()
    comps = [{"mean": m0.tolist(), "cov": cov, "weight": 1.0, "label": 0},
             {"mean": m1.tolist(), "cov": cov, "weight": float(np.exp(-r)), "label": 1}]
    for k in range(R.N_SEG):
        u = (k + 0.5) / R.N_SEG
        w = float(np.exp(-r * k / R.N_SEG) - np.exp(-r * (k + 1) / R.N_SEG))
        comps.append({"mean": (u * m1 + (1 - u) * m0).tolist(), "cov": cov,
                      "weight": max(w, 1e-12), "label": 1})
    return comps


def solve(data):
    M = Genie(data)
    kappa = M.nom["kappa_MHz"]
    fams = (*R.FAMS, "iq_stretch")
    fits = {(): M.fit(())}
    for fam in fams:
        fits[(fam,)] = M.fit_any((fam,))
    top = sorted(fams, key=lambda fm: -fits[(fm,)][0])[:2]
    for pr in combinations(fams, 2):
        if pr[0] in top or pr[1] in top:
            fits[pr] = M.fit_any(pr)
    best = max(fits, key=lambda k: fits[k][0] - R.PENALTY * len(k))
    ll, par = fits[best]
    mags = R.magnitudes({k: v for k, v in par.items() if k not in ("lr", "psi")}, M.f, kappa)
    kinds = {k: v for k, v in mags.items()
             if R.above_floor(k, v, data["no_change_floor"], kappa, M.rc)}
    rho = float(np.exp(par.get("lr", 0.0)))
    if rho >= MIN_RHO:
        kinds["other"] = rho
    m0, m1, C, r = M.device_full(**par)
    disc = [components(m0[b], m1[b], C, r) for b in range(M.nb)]
    return {"discriminator": disc,
            "diagnosis": {"changed": bool(kinds), "kinds": kinds, "model": list(best)}}
