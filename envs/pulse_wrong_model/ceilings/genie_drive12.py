"""Ceiling for the pulse hard tier: the reference, told the form of the unlisted change.

The hard tier adds an effect outside the design model's form: the drive reaches the 1-2 (and 2-3)
transition m times as strongly as the Duffing matrix elements say. This solution is the reference
plus that one parameter, whose form it is given and whose size it fits: after the reference's fit
of (s, delta, T2, k) it fits m and refits (s, k) on the Rabi and DRAG-check traces together,
flags `other` when m is more than 3 standard errors from 1, and runs the reference's robust GRAPE
with the fitted m. Its score shows how much of the tier a solver that discovers the form can reach.
"""
import numpy as np
from scipy.optimize import minimize

from _load import load

R = load("reference")


def set_m(M, m):
    n = len(M.n)
    b = np.diag(np.sqrt(np.arange(1, n)), 1).astype(complex)
    b[1:, 2:] *= m
    M.X, M.Y = b + b.conj().T, 1j * (b.conj().T - b)


def unitary(M, u, s, delta, kc, dt):
    w, V = np.linalg.eigh(M.H(np.asarray(u, dtype=complex), s, delta, kc))
    P = (V * np.exp(-1j * dt * w)[..., None, :]) @ np.swapaxes(V.conj(), -1, -2)
    U = np.eye(P.shape[-1], dtype=complex)
    for k in range(P.shape[0]):
        U = P[k] @ U
    return U


def check_trace(M, s, delta, kc):
    d = M.d
    u = np.asarray(d["nominal_pulse"])
    pair = unitary(M, -u, s, delta, kc, d["dt_ns"]) @ unitary(M, u, s, delta, kc, d["dt_ns"])
    n = np.asarray(d["drag_check_pairs"])
    psi, out = np.eye(pair.shape[0], dtype=complex)[:, 0], []
    for _ in range(int(n.max())):
        psi = pair @ psi
        out.append(1.0 - abs((psi[0] - psi[1]) / np.sqrt(2.0)) ** 2)  # then ideal pi/2 about y
    return np.clip(np.array(out)[n - 1], 0.0, 1.0)


def nll_rc(M, s, delta, kc, m):
    d = M.d
    set_m(M, m)
    kr = np.asarray(d["rabi_p_exc"]) * d["rabi_shots"]
    kch = np.asarray(d["drag_check_p_exc"]) * d["drag_check_shots"]
    v = R.nll(M.rabi(s, delta, kc), kr, d["rabi_shots"]) + \
        R.nll(check_trace(M, s, delta, kc), kch, d["drag_check_shots"])
    set_m(M, 1.0)
    return v


def solve(data):
    M = R.Model(data)
    s_d, f01 = data["drive_scale_design_ghz"], data["f01_design_ghz"]
    s, delta, t2, kc = R.fit(M)
    ms = np.linspace(0.8, 1.2, 161)
    m0 = ms[np.argmin([nll_rc(M, s, delta, kc, m) for m in ms])]
    r = minimize(lambda x: nll_rc(M, x[0] * s_d, delta, x[1] * 0.01, x[2]),
                 [s / s_d, kc / 0.01, m0], method="Nelder-Mead",
                 options=dict(xatol=1e-7, fatol=1e-6, maxiter=3000,
                              initial_simplex=[[s / s_d, kc / 0.01, m0],
                                               [s / s_d + 0.003, kc / 0.01, m0],
                                               [s / s_d, kc / 0.01 + 0.3, m0],
                                               [s / s_d, kc / 0.01, m0 + 0.01]]))
    s, kc, m = r.x[0] * s_d, r.x[1] * 0.01, float(r.x[2])
    set_m(M, m)
    sig_s, sig_d, sig_k = R.fisher_sigmas(M, s, delta, t2, kc)
    h = 1e-4  # standard error of m from the curvature of the joint likelihood
    curv = (nll_rc(M, s, delta, kc, m + h) - 2 * nll_rc(M, s, delta, kc, m)
            + nll_rc(M, s, delta, kc, m - h)) / h**2
    sig_m = 1.0 / np.sqrt(max(curv, 1e-12))
    set_m(M, m)
    s_ch = abs(s - s_d) > R.NSIG * sig_s
    d_ch = abs(delta) > R.NSIG * sig_d
    k_ch = abs(kc) > R.NSIG * sig_k
    m_ch = abs(m - 1.0) > R.NSIG * sig_m
    if not (s_ch or d_ch or k_ch or m_ch):
        return {"pulse": np.asarray(data["nominal_pulse"]),
                "diagnosis": {"changed": False, "drive_scale": s_d, "f01_ghz": f01,
                              "compression": 0.0, "other": False}}
    s_use = s if s_ch else s_d
    d_use = delta if d_ch else 0.0
    k_use = kc if k_ch else 0.0
    if not m_ch:
        set_m(M, 1.0)
    u0 = np.asarray(data["nominal_pulse"]) * s_d / s_use
    v = R.robust_pulse(M, u0, s_use, sig_s, d_use, cap=data["amp_cap"] * (1 - max(k_use, 0.0)))
    return {"pulse": R.predistort(v, k_use),
            "diagnosis": {"changed": True, "drive_scale": float(s_use),
                          "f01_ghz": float(f01 + d_use), "compression": float(k_use),
                          "other": bool(m_ch), "drive12_factor": m}}
