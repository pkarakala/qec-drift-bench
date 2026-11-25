"""Reference solution for qec_drift_d3: the strongest method we know, written as a solver would
write it (no harness imports; it sees only `data`).

1. Background: one global error scale fitted to the labeled pre-change shots (a per-qubit fit
   from 50k shots is noisier than the spread it tries to capture; review finding 6).
2. Burst: matched filter on per-shot detection counts with an exponential kernel, scanned over
   decay constants; threshold from the same scan on the pre-change shots; then a least-squares
   fit of onset, amplitude and decay. The burst shots are cut before step 3.
3. Stationary fault: two-sample comparison of detector rates and pairwise coincidences
   (p_ij statistics, Chen et al. 2021, arXiv 2102.06132) between pre and post shots, against
   exact moments of "hot qubit q at factor f" and "hot gate pair at factor f" hypotheses.
4. Onset: CUSUM change point on per-shot log-likelihood ratios, then a refit on the shots after it.
   Switching: a two-state Viterbi path over 250-shot windows of the same log-likelihood ratios;
   two or more transitions means a telegraph (TLS) rather than one onset. The factor is refitted
   on the "on" shots and the decoder uses the time-averaged model (1 + f) / 2.
5. Decoder: the fitted model as a detector error model (the grader decodes it with PyMatching).
Licence: Apache-2.0.
"""

from __future__ import annotations

import math

import numpy as np
import stim
from scipy.signal import fftconvolve

ARITY = {"DEPOLARIZE1": 1, "DEPOLARIZE2": 2, "X_ERROR": 1, "Z_ERROR": 1}
FACTORS = np.geomspace(1.25, 30.0, 24)
CUSUM_THRESHOLD = 25.0  # log-likelihood gain needed to call an onset inside the batch
BURST_Z = 6.0  # matched-filter z-score floor for a burst


def scaled(base: stim.Circuit, qf: dict, pf: dict | None = None, s: float = 1.0) -> stim.Circuit:
    pf = pf or {}
    out = stim.Circuit()
    for ins in base:
        if ins.name not in ARITY:
            out.append(ins)
            continue
        k, p, t = ARITY[ins.name], ins.gate_args_copy()[0], ins.targets_copy()
        groups: dict[float, list] = {}
        for i in range(0, len(t), k):
            g = t[i : i + k]
            f = s * max(qf.get(x.value, 1.0) for x in g)
            if k == 2:
                f *= pf.get(frozenset(x.value for x in g), 1.0)
            groups.setdefault(f, []).extend(g)
        for f, g in groups.items():
            out.append(ins.name, g, [min(p * f, 0.5)])
    return out


def dem_terms(dem: stim.DetectorErrorModel, n_det: int) -> tuple[np.ndarray, np.ndarray]:
    rows, w = [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        if dets:
            r = np.zeros(n_det)
            r[dets] = 1
            rows.append(r)
            w.append(math.log(1 - 2 * inst.args_copy()[0]))
    return np.array(rows), np.array(w)


class Moments:
    """Exact detector rates and pairwise coincidence rates of a model."""

    def __init__(self, n_det: int, pairs: tuple):
        self.n, self.pairs = n_det, pairs

    def model(self, c: stim.Circuit) -> np.ndarray:
        H, w = dem_terms(c.detector_error_model(), self.n)
        L, both = H.T @ w, (H * w[:, None]).T @ H
        i, j = self.pairs
        Eij = np.exp(L[i] + L[j] - 2 * both[i, j])
        return np.concatenate([(1 - np.exp(L)) / 2, (1 - np.exp(L[i]) - np.exp(L[j]) + Eij) / 4])

    def observed(self, det: np.ndarray) -> np.ndarray:
        x = det.astype(np.float64)
        i, j = self.pairs
        return np.concatenate([x.mean(0), (x.T @ x / len(x))[i, j]])


def fit_scale(base, mom, pre) -> float:
    o, n = mom.observed(pre), len(pre)
    grid = np.geomspace(0.7, 1.6, 46)
    cost = []
    for s in grid:
        m = mom.model(scaled(base, {}, s=s))
        cost.append(np.sum((o - m) ** 2 / (m * (1 - m) / n)))
    return float(grid[int(np.argmin(cost))])


def fit_background(base, mom, pre, qubits, s, prior_sigma=0.3, iters=6) -> dict:
    """Per-qubit error factors from the pre shots: Gauss-Newton on log factors with a log-normal
    prior of width prior_sigma around the global scale s (TASK.md states the spread)."""
    o, n = mom.observed(pre), len(pre)
    x = np.full(len(qubits), math.log(s))

    def model(v):
        return mom.model(scaled(base, dict(zip(qubits, np.exp(v)))))

    for _ in range(iters):
        m = model(x)
        w = 1 / (m * (1 - m) / n)
        J = np.empty((len(m), len(x)))
        for k in range(len(x)):
            e = x.copy()
            e[k] += 0.05
            J[:, k] = (model(e) - m) / 0.05
        A = J.T @ (J * w[:, None]) + np.eye(len(x)) / prior_sigma**2
        r = J.T @ (w * (o - m)) - (x - math.log(s)) / prior_sigma**2
        x = x + np.clip(np.linalg.solve(A, r), -0.5, 0.5)
    return {q: float(v) for q, v in zip(qubits, np.exp(x))}


def burst_scan(pre, post, det_xy):
    """Matched filter for a decaying burst: per-shot detection counts, weighted by a Gaussian
    spatial profile around candidate centres, convolved with exponential kernels. The threshold
    is the largest response of the same filter bank on the pre-change shots. Returns
    (onset, decay, amplitude proxy) or None."""
    xp, x = pre.astype(float), post.astype(float)
    mu = xp.mean(0)
    filt = []
    for cx in (1.0, 3.0, 5.0):
        for cy in (1.0, 3.0, 5.0):
            for rad in (2.0, 4.0):
                filt.append(np.exp(-((det_xy - (cx, cy)) ** 2).sum(1) / (2 * rad**2)))
    filt.append(np.ones(len(mu)))
    W = np.array(filt)
    sp, s = (xp - mu) @ W.T, (x - mu) @ W.T  # (shots, filters)
    sd = sp.std(0)

    def zmax(y):
        best = (-np.inf, 0, 0.0, 0)
        for tau in np.geomspace(30, 1500, 12):
            k = np.exp(-np.arange(int(5 * tau)) / tau)
            z = fftconvolve(y, k[::-1, None], mode="valid", axes=0) / (sd * math.sqrt(np.sum(k**2)))
            j, f = np.unravel_index(int(np.argmax(z)), z.shape)
            if z[j, f] > best[0]:
                best = (float(z[j, f]), int(j), tau, int(f))
        return best

    z, peak, _, fi = zmax(s)
    rng = np.random.default_rng(0)  # null: pre shots and time-shuffled post shots
    null = max([zmax(sp)[0]] + [zmax(s[rng.permutation(len(s))])[0] for _ in range(3)])
    if z < max(BURST_Z, 1.1 * null):
        return None
    # Fit onset and decay with a free amplitude per detector: maximize the projected
    # chi-square sum_d (x_d . shape)^2 / (v_d |shape|^2) over (onset, decay).
    v = xp.var(0)
    best = (-np.inf, None)
    t = np.arange(len(x))
    lo = max(0, peak - 40)
    xs = (x[lo : min(len(x), peak + 9000)] - mu) / np.sqrt(v)
    for t0 in range(lo, min(len(x) - 50, peak + 40)):
        tt = t[lo : lo + len(xs)] - t0
        for tau in np.geomspace(20, 4000, 70):
            g = np.where(tt >= 0, np.exp(-np.clip(tt, 0, None) / tau), 0.0)
            proj = xs.T @ g
            val = float(proj @ proj / (g @ g))
            if val > best[0]:
                best = (val, (t0, tau, float(np.abs(proj).max())))
    # Second stage: one spatial pattern (the per-detector amplitudes of stage one), so the
    # decay fit has two free parameters instead of 24 and is not pulled toward short decays.
    t0, tau, _ = best[1]
    tt = t[lo : lo + len(xs)] - t0
    g = np.where(tt >= 0, np.exp(-np.clip(tt, 0, None) / tau), 0.0)
    amp = np.clip(xs.T @ g, 0, None)
    y = xs @ (amp / max(np.linalg.norm(amp), 1e-12))
    best = (np.inf, None)
    for t1 in range(max(lo, t0 - 15), t0 + 16):
        tt = t[lo : lo + len(xs)] - t1
        for tau in np.geomspace(20, 4000, 90):
            g = np.where(tt >= 0, np.exp(-np.clip(tt, 0, None) / tau), 0.0)
            a = max(float(y @ g) / float(g @ g), 0.0)
            sse = float(np.sum((y - a * g) ** 2))
            if sse < best[0]:
                best = (sse, (t1, tau, a))
    if not 40 <= best[1][1] <= 1500:  # a spike, or no decay inside the batch (a step change)
        return None
    return best[1]


class Hypotheses:
    """Single-fault hypotheses (hot qubit q, hot gate pair) and their predicted moment shifts."""

    def __init__(self, base, mom, bg, data):
        self.base, self.mom, self.bg = base, mom, bg
        self.items = [("hot_qubit", q) for q in sorted(data["qubit_coords"])]
        self.items += [("hot_coupler", tuple(p)) for p in data["gate_pairs"]]
        self.m0 = mom.model(self.circuit(None, 1.0))
        self.dm = np.array([[mom.model(self.circuit(h, f)) - self.m0 for f in FACTORS]
                            for h in self.items])

    def circuit(self, h, f) -> stim.Circuit:
        if h is None:
            return scaled(self.base, self.bg)
        if h[0] == "hot_qubit":
            return scaled(self.base, {**self.bg, h[1]: self.bg[h[1]] * f})
        return scaled(self.base, self.bg, {frozenset(h[1]): f})

    def gains(self, d_obs, v) -> np.ndarray:
        """Chi-square gain of each (hypothesis, factor) over "no change"; shape (H, F)."""
        return 2 * (self.dm * (d_obs / v)).sum(-1) - (self.dm**2 / v).sum(-1)

    def var(self, n1, n2):
        return self.m0 * (1 - self.m0) * (1 / n1 + 1 / n2)

    def best(self, pre_o, post_o, n1, n2):
        g = self.gains(post_o - pre_o, self.var(n1, n2))
        i, j = np.unravel_index(int(np.argmax(g)), g.shape)
        return float(g[i, j]), self.items[i], float(FACTORS[j])

    def null_max(self, feats_pre, feats_post, reps=100):
        """Largest gain over all hypotheses when pre and post shots are randomly relabeled."""
        allf = np.vstack([feats_pre, feats_post])
        n1, rng, out = len(feats_pre), np.random.default_rng(1), []
        tot = allf.sum(0)
        v = self.var(n1, len(feats_post))
        for _ in range(reps):
            a = allf[rng.choice(len(allf), n1, replace=False)].sum(0)
            d = (tot - a) / (len(allf) - n1) - a / n1
            out.append(float(self.gains(d, v).max()))
        return max(out)

    def refine(self, h, f0, pre, post):
        """Fine factor fit of one hypothesis on post - pre moments."""
        mom = self.mom
        d = mom.observed(post) - mom.observed(pre)
        v = self.var(len(pre), len(post))
        fs = np.geomspace(max(f0 / 1.4, 1.0), f0 * 1.4, 25)
        c = [np.sum((d - (mom.model(self.circuit(h, f)) - self.m0)) ** 2 / v) for f in fs]
        return float(fs[int(np.argmin(c))])


def features(det: np.ndarray, pairs) -> np.ndarray:
    x = det.astype(np.float32)
    i, j = pairs
    return np.hstack([x, x[:, i] * x[:, j]])


def cusum_onset(post, c0, c1, n_det):
    """Change point maximizing the log-likelihood gain of model c1 over c0 from k on."""
    mom = Moments(n_det, (np.array([0]), np.array([0])))
    p0, p1 = mom.model(c0)[:n_det], mom.model(c1)[:n_det]
    x = post.astype(float)
    llr = x @ np.log(p1 / p0) + (1 - x) @ np.log((1 - p1) / (1 - p0))
    tail = np.cumsum(llr[::-1])[::-1]
    k = int(np.argmax(tail))
    return k, float(tail[k] - max(tail[0], 0.0))


TLS_WIN, TLS_SWITCH_P = 250, 0.001


def shot_llr(post, c0, c1, n_det):
    mom = Moments(n_det, (np.array([0]), np.array([0])))
    p0, p1 = mom.model(c0)[:n_det], mom.model(c1)[:n_det]
    x = post.astype(float)
    return x @ np.log(p1 / p0) + (1 - x) @ np.log((1 - p1) / (1 - p0))


def viterbi_on(llr):
    """Two-state path over windows: state 1 earns the window's LLR, state 0 earns 0."""
    n = len(llr) // TLS_WIN
    w = llr[: n * TLS_WIN].reshape(n, TLS_WIN).sum(1)
    stay, flip = math.log(1 - TLS_SWITCH_P), math.log(TLS_SWITCH_P)
    v = np.array([0.0, w[0]])
    back = np.zeros((n, 2), int)
    for i in range(1, n):
        c = np.array([[v[0] + stay, v[1] + flip], [v[0] + flip, v[1] + stay]])  # to x from
        back[i] = c.argmax(1)
        v = c.max(1) + np.array([0.0, w[i]])
    st = np.zeros(n, int)
    st[-1] = int(v.argmax())
    for i in range(n - 1, 0, -1):
        st[i - 1] = back[i, st[i]]
    return np.repeat(st, TLS_WIN), int(np.sum(st[1:] != st[:-1]))


def solve(data: dict) -> dict:
    base = stim.Circuit(data["circuit"])
    pre, post = data["pre_detectors"], data["post_detectors"]
    n_det = pre.shape[1]
    H, _ = dem_terms(base.detector_error_model(), n_det)
    pairs = np.nonzero(np.triu((H.T @ H) > 0, 1))
    mom = Moments(n_det, pairs)
    s = fit_scale(base, mom, pre)
    qubits = sorted(data["qubit_coords"])
    bg = fit_background(base, mom, pre, qubits, s)
    hyp = Hypotheses(base, mom, bg, data)
    bg_dem = str(hyp.circuit(None, 1.0).detector_error_model(decompose_errors=True))

    dc = data["detector_coords"]
    det_xy = np.array([dc[k][:2] for k in sorted(dc, key=int)], dtype=float)
    burst = burst_scan(pre, post, det_xy)
    if burst is not None:
        t0, tau, amp = burst
        keep = np.ones(len(post), bool)
        keep[max(0, t0 - 50) : t0 + int(15 * tau)] = False
        post = post[keep]

    fp, fq = features(pre, pairs), features(post, pairs)
    gain, h, f = hyp.best(fp.mean(0), fq.mean(0), len(fp), len(fq))
    threshold = hyp.null_max(fp, fq)
    if gain < threshold:
        if burst is not None:
            return {"dem": bg_dem, "diagnosis": {"changed": True, "kind": "burst",
                                                 "onset_shot": int(t0), "decay_shots": float(tau)}}
        return {"dem": bg_dem, "diagnosis": {"changed": False, "kind": "none", "gain": gain,
                                             "threshold": threshold}}

    onset, tail = 0, post
    if burst is None and h[0] == "hot_qubit":
        c0 = hyp.circuit(None, 1.0)
        on, n_sw = viterbi_on(shot_llr(post, c0, hyp.circuit(h, f), n_det))
        if n_sw >= 2:
            for _ in range(3):
                sel = post[: len(on)][on == 1]
                for _ in range(4):  # the refine grid spans 1.4x each way; walk to the optimum
                    f_new = hyp.refine(h, f, pre, sel)
                    if abs(math.log(f_new / f)) < 0.01:
                        break
                    f = f_new
                on, n_sw = viterbi_on(shot_llr(post, c0, hyp.circuit(h, f), n_det))
            if n_sw >= 2:
                dem = hyp.circuit(h, (1 + f) / 2).detector_error_model(decompose_errors=True)
                return {"dem": str(dem), "diagnosis": {
                    "changed": True, "kind": "tls_switching", "qubit": h[1], "factor": f,
                    "dwell_shots": float(len(post) / n_sw), "gain": gain}}
    if burst is None:
        k, lr = cusum_onset(post, hyp.circuit(None, 1.0), hyp.circuit(h, f), n_det)
        if k > 0.02 * len(post) and lr > CUSUM_THRESHOLD:
            for _ in range(2):
                gain, h, f = hyp.best(fp.mean(0), fq[k:].mean(0), len(fp), len(fq) - k)
                k, lr = cusum_onset(post, hyp.circuit(None, 1.0), hyp.circuit(h, f), n_det)
            onset, tail = k, post[k:]
    f = hyp.refine(h, f, pre, tail)
    kind, loc = h
    diag = {"changed": True, "kind": kind, "factor": f, "onset_shot": int(onset), "gain": gain,
            "threshold": threshold}
    diag["qubit" if kind == "hot_qubit" else "pair"] = loc if kind == "hot_qubit" else list(loc)
    dem = hyp.circuit(h, f).detector_error_model(decompose_errors=True)
    return {"dem": str(dem), "diagnosis": diag}
