"""Hidden T1 landscape, ground truth, baselines and scoring for coherence_tls.

A flux-tunable transmon relaxes into a smooth background and into two-level-system (TLS)
defects: each TLS adds a Lorentzian in the qubit frequency to the relaxation rate (Barends 2013,
arXiv:1304.2322; Klimov 2018, arXiv:1809.01043; Mueller 2019, arXiv:1705.01108). TLS move
(spectral diffusion), so the idle frequency chosen at the last calibration can land on one.
The solver sees the latest T1 scan (excited-state population after a fixed delay at every
frequency of the band) and the three scans taken in the hours before it, and returns an idle
frequency plus a diagnosis. The grader scores the relaxation rate the qubit will have at that
frequency, from the hidden truth. The solver runs in a separate process (eval/_isolate.py).
Licence: Apache-2.0.
"""
import numpy as np

import rubric
from _isolate import run_solver

# ---------------- device model ----------------
BAND_MHZ = 400.0                       # tunable idle band, offsets from its bottom
DF_MHZ = 1.0                           # scan step (Klimov 2018: 1 MHz)
FREQ = np.arange(0.0, BAND_MHZ + DF_MHZ / 2, DF_MHZ)
FINE = np.arange(0.0, BAND_MHZ + 0.025, 0.05)     # grid the grader optimizes on
# Background rate G_bg(f) = g0 + g1 f / BAND (1/us): T1 30 to 120 us away from TLS (Carroll
# 2022, arXiv:2105.15201: 31 to 113 us), a mild slope either way (assumption).
T1_BG_US, SLOPE = (30.0, 120.0), (-0.3, 0.3)
# TLS: rate 2 g^2 G / (G^2 + d^2) in angular units (Klimov 2018, Supplement S3), i.e.
# 4 pi g^2 w / (w^2 + d^2) per us with g = g/2pi, w = G/2pi, d in MHz. Klimov 2018: couplings
# 50 to 500 kHz, TLS decoherence 0.5 to 20 MHz, 13 defects on 5 qubits in 400 MHz; drawn
# log-uniform in those ranges, 2 to 5 TLS per band.
N_TLS = (2, 5)
G_MHZ, W_MHZ = (0.05, 0.5), (0.5, 20.0)
TLS_GAP_MHZ = 10.0                     # nominal TLS are at least this far apart (design choice)
# Wander between hourly scans: every TLS frequency moves by a normal 1 MHz (spectral diffusion;
# Carroll 2022 sigma 0.16 to 1.06 MHz, Klimov 2018 faster) and its coupling by up to 5%
# (assumption).
WANDER_MHZ, WANDER_G = 1.0, 0.05
# Listed changes. tls_moved: one nominal TLS jumps by 5 to 60 MHz (either way; Klimov 2018
# sees TLS jumps of up to ~60 MHz) and stays. tls_new: a TLS appears, at least TLS_GAP_MHZ from
# every other one, sizes as for nominal TLS (assumption).
MOVE_MHZ = (5.0, 60.0)
KINDS = ("tls_moved", "tls_new")
UNLISTED = "other"
# Hard tier: one nominal TLS switches between two frequencies f_a and f_b = f_a +- (4 to 40)
# MHz, driven by a thermal fluctuator it couples to (Klimov 2018: telegraphic jumps up to ~60
# MHz at rates 50 uHz to 5 mHz; Schlor 2019, arXiv:1901.05352), with mean dwell times 1 to 6
# hours in each state (within Klimov's rates). The hourly scans sample its state; the scan at
# hour -1 (the calibration) shows f_a and at least one of the scans at hours -3 and -2 shows f_b
# (draw rule, so the switching is visible in the history). The rate that counts is the
# long-run average.
SWITCH_MHZ, DWELL_H = (4.0, 40.0), (1.0, 6.0)

# ---------------- measurement ----------------
DELAY_US = 20.0                        # one delay per frequency (assumption; Carroll 2022: 50 us)
SHOTS = 1000                           # repetitions per frequency per scan (Carroll 2022: 1000)
E0, E1 = 0.02, 0.04                    # readout: P(read 1 | 0), P(read 0 | 1)
HOURS = (-3, -2, -1, 0)                # one scan an hour, the last at 0

# ---------------- idle constraint and scoring ----------------
N_FORBID, FORBID_MHZ = 2, (15.0, 35.0)  # windows taken by neighbour collisions (assumption)
EDGE_MHZ = 5.0                          # idle at least this far inside the band
FLOOR_FRAC = 0.1                        # floor F = 10% of the nominal rate at the idle point
CHANGE_MIN, CONTROL_MAX = 2.0, 0.5      # draw rules in units of F (TASK.md)
NOHARM_FREE, NOHARM_ZERO = 0.25, 1.0
TOL_FULL_SIG, TOL_ZERO_SIG = 2.0, 4.0
TIME_LIMIT_S = 120.0
N_INSTANCES, QUICK = 32, 8
TIERS = {"standard": dict(salt=[], public=0, families=("tls_moved", "tls_new", "both")),
         "hard": dict(salt=[5], public=20261127,
                      families=("switch",) * 8 + ("switch_moved",) * 2 + ("switch_new",) * 2
                      + ("tls_moved", "tls_new", "both", "tls_new"))}


def tls_rate(f, tls):
    """Relaxation rate (1/us) from TLS [(f0, g, w), ...] at frequencies f (MHz)."""
    f = np.asarray(f, float)[..., None]
    t = np.asarray(tls, float).reshape(-1, 3)
    return (4 * np.pi * t[:, 1] ** 2 * t[:, 2] / (t[:, 2] ** 2 + (f - t[:, 0]) ** 2)).sum(-1)


def rate(f, bg, tls):
    return bg[0] + bg[1] * np.asarray(f, float) / BAND_MHZ + tls_rate(f, tls)


def p1(f, bg, tls):
    return E0 + (1 - E0 - E1) * np.exp(-DELAY_US * rate(f, bg, tls))


def allowed(f, forbid):
    f = np.asarray(f, float)
    ok = (f >= EDGE_MHZ) & (f <= BAND_MHZ - EDGE_MHZ)
    for lo, hi in forbid:
        ok &= ~((f >= lo) & (f <= hi))
    return ok


def best_idle(gfun, forbid):
    """Allowed frequency on the fine grid with the lowest rate gfun(f)."""
    ok = allowed(FINE, forbid)
    g = gfun(FINE)
    return float(FINE[ok][np.argmin(g[ok])])


# ---------------- instances ----------------
def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _tls(rng, f0):
    return [float(f0), _logu(rng, *G_MHZ), _logu(rng, *W_MHZ)]


def _free_freq(rng, others, lo=0.0, hi=BAND_MHZ):
    for _ in range(1000):
        f = rng.uniform(lo, hi)
        if all(abs(f - o) >= TLS_GAP_MHZ for o in others):
            return float(f)
    return None


def draw_nominal(rng):
    t1 = rng.uniform(*T1_BG_US)
    g0 = 1.0 / t1
    s = rng.uniform(*SLOPE)
    bg = [g0 * (1 - s / 2), g0 * s]          # mean background rate 1/t1 over the band
    tls = []
    for _ in range(int(rng.integers(N_TLS[0], N_TLS[1] + 1))):
        f = _free_freq(rng, [t[0] for t in tls])
        tls.append(_tls(rng, f))
    tls.sort()
    forbid = []
    for _ in range(N_FORBID):
        w = rng.uniform(*FORBID_MHZ)
        lo = rng.uniform(EDGE_MHZ, BAND_MHZ - EDGE_MHZ - w)
        forbid.append([float(lo), float(lo + w)])
    return bg, tls, sorted(forbid)


def _wander(rng, tls):
    return [[f + rng.normal(0, WANDER_MHZ), g * (1 + rng.uniform(-WANDER_G, WANDER_G)), w]
            for f, g, w in tls]


def draw_truth(rng, nom_tls, family):
    """Truth for hours -3..0: (tls per scan, moved index, new TLS, switch) or None to redraw.
    Hour -1 is the nominal calibration state; listed changes happen between hour -1 and 0."""
    base = [list(t) for t in nom_tls]
    moved, new, sw = None, None, None
    others = [t[0] for t in base]
    if family in ("tls_moved", "both", "switch_moved"):
        moved = int(rng.integers(len(base)))
        f = base[moved][0] + rng.choice([-1, 1]) * rng.uniform(*MOVE_MHZ)
        if not 0 <= f <= BAND_MHZ or any(abs(f - o) < TLS_GAP_MHZ
                                         for j, o in enumerate(others) if j != moved):
            return None
        moved_to = float(f)
    if family in ("tls_new", "both", "switch_new"):
        f = _free_freq(rng, others)
        if f is None:
            return None
        new = _tls(rng, f)
    states = [0, 0, 0, 0]
    if family.startswith("switch"):
        k = int(rng.integers(len(base)))
        if k == moved:
            return None
        fb = base[k][0] + rng.choice([-1, 1]) * rng.uniform(*SWITCH_MHZ)
        if not 0 <= fb <= BAND_MHZ or any(abs(fb - o) < TLS_GAP_MHZ / 2
                                          for j, o in enumerate(others) if j != k):
            return None
        da, db = _logu(rng, *DWELL_H), _logu(rng, *DWELL_H)
        # Hour -1 (the calibration) is state 0 (f_a). Walk the two-state chain hour by hour.
        p_stay = (np.exp(-1 / da), np.exp(-1 / db))
        states = [0] * 4
        for d in (1, 0):                       # backwards from hour -1 to -3 (reversible)
            s = states[d + 1]
            states[d] = s if rng.random() < p_stay[s] else 1 - s
        s = states[2]
        states[3] = s if rng.random() < p_stay[s] else 1 - s
        if 1 not in states[:2]:               # an earlier scan shows the other state
            return None
        sw = dict(tls=k, f_a=base[k][0], f_b=float(fb), dwell_h=[da, db],
                  p_b=db / (da + db), states=states)
    days = []
    for d in range(4):
        t = [list(x) for x in base]
        if sw is not None and states[d] == 1:
            t[sw["tls"]][0] = sw["f_b"]
        if d == 3:
            if moved is not None:
                t[moved][0] = moved_to
            if new is not None:
                t.append(list(new))
        days.append(t)
    return days, moved, (moved_to if moved is not None else None), new, sw


def effective_rate(bg, today, sw):
    """The rate that counts at frequency f: today's landscape, or for a switching TLS the
    long-run average over its two states."""
    if sw is None:
        return lambda f: rate(f, bg, today)
    a = [list(t) for t in today]
    b = [list(t) for t in today]
    a[sw["tls"]][0], b[sw["tls"]][0] = sw["f_a"], sw["f_b"]
    return lambda f: (1 - sw["p_b"]) * rate(f, bg, a) + sw["p_b"] * rate(f, bg, b)


def fisher_sigmas(bg, tls, keys):
    """Cramer-Rao standard deviations, from today's scan alone, of the frequency (MHz) and
    log coupling of the TLS indexed by `keys`, with every TLS (f, log g, log w) and both
    background terms free."""
    th = np.array([*bg, *[v for f, g, w in tls for v in (f, np.log(g), np.log(w))]])
    def model(x):
        t = [(x[2 + 3 * i], np.exp(x[3 + 3 * i]), np.exp(x[4 + 3 * i])) for i in range(len(tls))]
        return p1(FREQ, x[:2], t)
    p0 = model(th)
    sd = np.sqrt(np.clip(p0 * (1 - p0), 1e-6, None) / SHOTS)
    J = []
    for i in range(th.size):
        h = 1e-6 if i < 2 else (1e-3 if (i - 2) % 3 == 0 else 1e-4)
        e = np.zeros_like(th)
        e[i] = h
        J.append((model(th + e) - model(th - e)) / (2 * h) / sd)
    J = np.array(J)
    C = np.linalg.pinv(J @ J.T)
    var = np.maximum(np.diag(C), 0)
    return {k: (float(np.sqrt(var[2 + 3 * k])), float(np.sqrt(var[3 + 3 * k]))) for k in keys}


def build(seed, i, family):
    rng = np.random.default_rng([seed, i])
    for _ in range(100 * 200):
        if _ % 200 == 0:                       # a fresh device every 200 tries
            bg, nom, forbid = draw_nominal(rng)
            idle = best_idle(lambda f: rate(f, bg, nom), forbid)
            floor = FLOOR_FRAC * float(rate(idle, bg, nom))
        drawn = draw_truth(rng, nom, "control" if family == "control" else family)
        if drawn is None:
            continue
        days, moved, moved_to, new, sw = drawn
        days = [_wander(rng, d) for d in days]
        geff = effective_rate(bg, days[3], sw)
        f_star = best_idle(geff, forbid)
        gap = float(geff(idle) - geff(f_star)) / floor
        if (family == "control" and gap <= CONTROL_MAX) or \
                (family != "control" and gap >= CHANGE_MIN):
            break
    else:
        raise RuntimeError(f"no draw for {family} (seed {seed}, instance {i})")
    scans = np.array([rng.binomial(SHOTS, p1(FREQ, bg, d)) / SHOTS for d in days])
    kinds = {k for k, on in (("tls_moved", moved is not None), ("tls_new", new is not None),
                             (UNLISTED, sw is not None)) if on}
    keys = ([moved] if moved is not None else []) + ([len(days[3]) - 1] if new is not None else [])
    sig = fisher_sigmas(bg, days[3], keys)
    data = {"freq_mhz": FREQ.copy(), "scans": scans, "hours": list(HOURS), "shots": SHOTS,
            "delay_us": DELAY_US, "readout_error": {"e0": E0, "e1": E1},
            "nominal": {"tls": [{"freq_mhz": f, "g_mhz": g, "width_mhz": w} for f, g, w in nom],
                        "bg_rate_per_us": list(bg)},
            "idle_mhz": idle, "band_mhz": [0.0, BAND_MHZ], "edge_mhz": EDGE_MHZ,
            "forbidden_mhz": forbid, "floor_rate_per_us": floor, "kinds": list(KINDS)}
    truth = dict(bg=bg, days=days, moved=moved, moved_to=moved_to, new=new, switch=sw,
                 kinds=sorted(kinds), sigmas={str(k): v for k, v in sig.items()},
                 best_idle=f_star, gap_floors=gap)
    return dict(family=family, rule="control" if family == "control" else "fix",
                truth=truth, g_stale=float(geff(idle)), g_oracle=float(geff(f_star)),
                floor=floor, data=data)


def instance_list(seed=None, quick=False, tier="standard"):
    t = TIERS[tier]
    fams = ["control"] * (N_INSTANCES // 2) + [t["families"][k % len(t["families"])]
                                               for k in range(N_INSTANCES // 2)]
    master = t["public"] if seed is None else int(seed)
    rng = np.random.default_rng([master, 424_242, *t["salt"]])
    rng.shuffle(fams)
    seeds = rng.integers(0, 2**31, N_INSTANCES)
    out = [(int(s), k, f) for k, (s, f) in enumerate(zip(seeds, fams))]
    if quick:
        ctrl = [o for o in out if o[2] == "control"][:QUICK // 2]
        return ctrl + [o for o in out if o[2] != "control"][:QUICK // 2]
    return out


# ---------------- scoring ----------------
def parse_answer(out, forbid):
    f = float(out["idle_mhz"])
    if not np.isfinite(f) or not allowed(f, forbid):
        raise ValueError("idle_mhz must be a finite allowed frequency (inside the band by "
                         "edge_mhz, outside every forbidden window)")
    d = out["diagnosis"]
    if not isinstance(d["changed"], bool) or not isinstance(d.get("kinds", {}), dict):
        raise ValueError("diagnosis.changed must be bool and diagnosis.kinds a dict")
    kinds = d.get("kinds", {})
    if not set(kinds) <= {*KINDS, UNLISTED}:
        raise ValueError(f"unknown kind in {sorted(kinds)}")
    return f, d["changed"], kinds


def sig_credit(err, sigma):
    return float(np.clip((TOL_ZERO_SIG - abs(err) / max(sigma, 1e-12))
                         / (TOL_ZERO_SIG - TOL_FULL_SIG), 0, 1))


def _num(x, default=np.inf):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def param_credit(tr, kinds):
    """Mean over the true listed kinds of their parameter credits."""
    parts = []
    if tr["moved"] is not None:
        rep = kinds.get("tls_moved") or {}
        sf = tr["sigmas"][str(tr["moved"])][0]
        ok = rep.get("tls") == tr["moved"]
        f_true = tr["days"][3][tr["moved"]][0]
        parts.append(ok * sig_credit(_num(rep.get("freq_mhz")) - f_true, sf))
    if tr["new"] is not None:
        rep = kinds.get("tls_new") or {}
        k = len(tr["days"][3]) - 1
        f_true, g_true, _ = tr["days"][3][k]
        sf, sg = tr["sigmas"][str(k)]
        g_rep = _num(rep.get("g_mhz"), 1e-9)
        parts.append(sig_credit(_num(rep.get("freq_mhz")) - f_true, sf) *
                     sig_credit(np.log(max(g_rep, 1e-9)) - np.log(g_true), sg))
    return float(np.mean(parts)) if parts else 1.0


def run_instance(solution_path, seed, i, family, tier):
    inst = build(seed, i, family)
    data = dict(inst.pop("data"), tier=tier)
    forbid = data["forbidden_mhz"]
    ok, out, secs = run_solver(solution_path, data, TIME_LIMIT_S)
    tr = inst["truth"]
    rec = dict(seed=seed, **inst, solve_s=round(secs, 2))
    try:
        if not ok:
            raise RuntimeError(out)
        f_sol, changed, kinds = parse_answer(out, forbid)
    except Exception as exc:  # noqa: BLE001
        rec.update(error=f"{type(exc).__name__}: {exc}"[:300], changed=None, fix=0.0,
                   score=0.0, rubric=rubric.invalid())
        return rec
    geff = effective_rate(tr["bg"], tr["days"][3], tr["switch"])
    g_sol = float(geff(f_sol))
    if inst["rule"] == "fix":
        fix = float(np.clip((inst["g_stale"] - g_sol) / (inst["g_stale"] - inst["g_oracle"]),
                            0, 1))
    else:
        excess = (g_sol - inst["g_stale"]) / inst["floor"]
        fix = float(np.clip((NOHARM_ZERO - excess) / (NOHARM_ZERO - NOHARM_FREE), 0, 1))
    true_k, rep_k = set(tr["kinds"]), set(kinds)
    kind = len(true_k & rep_k) / len(true_k | rep_k) if true_k | rep_k else 1.0
    par = (param_credit(tr, kinds) if true_k & set(KINDS) else kind) \
        if inst["rule"] == "fix" else 1.0
    checks = rubric.instance_checks(inst["rule"], changed == (inst["rule"] == "fix"), fix,
                                    kind, par)
    rec.update(idle_sol=f_sol, g_sol=g_sol, changed=changed, reported=kinds, fix=fix,
               kind=kind, params=par, score=rubric.total(checks), rubric=checks)
    return rec


def evaluate(solution_path, quick=False, seed=None, tier="standard"):
    recs = [run_instance(solution_path, s, i, f, tier)
            for s, i, f in instance_list(seed, quick, tier)]
    ctrl = [r for r in recs if r["rule"] == "control"]
    drift = [r for r in recs if r["rule"] != "control"]
    summary = dict(score=float(np.mean([r["score"] for r in recs])),
                   false_alarm_rate=float(np.mean([r["changed"] is not False for r in ctrl])),
                   miss_rate=float(np.mean([r["changed"] is not True for r in drift])),
                   n_fix=len(drift), n_control=len(ctrl),
                   errors=sum("error" in r for r in recs), tier=tier, rubric=rubric.VERSION)
    return {"score": summary["score"], "summary": summary, "instances": recs}


if __name__ == "__main__":
    import sys
    print(f"{evaluate(sys.argv[1] if len(sys.argv) > 1 else '../sandbox/solution.py')['score']:.4f}")
