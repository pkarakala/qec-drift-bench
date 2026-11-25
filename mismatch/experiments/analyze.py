"""Turn experiments/results/raw into the paper's tables and figures.

usage: python experiments/analyze.py [--tier hard]

Reads every report that run.py wrote and writes
  experiments/results/instances.csv   one row per (env, solution, seed, instance)
  experiments/results/summary.json    every number the paper quotes
  experiments/results/summary.md      the same, readable
  paper/tables/*.tex                  LaTeX tables
  paper/figures/*.pdf, *.png          figures
Headline numbers use the paper seeds only; the public seed is the smoke-test set used during
development and is reported apart. `--tier hard` reads results/raw_hard/ and writes only
instances_hard.csv, summary_hard.json and summary_hard.md. Licence: Apache-2.0.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import agents
from run import raw_dir, solutions
from stats import cluster_bootstrap, wilson

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "experiments" / "config.json").read_text())
RAW = ROOT / "experiments" / "results" / "raw"  # standard tier; see run.raw_dir
OUT = ROOT / "experiments" / "results"
TAB = ROOT / "paper" / "tables"
FIG = ROOT / "paper" / "figures"
BOOT = CONFIG["bootstrap"]
LABEL = {"qec_drift_d3": "QEC (d=3 surface code)", "readout_drift": "Readout",
         "pulse_wrong_model": "Pulse (X gate)"}
MARGIN = 0.05  # same gate as run_all.py
TIER = "hard"  # set by --tier (this track: hard only)


# ------------------------------------------------------------------ per-environment readers
def rows_qec(rep):
    for r in rep["instances"]:
        fam = r["family"]
        yield dict(family=fam, rule=r["rule"], truth_changed=fam != "none",
                   said=r.get("changed"), fix=r.get("fix"), diag=r.get("detail"),
                   score=r["score"], error="error" in r)


def rows_readout(rep):
    for r in rep["instances"]:
        kinds = sorted(r["truth"])
        fam = "none" if not kinds else kinds[0] if len(kinds) == 1 else "two_families"
        right = r.get("changed") == bool(kinds)
        diag = None
        if "kind" in r:
            diag = right * (0.15 * r["kind"] + 0.10 * r["magnitude"]) / 0.25
        yield dict(family=fam, rule=r["rule"], truth_changed=bool(kinds), said=r.get("changed"),
                   fix=r.get("fix"), diag=diag, score=r["score"], error="error" in r)


def rows_pulse(rep):
    for r in rep["instances"]:
        yield dict(family=r["kind"], rule=r["rule"], truth_changed=r["kind"] != "control",
                   said=r.get("changed"), fix=r.get("fix"), diag=r.get("diag"),
                   score=r["score"], error="error" in r)


READERS = {"qec_drift_d3": rows_qec, "readout_drift": rows_readout,
           "pulse_wrong_model": rows_pulse}


def load():
    recs = []
    for env in CONFIG["envs"]:
        for name, (_rel, kind) in solutions(env, TIER).items():
            for f in sorted((raw_dir(TIER) / env / name).glob("*.json")):
                rep = json.loads(f.read_text())
                for i, (inst, row) in enumerate(zip(rep["instances"], READERS[env](rep))):
                    # Pair instances across solutions by their own seed where the report has
                    # one (readout's order is shuffled per seed since 2026-11-17), else by order.
                    key = inst.get("seed", i) if env == "readout_drift" else i
                    recs.append(dict(env=env, solution=name, kind=kind, seed=f.stem, index=key,
                                     **row))
    return recs


# ------------------------------------------------------------------ summaries
def by_seed(recs, field="score"):
    """Scores grouped by seed, in instance-key order, so paired differences line up."""
    d = defaultdict(list)
    for r in sorted(recs, key=lambda r: (r["seed"], r["index"])):
        d[r["seed"]].append(r[field])
    return d


def summarize(recs):
    rng = np.random.default_rng(BOOT["rng_seed"])
    n, lev = BOOT["resamples"], BOOT["level"]
    paper = {str(s) for s in CONFIG["paper_seeds"]}
    groups = defaultdict(list)
    for r in recs:
        groups[(r["env"], r["solution"])].append(r)
    out = {}
    for env in CONFIG["envs"]:
        triv = [r for r in groups[(env, "trivial")] if r["seed"] in paper]
        ref = [r for r in groups[(env, "reference")] if r["seed"] in paper]
        if not triv or not ref:
            print(f"skipping {env}: no paper-seed trivial or reference runs yet")
            continue
        envd = {}
        for name, (_rel, kind) in solutions(env, TIER).items():
            allr = groups[(env, name)]
            rs = [r for r in allr if r["seed"] in paper]
            pub = [r for r in allr if r["seed"] == "public"]
            if not rs:
                continue
            common = sorted({r["seed"] for r in rs} & {r["seed"] for r in triv})
            sc = {s: v for s, v in by_seed(rs).items() if s in common}
            m, lo, hi = cluster_bootstrap(sc, n, rng, lev)
            d_t = cluster_bootstrap(sc, n, rng, lev, other={s: by_seed(triv)[s] for s in common})
            d_r = None
            if name != "reference" and ref:
                cr = sorted(set(common) & {r["seed"] for r in ref})
                d_r = cluster_bootstrap({s: sc[s] for s in cr}, n, rng, lev,
                                        other={s: by_seed(ref)[s] for s in cr})
            ctrl = [r for r in rs if not r["truth_changed"]]
            chg = [r for r in rs if r["truth_changed"]]
            fa = wilson(sum(r["said"] is not False for r in ctrl), len(ctrl), lev)
            miss = wilson(sum(r["said"] is not True for r in chg), len(chg), lev)
            fams = defaultdict(list)
            for r in rs:
                fams[r["family"]].append(r["score"])
            fam = {}
            for f, v in sorted(fams.items()):
                fam[f] = dict(n=len(v), mean=float(np.mean(v)))
            seed_means = [float(np.mean(v)) for v in sc.values()]
            envd[name] = dict(
                kind=kind, n_seeds=len(sc), n_instances=sum(map(len, sc.values())),
                score=[m, lo, hi], minus_trivial=list(d_t),
                minus_reference=list(d_r) if d_r else None,
                false_alarm=list(fa), miss=list(miss), families=fam,
                seed_sd=float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else None,
                seed_means=dict(zip(sc, seed_means)),
                public=float(np.mean([r["score"] for r in pub])) if pub else None,
                errors=sum(r["error"] for r in rs),
                gate_fail=kind in ("lazy", "leak", "data_ablation") and d_t[0] > MARGIN)
        out[env] = envd
    return out


# ------------------------------------------------------------------ writers
def fmt(x, lo=None, hi=None, d=2):
    if lo is None:
        return f"{x:.{d}f}"
    return f"{x:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def write_csv(recs):
    keys = ["env", "solution", "kind", "seed", "index", "family", "rule", "truth_changed",
            "said", "fix", "diag", "score", "error"]
    with open(OUT / f"instances{SUFFIX[TIER]}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(recs)


def write_md(S):
    title = "" if TIER == "standard" else f", {TIER} tier"
    lines = [f"# Results{title} (generated by experiments/analyze.py; do not edit)", "",
             f"Config {CONFIG['version']}. Mean over paper seeds with 95% cluster-bootstrap "
             "intervals; rates with Wilson intervals.", ""]
    for env, d in S.items():
        lines += [f"## {LABEL[env]}", "",
                  "| solution | kind | score | minus trivial | minus reference | false alarm | miss | public |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, s in d.items():
            mr = fmt(*s["minus_reference"]) if s["minus_reference"] else "-"
            pub = fmt(s["public"]) if s["public"] is not None else "-"
            flag = " (GATE FAIL)" if s["gate_fail"] else ""
            lines.append(f"| {name}{flag} | {s['kind']} | {fmt(*s['score'])} | "
                         f"{fmt(*s['minus_trivial'])} | {mr} | {fmt(*s['false_alarm'])} | "
                         f"{fmt(*s['miss'])} | {pub} |")
        lines.append("")
    (OUT / f"summary{SUFFIX[TIER]}.md").write_text("\n".join(lines))


def tex_main(S):
    show = ["trivial", "reference", "ref_no_fix", "ref_no_diag", "reference_swapped_data"]
    rows = []
    for env, d in S.items():
        heur = [k for k, v in d.items() if v["kind"] == "heuristic"]
        lazy = [k for k, v in d.items() if v["kind"] in ("lazy", "leak")]
        best_lazy = max(lazy, key=lambda k: d[k]["score"][0]) if lazy else None
        names = [k for k in show[:2] + heur + show[2:] if k in d]
        rows.append(r"\midrule" + "\n" + rf"\multicolumn{{5}}{{l}}{{\textit{{{LABEL[env]}}}}} \\")
        for k in names + ([best_lazy] if best_lazy else []):
            s = d[k]
            label = k.replace("_", r"\_")
            if k == best_lazy:
                label = rf"best of {len(lazy)} lazy/leak (\texttt{{{label}}})"
            else:
                label = rf"\texttt{{{label}}}"
            rows.append(f"{label} & {fmt(*s['score'])} & {fmt(*s['minus_trivial'])} & "
                        f"{fmt(*s['false_alarm'])} & {fmt(*s['miss'])} \\\\")
    body = "\n".join(rows)
    (TAB / "main.tex").write_text(
        "\\begin{tabular}{lllll}\n\\toprule\nsolution & score & $-$ trivial & false alarm & miss"
        " \\\\\n" + body + "\n\\bottomrule\n\\end{tabular}\n")


def tex_families(S):
    rows = []
    for env, d in S.items():
        heur = [k for k, v in d.items() if v["kind"] == "heuristic"][:1]
        cols = ["trivial", "reference"] + heur + ["ref_no_fix"]
        cols = [c for c in cols if c in d]
        fams = list(d["reference"]["families"])
        rows.append(r"\midrule" + "\n" + rf"\multicolumn{{6}}{{l}}{{\textit{{{LABEL[env]}}}"
                    + (rf" (heuristic: \texttt{{{heur[0].replace('_', chr(92) + '_')}}})"
                       if heur else "") + r"} \\")
        for f in fams:
            vals = [d[c]["families"].get(f, {}).get("mean", float("nan")) for c in cols]
            n = d["reference"]["families"][f]["n"]
            cells = vals + [float("nan")] * (4 - len(vals))
            cells = ["--" if np.isnan(v) else f"{v:.2f}" for v in cells]
            rows.append(f"\\texttt{{{f.replace('_', chr(92) + '_')}}} & {n} & "
                        + " & ".join(cells) + " \\\\")
    (TAB / "families.tex").write_text(
        "\\begin{tabular}{lrrrrr}\n\\toprule\nfamily & $n$ & trivial & reference & heuristic & "
        "ref.\\ no fix \\\\\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tex_checks(S):
    rows = []
    for env, d in S.items():
        for k, s in d.items():
            if s["kind"] not in ("lazy", "leak", "data_ablation"):
                continue
            ok = r"\textbf{fail}" if s["gate_fail"] else "pass"
            rows.append(f"{LABEL[env].split(' ')[0]} & \\texttt{{{k.replace('_', chr(92) + '_')}}}"
                        f" & {s['kind'].replace('_', ' ')} & {fmt(*s['minus_trivial'])} & {ok} \\\\")
    (TAB / "checks.tex").write_text(
        "\\begin{tabular}{lllll}\n\\toprule\nenv & solution & kind & score $-$ trivial & gate"
        " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tex_numbers(S, A):
    """Every number the prose quotes, as \\res{env.solution.field} (paper/numbers.tex)."""
    short = {"qec_drift_d3": "qec", "readout_drift": "ro", "pulse_wrong_model": "pulse"}
    out = ["% generated by experiments/analyze.py; do not edit",
           "\\providecommand{\\res}[1]{\\csname num@#1\\endcsname}"]

    def put(key, val):
        out.append(f"\\expandafter\\def\\csname num@{key}\\endcsname{{{val}}}")
    for env, d in S.items():
        e = short[env]
        for k, s in d.items():
            b = f"{e}.{k.replace('_', '')}"
            m, lo, hi = s["score"]
            put(f"{b}.score", f"{m:.2f}")
            put(f"{b}.ci", f"[{lo:.2f}, {hi:.2f}]")
            put(f"{b}.dt", f"{s['minus_trivial'][0]:+.2f}")
            if s["minus_reference"]:
                put(f"{b}.dr", f"{s['minus_reference'][0]:+.2f}")
                put(f"{b}.drci", "[{:.2f}, {:.2f}]".format(*s["minus_reference"][1:]))
            put(f"{b}.fa", f"{100 * s['false_alarm'][0]:.0f}")
            put(f"{b}.miss", f"{100 * s['miss'][0]:.0f}")
            if s["seed_sd"] is not None:
                put(f"{b}.seedsd", f"{s['seed_sd']:.3f}")
            put(f"{b}.n", str(s["n_instances"]))
            for f, v in s["families"].items():
                put(f"{b}.fam.{f.replace('_', '')}", f"{v['mean']:.2f}")
        put(f"{e}.nseeds", str(d["reference"]["n_seeds"]))
    checks = [s for d in S.values() for s in d.values() if s["kind"] in ("lazy", "leak")]
    put("all.nchecks", str(len(checks)))
    if checks:
        put("all.maxcheckdt", f"{max(s['minus_trivial'][0] for s in checks):+.2f}")
    put("all.nfail", str(sum(s["gate_fail"] for d in S.values() for s in d.values())))
    for key, val in agents.numbers(A, short):
        put(key, val)
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(out) + "\n")


def figures(S, recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"floor": "#8c8c8c", "reference": "#1f5fa8", "heuristic": "#d97a1c",
              "ablation": "#5aa36b", "data_ablation": "#a83a32", "lazy": "#b8b8b8",
              "leak": "#b8b8b8"}
    # Figure 1: score per solution with intervals, one panel per environment.
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), sharex=True)
    for ax, (env, d) in zip(axes, S.items()):
        order = sorted((k for k in d if d[k]["kind"] != "agent"),
                       key=lambda k: d[k]["score"][0])
        for y, k in enumerate(order):
            m, lo, hi = d[k]["score"]
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", ms=3,
                        color=colors[d[k]["kind"]], capsize=1.5, lw=0.8)
        ax.axvline(d["trivial"]["score"][0], color="#8c8c8c", lw=0.6, ls=":")
        ax.set_yticks(range(len(order)), [k.replace("_", " ") for k in order], fontsize=6)
        ax.set_title(LABEL[env], fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("score")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k.replace("_", " "))
               for k, c in colors.items() if k != "leak"]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"scores.{ext}", dpi=200)
    plt.close(fig)

    # Figure 2: per-seed score of reference, best heuristic and trivial (stability).
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0), sharey=True)
    for ax, (env, d) in zip(axes, S.items()):
        heur = [k for k, v in d.items() if v["kind"] == "heuristic"][:1]
        for x, k in enumerate(["trivial", "reference"] + heur):
            v = list(d[k]["seed_means"].values())
            ax.scatter(np.full(len(v), x) + np.linspace(-0.12, 0.12, len(v)), v, s=6,
                       color=colors[d[k]["kind"]])
        ax.set_xticks(range(2 + len(heur)), ["trivial", "reference"]
                      + [h.replace("_", " ") for h in heur], fontsize=6)
        ax.set_title(LABEL[env], fontsize=8)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("score per seed")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"seeds.{ext}", dpi=200)
    plt.close(fig)


SUFFIX = {"standard": "", "hard": "_hard"}


def main():
    global TIER
    if "--tier" in sys.argv:
        TIER = sys.argv[sys.argv.index("--tier") + 1]
    for p in (OUT, TAB, FIG):
        p.mkdir(parents=True, exist_ok=True)
    recs = load()
    write_csv(recs)
    S = summarize(recs)
    (OUT / f"summary{SUFFIX[TIER]}.json").write_text(json.dumps(S, indent=1))
    write_md(S)
    if TIER != "standard":
        paper = {str(s) for s in CONFIG["paper_seeds"]}
        A = agents.summarize_agents(recs, S, paper, BOOT)
        if A:
            (OUT / f"agents{SUFFIX[TIER]}.json").write_text(json.dumps(A, indent=1))
        print((OUT / f"summary{SUFFIX[TIER]}.md").read_text())
        return
    tex_main(S)
    tex_families(S)
    tex_checks(S)
    paper = {str(s) for s in CONFIG["paper_seeds"]}
    A = agents.summarize_agents(recs, S, paper, BOOT)
    (OUT / "agents.json").write_text(json.dumps(A, indent=1))
    agents.tex_agents(A, LABEL, TAB / "agents.tex")
    tex_numbers(S, A)
    figures(S, recs)
    agents.figure(A, S, LABEL, FIG)
    fails = [f"{e}/{k}" for e, d in S.items() for k, s in d.items() if s["gate_fail"]]
    print((OUT / "summary.md").read_text())
    if fails:
        print("GATE FAIL:", ", ".join(fails))


if __name__ == "__main__":
    main()
