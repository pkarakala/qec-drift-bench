"""Agent-pilot aggregation: one row per (model, environment) from the per-rollout runs.

Called by analyze.py. Agent solutions are named agent__<model>__r<k> (see run.solutions). For each
model and environment:
  score        mean over rollouts of the paper-seed score; the interval is the two-level cluster
               bootstrap (seeds, then instances) of the per-instance score averaged over rollouts,
               so it reflects instance and seed sampling for these rollouts, not rollout luck
  rollouts     each rollout's paper-seed mean (the spread across rollouts is reported apart)
  normalized   (score - trivial) / (reference - trivial): 0 = no better than the constant
               answer, 1 = as good as the reference solution
  false alarm, miss   pooled over rollouts, Wilson intervals
Only instances present in every rollout of that model are used, so rollouts stay paired.
Licence: Apache-2.0.
"""

from collections import defaultdict

import numpy as np

from stats import cluster_bootstrap, wilson


def short_model(m: str) -> str:
    return "".join(ch for ch in m.removeprefix("claude-") if ch.isalnum())


def summarize_agents(recs, S, paper, boot):
    rng = np.random.default_rng(boot["rng_seed"] + 1)
    n, lev = boot["resamples"], boot["level"]
    runs = defaultdict(lambda: defaultdict(dict))  # (env, model) -> rollout -> (seed, i) -> rec
    for r in recs:
        if r["kind"] == "agent" and r["seed"] in paper:
            _, model, rollout = r["solution"].split("__")
            runs[(r["env"], model)][rollout][(r["seed"], r["index"])] = r
    triv = defaultdict(dict)
    for r in recs:
        if r["solution"] == "trivial" and r["seed"] in paper:
            triv[r["env"]][(r["seed"], r["index"])] = r["score"]
    out = defaultdict(dict)
    for (env, model), rolls in sorted(runs.items()):
        keys = set.intersection(*(set(v) for v in rolls.values())) & set(triv[env])
        if not keys:
            continue
        keys = sorted(keys)
        per_inst = defaultdict(list)
        per_inst_triv = defaultdict(list)
        for k in keys:
            per_inst[k[0]].append(float(np.mean([v[k]["score"] for v in rolls.values()])))
            per_inst_triv[k[0]].append(triv[env][k])
        m, lo, hi = cluster_bootstrap(dict(per_inst), n, rng, lev)
        dt = cluster_bootstrap(dict(per_inst), n, rng, lev, other=dict(per_inst_triv))
        pooled = [v[k] for v in rolls.values() for k in keys]
        ctrl = [r for r in pooled if not r["truth_changed"]]
        chg = [r for r in pooled if r["truth_changed"]]
        t, ref = S[env]["trivial"]["score"][0], S[env]["reference"]["score"][0]
        out[env][model] = dict(
            n_rollouts=len(rolls), n_seeds=len(per_inst), n_instances=len(keys),
            score=[m, lo, hi], minus_trivial=list(dt),
            rollouts={ro: float(np.mean([v[k]["score"] for k in keys]))
                      for ro, v in sorted(rolls.items())},
            normalized=(m - t) / (ref - t) if ref > t else None,
            false_alarm=list(wilson(sum(r["said"] is not False for r in ctrl), len(ctrl), lev)),
            miss=list(wilson(sum(r["said"] is not True for r in chg), len(chg), lev)),
            errors=sum(r["error"] for r in pooled))
    return dict(out)


def tex_agents(A, label, path):
    rows = []
    for env, d in A.items():
        rows.append(r"\midrule" + "\n" + rf"\multicolumn{{6}}{{l}}{{\textit{{{label[env]}}}}} \\")
        for model, a in d.items():
            spread = list(a["rollouts"].values())
            norm = "--" if a["normalized"] is None else f"{a['normalized']:.2f}"
            rows.append(
                f"\\texttt{{{model}}} & {a['score'][0]:.2f} [{a['score'][1]:.2f}, "
                f"{a['score'][2]:.2f}] & {min(spread):.2f}--{max(spread):.2f} & {norm} & "
                f"{100 * a['false_alarm'][0]:.0f} & {100 * a['miss'][0]:.0f} \\\\")
    path.write_text(
        "\\begin{tabular}{llllrr}\n\\toprule\nmodel & score & rollout range & normalized & "
        "FA \\% & miss \\% \\\\\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def numbers(A, short_env):
    for env, d in A.items():
        for model, a in d.items():
            b = f"{short_env[env]}.agent.{short_model(model)}"
            yield f"{b}.score", f"{a['score'][0]:.2f}"
            yield f"{b}.ci", "[{:.2f}, {:.2f}]".format(*a["score"][1:])
            yield f"{b}.dt", f"{a['minus_trivial'][0]:+.2f}"
            if a["normalized"] is not None:
                yield f"{b}.norm", f"{a['normalized']:.2f}"
            yield f"{b}.fa", f"{100 * a['false_alarm'][0]:.0f}"
            yield f"{b}.miss", f"{100 * a['miss'][0]:.0f}"
            yield f"{b}.nroll", str(a["n_rollouts"])


def figure(A, S, label, fig_dir):
    import matplotlib.pyplot as plt
    if not A:
        return
    models = sorted({m for d in A.values() for m in d})
    fig, axes = plt.subplots(1, len(A), figsize=(7.0, 0.5 + 0.32 * len(models) + 0.9),
                             sharex=True, squeeze=False)
    for ax, (env, d) in zip(axes[0], A.items()):
        t, ref = S[env]["trivial"]["score"][0], S[env]["reference"]["score"][0]
        ax.axvline(t, color="#8c8c8c", lw=0.6, ls=":")
        ax.axvline(ref, color="#1f5fa8", lw=0.6, ls="--")
        for y, model in enumerate(models):
            if model not in d:
                continue
            a = d[model]
            m, lo, hi = a["score"]
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", ms=3, color="#6b3fa0",
                        capsize=1.5, lw=0.8)
            ax.scatter(list(a["rollouts"].values()), [y] * len(a["rollouts"]), s=8,
                       facecolors="none", edgecolors="#6b3fa0", lw=0.5)
        ax.set_yticks(range(len(models)), models, fontsize=6)
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.set_title(label[env], fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("score")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"agents.{ext}", dpi=200)
    plt.close(fig)
