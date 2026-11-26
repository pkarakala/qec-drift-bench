"""Which data-out tasks stay in the benchmark: the best agent must pass under the threshold.

    uv run python scripts/select_tasks.py                  # every graded arm in data/envs_hard
    uv run python scripts/select_tasks.py data/envs_hard/opus_5_5_cc.parquet ...

An arm table is a graded parquet from agent/envs_grade.py, or a CSV with the same columns (env,
model, rollout, outcome, pass_rate, score). data/envs_hard/pilot_heldout_2026-11-26.csv holds
the agent pilot of docs/envs-results.md, one row per rollout, graded by the same pass rule.

A task (one task folder in envs/, hard tier) is kept when the best model's pass rate is below
`threshold` in envs/config.json (0.75). The pass rate is `awareness.passed` per instance on the
paper seeds (agent/envs_grade.py), averaged over the model's rollouts. A rollout that ended
without a solution counts at the floor, as its starter scores. A task with no agent runs is
"pending": it is kept only provisionally, on the evidence that the reference (the strongest
known method) is below the threshold. This is a selection rule, not the per-run relative rule
of scripts/launch_scores.py.

Each model's pass rate also gets a 95% interval when its per-instance table is present
(<arm>_instances.parquet from agent/envs_grade.py, or a CSV with env, model, rollout, seed,
instance, passed). Instances share a seed's draw of devices, so the interval is a two-level
cluster bootstrap: seeds resampled first, instances within each chosen seed second, rollouts
pooled (Davison and Hinkley 1997, Sec. 3.8; Field and Welsh 2007,
doi:10.1111/j.1467-9868.2007.00593.x). The decision stays on the point estimate; a task whose
best interval spans the threshold is marked a close call.

Writes data/envs_hard/selection.json and docs/envs-selection.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qec_drift import paths
from qec_drift.agent import envs_grade as eg
from qec_drift.agent.envs_kit import TASKS

DOC = paths.REPO_ROOT / "docs" / "envs-selection.md"
BOOT = dict(resamples=10000, level=0.95, seed=20261126)


def _read(t: Path) -> pd.DataFrame:
    return pd.read_csv(t) if t.suffix == ".csv" else pd.read_parquet(t)


def instance_frame(tables: list[Path]) -> pd.DataFrame:
    """Per-instance pass flags (env, model, seed, passed) on the paper seeds. A table from
    envs_grade.py names its arm, not its model; the model comes from the arm table beside it."""
    frames = []
    for t in tables:
        df = _read(t).rename(columns={"task": "env"})
        if "model" not in df:
            arm = t.with_name(t.name.replace("_instances", ""))
            df["model"] = _read(arm).model.iloc[0]
        frames.append(df[["env", "model", "seed", "passed"]])
    if not frames:
        return pd.DataFrame(columns=["env", "model", "seed", "passed"])
    df = pd.concat(frames, ignore_index=True)
    return df[df.seed.astype(str) != "public"]


def cluster_bootstrap(by_seed: list[np.ndarray], rng: np.random.Generator) -> tuple[float, float]:
    """Percentile interval of the mean: seeds resampled, then instances within each seed."""
    boots = np.empty(BOOT["resamples"])
    for i in range(BOOT["resamples"]):
        pick = rng.integers(0, len(by_seed), len(by_seed))
        boots[i] = np.mean(
            np.concatenate([by_seed[j][rng.integers(0, len(by_seed[j]), len(by_seed[j]))]
                            for j in pick])
        )  # fmt: skip
    alpha = (1 - BOOT["level"]) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return float(lo), float(hi)


def intervals(inst: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    rng = np.random.default_rng(BOOT["seed"])
    out = {}
    for (env, model), g in inst.groupby(["env", "model"], sort=True):
        by_seed = [s.passed.to_numpy(float) for _, s in g.groupby(g.seed.astype(str), sort=True)]
        out[(env, model)] = cluster_bootstrap(by_seed, rng)
    return out


def model_rates(tables: list[Path]) -> pd.DataFrame:
    """Per (task, model): mean pass rate and native score over rollouts, and the rollout count."""
    frames = [_read(t) for t in tables]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty or "pass_rate" not in df:
        return pd.DataFrame(columns=["env", "model", "pass_rate", "score", "rollouts"])
    df = df[df.outcome != "error"].dropna(subset=["pass_rate"])
    return (
        df.groupby(["env", "model"])
        .agg(pass_rate=("pass_rate", "mean"), score=("score", "mean"), rollouts=("rollout", "size"))
        .reset_index()
    )


def decide(
    task: str, rates: pd.DataFrame, refs: dict, threshold: float, ci: dict | None = None
) -> dict:
    ref = refs.get(task, {}).get("reference", {})
    mine = rates[rates.env == task]
    if mine.empty:
        keep = (ref.get("pass_rate") or 1.0) < threshold
        return dict(task=task, status="pending" if keep else "excluded", best=None, best_pass=None)
    best = mine.loc[mine.pass_rate.idxmax()]
    status = "included" if best.pass_rate < threshold else "excluded"
    out = dict(task=task, status=status, best=best.model, best_pass=round(float(best.pass_rate), 4))
    span = (ci or {}).get((task, best.model))
    if span:
        out["best_interval"] = [round(span[0], 4), round(span[1], 4)]
        out["close_call"] = span[0] < threshold <= span[1]
    return out


def markdown(
    table: list[dict], rates: pd.DataFrame, refs: dict, threshold: float, ci: dict | None = None
) -> str:
    models = sorted(rates.model.unique())
    head = ["task", "status", "reference", *models]
    lines = [
        f"# Data-out task selection: keep a task if the best agent passes under {threshold:.0%}",
        "",
        "Generated by scripts/select_tasks.py; do not edit. Hard tier, paper seeds. Each cell: "
        "pass rate (awareness.passed per instance) / native score (trivial 0.50), rollouts "
        "averaged; [95% interval] where per-instance results exist. A close call is a task whose "
        "best interval spans the threshold.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for t in table:
        ref = refs.get(t["task"], {}).get("reference", {})
        cells = [t["task"], t["status"], _cell(ref.get("pass_rate"), ref.get("score"))]
        for m in models:
            r = rates[(rates.env == t["task"]) & (rates.model == m)]
            span = (ci or {}).get((t["task"], m))
            cells.append(_cell(*r[["pass_rate", "score"]].iloc[0], span) if len(r) else "not run")
        if t.get("close_call"):
            cells[1] += " (close call)"
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _cell(p: float | None, s: float | None, span: tuple | None = None) -> str:
    if p is None:
        return "-"
    return f"{p:.2f}" + (f" [{span[0]:.2f}, {span[1]:.2f}]" if span else "") + f" / {s:.2f}"


def _span(ci: dict, task: str, model: str) -> dict:
    span = ci.get((task, model))
    return dict(interval=[round(span[0], 4), round(span[1], 4)]) if span else {}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("tables", nargs="*", type=Path)
    a = ap.parse_args(argv)
    every = sorted([*eg.OUT.glob("*.parquet"), *eg.OUT.glob("*.csv")])
    tables = a.tables or [t for t in every if not t.stem.endswith(("_instances", "_failure_modes"))]
    ci = intervals(instance_frame([t for t in every if t.stem.endswith("_instances")]))
    threshold = eg.CONFIG["threshold"]
    refs_file = eg.OUT / "references.json"
    refs = json.loads(refs_file.read_text())["tasks"] if refs_file.exists() else eg.references()
    rates = model_rates(tables)
    table = [decide(task, rates, refs, threshold, ci) for task in TASKS]
    for t in table:
        mine = rates[rates.env == t["task"]].round(4)
        t["models"] = {r.model: dict(pass_rate=r.pass_rate, score=r.score, rollouts=int(r.rollouts),
                                     **_span(ci, t["task"], r.model))
                       for r in mine.itertuples()}  # fmt: skip
    eg.OUT.mkdir(parents=True, exist_ok=True)
    (eg.OUT / "selection.json").write_text(
        json.dumps(dict(threshold=threshold, config=eg.CONFIG["version"], tasks=table), indent=1)
        + "\n"
    )
    DOC.write_text(markdown(table, rates, refs, threshold, ci))
    print(DOC.read_text())


if __name__ == "__main__":
    main()
