"""Grading rubric for qec_drift_d3: every instance's score as named, weighted checks.

One fixed rubric grades every instance, solver, seed and tier; harness.evaluate scores with it,
and each instance's report carries its checks. Every check is computed by script from the hidden
truth and the solver's answer (no LLM judge). A check is a gate (it must pass, value 1, or the
instance scores 0), weighted (it adds weight x value, value in [0, 1]) or a multiplier (it scales
the weighted sum). The weights are the ones TASK.md states. Licence: Apache-2.0.
"""

from __future__ import annotations

VERSION = "qec_drift_d3/rubric-v1"
RULES = ("control", "fix", "diagnosis_only")
W_CHANGED, W_FIX, W_DIAG = 0.3, 0.35, 0.35
DIAG_PARTS = ("kind", "location", "size", "timing")

# id: (what failing it means, {rule: weight or role}). A diagnosis part weighs W_DIAG (fix) or
# W_FIX + W_DIAG (diagnosis only) times its share for the true kind (harness.detail_parts).
GATE, MULT = "gate", "multiplier"
_ALL = dict.fromkeys(RULES, GATE)
_DIAG = {"fix": "0.35 x share", "diagnosis_only": "0.7 x share"}
CHECKS = {
    "valid_answer": ("crashed, timed out, missing or ill-typed fields, or a DEM that fails "
                     "validation", _ALL),
    "change_call": ("false alarm on a control, or a missed change (gate)",
                    {"control": 1.0, "fix": W_CHANGED, "diagnosis_only": W_CHANGED}),
    "fix_gap": ("the returned decoder model does not close the decoding gap the change opened",
                {"fix": W_FIX}),
    "diag_kind": ("wrong kind of change (for an unlisted change: a listed kind named instead "
                  "of other)", _DIAG),
    "diag_location": ("wrong qubit or gate pair", _DIAG),
    "diag_size": ("factor outside the tolerance", _DIAG),
    "diag_timing": ("onset, decay or dwell time outside the tolerance", _DIAG),
    "no_harm": ("the shipped decoder is worse than both anchors (1% free, none at 10%)",
                {"control": MULT, "diagnosis_only": MULT}),
}


def check(value: float, weight: float = 0.0, gate: bool = False, mult: bool = False,
          part: bool = False) -> dict:
    out = {"v": float(value), "w": float(weight)}
    for k, on in (("gate", gate), ("mult", mult), ("part", part)):
        if on:
            out[k] = True
    return out


def total(checks: dict) -> float:
    """0 if a gate fails; else the weighted sum times every multiplier."""
    if any(c.get("gate") and c["v"] < 1 for c in checks.values()):
        return 0.0
    s = sum(c["w"] * c["v"] for c in checks.values())
    for c in checks.values():
        if c.get("mult"):
            s *= c["v"]
    return float(s)


def invalid() -> dict:
    return {"valid_answer": check(0.0, gate=True)}


def instance_checks(rule: str, call_right: bool, fix: float | None, harm: float,
                    parts: dict) -> dict:
    """parts: {part: (earned, available)} from harness.detail_parts."""
    c = {"valid_answer": check(1.0, gate=True),
         "change_call": check(call_right, CHECKS["change_call"][1][rule], gate=True)}
    if rule == "fix":
        c["fix_gap"] = check(fix, W_FIX)
    if rule != "control":
        w = W_DIAG if rule == "fix" else W_FIX + W_DIAG
        for p in DIAG_PARTS:
            got, avail = parts[p]
            if avail > 0:
                c[f"diag_{p}"] = check(got / avail, w * avail)
    if rule != "fix":
        c["no_harm"] = check(harm, mult=True)
    return c
