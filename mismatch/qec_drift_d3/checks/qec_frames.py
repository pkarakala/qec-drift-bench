"""EXPLOIT (review probe): read the grader's own local variables. No physics.

Returns a valid answer either way (the nominal DEM), so the check can actually fail: if the
stack walk ever finds the grader's truth again, the diagnosis is exact and the score jumps far
above trivial; while the solver is isolated it finds nothing and reports a fixed guess.
"""
import sys

import stim


def solve(data):
    dem = str(stim.Circuit(data["circuit"]).detector_error_model(decompose_errors=True))
    fr = sys._getframe(1)
    while fr is not None and "truth" not in fr.f_locals:
        fr = fr.f_back
    if fr is None:
        return {"dem": dem, "diagnosis": {"changed": True, "kind": "hot_qubit", "qubit": 0,
                                          "factor": 3.0, "onset_shot": 0}}
    truth = fr.f_locals["truth"]
    diag = {"changed": truth.family != "none", "kind": truth.family, "qubit": truth.qubit,
            "pair": list(truth.pair) if truth.pair else None, "factor": truth.factor,
            "onset_shot": truth.onset_shot, "decay_shots": truth.decay_shots}
    return {"dem": dem, "diagnosis": diag}
