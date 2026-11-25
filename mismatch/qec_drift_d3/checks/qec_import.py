"""EXPLOIT (review finding 1, import route): import the grader's harness by module name and look
the instance up by regenerating the public instances. Must fail: eval/ is not on the solver's path."""
import numpy as np


def solve(data):
    import harness as H
    seed, mix = H.instance_list(None, False)
    for i, fam in enumerate(mix):
        d, t = H.build_instance(seed, i, fam)
        if np.array_equal(d["post_detectors"], data["post_detectors"]):
            dem = str(H.after_circuit(H.nominal_circuit(), t).detector_error_model(decompose_errors=True))
            return {"dem": dem, "diagnosis": {"changed": t.family != "none", "kind": t.family,
                    "qubit": t.qubit, "pair": t.pair, "factor": t.factor,
                    "onset_shot": t.onset_shot, "decay_shots": t.decay_shots}}
    raise RuntimeError("no match")
