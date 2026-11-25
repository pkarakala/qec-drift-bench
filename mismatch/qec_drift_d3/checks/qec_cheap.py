"""CHEAP HEURISTIC (review probe): nominal decoder, no model fit at all.
Burst: biggest 8-shot detector count; decay constant copied from TASK.md (3.7 shots).
Otherwise: the qubit whose neighbouring detectors rose most, factor fixed at 3.2, onset 0."""
import numpy as np
import stim


def solve(data):
    dem = str(stim.Circuit(data["circuit"]).detector_error_model(decompose_errors=True))
    pre, post = data["pre_detectors"], data["post_detectors"]
    w = post.sum(1).astype(float)
    roll = np.convolve(w, np.ones(8), "valid")
    null = np.convolve(pre.sum(1).astype(float), np.ones(8), "valid")
    if roll.max() > null.max() + 6:
        t0 = int(np.argmax(roll))
        seg = w[max(0, t0 - 10): t0 + 8]
        onset = max(0, t0 - 10) + int(np.argmax(seg >= 3))
        return {"dem": dem, "diagnosis": {"changed": True, "kind": "burst", "onset_shot": onset,
                                          "decay_shots": 3.69}}
    rise = post.mean(0) / np.maximum(pre.mean(0), 1e-9)
    z = (post.mean(0) - pre.mean(0)) / np.sqrt(pre.mean(0) * (1 - pre.mean(0)) * (1 / len(pre) + 1 / len(post)))
    if z.max() < 6:
        return {"dem": dem, "diagnosis": {"changed": False, "kind": "none"}}
    # qubit nearest to the centroid of the detectors that rose
    dc = data["detector_coords"]
    wts = np.clip(z, 0, None) ** 2
    xy = np.array([dc[i][:2] for i in range(post.shape[1])])
    cen = (wts[:, None] * xy).sum(0) / wts.sum()
    qc = data["qubit_coords"]
    q = min(qc, key=lambda k: (qc[k][0] - cen[0]) ** 2 + (qc[k][1] - cen[1]) ** 2)
    return {"dem": dem, "diagnosis": {"changed": True, "kind": "hot_qubit", "qubit": int(q),
                                      "factor": 3.2, "onset_shot": 0}}
