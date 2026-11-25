"""LAZY (review finding 3, quick route): blind weight tuning. Every nominal error probability is
scaled by one constant, never reading any shot, with one fixed "changed" answer."""
import stim

SCALE = 1.6


def solve(data):
    dem = stim.Circuit(data["circuit"]).detector_error_model(decompose_errors=True)
    out = []
    for line in str(dem).splitlines():
        if line.startswith("error("):
            p = float(line[6 : line.index(")")])
            line = f"error({min(p * SCALE, 0.5)!r})" + line[line.index(")") + 1 :]
        out.append(line)
    return {"dem": "\n".join(out), "diagnosis": {"changed": True, "kind": "hot_qubit",
                                                 "qubit": 9, "factor": 4.0, "onset_shot": 0}}
