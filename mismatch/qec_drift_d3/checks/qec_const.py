"""CONSTANT GUESS (review probe, finding 2): nominal decoder, same diagnosis on every instance."""
import stim


def solve(data):
    dem = stim.Circuit(data["circuit"]).detector_error_model(decompose_errors=True)
    return {"dem": str(dem), "diagnosis": {"changed": True, "kind": "hot_qubit", "qubit": 12,
                                           "factor": 3.2, "onset_shot": 0}}
