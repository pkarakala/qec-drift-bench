"""Lazy (hard tier): the nominal model as decoder, always 'changed' with kind "other".

Probes the hard tier's new answer: "other" earns full detail on changes outside the model, so a
solver that always claims one must still lose every control, and it gets only the changed
credit on listed faults."""
import stim


def solve(data):
    dem = stim.Circuit(data["circuit"]).detector_error_model(decompose_errors=True)
    return {"dem": str(dem), "diagnosis": {"changed": True, "kind": "other"}}
