"""No-effort baseline: trust the nominal model and report no change."""

import stim


def solve(data: dict) -> dict:
    dem = stim.Circuit(data["circuit"]).detector_error_model(decompose_errors=True)
    return {"dem": str(dem), "diagnosis": {"changed": False, "kind": "none"}}
