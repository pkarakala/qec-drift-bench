"""Your solution. Edit `solve`; see TASK.md for the data and the return contract.

This starter trusts the nominal model and reports no change, which is the trivial baseline.
TODO: compare post_detectors with pre_detectors, decide whether and how the device changed,
refit the model, diagnose.
"""

import stim


def solve(data: dict) -> dict:
    dem = stim.Circuit(data["circuit"]).detector_error_model(decompose_errors=True)
    return {"dem": str(dem), "diagnosis": {"changed": False, "kind": "none"}}
