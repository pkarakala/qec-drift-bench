"""Starter solution for coherence_tls. Edit freely; keep the solve(data) -> dict interface.
As shipped it behaves like the trivial baseline: it keeps the current idle frequency."""


def solve(data):
    idle = data["idle_mhz"]  # chosen at the last calibration, from data["nominal"]

    # TODO: compare the latest scan data["scans"][-1] (and the earlier ones) with what the
    #       nominal TLS list predicts and decide whether the idle frequency must move (TASK.md).
    # TODO: return the allowed frequency where the qubit will now relax slowest.
    return {"idle_mhz": idle, "diagnosis": {"changed": False, "kinds": {}}}
