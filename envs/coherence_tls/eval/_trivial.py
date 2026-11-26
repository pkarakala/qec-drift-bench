"""No-effort baseline: keep the current idle frequency and report no change. Licence: Apache-2.0."""


def solve(data):
    return {"idle_mhz": data["idle_mhz"], "diagnosis": {"changed": False, "kinds": {}}}
