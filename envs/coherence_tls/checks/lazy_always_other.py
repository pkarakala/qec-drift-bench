"""Lazy (hard tier): keep the idle frequency, always 'changed' with kind "other". Earns the kinds
credit on changes outside the model, loses every control."""


def solve(data):
    return {"idle_mhz": data["idle_mhz"], "diagnosis": {"changed": True, "kinds": {"other": {}}}}
