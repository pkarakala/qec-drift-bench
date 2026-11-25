"""Ablation: the reference's pulse and changed flag, with the design numbers as the estimate."""
from _load import load


def solve(data):
    out = load("reference").solve(data)
    out["diagnosis"] = {"changed": out["diagnosis"]["changed"],
                        "drive_scale": data["drive_scale_design_ghz"],
                        "f01_ghz": data["f01_design_ghz"], "compression": 0.0}
    return out
