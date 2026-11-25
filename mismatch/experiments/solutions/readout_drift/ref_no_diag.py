"""Ablation: the reference's discriminator and changed flag, with no drift families named."""
from _load import load


def solve(data):
    out = load("reference").solve(data)
    out["diagnosis"] = {"changed": out["diagnosis"]["changed"], "kinds": {}}
    return out
