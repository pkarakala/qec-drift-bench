"""Load this environment's eval/_reference.py and eval/_trivial.py by path.

The ablations below reuse the reference instead of copying it. Loading by path is a
maintainer shortcut for these fixed baselines; a submitted solution cannot see eval/.
"""
import importlib.util
from pathlib import Path

ENV = Path(__file__).resolve().parents[3] / Path(__file__).resolve().parent.name


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", ENV / "eval" / f"_{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
