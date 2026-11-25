"""Helper for the lazy checks: load eval/_reference.py by path (a shortcut for the probe only)."""
import importlib.util
from pathlib import Path


def reference():
    path = Path(__file__).resolve().parent.parent / "eval" / "_reference.py"
    spec = importlib.util.spec_from_file_location("qec_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
