"""Load this task's eval/_reference.py by path, so a ceiling extends the reference instead of
copying it. Loading by path is a maintainer shortcut for these fixed baselines; a submitted
solution cannot see eval/."""
import importlib.util
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", TASK / "eval" / f"_{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
