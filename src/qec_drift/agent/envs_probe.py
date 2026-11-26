"""Isolation probe (acceptance gate A6, docs/task-standard.md): run as a solution, report which
leak channels are open.

It never answers. On every instance it raises `RuntimeError("PROBE open=<names>")`, so the
grader records the list as that instance's error and scores it 0; scripts/accept_task.py
reads it back.
Each channel is something a submitted solution could use to learn the answer without solving
the task. Licence: Apache-2.0.

Channels:
- env_vars: a variable naming the seed or the task is visible.
- ancestor_seed: a parent process's command line or environment holds a seed.
- parent_memory: the grader's memory can be read through /proc (it holds the hidden truth).
- grader_source: a task grader's files (eval/harness.py) can be read from the solution's folder
  or any folder above it.
- grader_writable: those files can be written (a solution could rewrite its own grader).
- stored_results: stored reports or other runs' solutions can be read.
- inherited_fds: open file descriptors other than stdin, stdout and stderr.
- network: an outside host answers.
"""

import glob
import os
import socket


def _ancestors(n=8):
    pid = os.getppid()
    for _ in range(n):
        if pid <= 1:
            return
        yield pid
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return


def _read(path, n=-1):
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except OSError:
        return None


def _parent_memory():
    pid = os.getppid()
    maps = _read(f"/proc/{pid}/maps")
    if not maps:
        return False
    for line in maps.decode(errors="replace").splitlines():
        rng, perms = line.split()[:2]
        if perms.startswith("rw"):
            lo = int(rng.split("-")[0], 16)
            try:
                with open(f"/proc/{pid}/mem", "rb") as f:
                    f.seek(lo)
                    return len(f.read(64)) > 0
            except (OSError, ValueError, OverflowError):
                return False
    return False


def _bases(n=6):
    here = os.path.dirname(os.path.abspath(__file__))
    return [os.path.abspath(os.path.join(here, *[".."] * up)) for up in range(n)]


def _grader_files():
    found = []
    for base in _bases():
        for depth in ("", "*", os.path.join("*", "*")):
            found += glob.glob(os.path.join(base, depth, "eval", "harness.py"))
    return sorted(set(found))


def _network():
    for host, port in (("1.1.1.1", 53), ("8.8.8.8", 53), ("example.com", 443)):
        try:
            socket.create_connection((host, port), timeout=2).close()
            return True
        except OSError:
            continue
    return False


def channels():
    open_ = []
    if any("MISMATCH" in k or "SEED" in k for k in os.environ):
        open_.append("env_vars")
    for pid in _ancestors():
        cmd = (_read(f"/proc/{pid}/cmdline") or b"").split(b"\0")
        env = _read(f"/proc/{pid}/environ") or b""
        if b"--seed" in cmd or b"MISMATCH_SEED=" in env:
            open_.append("ancestor_seed")
            break
    if _parent_memory():
        open_.append("parent_memory")
    files = _grader_files()
    if any(_read(f, 1) is not None for f in files):
        open_.append("grader_source")
    if any(os.access(f, os.W_OK) for f in files):
        open_.append("grader_writable")
    for base in _bases():
        if glob.glob(os.path.join(base, "data", "*", "reports")):
            open_.append("stored_results")
            break
    fds = (
        [int(x) for x in os.listdir("/proc/self/fd") if x.isdigit()]
        if os.path.isdir("/proc/self/fd")
        else []
    )
    if [
        fd
        for fd in fds
        if fd > 2
        and os.path.exists(f"/proc/self/fd/{fd}")
        and not os.readlink(f"/proc/self/fd/{fd}").startswith("/proc/")
    ]:
        open_.append("inherited_fds")
    if _network():
        open_.append("network")
    return open_


def solve(data):
    raise RuntimeError("PROBE open=" + (",".join(channels()) or "none"))
