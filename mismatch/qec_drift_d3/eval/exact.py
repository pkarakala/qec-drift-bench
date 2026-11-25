"""Exact, sampling-free logical error rates for the d = 3, 3-round circuit (24 detectors).

A detector error model is a list of independent mechanisms, each flipping a set of detectors and
maybe the logical observable. The joint distribution of (syndrome, observable) over all 2^25
outcomes is computed with two Walsh-Hadamard transforms (the characteristic function of a sum of
independent bits over GF(2) is a product). A decoder's logical error rate is then
sum_s P(s, not decoder(s)), summed over the most likely syndromes until the left-out probability
mass is below TAIL (1e-6), which bounds the error of every rate by TAIL.

Per-instance tables are cached on disk between runs, encrypted with a key derived from the
instance seed, so a cache file says nothing without the seed. Licence: Apache-2.0.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from io import BytesIO as _BytesIO
from pathlib import Path

import numpy as np
import pymatching
import stim

ND = 24
TAIL = 1e-6
CACHE_DIR = Path(tempfile.gettempdir()) / "qec_drift_d3_cache"
CACHE_MAX_BYTES = 4 << 30
_CODE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def _wht(x: np.ndarray) -> np.ndarray:
    """In-place unnormalized Walsh-Hadamard transform of a length 2^n float64 vector."""
    n, h = x.size, 1
    while h < n:
        y = x.reshape(-1, 2, h)
        a = y[:, 0, :].copy()
        y[:, 0, :] += y[:, 1, :]
        y[:, 1, :] *= -1
        y[:, 1, :] += a
        h *= 2
    return x


def joint_table(dem: stim.DetectorErrorModel) -> np.ndarray:
    """P[s, o] for every syndrome s (bit d = detector d) and observable value o; shape (2^24, 2)."""
    f = np.zeros(1 << (ND + 1))
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        idx = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                idx ^= 1 << (t.val + 1)
            elif t.is_logical_observable_id():
                idx ^= 1
        if idx:
            f[idx] += np.log1p(-2 * inst.args_copy()[0])
    total = f.sum()
    g = _wht(f)  # log chi(k) = (total - WHT(f)(k)) / 2
    g -= total
    g *= -0.5
    np.exp(g, out=g)
    p = _wht(g)
    p /= p.size
    np.clip(p, 0.0, None, out=p)
    return p.reshape(-1, 2)


class Table:
    """The likeliest syndromes of one model and their (P(s, 0), P(s, 1))."""

    def __init__(self, idx: np.ndarray, p: np.ndarray):
        self.idx, self.p = idx, p
        self.bits = ((idx[:, None] >> np.arange(ND)) & 1).astype(np.uint8)
        self.tail = float(max(0.0, 1.0 - p.sum()))

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> "Table":
        full = joint_table(dem)
        mass = full.sum(1)
        order = np.argsort(mass)  # ascending; drop the least likely while their sum < TAIL
        cut = int(np.searchsorted(np.cumsum(mass[order]), TAIL))
        idx = np.sort(order[cut:]).astype(np.int32)
        return cls(idx, full[idx].astype(np.float32))

    def ler(self, pred: np.ndarray) -> float:
        """Logical error rate of predictions pred[k] for syndrome idx[k] (lower bound; the
        left-out mass self.tail is at most TAIL)."""
        pred = np.asarray(pred, dtype=np.int64).reshape(-1)
        return float(self.p[np.arange(len(pred)), 1 - pred].astype(np.float64).sum())

    def decode(self, m: pymatching.Matching) -> np.ndarray:
        return m.decode_batch(self.bits)[:, 0].astype(np.uint8)

    def ler_of(self, m: pymatching.Matching) -> float:
        return self.ler(self.decode(m))


def _keystream(key: bytes, n: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key + b"stream").digest(), "little")
    return np.frombuffer(np.random.default_rng(seed).bytes(n), dtype=np.uint8)


def _path(key: bytes) -> Path:
    return CACHE_DIR / (hashlib.sha256(key + _CODE.encode()).hexdigest() + ".bin")


def cache_load(key: bytes) -> dict | None:
    path = _path(key)
    try:
        raw = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    raw ^= _keystream(key, raw.size)
    try:
        with np.load(_BytesIO(raw.tobytes())) as z:
            out = {k: z[k] for k in z.files}
        os.utime(path)
        return out
    except Exception:  # corrupt or stale file: rebuild
        return None


def cache_save(key: bytes, arrays: dict) -> None:
    buf = _BytesIO()
    np.savez(buf, **arrays)
    raw = np.frombuffer(buf.getvalue(), dtype=np.uint8).copy()
    raw ^= _keystream(key, raw.size)
    try:
        CACHE_DIR.mkdir(mode=0o700, exist_ok=True)
        tmp = _path(key).with_suffix(".tmp%d" % os.getpid())
        raw.tofile(tmp)
        os.replace(tmp, _path(key))
        files = sorted(CACHE_DIR.glob("*.bin"), key=lambda f: f.stat().st_mtime)
        size = sum(f.stat().st_size for f in files)
        while files and size > CACHE_MAX_BYTES:
            size -= files[0].stat().st_size
            files.pop(0).unlink()
    except OSError:
        pass  # a cache is an optimisation only

