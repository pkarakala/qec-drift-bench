"""Interval estimates used by analyze.py. Licence: Apache-2.0.

- Scores: a two-level (cluster) bootstrap. Instances share a seed's draw of devices, so seeds are
  resampled first and instances within each chosen seed second (Davison and Hinkley 1997,
  Sec. 3.8; Field and Welsh 2007, doi:10.1111/j.1467-9868.2007.00593.x). Percentile intervals.
- Paired differences between two solutions use the same resampled indices for both, so the
  interval reflects only how the two differ on the same instances.
- Rates (false alarm, miss) use the Wilson score interval (Wilson 1927,
  doi:10.1080/01621459.1927.10502953) from scipy.stats.binomtest; we pool instances and note
  that this ignores clustering by seed, which the bootstrap does not.
"""

import numpy as np
from scipy.stats import binomtest


def cluster_bootstrap(values: dict, n: int, rng: np.random.Generator, level: float,
                      other: dict | None = None):
    """values: {seed: array of per-instance numbers}. Returns (mean, lo, hi).

    With `other` (same keys, same shapes), returns the interval of mean(values - other)."""
    keys = sorted(values)
    a = [np.asarray(values[k], float) for k in keys]
    if other is not None:
        a = [x - np.asarray(other[k], float) for k, x in zip(keys, a)]
    point = float(np.mean(np.concatenate(a)))
    if len(keys) == 1:  # one seed: resample instances only
        x = a[0]
        boots = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    else:
        boots = np.empty(n)
        for i in range(n):
            pick = rng.integers(0, len(keys), len(keys))
            boots[i] = np.mean(np.concatenate(
                [a[j][rng.integers(0, len(a[j]), len(a[j]))] for j in pick]))
    alpha = (1 - level) / 2
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return point, float(lo), float(hi)


def wilson(k: int, n: int, level: float):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    ci = binomtest(k, n).proportion_ci(confidence_level=level, method="wilson")
    return k / n, float(ci.low), float(ci.high)
