from __future__ import annotations

import math

import numpy as np


def bootstrap_ci(values_or_fn, n: int = 1000, alpha: float = 0.05, rng=None, data=None):
    """Percentile bootstrap.

    Either pass a 1-D array of per-sample scores (CI of the mean), or a metric
    function `fn(idx)` plus `data` length via `data=n_samples`.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    if callable(values_or_fn):
        m = int(data)
        stats = [values_or_fn(rng.integers(0, m, m)) for _ in range(n)]
    else:
        v = np.asarray(values_or_fn, dtype=float)
        stats = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def wilson_ci(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
