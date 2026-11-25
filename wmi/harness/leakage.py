"""Leakage guards. These raise; they do not warn.

Unit of independence is the *episode*: samples from one episode are correlated,
so train/val/test must be split by episode id, never by timestep.
"""
from __future__ import annotations

import hashlib

import numpy as np


class LeakageError(RuntimeError):
    pass


def assert_disjoint(a, b, what: str = "ids") -> None:
    inter = np.intersect1d(np.asarray(a), np.asarray(b))
    if len(inter):
        raise LeakageError(f"{len(inter)} {what} shared between splits, e.g. {inter[:5].tolist()}")


def _row_hashes(x: np.ndarray) -> set:
    x = np.ascontiguousarray(x)
    return {hashlib.sha1(r.tobytes()).hexdigest() for r in x}


def observation_overlap(train_x: np.ndarray, test_x: np.ndarray) -> float:
    """Fraction of test rows that appear byte-identically in train.

    In small discrete worlds this is expected to be > 0 (states recur), so it is
    *reported*, not enforced, unless the caller asks for strict disjointness.
    """
    tr = _row_hashes(train_x)
    te = [hashlib.sha1(r.tobytes()).hexdigest() for r in np.ascontiguousarray(test_x)]
    return float(np.mean([h in tr for h in te])) if te else 0.0


def check_split(split: dict, n_total: int | None = None) -> dict:
    """Assert pairwise disjointness of every split; optionally full coverage.

    `split` maps name -> array of episode ids. Returns a small report dict that
    should be stored alongside results.
    """
    names = list(split)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert_disjoint(split[names[i]], split[names[j]], f"episode ids ({names[i]}/{names[j]})")
    if n_total is not None:
        covered = np.unique(np.concatenate([np.asarray(v) for v in split.values()]))
        if len(covered) != n_total:
            raise LeakageError(f"split covers {len(covered)} of {n_total} episodes")
    return {k: int(len(v)) for k, v in split.items()}
