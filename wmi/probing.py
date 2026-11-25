"""Probes with mandatory controls.

A probe accuracy is never reported alone. `probe_with_controls` always returns:
  real            probe on true labels
  shuffled        same pipeline, train/val labels permuted (should be ~chance)
  control_task    Hewitt & Liang (2019) control task: each state *type* gets a
                  fixed random label; selectivity = real - control_task
  majority        constant predictor fitted on train (classification) / train mean (regression)
Hyper-parameters are selected on VAL only. CIs are cluster-bootstrapped over
test *episodes*, because timesteps within an episode are not independent.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

CS = (0.01, 0.1, 1.0, 10.0)
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def _metric(kind, y, p):
    return float(np.mean(y == p)) if kind == "classification" else float(r2_score(y, p))


def fit_probe(Xtr, ytr, Xva, yva, Xte, kind: str, seed: int = 0):
    sc = StandardScaler().fit(Xtr)
    Xtr_, Xva_, Xte_ = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best = (-np.inf, None, None)
    for h in (CS if kind == "classification" else ALPHAS):
        if kind == "classification":
            if len(np.unique(ytr)) < 2:
                continue
            m = LogisticRegression(C=h, max_iter=3000, random_state=seed)
        else:
            m = Ridge(alpha=h)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m.fit(Xtr_, ytr)
        v = _metric(kind, yva, m.predict(Xva_))
        if v > best[0]:
            best = (v, h, m)
    if best[2] is None:  # degenerate single-class train set
        const = np.full(len(Xte), ytr[0])
        return {"val": float("nan"), "hparam": None, "test_pred": const, "model": None, "scaler": sc}
    return {"val": best[0], "hparam": best[1], "test_pred": best[2].predict(Xte_), "model": best[2], "scaler": sc}


def cluster_bootstrap(kind, y, p, groups, rng, n=1000, alpha=0.05):
    ug = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    stats = []
    for _ in range(n):
        gs = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        stats.append(_metric(kind, y[idx], p[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def control_labels(types_all: np.ndarray, y_pool: np.ndarray, rng) -> dict:
    """Fixed random label per unique type, drawn from the empirical train label distribution."""
    uniq = np.unique(types_all, axis=0)
    labs = rng.choice(y_pool, size=len(uniq), replace=True)
    return {tuple(u): l for u, l in zip(uniq.tolist(), labs.tolist())}


def probe_with_controls(X: dict, y: dict, types: dict, groups_test: np.ndarray, kind: str,
                        rng: np.random.Generator, n_boot: int = 1000) -> dict:
    """X, y, types: dicts with keys train/val/test."""
    out = {}

    def run(ytr, yva, yte, name):
        r = fit_probe(X["train"], ytr, X["val"], yva, X["test"], kind, seed=int(rng.integers(1 << 30)))
        s = _metric(kind, yte, r["test_pred"])
        lo, hi = cluster_bootstrap(kind, yte, r["test_pred"], groups_test, rng, n=n_boot)
        out[name] = {"test": s, "ci95": [lo, hi], "val": r["val"], "hparam": r["hparam"]}
        return r

    real = run(y["train"], y["val"], y["test"], "real")
    run(rng.permutation(y["train"]), rng.permutation(y["val"]), y["test"], "shuffled")

    cmap = control_labels(np.concatenate([types["train"], types["val"], types["test"]]), y["train"], rng)
    c = {k: np.array([cmap[tuple(t)] for t in types[k].tolist()]) for k in ("train", "val", "test")}
    run(c["train"], c["val"], c["test"], "control_task")

    if kind == "classification":
        vals, cnt = np.unique(y["train"], return_counts=True)
        maj = np.full(len(y["test"]), vals[np.argmax(cnt)])
    else:
        maj = np.full(len(y["test"]), float(np.mean(y["train"])))
    out["majority"] = {"test": _metric(kind, y["test"], maj)}
    out["selectivity"] = out["real"]["test"] - out["control_task"]["test"]
    out["_real_fit"] = real  # not serialised; used for interventions
    return out
