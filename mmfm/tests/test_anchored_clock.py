"""Addendum v2-2 harness: anchored clock puts exactly N_h simulated events in (0, h]; count quantile is a valid inverse CDF."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.rollout2 import Rollout2, cnt_quantile  # noqa: E402


def _fake(knots, n_ev, seed=0):
    R = Rollout2.__new__(Rollout2)
    R.clock, R.t0, R.knot_s = "anchored", 1_000_000_000, np.array([0.0, 1.0, 5.0, 10.0])
    rng = np.random.default_rng(seed)
    tau = np.cumsum(np.where(rng.random(n_ev) < 0.3, 0, rng.exponential(2000, n_ev)))  # bursty internal clock, ties
    p = dict(t=list(R.t0 + np.r_[0, tau]), knots=np.array(knots), us=int(R.t0 + tau[-1]))
    return R, p


def test_anchored_counts_equal_knots():
    for knots in ([0, 40, 300, 700], [0, 0, 0, 5], [0, 10, 10, 50], [0, 120, 600, 1300]):
        R, p = _fake(knots, knots[-1] + 200)
        t = R.times(p)
        assert np.all(np.diff(t) >= 0), "output times must be monotone"
        for k, h in enumerate((1.0, 5.0, 10.0)):
            assert int(np.sum(t[1:] - R.t0 <= h * 1e6)) == knots[k + 1], (knots, h)
        assert t[-1] > R.t0 + 10e6  # extrapolated events lie after the last knot


def test_anchored_last_time_incomplete():
    R, p = _fake([0, 40, 300, 700], 100)  # path still inside segment (1 s, 5 s]
    assert R.last_time(p) == R.t0 + 1e6
    t = R.times(p)
    assert np.all(t[41:] - R.t0 <= 5e6) and np.all(np.diff(t) >= 0)


def test_cnt_quantile_inverse_cdf():
    e = np.log1p(np.array([0, 3, 10, 30, 100, 300], dtype=float))
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(len(e) + 1))
    fc = dict(p_cnt=p, cnt_edges=e)
    u = np.sort(rng.random(4000))
    n = np.array([cnt_quantile(fc, x) for x in u])
    assert np.all(np.diff(n) >= 0), "quantile must be monotone in u"
    b = np.searchsorted(e, np.log1p(n.astype(float)))
    freq = np.bincount(b, minlength=len(p)) / len(n)
    assert np.abs(freq - p).max() < 0.03, (freq, p)
