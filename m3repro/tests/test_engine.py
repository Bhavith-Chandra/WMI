"""Matching-engine invariants + agreement between the two matching implementations (replay vs closed loop)."""
import math

import numpy as np

from m3.closedloop import ClosedLoop
from m3.engine import TICK, Book, replay


def _random_stream(rng, n=3000, mid=5800):
    px = mid + rng.integers(-15, 16, n)
    qty = rng.integers(1, 30, n) * 100
    act = (rng.random(n) < 0.7).astype(int)
    side = rng.integers(0, 2, n)
    dt = rng.exponential(0.05, n)
    return px, qty, dt, act, side


def _book0():
    b = Book()
    for k in range(1, 11):
        b.asks[5800 + k] = 1000
        b.bids[5800 - k] = 1000
    return b


def _depth(b):
    return sum(b.bids.values()) + sum(b.asks.values())


def test_never_crossed_and_quantity_conserved():
    rng = np.random.default_rng(0)
    for _ in range(5):
        b0 = _book0()
        px, qty, dt, act, side = _random_stream(rng)
        b1, s = replay(b0, px, qty, dt, act, side)
        ok = np.isfinite(s.bid) & np.isfinite(s.ask)
        assert np.all(s.bid[ok] < s.ask[ok]), "crossed book"
        added = int(qty[act == 1].sum())
        removed_by_cancel = s.cancel_qty - s.cancel_unmatched_qty
        assert _depth(b1) == _depth(b0) + added - 2 * s.traded_qty - removed_by_cancel, "quantity not conserved"


def test_price_priority_multi_level_fill():
    b = Book(bids={5799: 100}, asks={5801: 100, 5802: 200, 5805: 300})
    b1, s = replay(b, np.array([5802]), np.array([250]), np.array([0.0]), np.array([1]), np.array([1]))
    assert s.traded_qty == 250
    assert b1.asks == {5802: 50, 5805: 300}  # lowest first, partial at 5802, 5805 untouched (above limit)
    assert 5802 not in b1.bids  # fully filled -> nothing posted


def test_closedloop_matching_equals_replay_engine():
    """Two independent matching implementations must agree event by event."""
    rng = np.random.default_rng(1)
    px, qty, dt, act, side = _random_stream(rng, 2000)
    anchor = 58.0
    b0 = _book0()
    b_ref, s = replay(b0, px, qty, dt, act, side)
    cl = ClosedLoop.__new__(ClosedLoop)
    cl.anchor, cl.books, cl.t = anchor, [b0.copy()], np.zeros(1)
    cl.mid_path, cl.n_ev, cl.n_dt0 = [[]], np.zeros(1, int), np.zeros(1, int)
    for i in range(len(px)):
        f = np.array([px[i] * TICK / anchor - 1.0, math.log1p(qty[i]), dt[i], act[i], side[i]])
        cl._apply(0, f)
    assert cl.books[0].bids == b_ref.bids and cl.books[0].asks == b_ref.asks
    mids = np.array([m for _, m in cl.mid_path[0]])
    assert np.allclose(mids, s.mid, equal_nan=True)
