"""Matching-engine invariants + agreement between the two matching implementations (replay vs closed loop)."""
import numpy as np

from m3.engine import Book, apply_event, replay


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


def test_apply_event_edge_cases():
    b = Book(bids={5799: 100}, asks={5801: 100})
    assert apply_event(b, 5801, 0, 1, 1) == (0, 0, False) and apply_event(b, 5801, -50, 0, 0) == (0, 0, False)
    assert b.bids == {5799: 100} and b.asks == {5801: 100}, "q<=0 must be a no-op (review #11)"
    assert apply_event(b, 5790, 300, 0, 1) == (0, 0, False) and b.bids == {5799: 100}  # cancel at empty level
    e = Book(bids={5799: 100}, asks={})
    tr, _, mk = apply_event(e, 5810, 500, 1, 1)  # buy into empty ask side: nothing trades, posts as bid
    assert tr == 0 and not mk and e.bids[5810] == 500


def test_replay_uses_the_single_engine():
    """replay() and the closed loop share engine.apply_event (one matching implementation)."""
    import inspect
    from m3 import closedloop, engine
    assert "apply_event(" in inspect.getsource(engine.replay)
    assert "apply_event(" in inspect.getsource(closedloop.ClosedLoop._apply_generated)
    assert "_apply(" not in inspect.getsource(closedloop)
