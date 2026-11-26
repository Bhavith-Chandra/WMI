"""Regression tests for every Bitstamp quirk found while verifying the reconstruction (synthetic messages)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.l3 import ADD, BUY, CANCEL, SELL, TAKE, IdBook, rebuild  # noqa: E402

SNAP = {"bids": [["100.00", "1.0", "1"], ["99.99", "2.0", "2"]], "asks": [["100.02", "1.5", "3"], ["100.03", "3.0", "4"]],
        "microtimestamp": "1000"}


def o(ev, us, oid, typ, price, amount, traded="0"):
    return (us * 1000, {"event": ev, "channel": "live_orders_x", "data": {
        "id": oid, "order_type": typ, "price": price, "amount": amount, "amount_traded": traded,
        "microtimestamp": str(us)}})


def t(us, typ, price, amount, buy, sell):
    return (us * 1000, {"event": "trade", "channel": "live_trades_x", "data": {
        "type": typ, "price": price, "amount": amount, "buy_order_id": buy, "sell_order_id": sell,
        "microtimestamp": str(us)}})


def run(msgs):
    return rebuild("x", SNAP, 1000, msgs, [], [], 0, tick=0.01, lockstep_every=1)


def rows(r):
    return [(int(e["type"]), int(e["side"]), round(float(e["price"]), 2), round(float(e["qty"]), 6)) for e in r["events"]]


def test_plain_add_and_cancel():
    r = run([o("order_created", 2000, 9, 0, 99.98, 0.5), o("order_deleted", 3000, 9, 0, 99.98, 0.5)])
    assert rows(r) == [(ADD, BUY, 99.98, 0.5), (CANCEL, BUY, 99.98, 0.5)]
    assert r["lockstep"]["final_top50_match"] == {"bid": 1.0, "ask": 1.0}


def test_trade_before_created_and_full_fill_no_phantom_add():
    """Trade (other channel) arrives first; created carries full amount; delete with amount 0 = full fill."""
    r = run([t(2000, 0, 100.02, 1.5, 9, 3), o("order_created", 2000, 9, 0, 100.05, 1.5),
             o("order_deleted", 2000, 3, 1, 100.02, 0, "1.5"), o("order_deleted", 2000, 9, 0, 100.05, 0, "1.5")])
    assert rows(r) == [(TAKE, BUY, 100.02, 1.5)]
    assert r["final_book"].levels(SELL, 1)[0][0] == 100.03
    assert r["lockstep"]["final_top50_match"] == {"bid": 1.0, "ask": 1.0}


def test_trades_stamped_one_ms_after_created():
    r = run([o("order_created", 2000, 9, 1, 99.90, 3.0), t(2001, 1, 100.00, 1.0, 1, 9), t(2001, 1, 99.99, 2.0, 2, 9),
             o("order_changed", 2001, 9, 1, 100.00, 2.0, "1.0"), o("order_deleted", 2001, 1, 0, 100.00, 0, "1.0"),
             o("order_deleted", 2001, 2, 0, 99.99, 0, "2.0"), o("order_deleted", 2001, 9, 1, 99.99, 0, "2.0")])
    assert rows(r) == [(TAKE, SELL, 99.99, 3.0)]  # no phantom ADD of the aggressive amount
    assert r["lockstep"]["final_top50_match"] == {"bid": 1.0, "ask": 1.0}


def test_ioc_remainder_cancel_after_execution_price_change():
    """order_changed reports the execution price; the delete of the unfilled remainder is a CANCEL."""
    r = run([o("order_created", 2000, 9, 1, 99.95, 5.0), t(2000, 1, 100.00, 1.0, 1, 9),
             o("order_changed", 2000, 9, 1, 100.00, 4.0, "1.0"), o("order_deleted", 2000, 1, 0, 100.00, 0, "1.0"),
             o("order_deleted", 2000, 9, 1, 99.95, 4.0, "0")])
    assert rows(r) == [(TAKE, SELL, 100.0, 1.0), (ADD, SELL, 99.95, 4.0), (CANCEL, SELL, 99.95, 4.0)]
    assert r["lockstep"]["final_top50_match"] == {"bid": 1.0, "ask": 1.0}


def test_maker_partial_fill_then_cancel_rest():
    r = run([t(2000, 0, 100.02, 0.5, 9, 3), o("order_changed", 2000, 3, 1, 100.02, 1.0, "0.5"),
             o("order_deleted", 3000, 3, 1, 100.02, 1.0)])
    assert rows(r) == [(TAKE, BUY, 100.02, 0.5), (CANCEL, SELL, 100.02, 1.0)]
    assert r["lockstep"]["final_top50_match"] == {"bid": 1.0, "ask": 1.0}


def test_price_amendment_without_trade():
    r = run([o("order_changed", 2000, 2, 0, 99.97, 2.0)])
    assert rows(r) == [(CANCEL, BUY, 99.99, 2.0), (ADD, BUY, 99.97, 2.0)]


def test_idbook_levels_have_no_float_dust():
    b = IdBook()
    for i in range(1000):
        b.set(i, BUY, 1.0, 0.1)
    for i in range(1000):
        b.pop(i)
    assert b.levels(BUY, 5) == [] and b.agg[BUY] == {}


def test_events_before_anchor_snapshot_are_dropped():
    r = run([o("order_created", 900, 9, 0, 99.98, 0.5)])  # us <= snapshot microtimestamp
    assert len(r["events"]) == 0


def test_state_is_pre_event_and_aligned():
    """L7: the state attached to each event is the book BEFORE that event (no lookahead)."""
    msgs = [t(2000, 0, 100.02, 1.5, 9, 3), o("order_created", 2000, 9, 0, 100.05, 1.5),
            o("order_deleted", 2000, 3, 1, 100.02, 0, "1.5"), o("order_deleted", 2000, 9, 0, 100.05, 0, "1.5"),
            o("order_created", 3000, 10, 0, 100.01, 0.7)]
    r = rebuild("x", SNAP, 1000, msgs, [], [], 0, tick=0.01, want_state=True)
    assert r["state"].shape == (len(r["events"]), 6)
    take, add = r["state"][0], r["state"][1]
    assert take[1] == 100.02  # best ask before the TAKE consumed it
    assert add[1] == 100.03 and add[0] == 100.00  # state before the ADD, after the sweep


def test_pre_event_level_quantity():
    """lvq = quantity of the event's own level BEFORE it (ADD/CANCEL) or of the opposite touch (TAKE)."""
    msgs = [o("order_created", 2000, 9, 0, 99.98, 0.5), o("order_deleted", 3000, 9, 0, 99.98, 0.5),
            o("order_deleted", 4000, 1, 0, 100.00, 1.0), t(5000, 0, 100.02, 1.5, 19, 3),
            o("order_deleted", 5000, 3, 1, 100.02, 0, "1.5")]
    r = rebuild("x", SNAP, 1000, msgs, [], [], 0, tick=0.01, want_state=True)
    assert [int(e["type"]) for e in r["events"]] == [ADD, CANCEL, CANCEL, TAKE]
    assert r["lvq"].tolist() == [0.0, 0.5, 1.0, 1.5]
