"""Deterministic aggregate-depth matching engine, implemented to the paper's appendix spec.

Spec (paper, "Matching Engine"):
  * each side = aggregate volume per discrete price level; no order ids, no FIFO
  * decoded price -> absolute via open anchor, rounded to tick (0.01); volume rounded to unit
  * buy consumes asks from lowest upward while ask <= limit (partial fills, multi-level);
    sell symmetric; residual posts at limit price
  * cancel removes depth only at its own side/price; unmatched cancel qty is recorded, no other effect
  * best quotes / mid updated after every event
Extra diagnostics (not in the paper) are tracked so silent failure modes become measurable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TICK = 0.01


@dataclass
class Book:
    bids: dict = field(default_factory=dict)  # tick -> qty
    asks: dict = field(default_factory=dict)

    @classmethod
    def from_snapshot(cls, lob: np.ndarray) -> "Book":
        """lob: [2,10,2] with side order (ask, bid), features (price, volume)."""
        b = cls()
        for (p, v) in lob[0]:
            if p > 0 and v > 0:
                b.asks[int(round(p / TICK))] = b.asks.get(int(round(p / TICK)), 0) + int(round(v))
        for (p, v) in lob[1]:
            if p > 0 and v > 0:
                b.bids[int(round(p / TICK))] = b.bids.get(int(round(p / TICK)), 0) + int(round(v))
        return b

    def best_bid(self):
        return max(self.bids) if self.bids else None

    def best_ask(self):
        return min(self.asks) if self.asks else None

    def mid(self):
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return math.nan
        return 0.5 * (bb + ba) * TICK

    def copy(self):
        return Book(dict(self.bids), dict(self.asks))


def decode_to_orders(feats: np.ndarray, open_anchor: float):
    """feats [T,5] -> (price_tick int, qty int, dt float>=0, action{0,1}, side{0 ask/sell,1 bid/buy})."""
    px = np.rint(open_anchor * (1.0 + feats[:, 0].astype(np.float64)) / TICK).astype(np.int64)
    qty = np.rint(np.expm1(feats[:, 1].astype(np.float64))).astype(np.int64).clip(min=0)
    dt_raw = feats[:, 2].astype(np.float64)
    dt = np.clip(dt_raw, 0.0, None)
    act = np.rint(feats[:, 3]).astype(np.int64)
    side = np.rint(feats[:, 4]).astype(np.int64)
    return px, qty, dt, act, side, dt_raw


@dataclass
class ReplayStats:
    mid: np.ndarray
    t: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    n_events: int
    n_add: int
    n_cancel: int
    cancel_qty: int
    cancel_unmatched_qty: int       # qty requested but no resting depth at that side/price
    cancel_events_fully_unmatched: int
    cancel_unmatched_outside_known: int  # unmatched cancels at prices never seen in book (incl. beyond 10-level snapshot)
    marketable_adds: int
    traded_qty: int
    one_side_empty_events: int
    negative_dt_frac: float
    zero_qty_events: int


def replay(book0: Book, px, qty, dt, act, side, dt_raw=None) -> tuple[Book, ReplayStats]:
    b = book0.copy()
    seen = set(b.bids) | set(b.asks)
    T = len(px)
    mid = np.empty(T)
    bb = np.empty(T)
    ba = np.empty(T)
    t = np.cumsum(dt)
    n_add = n_can = can_q = can_un = can_full = can_out = mkt = traded = empty = zq = 0
    for i in range(T):
        p, q, a, s = int(px[i]), int(qty[i]), int(act[i]), int(side[i])
        if q <= 0:
            zq += 1
        if a == 1:  # add
            n_add += 1
            res = q
            if s == 1:  # buy
                if b.asks and min(b.asks) <= p:
                    mkt += 1
                while res > 0 and b.asks:
                    best = min(b.asks)
                    if best > p:
                        break
                    f = min(res, b.asks[best])
                    res -= f
                    traded += f
                    b.asks[best] -= f
                    if b.asks[best] <= 0:
                        del b.asks[best]
                if res > 0:
                    b.bids[p] = b.bids.get(p, 0) + res
            else:  # sell
                if b.bids and max(b.bids) >= p:
                    mkt += 1
                while res > 0 and b.bids:
                    best = max(b.bids)
                    if best < p:
                        break
                    f = min(res, b.bids[best])
                    res -= f
                    traded += f
                    b.bids[best] -= f
                    if b.bids[best] <= 0:
                        del b.bids[best]
                if res > 0:
                    b.asks[p] = b.asks.get(p, 0) + res
            seen.add(p)
        else:  # cancel
            n_can += 1
            can_q += q
            book = b.bids if s == 1 else b.asks
            have = book.get(p, 0)
            rem = min(have, q)
            un = q - rem
            can_un += un
            if have == 0 and q > 0:
                can_full += 1
                if p not in seen:
                    can_out += 1
            if rem:
                book[p] = have - rem
                if book[p] <= 0:
                    del book[p]
        bbi, bai = b.best_bid(), b.best_ask()
        if bbi is None or bai is None:
            empty += 1
        bb[i] = bbi * TICK if bbi is not None else np.nan
        ba[i] = bai * TICK if bai is not None else np.nan
        mid[i] = b.mid()
    ndt = float(np.mean(dt_raw < 0)) if dt_raw is not None and T else 0.0
    return b, ReplayStats(mid, t, bb, ba, T, n_add, n_can, can_q, can_un, can_full, can_out, mkt, traded, empty, ndt, zq)
