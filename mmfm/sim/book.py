"""Aggregated price-level book for MMFM (integer ticks). Single matching implementation for
replay, training-feature extraction and simulation. State features at event t use only events <= t.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from data.l3 import ADD, BUY, CANCEL, SELL, TAKE


@dataclass
class LevelBook:
    tick: float
    bids: dict = field(default_factory=dict)  # tick -> qty
    asks: dict = field(default_factory=dict)

    @classmethod
    def from_idbook(cls, idbook, tick):
        b = cls(tick)
        for s, p, a in idbook.orders.values():
            k = int(round(p / tick))
            side = b.bids if s == BUY else b.asks
            side[k] = side.get(k, 0.0) + a
        return b

    def copy(self):
        return LevelBook(self.tick, dict(self.bids), dict(self.asks))

    def best(self, side):
        d = self.bids if side == BUY else self.asks
        if not d:
            return None
        return max(d) if side == BUY else min(d)

    def mid_ticks(self):
        bb, ba = self.best(BUY), self.best(SELL)
        return None if bb is None or ba is None else 0.5 * (bb + ba)

    def apply(self, typ, side, k, qty, eps=1e-12):
        """Returns (traded_qty, unmatched_qty, crossed_add)."""
        if qty <= 0:
            return 0.0, 0.0, False
        if typ == ADD:
            opp = self.asks if side == BUY else self.bids
            crossed = bool(opp) and ((min(opp) <= k) if side == BUY else (max(opp) >= k))
            own = self.bids if side == BUY else self.asks
            own[k] = own.get(k, 0.0) + qty
            return 0.0, 0.0, crossed
        if typ == CANCEL:
            own = self.bids if side == BUY else self.asks
            have = own.get(k, 0.0)
            rem = min(have, qty)
            if have - rem <= 1e-9 * max(have, 1e-12):  # relative: no float dust levels
                own.pop(k, None)
            else:
                own[k] = have - rem
            return 0.0, qty - rem, False
        # TAKE: side = aggressor side; consume opposite book up to limit k
        opp = self.asks if side == BUY else self.bids
        res, tr = qty, 0.0
        while res > 1e-9 * qty and opp:
            best = min(opp) if side == BUY else max(opp)
            if (side == BUY and best > k) or (side == SELL and best < k):
                break
            f = min(res, opp[best])
            res -= f
            tr += f
            if opp[best] - f <= 1e-9 * opp[best]:
                del opp[best]
            else:
                opp[best] -= f
        return tr, res, False

    def levels(self, side, n):
        d = self.bids if side == BUY else self.asks
        ks = sorted(d, reverse=(side == BUY))[:n]
        return [(k, d[k]) for k in ks]


def state_features(b: LevelBook, mid_hist_ticks, n_levels=5):
    """Scale-invariant state at the current event (past only). Returns np.float32[F]."""
    bb, ba = b.best(BUY), b.best(SELL)
    if bb is None or ba is None:
        return np.zeros(8, np.float32)
    mid = 0.5 * (bb + ba)
    spread_bp = (ba - bb) / mid * 1e4
    lb, la = b.levels(BUY, n_levels), b.levels(SELL, n_levels)
    vb, va = sum(q for _, q in lb), sum(q for _, q in la)
    imb = (vb - va) / max(vb + va, 1e-12)
    tb, ta = lb[0][1], la[0][1]
    imb1 = (tb - ta) / max(tb + ta, 1e-12)
    notional_touch = math.log10(max((tb + ta) * mid * b.tick, 1e-6))
    r = [0.0, 0.0, 0.0]
    for j, lag in enumerate((10, 100, 1000)):
        if len(mid_hist_ticks) > lag:
            r[j] = (mid / mid_hist_ticks[-lag - 1] - 1) * 1e4
    return np.array([math.log1p(spread_bp), imb, imb1, notional_touch / 6.0,
                     np.tanh(r[0] / 5), np.tanh(r[1] / 10), np.tanh(r[2] / 20), 1.0], np.float32)
