"""Paper trading against the LIVE book (no real orders are ever sent anywhere).

Market orders: walk the live id-book levels at submission time (VWAP), consuming displayed depth only.
Limit orders: rest at their price behind the queue displayed at placement (we are last in queue);
filled only by live aggressive trades that reach our price: trades at our price first eat the queue
ahead, then us; a trade through our price fills us fully. Conservative: cancels ahead never help us.
"""
from __future__ import annotations

import itertools
import time

from data.l3 import BUY, SELL, TAKE


class Paper:
    def __init__(self, fee_bps=0.0):
        self.fee = fee_bps / 1e4
        self.reset()

    def reset(self):
        self.pos = 0.0
        self.cash = 0.0
        self.fills = []
        self.orders = {}
        self._ids = itertools.count(1)

    def market(self, side, qty, book):
        lv = book.levels(SELL if side == BUY else BUY, 500)
        left, cost = qty, 0.0
        for p, q in lv:
            x = min(left, q)
            cost += x * p
            left -= x
            if left <= 1e-12:
                break
        done = qty - left
        if done <= 0:
            return dict(ok=False, reason="no displayed liquidity")
        vwap = cost / done
        self._fill(side, done, vwap, "market")
        return dict(ok=True, filled=done, vwap=vwap, unfilled=left)

    def limit(self, side, qty, price, book):
        best_opp = book.levels(SELL if side == BUY else BUY, 1)
        if best_opp and ((side == BUY and price >= best_opp[0][0]) or (side == SELL and price <= best_opp[0][0])):
            return self.market(side, qty, book)  # marketable: treated as market up to displayed depth
        same = dict(book.levels(side, 500))
        oid = next(self._ids)
        self.orders[oid] = dict(id=oid, side=side, qty=qty, price=price, ahead=same.get(price, 0.0), t=time.time())
        return dict(ok=True, resting=oid, queue_ahead=self.orders[oid]["ahead"])

    def cancel(self, oid):
        return dict(ok=self.orders.pop(int(oid), None) is not None)

    def on_trade(self, row, book):
        """row = (us, TAKE, aggressor side, worst price, qty, taker id)."""
        _, typ, aside, px, qty, _ = row
        if typ != TAKE:
            return
        for oid, o in list(self.orders.items()):
            if o["side"] == aside:
                continue  # an aggressive buy only hits resting sells and vice versa
            reach = (px >= o["price"]) if aside == BUY else (px <= o["price"])
            if not reach:
                continue
            through = (px > o["price"]) if aside == BUY else (px < o["price"])
            if through:
                fill = o["qty"]
            else:
                avail = max(qty - o["ahead"], 0.0)
                o["ahead"] = max(o["ahead"] - qty, 0.0)
                fill = min(o["qty"], avail)
            if fill > 0:
                self._fill(o["side"], fill, o["price"], f"limit#{oid}")
                o["qty"] -= fill
                if o["qty"] <= 1e-12:
                    del self.orders[oid]

    def _fill(self, side, qty, price, kind):
        sgn = 1 if side == BUY else -1
        self.pos += sgn * qty
        self.cash -= sgn * qty * price + qty * price * self.fee
        self.fills.append(dict(t=time.time(), side="buy" if side == BUY else "sell", qty=qty, price=price, kind=kind))

    def summary(self, mid):
        return dict(position=self.pos, cash=self.cash, equity=self.cash + self.pos * mid if mid else None,
                    open_orders=list(self.orders.values()), fills=self.fills[-20:])
