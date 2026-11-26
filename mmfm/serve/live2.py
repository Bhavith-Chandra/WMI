"""v2 live feed: same verified L3Processor, v2 tokens/features (IncFeatures2) with pre-event level quantities
computed exactly as data/l3.rebuild does (train/serve equivalence tested in tests/test_live_equiv_v2.py)."""
from __future__ import annotations

import collections

import numpy as np

from data.l3 import BUY, SELL, TAKE
from model.infer2 import IncFeatures2
from serve.live import LiveFeed


class LiveFeed2(LiveFeed):
    def __init__(self, pair, tick, tok, ctx_len=64, hist=4000):
        super().__init__(pair, tick, tok, ctx_len=ctx_len, hist=hist)
        self.inc = IncFeatures2(tok)
        self.lvqs = collections.deque(maxlen=hist)
        self._pending = None

    def reset_stream_state(self):
        self.inc = IncFeatures2(self.tok)
        self._pending = None
        self._prev_us = None
        for q in (self.events, self.states, self.tokens, self.feats, self.mids, self.lvqs):
            q.clear()

    def _lv_before(self, m):
        book = self.proc.book
        out = {}
        if m["event"] == "trade":
            return out
        d = m["data"]
        try:
            sd = BUY if int(d["order_type"]) == 0 else SELL
            lv = book.agg[sd].get(float(d["price"]))
            out[(sd, float(d["price"]))] = lv[0] if lv else 0.0
            prev = book.orders.get(int(d["id"]))
            if prev is not None:
                lv2 = book.agg[prev[0]].get(prev[1])
                out[(prev[0], prev[1])] = lv2[0] if lv2 else 0.0
        except (KeyError, ValueError, TypeError):
            pass
        return out

    def _handle(self, m):
        pre = self.proc.book.state()
        lvb = self._lv_before(m)
        rows = self.proc.handle(m)
        if not rows or not np.isfinite(pre[0]):
            return
        self._finalize(pre)
        for j, r in enumerate(rows):
            us, typ, side, price, qty, _ = r
            lvq = (pre[3] if side == BUY else pre[2]) if typ == TAKE else lvb.get((side, price), np.nan)
            if self._prev_us is None:
                self.inc.seed(pre)
            row, dt = self.inc.tokens_for(typ, side, price, qty, us, self._prev_us, pre, lvq)
            if j < len(rows) - 1:
                self.feats.append(self.inc.push(us, dt, pre))
            else:
                self._pending = (us, dt)
            self.events.append((us, typ, side, price, qty))
            self.states.append(pre)
            self.lvqs.append(lvq)
            self.tokens.append(row)
            self.mids.append((us, 0.5 * (pre[0] + pre[1])))
            self._prev_us = us
            self.status["n_events"] += 1
            self.status["last_us"] = us
            if typ == TAKE:
                self.trades.append(r)
                for cb in self.listeners:
                    cb(r, self.proc.book)

    def _finalize(self, state_now):
        if self._pending is not None and np.isfinite(state_now[0]):
            us, dt = self._pending
            self.feats.append(self.inc.push(us, dt, state_now))
            self._pending = None

    def context_arrays(self):
        import copy
        toks = np.array(self.tokens)
        feats = list(self.feats)
        inc = copy.deepcopy(self.inc)
        now = self.proc.book.state()
        if self._pending is not None:
            us, dt = self._pending
            feats.append(inc.push(us, dt, now))
        n = min(len(toks), len(feats))
        return toks[-n:], np.array(feats[-n:], np.float32), inc
