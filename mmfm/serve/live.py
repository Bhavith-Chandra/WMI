"""Live Bitstamp L3 feed -> verified id-book (L3Processor, same code as the training data) -> model context.

Order messages are handled DELAY_MS after receipt so trades for the same matching step (other channel,
sometimes 1 ms later) are registered first - exactly the batch semantics. Snapshot anchoring follows
the same rule as training: the REST snapshot must post-date the subscription.
"""
from __future__ import annotations

import asyncio
import collections
import json
import time
import urllib.request

import numpy as np
import websockets

from data.l3 import ADD, BUY, CANCEL, SELL, TAKE, EV_DTYPE, L3Processor
from model.infer import IncFeatures
from sim.book import LevelBook

WS = "wss://ws.bitstamp.net"
REST = "https://www.bitstamp.net/api/v2/order_book/{pair}/?group=2"
DELAY_MS = 300


class LiveFeed:
    def __init__(self, pair, tick, tok, ctx_len=256, hist=4000):
        self.pair, self.tick, self.tok, self.ctx_len, self.hist = pair, tick, tok, ctx_len, hist
        self.proc = None
        self.inc = IncFeatures(tok)
        self.events = collections.deque(maxlen=hist)      # (us, type, side, price, qty)
        self.states = collections.deque(maxlen=hist)      # pre-event states
        self.tokens = collections.deque(maxlen=ctx_len)
        self.feats = collections.deque(maxlen=ctx_len)
        self.mids = collections.deque(maxlen=20000)       # (us, mid) after each event
        self.trades = collections.deque(maxlen=5000)      # TAKE rows (for the paper engine)
        self.listeners = []                               # callbacks(row) for TAKE rows
        self.status = dict(state="starting", n_events=0, reconnects=0, last_us=None, lag_ms=None)
        self._queue = collections.deque()
        self._prev_us = None
        self._pending_dt = None

    # ----------------------------------------------------------------------------------- public
    def ready(self):
        return self.proc is not None and len(self.tokens) >= self.ctx_len

    def level_book(self) -> LevelBook:
        return LevelBook.from_idbook(self.proc.book, self.tick)

    def ladder(self, n=12):
        b = self.proc.book
        return dict(bids=[(p, q) for p, q in b.levels(BUY, n)], asks=[(p, q) for p, q in b.levels(SELL, n)])

    def history_arrays(self):
        ev = np.array(list(self.events), dtype=[(k, t) for k, t in EV_DTYPE if k != "oid"])
        return ev, np.array(list(self.states), dtype=np.float64)

    def reset_stream_state(self):
        self.inc = IncFeatures(self.tok)
        self._prev_us = None
        self._pending_dt = None
        for q in (self.events, self.states, self.tokens, self.feats, self.mids):
            q.clear()

    # ----------------------------------------------------------------------------------- internals
    def _snapshot(self, not_before_us):
        for _ in range(15):
            with urllib.request.urlopen(urllib.request.Request(REST.format(pair=self.pair), headers={"User-Agent": "mmfm-live"}), timeout=20) as r:
                snap = json.loads(r.read())
            if int(snap["microtimestamp"]) >= not_before_us:
                return snap
            time.sleep(1.0)
        raise RuntimeError("REST snapshot stayed stale")

    def _handle(self, m):
        """Batch semantics (data/l3.rebuild + tokenizer.state_features):
        * every row of one message gets the book BEFORE the message as its pre-state
        * the feature row after event i = book just before event i+1 (includes silent updates in between,
          e.g. maker-fill messages that emit no event) -> the last row's feature is finalised lazily."""
        pre = self.proc.book.state()
        rows = self.proc.handle(m)
        if not rows or not np.isfinite(pre[0]):
            return
        self._finalize(pre)
        for j, r in enumerate(rows):
            us, typ, side, price, qty, _ = r
            if self._prev_us is None:
                self.inc.seed(pre)
            row, dt = self.inc.tokens_for(typ, side, price, qty, us, self._prev_us, pre)
            if j < len(rows) - 1:
                self.feats.append(self.inc.push(dt, pre))  # next row (same message) has the same pre-state
            else:
                self._pending_dt = dt
            self.events.append((us, typ, side, price, qty))
            self.states.append(pre)
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
        if self._pending_dt is not None and np.isfinite(state_now[0]):
            self.feats.append(self.inc.push(self._pending_dt, state_now))
            self._pending_dt = None

    def context_arrays(self):
        """tokens/feats for a forecast NOW: the pending last row gets its feature from the current book
        (provisional, not committed), exactly what the batch pipeline would compute if the next event came now."""
        import copy
        toks = np.array(self.tokens)
        feats = list(self.feats)
        inc = copy.deepcopy(self.inc)
        now = self.proc.book.state()
        if self._pending_dt is not None:
            feats.append(inc.push(self._pending_dt, now))
        n = min(len(toks), len(feats))
        return toks[-n:], np.array(feats[-n:], np.float32), inc

    async def run(self):
        while True:
            try:
                async with websockets.connect(WS, max_size=2**24, ping_interval=20) as ws:
                    for ch in ("live_orders", "live_trades"):
                        await ws.send(json.dumps({"event": "bts:subscribe", "data": {"channel": f"{ch}_{self.pair}"}}))
                    sub_us = time.time_ns() // 1000 + 500_000
                    self.status["state"] = "snapshot"
                    snap_task = asyncio.create_task(asyncio.to_thread(self._snapshot, sub_us))
                    buf = []
                    while not snap_task.done():
                        try:
                            msg = await asyncio.wait_for(ws.recv(), 0.2)
                            buf.append((time.time_ns(), json.loads(msg)))
                        except asyncio.TimeoutError:
                            pass
                    snap = snap_task.result()
                    self.proc = L3Processor(snap, int(snap["microtimestamp"]))
                    self.reset_stream_state()
                    self._queue = collections.deque(buf)
                    self.status["state"] = "live"
                    while True:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), DELAY_MS / 1000)
                            m = json.loads(msg)
                            if m.get("event") == "bts:request_reconnect":
                                break
                            self._queue.append((time.time_ns(), m))
                        except asyncio.TimeoutError:
                            pass
                        now = time.time_ns()
                        # register trades immediately; handle everything older than the delay, in order
                        for t, m in self._queue:
                            if m.get("event") == "trade" and not m.get("_reg"):
                                self.proc.add_trade(m["data"])
                                m["_reg"] = True
                        while self._queue and now - self._queue[0][0] > DELAY_MS * 1_000_000:
                            _, m = self._queue.popleft()
                            if m.get("event") in ("order_created", "order_changed", "order_deleted", "trade"):
                                self._handle(m)
                        if self.status["last_us"]:
                            self.status["lag_ms"] = (time.time_ns() // 1000 - self.status["last_us"]) / 1000
                            self.proc.prune(self.status["last_us"] - 10_000_000)
            except Exception as e:  # reconnect with a fresh snapshot; continuity is never fabricated
                self.status.update(state=f"reconnecting ({type(e).__name__})")
            self.status["reconnects"] += 1
            self.proc = None
            await asyncio.sleep(2)
