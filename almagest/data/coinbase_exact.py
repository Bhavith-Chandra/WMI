"""Replay-exact Coinbase tokenizer (v3). Fixes the v2 representation loss found by the replay test (log-depth error
1.2 after 1000 real events; mid-change correlation 0.31).

Principle: the CANONICAL book is the engine book itself. The tokenizer keeps an engine book (integer lots per coarse
bucket, absolute index = bucket - base) that is changed ONLY through the engine's own functions (W.apply_action), so
replaying the emitted tokens through the engine reproduces every feature bit-for-bit. Each real L2 change becomes the
token that RECONCILES the engine bucket with the real bucket (target = max(1, round(size / q0)) lots if size > 0):
  * diff > 0 inside the action window -> limit token (exact lots);
  * diff < 0 inside the window -> cancel token (exact lots); consumed liquidity needs no attribution: a market token
    already removed it from the engine book, the reconciliation only sees the residual;
  * a match group (same taker order) -> one market token with exact lots, applied by the engine (sweeps levels).
Changes the action space cannot express (deeper than D-1 buckets, improvement by > 1 bucket, crossed L2) are HIDDEN
updates: applied to the engine book, stored (position, side, index, new lots) so replay stays exact, never a model
target. Their share is reported (stats) and gated.
Every token also stores exact lots; the size bucket is floor(log2 lots) capped at NSB-1 as in v2.
"""
import math
import numpy as np
from almagest.sim import world as W, engine as E
from almagest.data.coinbase_tokenize import (CoarseBook, KAPPA_BP, Q0, D, NSB, read_lines, ts, dt_bucket,
                                             size_bucket, segments)

P = W.P
NB = 10
DR = 64                   # real-data action window (coarse buckets from own best): covers 84% of deep L2 changes
NF = E.NF3
KI = 8                    # explicit limit classes 'improve own best by k = 2..KI+1 buckets' (token lvl DR+1 .. DR+KI)
NLVL = DR + 1 + KI


def eng_lvl(typ, lvl):
    """token level -> engine level (vectorised). Improvement class DR-1+k -> engine lvl 1-k (price own best + k)."""
    lvl = np.asarray(lvl)
    return np.where((np.asarray(typ) == 0) & (lvl > DR), DR - lvl, lvl)
CK_EVERY = 2048           # full-book checkpoints (sparse) every CK_EVERY tokens -> exact deep seeding anywhere


def target_lots(v, q0):
    return 0 if v <= 1e-12 else max(1, int(round(v / q0)))


class ExactTokenizer:
    def __init__(self, product):
        self.product, self.q0 = product, Q0[product]
        self.real = None
        self.bid = np.zeros(P, np.int64)
        self.ask = np.zeros(P, np.int64)
        self.bb, self.ba, self.base = -1, P, 0
        self.cur, self.last_t = None, None
        self.toks, self.hid, self.cks = [], [], []
        self.need_ck = False
        self.stats = dict(l2_changes=0, limits=0, cancels=0, markets=0, hidden_deep=0, hidden_improve=0,
                          hidden_cross=0, hidden_lots=0, token_lots=0, resets=0, out_of_range=0)

    # ---- engine-book helpers (only engine semantics) ----
    def _idx(self, b):
        return b - self.base

    def _features(self):
        f = np.zeros(NF, np.float64); cb = np.zeros(DR, np.int64); ca = np.zeros(DR, np.int64)
        E.observe_row3(self.bid, self.ask, self.bb, self.ba, f, cb, ca)
        return f.astype(np.float32), np.stack([cb, ca]).astype(np.int8)

    def _valid(self):
        return 0 <= self.bb < self.ba < P and self.bid[self.bb] > 0 and self.ask[self.ba] > 0

    def _emit(self, typ, side, lvl, lots, t):
        if not self._valid():
            return False
        f, cm = self._features()
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        n = len(self.toks)
        if n % CK_EVERY == 0 or self.need_ck:
            self._checkpoint(n)
            self.need_ck = False
        self.toks.append(((typ, side, lvl, size_bucket(lots), dt_bucket(dt)), int(lots), f, cm,
                          0.5 * (self.bb + self.ba) + self.base, t))
        self.bb, self.ba, _ = W.apply_action(self.bid, self.ask, self.bb, self.ba, typ, side, int(eng_lvl(typ, lvl)),
                                             int(lots))
        self.stats['token_lots'] += int(lots)
        return True

    def _hidden(self, side, i, new):
        arr = self.bid if side == 0 else self.ask
        self.stats['hidden_lots'] += abs(int(new) - int(arr[i]))
        arr[i] = new
        self.hid.append((len(self.toks), side, i, int(new)))
        self._fix_best(side, i)

    def _fix_best(self, side, i):
        if side == 0:
            if self.bid[i] > 0 and i > self.bb:
                self.bb = i
            elif i == self.bb and self.bid[i] == 0:
                self.bb = W._best_bid(self.bid, self.bb)
        else:
            if self.ask[i] > 0 and i < self.ba:
                self.ba = i
            elif i == self.ba and self.ask[i] == 0:
                self.ba = W._best_ask(self.ask, self.ba)

    def _checkpoint(self, n):
        nb, na = np.nonzero(self.bid)[0], np.nonzero(self.ask)[0]
        self.cks.append((n, self.bb, self.ba, nb, self.bid[nb].copy(), na, self.ask[na].copy()))

    def _reset(self, m):
        mid = 0.5 * (float(m['bids'][0][0]) + float(m['asks'][0][0]))
        if self.real is None:
            self.real = CoarseBook(KAPPA_BP * mid, self.q0)       # kappa fixed per segment (first snapshot)
        else:
            self.real = CoarseBook(self.real.k, self.q0)
        for p, s in m['bids']:
            self.real.set_level(0, float(p), float(s))
        for p, s in m['asks']:
            self.real.set_level(1, float(p), float(s))
        self.base = int(self.real.best(0)) - P // 2
        self.bid[:] = 0; self.ask[:] = 0
        for side, arr in ((0, self.bid), (1, self.ask)):
            for b, v in self.real.coarse[side].items():
                i = self._idx(b)
                if 0 <= i < P:
                    arr[i] = target_lots(v, self.q0)
        self.bb = W._best_bid(self.bid, P - 1)
        self.ba = W._best_ask(self.ask, 0)
        self.cur, self.last_t = None, None
        self.stats['resets'] += 1
        self.hid.append((len(self.toks), -1, 0, 0))   # marker: full re-seed (replay reloads the checkpoint)
        self.need_ck = True                            # checkpoints are always taken at emission time

    # ---- feed ----
    def _flush_market(self):
        if self.cur is not None:
            lots = max(1, int(round(self.cur[2] / self.q0)))
            if self._emit(2, self.cur[1], 0, lots, self.cur[3]):
                self.stats['markets'] += 1
            self.cur = None

    def flush(self):
        self._flush_market()

    def _reconcile(self, side, b, t):
        i = self._idx(b)
        if not (0 <= i < P):
            self.stats['out_of_range'] += 1
            return
        arr = self.bid if side == 0 else self.ask
        T = target_lots(self.real.coarse[side].get(b, 0.0), self.q0)
        diff = T - int(arr[i])
        if diff == 0:
            return
        if not self._valid():
            self._hidden(side, i, T)
            return
        off = (self.bb - i) if side == 0 else (i - self.ba)
        if diff > 0:
            if off < -1:
                k = -off
                inside = (i < self.ba) if side == 0 else (i > self.bb)
                if k <= KI + 1 and inside:
                    if self._emit(0, side, DR - 1 + k, diff, t):
                        self.stats['limits'] += 1
                    return
                self.stats['hidden_improve'] += 1; self._hidden(side, i, T); return
            if off > DR - 1:
                self.stats['hidden_deep'] += 1; self._hidden(side, i, T); return
            if off == -1 and ((side == 0 and i >= self.ba) or (side == 1 and i <= self.bb)):
                self.stats['hidden_cross'] += 1; self._hidden(side, i, T); return
            if self._emit(0, side, off + 1, diff, t):
                self.stats['limits'] += 1
        else:
            if off < 0 or off > DR - 1:
                self.stats['hidden_deep'] += 1; self._hidden(side, i, T); return
            if self._emit(1, side, off, -diff, t):
                self.stats['cancels'] += 1

    def feed(self, m):
        t = m.get('type')
        if t == 'snapshot':
            self._flush_market()
            self._reset(m)
            return
        if self.real is None:
            return
        if t == 'match':
            tm = ts(m['time'])
            taker = 0 if m['side'] == 'sell' else 1          # Coinbase 'side' = maker side
            if self.cur is not None and self.cur[0] != m['taker_order_id']:
                self._flush_market()
            if self.cur is None:
                self.cur = [m['taker_order_id'], taker, 0.0, tm]
            self.cur[2] += float(m['size'])
        elif t == 'l2update':
            self._flush_market()
            tm = ts(m['time'])
            touched = {}
            for side_s, p, s in m['changes']:
                self.stats['l2_changes'] += 1
                side = 0 if side_s == 'buy' else 1
                b, _ = self.real.set_level(side, float(p), float(s))
                touched[(side, b)] = 1
            # reconcile AFTER the whole batch (no transient crosses), decreases first, then increases nearest-first;
            # the engine's best levels are reconciled too (market sweeps may precede their L2 update)
            for side in (0, 1):
                if self._valid():
                    i = self.bb if side == 0 else self.ba
                    touched[(side, i + self.base)] = 1
            todo = []
            rb = {0: self.real.best(0), 1: self.real.best(1)}
            for (side, b) in touched:
                i = self._idx(b)
                if not (0 <= i < P):
                    self.stats['out_of_range'] += 1
                    continue
                arr = self.bid if side == 0 else self.ask
                d = target_lots(self.real.coarse[side].get(b, 0.0), self.q0) - int(arr[i])
                if d != 0:
                    todo.append((0 if d < 0 else 1, abs(b - (rb[side] if rb[side] is not None else b)), side, b))
            for _, _, side, b in sorted(todo):
                self._reconcile(side, b, tm)

    def pack(self):
        toks = self.toks
        if not toks:
            z = np.zeros(0)
            return dict(ev=np.zeros((0, 5), np.int64), lots=np.zeros(0, np.int64), feat=np.zeros((0, NF), np.float32),
                        mids=z, cm=np.zeros((0, 2, DR), np.int8), times=z, hid=np.zeros((0, 4), np.int64),
                        ck=[], stats=self.stats, base=self.base)
        ev, lots, f, cm, mid, t = zip(*toks)
        return dict(ev=np.array(ev, np.int64), lots=np.array(lots, np.int64), feat=np.array(f, np.float32),
                    mids=np.array(mid), cm=np.array(cm, np.int8), times=np.array(t),
                    hid=np.array(self.hid, np.int64).reshape(-1, 4), ck=self.cks, stats=self.stats)


def tokenize_segment(paths, product, max_msgs=None):
    st = ExactTokenizer(product)
    k = 0
    for path in paths:
        for m in read_lines(path):
            st.feed(m)
            k += 1
            if max_msgs and k >= max_msgs:
                break
        if max_msgs and k >= max_msgs:
            break
    st.flush()
    return st.pack()


def ck_arrays(cks):
    """Flatten sparse checkpoints for npz storage."""
    pos = np.array([c[0] for c in cks], np.int64)
    best = np.array([[c[1], c[2]] for c in cks], np.int64).reshape(-1, 2)
    nb = np.array([len(c[3]) for c in cks]); na = np.array([len(c[5]) for c in cks])
    return dict(ck_pos=pos, ck_best=best, ck_nb=nb, ck_na=na,
                ck_bi=np.concatenate([c[3] for c in cks]) if cks else np.zeros(0, np.int64),
                ck_bv=np.concatenate([c[4] for c in cks]) if cks else np.zeros(0, np.int64),
                ck_ai=np.concatenate([c[5] for c in cks]) if cks else np.zeros(0, np.int64),
                ck_av=np.concatenate([c[6] for c in cks]) if cks else np.zeros(0, np.int64))


def _load_ck(z, j):
    ob = np.concatenate([[0], np.cumsum(z['ck_nb'])]); oa = np.concatenate([[0], np.cumsum(z['ck_na'])])
    bid = np.zeros(P, np.int64); ask = np.zeros(P, np.int64)
    bid[z['ck_bi'][ob[j]:ob[j + 1]]] = z['ck_bv'][ob[j]:ob[j + 1]]
    ask[z['ck_ai'][oa[j]:oa[j + 1]]] = z['ck_av'][oa[j]:oa[j + 1]]
    return bid, ask, int(z['ck_best'][j, 0]), int(z['ck_best'][j, 1])


def book_at(z, n):
    """Exact full engine book just BEFORE token n: nearest checkpoint <= n, then replay. Checkpoints are taken at
    emission time, so they already contain every hidden update recorded at their own position."""
    j = int(np.searchsorted(z['ck_pos'], n, side='right') - 1)
    bid, ask, bb, ba = _load_ck(z, j)
    return replay(z, int(z['ck_pos'][j]), n, bid, ask, bb, ba)


def replay(z, s, e, bid, ask, bb, ba, on_token=None):
    """Book is the exact state just before token s (hidden updates at position s included). Applies tokens s..e-1 and
    the hidden updates recorded before each later token; a reset marker reloads that position's checkpoint."""
    hid, ev, lots, pos = z['hid'], z['ev'], z['lots'], z['ck_pos']
    h = int(np.searchsorted(hid[:, 0], s, side='right')) if len(hid) else 0
    for n in range(s, e):
        if n > s:
            reset = False
            while h < len(hid) and hid[h, 0] == n:
                if hid[h, 1] < 0:
                    reset = True
                else:
                    side, i, v = int(hid[h, 1]), int(hid[h, 2]), int(hid[h, 3])
                    (bid if side == 0 else ask)[i] = v
                    bb, ba = _fix(bid, ask, bb, ba, side, i)
                h += 1
            if reset:
                j = int(np.searchsorted(pos, n, side='right') - 1)
                bid, ask, bb, ba = _load_ck(z, j)
        if on_token is not None:
            on_token(n, bid, ask, bb, ba)
        bb, ba, _ = W.apply_action(bid, ask, bb, ba, int(ev[n, 0]), int(ev[n, 1]), int(eng_lvl(ev[n, 0], ev[n, 2])),
                                   int(lots[n]))
    return bid, ask, bb, ba


def _fix(bid, ask, bb, ba, side, i):
    if side == 0:
        if bid[i] > 0 and i > bb:
            bb = i
        elif i == bb and bid[i] == 0:
            bb = W._best_bid(bid, bb)
    else:
        if ask[i] > 0 and i < ba:
            ba = i
        elif i == ba and ask[i] == 0:
            ba = W._best_ask(ask, ba)
    return bb, ba
