"""v2 incremental tokens/features (O(1) per event), identical to data/tokenizer_v2 batch encoding (tested)."""
from __future__ import annotations

import collections
import math

import numpy as np

from data.tokenizer_v2 import (CANCEL_EDGES, EPS, N_DEEP, N_INSIDE, TAKE, TAKE_EDGES, TAKE_SWEEP, WHOLE_CANCEL,
                               WHOLE_TAKE, CANCEL, Tokenizer2)


class IncFeatures2:
    LAGS = ((10, 2.0), (100, 5.0), (1000, 10.0))

    def __init__(self, tok: Tokenizer2):
        self.tok = tok
        self.lm = []
        self.dts = collections.deque(maxlen=32)
        self.win1 = collections.deque()
        self.win10 = collections.deque()
        self.prev_mid_after = None
        self.last_chg_us = None

    def seed(self, pre_state):
        self.lm = [math.log(0.5 * (pre_state[0] + pre_state[1]))]

    def tokens_for(self, typ, side, price, qty, us, prev_us, pre, lvq):
        bb, ba = np.float64(pre[0]), np.float64(pre[1])
        mid = 0.5 * (bb + ba)
        price = np.float64(price)
        tk = self.tok
        if typ == TAKE:
            px = max(((price - ba) if side == 1 else (bb - price)) / mid * 1e4, 0.0)
            pxt = N_INSIDE + 1 + N_DEEP + int(np.searchsorted(tk.edges_take, px))
        else:
            d = ((bb - price) if side == 1 else (price - ba)) / mid * 1e4
            if d < 0:
                pxt = int(np.searchsorted(tk.edges_inside, d))
            elif d == 0:
                pxt = N_INSIDE
            else:
                pxt = N_INSIDE + 1 + int(np.searchsorted(tk.edges_deep, np.log10(np.float64(d))))
        frac = qty / lvq if (lvq is not None and lvq > 0) else np.nan
        if typ == CANCEL:
            f = frac if np.isfinite(frac) else 1.0
            szt = WHOLE_CANCEL if f >= 1 - EPS else int(np.searchsorted(CANCEL_EDGES, f))
        elif typ == TAKE:
            f = frac if np.isfinite(frac) else 1.0
            szt = int(np.searchsorted(TAKE_EDGES, f)) if f < 1 - EPS else (WHOLE_TAKE if f <= 1 + EPS else WHOLE_TAKE + 1 + int(np.searchsorted(TAKE_SWEEP, f)))
        else:
            szt = int(np.searchsorted(tk.edges_add, np.log10(np.maximum(np.float64(qty) * mid, 1e-9))))
        dt = max(us - prev_us, 0) if prev_us is not None else 0
        dtt = 0 if dt <= 0 else 1 + int(np.searchsorted(tk.edges_dt, np.log10(np.maximum(np.float64(dt), 1e-9))))
        return np.array([typ, side, pxt, szt, dtt], np.int64), dt

    def push(self, us, dt, post):
        """Register an applied event (timestamp, dt, POST-event state) -> normalised feature row."""
        bb, ba, qb1, qa1, qb5, qa5 = (float(x) for x in post)
        mid = 0.5 * (bb + ba)
        lm = math.log(mid)
        if not self.lm:
            raise RuntimeError("seed() first")
        self.lm.append(lm)
        if len(self.lm) > 1102:
            self.lm = self.lm[-1102:]
        sp = (ba - bb) / mid * 1e4
        f = [math.log1p(max(sp, 0.0)), (qb1 - qa1) / max(qb1 + qa1, 1e-12), (qb5 - qa5) / max(qb5 + qa5, 1e-12),
             math.log10(max(qb1 * mid, 1e-9)), math.log10(max(qa1 * mid, 1e-9)), math.log10(max((qb5 + qa5) * mid, 1e-9))]
        n = len(self.lm)
        for k, s in self.LAGS:
            ref = self.lm[n - 1 - k] if n - 1 - k >= 0 else self.lm[0]
            f.append(math.tanh((lm - ref) * 1e4 / s))
        self.dts.append(float(dt))
        f.append(math.log1p(sum(self.dts) / 32))
        for w, span in ((self.win1, 1_000_000), (self.win10, 10_000_000)):
            w.append(us)
            while w and w[0] <= us - span:
                w.popleft()
        f.append(math.log1p(len(self.win1)))
        f.append(math.log1p(len(self.win10)))
        if self.prev_mid_after is None or mid != self.prev_mid_after:
            self.last_chg_us = us
        self.prev_mid_after = mid
        f.append(math.log1p((us - self.last_chg_us) / 1000.0))
        x = (np.array(f) - self.tok.feat_mean) / self.tok.feat_std
        return np.nan_to_num(x).astype(np.float32)
