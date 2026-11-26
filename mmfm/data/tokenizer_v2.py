"""MMFM v2 tokenizer: state-relative sizes, clock-anchored features, multi-horizon forecast targets.

Event fields (predicted in order): act, side, px, size, dt
  px   : as v1 (bp distance from same-side touch for ADD/CANCEL; sweep depth beyond the opposite touch for TAKE)
  size : depends on act
         ADD    -> log10 notional quantile bins (N_ADD)
         CANCEL -> fraction of its level's pre-event quantity: 5 partial bins + exact WHOLE-LEVEL bin
         TAKE   -> fraction of the opposite touch quantity: 4 partial bins + WHOLE-TOUCH + 3 sweep bins
  dt   : exact-0 bin + log-quantile bins (as v1)
Features (state AFTER event i, i.e. before event i+1; past only):
  spread, imbalance top1/top5, log notional bid1 / ask1 / top5, mid return over the last 10/100/1000 events,
  mean inter-arrival of the last 32 events, wall-clock event counts in the last 1 s / 10 s,
  time since the last mid change.
Forecast targets at event i (labels only, never inputs): for h in HORIZONS, R_h = log-return (bp) from the mid
after event i to the mid at t_i + h, and N_h = number of events in (t_i, t_i + h].
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

ADD, CANCEL, TAKE = 0, 1, 2
N_INSIDE, N_DEEP, N_TAKE_PX, N_DT = 8, 31, 8, 23
N_ADD = 20
CANCEL_EDGES = np.array([0.1, 0.25, 0.5, 0.75])        # bins 0..4 partial, 5 = whole level
TAKE_EDGES = np.array([0.1, 0.25, 0.5])                # bins 0..3 partial (<1), 4 = whole touch, 5..7 sweep
TAKE_SWEEP = np.array([1.5, 3.0])
WHOLE_CANCEL, WHOLE_TAKE = 5, 4
SIZE_V = 24
HORIZONS = (1.0, 5.0, 10.0)
N_RET, N_CNT = 41, 24
FIELDS = ("act", "side", "px", "size", "dt")
EPS = 1e-9


def _lag(x, k):
    y = np.full_like(x, x[0])
    if k < len(x):
        y[k:] = x[:-k]
    return y


def raw_fields(ev, state, lvq):
    bb, ba = state[:, 0], state[:, 1]
    mid = 0.5 * (bb + ba)
    typ, side, p, q = ev["type"].astype(np.int64), ev["side"].astype(np.int64), ev["price"], ev["qty"]
    dist = np.where(side == 1, (bb - p) / mid, (p - ba) / mid) * 1e4
    sweep = np.where(side == 1, (p - ba) / mid, (bb - p) / mid) * 1e4
    px = np.where(typ == TAKE, np.maximum(sweep, 0.0), dist)
    notional = np.log10(np.maximum(q * mid, 1e-9))
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(lvq > 0, q / lvq, np.nan)
    dt = np.maximum(np.diff(ev["us"], prepend=ev["us"][0]), 0).astype(np.float64)
    return typ, side, px, notional, frac, dt, mid


def state_features(ev, state):
    """Row i = state AFTER event i (= pre-state of event i+1; book-derived) + activity up to event i."""
    bb, ba, qb1, qa1, qb5, qa5 = (state[:, k] for k in range(6))
    mid = 0.5 * (bb + ba)
    sp = (ba - bb) / mid * 1e4
    lm = np.log(mid)
    book = np.stack([np.log1p(np.maximum(sp, 0)), (qb1 - qa1) / np.maximum(qb1 + qa1, 1e-12),
                     (qb5 - qa5) / np.maximum(qb5 + qa5, 1e-12), np.log10(np.maximum(qb1 * mid, 1e-9)),
                     np.log10(np.maximum(qa1 * mid, 1e-9)), np.log10(np.maximum((qb5 + qa5) * mid, 1e-9))]
                    + [np.tanh((lm - _lag(lm, k)) * 1e4 / s) for k, s in ((10, 2), (100, 5), (1000, 10))], 1)
    after = np.concatenate([book[1:], book[-1:]], 0)
    # time since last mid change, measured at the time of event i, using the mid AFTER event i
    us = ev["us"].astype(np.int64)
    mid_after = np.concatenate([mid[1:], mid[-1:]])
    chg = np.r_[True, mid_after[1:] != mid_after[:-1]]
    last_chg_us = np.maximum.accumulate(np.where(chg, us, us[0]))
    since = np.log1p((us - last_chg_us) / 1000.0)
    dt = np.maximum(np.diff(us, prepend=us[0]), 0).astype(np.float64)
    act = np.log1p(np.convolve(dt, np.ones(32) / 32, "full")[: len(dt)])
    c1 = np.log1p(np.arange(len(us)) - np.searchsorted(us, us - 1_000_000, side="right") + 1)
    c10 = np.log1p(np.arange(len(us)) - np.searchsorted(us, us - 10_000_000, side="right") + 1)
    return np.concatenate([after, np.stack([act, c1, c10, since], 1)], 1).astype(np.float32)


def forecast_targets(ev, state):
    """R_h (bp) and N_h for each event i; NaN / -1 where the horizon runs past the chunk (masked in the loss)."""
    us = ev["us"].astype(np.int64)
    mid = 0.5 * (state[:, 0] + state[:, 1])
    mid_after = np.concatenate([mid[1:], [np.nan]])
    R, N = [], []
    for h in HORIZONS:
        j = np.searchsorted(us, us + int(h * 1e6), side="right")  # first event strictly after t_i + h
        ok = j < len(us)
        r = np.full(len(us), np.nan)
        r[ok] = np.log(mid[j[ok]] / mid_after[ok]) * 1e4  # mid at t+h = pre-state of event j
        n = np.where(ok, j - np.arange(len(us)) - 1, -1)
        R.append(r)
        N.append(n)
    return np.stack(R, 1), np.stack(N, 1)


@dataclass
class Tokenizer2:
    edges_inside: np.ndarray
    edges_deep: np.ndarray
    edges_take: np.ndarray
    edges_add: np.ndarray
    edges_dt: np.ndarray
    edges_ret: list
    edges_cnt: list
    feat_mean: np.ndarray
    feat_std: np.ndarray
    fit_split: dict
    bin_q: dict = None

    @property
    def vocab(self):
        return dict(act=3, side=2, px=N_INSIDE + 1 + N_DEEP + N_TAKE_PX, size=SIZE_V, dt=N_DT + 1)

    @staticmethod
    def _q(x, n, default=(0.0,)):
        x = x[np.isfinite(x)]
        if len(x) < 2:
            return np.array(default, float)
        return np.unique(np.quantile(x, np.linspace(0, 1, n + 1)[1:-1]))

    @classmethod
    def fit(cls, segs, split_info):
        """segs: list of (events, state, lvq) from the TRAIN split only."""
        T, P, Z, D, F, RR, NN = [], [], [], [], [], [], []
        for ev, st, lv in segs:
            typ, side, px, nz, fr, dt, _ = raw_fields(ev, st, lv)
            T.append(typ); P.append(px); Z.append(nz); D.append(dt); F.append(state_features(ev, st))
            r, n = forecast_targets(ev, st)
            RR.append(r); NN.append(n)
        typ, px, nz, dt, feats, R, N = map(np.concatenate, (T, P, Z, D, F, RR, NN))
        ac = typ != TAKE
        edges_ret, edges_cnt = [], []
        for k in range(len(HORIZONS)):
            r = R[:, k][np.isfinite(R[:, k])]
            nzr = r[r != 0]
            # symmetric-ish quantile bins on non-zero moves + an exact-zero bin in the middle
            neg, pos = nzr[nzr < 0], nzr[nzr > 0]
            e_neg = np.unique(np.quantile(neg, np.linspace(0, 1, N_RET // 2 + 1)[1:-1])) if len(neg) else np.array([-1.0])
            e_pos = np.unique(np.quantile(pos, np.linspace(0, 1, N_RET // 2 + 1)[1:-1])) if len(pos) else np.array([1.0])
            edges_ret.append(np.concatenate([e_neg, [-1e-12, 1e-12], e_pos]))
            c = N[:, k][N[:, k] >= 0]
            edges_cnt.append(cls._q(np.log1p(c.astype(float)), N_CNT, (1.0,)))
        tok = cls(edges_inside=cls._q(px[ac & (px < 0)], N_INSIDE, (-1.0,)), edges_deep=cls._q(np.log10(px[ac & (px > 0)]), N_DEEP),
                  edges_take=cls._q(px[~ac], N_TAKE_PX) if (~ac).sum() > N_TAKE_PX else np.array([.5, 1, 2, 4, 8, 16, 32.]),
                  edges_add=cls._q(nz[typ == ADD], N_ADD), edges_dt=cls._q(np.log10(dt[dt > 0]), N_DT),
                  edges_ret=edges_ret, edges_cnt=edges_cnt, feat_mean=np.nanmean(feats, 0),
                  feat_std=np.nanstd(feats, 0) + 1e-6, fit_split=split_info)
        # per-bin empirical quantiles (train) for realistic within-bin decoding
        qs = np.linspace(0.0, 1.0, 21)
        toks = np.concatenate([tok.encode(ev, st, lv)[0] for ev, st, lv in segs])
        FR = np.concatenate([raw_fields(ev, st, lv)[4] for ev, st, lv in segs])
        raw = dict(px=px, dt=np.log10(np.maximum(dt, 1e-9)))
        bq = {}
        for k, f in ((2, "px"), (4, "dt")):
            arr = np.full((tok.vocab[f], len(qs)), np.nan)
            for b in range(tok.vocab[f]):
                x = raw[f][(toks[:, k] == b)]
                x = x[np.isfinite(x)]
                if len(x):
                    arr[b] = np.quantile(x, qs)
            bq[f] = arr
        for a, f, val in ((ADD, "size_add", nz), (CANCEL, "size_cancel", FR), (TAKE, "size_take", FR)):
            arr = np.full((SIZE_V, len(qs)), np.nan)
            for b in range(SIZE_V):
                x = val[(typ == a) & (toks[:, 3] == b)]
                x = x[np.isfinite(x)]
                if len(x):
                    arr[b] = np.quantile(x, qs)
            bq[f] = arr
        tok.bin_q = bq
        return tok

    def size_tokens(self, typ, nz, frac):
        out = np.searchsorted(self.edges_add, nz)
        c = typ == CANCEL
        fc = np.where(np.isfinite(frac), frac, 1.0)  # unknown level -> treat as whole (rare; counted in tests)
        out = np.where(c, np.where(fc >= 1 - EPS, WHOLE_CANCEL, np.searchsorted(CANCEL_EDGES, fc)), out)
        t = typ == TAKE
        ft = np.where(np.isfinite(frac), frac, 1.0)
        take_tok = np.where(ft < 1 - EPS, np.searchsorted(TAKE_EDGES, ft),
                            np.where(ft <= 1 + EPS, WHOLE_TAKE, WHOLE_TAKE + 1 + np.searchsorted(TAKE_SWEEP, ft)))
        return np.where(t, take_tok, out)

    def encode(self, ev, st, lvq):
        typ, side, px, nz, frac, dt, mid = raw_fields(ev, st, lvq)
        n = len(typ)
        pxt = np.zeros(n, np.int64)
        ac = typ != TAKE
        ins, at, dp = ac & (px < 0), ac & (px == 0), ac & (px > 0)
        pxt[ins] = np.searchsorted(self.edges_inside, px[ins])
        pxt[at] = N_INSIDE
        pxt[dp] = N_INSIDE + 1 + np.searchsorted(self.edges_deep, np.log10(px[dp]))
        pxt[~ac] = N_INSIDE + 1 + N_DEEP + np.searchsorted(self.edges_take, px[~ac])
        szt = self.size_tokens(typ, nz, frac)
        dtt = np.where(dt <= 0, 0, 1 + np.searchsorted(self.edges_dt, np.log10(np.maximum(dt, 1e-9))))
        tokens = np.stack([typ, side, pxt, szt, dtt], 1).astype(np.int64)
        feats = (state_features(ev, st) - self.feat_mean) / self.feat_std
        valid = np.isfinite(mid) & np.all(np.isfinite(feats), 1)
        return tokens, np.nan_to_num(feats).astype(np.float32), valid

    def encode_targets(self, ev, st):
        R, N = forecast_targets(ev, st)
        rt = np.full(R.shape, -1, np.int64)
        ct = np.full(N.shape, -1, np.int64)
        for k in range(len(HORIZONS)):
            ok = np.isfinite(R[:, k])
            rt[ok, k] = np.searchsorted(self.edges_ret[k], R[ok, k])
            okn = N[:, k] >= 0
            ct[okn, k] = np.searchsorted(self.edges_cnt[k], np.log1p(N[okn, k]))
        return rt, ct

    def n_ret_bins(self, k):
        return len(self.edges_ret[k]) + 1

    def n_cnt_bins(self, k):
        return len(self.edges_cnt[k]) + 1

    def ret_bin_values(self, k):
        """Representative bp value per return bin (midpoints; exact 0 for the zero bin)."""
        e = self.edges_ret[k]
        lo = np.r_[e[0] - (e[1] - e[0]) if len(e) > 1 else e[0] - 1, e]
        hi = np.r_[e, e[-1] + (e[-1] - e[-2]) if len(e) > 1 else e[-1] + 1]
        v = 0.5 * (lo + hi)
        v[np.searchsorted(e, 0.0)] = 0.0
        return v

    def cnt_bin_values(self, k):
        e = self.edges_cnt[k]
        lo = np.r_[0.0, e]
        hi = np.r_[e, e[-1] + 0.5]
        return np.expm1(0.5 * (lo + hi))

    def sample_q(self, field, b, rng):
        q = self.bin_q[field][b]
        q = q[np.isfinite(q)]
        if not len(q):
            return np.nan
        u = rng.random() * (len(q) - 1)
        i = int(u)
        return float(q[i] + (u - i) * (q[min(i + 1, len(q) - 1)] - q[i]))

    def to_json(self):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif k in ("edges_ret", "edges_cnt"):
                d[k] = [np.asarray(x).tolist() for x in v]
            elif k == "bin_q" and v is not None:
                d[k] = {kk: np.where(np.isfinite(vv), vv, None).tolist() for kk, vv in v.items()}
            else:
                d[k] = v
        d["version"] = 2
        d["hash"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
        return d

    @classmethod
    def from_json(cls, d):
        d = dict(d)
        d.pop("hash", None)
        d.pop("version", None)
        bq = d.pop("bin_q", None)
        t = cls(**{k: (v if k == "fit_split" else ([np.array(x) for x in v] if k in ("edges_ret", "edges_cnt") else np.array(v)))
                   for k, v in d.items()})
        if bq is not None:
            t.bin_q = {k: np.array([[np.nan if x is None else x for x in row] for row in v], float) for k, v in bq.items()}
        return t
