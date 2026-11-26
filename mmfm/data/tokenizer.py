"""Factorized discrete event tokenizer + causal state features. Every fitted quantity comes from the
split passed to `fit` (train only, EVAL.md L1) and records that split's identity.

Fields (predicted in this order): act, side, px, size, dt
  act  : ADD / CANCEL / TAKE
  side : 0 ask/sell, 1 bid/buy
  px   : ADD/CANCEL -> signed bp distance from the same-side touch (<0 inside the spread, 0 at touch, >0 deeper)
         TAKE       -> sweep depth: bp beyond the opposite touch reached by the worst fill (>= 0)
  size : log10 notional (qty * mid, quote currency)
  dt   : microseconds since the previous event of the pair (exact-0 bin + log-quantile bins)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

ADD, CANCEL, TAKE = 0, 1, 2
N_INSIDE, N_DEEP, N_TAKE, N_SIZE, N_DT = 8, 31, 8, 24, 23
FIELDS = ("act", "side", "px", "size", "dt")


def raw_fields(ev, state):
    """Continuous per-event quantities from events + PRE-event state (state row i = book before event i)."""
    bb, ba = state[:, 0], state[:, 1]
    mid = 0.5 * (bb + ba)
    typ, side, p = ev["type"].astype(np.int64), ev["side"].astype(np.int64), ev["price"]
    dist = np.where(side == 1, (bb - p) / mid, (p - ba) / mid) * 1e4          # ADD/CANCEL: vs same-side touch
    sweep = np.where(side == 1, (p - ba) / mid, (bb - p) / mid) * 1e4          # TAKE: beyond opposite touch
    px = np.where(typ == TAKE, np.maximum(sweep, 0.0), dist)
    notional = np.log10(np.maximum(ev["qty"] * mid, 1e-9))
    dt = np.diff(ev["us"], prepend=ev["us"][0]).astype(np.float64)
    return typ, side, px, notional, dt, mid


def state_features(ev, state):
    """Causal state vector AFTER event i (= state before event i+1); last row uses its own pre-state shifted.
    Uses only the book (pre-event states) and past events."""
    bb, ba, qb1, qa1, qb5, qa5 = (state[:, k] for k in range(6))
    mid = 0.5 * (bb + ba)
    spread_bp = (ba - bb) / mid * 1e4
    f = np.stack([
        np.log1p(np.maximum(spread_bp, 0)),
        (qb1 - qa1) / np.maximum(qb1 + qa1, 1e-12),
        (qb5 - qa5) / np.maximum(qb5 + qa5, 1e-12),
        np.log10(np.maximum((qb1 + qa1) * mid, 1e-9)),
        np.log10(np.maximum((qb5 + qa5) * mid, 1e-9)),
    ], 1)
    lm = np.log(mid)
    def lag(x, k):
        y = np.full_like(x, x[0])
        if k < len(x):
            y[k:] = x[:-k]
        return y

    rets = [np.tanh((lm - lag(lm, k)) * 1e4 / s) for k, s in ((10, 2), (100, 5), (1000, 10))]
    f = np.concatenate([f, np.stack(rets, 1)], 1)
    # book-derived features: row i of `after` = pre-state of event i+1 = book after event i (past only)
    after = np.concatenate([f[1:], f[-1:]], 0)
    # activity: mean inter-arrival of the last 32 events up to and including event i (NOT event i+1's dt,
    # which is a prediction target)
    dt = np.diff(ev["us"], prepend=ev["us"][0]).astype(np.float64)
    act_rate = np.log1p(np.convolve(dt, np.ones(32) / 32, "full")[: len(dt)])
    after = np.concatenate([after, act_rate[:, None]], 1)
    return after.astype(np.float32)


@dataclass
class Tokenizer:
    edges_inside: np.ndarray
    edges_deep: np.ndarray
    edges_take: np.ndarray
    edges_size: np.ndarray
    edges_dt: np.ndarray
    feat_mean: np.ndarray
    feat_std: np.ndarray
    fit_split: dict

    @property
    def vocab(self):
        return dict(act=3, side=2, px=N_INSIDE + 1 + N_DEEP + N_TAKE, size=N_SIZE, dt=N_DT + 1)

    @staticmethod
    def _q(x, n):
        x = x[np.isfinite(x)]
        e = np.unique(np.quantile(x, np.linspace(0, 1, n + 1)[1:-1]))
        return e

    @classmethod
    def fit(cls, segs, split_info: dict):
        """segs: list of (events, state) from the TRAIN split only."""
        T, S, P, Z, D, F = [], [], [], [], [], []
        for ev, st in segs:
            typ, side, px, sz, dt, _ = raw_fields(ev, st)
            T.append(typ); P.append(px); Z.append(sz); D.append(dt); F.append(state_features(ev, st))
        typ, px, sz, dt, feats = map(np.concatenate, (T, P, Z, D, F))
        ac = typ != TAKE
        inside = px[ac & (px < 0)]
        deep = px[ac & (px > 0)]
        take = px[~ac]
        tok = cls(edges_inside=cls._q(inside, N_INSIDE), edges_deep=cls._q(np.log10(deep), N_DEEP),
                  edges_take=cls._q(take, N_TAKE) if len(take) > N_TAKE else np.array([0.5, 1, 2, 4, 8, 16, 32.]),
                  edges_size=cls._q(sz, N_SIZE), edges_dt=cls._q(np.log10(dt[dt > 0]), N_DT),
                  feat_mean=np.nanmean(feats, 0), feat_std=np.nanstd(feats, 0) + 1e-6, fit_split=split_info)
        return tok

    def encode(self, ev, st):
        typ, side, px, sz, dt, mid = raw_fields(ev, st)
        n = len(typ)
        pxt = np.zeros(n, np.int64)
        ac = typ != TAKE
        ins = ac & (px < 0)
        at = ac & (px == 0)
        dp = ac & (px > 0)
        pxt[ins] = np.searchsorted(self.edges_inside, px[ins])
        pxt[at] = N_INSIDE
        pxt[dp] = N_INSIDE + 1 + np.searchsorted(self.edges_deep, np.log10(px[dp]))
        pxt[~ac] = N_INSIDE + 1 + N_DEEP + np.searchsorted(self.edges_take, px[~ac])
        szt = np.searchsorted(self.edges_size, sz)
        dtt = np.where(dt <= 0, 0, 1 + np.searchsorted(self.edges_dt, np.log10(np.maximum(dt, 1e-9))))
        tokens = np.stack([typ, side, pxt, szt, dtt], 1).astype(np.int64)
        feats = (state_features(ev, st) - self.feat_mean) / self.feat_std
        valid = np.isfinite(mid) & np.all(np.isfinite(feats), 1)
        return tokens, np.nan_to_num(feats).astype(np.float32), valid

    def to_json(self):
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        d["hash"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
        return d

    @classmethod
    def from_json(cls, d):
        d = dict(d)
        d.pop("hash", None)
        return cls(**{k: (np.array(v) if k != "fit_split" else v) for k, v in d.items()})
