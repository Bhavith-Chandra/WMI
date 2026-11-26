"""MMFM generative simulator (real-time capable): KV-cached model + incremental features + single engine.

State-aware decoding (no phantom liquidity by construction):
  ADD    price = same-side touch -/+ sampled bp distance, rounded to tick; post-only clamp (never crosses)
  CANCEL resolves to an EXISTING level on that side nearest to the sampled distance; qty <= resting qty
  TAKE   sweeps the opposite side up to the sampled depth
The event the engine actually applied is re-tokenised (with its pre-state) and fed back, so the model
conditions on what happened, exactly as in training. All paths step in lock-step (batched KV cache).
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from data.l3 import ADD, BUY, CANCEL, SELL, TAKE
from data.tokenizer import N_INSIDE, Tokenizer
from model.infer import FastMMFM, IncFeatures
from sim.book import LevelBook


def book_state(b: LevelBook):
    lb, la = b.levels(BUY, 5), b.levels(SELL, 5)
    if not lb or not la:
        return None
    t = b.tick
    return (lb[0][0] * t, la[0][0] * t, lb[0][1], la[0][1], sum(q for _, q in lb), sum(q for _, q in la))


class Context:
    """Everything a rollout needs at time `now`: last T tokens/feats, incremental-feature state, book."""

    def __init__(self, tokens, feats, inc: IncFeatures, book: LevelBook, last_us: int):
        self.tokens, self.feats, self.inc, self.book, self.last_us = tokens, feats, inc, book, last_us


class Rollout:
    def __init__(self, model, tok: Tokenizer, ctx: Context, n_paths=32, seed=0, temperature=1.0):
        self.tok, self.temp = tok, temperature
        self.rng = np.random.default_rng(seed)
        self.g = torch.Generator().manual_seed(seed)
        self.fm = FastMMFM(model)
        T = model.cfg["ctx"]
        tk = torch.from_numpy(ctx.tokens[-T:])[None].repeat(n_paths, 1, 1)
        ft = torch.from_numpy(ctx.feats[-T:])[None].repeat(n_paths, 1, 1)
        self.h = self.fm.prefill(tk, ft)
        self.paths = [dict(book=ctx.book.copy(), inc=copy.deepcopy(ctx.inc), us=ctx.last_us, prev_us=ctx.last_us,
                           t=[ctx.last_us], mid=[0.5 * (book_state(ctx.book)[0] + book_state(ctx.book)[1])],
                           alive=True, n_ev=0, injected=[]) for _ in range(n_paths)]
        self.t0 = ctx.last_us
        self.gen_seconds = 0.0
        self.n_steps = 0

    # ------------------------------------------------------------------ token -> concrete order
    def _concrete(self, p, row):
        tk = self.tok
        act, side, pxb, szb, dtb = (int(x) for x in row)
        b = p["book"]
        st = book_state(b)
        if st is None:
            return None
        bb, ba = st[0], st[1]
        mid = 0.5 * (bb + ba)
        dt_us = 0 if dtb == 0 else int(round(10 ** tk.sample_value("dt", dtb, self.rng)))
        qty = 10 ** tk.sample_value("size", szb, self.rng) / mid
        tick = b.tick
        if act == TAKE:
            sw = tk.sample_value("px", pxb, self.rng)
            sw = max(sw, 0.0) if np.isfinite(sw) else 0.0
            if side == BUY:
                k = int(math.ceil(ba * (1 + sw / 1e4) / tick - 1e-9))
            else:
                k = int(math.floor(bb * (1 - sw / 1e4) / tick + 1e-9))
            return act, side, k, qty, dt_us
        d = 0.0 if pxb == N_INSIDE else tk.sample_value("px", pxb, self.rng)
        d = d if np.isfinite(d) else 0.0
        ref = bb if side == BUY else ba
        k = int(round((ref * (1 - d / 1e4) if side == BUY else ref * (1 + d / 1e4)) / tick))
        if act == ADD:
            k = min(k, int(round(ba / tick)) - 1) if side == BUY else max(k, int(round(bb / tick)) + 1)
            return act, side, k, qty, dt_us
        lv = b.bids if side == BUY else b.asks
        if not lv:
            return None
        keys = np.fromiter(lv.keys(), dtype=np.int64)
        kk = int(keys[np.argmin(np.abs(keys - k))])
        return act, side, kk, min(qty, lv[kk]), dt_us

    def _apply(self, p, o):
        """Apply concrete order; return (token row, feature row) of the event as applied."""
        act, side, k, qty, dt_us = o
        b = p["book"]
        pre = book_state(b)
        us = p["us"] + dt_us
        price = k * b.tick
        if act == TAKE:  # the TAKE's recorded price is the worst fill, as in the data
            opp = b.asks if side == BUY else b.bids
            worst, res = None, qty
            for lvk in (sorted(opp) if side == BUY else sorted(opp, reverse=True)):
                if (side == BUY and lvk > k) or (side == SELL and lvk < k) or res <= 0:
                    break
                worst = lvk
                res -= opp[lvk]
            price = (worst if worst is not None else k) * b.tick
        row, dt = p["inc"].tokens_for(act, side, price, qty, us, p["prev_us"], pre)
        b.apply(act, side, k, qty)
        post = book_state(b)
        if post is None:
            p["alive"] = False
            return row, np.zeros_like(self.fm.hist_feat[0, -1].numpy())
        f = p["inc"].push(dt, post)
        p["prev_us"] = p["us"] = us
        p["t"].append(us)
        p["mid"].append(0.5 * (post[0] + post[1]))
        p["n_ev"] += 1
        return row, f

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def step(self):
        t0 = time.perf_counter()
        rows = self.fm.m.heads.sample(self.h, generator=self.g, temperature=self.temp).numpy()
        toks, feats = [], []
        for p, r in zip(self.paths, rows):
            o = self._concrete(p, r) if p["alive"] else None
            if o is None:
                p["alive"] = False
                toks.append(self.fm.hist_tok[0, -1].numpy())
                feats.append(self.fm.hist_feat[0, -1].numpy())
                continue
            t, f = self._apply(p, o)
            toks.append(t)
            feats.append(f)
        self.h = self.fm.step(torch.from_numpy(np.stack(toks)), torch.from_numpy(np.stack(feats)))
        self.gen_seconds += time.perf_counter() - t0
        self.n_steps += 1

    def inject(self, act, side, k, qty):
        """What-if: the same user order enters every path now (immediate, dt=0) and the model sees it."""
        toks, feats = [], []
        for p in self.paths:
            t, f = self._apply(p, (act, side, k, qty, 0))
            p["injected"].append((act, side, k, qty))
            toks.append(t)
            feats.append(f)
        self.h = self.fm.step(torch.from_numpy(np.stack(toks)), torch.from_numpy(np.stack(feats)))

    def run(self, horizon_s, max_steps=20000, wall_budget_s=None):
        t_start = time.perf_counter()
        for _ in range(max_steps):
            if all((not p["alive"]) or p["us"] - self.t0 >= horizon_s * 1e6 for p in self.paths):
                break
            if wall_budget_s is not None and time.perf_counter() - t_start > wall_budget_s:
                break
            self.step()
        return self

    def mid_grid(self, grid_s):
        """Mid of every path on a time grid (seconds after t0), previous-tick; NaN beyond a path's reach."""
        out = np.full((len(self.paths), len(grid_s)), np.nan)
        for i, p in enumerate(self.paths):
            t = (np.array(p["t"]) - self.t0) / 1e6
            m = np.array(p["mid"])
            idx = np.searchsorted(t, grid_s, side="right") - 1
            ok = (idx >= 0) & (grid_s <= t[-1])
            out[i, ok] = m[idx[ok]]
        return out


def context_from_history(ev, st, tok: Tokenizer, book_now: LevelBook, T):
    """Build a Context from a historical/live event history (events + PRE-states) and the current book.
    `book_now` must be the book AFTER the last event."""
    tokens, feats, _ = tok.encode(ev, st)
    inc = IncFeatures(tok)
    inc.seed(st[0])
    mids = 0.5 * (st[:, 0] + st[:, 1])
    inc.lm = [float(np.log(x)) for x in mids[-1101:]]
    post = book_state(book_now)
    inc.lm.append(math.log(0.5 * (post[0] + post[1])))
    dt = np.maximum(np.diff(ev["us"], prepend=ev["us"][0]), 0).astype(float)
    inc.dts = list(dt[-64:])
    return Context(tokens[-T:], feats[-T:], inc, book_now, int(ev["us"][-1]))
