"""MMFM v2 simulator: KV-cached model + v2 incremental features + single engine + state-relative decoding.

Decoding of a sampled token (act, side, px, size, dt):
  ADD    price = same-side touch -/+ sampled bp distance (post-only clamp); qty = sampled notional / mid
  CANCEL level = existing level nearest the sampled distance; qty = sampled FRACTION x that level's quantity
         (WHOLE-LEVEL bin removes the level exactly -> the mid moves as in real books)
  TAKE   qty = sampled FRACTION x opposite touch quantity (WHOLE-TOUCH / sweep bins), limit from sampled depth
The event actually applied is re-tokenised (with its pre-state and pre-event level quantity) and fed back.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from data.l3 import ADD, BUY, CANCEL, SELL, TAKE
from data.tokenizer_v2 import N_INSIDE, WHOLE_CANCEL, WHOLE_TAKE, Tokenizer2
from model.infer import FastMMFM
from model.infer2 import IncFeatures2
from sim.book import LevelBook


def book_state(b: LevelBook):
    lb, la = b.levels(BUY, 5), b.levels(SELL, 5)
    if not lb or not la:
        return None
    t = b.tick
    return (lb[0][0] * t, la[0][0] * t, lb[0][1], la[0][1], sum(q for _, q in lb), sum(q for _, q in la))


class Context2:
    def __init__(self, tokens, feats, inc: IncFeatures2, book: LevelBook, last_us: int):
        self.tokens, self.feats, self.inc, self.book, self.last_us = tokens, feats, inc, book, last_us


def context_from_history2(ev, st, lvq, tok: Tokenizer2, book_now: LevelBook, T):
    """Context from history (events + PRE-states + pre-event level quantities) and the CURRENT book.
    Features are produced by replaying the history through IncFeatures2 (equal to batch, tested); the last
    feature row describes the current book."""
    tokens, _, _ = tok.encode(ev, st, lvq)
    n = len(ev)
    k0 = max(0, n - 1100)
    inc = IncFeatures2(tok)
    inc.seed(st[k0])
    post_now = book_state(book_now)
    feats = []
    for i in range(k0, n):
        post = st[i + 1] if i + 1 < n else post_now
        dt = max(int(ev["us"][i]) - int(ev["us"][i - 1]), 0) if i > 0 else 0
        feats.append(inc.push(int(ev["us"][i]), dt, post))
    feats = np.array(feats, np.float32)
    return Context2(tokens[-T:], feats[-T:], inc, book_now, int(ev["us"][-1]))


class Rollout2:
    def __init__(self, model, tok: Tokenizer2, ctx: Context2, n_paths=16, seed=0, temperature=1.0, dt_scale=1.0):
        self.tok, self.temp, self.dt_scale = tok, temperature, dt_scale
        self.rng = np.random.default_rng(seed)
        self.g = torch.Generator().manual_seed(seed)
        self.fm = FastMMFM(model)
        T = model.cfg["ctx"]
        tk = torch.from_numpy(ctx.tokens[-T:])[None].repeat(n_paths, 1, 1)
        ft = torch.from_numpy(ctx.feats[-T:])[None].repeat(n_paths, 1, 1)
        self.h = self.fm.prefill(tk, ft)
        s0 = book_state(ctx.book)
        self.paths = [dict(book=ctx.book.copy(), inc=copy.deepcopy(ctx.inc), us=ctx.last_us, prev_us=ctx.last_us,
                           t=[ctx.last_us], mid=[0.5 * (s0[0] + s0[1])], alive=True, n_ev=0) for _ in range(n_paths)]
        self.t0 = ctx.last_us
        self.gen_seconds, self.n_steps = 0.0, 0

    def _concrete(self, p, row):
        tk = self.tok
        act, side, pxb, szb, dtb = (int(x) for x in row)
        b = p["book"]
        st = book_state(b)
        if st is None:
            return None
        bb, ba = st[0], st[1]
        mid = 0.5 * (bb + ba)
        dt = 0.0 if dtb == 0 else 10 ** tk.sample_q("dt", dtb, self.rng)
        dt_us = int(round(self.dt_scale * dt)) if np.isfinite(dt) else 0
        tick = b.tick
        if act == TAKE:
            opp_q = st[3] if side == BUY else st[2]
            fr = 1.0 if szb == WHOLE_TAKE else tk.sample_q("size_take", szb, self.rng)
            fr = fr if np.isfinite(fr) else 1.0
            qty = max(fr, 1e-6) * opp_q
            sw = tk.sample_q("px", pxb, self.rng)
            sw = max(sw, 0.0) if np.isfinite(sw) else 0.0
            k = int(math.ceil(ba * (1 + sw / 1e4) / tick - 1e-9)) if side == BUY else int(math.floor(bb * (1 - sw / 1e4) / tick + 1e-9))
            if fr >= 1.0:  # a whole-touch / sweep take must at least reach the touch
                k = max(k, int(round(ba / tick))) if side == BUY else min(k, int(round(bb / tick)))
            return act, side, k, qty, dt_us
        d = 0.0 if pxb == N_INSIDE else tk.sample_q("px", pxb, self.rng)
        d = d if np.isfinite(d) else 0.0
        ref = bb if side == BUY else ba
        k = int(round((ref * (1 - d / 1e4) if side == BUY else ref * (1 + d / 1e4)) / tick))
        if act == ADD:
            k = min(k, int(round(ba / tick)) - 1) if side == BUY else max(k, int(round(bb / tick)) + 1)
            nz = tk.sample_q("size_add", szb, self.rng)
            qty = 10 ** nz / mid if np.isfinite(nz) else 0.0
            return act, side, k, qty, dt_us
        lv = b.bids if side == BUY else b.asks
        kk = b.nearest(side, k)
        if kk is None:
            return None
        fr = 1.0 if szb == WHOLE_CANCEL else tk.sample_q("size_cancel", szb, self.rng)
        fr = min(max(fr if np.isfinite(fr) else 1.0, 0.0), 1.0)
        return act, side, kk, fr * lv[kk], dt_us

    def _apply(self, p, o):
        act, side, k, qty, dt_us = o
        b = p["book"]
        pre = book_state(b)
        us = p["us"] + dt_us
        price = k * b.tick
        if act == TAKE:
            opp = b.asks if side == BUY else b.bids
            lvq = pre[3] if side == BUY else pre[2]
            worst, res = None, qty
            for lk in (opp.keys() if side == BUY else reversed(opp.keys())):
                if (side == BUY and lk > k) or (side == SELL and lk < k) or res <= 1e-12:
                    break
                worst = lk
                res -= opp[lk]
            price = (worst if worst is not None else k) * b.tick
        else:
            lvq = (b.bids if side == BUY else b.asks).get(k, 0.0)
        row, dt = p["inc"].tokens_for(act, side, price, qty, us, p["prev_us"], pre, lvq)
        b.apply(act, side, k, qty)
        post = book_state(b)
        if post is None:
            p["alive"] = False
            return row, self.fm.hist_feat[0, -1].numpy()
        f = p["inc"].push(us, dt, post)
        p["prev_us"] = p["us"] = us
        p["t"].append(us)
        p["mid"].append(0.5 * (post[0] + post[1]))
        p["n_ev"] += 1
        return row, f

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
        toks, feats = [], []
        for p in self.paths:
            t, f = self._apply(p, (act, side, k, qty, 0))
            toks.append(t)
            feats.append(f)
        self.h = self.fm.step(torch.from_numpy(np.stack(toks)), torch.from_numpy(np.stack(feats)))

    def run(self, horizon_s, max_steps=40000, wall_budget_s=None):
        ts = time.perf_counter()
        for _ in range(max_steps):
            if all((not p["alive"]) or p["us"] - self.t0 >= horizon_s * 1e6 for p in self.paths):
                break
            if wall_budget_s is not None and time.perf_counter() - ts > wall_budget_s:
                break
            self.step()
        return self

    def mid_grid(self, grid_s):
        out = np.full((len(self.paths), len(grid_s)), np.nan)
        for i, p in enumerate(self.paths):
            t = (np.array(p["t"]) - self.t0) / 1e6
            m = np.array(p["mid"])
            idx = np.searchsorted(t, grid_s, side="right") - 1
            ok = (idx >= 0) & (grid_s <= t[-1])
            out[i, ok] = m[idx[ok]]
        return out

    def counts(self, grid_s):
        return np.array([[int(np.sum((np.array(p["t"][1:]) - self.t0) <= h * 1e6)) for h in grid_s] for p in self.paths])


@torch.no_grad()
def direct_forecast(model, tok: Tokenizer2, ctx: Context2):
    """Calibrated distributions of R_h (bp) and N_h from the forecast heads (no rollout; real-time)."""
    h = model.hidden(torch.from_numpy(ctx.tokens)[None], torch.from_numpy(ctx.feats)[None])[0, -1]
    r, c = model.forecast_logits(h)
    out = []
    for k in range(len(r)):
        pr = torch.softmax(r[k].float(), -1).numpy()
        pc = torch.softmax(c[k].float(), -1).numpy()
        out.append(dict(p_ret=pr, ret_values=tok.ret_bin_values(k), ret_edges=tok.edges_ret[k], p_cnt=pc,
                        cnt_values=tok.cnt_bin_values(k), cnt_edges=tok.edges_cnt[k]))
    return out
