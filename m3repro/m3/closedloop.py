"""Closed-loop simulation: AR generation + in-context decoding + matching engine + order injection.

Independence guarantee (v2): every row of the batch is a statistically independent path.
  * one shared KV cache; after every block each row gets `slots` injection positions (masked pad if it
    has nothing to inject) with explicit position ids, so the KV layout and every row's attention / RoPE
    state is identical whatever the other rows do (bit-exact, tested)
  * the rolling-context schedule depends only on the number of *generated* steps (identical for all
    rows, never on another row's injections); at a roll every row is re-prefilled from its own last
    `keep` real codes (equal length -> no padding)
  * sampling uses inverse-CDF with one uniform per row per step from the shared generator, so row i's
    random stream does not depend on other rows' probabilities
  * in-context decoding groups rows by history length (window = own last <=1024 codes)
Child orders are IOC: an add at the aggressive limit followed by a cancel of any unfilled residual,
both executed by the single engine (`engine.apply_event`) and both encoded into the model context,
exactly as an IOC appears in order-level data. Children execute only strictly before the horizon.
Injection timing granularity = one block (documented approximation).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from .engine import TICK, Book, apply_event


class ClosedLoop:
    def __init__(self, model, tok, prompt_codes, lob, st, book_after_prompt: Book, t0: float,
                 B: int, ctx=1024, keep=512, block=16, temperature=1.0, seed=0, log_every=10, reserve=64, slots=2):
        self.m, self.tok = model, tok
        self.B, self.ctx, self.keep, self.block, self.temp = B, ctx, keep, block, temperature
        self.log_every = log_every
        self.roll_at = ctx - keep - reserve  # generated steps between rolls (reserve = room for injected tokens)
        self.lob = lob.expand(B, -1, -1, -1).contiguous()
        self.st = st.expand(B, -1).contiguous()
        self.anchor = float(st[0, 4])
        self.codes = [list(map(int, prompt_codes[0])) for _ in range(B)]
        f = tok.decode(prompt_codes)[0].numpy()
        self.feats = [f.copy() for _ in range(B)]
        self.books = [book_after_prompt.copy() for _ in range(B)]
        self.t = np.full(B, float(t0))
        self.t_start = self.t.copy()
        self.g = torch.Generator().manual_seed(seed)
        self.mid_path = [[] for _ in range(B)]          # (t, mid) after every applied event
        self.injected = [[] for _ in range(B)]          # dict per child
        self.n_ev = np.zeros(B, dtype=np.int64)
        self.n_dt0 = np.zeros(B, dtype=np.int64)
        self.done = np.zeros(B, dtype=bool)
        self.pad = int(model.c["pad_token_id"])
        self.n_rolls = 0
        self.slots = slots
        self.queue = [[] for _ in range(B)]
        self.n_deferred = 0  # row-blocks where child events waited for the next block's slots
        assert keep + self.roll_at + slots * (self.roll_at // block + 1) <= ctx, 'context overflow'
        self._prefill_all()

    # ------------------------------------------------------------------ AR plumbing
    def _prefill_all(self):
        L = min(self.keep, min(len(c) for c in self.codes))
        ids = torch.tensor([c[-L:] for c in self.codes], dtype=torch.long)
        ids = torch.cat([torch.full((self.B, 1), self.m.bos, dtype=torch.long), ids], 1)
        emb = self.m.embeds(ids, self.lob, self.st)
        out = self.m.llama(inputs_embeds=emb, use_cache=True, logits_to_keep=1)
        self.pkv = out.past_key_values
        self.attn = torch.ones((self.B, emb.shape[1]), dtype=torch.long)
        self.pos = torch.full((self.B,), emb.shape[1], dtype=torch.long)
        self.logits = out.logits[:, -1, : self.m.K].float()
        self.gen_since_roll = 0

    def _feed(self, tokens: torch.Tensor, mask: torch.Tensor):
        """tokens [B] (pad where mask==0); masked rows keep state and logits unchanged."""
        self.attn = torch.cat([self.attn, mask.long()[:, None]], 1)
        out = self.m.llama(input_ids=tokens[:, None], attention_mask=self.attn, position_ids=self.pos[:, None],
                           past_key_values=self.pkv, use_cache=True)
        self.pkv = out.past_key_values
        new = out.logits[:, -1, : self.m.K].float()
        self.logits = torch.where(mask[:, None], new, self.logits)
        self.pos = self.pos + mask.long()

    def _sample(self) -> torch.Tensor:
        p = torch.softmax(self.logits / max(self.temp, 1e-6), -1).double()
        cdf = p.cumsum(-1)
        u = torch.rand(self.B, generator=self.g, dtype=torch.float64)[:, None] * cdf[:, -1:]
        return torch.searchsorted(cdf, u).clamp_max(self.m.K - 1)[:, 0]

    # ------------------------------------------------------------------ engine glue
    def _order(self, f):
        px = int(round(self.anchor * (1 + float(f[0])) / TICK))
        q = max(int(round(math.expm1(float(f[1])))), 0)
        return px, q, int(round(float(f[3]))), int(round(float(f[4])))

    def _apply_generated(self, i, f):
        dt = float(f[2])
        self.t[i] += max(dt, 0.0)
        self.n_ev[i] += 1
        self.n_dt0[i] += dt <= 1e-3
        apply_event(self.books[i], *self._order(f))
        self.mid_path[i].append((self.t[i], self.books[i].mid()))

    def _decode_rows(self, rows, n_last):
        """Decode, in context, the last n_last codes of each row (window = own last <=1024 codes)."""
        out, by_len = {}, {}
        for i in rows:
            by_len.setdefault(min(1024, len(self.codes[i])), []).append(i)
        for W, idx in by_len.items():
            win = torch.tensor([self.codes[i][-W:] for i in idx], dtype=torch.long)
            fd = self.tok.decode(win).numpy()[:, -n_last:]
            for k, i in enumerate(idx):
                out[i] = fd[k]
        return out

    def _encode_event(self, i, f5):
        hist = np.concatenate([self.feats[i], np.asarray(f5, np.float32)[None]])[-1024:]
        code = int(self.tok.encode(torch.from_numpy(hist)[None])[0, -1])
        self.feats[i] = hist
        self.codes[i].append(code)
        return code

    # ------------------------------------------------------------------ main loop
    @torch.no_grad()
    def run(self, horizon_s: float, schedule, max_steps=20000):
        """schedule[i] = list of (t_rel, side, qty) IOC child orders (limit = touch -/+ 10 ticks)."""
        ptr = [0] * self.B
        steps = 0
        w0 = time.time()
        allm = torch.ones(self.B, dtype=torch.bool)
        while not self.done.all() and steps < max_steps:
            for _ in range(self.block):
                nxt = self._sample()
                for i in range(self.B):
                    self.codes[i].append(int(nxt[i]))
                self._feed(nxt, allm)
                self.gen_since_roll += 1
                if self.gen_since_roll >= self.roll_at:
                    self._prefill_all()
                    self.n_rolls += 1
            steps += self.block
            fd = self._decode_rows(range(self.B), self.block)
            pending = {}
            for i in range(self.B):
                self.feats[i] = np.concatenate([self.feats[i], fd[i]])[-1024:]
                for j in range(self.block):
                    if self.done[i]:
                        break
                    self._apply_generated(i, fd[i][j])
                    if self.t[i] - self.t_start[i] >= horizon_s:
                        self.done[i] = True
                if self.done[i]:
                    continue
                el = self.t[i] - self.t_start[i]
                while ptr[i] < len(schedule[i]) and el >= schedule[i][ptr[i]][0]:
                    t_s, side, qty = schedule[i][ptr[i]]
                    ptr[i] += 1
                    b = self.books[i]
                    opp = b.asks if side == 1 else b.bids
                    rec = dict(t_sched=t_s, t_exec=el, side=side, qty=qty, filled=0, skipped=not opp)
                    if opp:
                        lim = (min(opp) + 10) if side == 1 else (max(opp) - 10)
                        f_add = [lim * TICK / self.anchor - 1.0, math.log1p(qty), 0.0, 1.0, float(side)]
                        filled, _, _ = apply_event(b, lim, qty, 1, side)
                        evs = [f_add]
                        resid = qty - filled
                        if resid > 0:  # IOC: cancel the residual immediately
                            apply_event(b, lim, resid, 0, side)
                            evs.append([lim * TICK / self.anchor - 1.0, math.log1p(resid), 0.0, 0.0, float(side)])
                        rec.update(filled=filled, limit_tick=lim, n_events=len(evs))
                        self.mid_path[i].append((self.t[i], b.mid()))
                        pending.setdefault(i, []).extend(evs)
                    self.injected[i].append(rec)
            for i, evs in pending.items():
                self.queue[i].extend(evs)
            # fixed injection slots every block for EVERY row (pad+mask when nothing queued): the KV layout
            # is then identical whatever other rows do -> bit-exact row independence
            fed = {}
            for r in range(self.slots):
                mask = torch.zeros(self.B, dtype=torch.bool)
                toks = torch.full((self.B,), self.pad, dtype=torch.long)
                for i in range(self.B):
                    if self.queue[i]:
                        toks[i] = self._encode_event(i, self.queue[i].pop(0))
                        mask[i] = True
                        fed[i] = fed.get(i, 0) + 1
                self._feed(toks, mask)
            self.n_deferred += sum(len(q) > 0 for q in self.queue)
            for i, k in fed.items():  # round-trip: what does the decoder think the model saw?
                dec = self._decode_rows([i], k)[i]
                rec = [x for x in self.injected[i] if "limit_tick" in x][-1]
                rec.setdefault("roundtrip_tick_err", []).extend(
                    abs(int(round(self.anchor * (1 + float(d[0])) / TICK)) - rec["limit_tick"]) for d in dec)
            if self.log_every and (steps // self.block) % self.log_every == 0:
                el = self.t - self.t_start
                print(f"[cl] steps={steps} sim_t min/med/max={el.min():.1f}/{np.median(el):.1f}/{el.max():.1f}s "
                      f"done={self.done.sum()}/{self.B} rolls={self.n_rolls} wall={time.time() - w0:.0f}s", flush=True)
        return steps

    def terminal(self, horizon_s):
        """Per row: mid at the last event with t-t_start <= horizon; stale=True if that mid was NaN
        (one side empty) and the last finite mid was used instead."""
        mids, stale = np.full(self.B, np.nan), np.zeros(self.B, dtype=bool)
        for i in range(self.B):
            ok = [m for t, m in self.mid_path[i] if t - self.t_start[i] <= horizon_s]
            if not ok:
                continue
            if np.isfinite(ok[-1]):
                mids[i] = ok[-1]
            else:
                fin = [m for m in ok if np.isfinite(m)]
                mids[i] = fin[-1] if fin else np.nan
                stale[i] = True
        return mids, stale
