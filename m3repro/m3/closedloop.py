"""Closed-loop simulation: AR generation + in-context decoding + matching engine + order injection.

Design (paper's TWAP protocol, plus the pieces the release lacks):
  * rolling AR context: LOB prefix kept, BOS + last `keep` codes re-prefilled when codes exceed `ctx`
    (paper: "LOB prefix retained ... only most recent order-event tokens kept"; NOT in released code)
  * decoding in context: every `block` generated tokens, the last <=1024 codes are decoded jointly
    (training layout), not the release default of decoding generated ids alone
  * injected child orders are encoded with the tokenizer *encoder over the feature history window*
    and appended to the AR context, so later background flow conditions on them (closed loop)
Injection timing granularity = one block (documented approximation).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .engine import TICK, Book


class ClosedLoop:
    def __init__(self, model, tok, prompt_codes, lob, st, book_after_prompt: Book, t0: float,
                 B: int, ctx=1024, keep=512, block=32, temperature=1.0, seed=0):
        self.m, self.tok = model, tok
        self.B, self.ctx, self.keep, self.block, self.temp = B, ctx, keep, block, temperature
        self.lob = lob.expand(B, -1, -1, -1).contiguous()
        self.st = st.expand(B, -1).contiguous()
        self.anchor = float(st[0, 4])
        self.codes = [list(map(int, prompt_codes[0])) for _ in range(B)]
        f = tok.decode(prompt_codes)[0].numpy()
        self.feats = [f.copy() for _ in range(B)]  # feature history aligned with codes
        self.books = [book_after_prompt.copy() for _ in range(B)]
        self.t = np.full(B, t0)
        self.g = torch.Generator().manual_seed(seed)
        self.mid_path = [[] for _ in range(B)]
        self.injected = [[] for _ in range(B)]  # (time, qty_req, qty_filled)
        self._prefill()

    # ---- AR plumbing
    def _prefill(self):
        L = min(self.keep, len(self.codes[0])) if len(self.codes[0]) > self.ctx else len(self.codes[0])
        L = min(L, min(len(c) for c in self.codes))
        ids = torch.tensor([c[-L:] for c in self.codes], dtype=torch.long)
        ids = torch.cat([torch.full((self.B, 1), self.m.bos, dtype=torch.long), ids], 1)
        out = self.m.llama(inputs_embeds=self.m.embeds(ids, self.lob, self.st), use_cache=True, logits_to_keep=1)
        self.pkv, self.ctx_len = out.past_key_values, L
        self.logits = out.logits[:, -1, : self.m.K].float()

    def _feed(self, tokens: torch.Tensor):  # tokens [B,1]
        out = self.m.llama(input_ids=tokens, past_key_values=self.pkv, use_cache=True)
        self.pkv = out.past_key_values
        self.ctx_len += 1
        self.logits = out.logits[:, -1, : self.m.K].float()

    def _sample(self):
        p = torch.softmax(self.logits / max(self.temp, 1e-6), -1)
        return torch.multinomial(p, 1, generator=self.g)

    # ---- engine
    def _apply(self, i, f, injected=False):
        px = int(round(self.anchor * (1 + float(f[0])) / TICK))
        q = int(round(math.expm1(float(f[1])))) if not injected else int(f[5])
        dt = max(float(f[2]), 0.0)
        a, s = int(round(f[3])), int(round(f[4]))
        b = self.books[i]
        if not injected:
            self.t[i] += dt
        filled = 0
        if a == 1:
            res = q
            opp = b.asks if s == 1 else b.bids
            while res > 0 and opp:
                best = min(opp) if s == 1 else max(opp)
                if (s == 1 and best > px) or (s == 0 and best < px):
                    break
                x = min(res, opp[best])
                res -= x
                filled += x
                opp[best] -= x
                if opp[best] <= 0:
                    del opp[best]
            if res > 0:
                own = b.bids if s == 1 else b.asks
                own[px] = own.get(px, 0) + res
        else:
            own = b.bids if s == 1 else b.asks
            have = own.get(px, 0)
            r = min(have, q)
            if r:
                own[px] = have - r
                if own[px] <= 0:
                    del own[px]
        self.mid_path[i].append((self.t[i], b.mid()))
        return filled

    # ---- main loop
    @torch.no_grad()
    def run(self, horizon_s: float, schedule, max_steps=40000):
        """schedule[i] = list of (t_rel, side, qty) child orders for sequence i (aggressive limit +-10 ticks)."""
        t_start = self.t.copy()
        self.t_start = t_start
        ptr = [0] * self.B
        steps = 0
        while (self.t - t_start < horizon_s).any() and steps < max_steps:
            blk = []
            for _ in range(self.block):
                nxt = self._sample()
                blk.append(nxt)
                for i in range(self.B):
                    self.codes[i].append(int(nxt[i]))
                self._feed(nxt)
                if self.ctx_len >= self.ctx:
                    self._prefill()
            steps += self.block
            # decode the block in context (last <=1024 codes, training layout)
            W = min(1024, len(self.codes[0]))
            win = torch.tensor([c[-W:] for c in self.codes], dtype=torch.long)
            fd = self.tok.decode(win).numpy()[:, -self.block:]
            inject_rows = {}
            for i in range(self.B):
                self.feats[i] = np.concatenate([self.feats[i], fd[i]])[-1024:]
                for j in range(self.block):
                    if self.t[i] - t_start[i] >= horizon_s:
                        break
                    self._apply(i, fd[i, j])
                # due child orders (injected after the block in which their time passed)
                while ptr[i] < len(schedule[i]) and self.t[i] - t_start[i] >= schedule[i][ptr[i]][0] \
                        and schedule[i][ptr[i]][0] < horizon_s:
                    _, side, qty = schedule[i][ptr[i]]
                    b = self.books[i]
                    ref = (min(b.asks) + 10) if side == 1 and b.asks else (max(b.bids) - 10 if b.bids else None)
                    if ref is None:
                        ptr[i] += 1
                        continue
                    f = np.array([ref * TICK / self.anchor - 1.0, math.log1p(qty), 0.0, 1.0, float(side), qty])
                    filled = self._apply(i, f, injected=True)
                    self.injected[i].append((self.t[i] - t_start[i], qty, filled))
                    inject_rows.setdefault(i, []).append(f[:5].astype(np.float32))
                    ptr[i] += 1
            if inject_rows:  # encode injected orders in context, append to AR context for ALL rows (pad = own token)
                for i, rows in inject_rows.items():
                    for r in rows:
                        hist = np.concatenate([self.feats[i], r[None]])[-1024:]
                        code = int(self.tok.encode(torch.from_numpy(hist)[None])[0, -1])
                        self.codes[i].append(code)
                        self.feats[i] = hist
                # equalise lengths: rows without injection get nothing; rebuild cache from codes (ragged -> re-prefill)
                self._prefill_ragged()
        return steps

    def _prefill_ragged(self):
        """After injections rows have different lengths; re-prefill each row's last L codes (L=min len, <=keep..ctx)."""
        L = min(min(len(c) for c in self.codes), self.keep)
        ids = torch.tensor([c[-L:] for c in self.codes], dtype=torch.long)
        ids = torch.cat([torch.full((self.B, 1), self.m.bos, dtype=torch.long), ids], 1)
        out = self.m.llama(inputs_embeds=self.m.embeds(ids, self.lob, self.st), use_cache=True, logits_to_keep=1)
        self.pkv, self.ctx_len = out.past_key_values, L
        self.logits = out.logits[:, -1, : self.m.K].float()

    def terminal_mid(self, horizon_s):
        out = np.full(self.B, np.nan)
        for i in range(self.B):
            ok = [m for t, m in self.mid_path[i] if t - self.t_start[i] <= horizon_s and np.isfinite(m)]
            out[i] = ok[-1] if ok else np.nan
        return out
