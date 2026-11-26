"""Real-time inference: KV-cached incremental MMFM + O(1) incremental features/tokens.

Both are tested for equality with the batch path (tests/test_infer.py), so the live interface and
the simulator compute exactly what the model was trained on.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from data.tokenizer import N_DEEP, N_INSIDE, TAKE, Tokenizer
from model.mmfm import FIELDS, MMFM


# ------------------------------------------------------------------------------------------ features/tokens
class IncFeatures:
    """Feature row after each event, identical to data.tokenizer.state_features on the same history."""

    LAGS = ((10, 2.0), (100, 5.0), (1000, 10.0))

    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.lm = []   # log mid of post-event states (= pre-state of the next event)
        self.dts = []
        self.first_lm = None

    def seed(self, pre_state):
        """Initialise with the book state before the first event (batch lags clamp to this state)."""
        self.lm = [math.log(0.5 * (pre_state[0] + pre_state[1]))]

    def tokens_for(self, ev_type, side, price, qty, us, prev_us, pre):
        """Token row for one event given its PRE-event state (bb, ba, ...)."""
        bb, ba = np.float64(pre[0]), np.float64(pre[1])
        mid = 0.5 * (bb + ba)
        price = np.float64(price)
        if ev_type == TAKE:
            px = max(((price - ba) if side == 1 else (bb - price)) / mid * 1e4, 0.0)
            pxt = N_INSIDE + 1 + N_DEEP + int(np.searchsorted(self.tok.edges_take, px))
        else:
            d = ((bb - price) if side == 1 else (price - ba)) / mid * 1e4
            if d < 0:
                pxt = int(np.searchsorted(self.tok.edges_inside, d))
            elif d == 0:
                pxt = N_INSIDE
            else:
                pxt = N_INSIDE + 1 + int(np.searchsorted(self.tok.edges_deep, np.log10(np.float64(d))))
        szt = int(np.searchsorted(self.tok.edges_size, np.log10(np.maximum(np.float64(qty) * mid, 1e-9))))
        dt = max(us - prev_us, 0) if prev_us is not None else 0
        dtt = 0 if dt <= 0 else 1 + int(np.searchsorted(self.tok.edges_dt, np.log10(np.maximum(np.float64(dt), 1e-9))))
        return np.array([ev_type, side, pxt, szt, dtt], np.int64), dt

    def push(self, dt, post):
        """Register an applied event (its dt and POST-event state); return the normalised feature row."""
        bb, ba, qb1, qa1, qb5, qa5 = post
        mid = 0.5 * (bb + ba)
        lm = math.log(mid)
        if not self.lm:
            raise RuntimeError("call seed(pre_state_of_first_event) first")
        self.lm.append(lm)
        self.dts.append(float(dt))
        if len(self.lm) > 1102:  # keep enough for lag 1000 (clamp semantics differ only before 1000 events)
            self.lm = self.lm[-1102:]
        if len(self.dts) > 64:
            self.dts = self.dts[-64:]
        spread_bp = (ba - bb) / mid * 1e4
        f = [math.log1p(max(spread_bp, 0.0)), (qb1 - qa1) / max(qb1 + qa1, 1e-12), (qb5 - qa5) / max(qb5 + qa5, 1e-12),
             math.log10(max((qb1 + qa1) * mid, 1e-9)), math.log10(max((qb5 + qa5) * mid, 1e-9))]
        n = len(self.lm)
        for k, s in self.LAGS:
            ref = self.lm[n - 1 - k] if n - 1 - k >= 0 else self.lm[0]
            f.append(math.tanh((lm - ref) * 1e4 / s))
        f.append(math.log1p(sum(self.dts[-32:]) / 32))
        x = (np.array(f, np.float64) - self.tok.feat_mean) / self.tok.feat_std
        return np.nan_to_num(x).astype(np.float32)


# ------------------------------------------------------------------------------------------ KV-cached model
class FastMMFM:
    """Incremental decoding with a KV cache over a sliding window (re-prefill when the window is full)."""

    def __init__(self, model: MMFM, keep=128):
        self.m = model.eval()
        self.ctx = model.cfg["ctx"]
        self.keep = keep
        self.layers = list(model.blocks.layers)
        self.H = model.cfg["n_heads"]
        self.d = model.cfg["d"]

    def _embed(self, tokens, feats, pos0):
        m = self.m
        x = sum(m.emb[f](tokens[..., k]) for k, f in enumerate(FIELDS)) + m.state(feats)
        T = tokens.shape[1]
        return x + m.pos(torch.arange(pos0, pos0 + T))[None]

    def _attn_layer(self, L, x, cache, causal, li=None):
        B, T, d = x.shape
        H, dh = self.H, d // self.H
        h = L.norm1(x)
        qkv = F.linear(h, L.self_attn.in_proj_weight, L.self_attn.in_proj_bias)
        q, k, v = qkv.split(d, -1)
        q, k, v = (t.view(B, T, H, dh).transpose(1, 2) for t in (q, k, v))
        if cache is not None:  # preallocated buffers: write in place, attend over the filled prefix
            Kb, Vb = cache
            Kb[:, :, self.n:self.n + T] = k
            Vb[:, :, self.n:self.n + T] = v
            k, v = Kb[:, :, :self.n + T], Vb[:, :, :self.n + T]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=causal and cache is None)
        a = a.transpose(1, 2).reshape(B, T, d)
        x = x + L.self_attn.out_proj(a)
        x = x + L.linear2(L.activation(L.linear1(L.norm2(x))))
        return x, (k, v)

    @torch.no_grad()
    def prefill(self, tokens, feats):
        """tokens [B,T,5], feats [B,T,F] -> last hidden [B,d]; builds the cache."""
        T = tokens.shape[1]
        x = self._embed(tokens, feats, 0)
        self.cache = []
        B = tokens.shape[0]
        dh = self.d // self.H
        for L in self.layers:
            x, (k, v) = self._attn_layer(L, x, None, True)
            Kb = torch.empty(B, self.H, self.ctx, dh)
            Vb = torch.empty(B, self.H, self.ctx, dh)
            Kb[:, :, :T] = k
            Vb[:, :, :T] = v
            self.cache.append((Kb, Vb))
        self.n = T
        self.hist_tok, self.hist_feat = tokens, feats
        return self.m.norm(x)[:, -1]

    @torch.no_grad()
    def step(self, tok_row, feat_row):
        """tok_row [B,5], feat_row [B,F] -> hidden [B,d] after consuming this event."""
        self.hist_tok = torch.cat([self.hist_tok, tok_row[:, None]], 1)[:, -self.ctx:]
        self.hist_feat = torch.cat([self.hist_feat, feat_row[:, None]], 1)[:, -self.ctx:]
        if self.n >= self.ctx:  # window full: re-prefill from the last `keep` events
            return self.prefill(self.hist_tok[:, -self.keep:], self.hist_feat[:, -self.keep:])
        x = self._embed(tok_row[:, None], feat_row[:, None], self.n)
        for L, kv in zip(self.layers, self.cache):
            x, _ = self._attn_layer(L, x, kv, False)
        self.n += 1
        return self.m.norm(x)[:, -1]
