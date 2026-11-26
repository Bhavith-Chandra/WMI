"""MMFM v2: one causal trunk, two kinds of heads.
  * event heads  : chained p(act) p(side|.) p(px|.) p(size|.) p(dt|.) -> generative simulator
  * forecast heads: for h in {1,5,10}s, categorical distributions of the return R_h (bp bins) and the event
                    count N_h (log bins), trained on REAL futures with proper scoring (log loss) and
                    temperature-calibrated on val -> calibrated, real-time forecasts without long rollouts.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from model.mmfm import FIELDS, ChainHeads


class MMFM2(nn.Module):
    def __init__(self, vocab, n_feat, n_ret, n_cnt, d=160, n_layers=4, n_heads=4, ctx=64, dropout=0.0):
        super().__init__()
        self.cfg = dict(vocab=vocab, n_feat=n_feat, n_ret=list(n_ret), n_cnt=list(n_cnt), d=d, n_layers=n_layers,
                        n_heads=n_heads, ctx=ctx, dropout=dropout, version=2)
        self.emb = nn.ModuleDict({f: nn.Embedding(vocab[f], d) for f in FIELDS})
        self.state = nn.Sequential(nn.Linear(n_feat, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Embedding(ctx, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.heads = ChainHeads(d, vocab)
        self.ret = nn.ModuleList(nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n)) for n in n_ret)
        self.cnt = nn.ModuleList(nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n)) for n in n_cnt)
        # calibration temperatures (fit on val after training; 1 = uncalibrated)
        self.register_buffer("t_ret", torch.ones(len(n_ret)))
        self.register_buffer("t_cnt", torch.ones(len(n_cnt)))

    def hidden(self, tokens, feats):
        B, T, _ = tokens.shape
        x = sum(self.emb[f](tokens[..., k]) for k, f in enumerate(FIELDS)) + self.state(feats)
        x = x + self.pos(torch.arange(T, device=tokens.device))[None]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), 1)
        return self.norm(self.blocks(x, mask=mask, is_causal=True))

    def logp(self, tokens, feats):
        h = self.hidden(tokens[:, :-1], feats[:, :-1])
        return self.heads(h, tokens[:, 1:])

    def forecast_logits(self, h, calibrated=True):
        r = [head(h) / (self.t_ret[k] if calibrated else 1.0) for k, head in enumerate(self.ret)]
        c = [head(h) / (self.t_cnt[k] if calibrated else 1.0) for k, head in enumerate(self.cnt)]
        return r, c

    def losses(self, tokens, feats, rt, ct, inputs=None):
        """bits: event NLL per event [B,T-1,5]; forecast NLL per position for returns/counts (masked).
        `inputs` (scheduled sampling, EVAL_v3): tokens fed to the trunk; targets are always the real `tokens`."""
        h_all = self.hidden(tokens if inputs is None else inputs, feats)
        ev = self.heads(h_all[:, :-1], tokens[:, 1:])
        r, c = self.forecast_logits(h_all, calibrated=False)
        lr, lc = [], []
        for k in range(len(r)):
            for logits, tgt, out in ((r[k], rt[..., k], lr), (c[k], ct[..., k], lc)):
                m = tgt >= 0
                lp = torch.log_softmax(logits.float(), -1).gather(-1, tgt.clamp_min(0)[..., None])[..., 0]
                out.append(-(lp * m).sum() / m.sum().clamp_min(1) / math.log(2))
        return -ev / math.log(2), torch.stack(lr), torch.stack(lc)
