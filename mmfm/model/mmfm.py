"""MMFM: state-event decoder over factorized order-flow tokens.

Input at step t : sum_f E_f(token_f[t]) + W_s * state_after[t]   (state after event t = f(past only))
Output at step t: p(event t+1) = p(act) p(side|act) p(px|act,side) p(size|..,px) p(dt|..,size)
The chained heads give an exact factorised likelihood (bits/event) comparable across models/baselines.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

FIELDS = ("act", "side", "px", "size", "dt")


class ChainHeads(nn.Module):
    def __init__(self, d, vocab):
        super().__init__()
        self.vocab = vocab
        self.cond = nn.ModuleDict({f: nn.Embedding(vocab[f], d) for f in FIELDS[:-1]})
        self.out = nn.ModuleDict({f: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, vocab[f])) for f in FIELDS})

    def forward(self, h, target):
        """h [B,T,d], target [B,T,5] -> per-field log-probs of the target tokens [B,T,5] (nats)."""
        lps, z = [], h
        for k, f in enumerate(FIELDS):
            logits = self.out[f](z)
            lps.append(torch.log_softmax(logits.float(), -1).gather(-1, target[..., k:k + 1])[..., 0])
            if k < len(FIELDS) - 1:
                z = z + self.cond[f](target[..., k])
        return torch.stack(lps, -1)

    @torch.no_grad()
    def sample(self, h, generator=None, temperature=1.0):
        """h [B,d] -> tokens [B,5]"""
        z, toks = h, []
        for k, f in enumerate(FIELDS):
            p = torch.softmax(self.out[f](z).float() / temperature, -1)
            t = torch.multinomial(p, 1, generator=generator)[:, 0]
            toks.append(t)
            if k < len(FIELDS) - 1:
                z = z + self.cond[f](t)
        return torch.stack(toks, -1)


class MMFM(nn.Module):
    def __init__(self, vocab, n_feat, d=128, n_layers=4, n_heads=4, ctx=256, dropout=0.0):
        super().__init__()
        self.cfg = dict(vocab=vocab, n_feat=n_feat, d=d, n_layers=n_layers, n_heads=n_heads, ctx=ctx, dropout=dropout)
        self.emb = nn.ModuleDict({f: nn.Embedding(vocab[f], d) for f in FIELDS})
        self.state = nn.Linear(n_feat, d)
        self.pos = nn.Embedding(ctx, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.heads = ChainHeads(d, vocab)

    def hidden(self, tokens, feats):
        B, T, _ = tokens.shape
        x = sum(self.emb[f](tokens[..., k]) for k, f in enumerate(FIELDS)) + self.state(feats)
        x = x + self.pos(torch.arange(T, device=tokens.device))[None]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), 1)
        return self.norm(self.blocks(x, mask=mask, is_causal=True))

    def logp(self, tokens, feats):
        """log p(event t+1 | <= t) for t = 0..T-2  ->  [B, T-1, 5] nats"""
        h = self.hidden(tokens[:, :-1], feats[:, :-1])
        return self.heads(h, tokens[:, 1:])


class FeatMLP(nn.Module):
    """B2 baseline: no sequence attention; last K events (embeddings) + current state -> MLP -> same chain heads."""

    def __init__(self, vocab, n_feat, d=128, K=8):
        super().__init__()
        self.cfg = dict(vocab=vocab, n_feat=n_feat, d=d, K=K, kind="B2_featmlp")
        self.K = K
        self.emb = nn.ModuleDict({f: nn.Embedding(vocab[f], d // 4) for f in FIELDS})
        self.mlp = nn.Sequential(nn.Linear(K * 5 * (d // 4) + n_feat, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.heads = ChainHeads(d, vocab)

    def logp(self, tokens, feats):
        B, T, _ = tokens.shape
        e = torch.cat([self.emb[f](tokens[..., k]) for k, f in enumerate(FIELDS)], -1)  # [B,T,5*d/4]
        pad = torch.zeros(B, self.K - 1, e.shape[-1], device=e.device)
        ep = torch.cat([pad, e], 1)
        win = torch.cat([ep[:, j:j + T] for j in range(self.K)], -1)  # last K events up to t
        h = self.mlp(torch.cat([win, feats], -1))
        return self.heads(h[:, :-1], tokens[:, 1:])


def bits(lp):  # nats -> bits
    return -lp / math.log(2)
