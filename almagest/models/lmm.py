"""Lightweight reactive order-level generator (CPU-trainable).

Per-event recurrent backbone: stacked diagonal linear recurrences (LRU-style, real) with
  * 'learned' timescales (init log-uniform [10, 1000] events, Griffin-like regime), or
  * 'modebank': half of channels pinned to FIXED log-spaced timescales 2^0..2^17 events (non-selective,
    FRAC-lite power-law memory), half learned.
Optional 'slowbank': parameter-free leaky integrators of (signed traded volume) and (price - EMA_tau(price))
over a GENERIC log grid tau = 8 * 4^k (k=0..7), computed only from the model-visible stream.
Factored heads: type -> side -> level -> size, with validity masks from the visible book.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 20
NLVL = D + 1
NSIZE = 4
SB_TAUS = np.array([8 * 4 ** k for k in range(8)], dtype=np.float64)  # 8 .. 131072


def book_transform(feat):
    """feat (..., 21) raw -> (..., 21) scaled."""
    v = torch.log1p(feat[..., :20].clamp(min=0))
    s = torch.log1p((feat[..., 20:21] - 1).clamp(min=0))
    return torch.cat([v, s], -1)


class DiagRec(nn.Module):
    def __init__(self, d_in, h, mode='learned'):
        super().__init__()
        self.inp = nn.Linear(d_in, h)
        self.out = nn.Linear(h, d_in)
        self.mode = mode
        if mode == 'learned':
            tau = torch.exp(torch.empty(h).uniform_(math.log(10), math.log(1000)))
            self.log_tau = nn.Parameter(torch.log(tau))
            self.register_buffer('fixed_log_tau', torch.zeros(0))
        else:
            hf = h // 2
            fixed = torch.log(2.0 ** torch.linspace(0, 17, hf))
            tau = torch.exp(torch.empty(h - hf).uniform_(math.log(10), math.log(1000)))
            self.log_tau = nn.Parameter(torch.log(tau))
            self.register_buffer('fixed_log_tau', fixed)
        self.norm = nn.LayerNorm(d_in)

    def lam(self):
        lt = torch.cat([self.fixed_log_tau, self.log_tau]) if self.mode != 'learned' else self.log_tau
        return torch.exp(-torch.exp(-lt))  # lambda = exp(-1/tau)

    def forward_seq(self, x, h0):
        """x (B,T,d), h0 (B,H) -> y (B,T,d), hT. Sequential scan (T small during training)."""
        lam = self.lam()
        u = self.inp(self.norm(x)) * (1 - lam)
        T = x.shape[1]
        Cn = 16 if T % 16 == 0 else 1
        ar = torch.arange(Cn, dtype=lam.dtype)
        diff = ar[:, None] - ar[None, :]                       # i - j
        loglam = torch.log(lam)                                # (H,)
        Lm = torch.exp(diff.clamp(min=0)[None] * loglam[:, None, None]) * (diff >= 0)[None]  # (H,C,C)
        powi = torch.exp((ar + 1)[:, None] * loglam[None, :])  # (C,H) lambda^(i+1)
        hs = []
        h = h0
        for c in range(0, T, Cn):
            uc = u[:, c:c + Cn]                                # (B,C,H)
            hc = torch.einsum('hij,bjh->bih', Lm, uc) + powi[None] * h[:, None, :]
            hs.append(hc)
            h = hc[:, -1]
        H = torch.cat(hs, 1)
        return x + self.out(F.gelu(H)), h

    def forward_step(self, x, h):
        lam = self.lam()
        h = lam * h + self.inp(self.norm(x)) * (1 - lam)
        return x + self.out(F.gelu(h)), h


class LMM(nn.Module):
    def __init__(self, d=64, h=128, layers=2, mode='learned', slowbank=False):
        super().__init__()
        self.cfg = dict(d=d, h=h, layers=layers, mode=mode, slowbank=slowbank)
        self.e_typ = nn.Embedding(3, d)
        self.e_side = nn.Embedding(2, d)
        self.e_lvl = nn.Embedding(NLVL, d)
        self.e_size = nn.Embedding(NSIZE, d)
        nsb = 2 * len(SB_TAUS) if slowbank else 0
        self.slowbank = slowbank
        self.inproj = nn.Linear(d + 21 + 1 + nsb, d)
        self.blocks = nn.ModuleList([DiagRec(d, h, mode) for _ in range(layers)])
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.h_typ = nn.Linear(d, 3)
        self.c_side = nn.Embedding(3, d)
        self.h_side = nn.Linear(d, 2)
        self.c_lvl = nn.Embedding(6, d)
        self.h_lvl = nn.Linear(d, NLVL)
        self.h_size = nn.Linear(d, NSIZE)
        self.hdim = h

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def embed(self, ev, feat, dmid, sb):
        e = self.e_typ(ev[..., 0]) + self.e_side(ev[..., 1]) + self.e_lvl(ev[..., 2]) + self.e_size(ev[..., 3] - 1)
        z = [e, book_transform(feat), torch.clamp(dmid, -5, 5).unsqueeze(-1)]
        if self.slowbank:
            z.append(sb)
        return self.inproj(torch.cat(z, -1))

    def init_state(self, B):
        return [torch.zeros(B, self.hdim) for _ in self.blocks]

    def trunk_seq(self, x, state):
        new = []
        for blk, h in zip(self.blocks, state):
            x, h = blk.forward_seq(x, h)
            new.append(h)
        return x + self.mlp(x), new

    def trunk_step(self, x, state):
        new = []
        for blk, h in zip(self.blocks, state):
            x, h = blk.forward_step(x, h)
            new.append(h)
        return x + self.mlp(x), new

    def heads_logits(self, z, typ, side):
        lt = self.h_typ(z)
        zs = z + self.c_side(typ)
        ls = self.h_side(zs)
        zl = zs + self.c_lvl(typ * 2 + side)
        ll = self.h_lvl(zl)
        lz = self.h_size(zl)
        return lt, ls, ll, lz

    def loss(self, z, tgt, cmask):
        """tgt (B,T,4); cmask (B,T,2,D) cancel validity of the book BEFORE the event."""
        typ, side, lvl, size = tgt[..., 0], tgt[..., 1], tgt[..., 2], tgt[..., 3]
        lt, ls, ll, lz = self.heads_logits(z, typ, side)
        ll = mask_lvl(ll, typ, side, cmask)
        nll = F.cross_entropy(lt.reshape(-1, 3), typ.reshape(-1), reduction='none')
        nll = nll + F.cross_entropy(ls.reshape(-1, 2), side.reshape(-1), reduction='none')
        m_l = (typ != 2).reshape(-1).float()
        nll = nll + m_l * F.cross_entropy(ll.reshape(-1, NLVL), lvl.reshape(-1), reduction='none')
        m_z = (typ == 2).reshape(-1).float()
        nll = nll + m_z * F.cross_entropy(lz.reshape(-1, NSIZE), (size - 1).reshape(-1), reduction='none')
        return nll.mean()


def mask_lvl(ll, typ, side, cmask):
    """cancel: only levels with volume on that side (levels 0..D-1 -> logits 0..D-1, index D invalid).
    limit: all 0..D valid (lvl 0 means inside-spread; engine falls back if spread==1)."""
    neg = torch.finfo(ll.dtype).min / 4
    cm = torch.where(side.unsqueeze(-1) == 0, cmask[..., 0, :], cmask[..., 1, :]).float()
    cm = torch.cat([cm, torch.zeros_like(cm[..., :1])], -1)
    is_c = (typ == 1).unsqueeze(-1)
    allow = torch.where(is_c, cm, torch.ones_like(cm))
    # if a cancel side has no volume (should not happen), allow everything to keep logits finite
    allow = torch.where(allow.sum(-1, keepdim=True) > 0, allow, torch.ones_like(allow))
    return torch.where(allow > 0, ll, torch.full_like(ll, neg))


def inv_cdf(logits, u):
    p = torch.softmax(logits, -1)
    c = torch.cumsum(p, -1)
    idx = (c < u.unsqueeze(-1)).sum(-1)
    return idx.clamp(max=logits.shape[-1] - 1)


class SlowBank:
    """Recursive slow features for generation: signed-volume EMAs and price - EMA(price)."""
    def __init__(self, B):
        self.lam = torch.tensor(np.exp(-1.0 / SB_TAUS), dtype=torch.float32)
        self.fv = torch.zeros(B, len(SB_TAUS))
        self.p = torch.zeros(B, 1)
        self.ep = torch.zeros(B, len(SB_TAUS))

    def update(self, signed_vol, dmid):
        self.fv = self.lam * self.fv + (1 - self.lam) * signed_vol.unsqueeze(-1)
        self.p = self.p + dmid.unsqueeze(-1)
        self.ep = self.lam * self.ep + (1 - self.lam) * self.p

    def feat(self):
        return sb_scale(self.fv, self.p - self.ep)


def sb_scale(fv, dev):
    return torch.cat([fv * torch.tensor(np.sqrt(SB_TAUS), dtype=torch.float32) / 4.0, dev / 5.0], -1)


def slowbank_offline(signed_vol, dmid):
    """Exact same recursion as SlowBank, vectorized over a whole stream (numpy, for training)."""
    from scipy.signal import lfilter
    lam = np.exp(-1.0 / SB_TAUS)
    p = np.cumsum(dmid)
    fv = np.stack([lfilter([1 - l], [1, -l], signed_vol) for l in lam], -1)
    ep = np.stack([lfilter([1 - l], [1, -l], p) for l in lam], -1)
    return fv, p[:, None] - ep
