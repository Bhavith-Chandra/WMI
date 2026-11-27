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
    """feat (..., 21[+bands]) raw -> scaled: log1p volumes, log1p(spread-1), log1p depth bands (v3)."""
    v = torch.log1p(feat[..., :20].clamp(min=0))
    s = torch.log1p((feat[..., 20:21] - 1).clamp(min=0))
    if feat.shape[-1] > 21:
        return torch.cat([v, s, torch.log1p(feat[..., 21:].clamp(min=0))], -1)
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
    def __init__(self, d=64, h=128, layers=2, mode='learned', slowbank=False, nsize=NSIZE, ndt=0, size_all=False,
                 src=False, cond_heads=False, tagch=False, ndom=0, tag_gain=1.0, gen_cal=None, nlvl=NLVL, nimp=0,
                 nfeat=21, nsb_extra=0, size_cond=False, side_skip=False,
                 mkt_side_w=1.0, lvl0_mask=False):
        """v1 (synthetic): nsize=4, ndt=0, size only for markets. v2 (real): nsize=12 for all types, ndt=11 (Dt head)."""
        super().__init__()
        self.cfg = dict(d=d, h=h, layers=layers, mode=mode, slowbank=slowbank, nsize=nsize, ndt=ndt, size_all=size_all,
                        src=src, cond_heads=cond_heads, tagch=tagch, ndom=ndom, tag_gain=tag_gain,
                        nlvl=nlvl, nimp=nimp, nfeat=nfeat, nsb_extra=nsb_extra, size_cond=size_cond,
                        side_skip=side_skip, mkt_side_w=mkt_side_w, lvl0_mask=lvl0_mask)
        # v3 (replay-exact real data): nlvl = DR + 1 + nimp level classes; the last nimp limit classes mean 'improve
        # own best by k = 2..nimp+1 buckets' (valid only if spread > k); cancels use the first DR classes (book mask)
        self.nlvl, self.nimp, self.nfeat = nlvl, nimp, nfeat
        self.tag_gain = float(tag_gain)   # Stage-3 post-training scalar: sensitivity to the own-order channel
        # generation-time calibration (realism post-training on FIT data): side temperature, market-logit bias,
        # type temperature. Teacher-forced likelihood is untouched (only sampling uses it).
        self.gen_cal = {**dict(side_temp=1.0, mkt_bias=0.0, typ_temp=1.0, lim_bias=0.0), **(gen_cal or {})}
        self.cfg['gen_cal'] = self.gen_cal
        # v2 token columns: 0 typ, 1 side, 2 lvl, 3 size bucket, 4 dt bucket, [5 src], [6 domain]
        self.ndom = ndom
        self.tagch = tagch
        # cond_heads: SEPARATE output weights per event type (side) and per (type, side) (level, size). Without it the
        # state->side mapping is shared by limits/cancels/markets (only the bias differs), so market-order sides cannot
        # depend on order-flow state (found 2026-11-26: model P(buy|market) flat at 0.5 vs empirical 0.32-0.67).
        self.cond_heads = cond_heads
        self.src = src   # v1 only: token column 4 = source tag (0 market, 1 own/exogenous order); never a target
        self.nsize, self.ndt, self.size_all = nsize, ndt, size_all
        self.e_typ = nn.Embedding(3, d)
        self.e_side = nn.Embedding(2, d)
        self.e_lvl = nn.Embedding(nlvl, d)
        self.e_size = nn.Embedding(nsize, d)
        self.e_dt = nn.Embedding(ndt, d) if ndt else None
        self.e_src = nn.Embedding(2, d) if src else None
        self.e_dom = nn.Embedding(ndom, d) if ndom else None
        nsb = ((2 + (1 if tagch else 0)) * len(SB_TAUS) if slowbank else 0) + nsb_extra   # extra: FlowBank (v3)
        # src models get a SEPARATE slow-bank channel of OWN/exogenous (tagged) signed volume: the market reacts to
        # how much of recent flow was exogenous; mixing it into the total-flow channel hides that (2026-11-26).
        self.slowbank = slowbank
        self.inproj = nn.Linear(d + nfeat + 1 + nsb, d)
        self.blocks = nn.ModuleList([DiagRec(d, h, mode) for _ in range(layers)])
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.h_typ = nn.Linear(d, 3)
        self.c_side = nn.Embedding(3, d)
        self.h_side = nn.Linear(d, 2)
        self.c_lvl = nn.Embedding(6, d)
        self.h_lvl = nn.Linear(d, nlvl)
        self.h_size = nn.Linear(d, nsize)
        if cond_heads:
            self.hc_side = nn.Linear(d, 3 * 2)
            self.hc_lvl = nn.Linear(d, 6 * nlvl)
            self.hc_size = nn.Linear(d, 6 * nsize)
        self.c_size = nn.Embedding(nsize, d) if ndt else None
        # size_cond (v3.4): size | (type, side, LEVEL, log queue volume at that level). Without it a cancel at the
        # best and one 40 levels deep share one size law; whole-queue cancels at the best (21% real) came out at 6%
        # in generation and limit sizes at the best x2 -> depth x2 within 200 events (found 2026-11-26).
        # side_skip (v3.5): direct linear path FlowBank -> market-order side logits. A logistic on the flow features
        # reaches implied trade-sign ACF(1) 0.39 vs the network's 0.26 (real 0.46): the signal is there, the deep
        # path under-uses it (markets are 2.5% of events). mkt_side_w upweights the market-side CE term (its optimum
        # is still the true conditional, so calibration is preserved).
        self.side_skip, self.mkt_side_w, self._fb = side_skip, float(mkt_side_w), None
        self.lvl0_mask = lvl0_mask
        if side_skip:
            self.skip_side = nn.Linear(nsb_extra, 2)
            nn.init.zeros_(self.skip_side.weight); nn.init.zeros_(self.skip_side.bias)
        self.size_cond = size_cond
        if size_cond:
            self.c_lvl_sz = nn.Embedding(nlvl, d)
            self.q_sz = nn.Linear(1, d)
        self.h_dt = nn.Linear(d, ndt) if ndt else None
        self.hdim = h

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def embed(self, ev, feat, dmid, sb):
        if self.ndt:   # v2 token (typ, side, lvl, size_bucket, dt_bucket, [src], [domain])
            e = (self.e_typ(ev[..., 0]) + self.e_side(ev[..., 1]) + self.e_lvl(ev[..., 2]) + self.e_size(ev[..., 3])
                 + self.e_dt(ev[..., 4]))
            if self.src and ev.shape[-1] > 5:
                e = e + self.e_src(ev[..., 5])
            if self.ndom and ev.shape[-1] > 6:
                e = e + self.e_dom(ev[..., 6])
        else:
            e = self.e_typ(ev[..., 0]) + self.e_side(ev[..., 1]) + self.e_lvl(ev[..., 2]) + self.e_size(ev[..., 3] - 1)
            if self.src and ev.shape[-1] > 4:
                e = e + self.e_src(ev[..., 4])
        z = [e, book_transform(feat), torch.clamp(dmid, -5, 5).unsqueeze(-1)]
        if self.slowbank:
            if self.tagch and self.tag_gain != 1.0:
                k = len(SB_TAUS)
                sb = torch.cat([sb[..., :2 * k], sb[..., 2 * k:] * self.tag_gain], -1)
            z.append(sb)
        if self.side_skip:
            self._fb = sb[..., -self.cfg['nsb_extra']:]
        return self.inproj(torch.cat(z, -1))

    def gen_typ_logits(self, z):
        lt = self.h_typ(z) / self.gen_cal['typ_temp']
        return torch.cat([lt[..., :1] + self.gen_cal.get('lim_bias', 0.0), lt[..., 1:2], lt[..., 2:] + self.gen_cal['mkt_bias']], -1)

    def gen_side_logits(self, z, typ):
        ls = self.side_logits(z, typ) / self.gen_cal['side_temp']
        mt = self.gen_cal.get('mkt_side_temp', 1.0)     # market-order side only (trade-sign persistence)
        if mt != 1.0:
            ls = torch.where((typ == 2).unsqueeze(-1), ls / mt, ls)
        return ls

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

    def size_input(self, zl, lvl=None, lq=None):
        """Size-head input: zl (+ level and log-queue conditioning when size_cond)."""
        if self.size_cond and lvl is not None:
            return zl + self.c_lvl_sz(lvl) + self.q_sz(lq.unsqueeze(-1).float())
        return zl

    def size_logits(self, z, typ, side, lvl=None, lq=None):
        zl = z + self.c_side(typ) + self.c_lvl(typ * 2 + side)
        zz = self.size_input(zl, lvl, lq)
        return _pick(self.hc_size(zz), typ * 2 + side, self.nsize) if self.cond_heads else self.h_size(zz)

    def heads_logits(self, z, typ, side, lvl=None, lq=None):
        lt = self.h_typ(z)
        zs = z + self.c_side(typ)
        zl = zs + self.c_lvl(typ * 2 + side)
        if self.cond_heads:
            ls = self._skip(_pick(self.hc_side(zs), typ, 2), typ)
            ll = _pick(self.hc_lvl(zl), typ * 2 + side, self.nlvl)
            lz = _pick(self.hc_size(self.size_input(zl, lvl, lq)), typ * 2 + side, self.nsize)
            return lt, ls, ll, lz
        ls = self.h_side(zs)
        ll = self.h_lvl(zl)
        lz = self.h_size(zl)
        return lt, ls, ll, lz

    # generation-time head access (same math as heads_logits, usable step by step)
    def _skip(self, ls, typ):
        if self.side_skip and self._fb is not None:
            return ls + (typ == 2).unsqueeze(-1).float() * self.skip_side(self._fb)
        return ls

    def side_logits(self, z, typ):
        zs = z + self.c_side(typ)
        return self._skip(_pick(self.hc_side(zs), typ, 2), typ) if self.cond_heads else self.h_side(zs)

    def lvl_size_logits(self, z, typ, side):
        zl = z + self.c_side(typ) + self.c_lvl(typ * 2 + side)
        if self.cond_heads:
            return _pick(self.hc_lvl(zl), typ * 2 + side, self.nlvl), _pick(self.hc_size(zl), typ * 2 + side, self.nsize)
        return self.h_lvl(zl), self.h_size(zl)

    def loss(self, z, tgt, cmask, spread=None, lq=None):
        """tgt (B,T,4); cmask (B,T,2,D) cancel validity of the book BEFORE the event; lq log1p queue volume."""
        typ, side, lvl, size = tgt[..., 0], tgt[..., 1], tgt[..., 2], tgt[..., 3]
        lt, ls, ll, lz = self.heads_logits(z, typ, side, lvl if self.size_cond else None, lq)
        ll = mask_lvl(ll, typ, side, cmask, spread, self.nimp, self.lvl0_mask)
        nll = F.cross_entropy(lt.reshape(-1, 3), typ.reshape(-1), reduction='none')
        sw = 1.0 + (self.mkt_side_w - 1.0) * (typ == 2).reshape(-1).float() if self.training else 1.0
        nll = nll + sw * F.cross_entropy(ls.reshape(-1, 2), side.reshape(-1), reduction='none')
        m_l = (typ != 2).reshape(-1).float()
        nll = nll + m_l * F.cross_entropy(ll.reshape(-1, self.nlvl), lvl.reshape(-1), reduction='none')
        if self.ndt:
            nll = nll + F.cross_entropy(lz.reshape(-1, self.nsize), size.reshape(-1), reduction='none')
            zl = z + self.c_side(typ) + self.c_lvl(typ * 2 + side)
            ld = self.h_dt(zl + self.c_size(size))
            nll = nll + F.cross_entropy(ld.reshape(-1, self.ndt), tgt[..., 4].reshape(-1), reduction='none')
            if self.src and tgt.shape[-1] > 5:     # own/exogenous orders are inputs, never targets
                w = (tgt[..., 5] == 0).reshape(-1).float()
                return (nll * w).sum() / w.sum().clamp(min=1)
            return nll.mean()
        m_z = (typ == 2).reshape(-1).float()
        nll = nll + m_z * F.cross_entropy(lz.reshape(-1, NSIZE), (size - 1).reshape(-1), reduction='none')
        if self.src and tgt.shape[-1] > 4:      # own/exogenous orders are inputs, never prediction targets
            w = (tgt[..., 4] == 0).reshape(-1).float()
            return (nll * w).sum() / w.sum().clamp(min=1)
        return nll.mean()

    def loss_parts(self, z, tgt, cmask, spread=None, lq=None):
        """v2 per-field NLL (B*T, 5): typ, side, lvl (non-market), size, dt."""
        typ, side, lvl, size = tgt[..., 0], tgt[..., 1], tgt[..., 2], tgt[..., 3]
        lt, ls, ll, lz = self.heads_logits(z, typ, side, lvl if self.size_cond else None, lq)
        ll = mask_lvl(ll, typ, side, cmask, spread, self.nimp, self.lvl0_mask)
        zl = z + self.c_side(typ) + self.c_lvl(typ * 2 + side)
        ld = self.h_dt(zl + self.c_size(size))
        p = [F.cross_entropy(lt.reshape(-1, 3), typ.reshape(-1), reduction='none'),
             F.cross_entropy(ls.reshape(-1, 2), side.reshape(-1), reduction='none'),
             (typ != 2).reshape(-1).float() * F.cross_entropy(ll.reshape(-1, self.nlvl), lvl.reshape(-1), reduction='none'),
             F.cross_entropy(lz.reshape(-1, self.nsize), size.reshape(-1), reduction='none'),
             F.cross_entropy(ld.reshape(-1, self.ndt), tgt[..., 4].reshape(-1), reduction='none')]
        return torch.stack(p, -1)


def _pick(flat, idx, k):
    """flat (..., G*k) -> (..., k) selecting group idx (...,)."""
    sh = flat.shape[:-1]
    g = flat.reshape(*sh, -1, k)
    return torch.gather(g, -2, idx.unsqueeze(-1).unsqueeze(-1).expand(*sh, 1, k)).squeeze(-2)


def mask_lvl(ll, typ, side, cmask, spread=None, nimp=0, lvl0_mask=False):
    """cancel: only levels with volume on that side (levels 0..D-1 -> logits 0..D-1, index D invalid).
    limit: all 0..D valid (lvl 0 means inside-spread; engine falls back if spread==1)."""
    neg = torch.finfo(ll.dtype).min / 4
    cm = torch.where(side.unsqueeze(-1) == 0, cmask[..., 0, :], cmask[..., 1, :]).float()
    cm = torch.cat([cm, torch.zeros_like(cm[..., :1 + nimp])], -1)
    is_c = (typ == 1).unsqueeze(-1)
    lim = torch.ones_like(cm)
    if nimp:
        k = torch.arange(2, nimp + 2, dtype=cm.dtype)                   # improvement sizes
        sp = spread.to(cm.dtype).unsqueeze(-1) if spread is not None else torch.zeros_like(cm[..., :1])
        lim = torch.cat([lim[..., :-nimp], (sp > k).to(cm.dtype)], -1)
        if spread is not None and lvl0_mask:
            # v3.7: 'improve by 1' (lvl 0) is impossible at spread 1 (it is the same price as joining the best,
            # tokenised as lvl 1; real P = 0 exactly). Unmasked, the head learned lvl 0 as globally rare (87% of states
            # have spread 1) and under-predicted book resiliency at spread 2 by half (0.038 vs 0.081 teacher-forced).
            lim = torch.cat([(sp > 1).to(cm.dtype), lim[..., 1:]], -1)
    allow = torch.where(is_c, cm, lim)
    # if a cancel side has no volume (should not happen), allow everything to keep logits finite
    allow = torch.where(allow.sum(-1, keepdim=True) > 0, allow, torch.ones_like(allow))
    return torch.where(allow > 0, ll, torch.full_like(ll, neg))


def inv_cdf(logits, u):
    p = torch.softmax(logits, -1)
    c = torch.cumsum(p, -1)
    idx = (c < u.unsqueeze(-1)).sum(-1)
    return idx.clamp(max=logits.shape[-1] - 1)


class SlowBank:
    """Recursive slow features for generation: signed-volume EMAs and price - EMA(price); with tag=True also EMAs of
    own/exogenous (tagged) signed volume."""
    def __init__(self, B, tag=False):
        self.lam = torch.tensor(np.exp(-1.0 / SB_TAUS), dtype=torch.float32)
        self.fv = torch.zeros(B, len(SB_TAUS))
        self.p = torch.zeros(B, 1)
        self.ep = torch.zeros(B, len(SB_TAUS))
        self.tag = tag
        self.ft = torch.zeros(B, len(SB_TAUS))

    def update(self, signed_vol, dmid, signed_tag=None):
        self.fv = self.lam * self.fv + (1 - self.lam) * signed_vol.unsqueeze(-1)
        self.p = self.p + dmid.unsqueeze(-1)
        self.ep = self.lam * self.ep + (1 - self.lam) * self.p
        if self.tag:
            st = torch.zeros_like(signed_vol) if signed_tag is None else signed_tag
            self.ft = self.lam * self.ft + (1 - self.lam) * st.unsqueeze(-1)

    def feat(self):
        f = sb_scale(self.fv, self.p - self.ep)
        return torch.cat([f, self.ft * TAG_SCALE], -1) if self.tag else f


TAG_SCALE = torch.tensor(np.sqrt(SB_TAUS), dtype=torch.float32) / 4.0


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


def tag_offline(signed_tag):
    """Tagged-flow channel, same recursion as SlowBank.ft, scaled (numpy)."""
    from scipy.signal import lfilter
    lam = np.exp(-1.0 / SB_TAUS)
    ft = np.stack([lfilter([1 - l], [1, -l], signed_tag) for l in lam], -1)
    return ft * (np.sqrt(SB_TAUS) / 4.0)


# ---------------- v3 FlowBank: order-flow memory that is NOT dominated by a few huge orders (found 2026-11-26: with
# signed-VOLUME EMAs only, the model's teacher-forced trade-sign ACF(1) was 0.27 vs real 0.47).
FB_MTAUS = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.float64)   # in MARKET-ORDER time
NFB = 2 * len(FB_MTAUS)


class FlowBank:
    """Online: EMAs of market-order signs (updated only on market orders, market-time taus) and event-time EMAs of
    sign * log1p(lots) of market orders (SB_TAUS)."""
    def __init__(self, B):
        self.lm = torch.tensor(np.exp(-1.0 / FB_MTAUS), dtype=torch.float32)
        self.le = torch.tensor(np.exp(-1.0 / SB_TAUS), dtype=torch.float32)
        self.s = torch.zeros(B, len(FB_MTAUS)); self.v = torch.zeros(B, len(SB_TAUS))

    def update(self, is_mkt, sign, lots):
        """is_mkt (B,) bool, sign (B,) +-1, lots (B,)"""
        m = is_mkt.unsqueeze(-1)
        self.s = torch.where(m, self.lm * self.s + (1 - self.lm) * sign.unsqueeze(-1), self.s)
        x = torch.where(is_mkt, sign * torch.log1p(lots), torch.zeros_like(lots))
        self.v = self.le * self.v + (1 - self.le) * x.unsqueeze(-1)

    def feat(self):
        return torch.cat([self.s * 2.0, self.v * torch.tensor(np.sqrt(SB_TAUS), dtype=torch.float32) / 4.0], -1)

    def select(self, mask, other):
        k = mask.unsqueeze(-1)
        self.s = torch.where(k, self.s, other.s); self.v = torch.where(k, self.v, other.v)

    def clone(self):
        c = FlowBank.__new__(FlowBank); c.lm, c.le = self.lm, self.le; c.s, c.v = self.s.clone(), self.v.clone()
        return c


def flowbank_offline(is_mkt, sign, lots):
    """Exactly FlowBank's recursion over a stream (numpy). Row t = state after updates with events 0..t."""
    from scipy.signal import lfilter
    n = len(is_mkt)
    lm = np.exp(-1.0 / FB_MTAUS)
    S = np.zeros((n, len(FB_MTAUS)))
    idx = np.nonzero(is_mkt)[0]
    if len(idx):
        sm = np.stack([lfilter([1 - l], [1, -l], sign[idx].astype(float)) for l in lm], -1)
        pos = np.searchsorted(idx, np.arange(n), side='right') - 1
        S = np.where((pos >= 0)[:, None], sm[np.maximum(pos, 0)], 0.0)
    x = np.where(is_mkt, sign * np.log1p(lots.astype(float)), 0.0)
    V = np.stack([lfilter([1 - l], [1, -l], x) for l in np.exp(-1.0 / SB_TAUS)], -1)
    return np.concatenate([S * 2.0, V * (np.sqrt(SB_TAUS) / 4.0)], -1).astype(np.float32)
