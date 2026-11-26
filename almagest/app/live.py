"""LiveModel: a stateful, single-market wrapper around ONE Almagest checkpoint.

The model is the market: it generates order flow event by event on its own matching engine. A user can submit orders
(fed back to the model as events, so it reacts) and ask what-if questions, answered by forking the current state into
paired rollouts on common random numbers (with vs without the hypothetical metaorder, plus a blind arm where the
order fills but the model never sees it). Nothing here routes orders to any venue.
"""
import time
import numpy as np
import torch
from almagest.sim import world as W, engine as E
from almagest.models import lmm as L
from almagest.experiments.e2_train import build_stream, make_model
from almagest.experiments.e1_calibrate_world import make

torch.set_num_threads(1)


def _sv(ev):
    return torch.where(ev[:, 0] == 2, torch.where(ev[:, 1] == 0, 1.0, -1.0) * ev[:, 3].float(),
                       torch.zeros(ev.shape[0]))


class Rows:
    """Batched market state (B rows): view book, model state, slow bank, pending token."""

    def __init__(self, B):
        self.B = B
        self.bid = np.zeros((B, W.P), np.int64)
        self.ask = np.zeros((B, W.P), np.int64)
        self.rbid = np.zeros((B, W.P), np.int64)
        self.rask = np.zeros((B, W.P), np.int64)
        self.bb = np.zeros(B, np.int64)
        self.ba = np.zeros(B, np.int64)
        self.rbb = np.zeros(B, np.int64)
        self.rba = np.zeros(B, np.int64)
        self.feat = np.zeros((B, 21))
        self.cmb = np.zeros((B, W.D), np.int64)
        self.cma = np.zeros((B, W.D), np.int64)

    def clone_rows(self, idx):
        o = Rows(len(idx))
        for k in ('bid', 'ask', 'rbid', 'rask', 'bb', 'ba', 'rbb', 'rba', 'feat', 'cmb', 'cma'):
            setattr(o, k, getattr(self, k)[idx].copy())
        return o


class LiveModel:
    def __init__(self, ckpt_path, context_seed=7_000_001, context_len=20480):
        ck = torch.load(ckpt_path, weights_only=False)
        self.arm = ck['arm']
        self.model, self.reset = make_model(ck['arm'], cfg=ck.get('cfg'))
        self.model.load_state_dict(ck['state'])
        self.model.eval()
        self.n_params = self.model.n_params()
        self.par = make(ck['theta'])
        self.ckpt = ckpt_path
        self._boot(context_seed, context_len)

    # ---------------------------------------------------------------- boot from an observed context stream
    @torch.no_grad()
    def _boot(self, seed, n):
        """Teacher-force the model on n events of an observed stream (synthetic World context for now; the same
        method takes a recorded real stream once the real-data tokenizer is in)."""
        s = build_stream(self.par, seed, n + 1)
        self.tag = bool(getattr(self.model, 'tagch', False))
        if self.tag and s['sb'].shape[-1] == 2 * len(L.SB_TAUS):   # observed context: no own orders
            s['sb'] = np.concatenate([s['sb'], np.zeros((len(s['sb']), len(L.SB_TAUS)), np.float32)], 1)
        st = self.model.init_state(1)
        for s0 in range(0, n, 256):
            sl = slice(s0, s0 + 256)
            x = self.model.embed(torch.tensor(s['prev'][None, sl]), torch.tensor(s['feat'][None, sl]),
                                 torch.tensor(s['dmid'][None, sl]), torch.tensor(s['sb'][None, sl]))
            _, st = self.model.trunk_seq(x, st)
        self.state = st
        self.rows = Rows(1)
        E.init_rows(self.par, np.array([seed], np.int64), np.array([20000 + n], np.int64), self.rows.bid,
                    self.rows.ask, self.rows.rbid, self.rows.rask, self.rows.bb, self.rows.ba, self.rows.rbb,
                    self.rows.rba)
        self.sbank = L.SlowBank(1, tag=self.tag)
        sv = np.where(s['prev'][:, 0] == 2, np.where(s['prev'][:, 1] == 0, 1, -1) * s['prev'][:, 3], 0)
        from scipy.signal import lfilter
        lam = np.exp(-1.0 / L.SB_TAUS)
        p = np.cumsum(s['dmid'].astype(float))
        fv = np.stack([lfilter([1 - l], [1, -l], sv.astype(float)) for l in lam], -1)
        ep = np.stack([lfilter([1 - l], [1, -l], p) for l in lam], -1)
        self.sbank.fv[0] = torch.tensor(fv[n]); self.sbank.p[0] = float(p[n]); self.sbank.ep[0] = torch.tensor(ep[n])
        self.pend_ev = torch.tensor(s['tgt'][n - 1][None])
        if self.model.src:
            self.pend_ev = torch.cat([self.pend_ev, torch.zeros(1, 1, dtype=self.pend_ev.dtype)], 1)
        self.pend_dmid = torch.tensor([float(s['dmid'][n])])
        self._observe(self.rows)
        self.pend_feat = torch.tensor(self.rows.feat, dtype=torch.float32)
        self.since = np.array([self.reset // 2 if self.reset else 0])
        self.t = 0
        self.mid_hist = [self.mid()]
        self.trade_hist = []
        self.surprise = []
        self.user = dict(position=0, cash=0.0, fills=[])
        self.lat = []

    def _observe(self, r):
        E.observe(r.bid, r.ask, r.bb, r.ba, r.feat, r.cmb, r.cma)

    def mid(self):
        return 0.5 * (self.rows.rbb[0] + self.rows.rba[0])

    def book(self, depth=10):
        r = self.rows
        bb, ba = r.rbb[0], r.rba[0]
        bids = [[int(bb - k), int(r.rbid[0, bb - k])] for k in range(depth) if bb - k >= 0]
        asks = [[int(ba + k), int(r.rask[0, ba + k])] for k in range(depth) if ba + k < W.P]
        return dict(bids=bids, asks=asks, mid=float(self.mid()), spread=int(ba - bb))

    # ---------------------------------------------------------------- core batched step
    @torch.no_grad()
    def _consume(self, rows, state, sbank, pend, since, mask):
        if self.reset:
            rz = torch.tensor((since % self.reset == 0) & mask).unsqueeze(-1)
            state = [torch.where(rz, torch.zeros_like(h), h) for h in state]
        x = self.model.embed(pend['ev'], pend['feat'], pend['dmid'], sbank.feat())
        z, ns = self.model.trunk_step(x, state)
        m = torch.tensor(mask).unsqueeze(-1)
        state = [torch.where(m, n, o) for n, o in zip(ns, state)]
        return z, state, since + mask.astype(np.int64)

    @torch.no_grad()
    def _sample(self, z, rows, u, observed=None):
        m = self.model
        cm = torch.stack([torch.tensor(rows.cmb), torch.tensor(rows.cma)], 1)
        lt = m.h_typ(z)
        typ = L.inv_cdf(lt, u[:, 0]) if observed is None else observed[:, 0]
        ls = m.side_logits(z, typ)
        side = L.inv_cdf(ls, u[:, 1]) if observed is None else observed[:, 1]
        ll, lz = m.lvl_size_logits(z, typ, side)
        ll = L.mask_lvl(ll, typ, side, cm)
        lvl = L.inv_cdf(ll, u[:, 2]) if observed is None else observed[:, 2]
        size = (L.inv_cdf(lz, u[:, 3]) + 1) if observed is None else observed[:, 3]
        lvl = torch.where(typ == 2, torch.zeros_like(lvl), lvl)
        size = torch.where(typ == 2, size, torch.ones_like(size))
        ev = torch.stack([typ, side, lvl, size], 1)
        # surprise (nats) of the realised event under the model
        nll = torch.nn.functional.cross_entropy(lt, typ, reduction='none') + \
            torch.nn.functional.cross_entropy(ls, side, reduction='none')
        nll = nll + torch.where(typ != 2, torch.nn.functional.cross_entropy(ll, lvl, reduction='none'), 0.0)
        nll = nll + torch.where(typ == 2, torch.nn.functional.cross_entropy(lz, size - 1, reduction='none'), 0.0)
        return ev, nll

    def _apply(self, rows, ev, active=None):
        evn = ev.numpy().astype(np.int64)
        act = np.ones(rows.B, np.int64) if active is None else active.astype(np.int64)
        old = 0.5 * (rows.bb + rows.ba).astype(float)
        E.step_events(rows.bid, rows.ask, rows.rbid, rows.rask, rows.bb, rows.ba, rows.rbb, rows.rba,
                      evn[:, 0].copy(), evn[:, 1].copy(), evn[:, 2].copy(), evn[:, 3].copy(), act)
        self._observe(rows)
        return torch.tensor(0.5 * (rows.bb + rows.ba) - old, dtype=torch.float32)

    # ---------------------------------------------------------------- public API
    def step(self, n=1, seed=None):
        """Advance the live market by n model-generated events."""
        g = torch.Generator().manual_seed(int(seed if seed is not None else time.time_ns() % 2**31))
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            pend = dict(ev=self.pend_ev, feat=self.pend_feat, dmid=self.pend_dmid)
            z, self.state, self.since = self._consume(self.rows, self.state, self.sbank, pend, self.since,
                                                      np.ones(1, bool))
            ev, nll = self._sample(z, self.rows, torch.rand(1, 4, generator=g))
            d = self._apply(self.rows, ev)
            if self.model.src:
                ev = torch.cat([ev, torch.zeros(1, 1, dtype=ev.dtype)], 1)
            self.sbank.update(_sv(ev), d)
            self.pend_ev, self.pend_dmid = ev, d
            self.pend_feat = torch.tensor(self.rows.feat, dtype=torch.float32)
            self.lat.append(time.perf_counter() - t0)
            self.t += 1
            e = ev[0].tolist()
            self.mid_hist.append(float(self.mid()))
            self.surprise.append(float(nll[0]))
            if e[0] == 2:
                self.trade_hist.append([self.t, 1 if e[1] == 0 else -1, e[3]])
            out.append(e)
        self.mid_hist = self.mid_hist[-5000:]
        self.surprise = self.surprise[-5000:]
        self.lat = self.lat[-2000:]
        return out

    def submit_market(self, side, size):
        """User aggressive order: executes on the engine and is fed to the model as an event (it reacts)."""
        side = 0 if side in ('buy', 0) else 1
        size = int(max(1, min(size, 40)))
        r = self.rows
        pre_bid, pre_ask = r.rbid[0].copy(), r.rask[0].copy()
        pend = dict(ev=self.pend_ev, feat=self.pend_feat, dmid=self.pend_dmid)
        _, self.state, self.since = self._consume(r, self.state, self.sbank, pend, self.since, np.ones(1, bool))
        old = 0.5 * (r.bb + r.ba).astype(float)
        E.inject(r.bid, r.ask, r.rbid, r.rask, r.bb, r.ba, r.rbb, r.rba, np.array([side]), size,
                 np.array([1]), np.array([1]))
        self._observe(r)
        # fill price from consumed liquidity
        book_before = pre_ask if side == 0 else pre_bid
        book_after = r.rask[0] if side == 0 else r.rbid[0]
        taken = book_before - book_after
        px = np.nonzero(taken > 0)[0]
        notional = float((px * taken[px]).sum())
        qty = int(taken.sum())
        sgn = 1 if side == 0 else -1
        self.user['position'] += sgn * qty
        self.user['cash'] -= sgn * notional
        self.user['fills'].append(dict(t=self.t, side='buy' if side == 0 else 'sell', qty=qty,
                                       avg_px=notional / max(qty, 1)))
        d = torch.tensor(0.5 * (r.bb + r.ba) - old, dtype=torch.float32)
        ev = torch.tensor([[2, side, 0, min(size, 4)] + ([1] if self.model.src else [])])
        self.sbank.update(torch.tensor([float(sgn * size)]), d, torch.tensor([float(sgn * size)]) if self.tag else None)
        self.pend_ev, self.pend_dmid = ev, d
        self.pend_feat = torch.tensor(r.feat, dtype=torch.float32)
        self.mid_hist.append(float(self.mid()))
        return self.user['fills'][-1]

    def pnl(self):
        return dict(position=self.user['position'], cash=self.user['cash'],
                    mark_to_mid=self.user['cash'] + self.user['position'] * float(self.mid()))

    @torch.no_grad()
    def whatif(self, side='buy', children=20, child_size=2, gap=40, tail_blocks=20, paths=64, seed=0):
        """Fork current state into paths x {visible, blind, free} on common random numbers.
        visible: metaorder executes and the model sees it; blind: it executes but the model never sees it
        (mechanical only); free: no metaorder. Returns mean mid paths minus free, with 95% bands."""
        side = 0 if side in ('buy', 0) else 1
        A = 3
        B = paths * A
        idx = np.zeros(B, np.int64)
        rows = self.rows.clone_rows(idx)
        state = [h.repeat(B, 1) for h in self.state]
        sb = L.SlowBank(B, tag=self.tag)
        sb.fv = self.sbank.fv.repeat(B, 1); sb.p = self.sbank.p.repeat(B, 1); sb.ep = self.sbank.ep.repeat(B, 1)
        sb.ft = self.sbank.ft.repeat(B, 1)
        SRC = bool(self.model.src)
        pend = dict(ev=self.pend_ev.repeat(B, 1), feat=self.pend_feat.repeat(B, 1), dmid=self.pend_dmid.repeat(B))
        since = np.repeat(self.since, B)
        arm = np.tile(np.arange(A), paths)       # 0 visible, 1 blind, 2 free
        path = np.repeat(np.arange(paths), A)
        g = torch.Generator().manual_seed(seed)
        nblk = children + tail_blocks
        mids = np.zeros((B, nblk + 1))
        mids[:, 0] = 0.5 * (rows.rbb + rows.rba)
        t0 = time.perf_counter()
        for blk in range(nblk):
            if blk < children:
                vis = arm == 0
                inj = arm <= 1
                z, state, since = self._consume(rows, state, sb, pend, since, vis)
                old = 0.5 * (rows.bb + rows.ba).astype(float)
                E.inject(rows.bid, rows.ask, rows.rbid, rows.rask, rows.bb, rows.ba, rows.rbb, rows.rba,
                         np.full(B, side, np.int64), child_size, vis.astype(np.int64), inj.astype(np.int64))
                self._observe(rows)
                d = torch.tensor(0.5 * (rows.bb + rows.ba) - old, dtype=torch.float32)
                m = torch.tensor(vis)
                cev = torch.tensor([[2, side, 0, min(child_size, 4)] + ([1] if SRC else [])]).repeat(B, 1)
                pend['ev'] = torch.where(m.unsqueeze(-1), cev, pend['ev'])
                pend['feat'] = torch.where(m.unsqueeze(-1), torch.tensor(rows.feat, dtype=torch.float32), pend['feat'])
                pend['dmid'] = torch.where(m, d, pend['dmid'])
                f0, p0, e0, t0_ = sb.fv.clone(), sb.p.clone(), sb.ep.clone(), sb.ft.clone()
                sb.update(_sv(cev), d, _sv(cev) if self.tag else None)
                mm = m.unsqueeze(-1)
                sb.fv = torch.where(mm, sb.fv, f0); sb.p = torch.where(mm, sb.p, p0); sb.ep = torch.where(mm, sb.ep, e0)
                sb.ft = torch.where(mm, sb.ft, t0_)
            for _ in range(gap):
                z, state, since = self._consume(rows, state, sb, pend, since, np.ones(B, bool))
                u = torch.rand(paths, 4, generator=g)[torch.tensor(path)]
                ev, _ = self._sample(z, rows, u)
                d = self._apply(rows, ev)
                if SRC:
                    ev = torch.cat([ev, torch.zeros(B, 1, dtype=ev.dtype)], 1)
                sb.update(_sv(ev), d)
                pend = dict(ev=ev, feat=torch.tensor(rows.feat, dtype=torch.float32), dmid=d)
            mids[:, blk + 1] = 0.5 * (rows.rbb + rows.rba)
        el = time.perf_counter() - t0
        sgn = 1 if side == 0 else -1
        rel = mids - mids[:, :1]
        v, b, f = rel[arm == 0], rel[arm == 1], rel[arm == 2]
        dv, db = sgn * (v - f), sgn * (b - f)   # paired with the free arm on common random numbers

        def band(x):
            mu = x.mean(0)
            se = x.std(0, ddof=1) / np.sqrt(len(x))
            return dict(mean=mu.tolist(), lo=(mu - 1.96 * se).tolist(), hi=(mu + 1.96 * se).tolist())
        return dict(side='buy' if side == 0 else 'sell', children=children, child_size=child_size, gap=gap,
                    paths=paths, blocks=nblk, visible=band(dv), blind=band(db),
                    peak=float(dv[:, children].mean()), end=float(dv[:, -1].mean()),
                    seconds=el, events_simulated=int(B * nblk * gap))

    def status(self):
        lat = np.array(self.lat) if self.lat else np.array([np.nan])
        s = np.array(self.surprise[-500:]) if self.surprise else np.array([np.nan])
        return dict(t=self.t, arm=self.arm, params=self.n_params, ckpt=self.ckpt.rsplit('/', 1)[-1],
                    lat_p50_ms=float(np.nanpercentile(lat, 50) * 1e3), lat_p99_ms=float(np.nanpercentile(lat, 99) * 1e3),
                    surprise_mean=float(np.nanmean(s)), pnl=self.pnl())
