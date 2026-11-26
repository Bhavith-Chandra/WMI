"""Track B2/B3/B5 on REAL data (Coinbase, held-out hours): does the foundation model's market look and react like the
real one? usage: e3_realism.py <cache_dir> <split.json> <model.pt> <out.json> <select|report> [n_windows]

Per window (start inside the chosen split's hours): teacher-force the model on the preceding real context (state
carried within the row, exactly as in training), seed the engine book from the observed top-10 levels (deeper levels
empty: stated approximation), then:
  B2/B3 free generation for GEN events vs the REAL continuation of the same window (and a real-vs-real noise floor):
        spread, top-of-book depth, imbalance, market share, Dt bucket distribution, trade-sign ACF(1..20),
        mid-change variance at horizons 10/100/1000 events (variance-ratio shape).
  B5   paired tagged what-ifs (own orders, src=1, own-order channel), BOTH sides -> directional impact
        (I_buy - I_sell)/2 in bp; R10 (10 children) and R100 (100 children) at the window's participation 10%;
        exponent/amplitude fitted on the build-up; compared with the empirical ranges in EVAL_PROTOCOL B5.
"""
import sys, os, json, time
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest.experiments.e3_real import load_rows, arm_model, WIN
from almagest.models import lmm as L
from almagest.sim import world as W, engine as E
from almagest.data import coinbase_tokenize as T

torch.set_num_threads(1)
GEN = 4000
DT_MID = np.concatenate([[0.0], np.sqrt(np.maximum(T.DT_EDGES[1:-1], 1e-6) * T.DT_EDGES[2:]), [T.DT_EDGES[-1] * 2]])


def seed_book(feat_rows):
    R = len(feat_rows)
    bid = np.zeros((R, W.P), np.int64); ask = np.zeros((R, W.P), np.int64)
    c = W.P // 2
    sp = np.maximum(1, feat_rows[:, 20].astype(int))
    bb = np.full(R, c, np.int64); ba = c + sp
    for k in range(10):
        bid[np.arange(R), bb - k] = np.maximum(feat_rows[:, k], 0).astype(np.int64)
        ask[np.arange(R), ba + k] = np.maximum(feat_rows[:, 10 + k], 0).astype(np.int64)
    bid[np.arange(R), bb] = np.maximum(bid[np.arange(R), bb], 1); ask[np.arange(R), ba] = np.maximum(ask[np.arange(R), ba], 1)
    return bid, ask, bb, ba.astype(np.int64)


class Gen:
    """Batched v2 generator from given per-row state (model state, slow bank, pending token, book)."""
    def __init__(self, model, state, sb, prev, pfeat, pdmid, bid, ask, bb, ba, U):
        self.m, self.state, self.sb, self.prev, self.pfeat, self.pdmid = model, state, sb, prev, pfeat, pdmid
        self.vb, self.va, self.rb, self.ra = bid, ask, bid.copy(), ask.copy()
        self.vbb, self.vba, self.rbb, self.rba = bb, ba, bb.copy(), ba.copy()
        self.B = len(bb); self.U = U
        self.feat = np.zeros((self.B, 21)); self.cmb = np.zeros((self.B, W.D), np.int64); self.cma = np.zeros_like(self.cmb)
        E.observe(self.vb, self.va, self.vbb, self.vba, self.feat, self.cmb, self.cma)

    def _consume(self, mask):
        x = self.m.embed(self.prev, self.pfeat, self.pdmid, self.sb.feat())
        z, ns = self.m.trunk_step(x, self.state)
        mm = torch.tensor(mask).unsqueeze(-1)
        self.state = [torch.where(mm, n, o) for n, o in zip(ns, self.state)]
        return z

    def _pend(self, mask, ev, old):
        E.observe(self.vb, self.va, self.vbb, self.vba, self.feat, self.cmb, self.cma)
        d = torch.tensor(0.5 * (self.vbb + self.vba) - old, dtype=torch.float32)
        mm = torch.tensor(mask)
        self.prev = torch.where(mm.unsqueeze(-1), ev, self.prev)
        self.pfeat = torch.where(mm.unsqueeze(-1), torch.tensor(self.feat, dtype=torch.float32), self.pfeat)
        self.pdmid = torch.where(mm, d, self.pdmid)
        lots = torch.pow(2.0, ev[:, 3].float())
        sv = torch.where(ev[:, 0] == 2, torch.where(ev[:, 1] == 0, 1.0, -1.0) * lots, torch.zeros(self.B))
        st = torch.where(ev[:, 5] == 1, sv, torch.zeros(self.B)) if (self.sb.tag and ev.shape[-1] > 5) else None
        f0, p0, e0, t0 = self.sb.fv.clone(), self.sb.p.clone(), self.sb.ep.clone(), self.sb.ft.clone()
        self.sb.update(sv, d, st)
        k = mm.unsqueeze(-1)
        self.sb.fv = torch.where(k, self.sb.fv, f0); self.sb.p = torch.where(k, self.sb.p, p0)
        self.sb.ep = torch.where(k, self.sb.ep, e0); self.sb.ft = torch.where(k, self.sb.ft, t0)

    def inject(self, side, lots, vis, inj):
        old = 0.5 * (self.vbb + self.vba).astype(float)
        if vis.any():
            self._consume(vis)
        E.inject(self.vb, self.va, self.rb, self.ra, self.vbb, self.vba, self.rbb, self.rba, side, int(lots),
                 vis.astype(np.int64), inj.astype(np.int64))
        cev = torch.stack([torch.full((self.B,), 2), torch.tensor(side), torch.zeros(self.B, dtype=torch.long),
                           torch.full((self.B,), T.size_bucket(lots)), torch.zeros(self.B, dtype=torch.long)], 1)
        if self.U:
            cev = torch.cat([cev, torch.ones(self.B, 1, dtype=torch.long), torch.ones(self.B, 1, dtype=torch.long)], 1)
        self._pend(vis, cev, old)

    def step(self, u):
        m = self.m
        z = self._consume(np.ones(self.B, bool))
        typ = L.inv_cdf(m.h_typ(z), u[:, 0])
        sd = L.inv_cdf(m.side_logits(z, typ), u[:, 1])
        ll, lz = m.lvl_size_logits(z, typ, sd)
        cm = torch.stack([torch.tensor(self.cmb), torch.tensor(self.cma)], 1)
        lvl = L.inv_cdf(L.mask_lvl(ll, typ, sd, cm), u[:, 2])
        szb = L.inv_cdf(lz, u[:, 3])
        zl = z + m.c_side(typ) + m.c_lvl(typ * 2 + sd)
        dtb = L.inv_cdf(m.h_dt(zl + m.c_size(szb)), u[:, 4])
        lvl = torch.where(typ == 2, torch.zeros_like(lvl), lvl)
        ev = torch.stack([typ, sd, lvl, szb, dtb], 1)
        if self.U:
            ev = torch.cat([ev, torch.zeros(self.B, 1, dtype=ev.dtype), torch.ones(self.B, 1, dtype=ev.dtype)], 1)
        evn = ev.numpy().astype(np.int64)
        old = 0.5 * (self.vbb + self.vba).astype(float)
        E.step_events(self.vb, self.va, self.rb, self.ra, self.vbb, self.vba, self.rbb, self.rba, evn[:, 0].copy(),
                      evn[:, 1].copy(), evn[:, 2].copy(), np.power(2, evn[:, 3]).astype(np.int64),
                      np.ones(self.B, np.int64))
        self._pend(np.ones(self.B, bool), ev, old)
        return evn

    def mid(self):
        return 0.5 * (self.rbb + self.rba).astype(float)


def stats_from(ev, feat, mid):
    """Distributional summary of an event block (generated or real)."""
    mk = ev[:, 0] == 2
    s = np.where(ev[mk, 1] == 0, 1.0, -1.0)
    acf = [float(np.mean(s[:-l] * s[l:])) if len(s) > l + 5 else np.nan for l in (1, 2, 5, 10, 20)]
    vr = []
    for h in (10, 100, 1000):
        dm = mid[h:] - mid[:-h]
        vr.append(float(np.var(dm)) if len(dm) > 5 else np.nan)
    return dict(spread=np.bincount(np.minimum(feat[:, 20].astype(int), 8), minlength=9) / len(feat),
                depth=np.log1p(feat[:, 0] + feat[:, 10]), imb=(feat[:, 0] - feat[:, 10]) / (feat[:, 0] + feat[:, 10] + 1e-9),
                mshare=float(mk.mean()), dt=np.bincount(ev[:, 4], minlength=T.NDT) / len(ev), acf=acf, var=vr)


def wass(a, b):
    a, b = np.sort(a), np.sort(b)
    q = np.linspace(0, 1, 200)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def compare(g, r):
    return dict(spread_L1=float(np.abs(g['spread'] - r['spread']).sum()), depth_W1=wass(g['depth'], r['depth']),
                imb_W1=wass(g['imb'], r['imb']), mshare_ratio=g['mshare'] / max(r['mshare'], 1e-9),
                dt_L1=float(np.abs(g['dt'] - r['dt']).sum()),
                acf1=(g['acf'][0], r['acf'][0]), acf10=(g['acf'][3], r['acf'][3]),
                vol100_ratio=g['var'][1] / max(r['var'][1], 1e-12), vol1000_ratio=g['var'][2] / max(r['var'][2], 1e-12))


def main():
    cache, split, mp, out, which = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4], sys.argv[5]
    nwin = int(sys.argv[6]) if len(sys.argv) > 6 else 64
    assert which in ('select', 'report') and not os.path.exists(out)
    ck = torch.load(mp, weights_only=False)
    model, reset = arm_model(ck['arm'], cfg=ck.get('cfg')); model.load_state_dict(ck['state']); model.eval()
    U = ck['arm'] == 'unified'
    rows = [r for r in load_rows(cache, split[which], unified=U) if len(r['tgt']) >= 4 * WIN + GEN * 2]
    rng = np.random.default_rng(0)
    picks = []
    for _ in range(nwin):
        i = int(rng.integers(len(rows))); r = rows[i]
        s = int(rng.integers(2 * WIN, (len(r['tgt']) - GEN * 2) // WIN)) * WIN // 1
        s = (s // WIN) * WIN
        picks.append((i, s))
    t0 = time.time()
    # teacher-forced states at window starts
    states, sbs, prevs, pfeats, pdmids, books = [], [], [], [], [], []
    with torch.no_grad():
        for i, s in picks:
            r = rows[i]; d = {k: torch.tensor(r[k][None, :s]) for k in r}
            st = model.init_state(1)
            for s0 in range(0, s, WIN):
                sl = slice(s0, s0 + WIN)
                x = model.embed(d['prev'][:, sl], d['feat'][:, sl], d['dmid'][:, sl], d['sb'][:, sl])
                _, st = model.trunk_seq(x, st)
            states.append(st)
            sb = L.SlowBank(1, tag=bool(getattr(model, 'tagch', False)))   # rebuild exactly from the row's history
            pr = r['prev'][:s + 1]; lots = np.power(2.0, pr[:, 3])
            sv = np.where(pr[:, 0] == 2, np.where(pr[:, 1] == 0, 1, -1) * lots, 0.0)
            fv, dev = L.slowbank_offline(sv, r['dmid'][:s + 1].astype(float))
            from scipy.signal import lfilter
            lam = np.exp(-1.0 / L.SB_TAUS); pcum = np.cumsum(r['dmid'][:s + 1].astype(float))
            sb.fv[0] = torch.tensor(fv[s], dtype=torch.float32); sb.p[0] = float(pcum[s])
            sb.ep[0] = torch.tensor(np.stack([lfilter([1 - l], [1, -l], pcum) for l in lam], -1)[s], dtype=torch.float32)
            sbs.append(sb); prevs.append(torch.tensor(r['prev'][s][None])); pfeats.append(torch.tensor(r['feat'][s][None]))
            pdmids.append(torch.tensor([float(r['dmid'][s])])); books.append(r['feat'][s])
    books = np.stack(books)

    def batch(rep):
        idx = np.repeat(np.arange(nwin), rep)
        st = [torch.cat([states[i][li] for i in idx]) for li in range(len(model.blocks))]
        sb = L.SlowBank(len(idx), tag=bool(getattr(model, 'tagch', False)))
        sb.fv = torch.cat([sbs[i].fv for i in idx]); sb.p = torch.cat([sbs[i].p for i in idx])
        sb.ep = torch.cat([sbs[i].ep for i in idx])
        bid, ask, bb, ba = seed_book(books[idx])
        return Gen(model, st, sb, torch.cat([prevs[i] for i in idx]), torch.cat([pfeats[i] for i in idx]),
                   torch.cat([pdmids[i] for i in idx]), bid, ask, bb, ba, U), idx

    res = dict(which=which, split=split, arm=ck['arm'], n_windows=nwin, gen_events=GEN)
    # ---------------- B2/B3: free generation vs real continuation
    with torch.no_grad():
        g, _ = batch(1)
        gen = []; mids = [g.mid()]; feats = []
        gu = torch.Generator().manual_seed(1)
        for t in range(GEN):
            feats.append(g.feat.copy())
            gen.append(g.step(torch.rand(nwin, 5, generator=gu)))
            mids.append(g.mid())
        gen = np.stack(gen, 1); mids = np.stack(mids, 1); feats = np.stack(feats, 1)
    comps, floor = [], []
    for w, (i, s) in enumerate(picks):
        r = rows[i]
        real = stats_from(r['tgt'][s:s + GEN], r['feat'][s:s + GEN], np.cumsum(r['dmid'][s:s + GEN]))
        real2 = stats_from(r['tgt'][s + GEN:s + 2 * GEN], r['feat'][s + GEN:s + 2 * GEN],
                           np.cumsum(r['dmid'][s + GEN:s + 2 * GEN]))
        gs = stats_from(gen[w], feats[w], mids[w, 1:])
        comps.append(compare(gs, real)); floor.append(compare(real2, real))
    def agg(cs):
        o = {}
        for k in cs[0]:
            v = np.array([c[k] for c in cs], float)
            o[k] = np.nanmedian(v, 0).tolist()
        return o
    res['B2_model_vs_real'] = agg(comps); res['B2_noise_floor_real_vs_real'] = agg(floor)
    print('B2 model', res['B2_model_vs_real'], '\nB2 floor', res['B2_noise_floor_real_vs_real'], flush=True)
    # ---------------- B5: directional impact of tagged own metaorders
    phi = float(np.median([np.mean(rows[i]['tgt'][s:s + GEN, 0] == 2) for i, s in picks]))
    gap = int(max(1, min(400, round(9 / phi))))
    lots = int(np.median(books[:, 0] + books[:, 10]) // 4) or 1
    B5 = {}
    for name, nch, nsil in (('R10', 10, 40), ('R100', 100, 40)):
        rep = 4
        out_side = {}
        with torch.no_grad():
            for side in (0, 1):
                arms = {}
                for arm in ('vis', 'free'):
                    g, idx = batch(rep)
                    gu = torch.Generator().manual_seed(7)
                    m0 = g.mid(); traj = []
                    for blk in range(nch + nsil):
                        if blk < nch and arm == 'vis':
                            g.inject(np.full(g.B, side), lots, np.ones(g.B, bool), np.ones(g.B, bool))
                        for _ in range(gap):
                            g.step(torch.rand(nwin, 5, generator=gu)[torch.tensor(idx)])
                        traj.append(g.mid() - m0)
                    arms[arm] = np.stack(traj, 1)
                out_side[side] = arms['vis'] - arms['free']
        D = 0.5 * (out_side[0] - out_side[1])            # directional impact, coarse ticks
        price = np.median(books[:, 20] * 0 + 1)          # placeholder unit: report in coarse ticks
        curve = D.mean(0)
        peak = curve[nch - 1]; end = curve[-5:].mean()
        q = np.arange(1, nch + 1)
        pos = curve[:nch] > 0
        delta = float(np.polyfit(np.log(q[pos]), np.log(curve[:nch][pos]), 1)[0]) if pos.sum() > 3 else float('nan')
        B5[name] = dict(peak_ticks=float(peak), end_over_peak=float(end / peak) if peak != 0 else float('nan'),
                        delta_buildup=delta, children=nch, child_lots=lots, gap=gap, curve=curve.tolist())
        print(name, {k: v for k, v in B5[name].items() if k != 'curve'}, flush=True)
    res['B5'] = B5
    res['seconds'] = time.time() - t0
    json.dump(res, open(out, 'x'), indent=1, default=float)


if __name__ == '__main__':
    main()
