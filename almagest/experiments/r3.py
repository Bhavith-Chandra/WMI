"""v3 REAL-DATA pipeline on the replay-exact representation (data/coinbase_exact.py).

usage:
  r3.py train <cache_v3> <split.json> <out.pt> [passes]
  r3.py b1    <cache_v3> <split.json> <model.pt> <out.json> <select|report>
  r3.py b2    <cache_v3> <split.json> <model.pt> <out.json> <select|report> [n_windows] [gen_events]

Representation: tokens (typ, side, lvl in 0..DR+KI, size bucket, dt bucket) + exact lots; book features = engine
observe_row3 (top-10, spread, depth bands) — the SAME function in tokenizer and generator; cancel mask width DR.
B2 seeds each window with the EXACT full engine book (checkpoint + replay) and the model state teacher-forced on the
real context; generated lots are drawn from the FIT-hour within-bucket lot distribution (per type, bucket).
Built-in plumbing check (B2 'oracle'): the generator fed the REAL tokens and lots must reproduce the recorded features
until the first hidden update (deep changes outside the action window); its divergence is the representation ceiling.
"""
import sys, os, json, glob, time
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest.models import lmm as L
from almagest.sim import world as W, engine as E
from almagest.data import coinbase_exact as X, coinbase_tokenize as T

torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 4)))
CHUNK = 32768
WIN = 256
NQ = 64
FLOW = os.environ.get('FLOW', '1') == '1'      # v3.3 FlowBank channel (default on)


def arm_model(cfg=None):
    if cfg is not None:
        return L.LMM(**cfg)
    return L.LMM(mode='modebank', slowbank=True, cond_heads=True, nsize=X.NSZ, ndt=T.NDT, size_all=True,
                 nlvl=X.NLVL, nimp=X.KI, nfeat=X.NF, nsb_extra=L.NFB if FLOW else 0,
                 size_cond=os.environ.get('SIZE_COND', '1') == '1',
                 d=int(os.environ.get('D_MODEL', 64)), h=int(os.environ.get('H_MODEL', 128)))


def inputs(ev, lots, feat, mids, cm, qv):
    ev = ev.copy()
    ev[:, 3] = X.size_class(ev[:, 0], lots, qv)          # v3.1 size classes (see coinbase_exact.size_class)
    first = np.array([[0, 0, 1, 0, 0]])
    prev = np.vstack([first, ev[:-1]])
    plots = np.concatenate([[0], lots[:-1]]).astype(float)
    dmid = np.concatenate([[0.0], np.diff(mids)]).astype(np.float32)
    sv = np.where(prev[:, 0] == 2, np.where(prev[:, 1] == 0, 1.0, -1.0) * plots, 0.0)
    fv, dev = L.slowbank_offline(sv, dmid.astype(float))
    sb = L.sb_scale(torch.tensor(fv, dtype=torch.float32), torch.tensor(dev, dtype=torch.float32)).numpy()
    fb = L.flowbank_offline(prev[:, 0] == 2, np.where(prev[:, 1] == 0, 1.0, -1.0), plots)
    sb = np.concatenate([sb, fb], 1)        # flow features always computed; the model uses them iff nsb_extra > 0
    return dict(prev=prev, feat=feat.astype(np.float32), dmid=dmid, sb=sb, tgt=ev, cm=cm,
                spread=feat[:, 20].astype(np.int64), lots=lots, qv=qv,
                lq=np.log1p(qv).astype(np.float32))


def load_rows(cache, hours, with_meta=False):
    """Contiguous runs of tokens whose hour is in `hours` (and complete), cut into CHUNK rows. Recurrent state starts
    at zero at each row start, identical at train and eval."""
    hours = set(hours)
    rows = []
    for f in sorted(glob.glob(os.path.join(cache, '*.npz'))):
        z = np.load(f)
        keep = np.isin(z['hours'], list(hours)) & z['ok']
        if keep.sum() < WIN * 2:
            continue
        idx = np.nonzero(keep)[0]
        for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
            if len(run) < WIN * 2:
                continue
            d = inputs(z['ev'][run], z['lots'][run], z['feat'][run], z['mids'][run], z['cm'][run], z['qv'][run])
            for s in range(0, len(run) - WIN + 1, CHUNK):
                e = min(len(run), s + CHUNK)
                n = ((e - s) // WIN) * WIN
                if n >= WIN:
                    r = {k: v[s:s + n] for k, v in d.items()}
                    if with_meta:
                        r['_file'] = f; r['_gidx'] = run[s:s + n]
                    rows.append(r)
    return rows


TK = ('prev', 'feat', 'dmid', 'sb', 'tgt', 'cm', 'spread', 'lq')


def batches(rows, bsz):
    rows = sorted(rows, key=lambda r: -len(r['tgt']))
    for i in range(0, len(rows), bsz):
        grp = rows[i:i + bsz]
        n = min(len(r['tgt']) for r in grp)
        yield {k: torch.tensor(np.stack([r[k][:n] for r in grp])) for k in TK}


def sbx(model, sb):
    """Slow-bank input for this model (drop FlowBank dims for models without them)."""
    return sb if getattr(model, 'cfg', {}).get('nsb_extra', 0) else sb[..., :2 * len(L.SB_TAUS)]


def run(model, data, opt=None, parts=False):
    B, N = data['tgt'].shape[:2]
    state = model.init_state(B)
    tot, cnt, acc = 0.0, 0, []
    for s in range(0, N - WIN + 1, WIN):
        sl = slice(s, s + WIN)
        x = model.embed(data['prev'][:, sl], data['feat'][:, sl], data['dmid'][:, sl], sbx(model, data['sb'][:, sl]))
        z, state = model.trunk_seq(x, state)
        if parts:
            acc.append(model.loss_parts(z, data['tgt'][:, sl], data['cm'][:, sl], data['spread'][:, sl], data['lq'][:, sl]).detach())
            continue
        loss = model.loss(z, data['tgt'][:, sl], data['cm'][:, sl], data['spread'][:, sl], data['lq'][:, sl])
        if opt is not None:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        state = [h.detach() for h in state]
        tot += loss.item(); cnt += 1
    return torch.cat(acc) if parts else tot / max(cnt, 1)


def lot_table(rows):
    """FIT-only within-class size quantiles. abs (3, NSZ, NQ): limits -> lots; markets -> fraction of the best
    queue (classes < MKT_CLEAR) or lots in excess of it (classes > MKT_CLEAR). frac (NSZ, NQ): cancel fractions."""
    ab = np.ones((3, X.NSZ, NQ)); fr = np.ones((X.NSZ, NQ))
    ev = np.concatenate([r['tgt'] for r in rows]); lots = np.concatenate([r['lots'] for r in rows]).astype(float)
    qv = np.concatenate([r['qv'] for r in rows]).astype(float)
    q = (np.arange(NQ) + 0.5) / NQ
    for b in range(X.NSZ):
        m = (ev[:, 0] == 0) & (ev[:, 3] == b)
        ab[0, b] = np.round(np.quantile(lots[m], q)) if m.sum() >= 20 else max(1, round(2 ** (b / 2 + 0.25)))
        m = (ev[:, 0] == 2) & (ev[:, 3] == b)
        val = np.where(b < X.MKT_CLEAR, lots / np.maximum(qv, 1), lots - qv)
        if m.sum() >= 20:
            ab[2, b] = np.quantile(val[m], q)
        else:
            ab[2, b] = 0.5 if b < X.MKT_CLEAR else max(1, round(2 ** ((b - 14) / 2 + 0.25)))
    for b in range(X.NSZ - 1):
        m = (ev[:, 0] == 1) & (ev[:, 3] == b)
        lo = 0.0 if b == 0 else X.FRAC_EDGES[b - 1]; hi = X.FRAC_EDGES[b] if b < len(X.FRAC_EDGES) else 1.0
        fr[b] = np.quantile(lots[m] / np.maximum(qv[m], 1), q) if m.sum() >= 20 else lo + q * (hi - lo)
    return dict(abs=ab, frac=fr)


def draw_lots(tab, evn, qv, u):
    """Generator size rule. cancels: fraction of the queue they hit (whole queue for FULL); markets: fraction of the
    opposite best queue / exactly that queue / queue + excess; limits: within-class lot quantiles."""
    k = np.minimum((u * NQ).astype(int), NQ - 1)
    q = np.maximum(qv, 1).astype(float)
    v = tab['abs'][evn[:, 0], evn[:, 3], k]
    lim = np.maximum(1, np.round(v))
    mk = np.where(evn[:, 3] < X.MKT_CLEAR, np.clip(np.round(v * q), 1, np.maximum(q - 1, 1)),
                  np.where(evn[:, 3] == X.MKT_CLEAR, q, q + np.maximum(1, np.round(v))))
    c = np.where(evn[:, 3] == X.NSZ - 1, q,     # partial cancels never empty the level (that is the FULL class)
                 np.clip(np.round(tab['frac'][np.minimum(evn[:, 3], X.NSZ - 1), k] * q), 1, np.maximum(q - 1, 1)))
    return np.where(evn[:, 0] == 1, np.minimum(c, q), np.where(evn[:, 0] == 2, mk, lim)).astype(np.int64)


def train(cache, split, out, passes=4):
    rows = load_rows(cache, split['fit'])
    vrows = load_rows(cache, split['select'])
    print('fit rows', len(rows), 'events', sum(len(r['tgt']) for r in rows), 'select events',
          sum(len(r['tgt']) for r in vrows), flush=True)
    torch.manual_seed(0)
    model = arm_model()
    print('params', model.n_params(), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get('LR', 2e-3)), weight_decay=0.01)
    sched = None
    hist = []
    t0 = time.time()
    for p in range(passes):
        if p == passes - 1:
            for g in opt.param_groups:
                g['lr'] *= 0.3
        model.train()
        trl = np.mean([run(model, b, opt) for b in batches(rows, 16)])
        model.eval()
        with torch.no_grad():
            val = np.mean([run(model, b) for b in batches(vrows, 16)]) if vrows else float('nan')
        hist.append((float(trl), float(val)))
        print(f'pass {p} fit {trl:.4f} select {val:.4f} t {time.time()-t0:.0f}s', flush=True)
    torch.save(dict(state=model.state_dict(), cfg=model.cfg, hist=hist, split=split, version='v3.4-real-exact',
                    lot_table=lot_table(rows), DR=X.DR, KI=X.KI, NF=X.NF), out)


# ------------------------------------------------------------------ B1
def b1(cache, split, mp, out, which):
    from almagest.experiments.e3_real import TableModel
    assert which in ('select', 'report') and not os.path.exists(out)
    ck = torch.load(mp, weights_only=False)
    model = arm_model(ck['cfg']); model.load_state_dict(ck['state']); model.eval()
    rows = load_rows(cache, split[which]); frows = load_rows(cache, split['fit'])
    with torch.no_grad():
        P = torch.cat([run(model, b, parts=True) for b in batches(rows, 16)]).numpy()
    base = {}
    for name, cond in (('marginal', False), ('table', True)):
        tm = TableModel(cond)
        tm.nlvl = X.NLVL
        T.NSB = X.NSZ  # table baseline over the same size classes
        tm.fit(frows)
        base[name] = np.concatenate([tm.nll(r).sum(1) for r in rows])
    m = P.sum(1)
    n = min(len(m), len(base['table']))
    res = dict(which=which, n=int(len(m)), model=float(m.mean()), parts=P.mean(0).tolist(),
               **{k: float(v.mean()) for k, v in base.items()})
    # block bootstrap of the difference (blocks of 4096 events)
    d = m[:n] - base['table'][:n]
    nb = n // 4096
    bm = d[:nb * 4096].reshape(nb, 4096).mean(1)
    rng = np.random.default_rng(0)
    bs = [bm[rng.integers(0, nb, nb)].mean() for _ in range(2000)]
    res['model_minus_table'] = float(d.mean()); res['ci95'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    print(json.dumps(res), flush=True)
    json.dump(res, open(out, 'x'), indent=1)


# ------------------------------------------------------------------ generator
class Gen3:
    """Batched generator on the v3 representation. VIEW/REAL books as in the synthetic engine (B5-ready)."""
    def __init__(self, model, state, sb, prev, pfeat, pdmid, bid, ask, bb, ba, lot_tab, fb=None):
        self.m, self.state, self.sb, self.prev, self.pfeat, self.pdmid = model, state, sb, prev, pfeat, pdmid
        self.fb = fb if fb is not None else L.FlowBank(len(bb))
        self.use_fb = bool(model.cfg.get('nsb_extra', 0))
        self.vb, self.va, self.rb, self.ra = bid, ask, bid.copy(), ask.copy()
        self.vbb, self.vba, self.rbb, self.rba = bb, ba, bb.copy(), ba.copy()
        self.B = len(bb); self.tab = lot_tab
        self.feat = np.zeros((self.B, X.NF)); self.cmb = np.zeros((self.B, X.DR), np.int64)
        self.cma = np.zeros_like(self.cmb)
        E.observe3(self.vb, self.va, self.vbb, self.vba, self.feat, self.cmb, self.cma)

    def _consume(self, mask):
        sbf = torch.cat([self.sb.feat(), self.fb.feat()], -1) if self.use_fb else self.sb.feat()
        x = self.m.embed(self.prev, self.pfeat, self.pdmid, sbf)
        z, ns = self.m.trunk_step(x, self.state)
        mm = torch.tensor(mask).unsqueeze(-1)
        self.state = [torch.where(mm, n, o) for n, o in zip(ns, self.state)]
        return z

    def _pend(self, mask, ev, lots, old):
        E.observe3(self.vb, self.va, self.vbb, self.vba, self.feat, self.cmb, self.cma)
        d = torch.tensor(0.5 * (self.vbb + self.vba) - old, dtype=torch.float32)
        mm = torch.tensor(mask)
        self.prev = torch.where(mm.unsqueeze(-1), ev, self.prev)
        self.pfeat = torch.where(mm.unsqueeze(-1), torch.tensor(self.feat, dtype=torch.float32), self.pfeat)
        self.pdmid = torch.where(mm, d, self.pdmid)
        lt = torch.tensor(lots, dtype=torch.float32)
        sv = torch.where(ev[:, 0] == 2, torch.where(ev[:, 1] == 0, 1.0, -1.0) * lt, torch.zeros(self.B))
        f0, p0, e0 = self.sb.fv.clone(), self.sb.p.clone(), self.sb.ep.clone()
        fb0 = self.fb.clone()
        self.fb.update(ev[:, 0] == 2, torch.where(ev[:, 1] == 0, 1.0, -1.0), lt)
        self.fb.select(mm, fb0)
        self.sb.update(sv, d, None)
        k = mm.unsqueeze(-1)
        self.sb.fv = torch.where(k, self.sb.fv, f0); self.sb.p = torch.where(k, self.sb.p, p0)
        self.sb.ep = torch.where(k, self.sb.ep, e0)

    def apply(self, evn, lots):
        old = 0.5 * (self.vbb + self.vba).astype(float)
        E.step_events(self.vb, self.va, self.rb, self.ra, self.vbb, self.vba, self.rbb, self.rba,
                      evn[:, 0].copy(), evn[:, 1].copy(), X.eng_lvl(evn[:, 0], evn[:, 2]).astype(np.int64),
                      lots.astype(np.int64), np.ones(self.B, np.int64))
        self._pend(np.ones(self.B, bool), torch.tensor(evn), lots, old)

    def sample(self, u):
        m = self.m
        z = self._consume(np.ones(self.B, bool))
        typ = L.inv_cdf(m.gen_typ_logits(z), u[:, 0])
        sd = L.inv_cdf(m.gen_side_logits(z, typ), u[:, 1])
        ll, lz = m.lvl_size_logits(z, typ, sd)
        cm = torch.stack([torch.tensor(self.cmb), torch.tensor(self.cma)], 1)
        sp = torch.tensor(self.feat[:, 20], dtype=torch.long)
        lvl = L.inv_cdf(L.mask_lvl(ll, typ, sd, cm, sp, m.nimp), u[:, 2])
        lvl = torch.where(typ == 2, torch.zeros_like(lvl), lvl)
        qv = self.queue(torch.stack([typ, sd, lvl, lvl, lvl], 1).numpy().astype(np.int64))
        if m.size_cond:
            lz = m.size_logits(z, typ, sd, lvl, torch.tensor(np.log1p(qv), dtype=torch.float32))
        szb = L.inv_cdf(lz, u[:, 3])
        zl = z + m.c_side(typ) + m.c_lvl(typ * 2 + sd)
        dtb = L.inv_cdf(m.h_dt(zl + m.c_size(szb)), u[:, 4])
        evn = torch.stack([typ, sd, lvl, szb, dtb], 1).numpy().astype(np.int64)
        return evn, draw_lots(self.tab, evn, qv, u[:, 5].numpy())

    def queue(self, evn):
        """Queue volume each event would hit on the VIEW book (cancel: its level; market: opposite best)."""
        r = np.arange(self.B)
        el = X.eng_lvl(evn[:, 0], evn[:, 2])
        qb = self.vb[r, np.clip(self.vbb - el, 0, W.P - 1)]; qa = self.va[r, np.clip(self.vba + el, 0, W.P - 1)]
        qc = np.where(evn[:, 1] == 0, qb, qa)
        qm = np.where(evn[:, 1] == 0, self.va[r, self.vba], self.vb[r, self.vbb])
        return np.where(evn[:, 0] == 1, qc, np.where(evn[:, 0] == 2, qm, 0))

    def step(self, u):
        evn, lots = self.sample(u)
        self.apply(evn, lots)
        return evn, lots

    def teacher(self, evn, lots):
        """Consume + apply GIVEN events (oracle plumbing check / context continuation)."""
        self._consume(np.ones(self.B, bool))
        self.apply(evn, lots)

    def inject(self, side, lots, vis, inj):
        """Our market order (child) of `lots`: hits REAL where inj, and VIEW (model sees it) where vis."""
        side = np.broadcast_to(np.asarray(side, np.int64), (self.B,)).copy()
        z0 = np.zeros(self.B, np.int64)
        probe = np.stack([np.full(self.B, 2), side, z0, z0, z0], 1)
        cls = X.size_class(probe[:, 0], np.full(self.B, lots), self.queue(probe))   # class w.r.t. the PRE-trade queue
        old = 0.5 * (self.vbb + self.vba).astype(float)
        if vis.any():
            self._consume(vis)
        E.inject(self.vb, self.va, self.rb, self.ra, self.vbb, self.vba, self.rbb, self.rba,
                 side, int(lots), vis.astype(np.int64), inj.astype(np.int64))
        probe[:, 3] = cls
        self._pend(vis, torch.as_tensor(probe), np.full(self.B, float(lots)), old)

    def mid(self):
        return 0.5 * (self.rbb + self.rba).astype(float)


def setup(model, rows, picks, tab):
    zs = {}
    states, sbs, prevs, pfeats, pdmids, books, fbs = [], [], [], [], [], [], []
    from scipy.signal import lfilter
    lam = np.exp(-1.0 / L.SB_TAUS)
    with torch.no_grad():
        for i, s in picks:
            r = rows[i]
            d = {k: torch.tensor(r[k][None, :s]) for k in ('prev', 'feat', 'dmid', 'sb')}
            st = model.init_state(1)
            for s0 in range(0, s, WIN):
                sl = slice(s0, s0 + WIN)
                _, st = model.trunk_seq(model.embed(d['prev'][:, sl], d['feat'][:, sl], d['dmid'][:, sl],
                                                    sbx(model, d['sb'][:, sl])), st)
            states.append(st)
            fbs.append(torch.tensor(r['sb'][s, 2 * len(L.SB_TAUS):][None]))
            sb = L.SlowBank(1)
            pr = r['prev'][:s + 1]; pl = np.concatenate([[0], r['lots'][:s]]).astype(float)
            sv = np.where(pr[:, 0] == 2, np.where(pr[:, 1] == 0, 1, -1) * pl, 0.0)
            fv, _ = L.slowbank_offline(sv, r['dmid'][:s + 1].astype(float))
            pc = np.cumsum(r['dmid'][:s + 1].astype(float))
            sb.fv[0] = torch.tensor(fv[s], dtype=torch.float32); sb.p[0] = float(pc[s])
            sb.ep[0] = torch.tensor(np.stack([lfilter([1 - l], [1, -l], pc) for l in lam], -1)[s], dtype=torch.float32)
            sbs.append(sb); prevs.append(torch.tensor(r['prev'][s][None])); pfeats.append(torch.tensor(r['feat'][s][None]))
            pdmids.append(torch.tensor([float(r['dmid'][s])]))
            f = r['_file']
            if f not in zs:     # only what exact seeding needs (features/masks are multi-GB across segments)
                with np.load(f) as zf:
                    zs[f] = {k: zf[k] for k in ('ev', 'lots', 'hid') + tuple(k for k in zf.files if k.startswith('ck_'))}
            books.append(X.book_at(zs[f], int(r['_gidx'][s])))

    def batch(rep=1, sel=None):
        idx = np.repeat(np.arange(len(picks)) if sel is None else np.asarray(sel), rep)
        st = [torch.cat([states[i][li] for i in idx]) for li in range(len(model.blocks))]
        sb = L.SlowBank(len(idx))
        sb.fv = torch.cat([sbs[i].fv for i in idx]); sb.p = torch.cat([sbs[i].p for i in idx])
        sb.ep = torch.cat([sbs[i].ep for i in idx])
        bid = np.stack([books[i][0] for i in idx]); ask = np.stack([books[i][1] for i in idx])
        bb = np.array([books[i][2] for i in idx], np.int64); ba = np.array([books[i][3] for i in idx], np.int64)
        fb = L.FlowBank(len(idx)); F = torch.cat([fbs[i] for i in idx]).float()
        k = len(L.FB_MTAUS)
        fb.s = F[:, :k] / 2.0; fb.v = F[:, k:] / (torch.tensor(np.sqrt(L.SB_TAUS), dtype=torch.float32) / 4.0)
        return Gen3(model, st, sb, torch.cat([prevs[i] for i in idx]), torch.cat([pfeats[i] for i in idx]),
                    torch.cat([pdmids[i] for i in idx]), bid, ask, bb, ba, tab, fb), idx
    return batch, zs


def pick_windows(rows, nwin, gen, seed=0):
    rng = np.random.default_rng(seed)
    ok = [i for i, r in enumerate(rows) if len(r['tgt']) >= 4 * WIN + 2 * gen]
    picks = []
    for _ in range(nwin):
        i = int(ok[rng.integers(len(ok))]); r = rows[i]
        s = int(rng.integers(2, (len(r['tgt']) - 2 * gen) // WIN)) * WIN
        picks.append((i, s))
    return picks


def stats_from(ev, feat, mid):
    mk = ev[:, 0] == 2
    s = np.where(ev[mk, 1] == 0, 1.0, -1.0)
    acf = [float(np.mean(s[:-l] * s[l:])) if len(s) > l + 5 else np.nan for l in (1, 2, 5, 10, 20)]
    var = []
    for h in (10, 100, 1000):
        dm = mid[h:] - mid[:-h]
        var.append(float(np.var(dm)) if len(dm) > 5 else np.nan)
    dep = feat[:, 0] + feat[:, 10]
    band = feat[:, 21:24].sum(1) + feat[:, 24:27].sum(1)
    return dict(spread=np.bincount(np.minimum(feat[:, 20].astype(int), 8), minlength=9) / len(feat),
                depth=np.log1p(dep), band=np.log1p(band),
                imb=(feat[:, 0] - feat[:, 10]) / (dep + 1e-9), mshare=float(mk.mean()),
                lvl=np.bincount(np.minimum(ev[~mk, 2], X.NLVL - 1), minlength=X.NLVL) / max((~mk).sum(), 1),
                dt=np.bincount(ev[:, 4], minlength=T.NDT) / len(ev), acf=acf, var=var,
                lim_share=float((ev[:, 0] == 0).mean()))


def wass(a, b):
    q = np.linspace(0, 1, 200)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def compare(g, r):
    return dict(spread_L1=float(np.abs(g['spread'] - r['spread']).sum()), depth_W1=wass(g['depth'], r['depth']),
                band_W1=wass(g['band'], r['band']), imb_W1=wass(g['imb'], r['imb']),
                mshare_ratio=g['mshare'] / max(r['mshare'], 1e-9), lim_share=(g['lim_share'], r['lim_share']),
                lvl_L1=float(np.abs(g['lvl'] - r['lvl']).sum()), dt_L1=float(np.abs(g['dt'] - r['dt']).sum()),
                acf1=(g['acf'][0], r['acf'][0]), acf10=(g['acf'][3], r['acf'][3]),
                vol10_ratio=g['var'][0] / max(r['var'][0], 1e-12),
                vol100_ratio=g['var'][1] / max(r['var'][1], 1e-12), vol1000_ratio=g['var'][2] / max(r['var'][2], 1e-12))


def agg(cs):
    out = {}
    for k in cs[0]:
        a = np.array([c[k] for c in cs], float)
        out[k] = np.nanmedian(a, 0).tolist()
    return out


def b2_run(model, rows, picks, batch, gen, seed=1, oracle=False, sampled=False):
    with torch.no_grad():
        g, _ = batch(1)
        evs, lts, feats, mids = [], [], [], [g.mid()]
        gu = torch.Generator().manual_seed(seed)
        for t in range(gen):
            feats.append(g.feat.copy())
            if oracle:
                evn = np.stack([rows[i]['tgt'][s + t] for i, s in picks]); lt = np.array([rows[i]['lots'][s + t] for i, s in picks])
                if sampled:      # real event classes, sizes drawn by the generator's rule (size-representation ceiling)
                    lt = draw_lots(g.tab, evn, g.queue(evn), torch.rand(len(picks), generator=gu).numpy())
                g.teacher(evn, lt); e = evn
            else:
                e, lt = g.step(torch.rand(len(picks), 6, generator=gu))
            evs.append(e); lts.append(lt); mids.append(g.mid())
    return np.stack(evs, 1), np.stack(feats, 1), np.stack(mids, 1)


def b2(cache, split, mp, out, which, nwin=64, gen=4000):
    assert which in ('select', 'report') and not os.path.exists(out)
    ck = torch.load(mp, weights_only=False)
    model = arm_model(ck['cfg']); model.load_state_dict(ck['state']); model.eval()
    rows = load_rows(cache, split[which], with_meta=True)
    picks = pick_windows(rows, nwin, gen)
    t0 = time.time()
    batch, zs = setup(model, rows, picks, ck['lot_table'])
    res = dict(which=which, model=mp, n_windows=nwin, gen_events=gen)
    ge, gf, gm = b2_run(model, rows, picks, batch, gen)
    oe, of, om = b2_run(model, rows, picks, batch, gen, oracle=True)
    se, sf, sm = b2_run(model, rows, picks, batch, gen, oracle=True, sampled=True)
    comps, floor, orc, fid, osm = [], [], [], [], []
    for w, (i, s) in enumerate(picks):
        r = rows[i]
        sl, sl2 = slice(s, s + gen), slice(s + gen, s + 2 * gen)
        real = stats_from(r['tgt'][sl], r['feat'][sl], np.cumsum(r['dmid'][sl]))
        real2 = stats_from(r['tgt'][sl2], r['feat'][sl2], np.cumsum(r['dmid'][sl2]))
        comps.append(compare(stats_from(ge[w], gf[w], gm[w, 1:]), real))
        floor.append(compare(real2, real))
        orc.append(compare(stats_from(oe[w], of[w], om[w, 1:]), real))
        osm.append(compare(stats_from(se[w], sf[w], sm[w, 1:]), real))
        # representation ceiling: log-depth error of oracle replay vs recorded features at horizons
        e = np.abs(np.log1p(of[w][:, :20]) - np.log1p(r['feat'][sl][:, :20])).mean(1)
        fid.append([float(e[h - 1]) for h in (10, 100, 1000, gen)])
    res['model_vs_real'] = agg(comps); res['floor_real_vs_real'] = agg(floor); res['oracle_vs_real'] = agg(orc)
    res['oracle_sampled_sizes_vs_real'] = agg(osm)
    res['oracle_logdepth_err_at_10_100_1000_gen'] = np.median(np.array(fid), 0).tolist()
    res['seconds'] = time.time() - t0
    for k in ('model_vs_real', 'floor_real_vs_real', 'oracle_vs_real', 'oracle_sampled_sizes_vs_real'):
        print(k, json.dumps(res[k]), flush=True)
    print('oracle depth err', res['oracle_logdepth_err_at_10_100_1000_gen'], flush=True)
    json.dump(res, open(out, 'x'), indent=1, default=float)
    return res


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'train':
        train(sys.argv[2], json.load(open(sys.argv[3])), sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 4)
    elif cmd == 'b1':
        b1(sys.argv[2], json.load(open(sys.argv[3])), sys.argv[4], sys.argv[5], sys.argv[6])
    elif cmd == 'b2':
        b2(sys.argv[2], json.load(open(sys.argv[3])), sys.argv[4], sys.argv[5], sys.argv[6],
           int(sys.argv[7]) if len(sys.argv) > 7 else 64, int(sys.argv[8]) if len(sys.argv) > 8 else 4000)
