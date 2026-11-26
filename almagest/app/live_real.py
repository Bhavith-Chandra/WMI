"""LiveReal: read-only live Coinbase market -> v2 Almagest model (teacher-forced on every real event) + what-if.

* Feed: public level2_batch + matches (no auth). Tokens come from the SAME StreamTokenizer used to build training data.
* Every real event is scored by the model before it updates state -> per-event surprise (NLL) = drift monitor.
* What-if forks the live state into paired paths (visible / blind / free) on a coarse-tick engine; physical time comes
  from the model's Dt head (bucket midpoints), so impact paths are indexed in seconds.
* Never routes orders.
"""
import asyncio, json, os, ssl, threading, time, queue
import numpy as np
import torch
from almagest.data import coinbase_tokenize as T
from almagest.models import lmm as L
from almagest.sim import world as W, engine as E
from almagest.experiments.e3_real import arm_model

CA = '/root/.ccr/ca-bundle.crt'
DT_MID = np.concatenate([[0.0], np.sqrt(np.maximum(T.DT_EDGES[1:-1], 1e-6) * T.DT_EDGES[2:]), [T.DT_EDGES[-1] * 2]])
# Dt bucket representative seconds: 0, geometric mid of each finite bucket, 2x last edge for the open bucket


class Feed(threading.Thread):
    def __init__(self, product, q):
        super().__init__(daemon=True)
        self.product, self.q = product, q
        self.status = 'connecting'

    def run(self):
        asyncio.run(self._run())

    async def _run(self):
        import websockets
        ctx = ssl.create_default_context(cafile=CA) if os.path.exists(CA) else ssl.create_default_context()
        while True:
            try:
                async with websockets.connect('wss://ws-feed.exchange.coinbase.com', ssl=ctx, max_size=2 ** 25) as ws:
                    await ws.send(json.dumps({'type': 'subscribe', 'product_ids': [self.product],
                                              'channels': ['level2_batch', 'matches']}))
                    self.status = 'live'
                    async for m in ws:
                        self.q.put(json.loads(m))
            except Exception as e:
                self.status = f'reconnecting ({type(e).__name__})'
                self.q.put({'type': '_reset'})
                await asyncio.sleep(2)


class LiveReal:
    def __init__(self, ckpt_path, product='BTC-USD'):
        ck = torch.load(ckpt_path, weights_only=False)
        assert ck.get('version') == 'v2-real', 'LiveReal needs a v2 real-data checkpoint'
        self.model, self.reset = arm_model(ck['arm'])
        self.model.load_state_dict(ck['state'])
        self.model.eval()
        self.arm, self.n_params, self.ckpt = ck['arm'], self.model.n_params(), ckpt_path
        self.dev_only = ck.get('dev_only', False)
        self.product = product
        self.q = queue.Queue()
        self.feed = Feed(product, self.q)
        self.feed.start()
        self.lock = threading.RLock()
        self._new_session()
        # keep consuming the feed even when no client polls (bounded memory, state always current)
        threading.Thread(target=self._pump_loop, daemon=True).start()

    def _pump_loop(self):
        while True:
            try:
                self.pump()
            except Exception:
                pass
            time.sleep(0.2)

    def _new_session(self):
        self.tok = T.StreamTokenizer(self.product)
        self.state = self.model.init_state(1)
        self.sb = L.SlowBank(1)
        self.prev = torch.tensor([[0, 0, 1, 0, 0]])
        self.last_mid = None
        self.since = 0
        self.n = 0
        self.surprise = []
        self.mids = []
        self.times = []
        self.lat = []
        self.n_mkt = 0

    # ------------------------------------------------------------------ teacher forcing on real events
    def pump(self, max_msgs=20000):
        with self.lock:
            return self._pump(max_msgs)

    @torch.no_grad()
    def _pump(self, max_msgs=20000):
        toks = []
        for _ in range(max_msgs):
            try:
                m = self.q.get_nowait()
            except queue.Empty:
                break
            if m.get('type') == '_reset':
                self._new_session()
                continue
            toks.extend(self.tok.feed(m))
        toks.extend(self.tok.flush())
        for (ev, f, cm, mid, t) in toks:
            t0 = time.perf_counter()
            dmid = 0.0 if self.last_mid is None else mid - self.last_mid
            self.last_mid = mid
            evt = torch.tensor([ev])
            pv = self.prev
            lots = 2.0 ** float(pv[0, 3])
            sv = (1.0 if pv[0, 1] == 0 else -1.0) * lots if pv[0, 0] == 2 else 0.0
            self.sb.update(torch.tensor([sv]), torch.tensor([dmid], dtype=torch.float32))
            if self.reset and self.since % self.reset == 0:
                self.state = self.model.init_state(1)
            x = self.model.embed(pv, torch.tensor(f[None], dtype=torch.float32),
                                 torch.tensor([dmid], dtype=torch.float32), self.sb.feat())
            z, self.state = self.model.trunk_step(x, self.state)
            nll = self.model.loss_parts(z[:, None], evt[:, None], torch.tensor(cm[None, None])).sum().item()
            self.prev = evt
            self.since += 1
            self.n += 1
            self.n_mkt += int(ev[0] == 2)
            self.surprise.append(nll)
            self.mids.append(float(mid))
            self.times.append(float(t))
            self.lat.append(time.perf_counter() - t0)
        for a in ('surprise', 'mids', 'times', 'lat'):
            setattr(self, a, getattr(self, a)[-20000:])
        return len(toks)

    def book(self, depth=10):
        b = self.tok.book
        if b is None:
            return dict(bids=[], asks=[], mid=0.0, spread=0)
        bb, ba = b.best(0), b.best(1)
        bids = [[int(bb - k), b.lots(0, bb - k)] for k in range(depth)]
        asks = [[int(ba + k), b.lots(1, ba + k)] for k in range(depth)]
        return dict(bids=bids, asks=asks, mid=float(0.5 * (bb + ba)), spread=int(ba - bb),
                    kappa_usd=b.k, lot=b.q0, price_usd=float(0.5 * (bb + ba) * b.k))

    def status(self):
        lat = np.array(self.lat[-2000:]) if self.lat else np.array([np.nan])
        s = np.array(self.surprise[-2000:]) if self.surprise else np.array([np.nan])
        rate = (len(self.times[-2000:]) / max(1e-9, self.times[-1] - self.times[-2000:][0])) if len(self.times) > 10 else 0
        return dict(t=self.n, arm=self.arm, params=self.n_params, ckpt=os.path.basename(self.ckpt),
                    feed=self.feed.status, product=self.product, dev_only=self.dev_only,
                    lat_p50_ms=float(np.nanpercentile(lat, 50) * 1e3), lat_p99_ms=float(np.nanpercentile(lat, 99) * 1e3),
                    surprise_mean=float(np.nanmean(s)), events_per_s=float(rate),
                    phi=float(self.n_mkt / max(1, self.n)))

    # ------------------------------------------------------------------ what-if from the live state
    def whatif(self, **kw):
        with self.lock:
            return self._whatif(**kw)

    @torch.no_grad()
    def _whatif(self, side='buy', children=20, child_lots=8, participation=0.1, tail_blocks=20, paths=48, seed=0):
        b = self.tok.book
        assert b is not None and self.n > 1000, 'need live context first'
        side = 0 if side in ('buy', 0) else 1
        phi = max(1e-3, self.n_mkt / max(1, self.n))
        gap = int(max(1, min(400, round((1 / participation - 1) / phi))))
        A = 3
        B = paths * A
        bb, ba = b.best(0), b.best(1)
        base = (bb + ba) // 2 - W.P // 2
        rows_bid = np.zeros((B, W.P), np.int64)
        rows_ask = np.zeros((B, W.P), np.int64)
        for k in range(3 * W.D):
            if 0 <= bb - k - base < W.P:
                rows_bid[:, bb - k - base] = b.lots(0, bb - k)
            if 0 <= ba + k - base < W.P:
                rows_ask[:, ba + k - base] = b.lots(1, ba + k)
        vb, va, rb, ra = rows_bid, rows_ask, rows_bid.copy(), rows_ask.copy()
        vbb = np.full(B, bb - base, np.int64); vba = np.full(B, ba - base, np.int64)
        rbb, rba = vbb.copy(), vba.copy()
        feat = np.zeros((B, 21)); cmb = np.zeros((B, W.D), np.int64); cma = np.zeros_like(cmb)
        E.observe(vb, va, vbb, vba, feat, cmb, cma)
        state = [h.repeat(B, 1) for h in self.state]
        sb = L.SlowBank(B)
        sb.fv = self.sb.fv.repeat(B, 1); sb.p = self.sb.p.repeat(B, 1); sb.ep = self.sb.ep.repeat(B, 1)
        prev = self.prev.repeat(B, 1)
        pfeat = torch.tensor(feat, dtype=torch.float32)
        pdmid = torch.zeros(B)
        arm = np.tile(np.arange(A), paths)
        path = np.repeat(np.arange(paths), A)
        g = torch.Generator().manual_seed(seed)
        m = self.model
        nblk = children + tail_blocks
        mids = np.zeros((B, nblk + 1)); secs = np.zeros((B, nblk + 1))
        mids[:, 0] = 0.5 * (rbb + rba)
        clock = np.zeros(B)
        t0 = time.perf_counter()

        def consume(mask):
            nonlocal state
            x = m.embed(prev, pfeat, pdmid, sb.feat())
            z, ns = m.trunk_step(x, state)
            mm = torch.tensor(mask).unsqueeze(-1)
            state = [torch.where(mm, n, o) for n, o in zip(ns, state)]
            return z

        def set_pending(mask, ev, old):
            nonlocal prev, pfeat, pdmid
            E.observe(vb, va, vbb, vba, feat, cmb, cma)
            d = torch.tensor(0.5 * (vbb + vba) - old, dtype=torch.float32)
            mm = torch.tensor(mask)
            prev = torch.where(mm.unsqueeze(-1), ev, prev)
            pfeat = torch.where(mm.unsqueeze(-1), torch.tensor(feat, dtype=torch.float32), pfeat)
            pdmid = torch.where(mm, d, pdmid)
            lots = torch.pow(2.0, ev[:, 3].float())
            sv = torch.where(ev[:, 0] == 2, torch.where(ev[:, 1] == 0, 1.0, -1.0) * lots, torch.zeros(B))
            f0, p0, e0 = sb.fv.clone(), sb.p.clone(), sb.ep.clone()
            sb.update(sv, d)
            k = mm.unsqueeze(-1)
            sb.fv = torch.where(k, sb.fv, f0); sb.p = torch.where(k, sb.p, p0); sb.ep = torch.where(k, sb.ep, e0)

        csb = T.size_bucket(child_lots)
        for blk in range(nblk):
            if blk < children:
                vis = arm == 0
                inj = arm <= 1
                consume(vis)
                old = 0.5 * (vbb + vba).astype(float)
                E.inject(vb, va, rb, ra, vbb, vba, rbb, rba, np.full(B, side, np.int64), int(child_lots),
                         vis.astype(np.int64), inj.astype(np.int64))
                set_pending(vis, torch.tensor([[2, side, 0, csb, 0]]).repeat(B, 1), old)
            for _ in range(gap):
                z = consume(np.ones(B, bool))
                u = torch.rand(paths, 5, generator=g)[torch.tensor(path)]
                typ = L.inv_cdf(m.h_typ(z), u[:, 0])
                sd = L.inv_cdf(m.side_logits(z, typ), u[:, 1])
                zl = z + m.c_side(typ) + m.c_lvl(typ * 2 + sd)
                cm = torch.stack([torch.tensor(cmb), torch.tensor(cma)], 1)
                ll, lzz = m.lvl_size_logits(z, typ, sd)
                lvl = L.inv_cdf(L.mask_lvl(ll, typ, sd, cm), u[:, 2])
                szb = L.inv_cdf(lzz, u[:, 3])
                dtb = L.inv_cdf(m.h_dt(zl + m.c_size(szb)), u[:, 4])
                lvl = torch.where(typ == 2, torch.zeros_like(lvl), lvl)
                ev = torch.stack([typ, sd, lvl, szb, dtb], 1)
                evn = ev.numpy().astype(np.int64)
                lots = np.power(2, evn[:, 3]).astype(np.int64)
                old = 0.5 * (vbb + vba).astype(float)
                E.step_events(vb, va, rb, ra, vbb, vba, rbb, rba, evn[:, 0].copy(), evn[:, 1].copy(),
                              evn[:, 2].copy(), lots, np.ones(B, np.int64))
                clock += DT_MID[evn[:, 4]]
                set_pending(np.ones(B, bool), ev, old)
            mids[:, blk + 1] = 0.5 * (rbb + rba)
            secs[:, blk + 1] = clock
        sgn = 1 if side == 0 else -1
        rel = (mids - mids[:, :1]) * b.k / max(1e-9, self.book()['price_usd']) * 1e4   # coarse ticks -> bp
        v, bl, f = rel[arm == 0], rel[arm == 1], rel[arm == 2]
        dv, db = sgn * (v - f), sgn * (bl - f)

        def band(x):
            mu = x.mean(0); se = x.std(0, ddof=1) / np.sqrt(len(x))
            return dict(mean=mu.tolist(), lo=(mu - 1.96 * se).tolist(), hi=(mu + 1.96 * se).tolist())
        return dict(side='buy' if side == 0 else 'sell', children=children, child_lots=child_lots,
                    lot=b.q0, gap_events=gap, participation=participation, paths=paths, blocks=nblk, units='bp',
                    visible=band(dv), blind=band(db), seconds=secs[arm == 2].mean(0).tolist(),
                    peak=float(dv[:, children].mean()), end=float(dv[:, -1].mean()),
                    compute_s=time.perf_counter() - t0, dev_only=self.dev_only)
