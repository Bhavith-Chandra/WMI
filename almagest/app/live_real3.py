"""LiveReal3: read-only live Coinbase feed -> v3 (replay-exact) Almagest model.

* The SAME ExactTokenizer object that built the training cache tokenises the live feed (no train/serve skew); its
  engine book IS the model's book (bit-exact features), so what-ifs fork from the exact full live book.
* Every real event is scored before the state update -> per-event surprise (NLL) drift monitor.
* What-if: paired paths (visible = the market sees our child orders / blind = it does not / free = no orders);
  directional impact in bp vs the free path, physical time from the model's Dt head. Never routes orders.
"""
import os, threading, time, queue
import numpy as np
import torch
from almagest.data import coinbase_exact as X, coinbase_tokenize as T
from almagest.models import lmm as L
from almagest.experiments import r3
from almagest.app.live_real import Feed, DT_MID

torch.set_num_threads(1)


class LiveReal3:
    def __init__(self, ckpt_path, product='BTC-USD'):
        ck = torch.load(ckpt_path, weights_only=False)
        assert ck.get('version') in ('v3-real-exact', 'v3.1-real-exact'), 'LiveReal3 needs a v3 checkpoint'
        self.model = r3.arm_model(ck['cfg']); self.model.load_state_dict(ck['state']); self.model.eval()
        self.tab = ck['lot_table']
        self.arm, self.n_params, self.ckpt, self.dev_only = 'v3', self.model.n_params(), ckpt_path, True
        self.product = product
        self.q = queue.Queue()
        self.feed = Feed(product, self.q); self.feed.start()
        self.lock = threading.RLock()
        self._new_session()
        threading.Thread(target=self._pump_loop, daemon=True).start()

    def _pump_loop(self):
        while True:
            try:
                self.pump()
            except Exception as e:      # never kill the pump; surface the error in status
                self.err = repr(e)
            time.sleep(0.2)

    def _new_session(self):
        self.tok = X.ExactTokenizer(self.product)
        self.state = self.model.init_state(1)
        self.sb = L.SlowBank(1)
        self.prev = torch.tensor([[0, 0, 1, 0, 0]]); self.prev_lots = 0.0
        self.last_mid = None
        self.n = self.n_mkt = 0
        self.surprise, self.mids, self.times, self.lat = [], [], [], []
        self.err = None

    def pump(self, max_msgs=20000):
        with self.lock:
            return self._pump(max_msgs)

    @torch.no_grad()
    def _pump(self, max_msgs):
        for _ in range(max_msgs):
            try:
                m = self.q.get_nowait()
            except queue.Empty:
                break
            if m.get('type') == '_reset':
                self._new_session(); continue
            self.tok.feed(m)
        toks = self.tok.toks; self.tok.toks = []
        self.tok.hid, self.tok.cks = [], []           # live: no replay archive needed (bounded memory)
        for (ev, lots, f, cm, mid, t, qv) in toks:
            t0 = time.perf_counter()
            dmid = 0.0 if self.last_mid is None else mid - self.last_mid
            self.last_mid = mid
            pv = self.prev
            sv = (1.0 if pv[0, 1] == 0 else -1.0) * self.prev_lots if pv[0, 0] == 2 else 0.0
            self.sb.update(torch.tensor([sv]), torch.tensor([dmid], dtype=torch.float32))
            x = self.model.embed(pv, torch.tensor(f[None], dtype=torch.float32),
                                 torch.tensor([dmid], dtype=torch.float32), self.sb.feat())
            z, self.state = self.model.trunk_step(x, self.state)
            evt = torch.tensor([list(ev)])
            nll = self.model.loss_parts(z[:, None], evt[:, None], torch.tensor(cm[None, None].astype(np.int64)),
                                        torch.tensor([[int(f[20])]])).sum().item()
            self.prev, self.prev_lots = evt, float(lots)
            self.n += 1; self.n_mkt += int(ev[0] == 2)
            self.surprise.append(nll); self.mids.append(float(mid)); self.times.append(float(t))
            self.lat.append(time.perf_counter() - t0)
        for a in ('surprise', 'mids', 'times', 'lat'):
            setattr(self, a, getattr(self, a)[-20000:])
        return len(toks)

    def book(self, depth=10):
        tk = self.tok
        if tk.real is None or not tk._valid():
            return dict(bids=[], asks=[], mid=0.0, spread=0)
        bb, ba = tk.bb, tk.ba
        k = tk.real.k
        return dict(bids=[[int(bb - j + tk.base), int(tk.bid[bb - j])] for j in range(depth)],
                    asks=[[int(ba + j + tk.base), int(tk.ask[ba + j])] for j in range(depth)],
                    mid=float(0.5 * (bb + ba) + tk.base), spread=int(ba - bb), kappa_usd=k, lot=tk.q0,
                    price_usd=float((0.5 * (bb + ba) + tk.base) * k))

    def status(self):
        lat = np.array(self.lat[-2000:]) if self.lat else np.array([np.nan])
        s = np.array(self.surprise[-2000:]) if self.surprise else np.array([np.nan])
        rate = (len(self.times[-2000:]) / max(1e-9, self.times[-1] - self.times[-2000:][0])) if len(self.times) > 10 else 0
        return dict(t=self.n, arm=self.arm, params=self.n_params, ckpt=os.path.basename(self.ckpt), feed=self.feed.status,
                    product=self.product, dev_only=self.dev_only, error=self.err,
                    lat_p50_ms=float(np.nanpercentile(lat, 50) * 1e3), lat_p99_ms=float(np.nanpercentile(lat, 99) * 1e3),
                    surprise_mean=float(np.nanmean(s)), events_per_s=float(rate), phi=float(self.n_mkt / max(1, self.n)))

    def whatif(self, **kw):
        with self.lock:
            return self._whatif(**kw)

    @torch.no_grad()
    def _whatif(self, side='buy', children=20, child_lots=8, participation=0.1, tail_blocks=20, paths=48, seed=0):
        tk = self.tok
        assert tk.real is not None and tk._valid() and self.n > 1000, 'need live context first'
        side = 0 if side in ('buy', 0) else 1
        phi = max(1e-3, self.n_mkt / max(1, self.n))
        gap = int(max(1, min(400, round((1 / participation - 1) / phi))))
        A = 3; B = paths * A
        st = [h.repeat(B, 1) for h in self.state]
        sb = L.SlowBank(B); sb.fv = self.sb.fv.repeat(B, 1); sb.p = self.sb.p.repeat(B, 1); sb.ep = self.sb.ep.repeat(B, 1)
        f, _ = tk._features()
        g = r3.Gen3(self.model, st, sb, self.prev.repeat(B, 1), torch.tensor(f[None], dtype=torch.float32).repeat(B, 1),
                    torch.zeros(B), np.tile(tk.bid, (B, 1)), np.tile(tk.ask, (B, 1)), np.full(B, tk.bb, np.int64),
                    np.full(B, tk.ba, np.int64), self.tab)
        arm = np.tile(np.arange(A), paths); path = np.repeat(np.arange(paths), A)
        gen = torch.Generator().manual_seed(seed)
        nblk = children + tail_blocks
        mids = np.zeros((B, nblk + 1)); secs = np.zeros((B, nblk + 1)); mids[:, 0] = g.mid()
        clock = np.zeros(B)
        t0 = time.perf_counter()
        for blk in range(nblk):
            if blk < children:
                g.inject(np.full(B, side, np.int64), int(child_lots), arm == 0, arm <= 1)
            for _ in range(gap):
                evn, _ = g.step(torch.rand(paths, 6, generator=gen)[torch.tensor(path)])
                clock += DT_MID[evn[:, 4]]
            mids[:, blk + 1] = g.mid(); secs[:, blk + 1] = clock
        sgn = 1 if side == 0 else -1
        px = self.book()['price_usd']
        rel = (mids - mids[:, :1]) * tk.real.k / max(1e-9, px) * 1e4
        v, bl, fr = rel[arm == 0], rel[arm == 1], rel[arm == 2]
        dv, db = sgn * (v - fr), sgn * (bl - fr)

        def band(x):
            mu = x.mean(0); se = x.std(0, ddof=1) / np.sqrt(len(x))
            return dict(mean=mu.tolist(), lo=(mu - 1.96 * se).tolist(), hi=(mu + 1.96 * se).tolist())
        return dict(side='buy' if side == 0 else 'sell', children=children, child_lots=child_lots, lot=tk.q0,
                    gap_events=gap, participation=participation, paths=paths, blocks=nblk, units='bp',
                    visible=band(dv), blind=band(db), seconds=secs[arm == 2].mean(0).tolist(),
                    peak=float(dv[:, children].mean()), end=float(dv[:, -1].mean()),
                    compute_s=time.perf_counter() - t0, dev_only=self.dev_only)
