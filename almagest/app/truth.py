"""Truth mode: the ground-truth World runs the market; the model shadows it (teacher-forced, live surprise).
What-if forks BOTH from the same state: the model's prediction vs the World's exact counterfactual (paired seeds).
This is the honest demo: every model answer is shown next to the true answer."""
import time
import numpy as np
import torch
from almagest.sim import world as W
from almagest.models import lmm as L
from almagest.app.live import LiveModel, _sv


class TruthLive(LiveModel):
    def __init__(self, ckpt_path, context_seed=7_000_001, context_len=20480):
        super().__init__(ckpt_path, context_seed=context_seed, context_len=context_len)
        n = context_len
        self.wb = np.zeros(W.P, np.int64); self.wa = np.zeros(W.P, np.int64)
        self.wema = np.zeros(W.NSTATE); ns = int(self.par[8]) + 1
        self.wss = np.zeros(ns, np.int64); self.wsr = np.zeros(ns, np.int64)
        self.wbb, self.wba = W.warm_state(self.par, context_seed, 20000 + n, self.wb, self.wa, self.wema,
                                          self.wss, self.wsr)
        assert np.array_equal(self.wb, self.rows.rbid[0]) and np.array_equal(self.wa, self.rows.rask[0])
        self.wseed = context_seed * 7 + 1
        self.mode = 'truth'

    def _sync_rows(self):
        r = self.rows
        r.bid[0] = self.wb; r.ask[0] = self.wa; r.rbid[0] = self.wb; r.rask[0] = self.wa
        r.bb[0] = r.rbb[0] = self.wbb; r.ba[0] = r.rba[0] = self.wba
        self._observe(r)

    @torch.no_grad()
    def step(self, n=1, seed=None):
        """World generates n events; the model consumes them as ONE teacher-forced sequence (same math as training)
        and scores each true event (surprise)."""
        t0 = time.perf_counter()
        ev = np.zeros((n, 4), np.int64); fa = np.zeros((n, 21)); md = np.zeros(n); cm = np.zeros((n, 2, W.D), np.int8)
        self.wseed += 1
        self.wbb, self.wba = W.world_run_events(self.par, self.wb, self.wa, self.wbb, self.wba, self.wema, self.wss,
                                                self.wsr, self.wseed, n, ev, fa, md, cm)
        src = bool(self.model.src)
        evt = torch.tensor(ev)
        if src:
            evt = torch.cat([evt, torch.zeros(n, 1, dtype=evt.dtype)], 1)
        prev = torch.cat([self.pend_ev, evt[:-1]], 0)
        feat = torch.cat([self.pend_feat, torch.tensor(fa[:-1], dtype=torch.float32)], 0)
        mids = np.concatenate([[self.mid_hist[-1]], md])
        dm = torch.tensor(np.diff(mids), dtype=torch.float32)          # dmid of each event (after - before)
        dmid = torch.cat([self.pend_dmid, dm[:-1]], 0)
        sbf = []
        for t in range(n):                                             # slow bank BEFORE consuming input t
            sbf.append(self.sbank.feat())
            self.sbank.update(_sv(evt[t:t + 1]), dm[t:t + 1])
        sbf = torch.cat(sbf, 0)
        if self.reset:
            self.state = self.model.init_state(1) if self.since[0] % self.reset == 0 else self.state
        x = self.model.embed(prev[None], feat[None], dmid[None], sbf[None])
        z, self.state = self.model.trunk_seq(x, self.state)
        self.since = self.since + n
        lt, ls, ll, lz = self.model.heads_logits(z, evt[None, :, 0], evt[None, :, 1])
        ll = L.mask_lvl(ll, evt[None, :, 0], evt[None, :, 1], torch.tensor(cm[None]))
        F = torch.nn.functional
        nll = F.cross_entropy(lt[0], evt[:, 0], reduction='none') + F.cross_entropy(ls[0], evt[:, 1], reduction='none')
        nll = nll + torch.where(evt[:, 0] != 2, F.cross_entropy(ll[0], evt[:, 2], reduction='none'), torch.zeros(n))
        nll = nll + torch.where(evt[:, 0] == 2, F.cross_entropy(lz[0], (evt[:, 3] - 1).clamp(0, 3), reduction='none'),
                                torch.zeros(n))
        self.pend_ev, self.pend_dmid = evt[-1:], dm[-1:]
        self.pend_feat = torch.tensor(fa[-1:], dtype=torch.float32)
        el = (time.perf_counter() - t0) / n
        self.lat.extend([el] * min(n, 50))
        self.surprise.extend(nll.tolist())
        self.mid_hist.extend(md.tolist())
        for t in np.nonzero(ev[:, 0] == 2)[0]:
            self.trade_hist.append([self.t + int(t), 1 if ev[t, 1] == 0 else -1, int(ev[t, 3])])
        self.t += n
        self._sync_rows()
        self.mid_hist = self.mid_hist[-5000:]; self.surprise = self.surprise[-5000:]; self.lat = self.lat[-2000:]
        self.trade_hist = self.trade_hist[-500:]
        return ev.tolist()

    def submit_market(self, side, size):
        self._sync_rows()
        fill = super().submit_market(side, size)          # executes on rows + model sees it (src=1, own-order channel)
        self.wb[:] = self.rows.rbid[0]; self.wa[:] = self.rows.rask[0]
        self.wbb, self.wba = int(self.rows.rbb[0]), int(self.rows.rba[0])
        s = 0 if side in ('buy', 0) else 1
        self.wema[W.IM] += fill['qty'] if s == 0 else -fill['qty']   # the World's traders see the trade
        return fill

    def whatif(self, side='buy', children=20, child_size=2, gap=40, tail_blocks=20, paths=64, seed=0):
        """Directional impact = (I_buy - I_sell)/2 (protocol definition; cancels direction-independent effects) for
        BOTH the model and the exact World counterfactual; raw effect of the chosen side also reported."""
        self._sync_rows()
        opp = 'sell' if side in ('buy', 0) else 'buy'
        mb = LiveModel.whatif(self, side=side, children=children, child_size=child_size, gap=gap,
                              tail_blocks=tail_blocks, paths=paths, seed=seed)
        self._sync_rows()
        ms = LiveModel.whatif(self, side=opp, children=children, child_size=child_size, gap=gap,
                              tail_blocks=tail_blocks, paths=paths, seed=seed)
        out = _antisym(mb, ms, children)
        s = 0 if side in ('buy', 0) else 1
        nblk = children + tail_blocks
        raw = {0: np.zeros((paths, nblk + 1)), 1: np.zeros((paths, nblk + 1))}
        t0 = time.perf_counter()
        for p in range(paths):
            for sd in (0, 1):
                mids = {}
                for mode in (1, 0):
                    om = np.zeros(nblk)
                    m0, _, _ = W.run_from(self.par, self.wb.copy(), self.wa.copy(), self.wbb, self.wba,
                                          self.wema.copy(), self.wss.copy(), self.wsr.copy(),
                                          90_000_000 + seed * 1000 + p, children, gap, tail_blocks, sd, child_size,
                                          mode, om)
                    mids[mode] = np.concatenate([[0.0], om - m0])
                raw[sd][p] = mids[1] - mids[0]
        sgn = 1 if s == 0 else -1
        d = 0.5 * (raw[0] - raw[1])          # (I_buy - I_sell)/2: directional impact, same for either chosen side
        mu = d.mean(0); se = d.std(0, ddof=1) / np.sqrt(paths)
        out['truth'] = dict(mean=mu.tolist(), lo=(mu - 1.96 * se).tolist(), hi=(mu + 1.96 * se).tolist(),
                            peak=float(mu[children]), end=float(mu[-1]))
        rs = raw[s] * sgn
        out['truth_raw'] = dict(mean=rs.mean(0).tolist(), peak=float(rs.mean(0)[children]), end=float(rs.mean(0)[-1]))
        out['truth_seconds'] = time.perf_counter() - t0
        return out

    def status(self):
        st = super().status()
        st['truth_mode'] = True
        return st


def _antisym(mb, ms, children):
    """Directional impact D = (I_buy - I_sell)/2 from forks each signed by its own side: D = (mb + ms)/2."""
    out = dict(mb)
    for k in ('visible', 'blind'):
        a = np.array(mb[k]['mean']); b = np.array(ms[k]['mean'])
        sa = (np.array(mb[k]['hi']) - a) / 1.96; sb = (np.array(ms[k]['hi']) - b) / 1.96
        mu = 0.5 * (a + b); se = 0.5 * np.sqrt(sa ** 2 + sb ** 2)   # both forks already signed by their own side
        out[k] = dict(mean=mu.tolist(), lo=(mu - 1.96 * se).tolist(), hi=(mu + 1.96 * se).tolist())
    out['peak'] = float(out['visible']['mean'][children]); out['end'] = float(out['visible']['mean'][-1])
    out['raw_visible'] = mb['visible']
    out['impact_definition'] = 'antisymmetric (I_side - I_opposite)/2, paired common random numbers'
    return out
