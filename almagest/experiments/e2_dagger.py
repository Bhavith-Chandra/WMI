"""E2-DAgger: fix closed-loop collapse (exposure bias) with the ground-truth World as oracle (synthetic stage only).

Each round: roll the CURRENT model forward from training-split windows (seeds TRAIN_BASE+), carrying the World's hidden
state along the model's own path; at every visited state record the model's inputs and the World's action there
(oracle label). Fine-tune on original observational streams + all aggregated DAgger trajectories.
No dev/test seeds are touched. usage: e2_dagger.py <base.pt> <out.pt> [rounds] [steps] [rows]
"""
import sys, os, json, time
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest import splits as S
from almagest.sim import world as W, engine as E
from almagest.models import lmm as L
from almagest.experiments.e1_calibrate_world import make
from almagest.experiments.e2_train import build_stream, build_stream_inj, make_model, stack, run_windows, inj_schedule


def run_traj(model, data, h0, T, reset, opt):
    B, N = data['tgt'].shape[:2]
    state = [h.clone() for h in h0]
    tot, cnt = 0.0, 0
    for s in range(0, N - T + 1, T):
        sl = slice(s, s + T)
        x = model.embed(data['prev'][:, sl], data['feat'][:, sl], data['dmid'][:, sl], data['sb'][:, sl])
        z, state = model.trunk_seq(x, state)
        loss = model.loss(z, data['tgt'][:, sl], data['cm'][:, sl])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        state = [h.detach() for h in state]
        tot += loss.item(); cnt += 1
    return tot / max(cnt, 1)

torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 2)))
CH = 256


@torch.no_grad()
def collect(model, reset, par, days, sidx, steps, seed, beta=0.0, inject=False):
    """inject=True: random tagged exogenous metaorders are executed during the rollout (as in evaluation), so the
    oracle labels the market's REACTION in closed loop - where impact errors compound."""
    R = len(days)
    udays = np.unique(days)
    st = {d: build_stream(par, int(d), int(sidx.max()) + 1) for d in udays}
    TAG = bool(getattr(model, 'tagch', False))
    if TAG:
        for d in st:
            if st[d]['sb'].shape[-1] == 2 * len(L.SB_TAUS):
                st[d]['sb'] = np.concatenate([st[d]['sb'], np.zeros((len(st[d]['sb']), len(L.SB_TAUS)), np.float32)], 1)
    rng = np.random.default_rng(seed + 5)
    if inject:
        sch = [inj_schedule(steps, 0.11, rng) for _ in range(R)]
        inj = np.stack([a for a, _, _ in sch]); iside = np.stack([b for _, b, _ in sch]); isize = np.stack([c for _, _, c in sch])
    else:
        inj = np.zeros((R, steps), np.int64); iside = inj.copy(); isize = np.ones_like(inj)
    # teacher-forced recurrent state at window starts (per row)
    state = []
    for i in range(R):
        s = st[days[i]]
        n = int(sidx[i])
        h = model.init_state(1)
        for s0 in range(0, n, CH):
            if reset and s0 % reset == (reset // 2):
                h = model.init_state(1)
            sl = slice(s0, s0 + CH)
            x = model.embed(torch.tensor(s['prev'][None, sl]), torch.tensor(s['feat'][None, sl]),
                            torch.tensor(s['dmid'][None, sl]), torch.tensor(s['sb'][None, sl]))
            _, h = model.trunk_seq(x, h)
        state.append(h)
    state = [torch.cat([h[li] for h in state]) for li in range(len(model.blocks))]
    snaps = W.Snapshots(par, udays, np.unique(20000 + sidx))
    rd, rw = snaps.index(days, 20000 + sidx)
    bid = snaps.sb[rd, rw].copy(); ask = snaps.sa[rd, rw].copy()
    bb = snaps.sbb[rd, rw].copy(); ba = snaps.sba[rd, rw].copy()
    ema = snaps.sema[rd, rw].copy(); ns = int(par[8]) + 1
    ss = snaps.sss[rd, rw, :ns].copy(); sr = snaps.ssr[rd, rw, :ns].copy()
    rb, ra, rbb, rba = bid.copy(), ask.copy(), bb.copy(), ba.copy()   # single book (view == real)
    sbank = L.SlowBank(R, tag=TAG)
    from scipy.signal import lfilter
    lam = np.exp(-1.0 / L.SB_TAUS)
    for i in range(R):
        s = st[days[i]]; j = int(sidx[i])
        pr = s['prev']
        sv = np.where(pr[:j + 1, 0] == 2, np.where(pr[:j + 1, 1] == 0, 1, -1) * pr[:j + 1, 3], 0).astype(float)
        p = np.cumsum(s['dmid'][:j + 1].astype(float))
        fv = np.stack([lfilter([1 - l], [1, -l], sv) for l in lam], -1)
        ep = np.stack([lfilter([1 - l], [1, -l], p) for l in lam], -1)
        sbank.fv[i] = torch.tensor(fv[j]); sbank.p[i] = float(p[j]); sbank.ep[i] = torch.tensor(ep[j])
    SRC = bool(getattr(model, 'src', False))
    pad = (lambda t, v: torch.cat([t, torch.full(t.shape[:-1] + (1,), v, dtype=t.dtype)], -1)) if SRC else (lambda t, v: t)
    pend_ev = pad(torch.tensor(np.stack([st[days[i]]['tgt'][sidx[i] - 1] for i in range(R)])), 0)
    pend_dmid = torch.tensor([float(st[days[i]]['dmid'][sidx[i]]) for i in range(R)], dtype=torch.float32)
    feat = np.zeros((R, 21)); cmb = np.zeros((R, W.D), np.int64); cma = np.zeros_like(cmb)
    E.observe(bid, ask, bb, ba, feat, cmb, cma)
    since = np.full(R, (reset // 2) if reset else 0)
    rec = {k: [] for k in ('prev', 'feat', 'dmid', 'sb', 'tgt', 'cm')}
    init_state = [h.clone() for h in state]
    g = torch.Generator().manual_seed(seed)
    lab = np.zeros((R, 4), np.int64)
    ones = np.ones(R, np.int64)
    for t in range(steps):
        if reset:
            rz = torch.tensor(since % reset == 0).unsqueeze(-1)
            state = [torch.where(rz, torch.zeros_like(h), h) for h in state]
        pf = torch.tensor(feat, dtype=torch.float32)
        sbf = sbank.feat()
        x = model.embed(pend_ev, pf, pend_dmid, sbf)
        z, state = model.trunk_step(x, state)
        since += 1
        cm = np.stack([cmb, cma], 1).astype(np.int8)
        # oracle label at the visited state (World's action); hidden state advanced as the World would
        W.oracle_labels(par, rb, ra, rbb, rba, ema, ss, sr,
                        (seed * 1_000_003 + t * 7919 + np.arange(R) * 104729) % (2 ** 31), lab)
        it = inj[:, t] == 1
        tgt = pad(torch.tensor(lab.copy()), 0)
        if it.any():   # exogenous child is the next event on these rows: it is an input, never a target (src=1)
            child = pad(torch.stack([torch.full((R,), 2), torch.tensor(iside[:, t]), torch.zeros(R, dtype=torch.long),
                                     torch.tensor(isize[:, t])], 1), 1)
            tgt = torch.where(torch.tensor(it).unsqueeze(-1), child, tgt)
        rec['prev'].append(pend_ev.numpy().copy()); rec['feat'].append(pf.numpy().copy())
        rec['dmid'].append(pend_dmid.numpy().copy()); rec['sb'].append(sbf.numpy().copy())
        rec['tgt'].append(tgt.numpy()); rec['cm'].append(cm)
        # model action executes (on-policy states)
        u = torch.rand(R, 4, generator=g)
        typ = L.inv_cdf(model.h_typ(z), u[:, 0])
        side = L.inv_cdf(model.side_logits(z, typ), u[:, 1])
        ll, lz = model.lvl_size_logits(z, typ, side)
        lvl = L.inv_cdf(L.mask_lvl(ll, typ, side, torch.tensor(cm)), u[:, 2])
        size = L.inv_cdf(lz, u[:, 3]) + 1
        lvl = torch.where(typ == 2, torch.zeros_like(lvl), lvl)
        size = torch.where(typ == 2, size, torch.ones_like(size))
        ev = torch.stack([typ, side, lvl, size], 1)
        if beta > 0:   # beta-mixture: oracle executes on a random subset of rows (keeps states recoverable)
            use_o = torch.rand(R, generator=g) < beta
            ev = torch.where(use_o.unsqueeze(-1), torch.tensor(lab.copy()), ev)
        if it.any():   # injected exogenous child replaces the agent event on these rows
            ev = torch.where(torch.tensor(it).unsqueeze(-1),
                             torch.stack([torch.full((R,), 2), torch.tensor(iside[:, t]), torch.zeros(R, dtype=torch.long),
                                          torch.tensor(isize[:, t])], 1), ev)
        evn = ev.numpy().astype(np.int64)
        old = 0.5 * (rbb + rba).astype(float)
        E.step_events(bid, ask, rb, ra, bb, ba, rbb, rba, evn[:, 0].copy(), evn[:, 1].copy(), evn[:, 2].copy(),
                      evn[:, 3].copy(), ones)
        W.credit_trade(par, ema, evn[:, 0].copy(), evn[:, 1].copy(), evn[:, 3].copy(), rbb, rba)
        E.observe(bid, ask, bb, ba, feat, cmb, cma)
        d = torch.tensor(0.5 * (rbb + rba) - old, dtype=torch.float32)
        sv = torch.where(ev[:, 0] == 2, torch.where(ev[:, 1] == 0, 1.0, -1.0) * ev[:, 3].float(), torch.zeros(R))
        itt = torch.tensor(it)
        sbank.update(sv, d, torch.where(itt, sv, torch.zeros(R)) if TAG else None)
        pe = pad(ev, 0)
        if SRC:
            pe[:, -1] = itt.long()
        pend_ev, pend_dmid = pe, d
    out = {k: np.stack(v, 1) for k, v in rec.items()}
    out['_h0'] = init_state
    depth_b = feat[:, :10].sum(1); depth_a = feat[:, 10:20].sum(1)
    collapsed = float(np.mean(np.minimum(depth_b, depth_a) <= 1))
    return out, collapsed


def main():
    base, out = sys.argv[1], sys.argv[2]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    rows = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    ck = torch.load(base, weights_only=False)
    par = make(ck['theta'])
    model, reset = make_model(ck['arm'], cfg=ck.get('cfg'))
    model.load_state_dict(ck['state'])
    if getattr(model, 'tagch', False):
        os.environ['TAG_CHANNEL'] = '1'
    bs = build_stream_inj if ck['arm'].endswith('_src') else build_stream
    orig = stack([bs(par, S.TRAIN_BASE + i, 120_000) for i in range(8)], 1)
    agg = []
    hist = []
    t0 = time.time()
    for r in range(rounds):
        rng = np.random.default_rng(1000 + r)
        days = (S.TRAIN_BASE + 20 + rng.integers(0, 40, rows)).astype(np.int64)   # training-range day seeds only
        sidx = (20480 + 2048 * rng.integers(0, 20, rows)).astype(np.int64)
        model.eval()
        beta = float(os.environ.get('BETA0', 0.5)) * (0.5 ** r)
        traj, coll = collect(model, reset, par, days, sidx, steps, seed=77 + r, beta=beta,
                             inject=bool(getattr(model, 'src', False)))
        h0 = traj.pop('_h0')
        agg.append(({k: torch.tensor(v) for k, v in traj.items()}, h0))
        print(f'round {r}: beta {beta:.2f} collected {rows}x{steps}, end-of-rollout one-sided books {coll:.2f}, {time.time()-t0:.0f}s',
              flush=True)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get('LR', 5e-4)), weight_decay=0.01)
        for ep in range(2):
            l1 = run_windows(model, orig, 256, reset, _Opt(opt))
            l2 = np.mean([run_traj(model, a, h0, 256, reset, opt) for a, h0 in agg])
            print(f'  epoch {ep}: orig {l1:.4f} dagger {l2:.4f} {time.time()-t0:.0f}s', flush=True)
        hist.append(dict(round=r, collapsed=coll, orig=l1, dagger=l2))
    torch.save(dict(state=model.state_dict(), cfg=model.cfg, arm=ck['arm'], theta=ck['theta'], hist=hist,
                    dagger=dict(base=base, rounds=rounds, steps=steps, rows=rows)), out)


class _Opt:
    def __init__(self, o):
        self.o = o

    def zero_grad(self):
        self.o.zero_grad()

    def step(self):
        self.o.step()


if __name__ == '__main__':
    main()
