"""E2: train lightweight generators on World's OBSERVATIONAL stream only (no injections ever seen).
Seeds from almagest/splits.py (train TRAIN_BASE+, val VAL_BASE+). Each training row is ONE contiguous stream from
the (burned-in) day start with recurrent state carried across all windows, matching evaluation, where state is
carried from day start to the window (audit fix #6).

usage: e2_train.py <world_theta.json> <arm> <out.pt> [passes]
arms: short (learned taus, state reset every 512 events = 500-msg-context analogue)
      stateful (learned taus, state carried across windows)
      modebank (fixed log-spaced 2^0..2^17 + learned, carried)
      modebank_sb (modebank + parameter-free slow bank)
"""
import sys, json, time
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest.sim import world as W
from almagest.experiments.e1_calibrate_world import make
from almagest.models import lmm as L
from almagest import splits as S

import os
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 4)))
NTR = int(os.environ.get("NTR", 16))
NTOK = int(os.environ.get("NTOK", 160000))


def build_stream(par, seed, n):
    ev = np.zeros((n, 4), np.int64)
    feat = np.zeros((n, 21), np.float64)
    mids = np.zeros(n)
    cm = np.zeros((n, 2, W.D), np.int8)
    W.gen_stream(par, seed, n, ev, feat, mids, cm)
    prev = np.vstack([[0, 0, 1, 1], ev[:-1]])
    dmid = np.concatenate([[0.0], np.diff(mids)])
    sv = np.where(prev[:, 0] == 2, np.where(prev[:, 1] == 0, 1, -1) * prev[:, 3], 0).astype(float)
    fv, dev = L.slowbank_offline(sv, dmid)
    sb = L.sb_scale(torch.tensor(fv, dtype=torch.float32), torch.tensor(dev, dtype=torch.float32)).numpy()
    return dict(prev=prev, feat=feat.astype(np.float32), dmid=dmid.astype(np.float32), sb=sb,
                tgt=ev, cm=cm)


def inj_schedule(n, phi, rng, busy=0.7):
    """Random exogenous metaorders over a stream of n events: participation log-U[0.02, 0.30], children log-U[10, 150],
    child size in 1..4, random side, idle gaps so ~`busy` of the time a metaorder is active. Deliberately covers
    a wide range around (not just at) the evaluation operating point."""
    inj = np.zeros(n, np.int64); side = np.zeros(n, np.int64); size = np.ones(n, np.int64)
    t = int(rng.integers(500, 5000))
    while t < n:
        eta = float(np.exp(rng.uniform(np.log(0.02), np.log(0.30))))
        N = int(np.exp(rng.uniform(np.log(10), np.log(150))))
        gap = max(1, int(round((1 / eta - 1) / phi)))
        sd, sz = int(rng.integers(2)), int(rng.integers(1, 5))
        for k in range(N):
            tk = t + k * (gap + 1)
            if tk >= n:
                break
            inj[tk], side[tk], size[tk] = 1, sd, sz
        dur = N * (gap + 1)
        t += dur + int(dur * (1 - busy) / busy * rng.uniform(0.5, 1.5)) + 1
    return inj, side, size


def build_stream_inj(par, seed, n, phi=0.11):
    rng = np.random.default_rng(seed + 7)
    inj, iside, isize = inj_schedule(n, phi, rng)
    ev = np.zeros((n, 4), np.int64)
    feat = np.zeros((n, 21), np.float64)
    mids = np.zeros(n)
    cm = np.zeros((n, 2, W.D), np.int8)
    src = np.zeros(n, np.int64)
    W.gen_stream_inj(par, seed, n, inj, iside, isize, ev, feat, mids, cm, src)
    ev5 = np.concatenate([ev, src[:, None]], 1)
    prev = np.vstack([[0, 0, 1, 1, 0], ev5[:-1]])
    dmid = np.concatenate([[0.0], np.diff(mids)])
    sv = np.where(prev[:, 0] == 2, np.where(prev[:, 1] == 0, 1, -1) * prev[:, 3], 0).astype(float)
    fv, dev = L.slowbank_offline(sv, dmid)
    sb = L.sb_scale(torch.tensor(fv, dtype=torch.float32), torch.tensor(dev, dtype=torch.float32)).numpy()
    return dict(prev=prev, feat=feat.astype(np.float32), dmid=dmid.astype(np.float32), sb=sb, tgt=ev5, cm=cm)


def mirror(d):
    """Exact buy/sell mirror of a stream (the World is side-symmetric): flip sides, swap bid/ask book features and
    cancel masks, negate price changes and slow-bank features (flow and price-deviation parts are both odd)."""
    m = {k: v.copy() for k, v in d.items()}
    for k in ('prev', 'tgt'):
        m[k][:, 1] = 1 - m[k][:, 1]
    f = m['feat']
    m['feat'] = np.concatenate([f[:, 10:20], f[:, 0:10], f[:, 20:21]], 1)
    m['cm'] = m['cm'][:, ::-1, :].copy()
    m['dmid'] = -m['dmid']
    m['sb'] = -m['sb']
    return m


def stack(streams, chunks):
    out = {}
    for k in streams[0]:
        arrs = []
        for s in streams:
            a = s[k]
            n = (len(a) // chunks) * chunks
            arrs += np.split(a[:n], chunks)
        out[k] = torch.tensor(np.stack(arrs))
    return out


def make_model(arm):
    if arm == 'short':
        return L.LMM(mode='learned'), 512
    if arm == 'stateful':
        return L.LMM(mode='learned'), None
    if arm == 'modebank':
        return L.LMM(mode='modebank'), None
    if arm == 'modebank_sb':
        return L.LMM(mode='modebank', slowbank=True), None
    if arm == 'modebank_sb_L':     # capacity test: ~4x params
        return L.LMM(d=128, h=256, layers=3, mode='modebank', slowbank=True), None
    if arm == 'short_src':
        return L.LMM(mode='learned', src=True), 512
    if arm == 'modebank_sb_src':
        return L.LMM(mode='modebank', slowbank=True, src=True), None
    raise ValueError(arm)


def run_windows(model, data, T, reset, opt=None):
    B, N = data['tgt'].shape[:2]
    state = model.init_state(B)
    tot, cnt = 0.0, 0
    for s in range(0, N - T + 1, T):
        if reset is not None and s % reset == 0:
            state = model.init_state(B)
        sl = slice(s, s + T)
        x = model.embed(data['prev'][:, sl], data['feat'][:, sl], data['dmid'][:, sl], data['sb'][:, sl])
        z, state = model.trunk_seq(x, state)
        loss = model.loss(z, data['tgt'][:, sl], data['cm'][:, sl])
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        state = [h.detach() for h in state]
        tot += loss.item()
        cnt += 1
    return tot / cnt


if __name__ == '__main__':
    theta = json.load(open(sys.argv[1]))['theta']
    arm, out = sys.argv[2], sys.argv[3]
    passes = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    par = make(theta)
    torch.manual_seed(0)
    t0 = time.time()
    bs = build_stream_inj if arm.endswith('_src') else build_stream
    tr_streams = [bs(par, S.TRAIN_BASE + i, NTOK) for i in range(NTR)]
    if os.environ.get('MIRROR'):           # side-flip augmentation: add the mirrored copy of every stream
        tr_streams = tr_streams + [mirror(x) for x in tr_streams]
    tr = stack(tr_streams, 1)
    va = stack([bs(par, S.VAL_BASE + i, 100_000) for i in range(4)], 1)
    print('data', time.time() - t0, 'train tokens', tr['tgt'].shape[0] * tr['tgt'].shape[1], flush=True)
    model, reset = make_model(arm)
    print(arm, 'params', model.n_params(), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    T = 256
    nsteps = passes * (tr['tgt'].shape[1] // T)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=nsteps + 5, pct_start=0.05)

    class _O:
        def zero_grad(self): opt.zero_grad()
        def step(self): opt.step(); sched.step()
    hist = []
    for p in range(passes):
        model.train()
        trl = run_windows(model, tr, T, reset, _O())
        model.eval()
        with torch.no_grad():
            val = run_windows(model, va, T, reset)
        hist.append((trl, val))
        print(f'pass {p} train {trl:.4f} val {val:.4f} t {time.time()-t0:.0f}s', flush=True)
    torch.save(dict(state=model.state_dict(), cfg=model.cfg, arm=arm, theta=theta, hist=hist), out)
