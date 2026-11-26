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

torch.set_num_threads(4)
import os
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
    tr = stack([build_stream(par, S.TRAIN_BASE + i, NTOK) for i in range(NTR)], 1)   # B=16 contiguous streams
    va = stack([build_stream(par, S.VAL_BASE + i, 100_000) for i in range(4)], 1)
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
