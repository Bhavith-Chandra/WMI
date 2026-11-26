"""E2 evaluation: LOB-Impact-style campaign for a trained lightweight generator vs ground-truth World.

Eval days 5000+ (never used in calibration or training). Per window: the model is teacher-forced on the World's
real stream from day start to the window (state carried), then 7 arms are rolled out on identical random numbers:
V+ V- (visible build-up), B+ B- (blind build-up), F (free), R+ R- (visible relaxation: 10 children, 100 silent blocks).
usage: e2_eval.py <model.pt> <out.json> [n_days] [win_per_day]
"""
import sys, json, time
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest.sim import world as W, engine as E, metrics as M, campaign as C
from almagest.experiments.e1_calibrate_world import make
from almagest.experiments.e2_train import build_stream, make_model
from almagest.models import lmm as L

torch.set_num_threads(4)
WIN0, STRIDE, CH = 20480, 4096, 256
N_EXEC, N_REL_EXEC, N_SIL = 100, 10, 100
NBLK = N_REL_EXEC + N_SIL  # 110 >= N_EXEC


def windows(n_days, wpd, base=5000):
    days, sidx = [], []
    for d in range(n_days):
        for w in range(wpd):
            days.append(base + d)
            sidx.append(WIN0 + w * STRIDE)
    return np.array(days, np.int64), np.array(sidx, np.int64)


def truth_campaign(par, days, sidx, gap, child):
    nw = 20000 + sidx
    rs = 10_000_000 + days * 1000 + (sidx - WIN0) // STRIDE
    out = {}
    for mode in (1, 2, 0):
        for side in (0, 1):
            o, m0, h, _ = W.run_batch(par, days, nw, rs, N_EXEC, gap, 0, side, child, mode)
            out[(mode, side)] = (o - m0[:, None], h)
    for side in (0, 1):
        o, m0, _, _ = W.run_batch(par, days, nw, rs, N_REL_EXEC, gap, N_SIL, side, child, 1)
        out[('R', side)] = (o - m0[:, None], None)
    return out


@torch.no_grad()
def model_campaign(model, par, days, sidx, gap, child, seed=0):
    nwin = len(days)
    udays = np.unique(days)
    # ---- teacher-forced conditioning on each day's real stream, snapshot state at window starts
    n_need = int(sidx.max())
    st = {}
    for d in udays:
        s = build_stream(par, int(d), n_need + 1)
        st[d] = s
    B = len(udays)
    state = model.init_state(B)
    snaps = {}
    want = {int(x) for x in np.unique(sidx)}
    tp = lambda k: torch.tensor(np.stack([st[d][k] for d in udays]))
    prev, feat_s, dmid, sb = tp('prev'), tp('feat'), tp('dmid'), tp('sb')
    for s0 in range(0, n_need, CH):
        sl = slice(s0, s0 + CH)
        x = model.embed(prev[:, sl], feat_s[:, sl], dmid[:, sl], sb[:, sl])
        _, state = model.trunk_seq(x, state)
        if s0 + CH in want:
            snaps[s0 + CH] = [h.clone() for h in state]
    assert want <= set(snaps), (want, sorted(snaps))
    dpos = {int(d): i for i, d in enumerate(udays)}
    # slow-bank internal state at window start (exact offline recursion up to sidx-1)
    from scipy.signal import lfilter
    lam = np.exp(-1.0 / L.SB_TAUS)
    sbst = {}
    for d in udays:
        s = st[d]
        pr = s['prev']
        sv = np.where(pr[:, 0] == 2, np.where(pr[:, 1] == 0, 1, -1) * pr[:, 3], 0).astype(float)
        p = np.cumsum(s['dmid'].astype(float))
        fv = np.stack([lfilter([1 - l], [1, -l], sv) for l in lam], -1)
        ep = np.stack([lfilter([1 - l], [1, -l], p) for l in lam], -1)
        sbst[d] = (fv, p, ep)
    # ---- arms
    ARMS = ['V+', 'V-', 'B+', 'B-', 'F', 'R+', 'R-']
    na = len(ARMS)
    R = nwin * na
    win_of = np.repeat(np.arange(nwin), na)
    arm_of = np.tile(np.arange(na), nwin)
    side_of = np.array([0 if ARMS[a].endswith('+') else 1 for a in arm_of], np.int64)
    vis = np.array([ARMS[a][0] in 'VR' for a in arm_of])
    inj = np.array([ARMS[a][0] in 'VBR' for a in arm_of])
    rel = np.array([ARMS[a][0] == 'R' for a in arm_of])
    # init engine rows from World books at window start
    rows_seed = days[win_of]
    rows_warm = 20000 + sidx[win_of]
    vb = np.zeros((R, W.P), np.int64); va = np.zeros_like(vb); rb = np.zeros_like(vb); ra = np.zeros_like(vb)
    vbb = np.zeros(R, np.int64); vba = np.zeros_like(vbb); rbb = np.zeros_like(vbb); rba = np.zeros_like(vbb)
    E.init_rows(par, rows_seed, rows_warm, vb, va, rb, ra, vbb, vba, rbb, rba)
    state = [torch.cat([snaps[int(sidx[w])][li][dpos[int(days[w])]:dpos[int(days[w])] + 1] for w in win_of])
             for li in range(len(model.blocks))]
    # pending token = last real event before window start
    pend_ev = torch.tensor(np.stack([st[days[w]]['tgt'][sidx[w] - 1] for w in win_of]))
    sbank = L.SlowBank(R)
    for i, w in enumerate(win_of):
        fv, p, ep = sbst[days[w]]
        j = sidx[w] - 1
        sbank.fv[i] = torch.tensor(fv[j]); sbank.p[i] = float(p[j]); sbank.ep[i] = torch.tensor(ep[j])
    pend_dmid = torch.tensor(np.array([float(st[days[w]]['dmid'][sidx[w]]) for w in win_of]), dtype=torch.float32)
    # dmid at index sidx = mid(before event sidx) - mid(before event sidx-1) = effect of event sidx-1: exact.
    # sbank state must then include index sidx; advance one step:
    sv_p = torch.where(pend_ev[:, 0] == 2, torch.where(pend_ev[:, 1] == 0, 1.0, -1.0) * pend_ev[:, 3].float(),
                       torch.zeros(R))
    sbank.update(sv_p, pend_dmid)
    feat = np.zeros((R, 21)); cmb = np.zeros((R, W.D), np.int64); cma = np.zeros_like(cmb)
    E.observe(vb, va, vbb, vba, feat, cmb, cma)
    pend_feat = torch.tensor(feat, dtype=torch.float32)
    pend_cm = torch.stack([torch.tensor(cmb), torch.tensor(cma)], 1)
    gen = torch.Generator().manual_seed(seed)
    out_mid = np.zeros((R, NBLK))
    mid0 = 0.5 * (rbb + rba).astype(float)
    free_mid = []
    free_sign = []
    hsh = np.zeros(R, np.uint64) + np.uint64(1469598103934665603)
    ones = np.ones(R, np.int64)

    def consume(mask):
        nonlocal state
        x = model.embed(pend_ev, pend_feat, pend_dmid, sbank.feat())
        z, ns = model.trunk_step(x, state)
        m = torch.tensor(mask).unsqueeze(-1)
        state = [torch.where(m, n, o) for n, o in zip(ns, state)]
        return z

    def set_pending(rows_mask, ev, old_vmid):
        nonlocal pend_ev, pend_feat, pend_dmid, pend_cm
        E.observe(vb, va, vbb, vba, feat, cmb, cma)
        newmid = 0.5 * (vbb + vba).astype(float)
        m = torch.tensor(rows_mask)
        d = torch.tensor(newmid - old_vmid, dtype=torch.float32)
        pend_ev = torch.where(m.unsqueeze(-1), ev, pend_ev)
        pend_feat = torch.where(m.unsqueeze(-1), torch.tensor(feat, dtype=torch.float32), pend_feat)
        pend_dmid = torch.where(m, d, pend_dmid)
        pend_cm = torch.stack([torch.tensor(cmb), torch.tensor(cma)], 1)
        sv = torch.where(ev[:, 0] == 2, torch.where(ev[:, 1] == 0, 1.0, -1.0) * ev[:, 3].float(), torch.zeros(R))
        fv0, p0, ep0 = sbank.fv.clone(), sbank.p.clone(), sbank.ep.clone()
        sbank.update(sv, d)
        mm = m.unsqueeze(-1)
        sbank.fv = torch.where(mm, sbank.fv, fv0); sbank.p = torch.where(mm, sbank.p, p0)
        sbank.ep = torch.where(mm, sbank.ep, ep0)

    fidx = np.where(np.array([ARMS[a] == 'F' for a in arm_of]))[0]
    for blk in range(NBLK):
        do_inj = inj & ((~rel & (blk < N_EXEC)) | (rel & (blk < N_REL_EXEC)))
        if do_inj.any():
            vmask = do_inj & vis
            if vmask.any():
                consume(vmask)  # consume pending (prediction discarded: next event is the forced child)
            old = 0.5 * (vbb + vba).astype(float)
            E.inject(vb, va, rb, ra, vbb, vba, rbb, rba, side_of, child, vmask.astype(np.int64),
                     do_inj.astype(np.int64))
            cev = torch.stack([torch.full((R,), 2), torch.tensor(side_of), torch.zeros(R, dtype=torch.long),
                               torch.full((R,), child)], 1)
            set_pending(vmask, cev, old)
        for g in range(gap):
            z = consume(np.ones(R, bool))
            u = torch.rand(nwin, 4, generator=gen)[torch.tensor(win_of)]
            lt = model.h_typ(z)
            typ = L.inv_cdf(lt, u[:, 0])
            ls = model.h_side(z + model.c_side(typ))
            side = L.inv_cdf(ls, u[:, 1])
            zl = z + model.c_side(typ) + model.c_lvl(typ * 2 + side)
            ll = L.mask_lvl(model.h_lvl(zl), typ, side, pend_cm)
            lvl = L.inv_cdf(ll, u[:, 2])
            size = L.inv_cdf(model.h_size(zl), u[:, 3]) + 1
            lvl = torch.where(typ == 2, torch.zeros_like(lvl), lvl)
            size = torch.where(typ == 2, size, torch.ones_like(size))
            ev = torch.stack([typ, side, lvl, size], 1)
            old = 0.5 * (vbb + vba).astype(float)
            evn = ev.numpy().astype(np.int64)
            E.step_events(vb, va, rb, ra, vbb, vba, rbb, rba, evn[:, 0].copy(), evn[:, 1].copy(),
                          evn[:, 2].copy(), evn[:, 3].copy(), ones)
            code = (evn[:, 0] * 1000003 + evn[:, 1] * 10007 + evn[:, 2] * 31 + evn[:, 3]).astype(np.uint64)
            hsh = (hsh ^ code) * np.uint64(1099511628211)
            set_pending(np.ones(R, bool), ev, old)
            free_mid.append(0.5 * (rbb[fidx] + rba[fidx]).astype(float))
            free_sign.append(np.where(evn[fidx, 0] == 2, np.where(evn[fidx, 1] == 0, 1, -1), 0))
        out_mid[:, blk] = 0.5 * (rbb + rba)
    I = out_mid - mid0[:, None]
    res = {}
    for a, name in enumerate(ARMS):
        res[name] = I[arm_of == a]
        res['h' + name] = hsh[arm_of == a]
    return res, np.array(free_mid).T, np.array(free_sign).T


def analyze(Ivis, Iblind, Irel, V, sig, did, child, free_drift, gamma, H, ref, ow_ep=None):
    q = (np.arange(1, N_EXEC + 1)[None, :] * child) / V[:, None]
    Iv = Ivis / sig[:, None]
    Ib = Iblind / sig[:, None]
    ok, d_l2, Y_l2 = M.identifiable(q, Iv)
    d, Y = M.law_from_pairs(Iv, q)
    bd, bY = M.day_bootstrap(Iv, q, did, B=300)
    tost, ci90 = M.tost_pass(bd)
    vb = Ivis[:, -1].mean() / Iblind[:, -1].mean() if Iblind[:, -1].mean() > 0 else float('inf')
    Ir = (Irel / sig[:, None]).mean(0)
    ep, slope = M.relaxation(Ir, N_REL_EXEC)
    r = dict(delta=d, Y=Y, delta_ci95=list(np.nanquantile(bd, [0.025, 0.975])), delta_ci90=list(ci90), tost=tost,
             Y_ci95=list(np.nanquantile(bY, [0.025, 0.975])), identified=bool(ok), delta_l2=d_l2, vb=vb,
             ep=ep, slope=slope, drift=free_drift, gamma=gamma, H=H,
             curve=list(Iv.mean(0)), relax_curve=list(Ir))
    r['ladder'] = {k: (None if v is None else bool(v)) for k, v in M.ladder(ref, r, ow_ep).items()}
    return r


def short_stats(mid_rows, sign_rows):
    g = np.nanmean([M.sign_acf_gamma(s, 5, 200) for s in sign_rows])
    H = np.mean([M.hurst(m, 3, 11) for m in mid_rows])
    return g, H


if __name__ == '__main__':
    ck = torch.load(sys.argv[1], weights_only=False)
    out_path = sys.argv[2]
    n_days = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    wpd = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    par = make(ck['theta'])
    model, _ = make_model(ck['arm'])
    model.load_state_dict(ck['state'])
    model.eval()
    days, sidx = windows(n_days, wpd)
    stats = C.day_stats(par, np.unique(days))
    phi = np.mean([s['phi'] for s in stats])
    child = int(round(np.median([s['med'] for s in stats])))
    gap = int(round((1 / 0.10 - 1) / phi))
    did = days - days.min()
    V = np.array([stats[d]['V'] for d in did], float)
    sig = np.array([stats[d]['sigma'] for d in did], float)
    t0 = time.time()
    tr = truth_campaign(par, days, sidx, gap, child)
    Tv = 0.5 * (tr[(1, 0)][0] - tr[(1, 1)][0])
    Tb = 0.5 * (tr[(2, 0)][0] - tr[(2, 1)][0])
    Tr = 0.5 * (tr[('R', 0)][0] - tr[('R', 1)][0])
    # truth free stats on the same horizon (gap*NBLK events) from the real stream
    hor = gap * NBLK
    tm, ts = [], []
    for d, s0 in zip(days, sidx):
        s = build_stream(par, int(d), int(s0) + hor)
        mids = np.cumsum(s['dmid'])[s0:s0 + hor]
        tm.append(mids)
        ts.append(np.where(s['tgt'][s0:s0 + hor, 0] == 2, np.where(s['tgt'][s0:s0 + hor, 1] == 0, 1, -1), 0))
    tg, tH = short_stats(tm, ts)
    ref = dict(gamma=tg, H=tH)
    truth = analyze(Tv, Tb, Tr, V, sig, did, child, 0.0, tg, tH, ref)
    print('truth', {k: v for k, v in truth.items() if k not in ('curve', 'relax_curve')}, f'{time.time()-t0:.0f}s',
          flush=True)
    res, fmid, fsign = model_campaign(model, par, days, sidx, gap, child)
    blind_inv = bool(np.all(res['hB+'] == res['hF']) and np.all(res['hB-'] == res['hF']))
    Iv = 0.5 * (res['V+'][:, :N_EXEC] - res['V-'][:, :N_EXEC])
    Ib = 0.5 * (res['B+'][:, :N_EXEC] - res['B-'][:, :N_EXEC])
    Ir = 0.5 * (res['R+'] - res['R-'])
    drift = float(res['F'][:, -1].mean())
    mg, mH = short_stats(fmid, fsign)
    mres = analyze(Iv, Ib, Ir, V, sig, did, child, drift, mg, mH, ref)
    mres['blind_invariant'] = blind_inv
    print('model', ck['arm'], {k: v for k, v in mres.items() if k not in ('curve', 'relax_curve')},
          f'{time.time()-t0:.0f}s', flush=True)
    json.dump(dict(arm=ck['arm'], params=model.n_params(), truth=truth, model=mres, gap=gap, child=child,
                   n_days=n_days, wpd=wpd, hist=ck.get('hist')), open(out_path, 'w'), default=float, indent=1)
