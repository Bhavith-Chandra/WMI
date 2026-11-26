"""E2 evaluation: LOB-Impact-style campaign for a trained lightweight generator vs ground-truth World.

Per window: the model is teacher-forced on the World's real stream from day start to the window, then 7 arms are
rolled out on identical random numbers: V+ V- (visible build-up), B+ B- (blind build-up), F (free),
R+ R- (visible relaxation: 10 children, 100 silent blocks). The truth campaign uses the SAME windows.

usage: e2_eval.py <model.pt|-> <out.json> <dev|test> [n_days] [win_per_day]
  '-' as model = truth-only certification (allowed on dev only).
Output files are never overwritten. Each output records split, day seeds, code revision and config.
Audit fixes (2026-11-26): reset-arm parity, pooled C1 stats on identical windows, fail-closed v/b with CI, truth drift
from the no-injection arm, truth blind-hash assertion, stronger model hash (absolute view prices), fixed q_edge,
windows inside the day span used for V/sigma, World burn-in (world.BURN).
"""
import os, sys, json, time, subprocess
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest import splits as S
from almagest.sim import world as W, engine as E, metrics as M, campaign as C
from almagest.experiments.e1_calibrate_world import make
from almagest.experiments.e2_train import build_stream, make_model
from almagest.models import lmm as L

torch.set_num_threads(1)   # oversubscription with numba + other jobs made ops ~100x slower
WIN0, STRIDE, CH = 20480, 2048, 256     # 32 windows/day -> last window ends at 20480+31*2048+8910 < T_DAY
N_EXEC, N_REL_EXEC, N_SIL = 100, 10, 100
NBLK = N_EXEC + N_SIL          # 200: V arms = 100-child build-up + 100 silent blocks (R100); R arms use first 110 (R10)
TRUTH_REPLICAS = 8             # truth is cheap (numba snapshots): 8 rollout seeds per window


def windows(n_days, wpd, base):
    days, sidx = [], []
    for d in range(n_days):
        for w in range(wpd):
            days.append(base + d)
            sidx.append(WIN0 + w * STRIDE)
    days, sidx = np.array(days, np.int64), np.array(sidx, np.int64)
    return days, sidx


def roll_seeds(days, sidx, replica=0):
    return np.array([S.roll_seed(int(d), int((s - WIN0) // STRIDE), replica) for d, s in zip(days, sidx)], np.int64)


def truth_campaign(par, days, sidx, gap, child, snaps, replicas=TRUTH_REPLICAS):
    D = np.repeat(days, replicas)
    Sx = np.repeat(sidx, replicas)
    rs = np.concatenate([[S.roll_seed(int(d), int((s - WIN0) // STRIDE), r) for r in range(replicas)]
                         for d, s in zip(days, sidx)]).astype(np.int64)
    nw = 20000 + Sx
    out = {'days': D}
    for mode in (1, 2, 0):
        for side in (0, 1):
            tail = N_SIL if mode == 1 else 0
            o, m0, h, _ = snaps.run(D, nw, rs, N_EXEC, gap, tail, side, child, mode)
            out[(mode, side)] = (o - m0[:, None], h)
    for side in (0, 1):
        o, m0, _, _ = snaps.run(D, nw, rs, N_REL_EXEC, gap, N_SIL, side, child, 1)
        out[('R', side)] = (o - m0[:, None], None)
    return out


@torch.no_grad()
def model_campaign(model, reset, par, days, sidx, gap, child, snaps, st, seed=0):
    nwin = len(days)
    udays = np.unique(days)
    n_need = int(sidx.max())
    B = len(udays)
    state = model.init_state(B)
    tf_snaps = {}
    want = {int(x) for x in np.unique(sidx)}
    tp = lambda k: torch.tensor(np.stack([st[d][k] for d in udays]))
    prev, feat_s, dmid, sb = tp('prev'), tp('feat'), tp('dmid'), tp('sb')
    SRC = bool(getattr(model, 'src', False))
    pad = (lambda t, v: torch.cat([t, torch.full(t.shape[:-1] + (1,), v, dtype=t.dtype)], -1)) if SRC else (lambda t, v: t)
    prev = pad(prev, 0)                      # real context events are market events (src=0)
    half = reset // 2 if reset else None
    for s0 in range(0, n_need, CH):
        if reset and s0 % reset == half:      # reset boundaries at sidx - reset/2 -> rollout starts mid-segment
            state = model.init_state(B)
        sl = slice(s0, s0 + CH)
        x = model.embed(prev[:, sl], feat_s[:, sl], dmid[:, sl], sb[:, sl])
        _, state = model.trunk_seq(x, state)
        if s0 + CH in want:
            tf_snaps[s0 + CH] = [h.clone() for h in state]
    assert want <= set(tf_snaps)
    dpos = {int(d): i for i, d in enumerate(udays)}
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
    ARMS = ['V+', 'V-', 'B+', 'B-', 'F', 'R+', 'R-']
    na = len(ARMS)
    R = nwin * na
    win_of = np.repeat(np.arange(nwin), na)
    arm_of = np.tile(np.arange(na), nwin)
    side_of = np.array([0 if ARMS[a].endswith('+') else 1 for a in arm_of], np.int64)
    vis = np.array([ARMS[a][0] in 'VR' for a in arm_of])
    inj = np.array([ARMS[a][0] in 'VBR' for a in arm_of])
    rel = np.array([ARMS[a][0] == 'R' for a in arm_of])
    rows_seed = days[win_of]
    rows_warm = 20000 + sidx[win_of]
    vb = np.zeros((R, W.P), np.int64); va = np.zeros_like(vb); rb = np.zeros_like(vb); ra = np.zeros_like(vb)
    vbb = np.zeros(R, np.int64); vba = np.zeros_like(vbb); rbb = np.zeros_like(vbb); rba = np.zeros_like(vbb)
    rd, rw = snaps.index(rows_seed, rows_warm)
    vb[:] = snaps.sb[rd, rw]; va[:] = snaps.sa[rd, rw]; rb[:] = vb; ra[:] = va
    vbb[:] = snaps.sbb[rd, rw]; vba[:] = snaps.sba[rd, rw]; rbb[:] = vbb; rba[:] = vba
    state = [torch.cat([tf_snaps[int(sidx[w])][li][dpos[int(days[w])]:dpos[int(days[w])] + 1] for w in win_of])
             for li in range(len(model.blocks))]
    # tokens consumed since last reset (reset arm only): reset/2 at window start, by construction above
    since = np.full(R, half if reset else 0, np.int64)
    pend_ev = pad(torch.tensor(np.stack([st[days[w]]['tgt'][sidx[w] - 1] for w in win_of])), 0)
    sbank = L.SlowBank(R)
    for i, w in enumerate(win_of):
        fv, p, ep = sbst[days[w]]
        j = sidx[w] - 1
        sbank.fv[i] = torch.tensor(fv[j]); sbank.p[i] = float(p[j]); sbank.ep[i] = torch.tensor(ep[j])
    pend_dmid = torch.tensor(np.array([float(st[days[w]]['dmid'][sidx[w]]) for w in win_of]), dtype=torch.float32)
    sv_p = torch.where(pend_ev[:, 0] == 2, torch.where(pend_ev[:, 1] == 0, 1.0, -1.0) * pend_ev[:, 3].float(),
                       torch.zeros(R))
    sbank.update(sv_p, pend_dmid)
    feat = np.zeros((R, 21)); cmb = np.zeros((R, W.D), np.int64); cma = np.zeros_like(cmb)
    E.observe(vb, va, vbb, vba, feat, cmb, cma)
    # sanity: engine start book == the real stream's book at the window (guards stale numba caches)
    for i in range(0, R, max(1, R // 16)):
        assert np.allclose(feat[i], st[days[win_of[i]]]['feat'][sidx[win_of[i]]]), 'engine/stream book mismatch'
    pend_feat = torch.tensor(feat, dtype=torch.float32)
    pend_cm = torch.stack([torch.tensor(cmb), torch.tensor(cma)], 1)
    gen = torch.Generator().manual_seed(seed)
    out_mid = np.zeros((R, NBLK))
    mid0 = 0.5 * (rbb + rba).astype(float)
    free_mid, free_sign, free_typ = [], [], []
    hsh = np.zeros(R, np.uint64) + np.uint64(1469598103934665603)
    ones = np.ones(R, np.int64)

    def consume(mask):
        nonlocal state, since
        if reset:
            rz = torch.tensor((since % reset == 0) & mask).unsqueeze(-1)
            state = [torch.where(rz, torch.zeros_like(h), h) for h in state]
        x = model.embed(pend_ev, pend_feat, pend_dmid, sbank.feat())
        z, ns = model.trunk_step(x, state)
        m = torch.tensor(mask).unsqueeze(-1)
        state = [torch.where(m, n, o) for n, o in zip(ns, state)]
        since = since + mask.astype(np.int64)
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
                consume(vmask)
            old = 0.5 * (vbb + vba).astype(float)
            E.inject(vb, va, rb, ra, vbb, vba, rbb, rba, side_of, child, vmask.astype(np.int64),
                     do_inj.astype(np.int64))
            cev = pad(torch.stack([torch.full((R,), 2), torch.tensor(side_of), torch.zeros(R, dtype=torch.long),
                                   torch.full((R,), child)], 1), 1)   # our own (exogenous) order: src=1
            set_pending(vmask, cev, old)
        for g in range(gap):
            z = consume(np.ones(R, bool))
            u = torch.rand(nwin, 4, generator=gen)[torch.tensor(win_of)]
            typ = L.inv_cdf(model.h_typ(z), u[:, 0])
            side = L.inv_cdf(model.h_side(z + model.c_side(typ)), u[:, 1])
            zl = z + model.c_side(typ) + model.c_lvl(typ * 2 + side)
            lvl = L.inv_cdf(L.mask_lvl(model.h_lvl(zl), typ, side, pend_cm), u[:, 2])
            size = L.inv_cdf(model.h_size(zl), u[:, 3]) + 1
            lvl = torch.where(typ == 2, torch.zeros_like(lvl), lvl)
            size = torch.where(typ == 2, size, torch.ones_like(size))
            ev = pad(torch.stack([typ, side, lvl, size], 1), 0)
            old = 0.5 * (vbb + vba).astype(float)
            evn = ev.numpy().astype(np.int64)
            E.step_events(vb, va, rb, ra, vbb, vba, rbb, rba, evn[:, 0].copy(), evn[:, 1].copy(),
                          evn[:, 2].copy(), evn[:, 3].copy(), ones)
            # hash: event + resulting ABSOLUTE view quotes (what the model will see next)
            code = (evn[:, 0] * 1000003 + evn[:, 1] * 10007 + evn[:, 2] * 31 + evn[:, 3]
                    + vbb * 7919 + vba * 104729).astype(np.uint64)
            hsh = (hsh ^ code) * np.uint64(1099511628211)
            set_pending(np.ones(R, bool), ev, old)
            free_mid.append(0.5 * (rbb[fidx] + rba[fidx]).astype(float))
            free_sign.append(np.where(evn[fidx, 0] == 2, np.where(evn[fidx, 1] == 0, 1, -1), 0))
            free_typ.append(evn[fidx, 0])
        out_mid[:, blk] = 0.5 * (rbb + rba)
    I = out_mid - mid0[:, None]
    res = {}
    for a, name in enumerate(ARMS):
        res[name] = I[arm_of == a]
        res['h' + name] = hsh[arm_of == a]
    ft = np.array(free_typ)
    res['phi_model'] = float((ft == 2).mean())
    return res, np.array(free_mid).T, np.array(free_sign).T


def analyze(Ivis, Iblind, Irel, V, sig, did, child, free_drift, gamma, H, ref, free_end=None):
    """Ivis: (n, N_EXEC + N_SIL) visible build-up + silent tail; Irel: (n, >=110) R10 arm."""
    free_sd = float(np.std(free_end / sig)) if free_end is not None else float('nan')  # free-run vol, sigma units
    I100 = Ivis / sig[:, None]
    ep100, slope100 = M.relaxation(I100.mean(0), N_EXEC)
    ep100_ci = list(M.relax_bootstrap(I100, did, N_EXEC, B=200))
    Ivis = Ivis[:, :N_EXEC]
    Irel = Irel[:, :N_REL_EXEC + N_SIL]
    q = (np.arange(1, N_EXEC + 1)[None, :] * child) / V[:, None]
    q_edge = float(np.median(q[:, -1]))
    Iv = Ivis / sig[:, None]
    ok, d_l2, _ = M.identifiable(q, Iv)
    d, Y = M.law_from_pairs(Iv, q, q_edge)
    bd, bY = M.day_bootstrap(Iv, q, did, B=300)
    tost, ci90 = M.tost_pass(bd)
    vb = M.vb_ratio(Ivis[:, -1], Iblind[:, -1])
    vb_ci = M.vb_bootstrap(Ivis[:, -1], Iblind[:, -1], did)
    Irows = Irel / sig[:, None]
    Ir = Irows.mean(0)
    ep, slope = M.relaxation(Ir, N_REL_EXEC)
    ep_ci = list(M.relax_bootstrap(Irows, did, N_REL_EXEC, B=200))
    r = dict(delta=d, Y=Y, q_edge=q_edge, delta_ci95=list(np.nanquantile(bd, [0.025, 0.975])), delta_ci90=list(ci90),
             tost=tost, Y_ci95=list(np.nanquantile(bY, [0.025, 0.975])), identified=bool(ok), delta_l2=d_l2,
             vb=vb, vb_ci95=list(vb_ci), ep=ep, free_sd=free_sd, ep_ci95=ep_ci, slope=slope, drift=free_drift, gamma=gamma, H=H,
             curve=list(Iv.mean(0)), relax_curve=list(Ir), ep100=ep100, ep100_ci95=ep100_ci, slope100=slope100,
             relax100_curve=list(I100.mean(0)))
    r['ladder'] = {k: (None if v is None else bool(v)) for k, v in M.ladder(ref, r).items()}
    return r


def git_rev():
    try:
        root = __file__.rsplit('/almagest/', 1)[0]
        rev = subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD']).decode().strip()
        dirty = subprocess.check_output(['git', '-C', root, 'status', '--porcelain', '--', 'almagest/sim',
                                         'almagest/models', 'almagest/experiments']).decode().strip()
        return rev + ('+dirty' if dirty else '')
    except Exception:
        return 'unknown'


if __name__ == '__main__':
    mpath, out_path, split = sys.argv[1], sys.argv[2], sys.argv[3]
    n_days = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    wpd = int(sys.argv[5]) if len(sys.argv) > 5 else 30   # 30 windows: last window + 200-block horizon stays inside T_DAY
    ETA = float(sys.argv[6]) if len(sys.argv) > 6 else 0.10   # participation; A8 uses held-out 0.05 and 0.20
    assert split in ('dev', 'test')
    assert not os.path.exists(out_path), f'refusing to overwrite {out_path}'
    base = S.DEV_BASE if split == 'dev' else S.TEST_BASE
    rev = git_rev()
    if split == 'test':
        assert not rev.endswith('+dirty') and rev != 'unknown', 'test runs require a clean committed tree'
    if mpath == '-':
        assert split == 'dev', 'truth-only certification is dev-only'
        theta = json.load(open(os.environ['WORLD']))['theta']
        ck = None
    else:
        ck = torch.load(mpath, weights_only=False)
        theta = ck['theta']
    par = make(theta)
    days, sidx = windows(n_days, wpd, base)
    stats = C.day_stats(par, np.unique(days))
    phi = np.mean([s['phi'] for s in stats])
    child = int(round(np.median([s['med'] for s in stats])))
    gap = int(round((1 / ETA - 1) / phi))
    if sidx.max() + gap * NBLK > C.T_DAY:      # low participation -> longer horizon: keep windows inside the day span
        keep = sidx + gap * NBLK <= C.T_DAY
        assert keep.any(), 'rollout horizon exceeds day span for every window'
        days, sidx = days[keep], sidx[keep]
    did = days - days.min()
    V = np.array([stats[d]['V'] for d in did], float)
    sig = np.array([stats[d]['sigma'] for d in did], float)
    t0 = time.time()
    snaps = W.Snapshots(par, np.unique(days), np.unique(20000 + sidx))
    st = {d: build_stream(par, int(d), int(sidx.max()) + gap * NBLK + 1) for d in np.unique(days)}
    tr = truth_campaign(par, days, sidx, gap, child, snaps)
    truth_blind_ok = bool(np.all(tr[(2, 0)][1] == tr[(0, 0)][1]) and np.all(tr[(2, 1)][1] == tr[(0, 1)][1]))
    assert truth_blind_ok, 'truth blind arm differs from no-injection arm'
    Tv = 0.5 * (tr[(1, 0)][0] - tr[(1, 1)][0])
    Tb = 0.5 * (tr[(2, 0)][0] - tr[(2, 1)][0])
    Tr = 0.5 * (tr[('R', 0)][0] - tr[('R', 1)][0])
    t_drift = float(tr[(0, 0)][0][:, N_EXEC - 1].mean())
    tdid = tr['days'] - days.min()
    tV = np.array([stats[d]['V'] for d in tdid], float)
    tsig = np.array([stats[d]['sigma'] for d in tdid], float)
    hor = gap * NBLK
    tm, ts = [], []
    for d, s0 in zip(days, sidx):
        s = st[d]
        mids = np.cumsum(s['dmid'])[s0:s0 + hor]
        tm.append(mids)
        ts.append(np.where(s['tgt'][s0:s0 + hor, 0] == 2, np.where(s['tgt'][s0:s0 + hor, 1] == 0, 1, -1), 0))
    tg, tH = M.pooled_gamma(ts), M.pooled_hurst(tm)
    ref = dict(gamma=tg, H=tH)
    truth = analyze(Tv, Tb, Tr, tV, tsig, tdid, child, t_drift, tg, tH, ref, free_end=tr[(0, 0)][0][:, N_EXEC - 1])
    truth['n_rollouts'] = int(len(tdid))
    truth['blind_invariant'] = truth_blind_ok
    print('truth', {k: v for k, v in truth.items() if k not in ('curve', 'relax_curve')}, f'{time.time()-t0:.0f}s',
          flush=True)
    rec = dict(split=split, base=base, days=sorted(set(days.tolist())), n_days=n_days, wpd=wpd, stride=STRIDE,
               truth_replicas=TRUTH_REPLICAS, nblk=NBLK, eta=ETA,
               win0=WIN0, gap=gap, child=child, phi_world=phi, code_rev=rev, burn=W.BURN, truth=truth, theta=theta)
    if ck is not None:
        model, reset = make_model(ck['arm'])
        model.load_state_dict(ck['state'])
        model.eval()
        res, fmid, fsign = model_campaign(model, reset, par, days, sidx, gap, child, snaps, st)
        blind_inv = bool(np.all(res['hB+'] == res['hF']) and np.all(res['hB-'] == res['hF']))
        vis_differs = bool(np.mean(res['hV+'] != res['hF']) > 0.5)
        assert blind_inv and vis_differs, (blind_inv, vis_differs)
        Iv = 0.5 * (res['V+'] - res['V-'])
        Ib = 0.5 * (res['B+'][:, :N_EXEC] - res['B-'][:, :N_EXEC])
        Ir = 0.5 * (res['R+'] - res['R-'])
        drift = float(res['F'][:, N_EXEC - 1].mean())
        mres = analyze(Iv, Ib, Ir, V, sig, did, child, drift, M.pooled_gamma(fsign), M.pooled_hurst(fmid), ref,
                       free_end=res['F'][:, N_EXEC - 1])
        mres.update(n_rollouts=int(len(did)), blind_invariant=blind_inv, visible_differs=vis_differs, phi_model=res['phi_model'],
                    participation_model=1.0 / (1.0 + gap * res['phi_model']))
        print('model', ck['arm'], {k: v for k, v in mres.items() if k not in ('curve', 'relax_curve')},
              f'{time.time()-t0:.0f}s', flush=True)
        rec.update(arm=ck['arm'], params=model.n_params(), reset=reset, model=mres, hist=ck.get('hist'))
    json.dump(rec, open(out_path, 'x'), default=float, indent=1)
