"""LOB-Impact-style campaign against the ground-truth World (or any batch generator with the same API).

Protocol constants fixed a priori (mirroring 2610.27100): eta = 10% participation, N = 100 build-up children,
relaxation shape i = 10 children + c = 100 silent blocks, child = median market-order size,
gap = Delta_N = (1/eta - 1)/phi, both directions, visible + blind on identical seeds.
"""
import numpy as np
from . import world as W
from . import metrics as M

T_DAY = 100_000      # events per synthetic "day" (V, sigma measured over [0, T_DAY) after warm-up)
WARM0 = 40_480       # = 20000 (stream offset) + 20480: SAME window placement as E2
STRIDE = 2048


def day_stats(par, day_seeds, n=T_DAY):
    """Per-day V (market volume), sigma (Parkinson, ticks), phi, signs/mid for C1."""
    out = []
    for s in day_seeds:
        mid = np.zeros(n)
        sg = np.zeros(n, np.int64)
        vol = np.zeros(n, np.int64)
        W.free_run(par, int(s), n, mid, sg, vol)
        out.append(dict(V=vol.sum(), sigma=(mid.max() - mid.min()) / M.KAPPA_P, phi=(sg != 0).mean(),
                        gamma=M.sign_acf_gamma(sg), H=M.hurst(mid), med=np.median(vol[sg != 0])))
    return out


def schedule(n_days, win_per_day, base_seed, stride=STRIDE, replicas=1):
    from almagest.splits import roll_seed
    ds, nw, rs, did = [], [], [], []
    for d in range(n_days):
        for w in range(win_per_day):
            for r in range(replicas):
                ds.append(base_seed + d)
                nw.append(WARM0 + w * stride)
                rs.append(roll_seed(base_seed + d, w, r))
                did.append(d)
    return (np.array(ds, np.int64), np.array(nw, np.int64), np.array(rs, np.int64), np.array(did))


def run_arms(runner, sched, n_child, gap, n_silent, child, modes=(1, 2, 0), vis_tail=0):
    """runner(day_seeds, n_warms, roll_seeds, n_child, gap, n_silent, side, child, mode) -> (out, mid0, hash, mvol).
    Returns antisymmetrised impact (ticks) per mode, plus free drift and blind-invariance check."""
    ds, nw, rs, _ = sched
    res = {}
    hashes = {}
    raw = {}
    for mode in modes:
        ns = n_silent + (vis_tail if mode == 1 else 0)
        ob, m0b, hb, _ = runner(ds, nw, rs, n_child, gap, ns, 0, child, mode)
        os_, m0s, hs, _ = runner(ds, nw, rs, n_child, gap, ns, 1, child, mode)
        res[mode] = 0.5 * ((ob - m0b[:, None]) - (os_ - m0s[:, None]))
        hashes[mode] = (hb, hs)
        raw[mode] = (ob - m0b[:, None], os_ - m0s[:, None])
    return res, hashes, raw


def world_runner(par):
    """Snapshot-backed World runner (bit-identical to W.run_batch, tested)."""
    cache = {}

    def f(ds, nw, rs, n_child, gap, n_silent, side, child, mode):
        key = (tuple(np.unique(ds)), tuple(np.unique(nw)))
        if key not in cache:
            cache.clear()
            cache[key] = W.Snapshots(par, np.unique(ds), np.unique(nw))
        f.snap = cache[key]
        return cache[key].run(ds, nw, rs, n_child, gap, n_silent, side, child, mode)
    f.snap = None
    return f


def evaluate(runner, par_world, n_days=20, win_per_day=16, base_seed=1000, ref=None, B=300, verbose=True,
             relax=True, stats=None, relax_mult=1, replicas=4):
    """Full campaign. `par_world` defines the target market (day statistics). Returns dict of results."""
    days = np.arange(base_seed, base_seed + n_days)
    stats = stats or day_stats(par_world, days)
    phi = np.mean([s['phi'] for s in stats])
    child = int(round(np.median([s['med'] for s in stats])))
    gap = int(round((1 / 0.10 - 1) / phi))
    sched = schedule(n_days, win_per_day, base_seed, replicas=replicas)
    did = sched[3]
    V = np.array([stats[d]['V'] for d in did], float)
    sig = np.array([stats[d]['sigma'] for d in did], float)
    N = 100
    res, hashes, raw = run_arms(runner, sched, N, gap, 0, child, modes=(1, 2, 0), vis_tail=100)
    # R100: relaxation after the FULL build-up (visible arm continues 100 silent blocks); higher SNR than R10
    I100 = res[1] / sig[:, None]
    ep100, slope100 = M.relaxation(I100.mean(0), N)
    ep100_ci = M.relax_bootstrap(I100, did, N, B=200)
    res[1] = res[1][:, :N]
    q = (np.arange(1, N + 1)[None, :] * child) / V[:, None]
    Iv = res[1] / sig[:, None]
    Ib = res[2] / sig[:, None]
    ok, d_l2, Y_l2 = M.identifiable(q, Iv)
    d, Y = M.law_from_pairs(Iv, q)
    bd, bY = M.day_bootstrap(Iv, q, did, B=B)
    ci95 = np.nanquantile(bd, [0.025, 0.975])
    tost, ci90 = M.tost_pass(bd)
    vis_peak = res[1][:, -1].mean()
    blind_floor = res[2][:, -1].mean()
    vb = M.vb_ratio(res[1][:, -1], res[2][:, -1])
    # free arm (no injection): drift = mean signed mid change over build-up horizon (buy-arm seeds)
    drift = raw[0][0][:, -1].mean()
    blind_equals_free = bool(np.all(hashes[2][0] == hashes[0][0]) and np.all(hashes[2][1] == hashes[0][1]))
    out = dict(phi=phi, child=child, gap=gap, delta=d, Y=Y, delta_ci95=ci95, delta_ci90=ci90, tost=tost,
               Y_ci95=np.nanquantile(bY, [0.025, 0.975]), identified=ok, delta_l2=d_l2, vb=vb,
               drift=drift, blind_equals_free=blind_equals_free, ep100=ep100, ep100_ci95=ep100_ci, slope100=slope100,
               gamma=np.nanmean([s['gamma'] for s in stats]), H=np.mean([s['H'] for s in stats]),
               curve=Iv.mean(0), q=q.mean(0))
    if relax:
        sched_r = schedule(n_days, win_per_day * relax_mult, base_seed, stride=STRIDE // relax_mult, replicas=replicas)
        V_r = np.array([stats[d]['V'] for d in sched_r[3]], float)
        sig_r = np.array([stats[d]['sigma'] for d in sched_r[3]], float)
        rr, _, _ = run_arms(runner, sched_r, 10, gap, 100, child, modes=(1,))
        Irows = rr[1] / sig_r[:, None]
        Irel = Irows.mean(0)
        ep, slope = M.relaxation(Irel, 10)
        out.update(ep=ep, slope=slope, relax_curve=Irel, ep_ci95=M.relax_bootstrap(Irows, sched_r[3], 10, B=200))
    # hidden-state excursions at window starts (realism): fundamental and long-reference gaps in daily-sigma units
    snap = getattr(runner, 'snap', None)
    if snap is not None:
        mid = 0.5 * (snap.sbb + snap.sba).astype(float)
        sig_d = np.array([stats[d]['sigma'] for d in range(len(stats))])[:, None]
        out['exc_F'] = float(np.std((snap.sema[:, :, W.IF] - mid) / sig_d))
        out['exc_r'] = float(np.std((snap.sema[:, :, W.NEMA - 1] - mid) / sig_d))
    if ref is not None:
        out['ladder'] = M.ladder(ref, out)
    if verbose:
        print({k: (np.round(v, 3) if isinstance(v, (float, np.floating, np.ndarray)) and np.size(v) < 5 else v)
               for k, v in out.items() if k not in ('curve', 'q', 'relax_curve')})
    return out
