"""E1-prop: calibrate the PROPAGATOR-ANCHOR World (impact law realistic by construction) on calibration days.
theta = [kappa_m, kappa_l, z_scale, sigma_F, kappa_F, p_inf, rho_split, alpha, c, beta, l0, G_inf, p_anchor, K_lat]
Liquidity (anchored placement share p_anchor within K_lat ticks, side skew) centres on A = F + P; market orders do not
chase A (no flow feedback loop).
usage: e1_prop.py <seed> <n_trials> <out.jsonl>   (random search; robust two-set scoring with ROBUST=1)
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest.sim import world as W, campaign as C
from almagest.experiments.e1_calibrate_world import loss

LO = np.array([0.0, 0.2, 1.0, 0.05, 0.2, 0.05, 0.5, 1.3, 0.3, 0.35, 5.0, 0.40, 0.2, 2.0])
HI = np.array([0.6, 1.5, 6.0, 0.40, 1.0, 0.25, 0.9, 1.7, 4.0, 0.65, 200.0, 0.75, 0.8, 20.0])
# v2 ranges (2026-11-26): slow informed correction (relaxation is set by how fast informed traders arbitrage
# uninformed impact away; 2/3 permanence after one duration needs a ~2.5-duration correction timescale)


def make_prop(theta):
    km, kl, zs, sF, kF, pinf, rho, alpha, c, beta, l0, ginf, pa, klat = theta
    p = W.default_params()
    p[5], p[4], p[9], p[23], p[24], p[25], p[6], p[7] = km, kl, zs, sF, kF, pinf, rho, alpha
    p[26] = 0.0; p[27] = 1e4; p[28] = pa; p[29] = klat
    W.set_propagator(p, c, beta, l0, ginf)
    return p


def score(theta, nd=16, nw=32):
    p = make_prop(theta)
    bases = (1000, 1032) if os.environ.get('ROBUST') else (1000,)
    rs = [C.evaluate(C.world_runner(p), p, n_days=nd, win_per_day=nw, base_seed=b, B=40, verbose=False) for b in bases]
    Ls = [loss(r) for r in rs]
    i = int(np.argmax(Ls))
    return Ls[i], rs[i]


if __name__ == '__main__':
    seed, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rng = np.random.default_rng(seed)
    with open(out, 'a') as f:
        for t in range(n):
            th = LO + rng.random(len(LO)) * (HI - LO)
            t0 = time.time()
            L, r = score(th)
            rec = dict(theta=th.tolist(), loss=L, **{k: float(r.get(k, np.nan)) for k in
                       ('delta', 'Y', 'ep100', 'ep', 'slope100', 'H', 'gamma', 'vb', 'exc_F', 'exc_r')}, sec=time.time() - t0)
            f.write(json.dumps(rec) + '\n'); f.flush()
            print(t, round(L, 1), {k: round(rec[k], 3) for k in ('delta', 'Y', 'ep100', 'H', 'gamma', 'exc_F')}, flush=True)
