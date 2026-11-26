"""E1a: search World physics for a ground-truth market that PASSES the LOB-Impact ladder at the benchmark
operating point (eta=10%, N=100, aggressive children): delta~0.5, Y in [0.5,1], relaxation to ~1/2-2/3,
diffusive price (H~0.5), persistent order flow (gamma in [0.3,0.8]).
Calibration uses days 1000..1007 only; the confirmation run (e1_confirm) uses disjoint days 5000+.
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest.sim import world as W, campaign as C



def make(theta):
    p = W.default_params()
    km, kl, zs, bw, rho, alpha, sF, kF, pinf, lK, ltK, pa, kl8 = theta
    p[5], p[4], p[9], p[6], p[7] = km, kl, zs, rho, alpha
    p[23], p[24], p[25], p[26], p[27] = sF, kF, pinf, lK, 10 ** ltK
    p[28], p[29] = pa, kl8
    taus = p[17:23]
    w = taus ** (-bw)
    p[11:17] = w / w.sum()
    return p


def loss(r):
    if not np.isfinite(r['delta']) or not np.isfinite(r.get('ep100', np.nan)):
        return 1e9
    L = ((r['delta'] - 0.5) / 0.05) ** 2 + (np.log(max(r['Y'], 1e-3) / 0.7) / 0.3) ** 2
    L += ((r['ep100'] - 0.67) / float(__import__('os').environ.get('EP100_SIGMA', 0.08))) ** 2 + ((r['ep'] - 0.5) / 0.25) ** 2 + ((r['H'] - 0.5) / 0.03) ** 2
    L += (max(0, 0.3 - r['gamma']) / 0.05) ** 2 + (max(0, r['gamma'] - 0.8) / 0.05) ** 2
    L += (max(0, r['slope100']) / 1.0) ** 2
    # realism: bounded hidden-state excursions (perverse responses arise in extreme states; found 2026-11-26)
    L += (max(0, r.get('exc_F', 0) - 0.25) / 0.05) ** 2 + (max(0, r.get('exc_r', 0) - 1.0) / 0.2) ** 2
    return float(L)


def main():
  rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
  n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 40
  out_path = sys.argv[3] if len(sys.argv) > 3 else 'e1_calib.jsonl'
  with open(out_path, 'a') as f:
    for t in range(n_trials):
        theta = [rng.uniform(0.1, 3.0), rng.uniform(0.0, 1.0), rng.uniform(1.0, 12.0),
                 rng.uniform(-0.6, 1.0), rng.uniform(0.4, 0.95), rng.uniform(1.2, 1.9),
                 rng.uniform(0.0, float(os.environ.get('SF_MAX', 0.6))),
                 rng.uniform(float(os.environ.get('KF_MIN', 0.1)), float(os.environ.get('KF_MAX', 2.0))),
                 rng.uniform(float(os.environ.get('PINF_MIN', 0.0)), float(os.environ.get('PINF_MAX', 0.6))),
                 rng.uniform(0.0, 1.0), rng.uniform(3.5, 6.5),
                 rng.uniform(0.0, 0.9), rng.uniform(2.0, 30.0)]
        p = make(theta)
        t0 = time.time()
        r = C.evaluate(C.world_runner(p), p, n_days=16, win_per_day=32, B=60, verbose=False)
        rec = dict(theta=theta, delta=r['delta'], Y=r['Y'], ep=r['ep'], ep100=r['ep100'], slope=r['slope'], H=r['H'],
                   gamma=r['gamma'], vb=r['vb'], loss=loss(r), sec=time.time() - t0)
        f.write(json.dumps({k: (float(v) if not isinstance(v, list) else v) for k, v in rec.items()}) + '\n')
        f.flush()
        print(t, np.round(theta, 2), {k: round(float(rec[k]), 3) for k in ('delta', 'Y', 'ep100', 'ep', 'H', 'gamma', 'loss')})


if __name__ == '__main__':
    main()
