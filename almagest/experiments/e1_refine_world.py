"""E1b: (mu,lambda)-style local refinement of World physics from the best random-search points.
Calibration days 1000..1007 only (disjoint from confirmation days)."""
import sys, json
import numpy as np
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest.sim import campaign as C
from almagest.experiments.e1_calibrate_world import make, loss  # noqa: E402  (module guarded below)

LO = np.array([0.05, 0.0, 0.5, -0.8, 0.3, 1.15, 0.0, 0.05, 0.0, 0.0, 3.0, 0.0, 2.0])
HI = np.array([4.0, 1.0, 14.0, 1.2, 0.97, 1.95, 0.8, 3.0, 0.8, 0.5, 5.7, 0.95, 40.0])


def score(theta, nd=12, nw=12):
    p = make(theta)
    r = C.evaluate(C.world_runner(p), p, n_days=nd, win_per_day=nw, B=40, verbose=False)
    return loss(r), r


if __name__ == '__main__':
    src, out_path, gens, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    recs = [json.loads(l) for l in open(src)]
    recs.sort(key=lambda r: r['loss'])
    rng = np.random.default_rng(seed)
    pop = [(r['loss'], np.array(r['theta'])) for r in recs[:4]]
    step = 0.15 * (HI - LO)
    with open(out_path, 'a') as f:
        for g in range(gens):
            pop.sort(key=lambda x: x[0])
            parents = pop[:3]
            kids = []
            for i in range(6):
                th = parents[i % 3][1] + rng.normal(0, 1, len(LO)) * step
                th = np.clip(th, LO, HI)
                L, r = score(th)
                kids.append((L, th))
                f.write(json.dumps(dict(gen=g, theta=th.tolist(), loss=L, delta=float(r['delta']), Y=float(r['Y']),
                                        ep=float(r['ep']), slope=float(r['slope']), H=float(r['H']),
                                        gamma=float(r['gamma']))) + '\n')
                f.flush()
            pop = sorted(parents + kids, key=lambda x: x[0])[:4]
            step *= 0.85
            b = pop[0]
            print(g, round(b[0], 2), np.round(b[1], 3), flush=True)
