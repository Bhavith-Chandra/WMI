"""E1c: confirm a calibrated World on DISJOINT days (5000+), full campaign size (20 days x 16 windows)."""
import sys, json
import numpy as np
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest.sim import campaign as C
from almagest.experiments.e1_calibrate_world import make

theta = json.loads(sys.argv[1])
base = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
p = make(theta)
r = C.evaluate(C.world_runner(p), p, n_days=32, win_per_day=32, base_seed=base, B=300, verbose=True)
ref = dict(gamma=r['gamma'], H=r['H'])
print('ladder(self-ref)', C.M.ladder(ref, r))
print('curve q,I:', [(float(np.round(a, 5)), float(np.round(b, 3))) for a, b in zip(r['q'][[0, 9, 49, 99]], r['curve'][[0, 9, 49, 99]])])
print('relax:', np.round(r['relax_curve'][[9, 19, 29, 59, 109]], 3))
np.save('world_confirm_%d.npy' % base, dict(theta=theta, **{k: v for k, v in r.items() if k != 'ladder'}), allow_pickle=True)
