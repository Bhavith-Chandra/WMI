"""E1d: trajectory exponent (LOB-Impact, within one 100-child metaorder) vs cross-sectional exponent (empirical
square-root-law convention: FINAL impact of completed metaorders of different sizes Q at fixed participation).
Ground-truth World, dev days only (no selection made on this output). usage: e1_delta_xsec.py <world.json>"""
import sys, json
import numpy as np
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest import splits as S
from almagest.sim import world as W, campaign as C, metrics as M
from almagest.experiments.e1_calibrate_world import make

par = make(json.load(open(sys.argv[1]))['theta'])
nd, wpd, rep = 16, 30, 4
days = np.arange(S.DEV_BASE + 40, S.DEV_BASE + 40 + nd)          # dev days not used by dev certification (5000-5031)
stats = C.day_stats(par, days)
phi = np.mean([s['phi'] for s in stats]); child = int(round(np.median([s['med'] for s in stats])))
gap = int(round(9 / phi))
sched = C.schedule(nd, wpd, int(days[0]), replicas=rep)
ds, nw, rs, did = sched
V = np.array([stats[d]['V'] for d in did], float); sig = np.array([stats[d]['sigma'] for d in did], float)
snap = W.Snapshots(par, np.unique(ds), np.unique(nw))
Ns = [5, 10, 20, 40, 70, 100, 150]
fin, qf = [], []
for N in Ns:
    ob, m0b, _, _ = snap.run(ds, nw, rs, N, gap, 0, 0, child, 1)
    os_, m0s, _, _ = snap.run(ds, nw, rs, N, gap, 0, 1, child, 1)
    I = 0.5 * ((ob - m0b[:, None]) - (os_ - m0s[:, None])) / sig[:, None]
    fin.append(I[:, -1]); qf.append(N * child / V)
    if N == 100:
        traj = I; qtraj = (np.arange(1, 101)[None, :] * child) / V[:, None]
fin = np.array(fin); qf = np.array(qf)
mi = fin.mean(1); mq = qf.mean(1)
xs = np.polyfit(np.log(mq), np.log(np.maximum(mi, 1e-9)), 1)[0]
dt, Yt = M.law_from_pairs(traj, qtraj)
rng = np.random.default_rng(0); ud = np.unique(did); boot = []
for _ in range(300):
    pick = np.concatenate([np.where(did == d)[0] for d in rng.choice(ud, len(ud))])
    m = fin[:, pick].mean(1)
    if (m > 0).all():
        boot.append(np.polyfit(np.log(qf[:, pick].mean(1)), np.log(m), 1)[0])
print(json.dumps(dict(trajectory_delta=dt, cross_sectional_delta=xs, xsec_ci95=list(np.quantile(boot, [0.025, 0.975])),
                      final_impact_by_N={int(n): float(v) for n, v in zip(Ns, mi)}, n_rollouts=int(len(ds))), default=float))
