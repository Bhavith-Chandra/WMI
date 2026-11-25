"""Single-plan test from a goal-change state (puck at rest at old goal, new goal far away)."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_mujoco_v3 import BASE, build
from wmi.mujoco_v3 import Planner3
from wmi.mujoco_exp import PuckBatch
from wmi.harness import set_seed
rng = set_seed(99); cfg = dict(BASE, K=20, H=25, E=40, beta=10.0); Mu, Mc, _, _ = build(cfg, rng)
E = cfg["E"]; r = np.random.default_rng(4); phi = np.tile([1.0, 0, 0], (E, 1))
old = r.uniform(-1, 1, (E, 2)); goal = r.uniform(-1, 1, (E, 2)); start = np.c_[old, np.zeros((E, 2))]
def true_traj(plan, n=25):
    env = PuckBatch(E); env.set_phi(phi); env.reset(old); c = np.zeros(E); S = []
    for t in range(n):
        a = np.clip(plan[:, t], -1, 1); s = env.step(a); S.append(s); c += ((s[:, :2] - goal) ** 2).sum(1) + 0.01 * (a ** 2).sum(1)
    return c, np.stack(S, 1)
@torch.no_grad()
def model_traj(pl, plan, n=25):
    st = torch.tensor(start, dtype=torch.float32)[:, None]; g = torch.tensor(goal, dtype=torch.float32); c = torch.zeros(E, 1); S = []
    for t in range(n):
        a = torch.tensor(np.clip(plan[:, t], -1, 1), dtype=torch.float32)[:, None]
        d, _ = pl.predict(st[..., 2:], a); st = st + d; S.append(st[:, 0].numpy())
        c += ((st[..., :2] - g[:, None]) ** 2).sum(-1) + 0.01 * (a ** 2).sum(-1)
    return c[:, 0].numpy(), np.stack(S, 1)
plans, pls = {}, {}
for k in ("forward", "oracle_latent"):
    pl = Planner3(k, Mu, Mc, E, cfg, np.random.default_rng(7)); pl.phi_true = phi; plans[k] = pl.act(start, goal); pls[k] = pl
for k, p in plans.items():
    tc, TS = true_traj(p); mc, MS = model_traj(pls["oracle_latent"], p); fc, FS = model_traj(pls["forward"], p)
    print(f"{k:14s} 25-step TRUE {tc.mean():.3f} | Mc {mc.mean():.3f} | Mu {fc.mean():.3f} | Mc pos err@25 {np.linalg.norm(MS[:, -1, :2]-TS[:, -1, :2], axis=1).mean():.4f} | max |v| true {np.abs(TS[..., 2:]).max():.2f}")
