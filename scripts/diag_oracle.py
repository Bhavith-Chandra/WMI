"""Diagnose why the true-phi planner (Mc) underperforms the phi-blind planner (Mu) on 'none'.
For identical start states/goals: plan once with each model (K=20 open loop), then score each plan
(a) in the TRUE simulator and (b) under each model's prediction."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_mujoco_v3 import BASE, build
from wmi.mujoco_v3 import Planner3
from wmi.mujoco_exp import PuckBatch
from wmi.harness import set_seed

rng = set_seed(99)
cfg = dict(BASE, K=20, H=25, E=40, beta=10.0)
Mu, Mc, _, _ = build(cfg, rng)
E = cfg["E"]; r = np.random.default_rng(3)
phi = np.tile([1.0, 0, 0], (E, 1))
start = np.zeros((E, 4)); goal = r.uniform(-1, 1, (E, 2))

def true_cost(plan):
    env = PuckBatch(E); env.set_phi(phi); s = env.reset(); c = np.zeros(E)
    for t in range(20):
        a = np.clip(plan[:, t], -1, 1); s = env.step(a); c += ((s[:, :2] - goal) ** 2).sum(1) + 0.01 * (a ** 2).sum(1)
    return c

@torch.no_grad()
def model_cost(pl, plan):
    st = torch.tensor(start, dtype=torch.float32)[:, None]; g = torch.tensor(goal, dtype=torch.float32); c = torch.zeros(E, 1)
    for t in range(20):
        a = torch.tensor(np.clip(plan[:, t], -1, 1), dtype=torch.float32)[:, None]
        d, _ = pl.predict(st[..., 2:], a); st = st + d
        c += ((st[..., :2] - g[:, None]) ** 2).sum(-1) + 0.01 * (a ** 2).sum(-1)
    return c[:, 0].numpy()

plans, pls = {}, {}
for k in ("forward", "oracle_latent"):
    pl = Planner3(k, Mu, Mc, E, cfg, np.random.default_rng(7)); pl.phi_true = phi
    plans[k] = pl.act(start, goal); pls[k] = pl
for k, p in plans.items():
  if True:
    print(f"plan from {k:14s}: TRUE cost {true_cost(p).mean():.3f} | Mu-model cost {model_cost(pls['forward'], p).mean():.3f} | Mc(true phi) cost {model_cost(pls['oracle_latent'], p).mean():.3f}")
print("mean |a| forward plan", np.abs(plans['forward'][:, :20]).mean().round(3), " oracle plan", np.abs(plans['oracle_latent'][:, :20]).mean().round(3))

# ---- full-episode per-chunk breakdown on 'none'
from wmi.mujoco_v3 import make_schedule3
g, s = make_schedule3("none", np.random.default_rng(9_999), E, 160)
for warm in (True, False):
  cfg["warm"] = warm
  for k in ("forward", "oracle_latent"):
      env = PuckBatch(E); st = env.reset(); pl = Planner3(k, Mu, Mc, E, cfg, np.random.default_rng(7)); cost = np.zeros((160, E))
      for t in range(160):
          env.set_phi(s[t]); pl.phi_true = s[t]
          if t % 20 == 0:
              plan = pl.act(st, g[t])
          a = np.clip(plan[:, t % 20], -1, 1); st2 = env.step(a)
          cost[t] = ((st2[:, :2] - g[t]) ** 2).sum(1) + 0.01 * (a ** 2).sum(1); st = st2
      print("warm" if warm else "cold", k, "per-chunk mean cost:", np.round(cost.reshape(8, 20, E).mean((1, 2)), 3), "total", round(float(cost.mean()), 4))
