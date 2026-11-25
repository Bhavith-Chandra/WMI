"""Capture the real episode state at t=40 under each planner, then replan from it with BOTH models."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_mujoco_v3 import BASE, build
from wmi.mujoco_v3 import Planner3, make_schedule3
from wmi.mujoco_exp import PuckBatch
from wmi.harness import set_seed
rng = set_seed(99); cfg = dict(BASE, K=20, H=25, E=40, beta=10.0); Mu, Mc, _, _ = build(cfg, rng)
E = cfg["E"]; g, s = make_schedule3("none", np.random.default_rng(9_999), E, 160)
phi = s[0]
def run_to(k, T):
    env = PuckBatch(E); st = env.reset(); pl = Planner3(k, Mu, Mc, E, cfg, np.random.default_rng(7))
    for t in range(T):
        env.set_phi(s[t]); pl.phi_true = s[t]
        if t % 20 == 0: plan = pl.act(st, g[t])
        st = env.step(np.clip(plan[:, t % 20], -1, 1))
    return st
def true_cost(start, plan, goal):
    env = PuckBatch(E); env.set_phi(phi); env.reset(start[:, :2])
    for e in range(E): env.d[e].qvel[:] = start[e, 2:]
    c = np.zeros(E)
    for t in range(20):
        a = np.clip(plan[:, t], -1, 1); st = env.step(a); c += ((st[:, :2] - goal) ** 2).sum(1) + 0.01 * (a ** 2).sum(1)
    return c
for who in ("forward", "oracle_latent"):
    st40 = run_to(who, 40)
    print(f"state at t=40 under {who}: |pos-oldgoal| {np.linalg.norm(st40[:, :2]-g[39], axis=1).mean():.4f}  |v| {np.linalg.norm(st40[:, 2:], axis=1).mean():.4f}")
    for k in ("forward", "oracle_latent"):
        pl = Planner3(k, Mu, Mc, E, cfg, np.random.default_rng(7)); pl.phi_true = phi
        print(f"   replan with {k:14s}: TRUE 20-step cost {true_cost(st40, pl.act(st40, g[40]), g[40]).mean():.3f}")
