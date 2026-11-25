"""Latent MPC with categorical CEM, and evaluation against oracle shortest paths.

NOTE: `cost="hindsight"` uses a temporal-distance function learned by hindsight
goal relabeling (train.train_temporal_distance). This is a *working definition*
of latent hindsight planning for the scaffold, to be replaced by the
collaborators' definition.
"""
from __future__ import annotations

import numpy as np
import torch

from .env import N_ACTIONS, KeyGrid
from .harness.stats import wilson_ci


@torch.no_grad()
def cem_first_action(wm, cost_fn, z0, zg, horizon, rng, pop=256, elites=32, iters=4, mix=0.05):
    probs = np.full((horizon, N_ACTIONS), 1.0 / N_ACTIONS)
    best_seq, best_cost = None, np.inf
    for _ in range(iters):
        cum = probs.cumsum(1)
        u = rng.random((pop, horizon, 1))
        seqs = (u > cum[None]).sum(-1).clip(0, N_ACTIONS - 1)
        z = z0.expand(pop, -1)
        g = zg.expand(pop, -1)
        cost = torch.full((pop,), float("inf"))
        a_t = torch.from_numpy(seqs)
        for t in range(horizon):
            z = wm.predict(z, a_t[:, t])
            cost = torch.minimum(cost, cost_fn(z, g) + 1e-3 * t)
        cost = cost.numpy()
        order = np.argsort(cost, kind="stable")
        if cost[order[0]] < best_cost:
            best_cost, best_seq = cost[order[0]], seqs[order[0]]
        el = seqs[order[:elites]]
        counts = np.stack([(el == k).mean(0) for k in range(N_ACTIONS)], 1)
        probs = (1 - mix) * counts + mix / N_ACTIONS
    return int(best_seq[0])


def run_episode(env: KeyGrid, policy, start: np.ndarray, goal_xy, max_steps: int):
    s = start.copy()
    for step in range(max_steps):
        if s[0] == goal_xy[0] and s[1] == goal_xy[1]:
            return True, step
        a = policy(s)
        s = env.step(s[None], np.array([a]))[0]
    return bool(s[0] == goal_xy[0] and s[1] == goal_xy[1]), max_steps


def sample_tasks(env: KeyGrid, rng, n, min_d=3, max_d=12):
    tasks = []
    while len(tasks) < n:
        s = env.sample_states(rng, 1, p_has_key=1.0)[0]
        g = env.free_cells[rng.integers(len(env.free_cells))]
        d = env.shortest_path_len(s[:2], g)
        if min_d <= d <= max_d:
            tasks.append((s, g, d))
    return tasks


def evaluate(env, make_policy, tasks, max_steps):
    succ, excess = [], []
    for s, g, d in tasks:
        ok, steps = run_episode(env, make_policy(g), s, g, max_steps)
        succ.append(ok)
        if ok:
            excess.append(steps - d)
    k = int(np.sum(succ))
    return {"success_rate": k / len(tasks), "success_ci95": list(wilson_ci(k, len(tasks))), "n": len(tasks),
            "mean_excess_steps_on_success": float(np.mean(excess)) if excess else None}
