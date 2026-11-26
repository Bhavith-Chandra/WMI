"""E-H: hierarchical navigation through gated hazard walls under a hidden persistent latent z = (gain, wind).

A high level supplies subgoals (approach/exit points of each gate, then the goal), in the role of the top level of a
hierarchical world-model planner. A low level tracks the current subgoal with CEM-MPC on a learned ensemble world model,
executing open-loop chunks of K steps, with a hazard penalty in its rollouts. Touching a hazard is an irreversible
failure of the current subgoal and therefore of the main goal. 'flat' planners skip the high level and plan to the goal.
Hindsight enters through the low-level predictor: every subgoal shares one belief q(z) (Planner3 kinds).
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch

from .mujoco_exp import PuckBatch, sample_phi
from .mujoco_v3 import Planner3

START = np.array([0.0, 0.0]); GOAL = np.array([4.0, 0.0])
WALLS = [(1.0, 0.5), (2.0, -0.5), (3.0, 0.5)]           # (x of wall, y of gap centre)
WALL_HALF, YLIM, REACH, GOAL_R = 0.15, 1.2, 0.2, 0.25
TASK = {"gap": 0.5}                                     # set by the gate ladder (pre-registered), never by outcomes


def subgoals():
    out = []
    for xw, yc in WALLS:
        out += [(xw - 0.35, yc), (xw + 0.35, yc)]
    return np.array(out + [tuple(GOAL)])


def hazard_np(p):
    x, y = p[..., 0], p[..., 1]
    h = np.abs(y) > YLIM
    for xw, yc in WALLS:
        h |= (np.abs(x - xw) < WALL_HALF) & (np.abs(y - yc) > TASK["gap"] / 2)
    return h


def hazard_t(p):
    x, y = p[..., 0], p[..., 1]
    h = y.abs() > YLIM
    for xw, yc in WALLS:
        h = h | (((x - xw).abs() < WALL_HALF) & ((y - yc).abs() > TASK["gap"] / 2))
    return h


@torch.no_grad()
def act_hazard(pl, s, goal, hz=50.0):
    """Planner3.act with a hazard penalty on predicted positions (and segment midpoints, against tunnelling)."""
    c = pl.cfg; E, H, P = pl.E, c["H"], c["pop"]
    std = np.full((E, H, 2), 0.6); g = torch.tensor(goal, dtype=torch.float32)
    for _ in range(c["iters"]):
        A = np.clip(pl.mu[:, None] + std[:, None] * pl.rng.standard_normal((E, P, H, 2)), -1, 1)
        At = torch.tensor(A, dtype=torch.float32)
        st = torch.tensor(s, dtype=torch.float32)[:, None].expand(-1, P, -1)
        cost = torch.zeros(E, P); prev = st[..., :2]; dead = torch.zeros(E, P, dtype=torch.bool)
        for t in range(H):
            d, dsd = pl.predict(st[..., 2:], At[:, :, t])
            st = st + d; p = st[..., :2]
            dead = dead | hazard_t(p) | hazard_t(0.5 * (p + prev)); prev = p
            cost += ((p - g[:, None]) ** 2).sum(-1) + 0.01 * (At[:, :, t] ** 2).sum(-1) \
                + c["beta"] * ((dsd / pl.rsd) ** 2).mean(-1) + hz * dead.float()
        el = torch.topk(cost, c["elites"], dim=1, largest=False).indices.numpy()
        EA = np.take_along_axis(A, el[:, :, None, None], 1)
        pl.mu, std = EA.mean(1), EA.std(1) + 0.05
    plan, K = pl.mu.copy(), c["K"]
    pl.mu = np.concatenate([pl.mu[:, K:], np.zeros((E, K, 2))], 1)
    return plan


def apply_physics(env, mass, damp):
    """Per-episode mass and damping multipliers. mj_setConst is REQUIRED: without it MuJoCo keeps the compiled mass
    matrix and a body_mass change is silently ignored (caught by tests/test_physics.py)."""
    for e in range(env.E):
        env.m[e].body_mass[env.body] *= mass[e]; env.m[e].dof_damping[:] *= damp[e]
        mujoco.mj_setConst(env.m[e], env.d[e])


def schedule(regime, rng, E, T, every=20):
    """Latent per step, (T, E, 3). none/wind/both persist for the whole episode; fresh redraws every `every` steps."""
    sched = np.zeros((T, E, 3)); sched[..., 0] = 1.0
    if regime == "none":
        return sched
    if regime in ("wind", "both"):
        ph = sample_phi(rng, E)
        if regime == "wind":
            ph[:, 0] = 1.0
        sched[:] = ph[None]
        return sched
    if regime == "fresh":
        for t0 in range(0, T, every):
            sched[t0:t0 + every] = sample_phi(rng, E)[None]
        return sched
    raise ValueError(regime)


def run_hier(planner, Mu, Mc, sched, cfg, seed, n_traj=6, world=None):
    """planner: '<flat|hier>:<Planner3 kind>'. Returns per-episode outcome arrays and a few trajectories.

    world (deployment realism, all optional):
      mass, damp: per-episode multipliers (E,) on body mass and joint damping -- physics the model never saw;
      obs_sd: (pos_sd, vel_sd) Gaussian sensor noise on everything the planner sees (act and observe);
      act_delay: control steps between command and actuation;
      compute_steps: control steps a decision takes in wall-clock (measured); the previous plan keeps executing
                     meanwhile, and the new plan starts from the state observed when planning began;
      seed: RNG for sensor noise.
    Hazards and success are always judged on the TRUE state."""
    mode, kind = planner.split(":")
    T, E = sched.shape[:2]; K = cfg["K"]; W = world or {}
    env = PuckBatch(E)
    if "mass" in W:
        apply_physics(env, W["mass"], W["damp"])
    s = env.reset(np.tile(START, (E, 1)))
    nrng = np.random.default_rng(W.get("seed", 0)); osd = np.array(list(W.get("obs_sd", (0.0, 0.0))))
    sensor = lambda x: x + (np.r_[osd[0], osd[0], osd[1], osd[1]] * nrng.standard_normal(x.shape) if osd.any() else 0.0)
    d_act, L = int(W.get("act_delay", 0)), int(W.get("compute_steps", 0))
    queue = [np.zeros((E, 2))] * d_act; o = sensor(s); pending = None
    pl = Planner3(kind, Mu, Mc, E, cfg, np.random.default_rng(seed))
    SG = subgoals(); idx = np.zeros(E, int) if mode == "hier" else np.full(E, len(SG) - 1)
    done = np.zeros(E, bool); success = np.zeros(E, bool); fail = np.zeros(E, bool)
    t_done = np.full(E, T); fail_sg = np.full(E, -1); traj = np.full((T + 1, min(n_traj, E), 2), np.nan)
    traj[0] = s[:n_traj, :2]
    plan, k0 = np.zeros((E, cfg["H"], 2)), 0
    for t in range(T):
        env.set_phi(sched[t]); pl.phi_true = sched[t]
        if t % K == 0:
            new = act_hazard(pl, o, SG[idx])
            if L == 0:
                plan, k0 = new, t
            else:
                pending = (new, t + L)
        if pending is not None and t == pending[1]:
            plan, k0 = pending[0], t - L; pending = None          # plan was made for the state at t - L
        cmd = np.clip(plan[:, min(t - k0, plan.shape[1] - 1)], -1, 1)
        queue.append(cmd); a = queue.pop(0)
        s2 = env.step(a)
        live = ~done
        o2 = sensor(s2)
        pl.observe(o, cmd, o2)                                     # the planner knows what it commanded, not what was applied
        p0, p1 = s[:, :2], s2[:, :2]
        hit = np.zeros(E, bool)
        for f in np.linspace(0, 1, 6)[1:]:                     # check the swept segment, not only the endpoint
            hit |= hazard_np(p0 + f * (p1 - p0))
        newfail = live & hit
        fail |= newfail; fail_sg[newfail] = idx[newfail]
        t_done[newfail] = t; done |= newfail
        live = ~done
        dist = np.linalg.norm(p1 - SG[idx], axis=1)
        at_goal = live & (idx == len(SG) - 1) & (np.linalg.norm(p1 - GOAL, axis=1) < GOAL_R)
        success |= at_goal; t_done[at_goal] = t; done |= at_goal
        adv = (~done) & (idx < len(SG) - 1) & (dist < REACH)
        idx[adv] += 1
        s, o = s2, o2                                              # finished episodes keep stepping; outcomes are frozen
        traj[t + 1] = np.where(done[:n_traj, None], traj[t], s[:n_traj, :2])
    return {"success": success.astype(int).tolist(), "fail": fail.astype(int).tolist(), "t_done": t_done.tolist(),
            "fail_subgoal": fail_sg.tolist(), "subgoals_reached": idx.tolist(), "traj": np.round(traj, 3).tolist()}
