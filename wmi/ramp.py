"""E-R2 (v11): subgoal redrawing when the latent decides what is physically possible (ramp task).

Derived from wmi/fork.py. Route A crosses a ramp (a known map feature: a region with a constant opposing force F_s
that the world model does not know); it is short, but impossible when the motor is too weak. Route B is a flat detour
around a central block: always possible, slower. Primary metric: time-discounted success, success * (1 - t / T).

Original fork docstring follows.
E-R (v10): subgoal redrawing from the hindsight posterior.

One hazard wall at x = 1.5 with two openings: a NARROW gap on the direct line (route A, short) and a WIDE gap far to
the side (route B, long). A strong crosswind makes route A crash; a weak motor and a headwind make route B time out.
Which route is right depends on the hidden latent. All planners first reach a common fork point; there the HIGH LEVEL
chooses the route:
  'A' / 'B'  fixed route;
  'prior'    the route with the higher calibrated success averaged over the prior (a fixed choice);
  'post'     the route with the higher calibrated success given the hindsight belief at the fork (redrawing);
  'oracle'   the route with the higher calibrated success given the true latent.
The LOW LEVEL is a Planner3 kind (forward, hindsight_posterior, oracle_latent). A shadow posterior tracks the belief
for every planner, so a forward low level can still be paired with a posterior high level.
Route-success models are fitted on calibration episodes only (scripts/run_fork.py calibrate).
"""
from __future__ import annotations

import numpy as np
import torch

from . import hier
from .mujoco_exp import PuckBatch
from .mujoco_v3 import Planner3

START, GOAL, FORK = np.array([0.0, 0.0]), np.array([3.0, 0.0]), np.array([0.7, 0.0])
BLOCK = (1.1, 1.9, -0.8, 0.8)          # central hazard block (x0, x1, y0, y1)
CORR = 0.25                            # half-width of the ramp corridor through the block (route A)
YLIM = 1.5
TASK = {"F_s": 3.5, "T": 160}          # ramp force (N) and time limit (steps), set by the gate ladder


def route_subgoals(r):
    if r == "A":
        return np.array([[BLOCK[0] - 0.25, 0.0], [BLOCK[1] + 0.25, 0.0], GOAL])
    yb = 0.5 * (BLOCK[2] - YLIM)       # middle of the lower passage
    return np.array([[BLOCK[0] - 0.25, yb], [BLOCK[1] + 0.25, yb], GOAL])


def hazard_np(p):
    x, y = p[..., 0], p[..., 1]
    in_block = (x > BLOCK[0]) & (x < BLOCK[1]) & (y > BLOCK[2]) & (y < BLOCK[3]) & (np.abs(y) >= CORR)
    return (np.abs(y) > YLIM) | in_block


def hazard_t(p):
    x, y = p[..., 0], p[..., 1]
    in_block = (x > BLOCK[0]) & (x < BLOCK[1]) & (y > BLOCK[2]) & (y < BLOCK[3]) & (y.abs() >= CORR)
    return (y.abs() > YLIM) | in_block


def ramp_force(pos):
    """Opposing force (-x) on the ramp: the corridor through the block. Unknown to the world model."""
    x, y = pos[:, 0], pos[:, 1]
    on = (x > BLOCK[0]) & (x < BLOCK[1]) & (np.abs(y) < CORR)
    return np.where(on, -TASK["F_s"], 0.0)


def belief_features(post):
    """Hindsight belief at the fork: posterior mean of the multiplicative (theta) and additive (w) corrections on the
    two velocity dimensions, the identifiable quantities (Section 'what the latent represents')."""
    m = post.mean()                                  # (E,4,2)
    return np.stack([m[:, 2, 0], m[:, 3, 0], m[:, 2, 1], m[:, 3, 1]], 1)


def feat_expand(f):
    """Quadratic features for the route-success logistic models (same for belief and true-latent inputs)."""
    f = np.asarray(f, float)
    sq = f[:, :, None] * f[:, None, :]
    iu = np.triu_indices(f.shape[1])
    return np.c_[np.ones(len(f)), f, sq[:, iu[0], iu[1]]]


class RouteModel:
    """P(success | features) per route: L2-regularised logistic regression, fitted on calibration episodes."""

    def __init__(self, lam=1.0):
        self.lam, self.w, self.mu, self.sd = lam, {}, None, None

    def fit(self, F, succ):          # F (n,d); succ: {route: (n,)}
        X = feat_expand(F); self.mu, self.sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-8
        X = np.c_[X[:, :1], (X[:, 1:] - self.mu) / self.sd]
        for r, y in succ.items():
            w = np.zeros(X.shape[1])
            for _ in range(500):                      # Newton steps
                p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) + self.lam * np.r_[0, w[1:]]
                H = X.T @ (X * (p * (1 - p))[:, None]) + self.lam * np.diag(np.r_[0, np.ones(len(w) - 1)])
                step = np.linalg.solve(H, g); w -= step
                if np.abs(step).max() < 1e-8:
                    break
            self.w[r] = w
        return self

    def predict(self, F):
        X = feat_expand(F); X = np.c_[X[:, :1], (X[:, 1:] - self.mu) / self.sd]
        return {r: 1 / (1 + np.exp(-X @ w)) for r, w in self.w.items()}

    def state(self):
        return {"lam": self.lam, "w": {r: w.tolist() for r, w in self.w.items()}, "mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def load(cls, d):
        m = cls(d["lam"]); m.w = {r: np.array(w) for r, w in d["w"].items()}; m.mu, m.sd = np.array(d["mu"]), np.array(d["sd"])
        return m


def choose(hl, E, feats_belief, z_true, models, b_penalty=0.0):
    """Route per episode: the higher predicted score (A on ties)."""
    if hl in ("A", "B"):
        return np.array([hl] * E)
    if hl == "prior":
        return np.array([models["prior_choice"]] * E)
    p = models["belief"].predict(feats_belief) if hl == "post" else models["latent"].predict(z_true)
    return np.where(p["A"] >= p["B"] - b_penalty, "A", "B")


def run_ramp(ll, hl, Mu, Mc, sched, cfg, seed, models=None, b_penalty=0.0, n_traj=6, world=None):
    """ll: Planner3 kind for the low level; hl: high-level rule. Returns outcomes, routes, belief and latent at the fork.
    world: deployment realism as in hier.run_hier (mass, damp, obs_sd, act_delay, compute_steps, seed)."""
    T, E = sched.shape[:2]; K = cfg["K"]; W = world or {}
    env = PuckBatch(E)
    if "mass" in W:
        hier.apply_physics(env, W["mass"], W["damp"])
    s = env.reset(np.tile(START, (E, 1)))
    nrng = np.random.default_rng(W.get("seed", 0)); osd = np.array(list(W.get("obs_sd", (0.0, 0.0))))
    sensor = lambda x: x + (np.r_[osd[0], osd[0], osd[1], osd[1]] * nrng.standard_normal(x.shape) if osd.any() else 0.0)
    d_act, L = int(W.get("act_delay", 0)), int(W.get("compute_steps", 0))
    queue = [np.zeros((E, 2))] * d_act; o = sensor(s); pending = None; plan, k0 = np.zeros((E, cfg["H"], 2)), 0
    pl = Planner3(ll, Mu, Mc, E, cfg, np.random.default_rng(seed))
    shadow = pl if ll == "hindsight_posterior" else Planner3("hindsight_posterior", Mu, Mc, E, cfg, np.random.default_rng(seed + 1))
    goal = np.tile(FORK, (E, 1)); sgs = [None] * E; idx = np.zeros(E, int)
    decided = np.zeros(E, bool); routes = np.array([""] * E, dtype=object)
    fb = np.zeros((E, 4)); zf = np.zeros((E, 3)); t_dec = np.full(E, -1)
    done = np.zeros(E, bool); success = np.zeros(E, bool); fail = np.zeros(E, bool); t_done = np.full(E, T)
    traj = np.full((T + 1, min(n_traj, E), 2), np.nan); traj[0] = s[:n_traj, :2]
    for t in range(T):
        env.set_phi(sched[t]); pl.phi_true = sched[t]; shadow.phi_true = sched[t]
        rf = ramp_force(s[:, :2])
        for e in np.flatnonzero(rf != 0):
            env.d[e].xfrc_applied[env.body, 0] += rf[e]
        # high-level decision at the fork (or at the latest after 3 s)
        ready = (~decided) & ((np.linalg.norm(o[:, :2] - FORK, axis=1) < hier.REACH) | (t >= 60))
        if ready.any():
            fb[ready] = belief_features(shadow.post)[ready]; zf[ready] = sched[t][ready]; t_dec[ready] = t
            ch = choose(hl, E, fb, zf, models, b_penalty)
            for e in np.flatnonzero(ready):
                routes[e] = ch[e]; sgs[e] = route_subgoals(ch[e]); goal[e] = sgs[e][0]; decided[e] = True
        if t % K == 0:
            new = hier.act_hazard(pl, o, goal, haz=hazard_t)
            if L == 0:
                plan, k0 = new, t
            else:
                pending = (new, t + L)
        if pending is not None and t == pending[1]:
            plan, k0 = pending[0], t - L; pending = None
        cmd = np.clip(plan[:, min(t - k0, plan.shape[1] - 1)], -1, 1)
        queue.append(cmd); a = queue.pop(0)
        s2 = env.step(a); o2 = sensor(s2)
        pl.observe(o, cmd, o2)
        if shadow is not pl:
            shadow.observe(o, cmd, o2)
        live = ~done; hit = np.zeros(E, bool)
        for f in np.linspace(0, 1, 6)[1:]:
            hit |= hazard_np(s[:, :2] + f * (s2[:, :2] - s[:, :2]))
        nf = live & hit; fail |= nf; t_done[nf] = t; done |= nf
        for e in np.flatnonzero(decided & ~done):
            if idx[e] == 2 and np.linalg.norm(s2[e, :2] - GOAL) < hier.GOAL_R:
                success[e] = True; t_done[e] = t; done[e] = True
            elif idx[e] < 2 and np.linalg.norm(o2[e, :2] - sgs[e][idx[e]]) < hier.REACH:
                idx[e] += 1; goal[e] = sgs[e][idx[e]]
        s, o = s2, o2
        traj[t + 1] = np.where(done[:n_traj, None], traj[t], s[:n_traj, :2])
    score = success * (1 - t_done / T)
    return {"success": success.astype(int).tolist(), "score": np.round(score, 6).tolist(), "fail": fail.astype(int).tolist(),
            "t_done": t_done.tolist(), "route": routes.tolist(), "t_decide": t_dec.tolist(), "belief": np.round(fb, 6).tolist(),
            "latent": np.round(zf, 6).tolist(), "traj": np.round(traj, 3).tolist()}
