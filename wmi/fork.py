"""E-R (v10): subgoal redrawing from the hindsight posterior.

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

START, GOAL, FORK = np.array([0.0, 0.0]), np.array([3.0, 0.0]), np.array([0.6, 0.0])
WALL_X, WALL_HALF, YLIM = 1.5, 0.15, 1.6
TASK = {"narrow": 0.35, "T": 200, "wide_c": 1.05, "wide_w": 0.7}     # narrow width and T set by the gate ladder


def route_subgoals(r):
    if r == "A":
        return np.array([[WALL_X - 0.4, 0.0], [WALL_X + 0.4, 0.0], GOAL])
    c = TASK["wide_c"]
    return np.array([[WALL_X - 0.4, c], [WALL_X + 0.4, c], GOAL])


def _gaps():
    return [(0.0, TASK["narrow"]), (TASK["wide_c"], TASK["wide_w"])]


def hazard_np(p):
    x, y = p[..., 0], p[..., 1]
    in_wall = np.abs(x - WALL_X) < WALL_HALF
    open_ = np.zeros_like(in_wall)
    for c, w in _gaps():
        open_ |= np.abs(y - c) < w / 2
    return (np.abs(y) > YLIM) | (in_wall & ~open_)


def hazard_t(p):
    x, y = p[..., 0], p[..., 1]
    in_wall = (x - WALL_X).abs() < WALL_HALF
    open_ = torch.zeros_like(in_wall)
    for c, w in _gaps():
        open_ = open_ | ((y - c).abs() < w / 2)
    return (y.abs() > YLIM) | (in_wall & ~open_)


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


def choose(hl, E, feats_belief, z_true, models, b_penalty):
    """Route per episode. b_penalty: pre-registered preference for the short route when success is predicted equal."""
    if hl in ("A", "B"):
        return np.array([hl] * E)
    if hl == "prior":
        return np.array([models["prior_choice"]] * E)
    p = models["belief"].predict(feats_belief) if hl == "post" else models["latent"].predict(z_true)
    return np.where(p["A"] >= p["B"] - b_penalty, "A", "B")


def run_fork(ll, hl, Mu, Mc, sched, cfg, seed, models=None, b_penalty=0.05, n_traj=6):
    """ll: Planner3 kind for the low level; hl: high-level rule. Returns outcomes, routes, belief and latent at the fork."""
    T, E = sched.shape[:2]; K = cfg["K"]
    env = PuckBatch(E); s = env.reset(np.tile(START, (E, 1)))
    pl = Planner3(ll, Mu, Mc, E, cfg, np.random.default_rng(seed))
    shadow = pl if ll == "hindsight_posterior" else Planner3("hindsight_posterior", Mu, Mc, E, cfg, np.random.default_rng(seed + 1))
    goal = np.tile(FORK, (E, 1)); sgs = [None] * E; idx = np.zeros(E, int)
    decided = np.zeros(E, bool); routes = np.array([""] * E, dtype=object)
    fb = np.zeros((E, 4)); zf = np.zeros((E, 3)); t_dec = np.full(E, -1)
    done = np.zeros(E, bool); success = np.zeros(E, bool); fail = np.zeros(E, bool); t_done = np.full(E, T)
    traj = np.full((T + 1, min(n_traj, E), 2), np.nan); traj[0] = s[:n_traj, :2]
    for t in range(T):
        env.set_phi(sched[t]); pl.phi_true = sched[t]; shadow.phi_true = sched[t]
        # high-level decision at the fork (or at the latest after 3 s)
        ready = (~decided) & ((np.linalg.norm(s[:, :2] - FORK, axis=1) < hier.REACH) | (t >= 60))
        if ready.any():
            fb[ready] = belief_features(shadow.post)[ready]; zf[ready] = sched[t][ready]; t_dec[ready] = t
            ch = choose(hl, E, fb, zf, models, b_penalty)
            for e in np.flatnonzero(ready):
                routes[e] = ch[e]; sgs[e] = route_subgoals(ch[e]); goal[e] = sgs[e][0]; decided[e] = True
        if t % K == 0:
            plan = hier.act_hazard(pl, s, goal, haz=hazard_t); k0 = t
        a = np.clip(plan[:, t - k0], -1, 1)
        s2 = env.step(a)
        pl.observe(s, a, s2)
        if shadow is not pl:
            shadow.observe(s, a, s2)
        live = ~done; hit = np.zeros(E, bool)
        for f in np.linspace(0, 1, 6)[1:]:
            hit |= hazard_np(s[:, :2] + f * (s2[:, :2] - s[:, :2]))
        nf = live & hit; fail |= nf; t_done[nf] = t; done |= nf
        for e in np.flatnonzero(decided & ~done):
            if idx[e] == 2 and np.linalg.norm(s2[e, :2] - GOAL) < hier.GOAL_R:
                success[e] = True; t_done[e] = t; done[e] = True
            elif idx[e] < 2 and np.linalg.norm(s2[e, :2] - sgs[e][idx[e]]) < hier.REACH:
                idx[e] += 1; goal[e] = sgs[e][idx[e]]
        s = s2
        traj[t + 1] = np.where(done[:n_traj, None], traj[t], s[:n_traj, :2])
    return {"success": success.astype(int).tolist(), "fail": fail.astype(int).tolist(), "t_done": t_done.tolist(),
            "route": routes.tolist(), "t_decide": t_dec.tolist(), "belief": np.round(fb, 6).tolist(),
            "latent": np.round(zf, 6).tolist(), "traj": np.round(traj, 3).tolist()}
