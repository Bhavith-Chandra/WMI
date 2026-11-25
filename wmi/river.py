"""River crossing: hierarchical planning with / without hindsight (see river_preregistration.md)."""
from __future__ import annotations

import numpy as np

from .mujoco_exp import PuckBatch

DT = 0.05
BAND, BRIDGE = 0.5, (-3.5, -2.5)
START, GOAL = np.array([0.0, -1.2]), np.array([0.0, 1.2])
ROUTES = {"direct": [(0.0, -0.75), (0.0, 0.75), tuple(GOAL)], "bridge": [(-3.0, -1.0), (-3.0, 1.0), tuple(GOAL)]}
C_GRID = np.linspace(0, 30, 301)


def prior_pdf(c):
    return 0.8 * ((c >= 0) & (c <= 3)) / 3 + 0.2 * ((c >= 15) & (c <= 25)) / 10


def sample_prior(rng, n):
    strong = rng.random(n) < 0.2
    return np.where(strong, rng.uniform(15, 25, n), rng.uniform(0, 3, n))


def in_current(p):
    return (np.abs(p[:, 1]) < BAND) & ~((p[:, 0] > BRIDGE[0]) & (p[:, 0] < BRIDGE[1]))


def fit_linear(rng, E=200, T=60):
    env = PuckBatch(E); env.set_phi(np.tile([1.0, 0, 0], (E, 1)))
    s = env.reset(rng.uniform(-3, 3, (E, 2))); X, Y = [], []
    a = np.zeros((E, 2))
    for _ in range(T):
        a = np.clip(0.7 * a + 0.7 * rng.standard_normal((E, 2)), -1, 1)
        s2 = env.step(a); X.append(np.c_[s[:, 2:], a]); Y.append(s2 - s); s = s2
    X, Y = np.concatenate(X), np.concatenate(Y)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)              # (4 inputs: v, u) -> (4 outputs: dp, dv)
    res = Y - X @ W
    kappa = W[2, 2] / 5.0                                   # d(dv_x)/d(force_x)
    return {"W": W, "kappa": float(kappa), "sigma": max(float(res[:, 2].std()), 0.002)}


def controller(p, v, w, c_hat, kp=10.0, kd=4.0):
    f = kp * (w - p) - kd * v
    f[:, 0] -= np.where(in_current(p), c_hat, 0.0)        # feed-forward cancel the believed current
    return np.clip(f / 5.0, -1, 1)


def believed_crossing_ok(c, lin, waterfall, steps=80):
    """Roll out the BELIEVED linear model + believed current for the direct crossing waypoint pair."""
    W, k = lin["W"], lin["kappa"]
    n = len(c); p = np.tile(ROUTES["direct"][0], (n, 1)).astype(float); v = np.zeros((n, 2))
    w = np.tile(ROUTES["direct"][1], (n, 1)); maxx = np.full(n, -9.0)
    for _ in range(steps):
        u = controller(p, v, w, c)
        d = np.c_[v, u] @ W
        d[:, 2] += k * np.where(in_current(p), c, 0.0)
        p, v = p + d[:, :2], v + d[:, 2:]
        maxx = np.where(in_current(p), np.maximum(maxx, p[:, 0]), maxx)
    return (maxx < waterfall - 0.3) & (np.linalg.norm(p - w, axis=1) < 0.15)


def run(kind, c_sched, lin, waterfall, T=400, budget=60, seed=0):
    """c_sched: (T, E) current per step. Returns success (E,), time (E,), fell (E,)."""
    E = c_sched.shape[1]
    env = PuckBatch(E); env.set_phi(np.tile([1.0, 0, 0], (E, 1))); s = env.reset(np.tile(START, (E, 1)))
    crit_ok = believed_crossing_ok(C_GRID, lin, waterfall)
    c_crit = C_GRID[crit_ok].max() if crit_ok.any() else -1.0
    prior_mean = float((C_GRID * prior_pdf(C_GRID)).sum() / prior_pdf(C_GRID).sum())
    logpost = np.tile(np.log(prior_pdf(C_GRID) + 1e-4), (E, 1))
    last_res_c = np.full(E, prior_mean)
    hier = kind.startswith("hier")
    route = np.array(["bridge" if kind == "hier_cautious" else "direct"] * E, dtype=object)
    wp_i = np.zeros(E, int); wp_t = np.zeros(E, int); extra = [None] * E
    done = np.zeros(E, bool); fell = np.zeros(E, bool); t_goal = np.full(E, T)
    for t in range(T):
        p, v = s[:, :2], s[:, 2:]
        post = np.exp(logpost - logpost.max(1, keepdims=True)); post /= post.sum(1, keepdims=True)
        c_hat = {"flat_forward": np.full(E, prior_mean), "hier_forward": np.full(E, prior_mean),
                 "hier_cautious": np.full(E, prior_mean), "hier_oracle": c_sched[t],
                 "hier_residual": last_res_c}.get(kind, post @ C_GRID)
        targets = np.zeros((E, 2))
        for e in range(E):
            if done[e]:
                continue
            if not hier:
                targets[e] = GOAL; continue
            want = "direct" if (kind != "hier_cautious" and c_hat[e] <= c_crit) else "bridge"
            if want != route[e]:                       # re-derive subgoals from the current state
                y = p[e, 1]
                if abs(y) < BAND:                      # in the river: retreat/finish to nearest bank first
                    extra[e] = (p[e, 0], -0.75 if y < 0 else 0.75)
                route[e], wp_i[e], wp_t[e] = want, 0, 0
                if want == "direct" and p[e, 1] > 0.5:
                    wp_i[e] = 2
            wps = ROUTES[route[e]]
            if extra[e] is not None:
                tgt = np.array(extra[e])
                if np.linalg.norm(p[e] - tgt) < 0.15:
                    extra[e] = None
                    if tgt[1] > 0:
                        wp_i[e] = 2                    # already across: go to goal
            if extra[e] is None:
                tgt = np.array(wps[min(wp_i[e], 2)])
                if np.linalg.norm(p[e] - tgt) < 0.15 and wp_i[e] < 2:
                    wp_i[e] += 1; wp_t[e] = 0; tgt = np.array(wps[wp_i[e]])
                wp_t[e] += 1
                if wp_t[e] > budget:                   # subgoal failure: re-plan with current belief (retry)
                    wp_t[e] = 0
                    wp_i[e] = 0 if p[e, 1] < -BAND else wp_i[e]
            targets[e] = tgt
        u = controller(p, v, targets, c_hat)
        u[done] = 0
        force = np.where(in_current(p), c_sched[t], 0.0)
        for e in range(E):
            env.d[e].xfrc_applied[env.body, 0] = force[e]
        s2 = env.step(u)
        # hindsight update from the surprise: residual of the learned no-current model
        d = s2 - s
        pred = np.c_[v, u] @ lin["W"]
        r = d[:, 2] - pred[:, 2]
        m = in_current(p) & ~done
        logpost[m] += -0.5 * ((r[m, None] - lin["kappa"] * C_GRID[None]) / lin["sigma"]) ** 2
        last_res_c = np.where(m, r / lin["kappa"], last_res_c)
        s = s2
        newfall = (~done) & in_current(s[:, :2]) & (s[:, 0] > waterfall)
        fell |= newfall; done |= newfall
        reached = (~done) & (np.linalg.norm(s[:, :2] - GOAL, axis=1) < 0.15)
        t_goal[reached] = t + 1; done |= reached
        # out-of-arena guard: treat leaving |x|>6 as failure
        lost = (~done) & (np.abs(s[:, 0]) > 6); fell |= lost; done |= lost
        if done.all():
            break
    success = (t_goal < T) & ~fell
    return success, t_goal, fell


def schedule(regime, rng, E, T=400):
    if regime == "calm":
        c = rng.uniform(0, 3, E)
    elif regime == "strong":
        c = rng.uniform(15, 25, E)
    elif regime == "mixed":
        c = sample_prior(rng, E)
    if regime == "fresh":
        return np.stack([sample_prior(rng, E) for _ in range(T)])
    return np.repeat(c[None], T, 0)
