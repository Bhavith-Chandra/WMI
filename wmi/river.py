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


# ============================== v2: hindsight as a re-planner ==============================
def configure_v2():
    global prior_pdf, sample_prior
    def prior_pdf(c):  # noqa: F811
        return 0.8 * ((c >= 0) & (c <= 3)) / 3 + 0.2 * ((c >= 30) & (c <= 40)) / 10
    def sample_prior(rng, n):  # noqa: F811
        strong = rng.random(n) < 0.2
        return np.where(strong, rng.uniform(30, 40, n), rng.uniform(0, 3, n))
    globals()["prior_pdf"], globals()["sample_prior"] = prior_pdf, sample_prior
    globals()["C_GRID"] = np.linspace(0, 45, 451)


def candidates(p):
    x = p[0]
    return {"cross": [(x, 0.75), tuple(GOAL)], "goal": [tuple(GOAL)],
            "retreat_bridge": [(x, -0.75), (-3.0, -1.0), (-3.0, 1.0), tuple(GOAL)],
            "bridge": [(-3.0, -1.0), (-3.0, 1.0), tuple(GOAL)]}


def simulate_plan(p0, v0, wps, c, lin, waterfall, steps=300, pass_tol=0.3):
    W, k = lin["W"], lin["kappa"]; p, v = p0.copy()[None], v0.copy()[None]; i = 0
    for t in range(steps):
        w = np.array(wps[i])[None]
        if i < len(wps) - 1 and np.linalg.norm(p[0] - w[0]) < pass_tol:
            i += 1; w = np.array(wps[i])[None]
        u = controller(p, v, w, np.array([c]))
        d = np.c_[v, u] @ W
        d[:, 2] += k * (c if in_current(p)[0] else 0.0)
        p, v = p + d[:, :2], v + d[:, 2:]
        if in_current(p)[0] and p[0, 0] > waterfall:
            return False, steps
        if i == len(wps) - 1 and np.linalg.norm(p[0] - GOAL) < 0.15:
            return True, t + 1
        if abs(p[0, 0]) > 6:
            return False, steps
    return False, steps


def choose(p, v, c, lin, waterfall):
    best, bt = None, 10 ** 9
    for name, wps in candidates(p).items():
        ok, tt = simulate_plan(p, v, wps, c, lin, waterfall)
        if ok and tt < bt:
            best, bt = wps, tt
    return best if best is not None else candidates(p)["bridge"]


def run_v2(kind, c_sched, lin, waterfall, T=400, budget=80, seed=0):
    E = c_sched.shape[1]
    env = PuckBatch(E); env.set_phi(np.tile([1.0, 0, 0], (E, 1))); s = env.reset(np.tile(START, (E, 1)))
    C = globals()["C_GRID"]; pp = globals()["prior_pdf"]
    prior_mean = float((C * pp(C)).sum() / pp(C).sum())
    logpost = np.tile(np.log(pp(C) + 1e-4), (E, 1))
    hier = kind.startswith("hier")
    plans = [None] * E; idx = np.zeros(E, int); since = np.zeros(E, int); c_at_plan = np.full(E, np.nan)
    done = np.zeros(E, bool); fell = np.zeros(E, bool); t_goal = np.full(E, T); entered = np.zeros(E, bool)
    for t in range(T):
        p, v = s[:, :2], s[:, 2:]
        post = np.exp(logpost - logpost.max(1, keepdims=True)); post /= post.sum(1, keepdims=True)
        if kind in ("flat_forward", "hier_replan_forward", "hier_cautious"):
            c_hat = np.full(E, prior_mean)
        elif kind == "hier_replan_oracle":
            c_hat = c_sched[t].astype(float)
        else:
            c_hat = post @ C
        targets = np.tile(GOAL, (E, 1)).astype(float)
        if hier:
            for e in range(E):
                if done[e]:
                    continue
                trigger = plans[e] is None or since[e] > budget or (
                    kind != "hier_cautious" and abs(c_hat[e] - c_at_plan[e]) > 2.0)
                if trigger:
                    plans[e] = candidates(p[e])["bridge"] if kind == "hier_cautious" else choose(p[e], v[e], c_hat[e], lin, waterfall)
                    idx[e], since[e], c_at_plan[e] = 0, 0, c_hat[e]
                wps = plans[e]
                if idx[e] < len(wps) - 1 and np.linalg.norm(p[e] - np.array(wps[idx[e]])) < 0.3:
                    idx[e] += 1; since[e] = 0
                since[e] += 1
                targets[e] = wps[idx[e]]
        u = controller(p, v, targets, c_hat); u[done] = 0
        cur = in_current(p)
        entered |= cur & ~done
        force = np.where(cur, c_sched[t], 0.0)
        for e in range(E):
            env.d[e].xfrc_applied[env.body, 0] = force[e]
        s2 = env.step(u)
        r = (s2 - s)[:, 2] - (np.c_[v, u] @ lin["W"])[:, 2]
        m = cur & ~done
        logpost[m] += -0.5 * ((r[m, None] - lin["kappa"] * C[None]) / lin["sigma"]) ** 2
        s = s2
        nf = (~done) & in_current(s[:, :2]) & (s[:, 0] > waterfall); fell |= nf; done |= nf
        rc = (~done) & (np.linalg.norm(s[:, :2] - GOAL, axis=1) < 0.15); t_goal[rc] = t + 1; done |= rc
        lost = (~done) & (np.abs(s[:, 0]) > 6); fell |= lost; done |= lost
        if done.all():
            break
    return (t_goal < T) & ~fell, t_goal, fell, entered


# ============================== v3: CEM-MPC low level on the believed model ==============================
def cem_mpc(p, v, w, c_hat, lin, waterfall, rng, H=15, P=200, iters=3):
    """Vectorised over episodes. p,v,w: (E,2); c_hat: (E,). Returns first action (E,2)."""
    W, k = lin["W"], lin["kappa"]; E = len(p)
    mu = np.zeros((E, H, 2)); sd = np.full((E, H, 2), 0.7)
    for _ in range(iters):
        A = np.clip(mu[:, None] + sd[:, None] * rng.standard_normal((E, P, H, 2)), -1, 1)
        pp = np.repeat(p[:, None], P, 1); vv = np.repeat(v[:, None], P, 1); cost = np.zeros((E, P)); dead = np.zeros((E, P), bool)
        for t in range(H):
            u = A[:, :, t]
            d = np.concatenate([vv, u], -1) @ W
            band = (np.abs(pp[..., 1]) < BAND) & ~((pp[..., 0] > BRIDGE[0]) & (pp[..., 0] < BRIDGE[1]))
            d[..., 2] += k * np.where(band, c_hat[:, None], 0.0)
            pp, vv = pp + d[..., :2], vv + d[..., 2:]
            band2 = (np.abs(pp[..., 1]) < BAND) & ~((pp[..., 0] > BRIDGE[0]) & (pp[..., 0] < BRIDGE[1]))
            dead |= band2 & (pp[..., 0] > waterfall - 0.2)
            cost += ((pp - w[:, None]) ** 2).sum(-1)
        cost += 1000.0 * dead
        el = np.argsort(cost, 1)[:, :20]
        EA = np.take_along_axis(A, el[:, :, None, None], 1)
        mu, sd = EA.mean(1), EA.std(1) + 0.05
    return mu[:, 0]


def run_v3(kind, c_sched, lin, waterfall, T=400, seed=0, guard=12.0):
    E = c_sched.shape[1]; rng = np.random.default_rng(seed)
    env = PuckBatch(E); env.set_phi(np.tile([1.0, 0, 0], (E, 1))); s = env.reset(np.tile(START, (E, 1)))
    C = globals()["C_GRID"]; pp = globals()["prior_pdf"]
    prior_mean = float((C * pp(C)).sum() / pp(C).sum())
    logpost = np.tile(np.log(pp(C) + 1e-4), (E, 1))
    plans = [None] * E; idx = np.zeros(E, int); c_at = np.full(E, np.nan)
    done = np.zeros(E, bool); fell = np.zeros(E, bool); t_goal = np.full(E, T); entered = np.zeros(E, bool)
    for t in range(T):
        p, v = s[:, :2], s[:, 2:]
        post = np.exp(logpost - logpost.max(1, keepdims=True)); post /= post.sum(1, keepdims=True)
        c_hat = {"forward": np.full(E, prior_mean), "cautious": np.full(E, prior_mean),
                 "oracle": c_sched[t].astype(float)}.get(kind, post @ C)
        targets = np.tile(GOAL, (E, 1)).astype(float)
        for e in range(E):
            if done[e]:
                continue
            if kind == "cautious":
                if plans[e] is None:
                    plans[e] = candidates(p[e])["bridge"]
            elif plans[e] is None or abs(c_hat[e] - c_at[e]) > 2.0:
                ok_direct = simulate_plan(p[e], v[e], candidates(p[e])["goal"], c_hat[e], lin, waterfall)[0]
                plans[e] = candidates(p[e])["goal"] if ok_direct else (
                    candidates(p[e])["retreat_bridge"] if p[e, 1] < 0 else [tuple(GOAL)])
                idx[e], c_at[e] = 0, c_hat[e]
            if idx[e] < len(plans[e]) - 1 and np.linalg.norm(p[e] - np.array(plans[e][idx[e]])) < 0.3:
                idx[e] += 1
            targets[e] = plans[e][idx[e]]
        u = cem_mpc(p, v, targets, c_hat, lin, waterfall, rng); u[done] = 0
        cur = in_current(p); entered |= cur & ~done
        force = np.where(cur, c_sched[t], 0.0)
        for e in range(E):
            env.d[e].xfrc_applied[env.body, 0] = force[e]
        s2 = env.step(u)
        r = (s2 - s)[:, 2] - (np.c_[v, u] @ lin["W"])[:, 2]
        m = cur & ~done
        logpost[m] += -0.5 * ((r[m, None] - lin["kappa"] * C[None]) / lin["sigma"]) ** 2
        s = s2
        nf = (~done) & in_current(s[:, :2]) & (s[:, 0] > waterfall); fell |= nf; done |= nf
        rc = (~done) & (np.linalg.norm(s[:, :2] - GOAL, axis=1) < 0.15); t_goal[rc] = t + 1; done |= rc
        lost = (~done) & (np.abs(s[:, 0]) > guard); fell |= lost; done |= lost
        if done.all():
            break
    return (t_goal < T) & ~fell, t_goal, fell, entered
