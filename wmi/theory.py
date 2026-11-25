"""Hindsight vs forward planning: a minimal model in which the theorems are exact.

Model (scalar state, one-step goals):
    Delta s_t = theta_t * a_t + w_t + eps_t,   eps_t ~ N(0, sigma^2) iid
    task at step t: realise the commanded displacement delta_t (iid, drawn before acting)
    cost c_t = (Delta s_t - delta_t)^2.
phi_t = (theta_t, w_t): theta is a GAIN fault (multiplies the command), w an ADDITIVE fault.
Prior phi ~ N(m0, S0), S0 = diag(s_theta^2, s_w^2).

Regret is measured against the oracle that knows phi_t. Because eps_t is common to every
planner, we use the conditional excess cost  e_t = (theta_t a_t + w_t - delta_t)^2,
whose expectation equals E[c_t] - E[c_t^oracle] exactly (lower variance than realised cost).

Planners
  oracle          a = (delta - w)/theta                         (excess 0)
  forward         Bayes-optimal under the PRIOR, ignores outcomes
  residual        residual reuse (Balestriero, LeCun, Varaku 2026, arXiv:2611.24778) applied
                  sequentially: add the last observed residual to the nominal prediction
  hindsight_bayes myopic Bayes action under the POSTERIOR from all observed transitions
  hindsight_ce    certainty-equivalent action under the posterior mean
  hindsight_forget posterior with exponential forgetting (for switching disturbances)

Disturbance regimes: persistent | persistent_additive | persistent_gain | fresh | switching | none.
NOTE: hindsight_* planners always ASSUME persistence; under "fresh" they are misspecified (the
correctly specified Bayes planner under fresh disturbances coincides with forward, Theorem 2).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class TheoryConfig:
    theta_bar: float = 1.0
    s_theta: float = 0.4
    s_w: float = 0.5
    sigma: float = 0.3
    s_delta: float = 1.0
    a_max: float = 10.0
    T: int = 4000
    M: int = 2000          # Monte-Carlo episodes
    switch_every: int = 500
    forget: float = 0.99   # forgetting factor for hindsight_forget
    seed: int = 0

    def dict(self):
        return asdict(self)


PLANNERS = ("forward", "residual", "hindsight_bayes", "hindsight_ce", "hindsight_forget")


def _draw_phi(rng, cfg, M):
    th = cfg.theta_bar + cfg.s_theta * rng.standard_normal(M)
    w = cfg.s_w * rng.standard_normal(M)
    return th, w


def bayes_action(mu, S, delta, a_max):
    """argmin_a E[(theta a + w - delta)^2] for (theta, w) ~ N(mu, S)."""
    mth, mw = mu[:, 0], mu[:, 1]
    a = (mth * (delta - mw) - S[:, 0, 1]) / (mth ** 2 + S[:, 0, 0])
    return np.clip(a, -a_max, a_max)


def ce_action(mu, delta, a_max):
    mth = np.where(np.abs(mu[:, 0]) < 1e-3, 1e-3, mu[:, 0])
    return np.clip((delta - mu[:, 1]) / mth, -a_max, a_max)




def simulate(regime: str, cfg: TheoryConfig) -> dict:
    """Return per-step mean excess cost (and std error) for every planner, shape (T,)."""
    rng = np.random.default_rng(cfg.seed)
    M, T = cfg.M, cfg.T
    m0 = np.array([cfg.theta_bar, 0.0])
    S0 = np.diag([cfg.s_theta ** 2, cfg.s_w ** 2])
    P0 = np.linalg.inv(S0)

    th, w = _draw_phi(rng, cfg, M)
    if regime == "none":
        th, w = np.full(M, cfg.theta_bar), np.zeros(M)
    elif regime == "persistent_additive":   # gain known (theta = theta_bar), additive offset persists
        th = np.full(M, cfg.theta_bar)
    elif regime == "persistent_gain":       # pure gain fault, no additive offset
        w = np.zeros(M)

    # Bayes state in information form: precision P, information vector b (P mu = b)
    state = {}
    for name in ("hindsight_bayes", "hindsight_ce", "hindsight_forget"):
        state[name] = {"P": np.broadcast_to(P0, (M, 2, 2)).copy(),
                       "b": np.broadcast_to(P0 @ m0, (M, 2)).copy()}
    last_res = np.zeros(M)  # residual reuse memory (prior mean of disturbance = 0)

    out = {p: np.zeros(T) for p in PLANNERS}
    out_se = {p: np.zeros(T) for p in PLANNERS}
    min_eig_over_t = np.zeros(T)

    for t in range(T):
        if regime == "fresh" or (regime == "switching" and t > 0 and t % cfg.switch_every == 0):
            th, w = _draw_phi(rng, cfg, M)
        delta = cfg.s_delta * rng.standard_normal(M)
        eps = cfg.sigma * rng.standard_normal(M)

        acts = {}
        acts["forward"] = bayes_action(np.broadcast_to(m0, (M, 2)), np.broadcast_to(S0, (M, 2, 2)), delta, cfg.a_max)
        acts["residual"] = np.clip((delta - last_res) / cfg.theta_bar, -cfg.a_max, cfg.a_max)
        for name in ("hindsight_bayes", "hindsight_ce", "hindsight_forget"):
            S = np.linalg.inv(state[name]["P"])
            mu = np.einsum("mij,mj->mi", S, state[name]["b"])
            acts[name] = ce_action(mu, delta, cfg.a_max) if name == "hindsight_ce" else bayes_action(mu, S, delta, cfg.a_max)

        for p in PLANNERS:
            e = (th * acts[p] + w - delta) ** 2
            out[p][t] = e.mean()
            out_se[p][t] = e.std(ddof=1) / np.sqrt(M)

        # observe the transition each planner produced, update its memory
        for p in PLANNERS:
            a = acts[p]
            y = th * a + w + eps
            if p == "residual":
                last_res = y - cfg.theta_bar * a
            elif p.startswith("hindsight"):
                x = np.stack([a, np.ones(M)], 1)
                st = state[p]
                if p == "hindsight_forget":
                    # discount evidence toward the prior: P <- lam P + (1-lam) P0
                    st["P"] = cfg.forget * st["P"] + (1 - cfg.forget) * P0
                    st["b"] = cfg.forget * st["b"] + (1 - cfg.forget) * (P0 @ m0)
                st["P"] = st["P"] + np.einsum("mi,mj->mij", x, x) / cfg.sigma ** 2
                st["b"] = st["b"] + x * (y / cfg.sigma ** 2)[:, None]
        min_eig_over_t[t] = np.linalg.eigvalsh(state["hindsight_bayes"]["P"]).min(1).mean()

    return {"excess": out, "excess_se": out_se, "min_precision_eig": min_eig_over_t}


def forward_excess_closed_form(cfg: TheoryConfig) -> float:
    """Theorem 2: c_F = s_w^2 + s_theta^2 E[(delta - w_bar)^2] / (theta_bar^2 + s_theta^2) (no clipping)."""
    return cfg.s_w ** 2 + cfg.s_theta ** 2 * cfg.s_delta ** 2 / (cfg.theta_bar ** 2 + cfg.s_theta ** 2)


def residual_excess_closed_form_additive(cfg: TheoryConfig) -> float:
    """Theorem 4 (theta known = theta_bar, persistent additive w): excess = sigma^2 every step t >= 1."""
    return cfg.sigma ** 2
