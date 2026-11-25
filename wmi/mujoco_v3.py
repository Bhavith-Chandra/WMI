"""MuJoCo v3: ensemble world models with disagreement penalty (anti-exploitation), particle posterior over phi,
AR(1)-persistent wind. See papers/latent-hindsight-planning/mujoco_v3_preregistration.md."""
from __future__ import annotations

import numpy as np
import torch

from .mujoco_exp import PHI_MU, DynMLP, PuckBatch, Posterior, fit, sample_phi

N_ENS = 5


def fit_ensemble(n_in, X, Y, ep, rng, steps):
    members, infos = [], []
    for m in range(N_ENS):
        torch.manual_seed(int(rng.integers(1 << 30)))
        ue = np.unique(ep)
        boot = rng.choice(ue, size=len(ue), replace=True)          # bootstrap over EPISODES
        idx = np.concatenate([np.flatnonzero(ep == e) for e in np.unique(boot)])
        mdl, info = fit(DynMLP(n_in), X[idx], Y[idx], ep[idx], rng, steps)
        for p in mdl.parameters():
            p.requires_grad_(False)
        members.append(mdl); infos.append(info)
    sd = np.mean([i["val_resid_sd"] for i in infos], 0)
    return members, {"val_resid_sd": sd.tolist(), "val_mse_norm": [i["val_mse_norm"] for i in infos]}


def make_schedule3(regime, rng, E, T, wind_sd=1.0, seg=40, switch=55):
    goals = rng.uniform(-1, 1, (T // seg, E, 2)).repeat(seg, 0)
    if regime.startswith("ar"):                       # AR(1) wind, g = 1
        rho = float(regime[2:])
        sched = np.zeros((T, E, 3)); sched[..., 0] = 1.0
        f = wind_sd * rng.standard_normal((E, 2))
        for t in range(T):
            if t > 0:
                f = rho * f + np.sqrt(1 - rho ** 2) * wind_sd * rng.standard_normal((E, 2))
            sched[t, :, 1:] = f
        return goals, sched
    if regime == "none":
        phi = np.tile(np.array([1.0, 0, 0]), (E, 1))
    elif regime == "ood_gain":
        phi = sample_phi(rng, E, "ood_gain"); phi[:, 1:] = 0
    else:
        phi = sample_phi(rng, E, wind_sd=wind_sd)
        if regime == "gain":
            phi[:, 1:] = 0
        if regime in ("wind", "fresh"):
            phi[:, 0] = 1.0
    sched = np.repeat(phi[None], T, 0).copy()
    for t in range(T):
        if regime == "switching" and t > 0 and t % switch == 0:
            sched[t:] = sample_phi(rng, E, wind_sd=wind_sd)[None]
        if regime == "fresh":
            sched[t, :, 1:] = wind_sd * rng.standard_normal((E, 2))
    return goals, sched


class Planner3:
    SGD = ("online_sgd", "hybrid")

    def __init__(self, kind, Mu, Mc, E, cfg, rng):
        self.kind, self.Mu, self.Mc, self.E, self.cfg, self.rng = kind, Mu, Mc, E, cfg, rng
        self.mu = np.zeros((E, cfg["H"], 2)); self.res = np.zeros((E, 4)); self.phi_true = None
        self.rsd = torch.tensor(cfg["resid_sd"], dtype=torch.float32)
        self.is_post = kind in ("hindsight_posterior", "hybrid")
        if kind in self.SGD:
            self.W = torch.stack([m.head.weight for m in Mu])[None].repeat(E, 1, 1, 1).clone()   # (E,M,4,h)
            self.b = torch.stack([m.head.bias for m in Mu])[None].repeat(E, 1, 1).clone()        # (E,M,4)
        if self.is_post:
            sd = np.array(cfg["resid_sd"]); self.post = Posterior(E, 0.5, 2 * sd, sd, cfg["forget"])
        if kind == "hindsight_latent":
            self.phi_hat = torch.tensor(np.tile(PHI_MU, (E, 1)), dtype=torch.float32); self.buf = []
        if kind == "hindsight_latent_pf":
            N = cfg["n_particles"]
            self.parts = self._prior(E * N).reshape(E, N, 3); self.logw = np.zeros((E, N))

    def _prior(self, n):
        w = 1.5 * self.cfg["wind_sd"]
        return np.stack([self.rng.uniform(0.1, 2.0, n), w * self.rng.standard_normal(n), w * self.rng.standard_normal(n)], 1)

    def phi_est(self):
        if self.kind == "oracle_latent":
            return self.phi_true
        if self.kind == "hindsight_latent":
            return self.phi_hat.numpy()
        w = np.exp(self.logw - self.logw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        return np.einsum("en,end->ed", w, self.parts)

    def _members(self, v, a):                        # base predictions per member: (M,E,P,4)
        x = torch.cat([v, a], -1)
        if self.kind in self.SGD:
            outs = []
            for m, mdl in enumerate(self.Mu):
                h = mdl.features(x)
                outs.append((torch.einsum("eoh,eph->epo", self.W[:, m], h) + self.b[:, m, None]) * mdl.out_sd + mdl.out_mu)
            return torch.stack(outs)
        return torch.stack([mdl(x) for mdl in self.Mu])

    def predict(self, v, a):
        if self.kind in ("hindsight_latent", "hindsight_latent_pf", "oracle_latent"):
            phi = torch.tensor(self.phi_est(), dtype=torch.float32)[:, None].expand(-1, v.shape[1], -1)
            ms = torch.stack([mdl(torch.cat([v, a, phi], -1)) for mdl in self.Mc])
        else:
            ms = self._members(v, a)
            if self.kind == "residual":
                ms = ms + torch.tensor(self.res, dtype=torch.float32)[None, :, None]
            if self.is_post:
                eff = ms - self._members(v, torch.zeros_like(a))
                pm = torch.tensor(self.post.mean(), dtype=torch.float32)
                ms = ms + pm[None, :, None, :, 0] * eff + pm[None, :, None, :, 1]
        return ms.mean(0), ms.std(0)

    @torch.no_grad()
    def act(self, s, goal):
        c = self.cfg; E, H, P = self.E, c["H"], c["pop"]
        if not c.get("warm", True):
            self.mu = np.zeros((E, H, 2))
        std = np.full((E, H, 2), 0.6); g = torch.tensor(goal, dtype=torch.float32)
        for _ in range(c["iters"]):
            A = np.clip(self.mu[:, None] + std[:, None] * self.rng.standard_normal((E, P, H, 2)), -1, 1)
            At = torch.tensor(A, dtype=torch.float32)
            st = torch.tensor(s, dtype=torch.float32)[:, None].expand(-1, P, -1)
            cost = torch.zeros(E, P)
            for t in range(H):
                d, dsd = self.predict(st[..., 2:], At[:, :, t])
                st = st + d
                cost += ((st[..., :2] - g[:, None]) ** 2).sum(-1) + 0.01 * (At[:, :, t] ** 2).sum(-1) \
                    + c["beta"] * ((dsd / self.rsd) ** 2).mean(-1)
            el = torch.topk(cost, c["elites"], dim=1, largest=False).indices.numpy()
            EA = np.take_along_axis(A, el[:, :, None, None], 1)
            self.mu, std = EA.mean(1), EA.std(1) + 0.05
        plan, K = self.mu.copy(), c["K"]
        self.mu = np.concatenate([self.mu[:, K:], np.zeros((E, K, 2))], 1)
        return plan

    @torch.no_grad()
    def observe(self, s, a, s2):
        v = torch.tensor(s[:, 2:], dtype=torch.float32)[:, None]; at = torch.tensor(a, dtype=torch.float32)[:, None]
        dobs = (s2 - s).astype(np.float32)
        if self.kind in ("residual", "hindsight_posterior", "hybrid", "online_sgd"):
            base = self._members(v, at).mean(0)[:, 0].numpy()
        if self.kind == "residual":
            self.res = dobs - base
        if self.is_post:
            eff = (self._members(v, at) - self._members(v, torch.zeros_like(at))).mean(0)[:, 0].numpy()
            self.post.update(eff, dobs - base)
        if self.kind in self.SGD:
            y = torch.tensor(dobs)
            for m, mdl in enumerate(self.Mu):
                h = mdl.features(torch.cat([v, at], -1))[:, 0]
                pred = torch.einsum("eoh,eh->eo", self.W[:, m], h) + self.b[:, m]
                gerr = 2 * (pred - (y - mdl.out_mu) / mdl.out_sd) / 4
                self.W[:, m] -= self.cfg["sgd_lr"] * gerr[:, :, None] * h[:, None, :]
                self.b[:, m] -= self.cfg["sgd_lr"] * gerr
        if self.kind == "hindsight_latent_pf":
            E, N = self.parts.shape[:2]
            ph = torch.tensor(self.parts.reshape(E * N, 3), dtype=torch.float32)
            vv = v[:, 0].repeat_interleave(N, 0); aa = at[:, 0].repeat_interleave(N, 0)
            pred = torch.stack([mdl(torch.cat([vv, aa, ph], -1)) for mdl in self.Mc]).mean(0).numpy().reshape(E, N, 4)
            sd = 2 * np.array(self.cfg["resid_sd_c"])
            self.logw += -0.5 * (((dobs[:, None] - pred) / sd) ** 2).sum(-1)
            self.logw -= self.logw.max(1, keepdims=True)
            w = np.exp(self.logw); w /= w.sum(1, keepdims=True)
            ess = 1 / (w ** 2).sum(1)
            for e in np.flatnonzero(ess < N / 2):           # systematic resampling
                u = (self.rng.random() + np.arange(N)) / N
                idx = np.minimum(np.searchsorted(np.cumsum(w[e]), u), N - 1)
                self.parts[e] = self.parts[e, idx]; self.logw[e] = 0
            rej = self.rng.random((E, N)) < 0.02
            if rej.any():
                self.parts[rej] = self._prior(int(rej.sum()))
                self.logw[rej] = np.log(np.exp(self.logw).mean(1, keepdims=True).repeat(N, 1)[rej] + 1e-300)
        if self.kind == "hindsight_latent":
            self.buf = (self.buf + [(s[:, 2:].copy(), a.copy(), dobs)])[-self.cfg["window"]:]
            Vb = torch.tensor(np.stack([b[0] for b in self.buf], 1), dtype=torch.float32)
            Ab = torch.tensor(np.stack([b[1] for b in self.buf], 1), dtype=torch.float32)
            Db = torch.tensor(np.stack([b[2] for b in self.buf], 1))
            phi = self.phi_hat.clone()
            with torch.enable_grad():
                phi.requires_grad_(True)
                opt = torch.optim.Adam([phi], lr=0.05)
                for _ in range(self.cfg["latent_steps"]):
                    pr = torch.stack([mdl(torch.cat([Vb, Ab, phi[:, None].expand(-1, Vb.shape[1], -1)], -1)) for mdl in self.Mc]).mean(0)
                    loss = (((pr - Db) / self.Mc[0].out_sd) ** 2).mean((1, 2)).sum()
                    opt.zero_grad(); loss.backward(); opt.step()
            self.phi_hat = phi.detach()


def run_regime3(kind, Mu, Mc, goals, sched, cfg, seed):
    T, E = sched.shape[:2]
    env = PuckBatch(E)
    starts = np.random.default_rng(seed + 1).uniform(-1, 1, (T // 40 + 1, E, 2))   # identical for every planner
    s = env.reset(starts[0]) if cfg.get("reset_each_segment") else env.reset()
    pl = Planner3(kind, Mu, Mc, E, cfg, np.random.default_rng(seed))
    cost = np.zeros((T, E)); K = cfg["K"]
    for t in range(T):
        env.set_phi(sched[t]); pl.phi_true = sched[t]
        if cfg.get("reset_each_segment") and t > 0 and t % 40 == 0:
            s = env.reset(starts[t // 40]); pl.mu[:] = 0
        if t % K == 0:
            plan = pl.act(s, goals[t])
        a = np.clip(plan[:, t % K], -1, 1)
        s2 = env.step(a)
        cost[t] = ((s2[:, :2] - goals[t]) ** 2).sum(1) + 0.01 * (a ** 2).sum(1)
        pl.observe(s, a, s2); s = s2
    return cost.mean(0)
