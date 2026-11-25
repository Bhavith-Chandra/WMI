"""Interactive MuJoCo test of hindsight planning (see papers/latent-hindsight-planning/mujoco_preregistration.md).

Puck with hidden actuator GAIN g and additive external force f. Learned world models, CEM-MPC, and seven
outcome-handling strategies, all evaluated on identical pre-generated disturbance/goal schedules (paired design).
"""
from __future__ import annotations

import mujoco
import numpy as np
import torch
import torch.nn as nn

XML = """<mujoco><option timestep="0.01"/><worldbody><body name="puck">
<joint name="x" type="slide" axis="1 0 0" damping="1"/><joint name="y" type="slide" axis="0 1 0" damping="1"/>
<geom type="sphere" size="0.05" mass="1"/></body></worldbody>
<actuator><motor joint="x" gear="5" ctrlrange="-1 1"/><motor joint="y" gear="5" ctrlrange="-1 1"/></actuator></mujoco>"""
FRAME_SKIP, GEAR = 5, 5.0
PHI_MU = np.array([1.0, 0.0, 0.0]); PHI_SD = np.array([0.3464, 1.0, 1.0])  # train-distribution moments


class PuckBatch:
    def __init__(self, E):
        self.m = [mujoco.MjModel.from_xml_string(XML) for _ in range(E)]
        self.d = [mujoco.MjData(m) for m in self.m]
        self.body = self.m[0].body("puck").id
        self.E = E

    def set_phi(self, phi):
        for e in range(self.E):
            self.m[e].actuator_gear[:, 0] = GEAR * phi[e, 0]
            self.d[e].xfrc_applied[self.body, :2] = phi[e, 1:3]

    def reset(self, pos=None):
        for e in range(self.E):
            self.d[e].qpos[:] = 0 if pos is None else pos[e]; self.d[e].qvel[:] = 0
            mujoco.mj_forward(self.m[e], self.d[e])
        return self.state()

    def state(self):
        return np.array([np.concatenate([d.qpos, d.qvel]) for d in self.d])

    def step(self, a):
        a = np.clip(a, -1, 1)
        for e in range(self.E):
            self.d[e].ctrl[:] = a[e]
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(self.m[e], self.d[e])
        return self.state()


def sample_phi(rng, n, regime="train", wind_sd=1.0):
    lo, hi = (0.2, 0.4) if regime == "ood_gain" else (0.4, 1.6)
    return np.stack([rng.uniform(lo, hi, n), wind_sd * rng.standard_normal(n), wind_sd * rng.standard_normal(n)], 1)


def phi_sd(wind_sd):
    return np.array([0.3464, wind_sd, wind_sd])


def collect(rng, n_batches=20, E=200, T=50, wind_sd=1.0):
    V, A, PHI, D, EP = [], [], [], [], []
    env = PuckBatch(E)
    for bidx in range(n_batches):
        phi = sample_phi(rng, E, wind_sd=wind_sd); env.set_phi(phi)
        s = env.reset(rng.uniform(-1, 1, (E, 2)))
        a = np.zeros((E, 2))
        for t in range(T):
            a = np.clip(0.8 * a + 0.6 * rng.standard_normal((E, 2)), -1, 1)
            s2 = env.step(a)
            V.append(s[:, 2:]); A.append(a.copy()); PHI.append(phi); D.append(s2 - s); EP.append(bidx * E + np.arange(E))
            s = s2
    cat = lambda x: np.concatenate(x).astype(np.float32)
    return cat(V), cat(A), cat(PHI), cat(D), np.concatenate(EP)


class DynMLP(nn.Module):
    def __init__(self, n_in, hidden=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, 4)
        for k in ("in_mu", "in_sd", "out_mu", "out_sd"):
            self.register_buffer(k, torch.zeros(n_in if k.startswith("in") else 4))

    def features(self, x):
        return self.body((x - self.in_mu) / self.in_sd)

    def forward(self, x):
        return self.head(self.features(x)) * self.out_sd + self.out_mu


def fit(model, X, Y, ep, rng, steps=6000, batch=1024, lr=1e-3):
    val_ep = rng.choice(np.unique(ep), size=len(np.unique(ep)) // 10, replace=False)
    va = np.isin(ep, val_ep); tr = ~va
    model.in_mu.copy_(torch.from_numpy(X[tr].mean(0))); model.in_sd.copy_(torch.from_numpy(X[tr].std(0) + 1e-6))
    model.out_mu.copy_(torch.from_numpy(Y[tr].mean(0))); model.out_sd.copy_(torch.from_numpy(Y[tr].std(0) + 1e-6))
    Xt, Yt, Xv, Yv = (torch.from_numpy(z) for z in (X[tr], Y[tr], X[va], Y[va]))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, state = np.inf, None
    for s in range(1, steps + 1):
        i = torch.from_numpy(rng.integers(0, len(Xt), batch))
        loss = (((model(Xt[i]) - Yt[i]) / model.out_sd) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 250 == 0:
            with torch.no_grad():
                v = float((((model(Xv) - Yv) / model.out_sd) ** 2).mean())
            if v < best:
                best, state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state)
    with torch.no_grad():
        res_sd = (model(Xv) - Yv).std(0).numpy()
    return model.eval(), {"val_mse_norm": best, "val_resid_sd": res_sd.tolist(), "n_val_episodes": int(len(val_ep))}


# ---------------------------------------------------------------- outcome-handling strategies
class Posterior:
    """Per-dimension Bayesian regression r_d = theta_d * eff_d + w_d with forgetting (vectorised over E)."""

    def __init__(self, E, s_theta, s_w, sigma, forget):
        self.P0 = np.zeros((4, 2, 2)); self.P0[:, 0, 0] = 1 / s_theta ** 2; self.P0[:, 1, 1] = 1 / s_w ** 2
        self.P = np.broadcast_to(self.P0, (E, 4, 2, 2)).copy(); self.b = np.zeros((E, 4, 2))
        self.s2, self.lam = sigma ** 2, forget

    def mean(self):
        return np.linalg.solve(self.P, self.b[..., None])[..., 0]   # (E,4,2): theta, w

    def update(self, eff, r):
        x = np.stack([eff, np.ones_like(eff)], -1)                 # (E,4,2)
        self.P = self.lam * self.P + (1 - self.lam) * self.P0
        self.b = self.lam * self.b
        self.P = self.P + np.einsum("edi,edj->edij", x, x) / self.s2[None, :, None, None]
        self.b = self.b + x * (r / self.s2)[..., None]


class Planner:
    def __init__(self, kind, Mu, Mc, E, cfg, rng):
        self.kind, self.Mu, self.Mc, self.E, self.cfg, self.rng = kind, Mu, Mc, E, cfg, rng
        self.is_post = kind.startswith("hindsight_posterior") or kind == "hybrid"
        self.is_lat = kind.startswith("hindsight_latent")
        self.cd = kind.endswith("_cd")
        self.S = np.zeros(E)
        self.PSD = cfg.get("phi_sd", PHI_SD)
        H = cfg["H"]
        self.mu = np.zeros((E, H, 2))
        self.res = np.zeros((E, 4))
        if kind in ("online_sgd", "online_sgd_lr1e-2", "hybrid"):
            self.W = Mu.head.weight.detach().clone().repeat(E, 1, 1); self.bh = Mu.head.bias.detach().clone().repeat(E, 1)
            self.lr = 1e-2 if kind == "online_sgd_lr1e-2" else cfg["sgd_lr"]
        if self.is_post:
            sd = np.array(cfg["resid_sd"])
            self.post = Posterior(E, 0.5, 2 * sd, sd, cfg["forget"])
        if self.is_lat:
            self.phi_hat = torch.zeros(E, 3)  # normalised phi
            self.buf = []
        self.phi_true = None

    # predictor: v (E,P,2), a (E,P,2) -> delta (E,P,4)
    def _base(self, v, a):
        x = torch.cat([v, a], -1)
        if self.kind in ("online_sgd", "online_sgd_lr1e-2", "hybrid"):
            h = self.Mu.features(x)
            return (torch.einsum("eoh,eph->epo", self.W, h) + self.bh[:, None]) * self.Mu.out_sd + self.Mu.out_mu
        return self.Mu(x)

    def predict(self, v, a):
        if self.is_lat or self.kind == "oracle_latent":
            phin = self.phi_hat if self.is_lat else torch.from_numpy(
                ((self.phi_true - PHI_MU) / self.PSD).astype(np.float32))
            phi = phin * torch.from_numpy(self.PSD.astype(np.float32)) + torch.from_numpy(PHI_MU.astype(np.float32))
            return self.Mc(torch.cat([v, a, phi[:, None].expand(-1, v.shape[1], -1)], -1))
        d = self._base(v, a)
        if self.kind == "residual":
            d = d + torch.from_numpy(self.res.astype(np.float32))[:, None]
        if self.is_post:
            eff = d - self._base(v, torch.zeros_like(a))
            m = torch.from_numpy(self.post.mean().astype(np.float32))       # (E,4,2)
            d = d + m[:, None, :, 0] * eff + m[:, None, :, 1]
        return d

    @torch.no_grad()
    def act(self, s, goal):
        c = self.cfg; E, H, P = self.E, c["H"], c["pop"]
        std = np.full((E, H, 2), 0.6)
        g = torch.from_numpy(goal.astype(np.float32))
        for _ in range(c["iters"]):
            A = np.clip(self.mu[:, None] + std[:, None] * self.rng.standard_normal((E, P, H, 2)), -1, 1)
            At = torch.from_numpy(A.astype(np.float32))
            st = torch.from_numpy(s.astype(np.float32))[:, None].expand(-1, P, -1)
            cost = torch.zeros(E, P)
            for t in range(H):
                st = st + self.predict(st[..., 2:], At[:, :, t])
                cost += ((st[..., :2] - g[:, None]) ** 2).sum(-1) + 0.01 * (At[:, :, t] ** 2).sum(-1)
            el = torch.topk(cost, c["elites"], dim=1, largest=False).indices.numpy()
            E_A = np.take_along_axis(A, el[:, :, None, None], 1)
            self.mu, std = E_A.mean(1), E_A.std(1) + 0.05
        plan = self.mu.copy()
        K = c.get("K", 1)
        self.mu = np.concatenate([self.mu[:, K:], np.zeros((E, K, 2))], 1)
        return plan

    def observe(self, s, a, s2):
        v = torch.from_numpy(s[:, 2:].astype(np.float32))[:, None]; at = torch.from_numpy(a.astype(np.float32))[:, None]
        dobs = (s2 - s).astype(np.float32)
        with torch.no_grad():
            base = self._base(v, at)[:, 0].numpy()
            eff = (self._base(v, at) - self._base(v, torch.zeros_like(at)))[:, 0].numpy()
        if self.kind == "residual":
            self.res = dobs - base
        if self.is_post:
            if self.cd:
                m = self.post.mean()
                pred = base + m[..., 0] * eff + m[..., 1]
                z = (((dobs - pred) / np.array(self.cfg["resid_sd"])) ** 2).mean(1)
                self.S = np.maximum(0, self.S + z - 1.5); hit = self.S > 10
                if hit.any():
                    self.post.P[hit] = self.post.P0; self.post.b[hit] = 0; self.S[hit] = 0
            self.post.update(eff, dobs - base)
        if self.kind in ("online_sgd", "online_sgd_lr1e-2", "hybrid"):
            with torch.no_grad():
                h = self.Mu.features(torch.cat([v, at], -1))[:, 0]                  # (E,hid)
                pred_n = torch.einsum("eoh,eh->eo", self.W, h) + self.bh
                y_n = (torch.from_numpy(dobs) - self.Mu.out_mu) / self.Mu.out_sd
                gerr = 2 * (pred_n - y_n) / 4                                        # d mean-sq / d pred
                self.W -= self.lr * gerr[:, :, None] * h[:, None, :]
                self.bh -= self.lr * gerr
        if self.is_lat:
            if self.cd:
                sdv0 = torch.from_numpy(self.PSD.astype(np.float32)); muv0 = torch.from_numpy(PHI_MU.astype(np.float32))
                with torch.no_grad():
                    pr = self.Mc(torch.cat([v, at, (self.phi_hat * sdv0 + muv0)[:, None]], -1))[:, 0].numpy()
                z = (((dobs - pr) / np.array(self.cfg["resid_sd_c"])) ** 2).mean(1)
                self.S = np.maximum(0, self.S + z - 1.5); hit = self.S > 10
                if hit.any():
                    self.phi_hat[torch.from_numpy(hit)] = 0.0; self.S[hit] = 0
                    self.reset_mask = hit
            self.buf.append((s[:, 2:].copy(), a.copy(), dobs))
            self.buf = self.buf[-self.cfg["window"]:]
            if self.cd:
                # per-episode effective window: transitions since last detected change (min 3)
                self.since = np.where(getattr(self, "reset_mask", np.zeros(self.E, bool)), 0,
                                      getattr(self, "since", np.zeros(self.E, int))) + 1
                self.reset_mask = np.zeros(self.E, bool)
            Vb = torch.from_numpy(np.stack([b[0] for b in self.buf], 1).astype(np.float32))
            Ab = torch.from_numpy(np.stack([b[1] for b in self.buf], 1).astype(np.float32))
            Db = torch.from_numpy(np.stack([b[2] for b in self.buf], 1))
            ph = self.phi_hat.clone().requires_grad_(True)
            opt = torch.optim.Adam([ph], lr=0.05)
            sdv = torch.from_numpy(self.PSD.astype(np.float32)); muv = torch.from_numpy(PHI_MU.astype(np.float32))
            for _ in range(self.cfg["latent_steps"]):
                phi = ph * sdv + muv
                pred = self.Mc(torch.cat([Vb, Ab, phi[:, None].expand(-1, Vb.shape[1], -1)], -1))
                err = (((pred - Db) / self.Mc.out_sd) ** 2).mean(-1)          # (E, W)
                if self.cd:
                    W = err.shape[1]
                    keep = torch.from_numpy((np.arange(W)[None, :] >= W - np.maximum(self.since, 3)[:, None]).astype(np.float32))
                    err = (err * keep).sum(1) / keep.sum(1)
                else:
                    err = err.mean(1)
                loss = err.sum() + 1e-3 * (ph ** 2).sum()
                opt.zero_grad(); loss.backward(); opt.step()
            self.phi_hat = ph.detach()


def make_schedule(regime, rng, E, T, seg=40, switch=55, wind_sd=1.0):
    goals = rng.uniform(-1, 1, (T // seg, E, 2)).repeat(seg, 0)
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


def run_regime(kind, Mu, Mc, goals, sched, cfg, seed):
    E, T = sched.shape[1], sched.shape[0]
    env = PuckBatch(E)
    s = env.reset()
    pl = Planner(kind, Mu, Mc, E, cfg, np.random.default_rng(seed))
    cost = np.zeros((T, E))
    K = cfg.get("K", 1)
    for t in range(T):
        env.set_phi(sched[t]); pl.phi_true = sched[t]
        if t % K == 0:
            plan = pl.act(s, goals[t])
        a = plan[:, t % K]
        s2 = env.step(a)
        cost[t] = ((s2[:, :2] - goals[t]) ** 2).sum(1) + 0.01 * (np.clip(a, -1, 1) ** 2).sum(1)
        pl.observe(s, np.clip(a, -1, 1), s2)
        s = s2
    seg = 40
    ss = np.sqrt(np.stack([cost[k * seg + seg - 10:(k + 1) * seg] for k in range(T // seg)]).mean((0, 1)))
    return cost.mean(0), ss  # per-episode mean cost, per-episode steady-state RMS error proxy
