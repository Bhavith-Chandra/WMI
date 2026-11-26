"""E-A Throw-and-retry: one-shot open-loop decisions with persistent hidden damping and wind (v6 pre-registration)."""
from __future__ import annotations

import mujoco
import numpy as np
import torch
import torch.nn as nn

XML = """<mujoco><option timestep="0.01"/><worldbody><body name="puck">
<joint name="x" type="slide" axis="1 0 0" damping="1"/><joint name="y" type="slide" axis="0 1 0" damping="1"/>
<geom type="sphere" size="0.05" mass="1"/></body></worldbody></mujoco>"""
STEPS = 200  # 2.0 s


class Thrower:
    def __init__(self, E):
        self.m = [mujoco.MjModel.from_xml_string(XML) for _ in range(E)]
        self.d = [mujoco.MjData(m) for m in self.m]; self.body = self.m[0].body("puck").id; self.E = E

    def throw(self, v0, phi):
        out = np.zeros((self.E, 2))
        for e in range(self.E):
            m, d = self.m[e], self.d[e]
            m.dof_damping[:] = phi[e, 0]; d.qpos[:] = 0; d.qvel[:] = v0[e]; d.xfrc_applied[self.body, :2] = phi[e, 1:]
            mujoco.mj_forward(m, d)
            for _ in range(STEPS):
                mujoco.mj_step(m, d)
            out[e] = d.qpos
        return out


def sample_phi(rng, n, broad=False):
    lo, hi, ws = (0.3, 2.5, 0.75) if broad else (0.5, 2.0, 0.5)
    return np.stack([rng.uniform(lo, hi, n), ws * rng.standard_normal(n), ws * rng.standard_normal(n)], 1)


class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))
        for k, n in (("mu", n_in), ("sd", n_in), ("omu", 2), ("osd", 2)):
            self.register_buffer(k, torch.zeros(n))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd) * self.osd + self.omu


def fit_ens(X, Y, rng, steps=4000, M=5):
    X, Y = X.astype(np.float32), Y.astype(np.float32)
    nva = len(X) // 10; idx = rng.permutation(len(X)); va, tr = idx[:nva], idx[nva:]
    members, sds = [], []
    for _ in range(M):
        torch.manual_seed(int(rng.integers(1 << 30)))
        b = rng.choice(tr, len(tr), replace=True)
        net = MLP(X.shape[1])
        net.mu.copy_(torch.tensor(X[tr].mean(0))); net.sd.copy_(torch.tensor(X[tr].std(0) + 1e-6))
        net.omu.copy_(torch.tensor(Y[tr].mean(0))); net.osd.copy_(torch.tensor(Y[tr].std(0) + 1e-6))
        opt = torch.optim.Adam(net.parameters(), 1e-3); Xt, Yt = torch.tensor(X[b]), torch.tensor(Y[b])
        for s in range(steps):
            i = torch.tensor(rng.integers(0, len(Xt), 512))
            loss = (((net(Xt[i]) - Yt[i]) / net.osd) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            sds.append((net(torch.tensor(X[va])) - torch.tensor(Y[va])).std(0).numpy())
        members.append(net)
    return members, np.mean(sds, 0)


@torch.no_grad()
def ens(members, x):
    return torch.stack([m(x) for m in members]).mean(0)


def train_models(rng, n=20000, E=200):
    th = Thrower(E); V, P, Y = [], [], []
    for _ in range(n // E):
        v0 = rng.uniform(-4, 4, (E, 2)); phi = sample_phi(rng, E)
        Y.append(th.throw(v0, phi)); V.append(v0); P.append(phi)
    V, P, Y = map(np.concatenate, (V, P, Y))
    Mu, sdu = fit_ens(V, Y, rng); Mc, sdc = fit_ens(np.c_[V, P], Y, rng)
    return Mu, Mc, sdu, sdc


@torch.no_grad()
def cem(cost_fn, E, rng, P=400, iters=5):
    mu = np.zeros((E, 2)); sd = np.full((E, 2), 2.0)
    for _ in range(iters):
        A = np.clip(mu[:, None] + sd[:, None] * rng.standard_normal((E, P, 2)), -4, 4)
        c = cost_fn(A)                                             # (E,P)
        el = np.argsort(c, 1)[:, :40]; EA = np.take_along_axis(A, el[..., None], 1)
        mu, sd = EA.mean(1), EA.std(1) + 0.02
    return mu


def run_episode(kind, Mu, Mc, sdc, phis, targets, rng, N=1000, S=16):
    """phis: (A, E, 3) per attempt; targets: (A, E, 2). Returns miss (A, E)."""
    A, E = targets.shape[:2]; th = Thrower(E); miss = np.zeros((A, E)); r_last = np.zeros((E, 2))
    parts = sample_phi(rng, E * N, broad=True).reshape(E, N, 3); logw = np.zeros((E, N))
    corr = np.zeros((E, 2)); gaps = []                         # goal-gap correction state (v9)
    HV, HY = [], []                                           # executed actions and achieved outcomes (relabeling)

    def anchor(tg):
        """Relabeling: the past attempt whose ACHIEVED outcome is closest to the current goal (per episode)."""
        Y, V = np.stack(HY, 1), np.stack(HV, 1)               # (E,k,2)
        k = np.argmin(((Y - tg[:, None]) ** 2).sum(-1), 1)
        return V[np.arange(E), k], Y[np.arange(E), k]
    nv = np.tile((2 * sdc) ** 2, (E, 1))
    for k in range(A):
        tg = targets[k]
        if kind == "forward":
            f = lambda Aa: ((ens(Mu, torch.tensor(Aa.reshape(-1, 2), dtype=torch.float32)).numpy().reshape(E, -1, 2) - tg[:, None]) ** 2).sum(-1)
        elif kind == "residual":
            f = lambda Aa: ((ens(Mu, torch.tensor(Aa.reshape(-1, 2), dtype=torch.float32)).numpy().reshape(E, -1, 2) + r_last[:, None] - tg[:, None]) ** 2).sum(-1)
        elif kind == "anchor" and HY:                         # predicted(a) = y* + f(a) - f(a*): model only predicts the difference
            va, ya = anchor(tg); ra = ya - ens(Mu, torch.tensor(va, dtype=torch.float32)).numpy()
            f = lambda Aa, ra=ra: ((ens(Mu, torch.tensor(Aa.reshape(-1, 2), dtype=torch.float32)).numpy().reshape(E, -1, 2) + ra[:, None] - tg[:, None]) ** 2).sum(-1)
        elif kind == "anchor":
            f = lambda Aa: ((ens(Mu, torch.tensor(Aa.reshape(-1, 2), dtype=torch.float32)).numpy().reshape(E, -1, 2) - tg[:, None]) ** 2).sum(-1)
        elif kind in ("gap_ilc", "gap_mean"):                 # aim so that predicted outcome + correction = target
            f = lambda Aa: ((ens(Mu, torch.tensor(Aa.reshape(-1, 2), dtype=torch.float32)).numpy().reshape(E, -1, 2) + corr[:, None] - tg[:, None]) ** 2).sum(-1)
        elif kind == "oracle":
            def f(Aa):
                Pp = Aa.shape[1]; ph = np.repeat(phis[k][:, None], Pp, 1)
                x = torch.tensor(np.c_[Aa.reshape(-1, 2), ph.reshape(-1, 3)], dtype=torch.float32)
                return ((ens(Mc, x).numpy().reshape(E, Pp, 2) - tg[:, None]) ** 2).sum(-1)
        else:  # hindsight: posterior-expected cost over S particle samples (+ relabeling anchor for hindsight_anchor)
            w = np.exp(logw - logw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
            idx = np.stack([rng.choice(N, S, p=w[e]) for e in range(E)]); samp = np.take_along_axis(parts, idx[..., None], 1)
            ca = np.zeros((E, S, 2))
            if kind == "hindsight_anchor" and HY:             # anchor residual under each posterior sample
                va, ya = anchor(tg)
                xa = torch.tensor(np.c_[np.repeat(va, S, 0), samp.reshape(-1, 3)], dtype=torch.float32)
                ca = ya[:, None] - ens(Mc, xa).numpy().reshape(E, S, 2)
            def f(Aa, samp=samp, ca=ca):
                Pp = Aa.shape[1]
                a = np.repeat(Aa[:, :, None], S, 2); ph = np.repeat(samp[:, None], Pp, 1)
                x = torch.tensor(np.c_[a.reshape(-1, 2), ph.reshape(-1, 3)], dtype=torch.float32)
                return ((ens(Mc, x).numpy().reshape(E, Pp, S, 2) + ca[:, None] - tg[:, None, None]) ** 2).sum(-1).mean(-1)
        v0 = cem(f, E, rng)
        y = th.throw(v0, phis[k]); miss[k] = np.linalg.norm(y - tg, axis=1)
        r_last = y - ens(Mu, torch.tensor(v0, dtype=torch.float32)).numpy()
        if kind == "gap_ilc":                                 # integral update on the outcome-to-goal gap
            corr = corr + (y - tg)
        if kind == "gap_mean":                                # mean of past gaps between outcome and intended outcome
            gaps.append(r_last); corr = np.mean(gaps, 0)
        HV.append(v0.copy()); HY.append(y.copy())
        if kind in ("hindsight", "hindsight_anchor"):
            x = torch.tensor(np.c_[np.repeat(v0, N, 0), parts.reshape(-1, 3)], dtype=torch.float32)
            pred = ens(Mc, x).numpy().reshape(E, N, 2)
            w = np.exp(logw - logw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
            pm = np.einsum("en,end->ed", w, pred)
            nv = 0.98 * nv + 0.02 * (y - pm) ** 2
            sd = np.sqrt(np.maximum(nv, (2 * sdc) ** 2))[:, None]
            logw += -0.5 * (((y[:, None] - pred) / sd) ** 2).sum(-1); logw -= logw.max(1, keepdims=True)
            w = np.exp(logw); w /= w.sum(1, keepdims=True)
            for e in np.flatnonzero(1 / (w ** 2).sum(1) < N / 2):
                u = (rng.random() + np.arange(N)) / N; ii = np.minimum(np.searchsorted(np.cumsum(w[e]), u), N - 1)
                parts[e] = parts[e, ii]; logw[e] = 0
            rej = rng.random((E, N)) < 0.02
            if rej.any():
                parts[rej] = sample_phi(rng, int(rej.sum()), broad=True)
    return miss


def schedule(regime, rng, E, A=10):
    targets = np.stack([rng.uniform([1, -1], [3, 1], (E, 2)) for _ in range(A)])
    def draw():
        ph = sample_phi(rng, E)
        if regime == "wind":
            ph[:, 0] = 1.0
        if regime == "damping":
            ph[:, 1:] = 0.0
        return ph
    if regime == "fresh":
        return np.stack([draw() for _ in range(A)]), targets
    return np.repeat(draw()[None], A, 0), targets
