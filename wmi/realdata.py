"""Hindsight vs forward prediction on REAL measured system-identification benchmarks.

What this tests (and what it does not): offline logs cannot be replanned, so we test the quantity the
theory says drives planning regret: the error of the world model's prediction of the consequence of the
action actually applied, one step and k steps ahead, after observing the outcomes so far.

World model: NARX MLP, y_{t+1} = f(y_{t-n+1..t}, u_{t-n+1..t+? }) trained ONLY on the official train split.
Hindsight model (same frozen f): y_{t+1} = y_t + (1 + theta) (f_t - y_t) + w + eps, i.e. a persistent GAIN
fault theta on the predicted increment and an ADDITIVE fault w, with a Gaussian posterior over (theta, w)
updated from every observed transition (exactly the posterior of wmi/theory.py).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .harness.leakage import LeakageError


class Norm:
    def __init__(self, u, y):
        self.mu_u, self.sd_u = float(np.mean(u)), float(np.std(u) + 1e-12)
        self.mu_y, self.sd_y = float(np.mean(y)), float(np.std(y) + 1e-12)

    def nu(self, u):
        return (np.asarray(u) - self.mu_u) / self.sd_u

    def ny(self, y):
        return (np.asarray(y) - self.mu_y) / self.sd_y


def windows(u, y, n):
    """Features for predicting y[t+1]: y[t-n+1..t], u[t-n+1..t] (strictly causal). Targets y[t+1]."""
    T = len(y)
    idx = np.arange(n - 1, T - 1)
    Y = np.stack([y[idx - k] for k in range(n)], 1)
    U = np.stack([u[idx - k] for k in range(n)], 1)
    return np.concatenate([Y, U], 1).astype(np.float32), y[idx + 1].astype(np.float32), idx


class NARX(nn.Module):
    def __init__(self, n, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2 * n, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1) + x[:, 0]  # residual on y_t


def fit_narx(train_recs, val_recs, norm: Norm, n: int, steps: int, rng, lr=1e-3, batch=256):
    X = np.concatenate([windows(norm.nu(r[0]), norm.ny(r[1]), n)[0] for r in train_recs])
    Y = np.concatenate([windows(norm.nu(r[0]), norm.ny(r[1]), n)[1] for r in train_recs])
    Xv = np.concatenate([windows(norm.nu(r[0]), norm.ny(r[1]), n)[0] for r in val_recs])
    Yv = np.concatenate([windows(norm.nu(r[0]), norm.ny(r[1]), n)[1] for r in val_recs])
    X, Y, Xv, Yv = map(torch.from_numpy, (X, Y, Xv, Yv))
    m = NARX(n)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    best, best_state = np.inf, None
    for s in range(1, steps + 1):
        i = torch.from_numpy(rng.integers(0, len(X), batch))
        loss = ((m(X[i]) - Y[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 200 == 0 or s == steps:
            with torch.no_grad():
                v = float(((m(Xv) - Yv) ** 2).mean())
            if v < best:
                best, best_state = v, {k: t.clone() for k, t in m.state_dict().items()}
    m.load_state_dict(best_state)
    with torch.no_grad():
        sigma_val = float(torch.sqrt(((m(Xv) - Yv) ** 2).mean()))
    return m.eval(), {"val_rmse_norm": sigma_val, "n_train_windows": int(len(X))}


class GainOffsetPosterior:
    """Bayesian linear regression for (theta, w) in  dy - df = theta * df + w + eps,  df = f_t - y_t."""

    def __init__(self, s_theta, s_w, sigma, forget=1.0):
        self.P0 = np.diag([1 / s_theta ** 2, 1 / s_w ** 2])
        self.P, self.b = self.P0.copy(), np.zeros(2)
        self.sigma2, self.forget = sigma ** 2, forget

    def mean(self):
        return np.linalg.solve(self.P, self.b)

    def update(self, df, resid):
        if self.forget < 1.0:
            self.P = self.forget * self.P + (1 - self.forget) * self.P0
            self.b = self.forget * self.b
        x = np.array([df, 1.0])
        self.P = self.P + np.outer(x, x) / self.sigma2
        self.b = self.b + x * resid / self.sigma2


def stream_eval(model: NARX, u_n, y_n, n, k, cfg, rng, stride=10):
    """Causal streaming over ONE test record (normalised units). Returns per-step squared errors."""
    X, Y, idx = windows(u_n, y_n, n)
    with torch.no_grad():
        F = model(torch.from_numpy(X)).numpy()
    yt = X[:, 0]
    N = len(Y)
    methods = ("forward", "residual", "hindsight", "hindsight_forget", "online_sgd", "naive_last", "naive_velocity")
    pred1 = {m: np.zeros(N) for m in methods}
    post = GainOffsetPosterior(cfg["s_theta"], cfg["s_w"], cfg["sigma"])
    postf = GainOffsetPosterior(cfg["s_theta"], cfg["s_w"], cfg["sigma"], forget=cfg["forget"])
    phis, phisf, res_hist = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros(N)
    # online SGD (AdaJEPA-style): copy of the model, last layer only, one step per observed transition
    sgd = NARX(n); sgd.load_state_dict(model.state_dict())
    for p in sgd.body.parameters():
        p.requires_grad_(False)
    opt = torch.optim.SGD(sgd.head.parameters(), lr=cfg["sgd_lr"])
    last_res = 0.0
    Xt = torch.from_numpy(X)
    for t in range(N):
        df = F[t] - yt[t]
        th, w = post.mean(); thf, wf = postf.mean()
        phis[t], phisf[t], res_hist[t] = (th, w), (thf, wf), last_res
        pred1["forward"][t] = F[t]
        pred1["residual"][t] = F[t] + last_res
        pred1["hindsight"][t] = yt[t] + (1 + th) * df + w
        pred1["hindsight_forget"][t] = yt[t] + (1 + thf) * df + wf
        xt = Xt[t:t + 1]
        out = sgd(xt)
        pred1["online_sgd"][t] = float(out)
        pred1["naive_last"][t] = yt[t]                       # model-free: y_{t+1} = y_t
        pred1["naive_velocity"][t] = 2 * yt[t] - X[t, 1]     # model-free: constant-velocity extrapolation
        # ---- outcome y_{t+1} revealed; update every method's memory (strictly after predicting)
        loss = (out - torch.tensor([Y[t]])).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        last_res = float(Y[t] - F[t])
        post.update(df, Y[t] - F[t]); postf.update(df, Y[t] - F[t])
    se1 = {m: (pred1[m] - Y) ** 2 for m in methods}

    # ---- k-step-ahead: from start t, roll the model k steps on its own outputs with the TRUE future inputs,
    # using only information available at t (posterior / residual / weights frozen at t).
    starts = np.arange(0, N - k, stride)
    sek = {m: np.zeros(len(starts)) for m in methods if m not in ("online_sgd", "naive_last", "naive_velocity")}
    for m in sek:
        ybuf = np.stack([X[starts, j] for j in range(n)], 1)  # y_t, y_{t-1}, ...
        for s in range(k):
            ub = np.stack([X[starts + s, n + j] for j in range(n)], 1)
            feats = np.concatenate([ybuf, ub], 1).astype(np.float32)
            with torch.no_grad():
                f = model(torch.from_numpy(feats)).numpy()
            yprev = ybuf[:, 0]
            if m == "forward":
                ynext = f
            elif m == "residual":
                ynext = f + res_hist[starts]
            else:
                ph = phis[starts] if m == "hindsight" else phisf[starts]
                ynext = yprev + (1 + ph[:, 0]) * (f - yprev) + ph[:, 1]
            ybuf = np.concatenate([ynext[:, None], ybuf[:, :-1]], 1)
        sek[m] = (ybuf[:, 0] - Y[starts + k - 1]) ** 2
    y0, y1 = X[starts, 0], X[starts, 1]
    sek["naive_last"] = (y0 - Y[starts + k - 1]) ** 2
    sek["naive_velocity"] = (y0 + k * (y0 - y1) - Y[starts + k - 1]) ** 2
    return se1, sek, {"theta_final": float(phis[-1, 0]), "w_final": float(phis[-1, 1])}


def block_bootstrap_diff(a, b, rng, block=500, n_boot=1000):
    """95% CI of mean(a) - mean(b) for paired per-step series, moving-block bootstrap (time correlation)."""
    d = a - b
    T = len(d)
    nb = max(1, T // block)
    starts_max = max(1, T - block)
    stats = []
    for _ in range(n_boot):
        s = rng.integers(0, starts_max, nb)
        stats.append(np.concatenate([d[i:i + block] for i in s]).mean())
    return [float(x) for x in np.quantile(stats, [0.025, 0.975])]


def assert_no_test_in_train(train_recs, test_recs):
    for tr in train_recs:
        for te in test_recs:
            if len(tr[1]) == len(te[1]) and np.array_equal(tr[1], te[1]):
                raise LeakageError("a test record is identical to a train record")
