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


def stream_eval(model: NARX, u_n, y_n, n, k, cfg, rng, stride=10, extra_est=None):
    """Causal streaming over ONE test record (normalised units). Returns per-step squared errors."""
    X, Y, idx = windows(u_n, y_n, n)
    with torch.no_grad():
        F = model(torch.from_numpy(X)).numpy()
    yt = X[:, 0]
    N = len(Y)
    methods = ("forward", "residual", "hindsight", "hindsight_forget", "online_sgd", "naive_last", "naive_velocity") + \
        (("hindsight_imm",) if extra_est is not None else ())
    if extra_est is not None:
        imm = extra_est(1, np.zeros(2), np.diag([cfg["s_theta"] ** 2, cfg["s_w"] ** 2]), cfg["sigma"])
        phis_imm = np.zeros((len(windows(u_n, y_n, n)[1]), 2))
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
        if extra_est is not None:
            mi = imm.belief()[0][0]
            phis_imm[t] = mi
            pred1["hindsight_imm"][t] = yt[t] + (1 + mi[0]) * df + mi[1]
            imm.update(np.array([[df, 1.0]]), np.array([Y[t] - F[t]]))
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
                ph = {"hindsight": phis, "hindsight_forget": phisf}.get(m, phis_imm if extra_est is not None else None)[starts]
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


def stream_generic(model: NARX, u_n, y_n, n, k, est_module, cfg, stride=10, max_steps=None):
    """Causal streaming evaluation shared by autoresearch dev (eval_v3) and the sealed test.

    Methods: 'forward' (frozen model), 'sgd' (AdaJEPA-style: last layer, 1 SGD step per transition),
    'candidate' (est_module: BASE in {'frozen','sgd'} + Estimator posterior over gain/offset of the base's
    predicted increment). k-step rollouts start every `stride` steps and use ONLY information available at the
    start time (weights and belief frozen at t; true future inputs, own predicted outputs).
    Returns per-step squared errors: se1[m] (N,), sek[m] (n_starts,).
    """
    X, Y, _ = windows(u_n, y_n, n)
    if max_steps:
        X, Y = X[:max_steps], Y[:max_steps]
    N = len(Y)
    base_kind = getattr(est_module, "BASE", "frozen")
    lr = getattr(est_module, "SGD_LR", cfg["sgd_lr"])

    def make_sgd():
        m = NARX(n); m.load_state_dict(model.state_dict())
        for p in m.body.parameters():
            p.requires_grad_(False)
        return m, torch.optim.SGD(m.head.parameters(), lr=lr)

    sgd, opt_s = make_sgd()                    # reference online-SGD model
    cand_base, opt_c = make_sgd() if base_kind == "sgd" else (model, None)
    est = est_module.Estimator(1, np.zeros(2), np.diag([cfg["s_theta"] ** 2, cfg["s_w"] ** 2]), cfg["sigma"])
    se1 = {m: np.zeros(N) for m in ("forward", "sgd", "candidate")}
    starts = [t for t in range(0, N - k, stride)]
    sek = {m: [] for m in ("forward", "sgd", "candidate")}
    Xt = torch.from_numpy(X)

    def rollout(net, t, phi=None):
        ybuf = X[t, :n].copy()
        for s in range(k):
            feats = np.concatenate([ybuf, X[t + s, n:]]).astype(np.float32)[None]
            with torch.no_grad():
                f = float(net(torch.from_numpy(feats)))
            yprev = ybuf[0]
            ynext = f if phi is None else yprev + (1 + phi[0]) * (f - yprev) + phi[1]
            ybuf = np.concatenate([[ynext], ybuf[:-1]])
        return (ybuf[0] - Y[t + k - 1]) ** 2

    for t in range(N):
        xt = Xt[t:t + 1]
        with torch.no_grad():
            f_fwd = float(model(xt))
            f_cb = float(cand_base(xt))
        mean, _ = est.belief()
        phi = mean[0]
        yt = X[t, 0]
        df = f_cb - yt
        pred_c = yt + (1 + phi[0]) * df + phi[1]
        out_s = sgd(xt)
        se1["forward"][t] = (f_fwd - Y[t]) ** 2
        se1["sgd"][t] = (float(out_s) - Y[t]) ** 2
        se1["candidate"][t] = (pred_c - Y[t]) ** 2
        if t % stride == 0 and t < N - k:
            sek["forward"].append(rollout(model, t))
            sek["sgd"].append(rollout(sgd, t))
            sek["candidate"].append(rollout(cand_base, t, phi))
        # ---- outcome revealed: update everything strictly after predicting
        loss = (out_s - torch.tensor([Y[t]])).pow(2).mean()
        opt_s.zero_grad(); loss.backward(); opt_s.step()
        if opt_c is not None:
            lc = (cand_base(xt) - torch.tensor([Y[t]])).pow(2).mean()
            opt_c.zero_grad(); lc.backward(); opt_c.step()
        est.update(np.array([[df, 1.0]]), np.array([Y[t] - f_cb]))
    return se1, {m: np.array(v) for m, v in sek.items()}
