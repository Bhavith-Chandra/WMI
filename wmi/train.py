"""Training loops. Only ever pass *train* episodes for fitting; val for model selection."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import Trajectories
from .env import KeyGrid
from .models import LatentWorldModel, RecurrentWorldModel, TemporalDistance


def _local_targets(env: KeyGrid, traj: Trajectories) -> np.ndarray:
    return traj.obs_pomdp[:, 1:, env.full_dim:env.full_dim + env.local_dim]


@torch.no_grad()
def recurrent_loss(model, env, traj) -> float:
    obs = torch.from_numpy(traj.obs_pomdp)
    logits, _ = model(obs, torch.from_numpy(traj.actions))
    return float(F.binary_cross_entropy_with_logits(logits, torch.from_numpy(_local_targets(env, traj))))


def train_recurrent(model: RecurrentWorldModel, env: KeyGrid, train: Trajectories, val: Trajectories,
                    steps: int, rng: np.random.Generator, batch: int = 128, lr: float = 3e-3,
                    eval_every: int = 100) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    obs = torch.from_numpy(train.obs_pomdp)
    act = torch.from_numpy(train.actions)
    tgt = torch.from_numpy(_local_targets(env, train))
    hist = {"step": [], "val_bce": []}
    best, best_state = float("inf"), None
    for step in range(1, steps + 1):
        idx = torch.from_numpy(rng.integers(0, len(obs), batch))
        logits, _ = model(obs[idx], act[idx])
        loss = F.binary_cross_entropy_with_logits(logits, tgt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            v = recurrent_loss(model, env, val)
            hist["step"].append(step)
            hist["val_bce"].append(v)
            if v < best:
                best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)  # early-stopped on VAL, never test
    hist["best_val_bce"] = best
    return hist


def _windows(traj: Trajectories, k: int):
    E, T = traj.actions.shape
    starts = np.arange(T - k + 1)
    o = np.stack([traj.obs_full[:, s:s + k + 1] for s in starts], 1).reshape(-1, k + 1, traj.obs_full.shape[-1])
    a = np.stack([traj.actions[:, s:s + k] for s in starts], 1).reshape(-1, k)
    return torch.from_numpy(o), torch.from_numpy(a)


def latent_loss(model: LatentWorldModel, o, a):
    z = model.encode(o[:, 0])
    loss = F.binary_cross_entropy_with_logits(model.decode(z), o[:, 0])
    for t in range(a.shape[1]):
        z = model.predict(z, a[:, t])
        loss = loss + F.binary_cross_entropy_with_logits(model.decode(z), o[:, t + 1])
        loss = loss + F.mse_loss(z, model.encode(o[:, t + 1]).detach())
    return loss / (a.shape[1] + 1)


def train_latent(model: LatentWorldModel, train: Trajectories, val: Trajectories, steps: int,
                 rng: np.random.Generator, k: int = 5, batch: int = 256, lr: float = 3e-3,
                 eval_every: int = 100) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    o, a = _windows(train, k)
    vo, va = _windows(val, k)
    hist = {"step": [], "val_loss": []}
    best, best_state = float("inf"), None
    for step in range(1, steps + 1):
        idx = torch.from_numpy(rng.integers(0, len(o), batch))
        loss = latent_loss(model, o[idx], a[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                v = float(latent_loss(model, vo, va))
            hist["step"].append(step)
            hist["val_loss"].append(v)
            if v < best:
                best, best_state = v, {kk: t.clone() for kk, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    hist["best_val_loss"] = best
    return hist


def hindsight_pairs(traj: Trajectories, max_k: int, rng: np.random.Generator, n: int):
    """Hindsight relabeling: goal := a state actually reached k steps later in the same episode.

    Label is k (temporal offset along the behaviour trajectory). Pairs never cross episodes.
    """
    E, T1 = traj.obs_full.shape[:2]
    e = rng.integers(0, E, n)
    k = rng.integers(0, max_k + 1, n)
    t = rng.integers(0, T1 - k)
    return traj.obs_full[e, t], traj.obs_full[e, t + k], k.astype(np.float32)


def train_temporal_distance(dist: TemporalDistance, wm: LatentWorldModel, train: Trajectories,
                            val: Trajectories, steps: int, rng: np.random.Generator, max_k: int = 12,
                            batch: int = 512, lr: float = 1e-3, eval_every: int = 100) -> dict:
    """Regress d(enc(o_t), enc(o_{t+k})) onto k with a frozen encoder.

    Random-walk offsets over-estimate shortest-path distance; an expectile-style
    asymmetric loss (tau < 0.5) biases d toward the lower envelope (~shortest path).
    """
    tau = 0.1
    opt = torch.optim.Adam(dist.parameters(), lr=lr)
    vo, vg, vk = hindsight_pairs(val, max_k, np.random.default_rng(12345), 4096)
    with torch.no_grad():
        vz, vgz = wm.encode(torch.from_numpy(vo)), wm.encode(torch.from_numpy(vg))
    vk = torch.from_numpy(vk)

    def loss_fn(pred, k):
        u = k - pred
        w = torch.where(u < 0, 1 - tau, tau)
        return (w * u.pow(2)).mean()

    hist = {"step": [], "val_loss": []}
    best, best_state = float("inf"), None
    for step in range(1, steps + 1):
        o, g, k = hindsight_pairs(train, max_k, rng, batch)
        with torch.no_grad():
            z, gz = wm.encode(torch.from_numpy(o)), wm.encode(torch.from_numpy(g))
        loss = loss_fn(dist(z, gz), torch.from_numpy(k))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                v = float(loss_fn(dist(vz, vgz), vk))
            hist["step"].append(step)
            hist["val_loss"].append(v)
            if v < best:
                best, best_state = v, {kk: t.clone() for kk, t in dist.state_dict().items()}
    dist.load_state_dict(best_state)
    hist["best_val_loss"] = best
    return hist
