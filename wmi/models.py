"""World models with explicit activation caches (hook sites are named)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentWorldModel(nn.Module):
    """Deterministic recurrent world model for the POMDP observation stream.

    h_t = GRU(h_{t-1}, [enc(o_t), onehot(a_{t-1})]);  predict local view o_{t+1} from (h_t, a_t).
    Hook sites: "enc" (e_t), "h" (h_t).
    """

    SITES = ("enc", "h")

    def __init__(self, obs_dim: int, n_actions: int, pred_dim: int, latent: int = 64, hidden: int = 128):
        super().__init__()
        self.n_actions = n_actions
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.gru = nn.GRUCell(hidden + n_actions, latent)
        self.head = nn.Sequential(nn.Linear(latent + n_actions, hidden), nn.ReLU(), nn.Linear(hidden, pred_dim))
        self.latent = latent

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        """obs (B, T+1, D), actions (B, T). Returns logits (B, T, pred_dim) and cache."""
        B, T1, _ = obs.shape
        h = obs.new_zeros(B, self.latent)
        a_prev = obs.new_zeros(B, self.n_actions)
        enc_all, h_all = [], []
        for t in range(T1):
            e = self.enc(obs[:, t])
            h = self.gru(torch.cat([e, a_prev], -1), h)
            enc_all.append(e)
            h_all.append(h)
            if t < T1 - 1:
                a_prev = F.one_hot(actions[:, t], self.n_actions).float()
        h_all = torch.stack(h_all, 1)
        a_oh = F.one_hot(actions, self.n_actions).float()
        logits = self.head(torch.cat([h_all[:, :-1], a_oh], -1))
        return logits, {"enc": torch.stack(enc_all, 1), "h": h_all}

    def predict_from_h(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([h, F.one_hot(a, self.n_actions).float()], -1))


class LatentWorldModel(nn.Module):
    """Feed-forward latent dynamics model on fully observed frames (used for planning).

    z = enc(o); z' = z + dyn([z, onehot(a)]); dec(z) -> obs logits.
    """

    def __init__(self, obs_dim: int, n_actions: int, latent: int = 32, hidden: int = 128):
        super().__init__()
        self.n_actions = n_actions
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.dyn = nn.Sequential(nn.Linear(latent + n_actions, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, obs_dim))

    def encode(self, o):
        return self.enc(o)

    def predict(self, z, a):
        return z + self.dyn(torch.cat([z, F.one_hot(a, self.n_actions).float()], -1))

    def decode(self, z):
        return self.dec(z)


class TemporalDistance(nn.Module):
    """d(z, g) >= 0, trained by hindsight relabeling (see planning.py)."""

    def __init__(self, latent: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * latent, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, z, g):
        return F.softplus(self.net(torch.cat([z, g, z - g], -1))).squeeze(-1)
