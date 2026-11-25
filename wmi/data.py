"""Trajectory collection and episode-level splits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import N_ACTIONS, KeyGrid
from .harness.leakage import check_split


@dataclass
class Trajectories:
    states: np.ndarray       # (E, T+1, 3) int64
    actions: np.ndarray      # (E, T)      int64
    obs_full: np.ndarray     # (E, T+1, full_dim)
    obs_pomdp: np.ndarray    # (E, T+1, pomdp_dim)
    episode_ids: np.ndarray  # (E,)

    @property
    def n_episodes(self) -> int:
        return len(self.episode_ids)

    def subset(self, ep_idx) -> "Trajectories":
        ep_idx = np.asarray(ep_idx)
        return Trajectories(self.states[ep_idx], self.actions[ep_idx], self.obs_full[ep_idx],
                            self.obs_pomdp[ep_idx], self.episode_ids[ep_idx])


def collect(env: KeyGrid, n_episodes: int, horizon: int, rng: np.random.Generator,
            id_offset: int = 0) -> Trajectories:
    """Uniform-random behaviour policy. Episode ids are unique integers."""
    s = env.sample_states(rng, n_episodes)
    acts = rng.integers(0, N_ACTIONS, size=(n_episodes, horizon))
    states = [s]
    for t in range(horizon):
        s = env.step(s, acts[:, t])
        states.append(s)
    states = np.stack(states, 1)
    E, T1 = states.shape[:2]
    flat = states.reshape(-1, 3)
    tt = np.tile(np.arange(T1), E)
    obs_full = env.render_full(flat).reshape(E, T1, -1)
    obs_pomdp = env.render_pomdp(flat, tt).reshape(E, T1, -1)
    return Trajectories(states, acts, obs_full, obs_pomdp, np.arange(n_episodes) + id_offset)


def split_episodes(traj: Trajectories, fractions=(0.7, 0.15, 0.15), rng=None) -> dict:
    """Random episode-level train/val/test split; disjointness is asserted."""
    rng = rng if rng is not None else np.random.default_rng(0)
    assert abs(sum(fractions) - 1) < 1e-9
    perm = rng.permutation(traj.n_episodes)
    n_tr = int(round(fractions[0] * len(perm)))
    n_va = int(round(fractions[1] * len(perm)))
    split = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    check_split({k: traj.episode_ids[v] for k, v in split.items()}, n_total=traj.n_episodes)
    return split
