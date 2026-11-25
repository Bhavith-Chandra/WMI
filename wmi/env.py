"""KeyGrid: a small deterministic gridworld with exact ground-truth state.

State per agent: (x, y, has_key) as int64. The world is fully known, so every
latent variable a world model might represent has an exact label. That is the
point: interpretability claims can be checked against ground truth instead of
against another model.

Two observation modes:
  * full  : agent one-hot + key one-hot (key channel cleared once picked up).
  * pomdp : full view at t=0 only; afterwards only a 3x3 egocentric local view
            (blocked cells + key). Position and key status at t>0 are then NOT
            in the current observation and must be carried in memory, which is
            what makes probing the latent non-trivial.
"""
from __future__ import annotations

from collections import deque

import numpy as np

# action -> (dx, dy)
ACTIONS = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int64)
N_ACTIONS = len(ACTIONS)


class KeyGrid:
    def __init__(self, size: int = 7, wall_x: int = 3, gap_y: int = 3, key_pos=(5, 1)):
        self.size = size
        self.key_pos = tuple(key_pos)
        self.wall = np.zeros((size, size), dtype=bool)  # indexed [x, y]
        for y in range(size):
            if y != gap_y:
                self.wall[wall_x, y] = True
        assert not self.wall[self.key_pos]
        self.free_cells = np.array(
            [(x, y) for x in range(size) for y in range(size) if not self.wall[x, y]], dtype=np.int64
        )

    # ------------------------------------------------------------ dynamics
    def step(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=np.int64)
        a = np.asarray(a, dtype=np.int64)
        nxt = s.copy()
        cand = s[..., :2] + ACTIONS[a]
        inb = ((cand >= 0) & (cand < self.size)).all(-1)
        cx = np.clip(cand[..., 0], 0, self.size - 1)
        cy = np.clip(cand[..., 1], 0, self.size - 1)
        ok = inb & ~self.wall[cx, cy]
        nxt[..., :2] = np.where(ok[..., None], cand, s[..., :2])
        on_key = (nxt[..., 0] == self.key_pos[0]) & (nxt[..., 1] == self.key_pos[1])
        nxt[..., 2] = np.maximum(s[..., 2], on_key.astype(np.int64))
        return nxt

    def sample_states(self, rng: np.random.Generator, n: int, p_has_key: float = 0.5) -> np.ndarray:
        cells = self.free_cells[~((self.free_cells[:, 0] == self.key_pos[0]) & (self.free_cells[:, 1] == self.key_pos[1]))]
        idx = rng.integers(0, len(cells), size=n)
        hk = (rng.random(n) < p_has_key).astype(np.int64)
        return np.concatenate([cells[idx], hk[:, None]], axis=1)

    # ------------------------------------------------------------ rendering
    @property
    def full_dim(self) -> int:
        return 2 * self.size * self.size

    @property
    def local_dim(self) -> int:
        return 2 * 9

    @property
    def pomdp_dim(self) -> int:
        return self.full_dim + self.local_dim + 1

    def render_full(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s).reshape(-1, 3)
        b = len(s)
        o = np.zeros((b, 2, self.size, self.size), dtype=np.float32)
        o[np.arange(b), 0, s[:, 0], s[:, 1]] = 1.0
        o[:, 1, self.key_pos[0], self.key_pos[1]] = 1.0 - s[:, 2]
        return o.reshape(b, -1)

    def render_local(self, s: np.ndarray) -> np.ndarray:
        """3x3 egocentric view: channel 0 = blocked (wall/out of bounds), 1 = key visible."""
        s = np.asarray(s).reshape(-1, 3)
        b = len(s)
        o = np.zeros((b, 2, 3, 3), dtype=np.float32)
        for i, dx in enumerate((-1, 0, 1)):
            for j, dy in enumerate((-1, 0, 1)):
                x = s[:, 0] + dx
                y = s[:, 1] + dy
                inb = (x >= 0) & (x < self.size) & (y >= 0) & (y < self.size)
                xc = np.clip(x, 0, self.size - 1)
                yc = np.clip(y, 0, self.size - 1)
                o[:, 0, i, j] = (~inb | self.wall[xc, yc]).astype(np.float32)
                o[:, 1, i, j] = (inb & (xc == self.key_pos[0]) & (yc == self.key_pos[1]) & (s[:, 2] == 0)).astype(np.float32)
        return o.reshape(b, -1)

    def render_pomdp(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        s = np.asarray(s).reshape(-1, 3)
        t = np.broadcast_to(np.asarray(t), (len(s),))
        first = (t == 0)
        full = self.render_full(s) * first[:, None]
        return np.concatenate([full, self.render_local(s), first[:, None].astype(np.float32)], axis=1)

    # ------------------------------------------------------------ oracle
    def shortest_path_len(self, start_xy, goal_xy) -> int:
        """BFS over positions (key status does not affect movement). -1 if unreachable."""
        start, goal = tuple(start_xy), tuple(goal_xy)
        if start == goal:
            return 0
        seen = {start}
        q = deque([(start, 0)])
        while q:
            (x, y), d = q.popleft()
            for dx, dy in ACTIONS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and not self.wall[nx, ny] and (nx, ny) not in seen:
                    if (nx, ny) == goal:
                        return d + 1
                    seen.add((nx, ny))
                    q.append(((nx, ny), d + 1))
        return -1
