"""Gym-style execution environment on top of ONE Almagest checkpoint (use case 4: RL environment).

    env = AlmagestExecEnv(ckpt, market='model'|'truth', target=+200, horizon_events=20000, k=50)
    obs = env.reset(seed)
    obs, reward, done, info = env.step(action)     # action: integer child size in lots (0 = wait), capped by remaining

The agent must buy (target>0) or sell (target<0) `|target|` lots within `horizon_events`. Every step the agent may send
one aggressive child; then the market advances `k` events (model-generated, or the ground-truth World in 'truth' mode).
The market sees the agent's orders (own-order channel), so the agent faces the market's learned reaction.
Reward = - signed implementation shortfall of this step's fill vs arrival mid (ticks x lots); unfinished quantity at
the deadline is executed at once (penalising procrastination with the book's own impact).
'model' vs 'truth' lets you train in the model and test in the exact World (sim-to-truth transfer check, gate E2).
"""
import numpy as np
from almagest.app.live import LiveModel
from almagest.app.truth import TruthLive


class AlmagestExecEnv:
    def __init__(self, ckpt, market='model', target=200, horizon_events=20000, k=50, max_child=8):
        self.ckpt, self.market, self.target = ckpt, market, int(target)
        self.H, self.k, self.max_child = int(horizon_events), int(k), int(max_child)
        self.m = None

    def _obs(self):
        b = self.m.book(10)
        bids = np.array([v for _, v in b['bids']] + [0] * (10 - len(b['bids'])), float)
        asks = np.array([v for _, v in b['asks']] + [0] * (10 - len(b['asks'])), float)
        return dict(bids=bids, asks=asks, spread=b['spread'], mid_rel=b['mid'] - self.arrival,
                    remaining=self.remaining, t_frac=self.t / self.H)

    def reset(self, seed=7_000_001):
        cls = TruthLive if self.market == 'truth' else LiveModel
        self.m = cls(self.ckpt, context_seed=int(seed))
        self.arrival = float(self.m.mid())
        self.remaining = abs(self.target)
        self.side = 'buy' if self.target > 0 else 'sell'
        self.sgn = 1 if self.target > 0 else -1
        self.t = 0
        self.cost = 0.0
        return self._obs()

    def _fill(self, q):
        f = self.m.submit_market(self.side, int(q))
        got = int(f['qty'])
        self.remaining -= got
        c = self.sgn * (f['avg_px'] - self.arrival) * got
        self.cost += c
        return -c, got

    def step(self, action):
        q = int(max(0, min(int(action), self.max_child, self.remaining)))
        reward, got = (0.0, 0)
        if q > 0:
            reward, got = self._fill(q)
        self.m.step(self.k)
        self.t += self.k
        done = self.remaining <= 0 or self.t >= self.H
        if done and self.remaining > 0:           # forced completion at the deadline
            while self.remaining > 0:
                r, g = self._fill(min(self.max_child, self.remaining))
                reward += r
                if g == 0:
                    break
        info = dict(filled=got, cost_ticks_x_lots=self.cost, remaining=self.remaining,
                    shortfall_per_lot=self.cost / max(1, abs(self.target) - self.remaining))
        return self._obs(), reward, done, info


def twap_policy(env):
    """Baseline: equal children spread over the horizon."""
    n_steps = env.H // env.k
    per = abs(env.target) / max(1, n_steps)
    acc = 0.0
    def act(obs):
        nonlocal acc
        acc += per
        q = int(acc); acc -= q
        return q
    return act
