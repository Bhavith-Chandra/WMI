"""Latent planning: CEM-MPC with L2 latent cost vs hindsight-learned temporal distance.

Usage: python scripts/run_planning.py --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmi.data import collect, split_episodes  # noqa: E402
from wmi.env import N_ACTIONS, KeyGrid  # noqa: E402
from wmi.harness import run_record, save_result, set_seed  # noqa: E402
from wmi.models import LatentWorldModel, TemporalDistance  # noqa: E402
from wmi.planning import cem_first_action, evaluate, sample_tasks  # noqa: E402
from wmi.train import train_latent, train_temporal_distance  # noqa: E402


@torch.no_grad()
def open_loop_position_acc(wm, env, traj, k_max=8):
    """Decode agent position after k imagined steps; compare to ground truth (test episodes)."""
    o0 = torch.from_numpy(traj.obs_full[:, 0])
    z = wm.encode(o0)
    acc = {}
    n = env.size * env.size
    for k in range(1, k_max + 1):
        z = wm.predict(z, torch.from_numpy(traj.actions[:, k - 1]))
        pos = wm.decode(z)[:, :n].argmax(-1).numpy()
        true = traj.states[:, k, 0] * env.size + traj.states[:, k, 1]
        acc[k] = float((pos == true).mean())
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--wm_steps", type=int, default=3000)
    ap.add_argument("--dist_steps", type=int, default=3000)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=25)
    ap.add_argument("--plan_horizon", type=int, default=10)
    args = ap.parse_args()
    cfg = {"experiment": "planning", **vars(args)}

    rng = set_seed(args.seed)
    env = KeyGrid()
    traj = collect(env, args.episodes, args.horizon, rng)
    split = split_episodes(traj, rng=rng)
    S = {k: traj.subset(v) for k, v in split.items()}

    torch.manual_seed(args.seed)
    wm = LatentWorldModel(env.full_dim, N_ACTIONS, latent=args.latent)
    h_wm = train_latent(wm, S["train"], S["val"], args.wm_steps, rng)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    dist = TemporalDistance(args.latent)
    h_d = train_temporal_distance(dist, wm, S["train"], S["val"], args.dist_steps, rng)
    dist.eval()

    model_quality = {"open_loop_pos_acc_test": open_loop_position_acc(wm, env, S["test"])}

    # Evaluation tasks come from an independent RNG stream; they are fresh start/goal pairs,
    # not states drawn from the training episodes.
    eval_rng = np.random.default_rng(10_000 + args.seed)
    tasks = sample_tasks(env, eval_rng, args.tasks)
    with torch.no_grad():
        d_true = np.array([d for *_, d in tasks])
        zs = wm.encode(torch.from_numpy(env.render_full(np.stack([s for s, *_ in tasks]))))
        zg = wm.encode(torch.from_numpy(env.render_full(np.stack([[g[0], g[1], 1] for _, g, _ in tasks]))))
        d_hat = dist(zs, zg).numpy()
        l2 = ((zs - zg) ** 2).sum(-1).numpy()
    model_quality["spearman_dist_vs_shortest_path"] = float(
        np.corrcoef(np.argsort(np.argsort(d_hat)), np.argsort(np.argsort(d_true)))[0, 1])
    model_quality["spearman_l2_vs_shortest_path"] = float(
        np.corrcoef(np.argsort(np.argsort(l2)), np.argsort(np.argsort(d_true)))[0, 1])

    costs = {"l2": lambda z, g: ((z - g) ** 2).sum(-1), "hindsight": lambda z, g: dist(z, g)}
    results = {}
    pr = np.random.default_rng(20_000 + args.seed)
    results["random"] = evaluate(env, lambda g: (lambda s: int(pr.integers(N_ACTIONS))), tasks, args.max_steps)
    for name, cost in costs.items():
        prng = np.random.default_rng(30_000 + args.seed)

        def make_policy(g, cost=cost, prng=prng):
            with torch.no_grad():
                zgoal = wm.encode(torch.from_numpy(env.render_full(np.array([[g[0], g[1], 1]]))))

            def pol(s):
                with torch.no_grad():
                    z0 = wm.encode(torch.from_numpy(env.render_full(s[None])))
                return cem_first_action(wm, cost, z0, zgoal, args.plan_horizon, prng)
            return pol

        results[f"cem_{name}"] = evaluate(env, make_policy, tasks, args.max_steps)
        print(name, results[f"cem_{name}"], flush=True)
    results["oracle_mean_shortest_path"] = float(d_true.mean())
    print(model_quality, results["random"], flush=True)

    rec = run_record(cfg)
    path = save_result("planning", rec, {"model_quality": model_quality, "planning": results},
                       extra={"train_history": {"wm": h_wm, "dist": h_d},
                              "leakage": {"episode_split_sizes": {k: int(len(v)) for k, v in split.items()},
                                          "eval_tasks": "fresh start/goal pairs from independent RNG"}})
    print("wrote", path)


if __name__ == "__main__":
    main()
