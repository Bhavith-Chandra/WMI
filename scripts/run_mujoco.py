"""MuJoCo interactive hindsight-planning experiment (pre-registered). Usage: python scripts/run_mujoco.py --seed 0"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmi.harness import run_record, save_result, set_seed  # noqa: E402
from wmi.mujoco_exp import DynMLP, collect, fit, make_schedule, run_regime  # noqa: E402

PLANNERS = ["forward", "residual", "online_sgd", "online_sgd_lr1e-2", "hindsight_posterior", "hybrid",
            "hindsight_latent", "oracle_latent"]
REGIMES = ["none", "gain", "wind", "both", "switching", "fresh", "ood_gain"]
CFG = {"H": 10, "pop": 300, "elites": 30, "iters": 3, "sgd_lr": 1e-3, "forget": 0.98, "window": 30,
       "latent_steps": 10, "E": 20, "T": 160, "train_batches": 20, "train_E": 200, "train_T": 50, "fit_steps": 6000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp", default="mujoco_hindsight")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = dict(CFG)
    if args.quick:
        cfg.update(E=4, T=80, train_batches=4, fit_steps=1000, pop=100)
    rec = run_record({"experiment": args.exp, "seed": args.seed, **cfg})
    rng = set_seed(args.seed)
    torch.set_num_threads(4)
    t0 = time.time()
    V, A, PHI, D, EP = collect(rng, cfg["train_batches"], cfg["train_E"], cfg["train_T"])
    Mu, info_u = fit(DynMLP(4), np.concatenate([V, A], 1), D, EP, rng, cfg["fit_steps"])
    Mc, info_c = fit(DynMLP(7), np.concatenate([V, A, PHI], 1), D, EP, rng, cfg["fit_steps"])
    for p in list(Mu.parameters()) + list(Mc.parameters()):
        p.requires_grad_(False)
    cfg["resid_sd"] = info_u["val_resid_sd"]
    print("models", info_u["val_mse_norm"], info_c["val_mse_norm"], f"{time.time() - t0:.0f}s", flush=True)
    test_rng = np.random.default_rng(10_000 + args.seed)      # test schedules: independent stream
    out = {}
    for reg in REGIMES:
        goals, sched = make_schedule(reg, test_rng, cfg["E"], cfg["T"])
        out[reg] = {}
        for k in PLANNERS:
            c, ss = run_regime(k, Mu, Mc, goals, sched, cfg, seed=20_000 + args.seed)
            out[reg][k] = {"per_episode_cost": c.tolist(), "per_episode_ss": ss.tolist(), "mean_cost": float(c.mean())}
        print(reg, {k: round(v["mean_cost"], 4) for k, v in out[reg].items()}, f"{time.time() - t0:.0f}s", flush=True)
    path = save_result(args.exp, rec, {"regimes": out, "models": {"Mu": info_u, "Mc": info_c}})
    print("wrote", path)


if __name__ == "__main__":
    main()
