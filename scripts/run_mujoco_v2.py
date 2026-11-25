"""MuJoCo v2 (see papers/latent-hindsight-planning/mujoco_v2_preregistration.md).

  python scripts/run_mujoco_v2.py calibrate          # step 1: forward vs oracle only, seed 99
  python scripts/run_mujoco_v2.py main --seed 0      # step 3: uses the setting chosen in calibration
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmi.harness import run_record, save_result, set_seed  # noqa: E402
from wmi.mujoco_exp import DynMLP, collect, fit, make_schedule, phi_sd, run_regime  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [("C1", 1, 1.0), ("C2", 5, 1.0), ("C3", 5, 2.0), ("C4", 10, 2.0)]
BASE = {"pop": 300, "elites": 30, "iters": 3, "sgd_lr": 1e-3, "forget": 0.98, "window": 30, "latent_steps": 10,
        "E": 20, "T": 160, "train_batches": 20, "train_E": 200, "train_T": 50, "fit_steps": 6000}
PLANNERS = ["forward", "residual", "online_sgd", "hindsight_posterior", "hindsight_posterior_cd", "hybrid",
            "hindsight_latent", "hindsight_latent_cd", "oracle_latent"]
REGIMES = ["none", "gain", "wind", "both", "switching", "fresh", "ood_gain"]
CAL_FILE = ROOT / "results" / "mujoco_v2_calibration.json"


def build(cfg, rng):
    V, A, PHI, D, EP = collect(rng, cfg["train_batches"], cfg["train_E"], cfg["train_T"], wind_sd=cfg["wind_sd"])
    Mu, iu = fit(DynMLP(4), np.concatenate([V, A], 1), D, EP, rng, cfg["fit_steps"])
    Mc, ic = fit(DynMLP(7), np.concatenate([V, A, PHI], 1), D, EP, rng, cfg["fit_steps"])
    for p in list(Mu.parameters()) + list(Mc.parameters()):
        p.requires_grad_(False)
    cfg["resid_sd"], cfg["resid_sd_c"], cfg["phi_sd"] = iu["val_resid_sd"], ic["val_resid_sd"], phi_sd(cfg["wind_sd"])
    return Mu, Mc, iu, ic


def calibrate():
    rec = run_record({"experiment": "mujoco_v2_calibration", "seed": 99, **BASE})
    out, chosen = [], None
    for name, K, wsd in CANDIDATES:
        rng = set_seed(99)
        cfg = dict(BASE, K=K, wind_sd=wsd, H=max(10, K + 5), E=10)
        Mu, Mc, _, _ = build(cfg, rng)
        goals, sched = make_schedule("both", np.random.default_rng(9_999), cfg["E"], cfg["T"], wind_sd=wsd)
        cf, _ = run_regime("forward", Mu, Mc, goals, sched, cfg, seed=99)
        co, _ = run_regime("oracle_latent", Mu, Mc, goals, sched, cfg, seed=99)
        head = float(1 - co.mean() / cf.mean())
        out.append({"name": name, "K": K, "wind_sd": wsd, "forward": float(cf.mean()), "oracle": float(co.mean()), "headroom": head})
        print(out[-1], flush=True)
        if chosen is None and head >= 0.20:
            chosen = out[-1]
            break
    if chosen is None:
        chosen = out[-1]
    CAL_FILE.write_text(json.dumps({"provenance": rec, "candidates": out, "chosen": chosen}, indent=2, default=float))
    print("CHOSEN", chosen)


def main(seed):
    cal = json.loads(CAL_FILE.read_text())["chosen"]
    cfg = dict(BASE, K=cal["K"], wind_sd=cal["wind_sd"], H=max(10, cal["K"] + 5))
    rec = run_record({"experiment": "mujoco_v2", "seed": seed, **cfg})
    rng = set_seed(seed); torch.set_num_threads(4)
    t0 = time.time()
    Mu, Mc, iu, ic = build(cfg, rng)
    test_rng = np.random.default_rng(10_000 + seed)
    res = {}
    for reg in REGIMES:
        goals, sched = make_schedule(reg, test_rng, cfg["E"], cfg["T"], wind_sd=cfg["wind_sd"])
        res[reg] = {}
        for k in PLANNERS:
            c, ss = run_regime(k, Mu, Mc, goals, sched, cfg, seed=20_000 + seed)
            res[reg][k] = {"per_episode_cost": c.tolist(), "per_episode_ss": ss.tolist(), "mean_cost": float(c.mean())}
        print(reg, {k: round(v["mean_cost"], 4) for k, v in res[reg].items()}, f"{time.time() - t0:.0f}s", flush=True)
    print("wrote", save_result("mujoco_v2", rec, {"regimes": res, "models": {"Mu": iu, "Mc": ic}}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calibrate", "main"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    calibrate() if a.mode == "calibrate" else main(a.seed)
