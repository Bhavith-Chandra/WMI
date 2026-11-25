"""Identification: does the hindsight particle posterior recover the TRUE disturbance? R^2(phi_hat, phi) vs number
of observed transitions, per component (gain, wind_x, wind_y), on persistent 'both' episodes. Planner acts with
latent_pf (closed loop), K=20, per-segment resets, as in v3 main. Usage: python scripts/run_identification.py --seed 0"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_mujoco_v3 import BASE, build  # noqa: E402
from wmi.harness import run_record, save_result, set_seed  # noqa: E402
from wmi.mujoco_exp import PuckBatch  # noqa: E402
from wmi.mujoco_v3 import Planner3, make_schedule3  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
cfg = dict(BASE, K=20, H=25, beta=0.1, E=40)
rec = run_record({"experiment": "identification", "seed": a.seed, **cfg})
rng = set_seed(a.seed); torch.set_num_threads(4)
Mu, Mc, _, _ = build(cfg, rng)
out = {}
for reg in ("both", "gain", "wind"):
    goals, sched = make_schedule3(reg, np.random.default_rng(30_000 + a.seed), cfg["E"], cfg["T"])
    env = PuckBatch(cfg["E"]); starts = np.random.default_rng(31_000 + a.seed).uniform(-1, 1, (5, cfg["E"], 2))
    s = env.reset(starts[0]); pl = Planner3("hindsight_latent_pf", Mu, Mc, cfg["E"], cfg, np.random.default_rng(a.seed))
    est = []
    for t in range(cfg["T"]):
        env.set_phi(sched[t])
        if t > 0 and t % 40 == 0:
            s = env.reset(starts[t // 40]); pl.mu[:] = 0
        if t % 20 == 0:
            plan = pl.act(s, goals[t])
        act = np.clip(plan[:, t % 20], -1, 1); s2 = env.step(act); pl.observe(s, act, s2); s = s2
        est.append(pl.phi_est().copy())
    est = np.stack(est); true = sched
    r2 = {}
    for j, name in enumerate(("gain", "wind_x", "wind_y")):
        v = true[:, :, j].var()
        if v < 1e-9:
            continue
        r2[name] = [float(1 - ((est[t, :, j] - true[t, :, j]) ** 2).mean() / true[t, :, j].var()) for t in range(cfg["T"])]
    out[reg] = {"r2_by_t": r2, "t": list(range(1, cfg["T"] + 1))}
    print(reg, {k: [round(v[i], 3) for i in (0, 4, 9, 19, 39, 159)] for k, v in r2.items()}, flush=True)
print("wrote", save_result("identification", rec, out))
