"""Test: are the chunk-boundary R^2 dips caused by gain/wind confounding? Regime 'wind' (true g = 1).
Variant A: standard particle posterior over (g, fx, fy). Variant B: same, but gain particles pinned to the true g = 1.
Metric: dip = mean R^2(wind) at boundary steps t in {20,...,140} minus mean over the 5 steps before each boundary."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_mujoco_v3 import BASE, build
from wmi.harness import run_record, save_result, set_seed
from wmi.mujoco_exp import PuckBatch
from wmi.mujoco_v3 import Planner3, make_schedule3
cfg = dict(BASE, K=20, H=25, beta=0.1, E=40)
rec = run_record({"experiment": "diag_confound", "seed": 0, **cfg}); res = {}
rng = set_seed(0); torch.set_num_threads(4); Mu, Mc, _, _ = build(cfg, rng)
for pin in (False, True):
    dips = []
    for sd in (3, 4, 5):
        goals, sched = make_schedule3("wind", np.random.default_rng(30_000 + sd), cfg["E"], cfg["T"])
        env = PuckBatch(cfg["E"]); starts = np.random.default_rng(31_000 + sd).uniform(-1, 1, (5, cfg["E"], 2))
        s = env.reset(starts[0]); pl = Planner3("hindsight_latent_pf", Mu, Mc, cfg["E"], cfg, np.random.default_rng(sd))
        if pin: pl.parts[..., 0] = 1.0
        est = []
        for t in range(cfg["T"]):
            env.set_phi(sched[t])
            if t > 0 and t % 40 == 0: s = env.reset(starts[t // 40]); pl.mu[:] = 0
            if t % 20 == 0: plan = pl.act(s, goals[t])
            a = np.clip(plan[:, t % 20], -1, 1); s2 = env.step(a); pl.observe(s, a, s2); s = s2
            if pin: pl.parts[..., 0] = 1.0            # rejuvenation must not reintroduce gain uncertainty
            est.append(pl.phi_est().copy())
        est = np.stack(est)
        r2 = np.mean([[1 - ((est[t, :, j] - sched[t, :, j]) ** 2).mean() / sched[t, :, j].var() for t in range(cfg["T"])] for j in (1, 2)], 0)
        bnd = [20, 40, 60, 80, 100, 120, 140]
        dips.append(np.mean([r2[b] - r2[b - 5:b].mean() for b in bnd]))
        print(("PINNED gain" if pin else "standard   "), "seed", sd, "dip", round(float(dips[-1]), 3), "R2@159", round(float(r2[-1]), 3), flush=True)
        res.setdefault("final_r2_" + ("pinned" if pin else "standard"), []).append(float(r2[-1]))
    print(("PINNED gain" if pin else "standard   "), "mean dip", round(float(np.mean(dips)), 3))
    res["pinned" if pin else "standard"] = {"dips": [float(d) for d in dips], "mean_dip": float(np.mean(dips)),
                                            "final_r2_last_seed": float(r2[-1])}
print("wrote", save_result("diag_confound", rec, res))
