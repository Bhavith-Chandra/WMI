"""Diagnose the chunk-boundary identification artefact (see mujoco_v4_preregistration.md, part A)."""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_mujoco_v3 import BASE, build
from wmi.harness import set_seed
from wmi.mujoco_exp import PuckBatch
from wmi.mujoco_v3 import Planner3, make_schedule3
cfg = dict(BASE, K=20, H=25, beta=0.1, E=40)
rng = set_seed(0); torch.set_num_threads(4)
Mu, Mc, _, _ = build(cfg, rng)
goals, sched = make_schedule3("both", np.random.default_rng(30_000), cfg["E"], cfg["T"])
env = PuckBatch(cfg["E"]); starts = np.random.default_rng(31_000).uniform(-1, 1, (5, cfg["E"], 2))
s = env.reset(starts[0]); pl = Planner3("hindsight_latent_pf", Mu, Mc, cfg["E"], cfg, np.random.default_rng(0))
sd = np.array(cfg["resid_sd_c"]); Z, DA, ENS = [], [], []; a_prev = np.zeros((cfg["E"], 2))
for t in range(cfg["T"]):
    env.set_phi(sched[t])
    if t > 0 and t % 40 == 0:
        s = env.reset(starts[t // 40]); pl.mu[:] = 0
    if t % 20 == 0:
        plan = pl.act(s, goals[t])
    a = np.clip(plan[:, t % 20], -1, 1); s2 = env.step(a)
    x = torch.tensor(np.c_[s[:, 2:], a, sched[t]], dtype=torch.float32)
    with torch.no_grad():
        preds = torch.stack([m(x) for m in Mc]).numpy()
    Z.append(((((s2 - s) - preds.mean(0)) / sd) ** 2).mean(1)); ENS.append(((preds.std(0) / sd) ** 2).mean(1))
    DA.append(np.abs(a - a_prev).sum(1)); a_prev = a
    pl.observe(s, a, s2); s = s2
Z, DA, ENS = np.array(Z), np.array(DA), np.array(ENS); t = np.arange(cfg["T"])
b = (t % 20 == 0) & (t > 0); m = ~b & (t > 0)
print(f"normalised error z (true phi): boundary {Z[b].mean():.1f}  mid-chunk {Z[m].mean():.1f}")
print(f"|a_t - a_(t-1)|:              boundary {DA[b].mean():.3f}  mid-chunk {DA[m].mean():.3f}")
print(f"ensemble var / sd^2:          boundary {ENS[b].mean():.2f}  mid-chunk {ENS[m].mean():.2f}")
print("reset steps (t%40==0) z:", Z[(t % 40 == 0) & (t > 0)].mean().round(1), " non-reset boundaries z:", Z[(t % 40 == 20)].mean().round(1))
