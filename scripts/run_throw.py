"""E-A throw-and-retry (v6 pre-registration). Usage: python scripts/run_throw.py --seed 0"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmi.harness import run_record, save_result, set_seed  # noqa: E402
from wmi.throw import run_episode, schedule, train_models  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--E", type=int, default=200)
ap.add_argument("--n_train", type=int, default=20000); ap.add_argument("--exp", default="throw"); a = ap.parse_args()
rec = run_record({"experiment": a.exp, "seed": a.seed, "E": a.E, "n_train": a.n_train})
rng = set_seed(a.seed); torch.set_num_threads(4); t0 = time.time()
Mu, Mc, sdu, sdc = train_models(rng, a.n_train)
print("models trained", f"{time.time() - t0:.0f}s", "val sd Mu", sdu.round(3), "Mc", sdc.round(3), flush=True)
out = {}
for reg in ("wind", "damping", "both", "fresh"):
    phis, targets = schedule(reg, np.random.default_rng(10_000 + a.seed), a.E)
    out[reg] = {}
    for k in ("forward", "residual", "hindsight", "oracle"):
        miss = run_episode(k, Mu, Mc, sdc, phis, targets, np.random.default_rng(20_000 + a.seed))
        out[reg][k] = miss.tolist()
    print(reg, {k: [round(float(np.mean(v[i])), 3) for i in (0, 1, 4, 9)] for k, v in ((k, np.array(v)) for k, v in out[reg].items())},
          f"{time.time() - t0:.0f}s", flush=True)
print("wrote", save_result(a.exp, rec, {"miss": out, "val_sd": {"Mu": sdu.tolist(), "Mc": sdc.tolist()}}))
