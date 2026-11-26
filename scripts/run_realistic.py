"""v8 realistic deployment benchmark (see papers/latent-hindsight-planning/v8_preregistration.md).
  python scripts/run_realistic.py --sealed 100     # pairs sealed seed 100 with the seed-0 models (101->1, 102->2)
Requires results/hier_gate.json (task) and results/latency/ (measured compute delay per planner)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_hier import HGATE, T_EP, cfg_for  # noqa: E402
from wmi import hier  # noqa: E402
from wmi.harness import load_results, run_record, save_result, set_seed  # noqa: E402

E = 50
PLANNERS = ["hier:forward", "hier:residual", "hier:online_sgd", "hier:hindsight_posterior",
            "hier:hindsight_latent_pf_adapt", "hier:oracle_latent"]
ABL = ["hier:forward", "hier:hindsight_posterior", "hier:oracle_latent"]
REALISM = {"mass": (0.8, 1.25), "damp": (0.7, 1.4), "obs_sd": (0.01, 0.05), "act_delay": 1}


def compute_steps():
    L = [r for r in load_results("latency") if r["provenance"]["config"]["threads"] == 1]
    assert len(L) == 1, "need exactly one single-thread latency measurement"
    return {k: v["compute_steps"] for k, v in L[0]["metrics"]["planners"].items()}, L[0]["provenance"]["git_commit"]


def world(rng, factors, kind, cs):
    w = {"seed": int(rng.integers(1 << 31))}
    m, d = rng.uniform(*REALISM["mass"], E), rng.uniform(*REALISM["damp"], E)   # drawn always: same worlds for all
    if "mismatch" in factors:
        w["mass"], w["damp"] = m, d
    if "noise" in factors:
        w["obs_sd"] = REALISM["obs_sd"]
    if "delay" in factors:
        w["act_delay"] = REALISM["act_delay"]; w["compute_steps"] = cs[kind]
    return w


def main(sealed):
    assert sealed in (100, 101, 102)
    mseed = sealed - 100; hier.TASK["gap"] = json.loads(HGATE.read_text())["gap"]
    cs, lat_commit = compute_steps()
    rec = run_record({"experiment": "realistic_v8", "seed": sealed, "model_seed": mseed, "E": E, "T": T_EP,
                      "gap": hier.TASK["gap"], "realism": REALISM, "compute_steps": cs, "latency_commit": lat_commit})
    set_seed(sealed); torch.set_num_threads(4); t0 = time.time()
    cfg, Mu, Mc = cfg_for(10, mseed); res = {}
    settings = [(f"deploy/{reg}", reg, ("mismatch", "noise", "delay"), PLANNERS) for reg in ("both", "none", "fresh")]
    settings += [(f"only_{f}/both", "both", (f,), ABL) for f in ("mismatch", "noise", "delay")]
    for key, reg, factors, planners in settings:
        sch = hier.schedule(reg, np.random.default_rng(100_000 + sealed), E, T_EP); res[key] = {}
        for p in planners:
            w = world(np.random.default_rng(300_000 + sealed), factors, p.split(":")[1], cs)
            res[key][p] = hier.run_hier(p, Mu, Mc, sch, cfg, 200_000 + sealed, world=w)
        print(key, {p: float(np.mean(v["success"])) for p, v in res[key].items()}, f"{time.time() - t0:.0f}s", flush=True)
    print("wrote", save_result("realistic_v8", rec, {"regimes": res}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sealed", type=int, required=True); main(ap.parse_args().sealed)
