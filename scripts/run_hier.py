"""E-H hierarchical vs hindsight (v7 pre-registration).
  python scripts/run_hier.py gate            # forward vs oracle only; picks the gap width from the pre-registered ladder
  python scripts/run_hier.py main --seed 0   # all planners x regimes, K=10
  python scripts/run_hier.py ksweep --seed 0 # commitment sweep, regime both
Models are the cached v3 ensembles (identical to those of the latency experiment)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_mujoco_v3 import BASE, GATE_FILE, build  # noqa: E402
from wmi import hier  # noqa: E402
from wmi.harness import run_record, save_result, set_seed  # noqa: E402

HGATE = ROOT / "results" / "hier_gate.json"
PLANNERS = ["flat:forward", "flat:hindsight_posterior", "hier:forward", "hier:residual", "hier:online_sgd",
            "hier:hindsight_posterior", "hier:hindsight_latent_pf_adapt", "hier:oracle_latent"]
REGIMES = ["none", "wind", "both", "fresh"]
LADDER = [0.5, 0.7, 0.4]          # gap widths (m), tried in this order
T_EP, E_MAIN = 300, 50


def cfg_for(K, seed):
    beta = json.loads(GATE_FILE.read_text())["beta"]
    cfg = dict(BASE, beta=beta, K=K, H=max(10, K + 5))
    Mu, Mc, _, _ = build(cfg, np.random.default_rng(seed), cache_seed=seed)
    return cfg, Mu, Mc


def gate():
    rec = run_record({"experiment": "hier_gate", "seed": 0, "ladder": LADDER, "T": T_EP})
    set_seed(0); torch.set_num_threads(4)
    cfg, Mu, Mc = cfg_for(10, 0)
    rows, chosen = [], None
    for gap in LADDER:
        hier.TASK["gap"] = gap; row = {"gap": gap}
        for reg in ("none", "both"):
            sch = hier.schedule(reg, np.random.default_rng(99_999), 20, T_EP)
            for p in ("hier:forward", "hier:oracle_latent"):
                row[f"{reg}/{p}"] = float(np.mean(hier.run_hier(p, Mu, Mc, sch, cfg, 99)["success"]))
        row["pass"] = row["none/hier:oracle_latent"] >= 0.8 and \
            row["both/hier:oracle_latent"] - row["both/hier:forward"] >= 0.10
        rows.append(row); print(row, flush=True)
        if row["pass"]:
            chosen = gap; break
    HGATE.write_text(json.dumps({"provenance": rec, "ladder": rows, "gap": chosen}, indent=2)); print("GAP", chosen)


def run(mode, seed):
    gap = json.loads(HGATE.read_text())["gap"]
    assert gap is not None, "gate failed: no valid task"
    hier.TASK["gap"] = gap
    if mode == "main":
        settings = [(10, REGIMES, PLANNERS)]
    else:
        settings = [(K, ["both"], ["hier:forward", "hier:hindsight_posterior", "hier:oracle_latent"]) for K in (1, 5, 20)]
    rec = run_record({"experiment": f"hier_{mode}", "seed": seed, "gap": gap, "T": T_EP, "E": E_MAIN})
    set_seed(seed); torch.set_num_threads(4); t0 = time.time(); res = {}
    for K, regimes, planners in settings:
        cfg, Mu, Mc = cfg_for(K, seed)
        for reg in regimes:
            sch = hier.schedule(reg, np.random.default_rng(10_000 + seed), E_MAIN, T_EP)
            key = f"K{K}/{reg}"; res[key] = {}
            for p in planners:
                res[key][p] = hier.run_hier(p, Mu, Mc, sch, cfg, 20_000 + seed)
            print(key, {p: float(np.mean(v["success"])) for p, v in res[key].items()}, f"{time.time() - t0:.0f}s", flush=True)
    print("wrote", save_result(f"hier_{mode}", rec, {"regimes": res}))


def smoke():
    """Crash test only: 2 episodes x 30 steps; prints no outcomes."""
    hier.TASK["gap"] = LADDER[0]; set_seed(0)
    cfg, Mu, Mc = cfg_for(10, 0)
    sch = hier.schedule("both", np.random.default_rng(1), 2, 30)
    for p in PLANNERS:
        out = hier.run_hier(p, Mu, Mc, sch, cfg, 1)
        assert len(out["success"]) == 2 and np.array(out["traj"]).shape == (31, 2, 2)
    print("smoke ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["gate", "main", "ksweep", "smoke"])
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    {"gate": gate, "smoke": smoke}.get(a.mode, lambda: run(a.mode, a.seed))()
