"""v10 subgoal redrawing from the posterior (papers/latent-hindsight-planning/v10_preregistration.md).
  python scripts/run_fork.py gate                 # oracle low level only: picks (narrow width, time limit) from the ladder
  python scripts/run_fork.py calibrate --seed s   # forced routes A and B, calibration episodes only
  python scripts/run_fork.py fit                  # route-success models from calibration results
  python scripts/run_fork.py main --seed s        # test (sealed schedules)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_hier import cfg_for  # noqa: E402
from wmi import fork, hier  # noqa: E402
from wmi.harness import load_results, run_record, save_result, set_seed  # noqa: E402

GATE = ROOT / "results" / "fork_gate.json"; MODELS = ROOT / "results" / "fork_route_models.json"
LADDER = [(0.35, 150), (0.35, 120), (0.35, 100), (0.30, 150), (0.30, 120), (0.30, 100), (0.40, 120), (0.40, 100)]
LLS = ["forward", "hindsight_posterior", "oracle_latent"]
TEST = [("forward", "prior"), ("forward", "post"), ("hindsight_posterior", "prior"), ("hindsight_posterior", "post"),
        ("hindsight_posterior", "oracle"), ("hindsight_posterior", "A"), ("hindsight_posterior", "B"), ("oracle_latent", "oracle")]
E_CAL, E_TEST, B_PEN = 100, 50, 0.05


def setup_task():
    g = json.loads(GATE.read_text()); assert g["choice"] is not None, "gate failed"
    fork.TASK["narrow"], fork.TASK["T"] = g["choice"]


def gate():
    rec = run_record({"experiment": "fork_gate", "seed": 0, "ladder": LADDER})
    set_seed(0); torch.set_num_threads(4); cfg, Mu, Mc = cfg_for(10, 0); rows, choice = [], None
    for narrow, T in LADDER:
        fork.TASK["narrow"], fork.TASK["T"] = narrow, T
        sch = hier.schedule("both", np.random.default_rng(99_998), 60, T)
        sA = np.array(fork.run_fork("oracle_latent", "A", Mu, Mc, sch, cfg, 98)["success"])
        sB = np.array(fork.run_fork("oracle_latent", "B", Mu, Mc, sch, cfg, 98)["success"])
        row = {"narrow": narrow, "T": T, "A": sA.mean(), "B": sB.mean(), "A_only": float(np.mean(sA & ~sB)), "B_only": float(np.mean(sB & ~sA))}
        row["pass"] = row["A_only"] >= 0.15 and row["B_only"] >= 0.15
        rows.append(row); print(row, flush=True)
        if row["pass"]:
            choice = (narrow, T); break
    GATE.write_text(json.dumps({"provenance": rec, "ladder": rows, "choice": choice}, indent=2, default=float)); print("CHOICE", choice)


def calibrate(seed):
    setup_task(); rec = run_record({"experiment": "fork_calib", "seed": seed, "task": fork.TASK, "E": E_CAL})
    set_seed(seed); torch.set_num_threads(4); cfg, Mu, Mc = cfg_for(10, seed); out = {}; t0 = time.time()
    sch = hier.schedule("both", np.random.default_rng(400_000 + seed), E_CAL, fork.TASK["T"])
    for ll in LLS:
        out[ll] = {r: fork.run_fork(ll, r, Mu, Mc, sch, cfg, 410_000 + seed) for r in ("A", "B")}
        print(ll, {r: float(np.mean(v["success"])) for r, v in out[ll].items()}, f"{time.time() - t0:.0f}s", flush=True)
    print("wrote", save_result("fork_calib", rec, {"runs": out}))


def fit():
    C = sorted(load_results("fork_calib"), key=lambda r: r["provenance"]["config"]["seed"]); setup_task()
    models = {"provenance": run_record({"experiment": "fork_fit", "seed": 0, "calib_seeds": [r["provenance"]["config"]["seed"] for r in C]})}
    for ll in LLS:
        g = lambda r, k: np.concatenate([np.array(c["metrics"]["runs"][ll][r][k]) for c in C])
        F, Z = g("A", "belief"), g("A", "latent")
        assert np.allclose(F, g("B", "belief")), "belief at the fork must not depend on the route"
        succ = {r: g(r, "success").astype(float) for r in ("A", "B")}
        mA, mB = succ["A"].mean(), succ["B"].mean()
        models[ll] = {"belief": fork.RouteModel().fit(F, succ).state(), "latent": fork.RouteModel().fit(Z, succ).state(),
                      "prior_choice": "A" if mA >= mB - B_PEN else "B", "calib_success": {"A": mA, "B": mB}}
        print(ll, models[ll]["prior_choice"], models[ll]["calib_success"])
    MODELS.write_text(json.dumps(models, indent=1, default=float))


def load_models(ll):
    d = json.loads(MODELS.read_text())[ll]
    return {"belief": fork.RouteModel.load(d["belief"]), "latent": fork.RouteModel.load(d["latent"]), "prior_choice": d["prior_choice"]}


def main(seed):
    setup_task(); rec = run_record({"experiment": "fork_main", "seed": seed, "task": fork.TASK, "E": E_TEST, "b_penalty": B_PEN})
    set_seed(seed); torch.set_num_threads(4); cfg, Mu, Mc = cfg_for(10, seed); res = {}; t0 = time.time()
    for reg in ("none", "wind", "both", "fresh"):
        sch = hier.schedule(reg, np.random.default_rng(500_000 + seed), E_TEST, fork.TASK["T"]); res[reg] = {}
        for ll, hl in TEST:
            res[reg][f"{ll}|{hl}"] = fork.run_fork(ll, hl, Mu, Mc, sch, cfg, 600_000 + seed, models=load_models(ll), b_penalty=B_PEN)
        print(reg, {k: float(np.mean(v["success"])) for k, v in res[reg].items()}, f"{time.time() - t0:.0f}s", flush=True)
    print("wrote", save_result("fork_main", rec, {"regimes": res}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["gate", "calibrate", "fit", "main", "smoke"])
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    if a.mode == "smoke":
        cfg, Mu, Mc = cfg_for(10, 0); sch = hier.schedule("both", np.random.default_rng(1), 3, 40)
        for ll in LLS:
            for r in ("A", "B"):
                o = fork.run_fork(ll, r, Mu, Mc, sch, cfg, 1); assert len(o["route"]) == 3
        print("smoke ok")
    else:
        {"gate": gate, "fit": fit}.get(a.mode, lambda: (calibrate if a.mode == "calibrate" else main)(a.seed))()
