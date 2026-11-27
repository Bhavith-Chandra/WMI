"""v11 subgoal redrawing when the latent decides what is possible (ramp task; v11_preregistration.md).
  python scripts/run_ramp.py gate | calibrate --seed s | fit | main --seed s | smoke"""
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
from wmi import hier, ramp  # noqa: E402
from wmi.harness import load_results, run_record, save_result, set_seed  # noqa: E402

GATE = ROOT / "results" / "ramp_gate.json"; MODELS = ROOT / "results" / "ramp_route_models.json"
LADDER = [(3.5, 160), (3.5, 120), (3.0, 160), (4.0, 160), (3.0, 120), (4.0, 120)]
LLS = ["forward", "hindsight_posterior", "oracle_latent"]
TEST = [("forward", "prior"), ("forward", "post"), ("hindsight_posterior", "prior"), ("hindsight_posterior", "post"),
        ("hindsight_posterior", "oracle"), ("hindsight_posterior", "A"), ("hindsight_posterior", "B"), ("oracle_latent", "oracle")]
E_CAL, E_TEST = 100, 50


def setup_task():
    g = json.loads(GATE.read_text()); assert g["choice"] is not None, "gate failed"
    ramp.TASK["F_s"], ramp.TASK["T"] = g["choice"]


def headroom(sA, sB):
    return float(np.maximum(sA, sB).mean() - max(sA.mean(), sB.mean()))


def gate():
    rec = run_record({"experiment": "ramp_gate", "seed": 0, "ladder": LADDER})
    set_seed(0); torch.set_num_threads(4); cfg, Mu, Mc = cfg_for(10, 0); rows, choice = [], None
    for F_s, T in LADDER:
        ramp.TASK["F_s"], ramp.TASK["T"] = F_s, T
        sch = hier.schedule("both", np.random.default_rng(99_995), 60, T); row = {"F_s": F_s, "T": T}
        for ll in ("forward", "oracle_latent"):
            o = {r: ramp.run_ramp(ll, r, Mu, Mc, sch, cfg, 95) for r in ("A", "B")}
            sA, sB = (np.array(o[r]["score"]) for r in ("A", "B"))
            row[ll] = {"score_A": sA.mean(), "score_B": sB.mean(), "succ_A": np.mean(o["A"]["success"]),
                       "succ_B": np.mean(o["B"]["success"]), "headroom": headroom(sA, sB)}
        row["pass"] = row["oracle_latent"]["headroom"] >= 0.10
        rows.append(row); print(row, flush=True)
        if row["pass"]:
            choice = (F_s, T); break
    GATE.write_text(json.dumps({"provenance": rec, "ladder": rows, "choice": choice}, indent=2, default=float)); print("CHOICE", choice)


def calibrate(seed):
    setup_task(); rec = run_record({"experiment": "ramp_calib", "seed": seed, "task": dict(ramp.TASK), "E": E_CAL})
    set_seed(seed); torch.set_num_threads(4); cfg, Mu, Mc = cfg_for(10, seed); out = {}; t0 = time.time()
    sch = hier.schedule("both", np.random.default_rng(800_000 + seed), E_CAL, ramp.TASK["T"])
    for ll in LLS:
        out[ll] = {r: ramp.run_ramp(ll, r, Mu, Mc, sch, cfg, 810_000 + seed) for r in ("A", "B")}
        print(ll, {r: float(np.mean(v["score"])) for r, v in out[ll].items()}, f"{time.time() - t0:.0f}s", flush=True)
    print("wrote", save_result("ramp_calib", rec, {"runs": out}))


def fit():
    C = sorted(load_results("ramp_calib"), key=lambda r: r["provenance"]["config"]["seed"]); setup_task()
    models = {"provenance": run_record({"experiment": "ramp_fit", "seed": 0, "calib_seeds": [r["provenance"]["config"]["seed"] for r in C]})}
    for ll in LLS:
        g = lambda r, k: np.concatenate([np.array(c["metrics"]["runs"][ll][r][k]) for c in C])
        F, Z = g("A", "belief"), g("A", "latent")
        assert np.allclose(F, g("B", "belief")) and np.allclose(Z, g("B", "latent")), "fork state must not depend on the route"
        sc = {r: g(r, "score") for r in ("A", "B")}
        models[ll] = {"belief": ramp.RouteModel().fit(F, sc).state(), "latent": ramp.RouteModel().fit(Z, sc).state(),
                      "prior_choice": "A" if sc["A"].mean() >= sc["B"].mean() else "B",
                      "calib_score": {r: float(v.mean()) for r, v in sc.items()}}
        print(ll, models[ll]["prior_choice"], models[ll]["calib_score"])
    MODELS.write_text(json.dumps(models, indent=1, default=float))


def load_models(ll):
    d = json.loads(MODELS.read_text())[ll]
    return {"belief": ramp.RouteModel.load(d["belief"]), "latent": ramp.RouteModel.load(d["latent"]), "prior_choice": d["prior_choice"]}


def main(seed):
    setup_task(); rec = run_record({"experiment": "ramp_main", "seed": seed, "task": dict(ramp.TASK), "E": E_TEST})
    set_seed(seed); torch.set_num_threads(4); cfg, Mu, Mc = cfg_for(10, seed); res = {}; t0 = time.time()
    for reg in ("none", "wind", "both", "fresh"):
        sch = hier.schedule(reg, np.random.default_rng(900_000 + seed), E_TEST, ramp.TASK["T"]); res[reg] = {}
        for ll, hl in TEST:
            res[reg][f"{ll}|{hl}"] = ramp.run_ramp(ll, hl, Mu, Mc, sch, cfg, 910_000 + seed, models=load_models(ll))
        print(reg, {k: round(float(np.mean(v["score"])), 3) for k, v in res[reg].items()}, f"{time.time() - t0:.0f}s", flush=True)
    print("wrote", save_result("ramp_main", rec, {"regimes": res}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["gate", "calibrate", "fit", "main", "smoke"])
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    if a.mode == "smoke":
        cfg, Mu, Mc = cfg_for(10, 0); sch = hier.schedule("both", np.random.default_rng(1), 3, 40)
        for ll in LLS:
            for r in ("A", "B"):
                o = ramp.run_ramp(ll, r, Mu, Mc, sch, cfg, 1); assert len(o["score"]) == 3
        print("smoke ok")
    else:
        {"gate": gate, "fit": fit}.get(a.mode, lambda: (calibrate if a.mode == "calibrate" else main)(a.seed))()
