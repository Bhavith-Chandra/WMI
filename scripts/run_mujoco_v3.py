"""MuJoCo v3 (pre-registered). Modes: gate | main --seed S | ksweep --seed S | rho --seed S"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmi.harness import run_record, save_result, set_seed  # noqa: E402
from wmi.mujoco_exp import collect  # noqa: E402
from wmi.mujoco_v3 import fit_ensemble, make_schedule3, run_regime3, run_regime_delay  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GATE_FILE = ROOT / "results" / "mujoco_v3_gate.json"
BASE = {"pop": 300, "elites": 30, "iters": 3, "sgd_lr": 1e-3, "forget": 0.98, "window": 30, "latent_steps": 10,
        "n_particles": 256, "E": 20, "T": 160, "train_batches": 20, "train_E": 200, "train_T": 50, "fit_steps": 6000,
        "wind_sd": 1.0, "reset_each_segment": True}
ALL = ["forward", "residual", "online_sgd", "hindsight_posterior", "hybrid", "hindsight_latent", "hindsight_latent_pf", "oracle_latent"]


def build(cfg, rng, cache_seed=None):
    """cache_seed: if given, trained ensembles are cached under data_cache/models keyed by (BASE, seed) so that
    splitting one seed's run into several processes reuses the *identical* models instead of retraining."""
    if cache_seed is not None:
        import hashlib, json as _j
        key = hashlib.sha1(_j.dumps({**BASE, "seed": cache_seed}, sort_keys=True).encode()).hexdigest()[:12]
        f = Path(__file__).resolve().parents[1] / "data_cache" / "models" / f"v3_{key}.pt"
        if f.exists():
            Mu, Mc, iu, ic = torch.load(f, weights_only=False)
            cfg["resid_sd"], cfg["resid_sd_c"] = iu["val_resid_sd"], ic["val_resid_sd"]
            print("loaded cached models", f.name, flush=True)
            return Mu, Mc, iu, ic
        out = build(cfg, rng)
        f.parent.mkdir(parents=True, exist_ok=True); torch.save(out, f)
        return out
    V, A, PHI, D, EP = collect(rng, cfg["train_batches"], cfg["train_E"], cfg["train_T"], wind_sd=cfg["wind_sd"])
    Mu, iu = fit_ensemble(4, np.concatenate([V, A], 1), D, EP, rng, cfg["fit_steps"])
    Mc, ic = fit_ensemble(7, np.concatenate([V, A, PHI], 1), D, EP, rng, cfg["fit_steps"])
    cfg["resid_sd"], cfg["resid_sd_c"] = iu["val_resid_sd"], ic["val_resid_sd"]
    return Mu, Mc, iu, ic


def gate():
    rec = run_record({"experiment": "mujoco_v3_gate", "seed": 99, **BASE})
    rng = set_seed(99)
    cfg = dict(BASE, K=20, H=25, E=10)
    Mu, Mc, _, _ = build(cfg, rng)
    out, chosen = [], None
    for beta in (0.1, 1.0, 10.0):
        cfg["beta"] = beta; row = {"beta": beta}
        for reg in ("none", "gain", "wind", "both"):
            g, s = make_schedule3(reg, np.random.default_rng(9_999), cfg["E"], cfg["T"])
            f = run_regime3("forward", Mu, Mc, g, s, cfg, 99).mean(); o = run_regime3("oracle_latent", Mu, Mc, g, s, cfg, 99).mean()
            row[reg] = float(o / f)
        row["pass"] = all(row[r] <= 1.0 for r in ("none", "gain", "wind", "both"))
        out.append(row); print(row, flush=True)
        if row["pass"]:
            chosen = beta; break
    GATE_FILE.write_text(json.dumps({"provenance": rec, "ladder": out, "beta": chosen}, indent=2))
    print("BETA", chosen)


def run(mode, seed, delays=(0, 5, 10, 20)):
    beta = json.loads(GATE_FILE.read_text())["beta"]
    gate_pass = beta is not None
    beta = 10.0 if beta is None else beta
    if mode == "main":
        settings = [("K20", 20, ["none", "gain", "wind", "both", "switching", "fresh", "ood_gain"] if gate_pass else ["wind", "both"], ALL)]
    elif mode == "ksweep":
        settings = [(f"K{K}", K, ["wind", "both"], ["forward", "online_sgd", "hindsight_posterior", "hindsight_latent_pf", "oracle_latent"]) for K in (1, 5, 10, 20, 40)]
    elif mode == "delay":
        settings = [(f"delay{d}", 5, ["both"], ["forward", "online_sgd", "hindsight_posterior", "hindsight_latent_pf_adapt",
                                                "oracle_latent"]) for d in delays]
    elif mode == "noise2":
        settings = [(f"gust{g}", 20, ["wind"], ["forward", "residual", "hindsight_posterior", "hindsight_latent_pf",
                                                 "hindsight_latent_pf_adapt", "oracle_latent"]) for g in (0.0, 0.25, 0.5, 1.0)]
    elif mode == "noise":
        settings = [(f"gust{g}", 20, ["wind"], ["forward", "residual", "hindsight_posterior", "hindsight_latent_pf",
                                                 "hindsight_latent_pf_bayes", "oracle_latent"]) for g in (0.0, 0.25, 0.5, 1.0)]
    else:
        settings = [("K20", 20, [f"ar{r}" for r in (0.0, 0.9, 0.99, 1.0)], ["forward", "residual", "hindsight_posterior", "hindsight_latent_pf", "oracle_latent"])]
    rec = run_record({"experiment": f"mujoco_v3_{mode}", "seed": seed, "beta": beta, "gate_pass": gate_pass, **BASE,
                      **({"delays": list(delays)} if mode == "delay" else {})})
    rng = set_seed(seed); torch.set_num_threads(4); t0 = time.time()
    cfg = dict(BASE, beta=beta, K=20, H=25)
    Mu, Mc, iu, ic = build(cfg, rng, cache_seed=seed if mode == "delay" else None)
    if not settings:                                   # --delays "" : train + cache models only, write no result
        print("models cached", f"{time.time() - t0:.0f}s"); return
    res = {}
    for tag, K, regimes, planners in settings:
        cfg.update(K=K, H=max(10, K + 5), gust_sd=float(tag[4:]) if tag.startswith("gust") else 0.0)
        test_rng = np.random.default_rng(10_000 + seed)
        for reg in regimes:
            g, s = make_schedule3(reg, test_rng, cfg["E"], cfg["T"])
            key = f"{tag}/{reg}"
            if tag.startswith("delay"):
                res[key] = {k: {"per_episode_cost": run_regime_delay(k, Mu, Mc, g, s, cfg, 20_000 + seed, int(tag[5:])).tolist()} for k in planners}
            else:
                res[key] = {k: {"per_episode_cost": run_regime3(k, Mu, Mc, g, s, cfg, 20_000 + seed).tolist()} for k in planners}
            print(key, {k: round(float(np.mean(v["per_episode_cost"])), 4) for k, v in res[key].items()}, f"{time.time() - t0:.0f}s", flush=True)
    print("wrote", save_result(f"mujoco_v3_{mode}", rec, {"regimes": res, "models": {"Mu": iu, "Mc": ic}}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["gate", "main", "ksweep", "rho", "noise", "noise2", "delay"]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delays", default="0,5,10,20")
    a = ap.parse_args()
    gate() if a.mode == "gate" else run(a.mode, a.seed, tuple(int(x) for x in a.delays.split(",") if x))
