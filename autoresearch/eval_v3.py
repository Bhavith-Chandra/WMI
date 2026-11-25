"""IMMUTABLE evaluation v3 for autoresearch (pre-registered 2026-11-25, BEFORE any v3 candidate was run).

Changes vs v2, motivated by the sealed-test failure of AR-3 (see hypotheses.md):
  * real DEV tasks are scored on the 10-STEP planning horizon (k=10 rollouts), not 1-step error;
  * a real task's ratio is  MSE10(candidate) / min(MSE10(forward), MSE10(online SGD))  -> must beat BOTH;
  * SCORE = sqrt(geomean(synthetic ratios) * geomean(real ratios))  -> real and synthetic weigh equally;
  * WORST = max over all ratios; rule: KEEP iff WORST <= 1.05 and SCORE improves > 0.5% over the best kept v3.
Real DEV = held-out part of the official TRAIN data only (TEST never loaded). Streams capped at 10k steps.
Usage: python autoresearch/eval_v3.py "<description>"
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import eval as ev1  # noqa: E402  (reuses the v1/v2 synthetic suite unchanged)

K, STRIDE, CAP, SEED = 10, 10, 10_000, 1000


def load_module():
    spec = importlib.util.spec_from_file_location("estimator", HERE / "estimator.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def dev_sets():
    import nonlinear_benchmarks as nb
    cache = str(ROOT / "data_cache")
    sets = {}
    for name in ("Cascaded_Tanks", "Silverbox", "WienerHammerBenchMark"):
        tr, _ = getattr(nb, name)(dir_placement=cache)
        T = len(tr.y); a, b = int(.6 * T), int(.8 * T)
        sets[name] = {"fit": [(tr.u[:a], tr.y[:a])], "es": [(tr.u[a:b], tr.y[a:b])], "dev": [(tr.u[b:], tr.y[b:])]}
    tr, _ = nb.ParWH(dir_placement=cache)
    lv = np.round([np.std(r.u) for r in tr], 3)
    fit, es, dev = [], [], []
    for L in sorted(set(lv)):
        recs = [(tr[i].u, tr[i].y) for i in np.flatnonzero(lv == L)]
        fit += recs[:16]; es += recs[16:18]; dev += recs[18:19]
    sets["ParWH"] = {"fit": fit, "es": es, "dev": dev}
    return sets


def get_model(name, d):
    from wmi.realdata import NARX, Norm, fit_narx
    norm = Norm(np.concatenate([r[0] for r in d["fit"]]), np.concatenate([r[1] for r in d["fit"]]))
    p = ROOT / "data_cache" / f"ar3_model_{name}.pt"
    if p.exists():
        ck = torch.load(p, weights_only=False)
        m = NARX(10); m.load_state_dict(ck["state"]); return m.eval(), norm, ck["sigma"]
    torch.manual_seed(SEED)
    m, info = fit_narx(d["fit"], d["es"], norm, 10, 4000, np.random.default_rng(SEED))
    torch.save({"state": m.state_dict(), "sigma": info["val_rmse_norm"]}, p)
    return m, norm, info["val_rmse_norm"]


def main():
    from wmi.realdata import stream_generic
    desc = sys.argv[1] if len(sys.argv) > 1 else "(no description)"
    mod = load_module()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    synth, real, detail = [], [], {}
    for reg in ev1.REGIMES:
        le, lf = ev1.synth(reg, mod.Estimator, rng)
        r = max(le, 1e-9) / max(lf, 1e-9); synth.append(r); detail[reg] = round(r, 4)
    torch.manual_seed(SEED)
    for name, d in dev_sets().items():
        model, norm, sig = get_model(name, d)
        cfg = {"s_theta": 0.5, "s_w": 2 * sig, "sigma": sig, "sgd_lr": 1e-3}
        acc = {m: [] for m in ("forward", "sgd", "candidate")}
        acc1 = {m: [] for m in ("forward", "sgd", "candidate")}
        for u, y in d["dev"]:
            se1, sek = stream_generic(model, norm.nu(u), norm.ny(y), 10, K, mod, cfg, STRIDE, CAP)
            for m in acc:
                acc[m].append(sek[m]); acc1[m].append(se1[m])
        mse = {m: np.concatenate(v).mean() for m, v in acc.items()}
        mse1 = {m: np.concatenate(v).mean() for m, v in acc1.items()}
        r = mse["candidate"] / min(mse["forward"], mse["sgd"])
        real.append(r)
        detail[f"real10_{name}"] = round(float(r), 4)
        detail[f"info_{name}"] = {"10step_cand/fwd": round(float(mse["candidate"] / mse["forward"]), 4),
                                  "10step_sgd/fwd": round(float(mse["sgd"] / mse["forward"]), 4),
                                  "1step_cand/fwd": round(float(mse1["candidate"] / mse1["forward"]), 4),
                                  "1step_sgd/fwd": round(float(mse1["sgd"] / mse1["forward"]), 4)}
    gm = lambda x: float(np.exp(np.mean(np.log(x))))
    score = float(np.sqrt(gm(synth) * gm(real)))
    worst = float(max(synth + real))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    sha = hashlib.sha1((HERE / "estimator.py").read_bytes()).hexdigest()[:10]
    row = [time.strftime("%Y-%m-%dT%H:%M:%S"), commit, sha, f"{score:.5f}", f"{time.time() - t0:.0f}s",
           json.dumps(detail), f"v3 worst={worst:.4f} | " + desc]
    tsv = HERE / "results_v3.tsv"
    if not tsv.exists():
        tsv.write_text("time\tcommit\testimator_sha\tscore\tduration\tdetail\tdescription\n")
    with tsv.open("a") as f:
        f.write("\t".join(row) + "\n")
    print("DETAIL", json.dumps(detail))
    print(f"SCORE: {score:.5f}\nWORST: {worst:.5f}")


if __name__ == "__main__":
    main()
