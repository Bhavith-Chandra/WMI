"""IMMUTABLE evaluation for autoresearch (the agent must NOT edit this file).

Score = geometric mean, over a fixed suite of DEV tasks, of (loss of estimator.py) / (loss of FORWARD).
Lower is better; 1.0 = no better than ignoring outcomes. Tasks:
  * 9 synthetic disturbance regimes (gain + additive faults; persistent, fresh, switching, random-walk drift
    at three rates, mixed persistent+fresh, low excitation, none) -- planning regret vs action-bounded oracle;
  * real measured benchmarks, DEV split only = last 20% of each official TRAIN series (models fit on the first
    60%, early-stopped on the next 20%). Official TEST splits are never loaded here.
Seeds are fixed and disjoint from the paper's seeds. Appends one row to results.tsv.
Usage: python autoresearch/eval.py "<one-line description of the change>"
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
sys.path.insert(0, str(ROOT))

THETA_BAR, S_THETA, S_W, SIGMA, S_DELTA, A_MAX = 1.0, 0.4, 0.5, 0.3, 1.0, 10.0
T, M, SEED = 2000, 400, 1000
REGIMES = ["persistent", "fresh", "switching", "drift_slow", "drift_mid", "drift_fast", "mixed", "low_excitation", "none"]


def load_estimator():
    spec = importlib.util.spec_from_file_location("estimator", HERE / "estimator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Estimator


def bayes_action(mean, cov, delta):
    """argmin_b E[(b(1+c1) + c0 - delta)^2], b = theta_bar * a, c ~ N(mean, cov)."""
    g = 1 + mean[:, 0]
    b = (g * (delta - mean[:, 1]) - cov[:, 0, 1]) / (g ** 2 + cov[:, 0, 0])
    return np.clip(b / THETA_BAR, -A_MAX, A_MAX)


def synth(regime, Est, rng):
    th = THETA_BAR + S_THETA * rng.standard_normal(M)
    w = S_W * rng.standard_normal(M)
    if regime == "none":
        th, w = np.full(M, THETA_BAR), np.zeros(M)
    s_delta = 0.1 if regime == "low_excitation" else S_DELTA
    q = {"drift_slow": 1e-4, "drift_mid": 1e-3, "drift_fast": 1e-2}.get(regime, 0.0)
    prior_mean, prior_cov = np.zeros(2), np.diag([(S_THETA / THETA_BAR) ** 2, S_W ** 2])
    est, fwd = Est(M, prior_mean, prior_cov, SIGMA), None
    loss_e, loss_f = 0.0, 0.0
    for t in range(T):
        if regime == "fresh" or (regime == "switching" and t > 0 and t % 400 == 0):
            th = THETA_BAR + S_THETA * rng.standard_normal(M); w = S_W * rng.standard_normal(M)
        if q:
            th = th + np.sqrt(q) * rng.standard_normal(M); w = w + np.sqrt(q) * rng.standard_normal(M)
        w_t = w + (0.3 * rng.standard_normal(M) if regime == "mixed" else 0.0)
        delta = s_delta * rng.standard_normal(M)
        eps = SIGMA * rng.standard_normal(M)
        mean, cov = est.belief()
        a_e = bayes_action(mean, cov, delta)
        a_f = bayes_action(np.broadcast_to(prior_mean, (M, 2)), np.broadcast_to(prior_cov, (M, 2, 2)), delta)
        a_o = np.clip((delta - w_t) / np.where(np.abs(th) < 1e-6, 1e-6, th), -A_MAX, A_MAX)
        e_o = (th * a_o + w_t - delta) ** 2
        loss_e += float(((th * a_e + w_t - delta) ** 2 - e_o).mean())
        loss_f += float(((th * a_f + w_t - delta) ** 2 - e_o).mean())
        y = th * a_e + w_t + eps
        x = np.stack([THETA_BAR * a_e, np.ones(M)], 1)
        est.update(x, y - THETA_BAR * a_e)
    return loss_e, loss_f


def real_dev(Est):
    import nonlinear_benchmarks as nb
    from wmi.realdata import Norm, fit_narx, windows
    cache = ROOT / "data_cache"
    out = {}
    for name in ("Cascaded_Tanks", "EMPS", "Silverbox", "WienerHammerBenchMark"):
        tr, _ = getattr(nb, name)(dir_placement=str(cache))     # TEST split discarded unread
        Tn = len(tr.y)
        a, b, c = int(.6 * Tn), int(.8 * Tn), Tn
        fit, val, dev = (tr.u[:a], tr.y[:a]), (tr.u[a:b], tr.y[a:b]), (tr.u[b:c], tr.y[b:c])
        norm = Norm(fit[0], fit[1])
        mpath = cache / f"ar_model_{name}.pt"
        if mpath.exists():
            ck = torch.load(mpath, weights_only=False)
            from wmi.realdata import NARX
            model = NARX(10); model.load_state_dict(ck["state"]); model.eval(); sig = ck["sigma"]
        else:
            torch.manual_seed(SEED)
            model, info = fit_narx([fit], [val], norm, 10, 4000, np.random.default_rng(SEED))
            sig = info["val_rmse_norm"]
            torch.save({"state": model.state_dict(), "sigma": sig}, mpath)
        X, Y, _ = windows(norm.nu(dev[0]), norm.ny(dev[1]), 10)
        with torch.no_grad():
            F = model(torch.from_numpy(X)).numpy()
        yt = X[:, 0]
        est = Est(1, np.zeros(2), np.diag([0.5 ** 2, (2 * sig) ** 2]), sig)
        se_e, se_f = 0.0, 0.0
        for t in range(len(Y)):
            df = F[t] - yt[t]
            mean, _ = est.belief()
            pred = F[t] + mean[0, 0] * df + mean[0, 1]
            se_e += (pred - Y[t]) ** 2; se_f += (F[t] - Y[t]) ** 2
            est.update(np.array([[df, 1.0]]), np.array([Y[t] - F[t]]))
        out[name] = (float(se_e), float(se_f))
    return out


def main():
    desc = sys.argv[1] if len(sys.argv) > 1 else "(no description)"
    Est = load_estimator()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    ratios, detail = [], {}
    for reg in REGIMES:
        le, lf = synth(reg, Est, rng)
        r = max(le, 1e-9) / max(lf, 1e-9)
        ratios.append(r); detail[reg] = round(r, 4)
    for name, (se, sf) in real_dev(Est).items():
        r = se / sf
        ratios.append(r); detail["real_" + name] = round(r, 4)
    score = float(np.exp(np.mean(np.log(ratios))))
    src = (HERE / "estimator.py").read_bytes()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    row = [time.strftime("%Y-%m-%dT%H:%M:%S"), commit, hashlib.sha1(src).hexdigest()[:10], f"{score:.5f}",
           f"{time.time() - t0:.0f}s", json.dumps(detail), desc]
    tsv = HERE / "results.tsv"
    if not tsv.exists():
        tsv.write_text("time\tcommit\testimator_sha\tscore\tduration\tdetail\tdescription\n")
    with tsv.open("a") as f:
        f.write("\t".join(row) + "\n")
    print("DETAIL", json.dumps(detail))
    print(f"SCORE: {score:.5f}")


if __name__ == "__main__":
    main()
