"""Real-time feasibility: wall-clock per decision (act) and per belief update (observe) for ONE robot (E=1),
with the production planner configuration. Run on an otherwise idle machine.
Output: results/latency/*.json with p50/p95/max, the hardware, and compute_steps = ceil(p95_act / dt)."""
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_hier import cfg_for  # noqa: E402
from wmi import hier  # noqa: E402
from wmi.harness import run_record, save_result  # noqa: E402
from wmi.mujoco_v3 import Planner3  # noqa: E402

DT = 0.05
KINDS = ["forward", "residual", "online_sgd", "hindsight_posterior", "hindsight_latent_pf_adapt", "oracle_latent"]


def cpu():
    try:
        return next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        return platform.processor()


def main(threads):
    torch.set_num_threads(threads)
    cfg, Mu, Mc = cfg_for(10, 0)
    rec = run_record({"experiment": "latency", "seed": threads, "threads": threads, "E": 1, "K": 10, "H": cfg["H"],
                      "pop": cfg["pop"], "iters": cfg["iters"], "n_particles": cfg["n_particles"]})
    rng = np.random.default_rng(0); out = {}
    for kind in KINDS:
        pl = Planner3(kind, Mu, Mc, 1, cfg, np.random.default_rng(1)); pl.phi_true = np.array([[1.0, 0.3, -0.2]])
        s = np.array([[0.2, 0.1, 0.5, -0.2]]); g = np.array([[0.65, 0.5]])
        ta, to = [], []
        for i in range(60):
            t0 = time.perf_counter(); hier.act_hazard(pl, s, g); dt = time.perf_counter() - t0
            if i >= 5:
                ta.append(dt)
            for _ in range(4):
                a = rng.uniform(-1, 1, (1, 2)); s2 = s + 0.02 * rng.standard_normal(s.shape)
                t0 = time.perf_counter(); pl.observe(s, a, s2); to.append(time.perf_counter() - t0); s = s2
        ta, to = np.array(ta) * 1e3, np.array(to[20:]) * 1e3
        out[kind] = {"act_ms": {"p50": float(np.median(ta)), "p95": float(np.quantile(ta, .95)), "max": float(ta.max())},
                     "observe_ms": {"p50": float(np.median(to)), "p95": float(np.quantile(to, .95)), "max": float(to.max())},
                     "compute_steps": int(math.ceil(np.quantile(ta, .95) / 1e3 / DT))}
        print(threads, kind, out[kind], flush=True)
    hw = {"cpu": cpu(), "nproc": __import__("os").cpu_count(), "torch": torch.__version__, "python": platform.python_version()}
    print("wrote", save_result("latency", rec, {"planners": out, "hardware": hw, "dt_s": DT}))


if __name__ == "__main__":
    for th in (1, 4):
        main(th)
