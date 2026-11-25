"""Numerically verify the hindsight-planning theorems (wmi/theory.py) and write provenance-stamped results."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmi.harness import run_record, save_result, set_seed  # noqa: E402
from wmi.theory import (PLANNERS, TheoryConfig, forward_excess_closed_form,  # noqa: E402
                        residual_excess_closed_form_additive, simulate)


def slope_loglog(R, ts):
    return float(np.polyfit(np.log(ts), np.log(R[ts - 1]), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=4000)
    ap.add_argument("--M", type=int, default=2000)
    ap.add_argument("--exp", default="theory_hindsight")
    args = ap.parse_args()
    cfg = TheoryConfig(T=args.T, M=args.M, seed=args.seed)
    rec = run_record({"experiment": "theory_hindsight", **cfg.dict()})
    set_seed(args.seed)

    ts = np.unique(np.logspace(2, np.log10(cfg.T), 12).astype(int))
    metrics, curves = {}, {}
    for regime in ("persistent", "persistent_additive", "persistent_gain", "fresh", "switching", "none"):
        r = simulate(regime, cfg)
        m = {}
        for p in PLANNERS:
            e = r["excess"][p]
            R = np.cumsum(e)
            m[p] = {"cum_regret_T": float(R[-1]), "mean_excess_last_quarter": float(e[-cfg.T // 4:].mean()),
                    "loglog_slope_100_to_T": slope_loglog(R, ts),
                    "regret_over_logT": float(R[-1] / np.log(cfg.T))}
            curves[f"{regime}/{p}"] = R[ts - 1].tolist()
        # Theorem 1 check: per-step dominance of hindsight_bayes over forward (3 s.e. tolerance)
        diff = r["excess"]["hindsight_bayes"] - r["excess"]["forward"]
        se = np.sqrt(r["excess_se"]["hindsight_bayes"] ** 2 + r["excess_se"]["forward"] ** 2)
        # Theorem 1 is a statement in expectation over the prior; it applies only where phi ~ prior and
        # the planner's persistence assumption is correct, i.e. regime == "persistent".
        m["steps_hindsight_worse_than_forward_beyond_3se"] = int(np.sum(diff > 3 * se))
        m["thm1_applies"] = regime == "persistent"
        m["min_precision_eig_growth_per_step"] = float(np.polyfit(np.arange(cfg.T), r["min_precision_eig"], 1)[0])
        metrics[regime] = m
        print(regime, {p: round(m[p]["cum_regret_T"], 1) for p in PLANNERS},
              "slopes", {p: round(m[p]["loglog_slope_100_to_T"], 2) for p in PLANNERS},
              "worse_steps", m["steps_hindsight_worse_than_forward_beyond_3se"], flush=True)

    metrics["closed_form"] = {"forward_excess_per_step": forward_excess_closed_form(cfg),
                              "residual_excess_per_step_additive_only": residual_excess_closed_form_additive(cfg)}
    path = save_result(args.exp, rec, metrics, extra={"curves_at_t": {"t": ts.tolist(), **curves}})
    print("wrote", path)


if __name__ == "__main__":
    main()
