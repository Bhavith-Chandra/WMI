"""River crossing experiment. Modes: calibrate | main --seed S"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmi.harness import run_record, save_result, set_seed  # noqa: E402
from wmi.river import fit_linear, run, schedule  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "results" / "river_calibration.json"
PLANNERS = ["flat_forward", "flat_hindsight", "hier_forward", "hier_cautious", "hier_residual", "hier_hindsight", "hier_oracle"]
REGIMES = ["calm", "strong", "mixed", "fresh"]


def calibrate():
    rec = run_record({"experiment": "river_calibration", "seed": 99})
    rng = set_seed(99); lin = fit_linear(rng); rows, chosen = [], None
    for wf in (2.0, 3.0, 1.5):
        row = {"waterfall": wf}
        for reg in REGIMES:
            cs = schedule(reg, np.random.default_rng(999), 100)
            row[reg] = {k: float(run(k, cs, lin, wf)[0].mean()) for k in ("hier_oracle", "hier_forward")}
        row["pass"] = all(row[r]["hier_oracle"] >= 0.9 for r in REGIMES) and row["strong"]["hier_forward"] <= 0.5
        rows.append(row); print(row, flush=True)
        if row["pass"]:
            chosen = wf; break
    CAL.write_text(json.dumps({"provenance": rec, "lin": {"kappa": lin["kappa"], "sigma": lin["sigma"]}, "rows": rows,
                               "waterfall": chosen if chosen is not None else 2.0, "passed": chosen is not None}, indent=2))
    print("CHOSEN", chosen)


def main(seed):
    cal = json.loads(CAL.read_text()); wf = cal["waterfall"]
    rec = run_record({"experiment": "river", "seed": seed, "waterfall": wf, "calibration_passed": cal["passed"]})
    rng = set_seed(seed); lin = fit_linear(rng)
    out = {}
    for reg in REGIMES:
        cs = schedule(reg, np.random.default_rng(10_000 + seed), 200)
        out[reg] = {}
        for k in PLANNERS:
            succ, tg, fell = run(k, cs, lin, wf)
            out[reg][k] = {"success": succ.astype(int).tolist(), "time": tg.tolist(), "fell": fell.astype(int).tolist()}
        print(reg, {k: (round(float(np.mean(v["success"])), 3), round(float(np.mean(np.array(v["time"])[np.array(v["success"]) == 1])) if any(v["success"]) else -1, 1)) for k, v in out[reg].items()}, flush=True)
    print("wrote", save_result("river", rec, {"regimes": out, "lin": {"kappa": lin["kappa"], "sigma": lin["sigma"]}}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["calibrate", "main"]); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); calibrate() if a.mode == "calibrate" else main(a.seed)
