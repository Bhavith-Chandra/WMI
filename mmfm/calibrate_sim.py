"""Choose the simulator's decoding calibration (clock scale, sampling temperature) on v2-VAL only.

python -I calibrate_sim.py --run runs/v2 --events /home/user/data/mmfm_v2
Loss per config = mean over (pair, horizon) of |log event_ratio| + |coverage80 - 0.8| / 0.1 + |PIT - 0.5| / 0.1,
i.e. distance to the EVAL_v2 target centres in units of the target half-widths. Writes runs/v2/sim_calibration.json,
which is frozen together with the model (part of the forward-test hash).
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def run(cfg, a, out):
    cmd = [sys.executable, "-I", str(ROOT / "sim_eval_v2.py"), "--events", a.events, "--run", a.run, "--split", "val",
           "--pairs", a.pairs, "--starts", str(a.starts), "--paths", str(a.paths), "--dt-scale", str(cfg[0]),
           "--temp", str(cfg[1]), "--clock", "anchored", "--gens", "MMFM-sim", "--out", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)
    r = json.loads(Path(out).read_text())
    loss, cells = [], []
    for pair, gs in r["per_pair"].items():
        g = gs["MMFM-sim"]
        for h in ("1.0", "5.0", "10.0"):
            x = g[h]
            l_ = abs(np.log(max(x["event_ratio_median"], 1e-3))) + abs(x["coverage80"] - 0.8) / 0.1 + abs(x["pit_mean"] - 0.5) / 0.1
            loss.append(l_)
            cells.append(dict(pair=pair, h=h, ev=x["event_ratio_median"], cov=x["coverage80"], pit=x["pit_mean"], crps=x["crps_bp"]))
    return float(np.mean(loss)), cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/v2"))
    ap.add_argument("--events", default="/home/user/data/mmfm_v2")
    ap.add_argument("--pairs", default="ethusd,xrpusd,dogeusd")
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--paths", type=int, default=16)
    ap.add_argument("--dt", default="1.0")
    ap.add_argument("--temp", default="0.85,1.0,1.15")
    a = ap.parse_args()
    grid = list(itertools.product([float(x) for x in a.dt.split(",")], [float(x) for x in a.temp.split(",")]))
    out = Path(a.run) / "calib_tmp.json"
    res = []
    for cfg in grid:
        l_, cells = run(cfg, a, out)
        res.append(dict(dt_scale=cfg[0], temp=cfg[1], loss=l_, cells=cells))
        print(f"dt_scale {cfg[0]} temp {cfg[1]} loss {l_:.3f}", flush=True)
    best = min(res, key=lambda r: r["loss"])
    Path(a.run, "sim_calibration.json").write_text(json.dumps(dict(chosen=dict(dt_scale=best["dt_scale"], temp=best["temp"], clock="anchored"),
                                                                   split="val", grid=res), indent=1))
    out.unlink(missing_ok=True)
    print("chosen", best["dt_scale"], best["temp"], best["loss"])


if __name__ == "__main__":
    main()
