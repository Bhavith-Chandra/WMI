"""EVAL_v2 criterion 7: real-time factor of the v2 simulator (and direct-head latency), design data only.

python -I bench_v2.py --run runs/v2 --events /home/user/data/mmfm_v2 --out results/v2_m5.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data.dataset_v2 import raw_chunks  # noqa: E402
from data.tokenizer_v2 import Tokenizer2  # noqa: E402
from sim.book import LevelBook  # noqa: E402
from sim.rollout2 import Rollout2, context_from_history2, direct_forecast  # noqa: E402
from sim_eval_v2 import load_v2  # noqa: E402

torch.set_num_threads(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m, mh = load_v2(a.run)
    tok = Tokenizer2.from_json(json.loads((Path(a.run) / "tokenizer.json").read_text()))
    cal = json.loads((Path(a.run) / "sim_calibration.json").read_text())["chosen"] if (Path(a.run) / "sim_calibration.json").exists() else dict(dt_scale=1.0, temp=1.0, clock="model")
    ch, _ = raw_chunks(Path(a.events), "train")
    rates = {}
    for c in ch:
        us = c["events"]["us"]
        if len(us) > 5000:
            w = np.histogram(us, bins=np.arange(us[0], us[-1], 10_000_000))[0] / 10.0
            rates.setdefault(c["pair"], []).append(w)
    rates = {p: dict(median=float(np.median(np.concatenate(v))), p90=float(np.percentile(np.concatenate(v), 90))) for p, v in rates.items()}
    out = dict(model_hash=mh, calibration=cal, threads=torch.get_num_threads(), pair_event_rates=rates, per_pair={})
    for pair in ("ethusd", "btcusd", "xrpusd", "dogeusd"):
        c = max((c for c in ch if c["pair"] == pair), key=lambda c: len(c["events"]))
        z = np.load(Path(a.events) / c["file"])
        j = len(z["ck_idx"]) // 2
        idx, tick = int(z["ck_idx"][j]), float(z["tick"])
        evs = np.zeros(idx, dtype=[("us", "i8"), ("type", "i1"), ("side", "i1"), ("price", "f8"), ("qty", "f8")])
        for k in evs.dtype.names:
            evs[k] = z[k][:idx]
        b = LevelBook(tick)
        for arr, d in ((z["ck_bids"][j], b.bids), (z["ck_asks"][j], b.asks)):
            for p, q in arr:
                if np.isfinite(p):
                    d[int(round(p / tick))] = float(q)
        ctx = context_from_history2(evs[-1100:], z["state"][idx - 1100:idx], z["lvq"][idx - 1100:idx], tok, b, m.cfg["ctx"])
        t0 = time.perf_counter()
        for _ in range(50):
            direct_forecast(m, tok, ctx)
        dms = (time.perf_counter() - t0) / 50 * 1e3
        rr = {}
        for n in (8, 16):
            R = Rollout2(m, tok, ctx, n_paths=n, seed=0, temperature=cal["temp"], dt_scale=cal["dt_scale"],
                       clock=cal.get("clock", "model"))
            t0 = time.perf_counter()
            for _ in range(400):
                R.step()
            el = time.perf_counter() - t0
            evps = 400 / el
            rr[str(n)] = dict(ms_per_step=1e3 * el / 400, sim_events_per_path_per_s=evps,
                              realtime_factor_median_rate=evps / rates[pair]["median"], realtime_factor_p90_rate=evps / rates[pair]["p90"])
        out["per_pair"][pair] = dict(direct_head_ms=dms, rollout=rr)
        print(pair, json.dumps(out["per_pair"][pair]), flush=True)
    out["_provenance"] = dict(git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                              argv=sys.argv, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
