"""M5 real-time viability: CPU latency of the live update path and of rollouts, vs each pair's event rate.

python -I bench_latency.py --events DIR --run runs/X --out results/m5.json
* update latency: tokenise + incremental features + one KV-cached model step for the live context (1 path)
* rollout throughput: simulated events per wall second for 1/8/16/32 paths
* pair event rates (median and p99 over 10-s windows) from the TRAIN split (no test data needed)
Real-time factor = rollout events per wall second / pair event rate (>1 = faster than the market).
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
from data.dataset import raw_chunks  # noqa: E402
from data.tokenizer import Tokenizer  # noqa: E402
from evaluate import load_model  # noqa: E402
from model.infer import FastMMFM  # noqa: E402
from sim.book import LevelBook  # noqa: E402
from sim.rollout import Rollout, context_from_history  # noqa: E402

torch.set_num_threads(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    events = Path(a.events)
    model, mhash = load_model(a.run)
    tok = Tokenizer.from_json(json.loads((Path(a.run) / "tokenizer.json").read_text()))
    chunks, _ = raw_chunks(events, "train")
    rates = {}
    for c in chunks:
        us = c["events"]["us"]
        if len(us) < 5000:
            continue
        w = np.histogram(us, bins=np.arange(us[0], us[-1], 10_000_000))[0] / 10.0
        rates.setdefault(c["pair"], []).append(w)
    rates = {p: dict(median=float(np.median(np.concatenate(v))), p99=float(np.percentile(np.concatenate(v), 99))) for p, v in rates.items()}
    # context: a train checkpoint of the busiest pair
    pair = max(rates, key=lambda p: rates[p]["median"])
    c = max((c for c in chunks if c["pair"] == pair), key=lambda c: len(c["events"]))
    z = np.load(events / c["file"])
    j = [k for k, i in enumerate(z["ck_idx"]) if i > 1200][0]
    idx = int(z["ck_idx"][j])
    tick = float(z["tick"])
    evs = np.zeros(idx, dtype=[("us", "i8"), ("type", "i1"), ("side", "i1"), ("price", "f8"), ("qty", "f8")])
    for k in evs.dtype.names:
        evs[k] = z[k][:idx]
    b = LevelBook(tick)
    for arr, d in ((z["ck_bids"][j], b.bids), (z["ck_asks"][j], b.asks)):
        for p, q in arr:
            if np.isfinite(p):
                d[int(round(p / tick))] = float(q)
    ctx = context_from_history(evs[-1100:], z["state"][idx - 1100:idx], tok, b, model.cfg["ctx"])
    # live update latency (1 path): KV step with a new token/feature row
    fm = FastMMFM(model)
    tk = torch.from_numpy(ctx.tokens)[None]
    ft = torch.from_numpy(ctx.feats)[None]
    fm.prefill(tk, ft)
    lat = []
    for i in range(400):
        t0 = time.perf_counter()
        fm.step(tk[:, i % tk.shape[1]], ft[:, i % ft.shape[1]])
        lat.append(time.perf_counter() - t0)
    lat = np.array(lat[50:]) * 1e3
    out = dict(model_hash=mhash, n_params=sum(p.numel() for p in model.parameters()), threads=torch.get_num_threads(),
               update_latency_ms=dict(p50=float(np.percentile(lat, 50)), p99=float(np.percentile(lat, 99))),
               pair_event_rates_per_s=rates, rollout={})
    for n in (1, 8, 16, 32):
        R = Rollout(model, tok, ctx, n_paths=n, seed=0)
        t0 = time.perf_counter()
        for _ in range(300):
            R.step()
        el = time.perf_counter() - t0
        ev_per_s = 300 / el  # events per path per wall second
        out["rollout"][str(n)] = dict(ms_per_step=1000 * el / 300, sim_events_per_path_per_s=ev_per_s,
                                      realtime_factor_vs_median_rate={p: ev_per_s / r["median"] for p, r in rates.items()},
                                      realtime_factor_vs_p99_rate={p: ev_per_s / r["p99"] for p, r in rates.items()})
    out["_provenance"] = dict(git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                              argv=sys.argv, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    p = Path(a.out or ROOT / "results" / "m5.json")
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "_provenance"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
