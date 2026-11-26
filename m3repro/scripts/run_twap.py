"""Closed-loop TWAP meta-order impact on the released example (paper Fig. 8b protocol, single prompt).

python -I scripts/run_twap.py --model small --horizon 60 --children 10 --reps 12 --qgrid 5000,12500,31250,78000,195000
* rows are independent paths (see m3/closedloop.py); children are IOC at touch -/+ 10 ticks,
  scheduled at (k-0.5)*H/N, k=1..N, so every child executes strictly before the horizon
* impact_i = side_sign * (log m_i(T) - mean_baseline log m(T)) in bp
* exponent: NLS fit I = A * (Q_exec/1e4)^beta over ALL paths of a side (non-positive impacts kept),
  path-level bootstrap (resample baseline paths and paths within each cell); see scripts/twap_bootstrap.py
Daily volume / volatility are not released, so Q is in shares and only beta is comparable (paper 0.532/0.524).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from m3.closedloop import ClosedLoop  # noqa: E402
from m3.engine import Book, decode_to_orders, replay  # noqa: E402
from m3.model import M3, VQTokenizer  # noqa: E402
from m3.provenance import INPUTS, stamp  # noqa: E402

OUT = ROOT / "results"
torch.set_num_threads(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--horizon", type=float, default=60)
    ap.add_argument("--children", type=int, default=10)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--qgrid", default="10000,40000,160000")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    codes = torch.from_numpy(np.load(INPUTS["example_prompt_ids"]).astype(np.int64))
    c = np.load(INPUTS["example_conditioning"])
    lob, st = torch.tensor(c["lob"]).float(), torch.tensor(c["initial_state_float"]).float()
    tok, m = VQTokenizer.from_checkpoint(INPUTS["tokenizer"]), M3.from_checkpoint(INPUTS[a.model])
    anchor = float(st[0, 4])
    bp, sp = replay(Book.from_snapshot(lob[0].numpy()), *decode_to_orders(tok.decode(codes)[0].numpy(), anchor))
    Q = [int(x) for x in a.qgrid.split(",")]
    cells = [("base", -1, 0)] + [(n, s, q) for q in Q for n, s in (("buy", 1), ("sell", 0))]
    rows, sched = [], []
    dtc = a.horizon / a.children
    for n, s, q in cells:
        per = max(int(round(q / a.children / 100.0)) * 100, 100)
        for r in range(a.reps):
            rows.append((n, s, q, r))
            sched.append([] if n == "base" else [((k - 0.5) * dtc, s, per) for k in range(1, a.children + 1)])
    B = len(rows)
    print("batch", B, "cells", len(cells), flush=True)
    t0 = time.time()
    sim = ClosedLoop(m, tok, codes, lob, st, bp, float(sp.t[-1]), B, block=a.block, seed=a.seed)
    steps = sim.run(a.horizon, sched, max_steps=a.max_steps)
    mT, stale = sim.terminal(a.horizon)
    mid_start = bp.mid()
    reached = sim.done.copy()
    lr = np.log(mT / mid_start)
    paths = []
    for i, (n, s, q, r) in enumerate(rows):
        inj = sim.injected[i]
        rt = [e for x in inj for e in x.get("roundtrip_tick_err", [])]
        paths.append(dict(cell=n, side=int(s), Q=int(q), rep=int(r), reached=bool(reached[i]), stale_mid=bool(stale[i]),
                          logret=None if not np.isfinite(lr[i]) else float(lr[i]),
                          sim_t=float(sim.t[i] - sim.t_start[i]), n_events=int(sim.n_ev[i]),
                          dt_le_1ms_frac=float(sim.n_dt0[i] / max(sim.n_ev[i], 1)),
                          children_sched=len(sched[i]), children_exec=sum(1 for x in inj if not x["skipped"]),
                          children_skipped=sum(1 for x in inj if x["skipped"]),
                          exec_qty=int(sum(x["filled"] for x in inj)), req_qty=int(sum(x["qty"] for x in inj)),
                          roundtrip_tick_err_med=float(np.median(rt)) if rt else None))
    res = dict(meta=dict(model=a.model, horizon=a.horizon, children=a.children, reps=a.reps, qgrid=Q, steps=steps, B=B,
                         block=a.block, seed=a.seed, max_steps=a.max_steps, mid_start=mid_start, n_rolls=sim.n_rolls,
                         roll_at=sim.roll_at, wall_s=time.time() - t0, simulator="closedloop v2 (independent rows, IOC)"),
               paths=paths)
    res["_provenance"] = stamp(["tokenizer", a.model, "example_prompt_ids", "example_conditioning"])
    out = OUT / f"twap_{a.model}_h{int(a.horizon)}{a.tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print("->", out)


if __name__ == "__main__":
    main()
