"""Closed-loop TWAP meta-order impact on the released example (paper Fig. 8b protocol, single prompt).

python -I scripts/run_twap.py --model tiny --horizon 300 --children 20 --reps 16
Impact_i = side * (log m_i(T) - mean_baseline log m(T)); fit log I vs log Q (bin = Q level, median over reps).
Daily volume / daily vol for this security-day are NOT in the release, so Q is in shares and the
exponent (scale-free) is the comparable quantity; paper reports 0.532 (buy) / 0.524 (sell).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from m3.model import M3, VQTokenizer
from m3.engine import Book, decode_to_orders, replay
from m3.closedloop import ClosedLoop
REL = Path("/home/user/ext/m3-market-microstructure"); OUT = ROOT / "results"
torch.set_num_threads(4)
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="tiny"); ap.add_argument("--horizon", type=float, default=300)
ap.add_argument("--children", type=int, default=20); ap.add_argument("--reps", type=int, default=16)
ap.add_argument("--qgrid", default="2000,5000,10000,20000,50000,100000,200000")
ap.add_argument("--seed", type=int, default=7); ap.add_argument("--max-steps", type=int, default=4000); ap.add_argument("--block", type=int, default=32)
a = ap.parse_args()
codes = torch.from_numpy(np.load(REL / "examples/prompt_ids.npy").astype(np.int64))
c = np.load(REL / "examples/conditioning.npz"); lob = torch.tensor(c["lob"]).float(); st = torch.tensor(c["initial_state_float"]).float()
tok = VQTokenizer.from_checkpoint(REL / "tokenizer/base/best.pt"); m = M3.from_checkpoint(REL / a.model / "best.pt")
anchor = float(st[0, 4]); book0 = Book.from_snapshot(lob[0].numpy())
bp, sp = replay(book0, *decode_to_orders(tok.decode(codes)[0].numpy(), anchor))
Q = [int(x) for x in a.qgrid.split(",")]
cells = [("base", 0, 0)] + [(sd, s, q) for q in Q for sd, s in (("buy", 1), ("sell", 0))]
cells = [(n, s, q) for n, s, q in cells]
rows, sched = [], []
dt_child = a.horizon / a.children
for n, s, q in cells:
    for r in range(a.reps):
        rows.append((n, s, q, r))
        if n == "base":
            sched.append([])
        else:
            per = int(round(q / a.children / 100.0)) * 100
            sched.append([(dt_child * (k + 1) - 1e-9, s, max(per, 100)) for k in range(a.children)])
B = len(rows); print("batch", B, "cells", len(cells))
t0 = time.time()
sim = ClosedLoop(m, tok, codes, lob, st, bp, float(sp.t[-1]), B, block=a.block, seed=a.seed)
steps = sim.run(a.horizon, sched, max_steps=a.max_steps)
reached = (sim.t - sim.t_start) >= a.horizon
mT = sim.terminal_mid(a.horizon); mid_start = bp.mid()
lr = np.where(reached, np.log(mT / mid_start), np.nan)  # stalled paths excluded
base = lr[[i for i, r in enumerate(rows) if r[0] == "base"]]
res = dict(meta=dict(model=a.model, horizon=a.horizon, children=a.children, reps=a.reps, steps=steps, B=B,
                     wall_s=time.time() - t0, block=a.block, seed=a.seed, mid_start=mid_start),
           baseline=dict(mean_bp=float(np.nanmean(base) * 1e4), sd_bp=float(np.nanstd(base) * 1e4), n=int(np.isfinite(base).sum())),
           cells=[],
           stall=dict(rule=f"path stalled if sim clock < horizon after max_steps={a.max_steps} generated events",
                      n_stalled=int((~reached).sum()), n_total=B,
                      stalled_by_cell={f"{n}_{q}": int(sum(1 for i, r in enumerate(rows) if r[0] == n and r[2] == q and not reached[i])) for n, _, q in cells},
                      sim_t_reached_stalled=[float(x) for x in (sim.t - sim.t_start)[~reached]],
                      dt_le_1ms_frac_stalled=[float(x) for x in (sim.n_dt0 / np.maximum(sim.n_ev, 1))[~reached]],
                      dt_le_1ms_frac_ok_median=float(np.median((sim.n_dt0 / np.maximum(sim.n_ev, 1))[reached])) if reached.any() else None))
for n, s, q in cells[1:]:
    idx = [i for i, r in enumerate(rows) if r[0] == n and r[2] == q]
    sign = 1 if s == 1 else -1
    imp = sign * (lr[idx] - np.nanmean(base)) * 1e4
    filled = [sum(f for _, _, f in sim.injected[i]) for i in idx]
    nch = [len(sim.injected[i]) for i in idx]
    res["cells"].append(dict(side=n, Q=q, impact_bp_mean=float(np.nanmean(imp)), impact_bp_median=float(np.nanmedian(imp)),
                             impact_bp_se=float(np.nanstd(imp) / np.sqrt(max(np.isfinite(imp).sum(), 1))), n_ok=int(np.isfinite(imp).sum()),
                             exec_qty_median=float(np.median(filled)), children_sent_median=float(np.median(nch))))
for side in ("buy", "sell"):
    cs = [x for x in res["cells"] if x["side"] == side]
    q = np.array([x["exec_qty_median"] for x in cs]); I = np.array([x["impact_bp_median"] for x in cs])
    ok = (q > 0) & (I > 0)
    if ok.sum() >= 3:
        b, a0 = np.polyfit(np.log(q[ok]), np.log(I[ok]), 1)
        pred = a0 + b * np.log(q[ok]); ss = np.sum((np.log(I[ok]) - pred) ** 2); st_ = np.sum((np.log(I[ok]) - np.log(I[ok]).mean()) ** 2)
        res[f"fit_{side}"] = dict(beta=float(b), r2=float(1 - ss / st_), n_points=int(ok.sum()), n_excluded_nonpositive=int((~ok).sum()))
    else:
        res[f"fit_{side}"] = dict(beta=None, n_points=int(ok.sum()), note="<3 positive-impact bins")
print(json.dumps(res, indent=1))
(OUT / f"twap_{a.model}_h{int(a.horizon)}.json").write_text(json.dumps(res, indent=2))
