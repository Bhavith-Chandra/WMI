"""Rollout-level bootstrap of the TWAP impact exponent.

python -I scripts/twap_bootstrap.py results/twap_small_h60_q5r12.json
Each replicate resamples baseline paths and, within every (side,Q) cell, treatment paths with
replacement; impact = side_sign*(logret - mean baseline logret) in bp; slope of log(median impact)
on log(median executed qty). Replicates with any non-positive cell median are counted, not dropped silently.
Also tests H0: beta=0.5 vs H1: beta=1 by reporting the fraction of replicates closer to each.
"""
import json
import sys

import numpy as np

r = json.load(open(sys.argv[1]))
P = [p for p in r["paths"] if p["reached"]]
base = np.array([p["logret"] for p in P if p["cell"] == "base"])
rng = np.random.default_rng(0)
NB = 20000
out = dict(source=sys.argv[1], n_base=len(base), B=NB)
for side, sgn in (("buy", 1), ("sell", -1)):
    Qs = sorted({p["Q"] for p in P if p["cell"] == side})
    cells = {q: [(p["logret"], p["exec_qty"]) for p in P if p["cell"] == side and p["Q"] == q] for q in Qs}
    point = []
    for q in Qs:
        lr = np.array([x for x, _ in cells[q]])
        point.append((np.median([e for _, e in cells[q]]), np.median(sgn * (lr - base.mean()) * 1e4)))
    betas, nonpos = [], 0
    for _ in range(NB):
        bm = rng.choice(base, len(base)).mean()
        xs, ys, ok = [], [], True
        for q in Qs:
            idx = rng.integers(0, len(cells[q]), len(cells[q]))
            lr = np.array([cells[q][i][0] for i in idx])
            ex = np.array([cells[q][i][1] for i in idx])
            I = np.median(sgn * (lr - bm) * 1e4)
            if I <= 0:
                ok = False
                break
            xs.append(np.log(np.median(ex)))
            ys.append(np.log(I))
        if not ok:
            nonpos += 1
            continue
        betas.append(np.polyfit(xs, ys, 1)[0])
    betas = np.array(betas)
    x = np.log([a for a, _ in point])
    y = np.array([b for _, b in point])
    pos = y > 0
    b_pt = float(np.polyfit(x[pos], np.log(y[pos]), 1)[0]) if pos.sum() >= 3 else None
    out[side] = dict(
        cells=[dict(Q=q, n=len(cells[q]), exec_qty_median=float(a), impact_bp_median=float(b)) for q, (a, b) in zip(Qs, point)],
        beta_point=b_pt, n_positive_bins=int(pos.sum()),
        beta_ci90=[float(v) for v in np.percentile(betas, [5, 95])] if len(betas) else None,
        beta_median=float(np.median(betas)) if len(betas) else None,
        frac_replicates_nonpositive_bin=nonpos / NB,
        frac_closer_to_sqrt_than_linear=float(np.mean(np.abs(betas - 0.5) < np.abs(betas - 1.0))) if len(betas) else None,
        monotone_increasing_point=bool(np.all(np.diff(y) > 0)),
    )
print(json.dumps(out, indent=1))
open(sys.argv[1].replace(".json", "_bootstrap.json"), "w").write(json.dumps(out, indent=2))
