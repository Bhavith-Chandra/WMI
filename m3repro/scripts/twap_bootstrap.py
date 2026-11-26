"""Analysis of a TWAP run: cell table, stall audit, impact exponent with path-level bootstrap.

python -I scripts/twap_bootstrap.py results/twap_small_h60.json
Estimator: NLS  I_p = A * (Q_exec,p / 1e4)^beta  over all reached paths of one side (no log, so
non-positive impacts stay in -> no selection on the outcome). Bootstrap: resample baseline paths and,
within each (side,Q) cell, treatment paths, with replacement; recompute baseline mean and refit.
Stall audit: per-cell counts + Fisher exact test (stall vs buy/sell vs baseline) because excluding
stalled paths is selection on a post-treatment outcome. Sensitivity: refit including stalled paths
(their mid at the last reached event).
Writes <input>_analysis.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import fisher_exact

src = Path(sys.argv[1])
r = json.loads(src.read_text())
P = r["paths"]
rng = np.random.default_rng(0)
NB = 4000


def f(q, A, b):
    return A * (q / 1e4) ** b


def fit(q, I):
    try:
        (A, b), _ = curve_fit(f, q, I, p0=(max(np.mean(I), 1.0), 0.5), bounds=([-1e5, -3], [1e5, 3]), maxfev=20000)
        return float(A), float(b)
    except Exception:
        return np.nan, np.nan


def analyse(include_stalled: bool):
    use = [p for p in P if p["logret"] is not None and (p["reached"] or include_stalled)]
    base = np.array([p["logret"] for p in use if p["cell"] == "base"])
    out = dict(n_base=len(base), baseline_mean_bp=float(base.mean() * 1e4), baseline_sd_bp=float(base.std(ddof=1) * 1e4),
               baseline_se_bp=float(base.std(ddof=1) / np.sqrt(len(base)) * 1e4))
    for side, sgn in (("buy", 1), ("sell", -1)):
        cellsQ = sorted({p["Q"] for p in use if p["cell"] == side})
        cells = {q: [p for p in use if p["cell"] == side and p["Q"] == q] for q in cellsQ}

        def stat(bm, sel):
            q = np.concatenate([[p["exec_qty"] for p in sel[k]] for k in cellsQ]).astype(float)
            I = np.concatenate([[sgn * (p["logret"] - bm) * 1e4 for p in sel[k]] for k in cellsQ])
            ok = q > 0
            return fit(q[ok], I[ok])

        A, b = stat(base.mean(), cells)
        bs = []
        for _ in range(NB):
            bm = rng.choice(base, len(base)).mean()
            sel = {k: [v[j] for j in rng.integers(0, len(v), len(v))] for k, v in cells.items()}
            bs.append(stat(bm, sel)[1])
        bs = np.array(bs)
        bs = bs[np.isfinite(bs)]
        tab = []
        for k in cellsQ:
            I = np.array([sgn * (p["logret"] - base.mean()) * 1e4 for p in cells[k]])
            tab.append(dict(Q=k, n=len(I), impact_bp_mean=float(I.mean()), impact_bp_median=float(np.median(I)),
                            impact_bp_se_incl_baseline=float(np.sqrt(I.var(ddof=1) / len(I) + (base.std(ddof=1) * 1e4) ** 2 / len(base))),
                            exec_qty_median=float(np.median([p["exec_qty"] for p in cells[k]]))))
        out[side] = dict(cells=tab, A=A, beta=b, beta_ci90=[float(x) for x in np.percentile(bs, [5, 95])],
                         beta_ci95=[float(x) for x in np.percentile(bs, [2.5, 97.5])], n_boot_ok=int(len(bs)),
                         ci90_contains_0_5=bool(np.percentile(bs, 5) <= 0.5 <= np.percentile(bs, 95)),
                         ci90_contains_1_0=bool(np.percentile(bs, 5) <= 1.0 <= np.percentile(bs, 95)),
                         monotone_mean_impact=bool(np.all(np.diff([t["impact_bp_mean"] for t in tab]) > 0)))
    return out


stall = {}
for cell in ("base", "buy", "sell"):
    ps = [p for p in P if p["cell"] == cell]
    stall[cell] = dict(stalled=sum(not p["reached"] for p in ps), n=len(ps))
tbl = [[stall["buy"]["stalled"], stall["buy"]["n"] - stall["buy"]["stalled"]],
       [stall["sell"]["stalled"], stall["sell"]["n"] - stall["sell"]["stalled"]]]
tb = [[stall["base"]["stalled"], stall["base"]["n"] - stall["base"]["stalled"]],
      [stall["buy"]["stalled"] + stall["sell"]["stalled"], stall["buy"]["n"] + stall["sell"]["n"] - stall["buy"]["stalled"] - stall["sell"]["stalled"]]]
res = dict(
    source=str(src), meta=r["meta"],
    stall=dict(counts=stall, fisher_p_buy_vs_sell=float(fisher_exact(tbl)[1]),
               fisher_p_base_vs_treated=float(fisher_exact(tb)[1]),
               dt_le_1ms_frac_median_stalled=float(np.median([p["dt_le_1ms_frac"] for p in P if not p["reached"]])) if any(not p["reached"] for p in P) else None,
               dt_le_1ms_frac_median_reached=float(np.median([p["dt_le_1ms_frac"] for p in P if p["reached"]]))),
    stale_terminal_mid=sum(p["stale_mid"] for p in P),
    children=dict(skipped=sum(p["children_skipped"] for p in P),
                  exec_frac_of_requested=float(sum(p["exec_qty"] for p in P) / max(sum(p["req_qty"] for p in P), 1)),
                  roundtrip_tick_err_median=float(np.median([p["roundtrip_tick_err_med"] for p in P if p["roundtrip_tick_err_med"] is not None]))
                  if any(p["roundtrip_tick_err_med"] is not None for p in P) else None),
    main_excluding_stalled=analyse(False),
    sensitivity_including_stalled=analyse(True),
    _provenance=r.get("_provenance"),
)
print(json.dumps({k: v for k, v in res.items() if k != "_provenance"}, indent=1))
src.with_name(src.stem + "_analysis.json").write_text(json.dumps(res, indent=2))
