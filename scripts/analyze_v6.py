"""Analysis for v6 (T1-T4 throw, D1-D2 latency), written before results."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_mujoco import paired_ratio, verdict
from wmi.harness import load_results
rng = np.random.default_rng(0); out, checks = [], {}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


T = load_results("throw")
if T:
    out.append(f"throw: commits {sorted({r['provenance']['git_commit'][:7] for r in T})}, seeds {len(T)}")
    miss = {reg: {k: np.concatenate([np.array(r["metrics"]["miss"][reg][k]) for r in T], axis=1) for k in T[0]["metrics"]["miss"][reg]}
            for reg in T[0]["metrics"]["miss"]}                                            # (A, E_total)
    summ = lambda reg, k: miss[reg][k][1:].mean(0)                                         # per-episode mean over attempts 2-10
    for reg in miss:
        out.append(f"{reg}: " + str({k: [round(float(miss[reg][k][i].mean()), 3) for i in (0, 1, 2, 4, 9)] for k in miss[reg]}))
        out.append(f"   summary ratio vs forward: " + str({k: (round(paired_ratio(summ(reg, k), summ(reg, 'forward'), rng)[0], 3),
                   [round(c, 3) for c in paired_ratio(summ(reg, k), summ(reg, 'forward'), rng)[1]]) for k in miss[reg]}))
    checks["T1 wind/damping/both: hindsight summary miss <= 0.5 x forward (upper CI < 0.5)"] = all(
        paired_ratio(summ(r, "hindsight"), summ(r, "forward"), rng)[1][1] < 0.5 for r in ("wind", "damping", "both"))
    checks["T2 damping/both: hindsight beats residual"] = all(
        verdict(paired_ratio(summ(r, "hindsight"), summ(r, "residual"), rng)[1]) == "beats" for r in ("damping", "both"))
    checks["T3 fresh: hindsight not better than forward"] = verdict(paired_ratio(summ("fresh", "hindsight"), summ("fresh", "forward"), rng)[1]) != "beats"
    checks["T4 wind/damping/both: attempt-10 miss hindsight <= 2 x oracle"] = all(
        miss[r]["hindsight"][9].mean() <= 2 * miss[r]["oracle"][9].mean() for r in ("wind", "damping", "both"))
D = load_results("mujoco_v3_delay")
if D:
    out.append(f"delay: commits {sorted({r['provenance']['git_commit'][:7] for r in D})}, seeds {len(D)}")
    # files may hold one seed x all delays, or one seed x one delay: merge per key, ordered by seed
    D = sorted(D, key=lambda r: r["provenance"]["config"]["seed"])
    keys = sorted({k for r in D for k in r["metrics"]["regimes"]})
    P = {k: {p: np.concatenate([np.array(r["metrics"]["regimes"][k][p]["per_episode_cost"]) for r in D if k in r["metrics"]["regimes"]])
             for p in next(r for r in D if k in r["metrics"]["regimes"])["metrics"]["regimes"][k]} for k in keys}
    out.append("delay seeds per key: " + str({k: sum(k in r["metrics"]["regimes"] for r in D) for k in keys}))
    ds = [0, 5, 10, 20]; rel = lambda d, p, q="forward": paired_ratio(P[f"delay{d}/both"][p], P[f"delay{d}/both"][q], rng)
    for d in ds:
        out.append(f"delay {d}: " + str({p: (round(rel(d, p)[0], 3), [round(c, 3) for c in rel(d, p)[1]]) for p in P[f'delay{d}/both']}))
    gains = [1 - rel(d, "hindsight_posterior")[0] for d in ds]
    checks["D1 posterior gain(d=20) > gain(d=0) and Spearman(d, gain) > 0"] = gains[-1] > gains[0] and spearman(ds, gains) > 0
    checks["D2 d=20: posterior and pf_adapt beat forward AND online_sgd"] = all(
        verdict(rel(20, p, q)[1]) == "beats" for p in ("hindsight_posterior", "hindsight_latent_pf_adapt") for q in ("forward", "online_sgd"))
out += [("SUPPORTED " if v else "NOT SUPPORTED ") + k for k, v in checks.items()]
print("\n".join(out))
(Path(__file__).resolve().parents[1] / "papers" / "latent-hindsight-planning" / "generated" / "v6_summary.txt").write_text("\n".join(out) + "\n")
