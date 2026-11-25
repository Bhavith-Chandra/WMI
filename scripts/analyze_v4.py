"""Analysis for MuJoCo v4 (N1-N4, F2, F3), written before results."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_mujoco import paired_ratio, verdict
from wmi.harness import load_results
rng = np.random.default_rng(0); out, checks = [], {}
runs = load_results("mujoco_v3_noise")
if runs:
    keys = runs[0]["metrics"]["regimes"]
    P = {k: {p: np.concatenate([np.array(r["metrics"]["regimes"][k][p]["per_episode_cost"]) for r in runs]) for p in keys[k]} for k in keys}
    G = (0.0, 0.25, 0.5, 1.0); key = lambda g: f"gust{g}/wind"
    rel = lambda g, p, q="forward": paired_ratio(P[key(g)][p], P[key(g)][q], rng)
    for g in G:
        out.append(f"gust {g}: " + str({p: (round(rel(g, p)[0], 3), [round(c, 3) for c in rel(g, p)[1]]) for p in P[key(g)]}))
    checks["N1 s=0: residual <= latent_pf (not worse)"] = verdict(rel(0.0, "residual", "hindsight_latent_pf")[1]) != "worse"
    rr = [rel(g, "residual")[0] for g in G]
    checks["N2 residual ratio vs forward increases monotonically in s"] = all(np.diff(rr) > 0)
    checks["N3 s=1: posterior AND latent_pf beat residual"] = all(verdict(rel(1.0, p, "residual")[1]) == "beats" for p in ("hindsight_posterior", "hindsight_latent_pf"))
    g0, g1 = 1 - rel(0.0, "hindsight_posterior")[0], 1 - rel(1.0, "hindsight_posterior")[0]
    checks["N4 posterior gain at s=1 >= half its gain at s=0"] = g1 >= 0.5 * g0
    checks["F3 pf_bayes not worse than pf at every s"] = all(verdict(rel(g, "hindsight_latent_pf_bayes", "hindsight_latent_pf")[1]) != "worse" for g in G)
    out.append(f"provenance: commits {sorted({r['provenance']['git_commit'][:7] for r in runs})}, seeds {len(runs)}")
I = load_results("identification_v4")
if I:
    bnd = [20, 40, 60, 80, 100, 120, 140]
    for reg in ("both", "wind"):
        for comp in ("wind_x", "wind_y", "predictive_dv"):
            A = np.array([r["metrics"][reg]["r2_by_t"][comp] for r in I]).mean(0)
            dip = np.mean([A[b] - A[b - 5:b].mean() for b in bnd])
            out.append(f"identification {reg} {comp}: dip {dip:.3f}, R2@20 {A[19]:.3f}, R2@160 {A[-1]:.3f}")
            if comp == "predictive_dv":
                checks[f"F2 {reg}: predictive R^2 boundary dip > -0.05"] = dip > -0.05
out += [("SUPPORTED " if v else "NOT SUPPORTED ") + k for k, v in checks.items()]
print("\n".join(out))
(Path(__file__).resolve().parents[1] / "papers" / "latent-hindsight-planning" / "generated" / "mujoco_v4_summary.txt").write_text("\n".join(out) + "\n")
