"""Analysis for MuJoCo v3 (rules R1-R6 from mujoco_v3_preregistration.md, written before results)."""
import sys
from pathlib import Path

import numpy as np
def spearmanr(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return (float(np.corrcoef(ra, rb)[0, 1]),)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_mujoco import paired_ratio, verdict  # noqa: E402
from wmi.harness import load_results  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)


def pooled(exp):
    runs = load_results(exp)
    if not runs:
        return None, None
    keys = runs[0]["metrics"]["regimes"]
    P = {k: {p: np.concatenate([np.array(r["metrics"]["regimes"][k][p]["per_episode_cost"]) for r in runs])
             for p in keys[k]} for k in keys}
    prov = sorted({r["provenance"]["git_commit"][:7] for r in runs}), any(r["provenance"]["git_dirty"] for r in runs), len(runs)
    return P, prov


def rel(P, k, p, q="forward"):
    return paired_ratio(P[k][p], P[k][q], rng)


out, checks = [], {}
M, prov = pooled("mujoco_v3_main")
if M:
    out.append(f"main: commits {prov[0]} dirty {prov[1]} seeds {prov[2]}")
    for k in M:
        out.append(k + " " + str({p: (round(rel(M, k, p)[0], 3), [round(c, 3) for c in rel(M, k, p)[1]]) for p in M[k]}))
    regs = [r for r in ("wind", "both", "gain", "ood_gain") if f"K20/{r}" in M]
    checks["R1 latent_pf beats forward AND online_sgd on wind/both/gain/ood_gain"] = all(
        verdict(rel(M, f"K20/{r}", "hindsight_latent_pf")[1]) == "beats" and
        verdict(rel(M, f"K20/{r}", "hindsight_latent_pf", "online_sgd")[1]) == "beats" for r in regs)
    caps = {r: (1 - rel(M, f"K20/{r}", "hindsight_latent_pf")[0]) / (1 - rel(M, f"K20/{r}", "oracle_latent")[0]) for r in ("wind", "both")}
    out.append(f"headroom captured by latent_pf: {caps}")
    checks["R2 latent_pf captures >= 50% of oracle headroom on wind and both"] = all(v >= 0.5 for v in caps.values())
    if "K20/fresh" in M:
        checks["R6 fresh: no hindsight method beats forward"] = all(
            verdict(rel(M, "K20/fresh", p)[1]) != "beats" for p in M["K20/fresh"] if p not in ("forward", "oracle_latent"))
K, provk = pooled("mujoco_v3_ksweep")
if K:
    out.append(f"ksweep: commits {provk[0]} seeds {provk[2]}")
    for reg in ("wind", "both"):
        gains = {kk: 1 - rel(K, f"K{kk}/{reg}", "hindsight_latent_pf")[0] for kk in (1, 5, 10, 20, 40)}
        heads = {kk: 1 - rel(K, f"K{kk}/{reg}", "oracle_latent")[0] for kk in (1, 5, 10, 20, 40)}
        out.append(f"{reg} latent_pf gain by K {gains} | oracle headroom by K {heads}")
        checks[f"R3 {reg}: gain(K=40) > gain(K=1) and Spearman(K, gain) > 0"] = gains[40] > gains[1] and spearmanr(list(gains), list(gains.values()))[0] > 0
R, provr = pooled("mujoco_v3_rho")
if R:
    out.append(f"rho: commits {provr[0]} seeds {provr[2]}")
    g = {r: rel(R, f"K20/ar{r}", "hindsight_latent_pf") for r in (0.0, 0.9, 0.99, 1.0)}
    out.append("rho latent_pf ratio " + str({r: (round(v[0], 3), [round(c, 3) for c in v[1]]) for r, v in g.items()}))
    out.append("rho residual ratio " + str({r: round(rel(R, f'K20/ar{r}', 'residual')[0], 3) for r in (0.0, 0.9, 0.99, 1.0)}))
    out.append("rho oracle ratio " + str({r: round(rel(R, f'K20/ar{r}', 'oracle_latent')[0], 3) for r in (0.0, 0.9, 0.99, 1.0)}))
    gr = [1 - g[r][0] for r in (0.0, 0.9, 0.99, 1.0)]
    checks["R4 latent_pf gain monotone in rho and no significant gain at rho=0"] = all(np.diff(gr) > 0) and verdict(g[0.0][1]) != "beats"
    checks["R5 residual worse than forward at rho=0 and 0.9"] = all(verdict(rel(R, f"K20/ar{r}", "residual")[1]) == "worse" for r in (0.0, 0.9))
out += [("SUPPORTED " if v else "NOT SUPPORTED ") + k for k, v in checks.items()]
print("\n".join(out))
(ROOT / "papers" / "latent-hindsight-planning" / "generated" / "mujoco_v3_summary.txt").write_text("\n".join(out) + "\n")
