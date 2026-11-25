"""Figures for paper 1 from committed results: (a) gain vs K, (b) gain vs persistence rho, (c) identification R^2."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmi.harness import load_results
ROOT = Path(__file__).resolve().parents[1]


def pooled(exp):
    runs = load_results(exp); keys = runs[0]["metrics"]["regimes"]
    return {k: {p: np.concatenate([np.array(r["metrics"]["regimes"][k][p]["per_episode_cost"]) for r in runs]) for p in keys[k]} for k in keys}


fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
K = pooled("mujoco_v3_ksweep"); Ks = [1, 5, 10, 20, 40]
for reg, c in (("wind", "#26a"), ("both", "#a62")):
    g = [1 - K[f"K{k}/{reg}"]["hindsight_latent_pf"].mean() / K[f"K{k}/{reg}"]["forward"].mean() for k in Ks]
    o = [1 - K[f"K{k}/{reg}"]["oracle_latent"].mean() / K[f"K{k}/{reg}"]["forward"].mean() for k in Ks]
    ax[0].plot(Ks, g, "o-", color=c, label=f"hindsight ({reg})"); ax[0].plot(Ks, o, "--", color=c, alpha=.6, label=f"oracle headroom ({reg})")
ax[0].set_xscale("log"); ax[0].set_xlabel("open-loop chunk length K"); ax[0].set_ylabel("cost reduction vs forward"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
R = pooled("mujoco_v3_rho"); rhos = [0.0, 0.9, 0.99, 1.0]
for p, c in (("hindsight_latent_pf", "#26a"), ("residual", "#c44"), ("oracle_latent", "#555")):
    ax[1].plot(range(4), [1 - R[f"K20/ar{r}"][p].mean() / R[f"K20/ar{r}"]["forward"].mean() for r in rhos], "o-", color=c, label=p)
ax[1].axhline(0, color="k", lw=.8); ax[1].set_xticks(range(4)); ax[1].set_xticklabels([str(r) for r in rhos])
ax[1].set_xlabel("disturbance persistence rho"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
I = load_results("identification")
for reg, comp, c in (("both", "wind_x", "#26a"), ("both", "wind_y", "#2a6"), ("both", "gain", "#a62")):
    A = np.array([r["metrics"][reg]["r2_by_t"][comp] for r in I]).mean(0)
    ax[2].plot(np.arange(1, len(A) + 1), A, color=c, label=comp)
ax[2].set_ylim(-0.5, 1); ax[2].set_xlabel("observed transitions"); ax[2].set_ylabel("R^2 (inferred vs true)"); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
fig.tight_layout(); out = ROOT / "papers" / "latent-hindsight-planning" / "generated" / "mujoco_v3_figure.png"
fig.savefig(out, dpi=150); print("wrote", out)
