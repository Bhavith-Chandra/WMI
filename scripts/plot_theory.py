"""Figure: cumulative regret vs t (log-log) for each planner and regime, from results/theory_hindsight (curves_at_t)."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmi.harness import load_results
runs = load_results("theory_hindsight")
t = np.array(runs[0]["curves_at_t"]["t"])
regs = ["persistent", "persistent_gain", "fresh", "switching"]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, reg in zip(axes, regs):
    for p, c in (("forward", "#555"), ("residual", "#c44"), ("hindsight_bayes", "#26a"), ("hindsight_forget", "#2a6")):
        Y = np.array([r["curves_at_t"][f"{reg}/{p}"] for r in runs]); m = Y.mean(0)
        ax.loglog(t, np.maximum(m, 1e-3), label=p, color=c)
    ax.set_title(reg.replace("_", " ")); ax.set_xlabel("t"); ax.grid(alpha=.3)
axes[0].set_ylabel("cumulative regret"); axes[0].legend(fontsize=8)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "papers" / "latent-hindsight-planning" / "generated" / "theory_regret.png"
fig.savefig(out, dpi=150); print("wrote", out)
