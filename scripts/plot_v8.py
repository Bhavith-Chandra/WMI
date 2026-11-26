"""Figure and table for the realistic deployment benchmark (v8), from results/ only."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import NAME, PCOL, panel, setup  # noqa: E402
from wmi.harness import load_results  # noqa: E402

setup()
G = ROOT / "papers" / "latent-hindsight-planning" / "generated"
rng = np.random.default_rng(0)
R = sorted(load_results("realistic_v8"), key=lambda r: r["provenance"]["config"]["seed"])
R7 = sorted(load_results("hier_main"), key=lambda r: r["provenance"]["config"]["seed"])
cat = lambda RR, key, p: np.concatenate([np.array(r["metrics"]["regimes"][key][p]["success"]) for r in RR])


def ci(x, B=5000):
    i = rng.integers(0, len(x), (B, len(x))); m = x[i].mean(1)
    return x.mean(), np.quantile(m, .025), np.quantile(m, .975)


PL = ["hier:forward", "hier:residual", "hier:online_sgd", "hier:hindsight_posterior", "hier:hindsight_latent_pf_adapt", "hier:oracle_latent"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
cols = [("K10/both", R7, "idealised"), ("deploy/both", R, "deployment"), ("deploy/none", R, "deployment,\nno latent"),
        ("deploy/fresh", R, "deployment,\nfresh latent")]
w = 0.8 / len(PL)
for j, p in enumerate(PL):
    for i, (key, RR, _) in enumerate(cols):
        m, lo, hi = ci(cat(RR, key, p))
        ax[0].bar(i + (j - (len(PL) - 1) / 2) * w, m, width=w * .92, color=PCOL[p], yerr=[[m - lo], [hi - m]], capsize=1.2,
                  ecolor="#333333", lw=0, label=NAME[p] if i == 0 else None)
ax[0].set_xticks(range(len(cols))); ax[0].set_xticklabels([c[2] for c in cols]); ax[0].set_ylabel("main-goal success")
ax[0].set_ylim(0, 1.08); ax[0].set_title("persistent gain + wind unless stated", fontsize=8.5)
ax[0].legend(ncol=3, frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(.5, 1.38)); panel(ax[0], "a")
abl = [("K10/both", R7, "none"), ("only_mismatch/both", R, "unmodelled\nmass, damping"), ("only_noise/both", R, "sensor\nnoise"),
       ("only_delay/both", R, "actuation +\ncompute delay"), ("deploy/both", R, "all three")]
for p in ("hier:forward", "hier:hindsight_posterior", "hier:oracle_latent"):
    ys = [cat(RR, key, p).mean() for key, RR, _ in abl]
    ax[1].plot(range(len(abl)), ys, "-o", ms=3.5, color=PCOL[p], label=NAME[p], lw=1.4)
ax[1].set_xticks(range(len(abl))); ax[1].set_xticklabels([a[2] for a in abl], fontsize=7); ax[1].set_ylim(0, 1.05)
ax[1].set_ylabel("main-goal success"); ax[1].set_title("realism factors one at a time", fontsize=8.5); ax[1].legend(frameon=False, fontsize=6.8)
panel(ax[1], "b")
fig.tight_layout(); fig.savefig(G / "v8_figure.png"); print("wrote v8_figure.png")
