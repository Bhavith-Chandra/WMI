"""Figure for the throw-and-retry (E-A) and latency (E-B) experiments, from results/ only."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.figstyle import COL, panel, setup  # noqa: E402
from wmi.harness import load_results  # noqa: E402

setup()

G = Path(__file__).resolve().parents[1] / "papers" / "latent-hindsight-planning" / "generated"
TH = load_results("throw")
miss = {reg: {pl: np.concatenate([np.array(r["metrics"]["miss"][reg][pl]) for r in TH], 1) for pl in TH[0]["metrics"]["miss"][reg]}
        for reg in TH[0]["metrics"]["miss"]}
DL = load_results("mujoco_v3_delay")
cost = {}
for r in DL:
    for k, v in r["metrics"]["regimes"].items():
        for p, d in v.items():
            cost.setdefault(k, {}).setdefault(p, []).append(np.array(d["per_episode_cost"]))
STY = {"forward": (COL["forward"], "forward"), "residual": (COL["residual"], "residual reuse"), "hindsight": (COL["posterior"], "hindsight"),
       "oracle": (COL["oracle"], "oracle (true $z$)"), "online_sgd": (COL["online_sgd"], "online SGD"),
       "hindsight_posterior": (COL["posterior"], "hindsight (posterior)"), "hindsight_latent_pf_adapt": (COL["particles"], "hindsight (particles)"),
       "oracle_latent": (COL["oracle"], "oracle (true $z$)")}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
for a, reg, title in ((ax[0], "both", "persistent damping + wind"), (ax[1], "fresh", "fresh draw every attempt (control)")):
    att = np.arange(1, miss[reg]["forward"].shape[0] + 1)
    for pl in ("forward", "residual", "hindsight", "oracle"):
        y = miss[reg][pl]; mu = y.mean(1); se = y.std(1) / np.sqrt(y.shape[1])
        c, lab = STY[pl]; a.plot(att, mu, "-o", ms=3, color=c, label=lab); a.fill_between(att, mu - 2 * se, mu + 2 * se, color=c, alpha=.15)
    a.set_xlabel("attempt"); a.set_ylabel("miss distance (m)"); a.set_title(title, fontsize=9); a.set_ylim(bottom=0)
ax[0].legend(fontsize=7, frameon=False)
ds = [0, 5, 10, 20]
for p in ("online_sgd", "hindsight_posterior", "hindsight_latent_pf_adapt", "oracle_latent"):
    y = [np.concatenate(cost[f"delay{d}/both"][p]).mean() / np.concatenate(cost[f"delay{d}/both"]["forward"]).mean() for d in ds]  # same statistic as tab_delay (ratio of means)
    c, lab = STY[p]; ax[2].plot(ds, y, "-o", ms=3, color=c, label=lab)
ax[2].set_xticks(ds); ax[2].axhline(1, color=COL["forward"], lw=.8, ls="--"); ax[2].set_xlabel("observation delay (steps)")
ax[2].set_ylabel("cost / forward"); ax[2].set_title("latency", fontsize=9); ax[2].legend(fontsize=7, frameon=False)
for a, l in zip(ax, "abc"):
    panel(a, l)
fig.tight_layout(); fig.savefig(G / "v6_figure.png", dpi=200); print("wrote", G / "v6_figure.png")
