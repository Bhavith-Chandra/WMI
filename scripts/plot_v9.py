"""Figure for v9 (goal-gap correction and relabeling), from results/ only."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.figstyle import COL, panel, setup  # noqa: E402
from wmi.harness import load_results  # noqa: E402

setup()
G = Path(__file__).resolve().parents[1] / "papers" / "latent-hindsight-planning" / "generated"
srt = lambda R: sorted(R, key=lambda r: r["provenance"]["config"]["seed"])
V6, V9 = srt(load_results("throw")), srt(load_results("throw_v9"))
get = lambda R, reg, p: np.concatenate([np.array(r["metrics"]["miss"][reg][p]) for r in R], 1)
STY = [("forward", V6, COL["forward"], "forward", "-"), ("gap_ilc", V9, COL["residual"], "goal-gap correction", "--"),
       ("anchor", V9, "#8e5cc8", "relabeling anchor", "-"), ("hindsight", V6, COL["posterior"], "latent inference (hindsight)", "-"),
       ("hindsight_anchor", V9, "#0b2f6b", "latent inference + relabeling", "-"), ("oracle", V6, COL["oracle"], "oracle $z$", ":")]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
for a, reg, title in zip(ax, ("wind", "damping", "both"), ("persistent wind (additive)", "persistent damping (multiplicative)", "both")):
    for p, R, c, lab, ls in STY:
        y = get(R, reg, p); mu = y.mean(1); se = y.std(1) / np.sqrt(y.shape[1]); att = np.arange(1, len(mu) + 1)
        a.plot(att, mu, ls, marker="o", ms=2.5, color=c, label=lab, lw=1.5); a.fill_between(att, mu - 2 * se, mu + 2 * se, color=c, alpha=.12)
    a.set_yscale("log"); a.set_xlabel("attempt"); a.set_title(title, fontsize=9); a.set_xticks(range(1, 11))
ax[0].set_ylabel("miss distance (m, log scale)"); ax[2].legend(frameon=False, fontsize=7)
for a, l in zip(ax, "abc"):
    panel(a, l)
fig.tight_layout(); fig.savefig(G / "v9_figure.png"); print("wrote v9_figure.png")
