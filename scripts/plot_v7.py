"""Figures for the hierarchical-vs-hindsight experiment (E-H), from results/ only."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.figstyle import NAME, PCOL, panel, setup  # noqa: E402
from wmi import hier  # noqa: E402
from wmi.harness import load_results  # noqa: E402

setup()
G = ROOT / "papers" / "latent-hindsight-planning" / "generated"
rng = np.random.default_rng(0)
R = sorted(load_results("hier_main"), key=lambda r: r["provenance"]["config"]["seed"])
hier.TASK["gap"] = R[0]["provenance"]["config"]["gap"]
RG = R[0]["metrics"]["regimes"]
cat = lambda key, p, f: np.concatenate([np.array(r["metrics"]["regimes"][key][p][f]) for r in R])
PL = list(RG["K10/none"].keys())


def ci(x, B=5000):
    i = rng.integers(0, len(x), (B, len(x))); m = x[i].mean(1)
    return x.mean(), np.quantile(m, .025), np.quantile(m, .975)


def draw_map(ax):
    for xw, yc in hier.WALLS:
        g = hier.TASK["gap"] / 2
        ax.add_patch(Rectangle((xw - hier.WALL_HALF, -hier.YLIM), 2 * hier.WALL_HALF, yc - g + hier.YLIM, color="#d9534f", alpha=.35, lw=0))
        ax.add_patch(Rectangle((xw - hier.WALL_HALF, yc + g), 2 * hier.WALL_HALF, hier.YLIM - yc - g, color="#d9534f", alpha=.35, lw=0))
    for y in (-hier.YLIM, hier.YLIM):
        ax.axhline(y, color="#d9534f", alpha=.5, lw=1.2)
    sg = hier.subgoals()
    ax.scatter(sg[:-1, 0], sg[:-1, 1], s=10, marker="x", color="#555555", zorder=3, lw=.8)
    ax.scatter(*hier.START, s=30, color="#2ca02c", zorder=4); ax.scatter(*hier.GOAL, s=60, marker="*", color="#2ca02c", zorder=4)
    ax.set_xlim(-0.3, 4.4); ax.set_ylim(-1.35, 1.35); ax.set_aspect("equal"); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])


# ---------- Figure 1 (teaser): trajectories + success bars
fig = plt.figure(figsize=(11, 3.1)); gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1.0], wspace=0.25)
r0 = R[0]["metrics"]["regimes"]["K10/both"]
for j, p in enumerate(("hier:forward", "hier:hindsight_posterior")):
    ax = fig.add_subplot(gs[j]); draw_map(ax)
    tr = np.array(r0[p]["traj"]); ok = np.array(r0[p]["success"])
    for e in range(tr.shape[1]):
        ax.plot(tr[:, e, 0], tr[:, e, 1], color=PCOL[p], alpha=.85, lw=1.1)
        last = tr[~np.isnan(tr[:, e, 0]), e][-1]
        ax.scatter(*last, marker="o" if ok[e] else "X", s=22 if ok[e] else 40, color=PCOL[p] if ok[e] else "#d9534f", zorder=5)
    ax.set_title(f"{NAME[p]}: {ok[:tr.shape[1]].sum()}/{tr.shape[1]} reach the goal", fontsize=8.5)
    panel(ax, "ab"[j])
ax = fig.add_subplot(gs[2]); show = ["flat:forward", "hier:forward", "hier:hindsight_posterior", "hier:oracle_latent"]
for i, p in enumerate(show):
    m, lo, hi = ci(cat("K10/both", p, "success"))
    ax.bar(i, m, color=PCOL[p], width=.7, yerr=[[m - lo], [hi - m]], capsize=2, ecolor="#333333", lw=0)
    ax.text(i, m + .04, f"{m:.2f}", ha="center", fontsize=7.5)
ax.set_xticks(range(len(show))); ax.set_xticklabels(["flat\nforward", "hier.\nforward", "hier.\n+ hindsight", "hier.\noracle $z$"], fontsize=7.5)
ax.set_ylim(0, 1.1); ax.set_ylabel("main-goal success"); ax.set_title("persistent gain + wind", fontsize=8.5); panel(ax, "c")
fig.savefig(G / "v7_teaser.png"); plt.close(fig)

# ---------- Figure 2: success by regime for every planner
regs = ["none", "wind", "both", "fresh"]
fig, ax = plt.subplots(figsize=(11, 2.9)); w = 0.8 / len(PL)
for j, p in enumerate(PL):
    for i, reg in enumerate(regs):
        m, lo, hi = ci(cat(f"K10/{reg}", p, "success"))
        ax.bar(i + (j - (len(PL) - 1) / 2) * w, m, width=w * .92, color=PCOL[p], yerr=[[m - lo], [hi - m]], capsize=1.2,
               ecolor="#333333", lw=0, label=NAME[p] if i == 0 else None, hatch="///" if p.startswith("flat") else None,
               edgecolor="white")
ax.set_xticks(range(len(regs))); ax.set_xticklabels(["no latent", "persistent wind", "persistent gain + wind", "fresh (redrawn every 1 s)"])
ax.set_ylabel("main-goal success"); ax.set_ylim(0, 1.08); ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.32))
fig.savefig(G / "v7_success.png"); plt.close(fig)

# ---------- Figure 3: where the main goal is lost (gate of first failure) + survival
fig, ax = plt.subplots(1, 2, figsize=(9, 2.8))
show = ["hier:forward", "hier:residual", "hier:online_sgd", "hier:hindsight_posterior", "hier:oracle_latent"]
for j, p in enumerate(show):
    fs = cat("K10/both", p, "fail_subgoal"); n = len(fs)
    gate = np.where(fs >= 0, np.minimum(fs // 2, 2), -1)
    frac = [np.mean(gate == g) for g in (0, 1, 2)]
    ax[0].bar(np.arange(3) + (j - 2) * .16, frac, width=.15, color=PCOL[p], label=NAME[p], lw=0)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["gate 1", "gate 2", "gate 3"]); ax[0].set_ylabel("episodes failing here")
ax[0].set_title("where the subgoal fails (persistent gain + wind)", fontsize=8.5); ax[0].legend(frameon=False, fontsize=6.5); panel(ax[0], "a")
T = R[0]["provenance"]["config"]["T"]; tt = np.arange(T + 1) * 0.05
for p in show:
    fl, td = cat("K10/both", p, "fail"), cat("K10/both", p, "t_done")
    alive = [(1 - (fl & (td < t / 0.05))).mean() for t in tt]
    ax[1].plot(tt, alive, color=PCOL[p], lw=1.4, label=NAME[p])
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("episodes without failure"); ax[1].set_ylim(0, 1.02)
ax[1].set_title("survival of the main goal", fontsize=8.5); panel(ax[1], "b")
fig.tight_layout(); fig.savefig(G / "v7_failures.png"); plt.close(fig)

# ---------- Figure 4: commitment sweep
KS = load_results("hier_ksweep")
if KS:
    fig, ax = plt.subplots(figsize=(4.2, 2.8)); Ks = [1, 5, 10, 20]
    for p in ("hier:forward", "hier:hindsight_posterior", "hier:oracle_latent"):
        ys = []
        for K in Ks:
            src = R if K == 10 else KS
            ys.append(np.concatenate([np.array(r["metrics"]["regimes"][f"K{K}/both"][p]["success"]) for r in src]).mean())
        ax.plot(Ks, ys, "-o", ms=3.5, color=PCOL[p], label=NAME[p], lw=1.4)
    ax.set_xscale("log", base=2); ax.set_xticks(Ks); ax.set_xticklabels(Ks); ax.set_xlabel("open-loop chunk length $K$ (steps)")
    ax.set_ylabel("main-goal success"); ax.set_ylim(0, 1.05); ax.legend(frameon=False)
    fig.savefig(G / "v7_ksweep.png"); plt.close(fig)
print("figures written")
