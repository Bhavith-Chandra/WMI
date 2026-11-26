"""Shared figure style for the hindsight-planning paper (clean, H-JEPA-like: light grid, no top/right spines,
one fixed colour per method across every figure)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"forward": "#8c8c8c", "residual": "#e08a2c", "online_sgd": "#c9a227", "posterior": "#1f5fbf",
       "particles": "#6c9be8", "oracle": "#1a1a1a", "flat_forward": "#c7c7c7", "flat_hindsight": "#a9c4ef"}
NAME = {"flat:forward": "flat, forward", "flat:hindsight_posterior": "flat, hindsight",
        "hier:forward": "hierarchical, forward", "hier:residual": "hierarchical + residual reuse",
        "hier:online_sgd": "hierarchical + online SGD", "hier:hindsight_posterior": "hierarchical + hindsight",
        "hier:hindsight_latent_pf_adapt": "hierarchical + hindsight (particles)", "hier:oracle_latent": "hierarchical, oracle $z$"}
PCOL = {"flat:forward": COL["flat_forward"], "flat:hindsight_posterior": COL["flat_hindsight"], "hier:forward": COL["forward"],
        "hier:residual": COL["residual"], "hier:online_sgd": COL["online_sgd"], "hier:hindsight_posterior": COL["posterior"],
        "hier:hindsight_latent_pf_adapt": COL["particles"], "hier:oracle_latent": COL["oracle"]}


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                         "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "axes.edgecolor": "#444444", "xtick.color": "#444444", "ytick.color": "#444444",
                         "savefig.dpi": 220, "savefig.bbox": "tight"})


def panel(ax, label):
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
