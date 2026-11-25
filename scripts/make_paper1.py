"""Generate every number and table used in papers/latent-hindsight-planning/main.tex from results/ ONLY.
Output: generated/numbers.tex (\\newcommand macros) and generated/tab_*.tex. Re-run after any new result."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_mujoco import paired_ratio  # noqa: E402
from wmi.harness import load_results  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "papers" / "latent-hindsight-planning" / "generated"
rng = np.random.default_rng(0)
macros, prov = {}, {}


def m(name, val, fmt="{:.3f}"):
    assert name.isalpha(), name
    macros[name] = fmt.format(val) if not isinstance(val, str) else val


def pct(x):
    return f"{100 * x:.1f}\\%"


def note(exp, runs):
    prov[exp] = (sorted({r["provenance"]["git_commit"][:7] for r in runs}), any(r["provenance"]["git_dirty"] for r in runs), len(runs))


def pooled(exp):
    runs = load_results(exp); note(exp, runs); keys = runs[0]["metrics"]["regimes"]
    return {k: {p: np.concatenate([np.array(r["metrics"]["regimes"][k][p]["per_episode_cost"]) for r in runs]) for p in keys[k]} for k in keys}


def ratio(P, k, p, q="forward"):
    return paired_ratio(P[k][p], P[k][q], rng)


# ---------------- theory
T = load_results("theory_hindsight"); note("theory_hindsight", T)
mean = lambda reg, p, key="cum_regret_T": float(np.mean([r["metrics"][reg][p][key] for r in T]))
m("thForward", mean("persistent", "forward"), "{:.0f}"); m("thHindsight", mean("persistent", "hindsight_bayes"), "{:.1f}")
m("thResidual", mean("persistent", "residual"), "{:.0f}"); m("thTexcess", mean("persistent", "hindsight_bayes", "t_times_excess_q4"), "{:.2f}")
cfg = T[0]["provenance"]["config"]; m("thSigmaSqD", 2 * cfg["sigma"] ** 2, "{:.2f}"); m("thT", cfg["T"], "{:d}")
m("thResidAdd", mean("persistent_additive", "residual") / cfg["T"], "{:.3f}"); m("thSigmaSq", cfg["sigma"] ** 2, "{:.3f}")
m("thFreshFwd", mean("fresh", "forward"), "{:.0f}"); m("thFreshHind", mean("fresh", "hindsight_bayes"), "{:.0f}")
m("thSwitchHind", mean("switching", "hindsight_bayes"), "{:.0f}"); m("thSwitchForget", mean("switching", "hindsight_forget"), "{:.0f}")
m("thSwitchFwd", mean("switching", "forward"), "{:.0f}")

# ---------------- real measured systems (EMPS excluded: model-free velocity extrapolation beats every model)
R = load_results("realdata_hindsight"); note("realdata_hindsight", R)
gm = lambda x: float(np.exp(np.mean(np.log(x))))
rows, r1, rk, s1, res_k = [], [], [], [], []
for ds in R[0]["metrics"]:
    if ds == "EMPS":
        continue
    for j, rec in enumerate(R[0]["metrics"][ds]["records"]):
        g = lambda key, mm: np.mean([r["metrics"][ds]["records"][j][key][mm] for r in R])
        f1, fk = g("rmse1", "forward"), g("rmse10", "forward")
        r1.append(g("rmse1", "hindsight_forget") / f1); rk.append(g("rmse10", "hindsight_forget") / fk)
        s1.append(g("rmse1", "online_sgd") / f1); res_k.append(g("rmse10", "residual") / fk)
m("rdOneStep", gm(r1)); m("rdTenStep", gm(rk)); m("rdSgdOne", gm(s1)); m("rdResidTen", gm(res_k))
m("rdRecords", len(r1), "{:d}")
S = load_results("sealed_v3_V31_hybrid_sgd_static_forget"); note("sealed_v3_hybrid", S)
cf, cs, sf, nb, nw, n = [], [], [], 0, 0, 0
for ds in S[0]["metrics"]:
    if ds == "EMPS":
        continue
    for j in range(len(S[0]["metrics"][ds]["records"])):
        g = lambda key, mm: np.mean([r["metrics"][ds]["records"][j][key][mm] for r in S])
        cf.append(g("rmse10", "candidate") / g("rmse10", "forward")); cs.append(g("rmse10", "candidate") / g("rmse10", "sgd"))
        sf.append(g("rmse10", "sgd") / g("rmse10", "forward"))
        for r in S:
            ci = r["metrics"][ds]["records"][j]["ci95_mse10_cand_minus"]; n += 1
            nb += ci["forward"][1] < 0; nw += ci["sgd"][0] > 0
m("rdHybridFwd", gm(cf)); m("rdHybridSgd", gm(cs)); m("rdSgdFwdTen", gm(sf))
m("rdHybridSigFwd", f"{nb}/{n}"); m("rdHybridWorseSgd", f"{nw}/{n}")

# ---------------- MuJoCo v3 main
M = pooled("mujoco_v3_main")
PL = [("hindsight_posterior", "posterior"), ("hindsight_latent", "latent (GD)"), ("hindsight_latent_pf", "latent (particles)"),
      ("hybrid", "hybrid"), ("online_sgd", "online SGD"), ("residual", "residual reuse"), ("oracle_latent", "oracle")]
REG = ["wind", "both", "switching", "ood_gain", "gain", "none", "fresh"]
lines = ["\\begin{tabular}{l" + "c" * len(PL) + "}\n\\toprule\nRegime & " + " & ".join(n for _, n in PL) + " \\\\\n\\midrule\n"]
for reg in REG:
    cells = []
    for p, _ in PL:
        v, ci = ratio(M, f"K20/{reg}", p)
        mark = "$^{\\downarrow}$" if ci[1] < 1 else ("$^{\\uparrow}$" if ci[0] > 1 else "")
        cells.append(f"{v:.3f}{mark}")
    lines.append(reg.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\\n")
lines.append("\\bottomrule\n\\end{tabular}\n"); (G / "tab_mujoco_main.tex").write_text("".join(lines))
for reg, nm in (("wind", "Wind"), ("both", "Both")):
    m(f"mjPost{nm}", ratio(M, f"K20/{reg}", "hindsight_posterior")[0]); m(f"mjSgd{nm}", ratio(M, f"K20/{reg}", "online_sgd")[0])
    m(f"mjOracle{nm}", ratio(M, f"K20/{reg}", "oracle_latent")[0]); m(f"mjLatent{nm}", ratio(M, f"K20/{reg}", "hindsight_latent")[0])
    m(f"mjPostVsSgd{nm}", ratio(M, f"K20/{reg}", "hindsight_posterior", "online_sgd")[0])
    cap = (1 - ratio(M, f"K20/{reg}", "hindsight_latent_pf")[0]) / (1 - ratio(M, f"K20/{reg}", "oracle_latent")[0])
    m(f"mjCapture{nm}", pct(cap))
m("mjFreshResid", ratio(M, "K20/fresh", "residual")[0]); m("mjFreshPost", ratio(M, "K20/fresh", "hindsight_posterior")[0])
m("mjGainLatent", ratio(M, "K20/gain", "hindsight_latent")[0]); m("mjGainResid", ratio(M, "K20/gain", "residual")[0])

# ---------------- sweeps
K = pooled("mujoco_v3_ksweep"); Ks = [1, 5, 10, 20, 40]
lines = ["\\begin{tabular}{lccccc}\n\\toprule\n$K$ & " + " & ".join(map(str, Ks)) + " \\\\\n\\midrule\n"]
for reg in ("wind", "both"):
    lines.append(f"hindsight gain ({reg}) & " + " & ".join(pct(1 - ratio(K, f'K{k}/{reg}', 'hindsight_latent_pf')[0]) for k in Ks) + " \\\\\n")
    lines.append(f"oracle headroom ({reg}) & " + " & ".join(pct(1 - ratio(K, f'K{k}/{reg}', 'oracle_latent')[0]) for k in Ks) + " \\\\\n")
lines.append("\\bottomrule\n\\end{tabular}\n"); (G / "tab_ksweep.tex").write_text("".join(lines))
m("ksGainOne", pct(1 - ratio(K, "K1/wind", "hindsight_latent_pf")[0])); m("ksGainForty", pct(1 - ratio(K, "K40/wind", "hindsight_latent_pf")[0]))
m("ksHeadForty", pct(1 - ratio(K, "K40/wind", "oracle_latent")[0]))
RH = pooled("mujoco_v3_rho"); rhos = [0.0, 0.9, 0.99, 1.0]
lines = ["\\begin{tabular}{lcccc}\n\\toprule\n$\\rho$ & " + " & ".join(map(str, rhos)) + " \\\\\n\\midrule\n"]
for p, nme in (("hindsight_latent_pf", "latent (particles)"), ("hindsight_posterior", "posterior"), ("residual", "residual reuse"), ("oracle_latent", "oracle")):
    lines.append(nme + " & " + " & ".join(f"{ratio(RH, f'K20/ar{r}', p)[0]:.3f}" for r in rhos) + " \\\\\n")
lines.append("\\bottomrule\n\\end{tabular}\n"); (G / "tab_rho.tex").write_text("".join(lines))
m("rhoZero", ratio(RH, "K20/ar0.0", "hindsight_latent_pf")[0]); m("rhoOne", ratio(RH, "K20/ar1.0", "hindsight_latent_pf")[0])
N = pooled("mujoco_v3_noise2"); gs = [0.0, 0.25, 0.5, 1.0]
lines = ["\\begin{tabular}{lcccc}\n\\toprule\ngust sd (N) & " + " & ".join(map(str, gs)) + " \\\\\n\\midrule\n"]
for p, nme in (("hindsight_posterior", "posterior"), ("hindsight_latent_pf_adapt", "latent, adaptive likelihood"),
               ("hindsight_latent_pf", "latent, fixed likelihood"), ("residual", "residual reuse"), ("oracle_latent", "oracle")):
    lines.append(nme + " & " + " & ".join(f"{ratio(N, f'gust{g}/wind', p)[0]:.3f}" for g in gs) + " \\\\\n")
lines.append("\\bottomrule\n\\end{tabular}\n"); (G / "tab_noise.tex").write_text("".join(lines))
for p, nm in (("residual", "Resid"), ("hindsight_posterior", "Post"), ("hindsight_latent_pf_adapt", "Adapt"), ("hindsight_latent_pf", "Fixed")):
    m(f"nz{nm}Zero", ratio(N, "gust0.0/wind", p)[0]); m(f"nz{nm}One", ratio(N, "gust1.0/wind", p)[0])

# ---------------- identification
I = load_results("identification_v4"); note("identification_v4", I)
A = lambda reg, c: np.array([r["metrics"][reg]["r2_by_t"][c] for r in I]).mean(0)
m("idWindTwenty", float(np.mean([A("both", "wind_x")[19], A("both", "wind_y")[19]])))
m("idPredTwenty", float(A("both", "predictive_dv")[19])); m("idPredEnd", float(A("both", "predictive_dv")[-1]))
bnd = [20, 40, 60, 80, 100, 120, 140]
m("idPredDip", float(np.mean([A("both", "predictive_dv")[b] - A("both", "predictive_dv")[b - 5:b].mean() for b in bnd])))
C = load_results("diag_confound"); note("diag_confound", C)
m("cfDipStd", C[0]["metrics"]["standard"]["mean_dip"]); m("cfDipPin", C[0]["metrics"]["pinned"]["mean_dip"])
m("cfRPin", float(np.mean(C[0]["metrics"]["final_r2_pinned"])))

# ---------------- river (hierarchical)
def river_rates(exp):
    runs = load_results(exp); note(exp, runs); out = {}
    for reg in runs[0]["metrics"]["regimes"]:
        out[reg] = {k: float(np.mean(np.concatenate([np.array(r["metrics"]["regimes"][reg][k]["success"]) for r in runs])))
                    for k in runs[0]["metrics"]["regimes"][reg]}
    return out
V1, V2 = river_rates("river"), river_rates("river_v2")
m("rvOneHind", V1["strong"]["hier_hindsight"]); m("rvOneFwd", V1["strong"]["hier_forward"]); m("rvOneFlat", V1["strong"]["flat_forward"])
m("rvTwoHind", V2["strong"]["hier_replan_hindsight"]); m("rvTwoFwd", V2["strong"]["hier_replan_forward"]); m("rvTwoOracle", V2["strong"]["hier_replan_oracle"])
cal3 = json.loads((ROOT / "results" / "river_v3_calibration.json").read_text())["rows"][0]
m("rvThreeFwd", cal3["strong"]["forward"], "{:.2f}"); m("rvThreeOracle", cal3["strong"]["oracle"], "{:.2f}")

body = "% generated by scripts/make_paper1.py -- DO NOT EDIT\n"
body += "".join(f"% provenance {k}: commits {v[0]}, dirty={v[1]}, seeds/runs={v[2]}\n" for k, v in prov.items())
body += "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items()))
(G / "numbers.tex").write_text(body)
print(f"wrote {len(macros)} macros; dirty in any source: {any(v[1] for v in prov.values())}")
