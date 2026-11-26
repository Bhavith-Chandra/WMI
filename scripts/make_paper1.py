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

# ---------------- v6: throw-and-retry (E-A) and observation latency (E-B)
TH = load_results("throw"); note("throw", TH)
miss = {reg: {pl: np.concatenate([np.array(r["metrics"]["miss"][reg][pl]) for r in TH], 1) for pl in TH[0]["metrics"]["miss"][reg]}
        for reg in TH[0]["metrics"]["miss"]}
summ = lambda reg, pl: miss[reg][pl][1:].mean(0)                       # per-episode mean over attempts 2..10
lines = ["\\begin{tabular}{lcccc}\n\\toprule\nRegime & hindsight & residual reuse & oracle & hindsight / oracle (attempt 10) \\\\\n\\midrule\n"]
for reg in ("wind", "damping", "both", "fresh"):
    cells = []
    for pl in ("hindsight", "residual", "oracle"):
        v, ci = paired_ratio(summ(reg, pl), summ(reg, "forward"), rng)
        cells.append(f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if pl == "hindsight" else f"{v:.3f}")
    cells.append(f"{miss[reg]['hindsight'][9].mean() / miss[reg]['oracle'][9].mean():.1f}$\\times$")
    lines.append(reg + " & " + " & ".join(cells) + " \\\\\n")
lines.append("\\bottomrule\n\\end{tabular}\n"); (G / "tab_throw.tex").write_text("".join(lines))
for reg, nm in (("wind", "Wind"), ("damping", "Damp"), ("both", "Both"), ("fresh", "Fresh")):
    m(f"twHind{nm}", paired_ratio(summ(reg, "hindsight"), summ(reg, "forward"), rng)[0])
    m(f"twResid{nm}", paired_ratio(summ(reg, "residual"), summ(reg, "forward"), rng)[0])
    m(f"twOracleGap{nm}", miss[reg]["hindsight"][9].mean() / miss[reg]["oracle"][9].mean(), "{:.1f}")
    m(f"twFirst{nm}", pct(miss[reg]["hindsight"][0].mean() / miss[reg]["forward"][0].mean() - 1))
    m(f"twFwdMiss{nm}", miss[reg]["forward"][1:].mean(), "{:.2f}"); m(f"twHindLast{nm}", miss[reg]["hindsight"][9].mean(), "{:.2f}")
m("twEpisodes", miss["wind"]["forward"].shape[1], "{:d}"); m("twAttempts", miss["wind"]["forward"].shape[0], "{:d}")
DL = load_results("mujoco_v3_delay"); note("mujoco_v3_delay", DL)
DL = sorted(DL, key=lambda r: r["provenance"]["config"]["seed"])
dk = sorted({k for r in DL for k in r["metrics"]["regimes"]})
PD = {k: {p: np.concatenate([np.array(r["metrics"]["regimes"][k][p]["per_episode_cost"]) for r in DL if k in r["metrics"]["regimes"]])
          for p in next(r for r in DL if k in r["metrics"]["regimes"])["metrics"]["regimes"][k]} for k in dk}
ds = [0, 5, 10, 20]
lines = ["\\begin{tabular}{lcccc}\n\\toprule\ndelay (steps) & " + " & ".join(map(str, ds)) + " \\\\\n\\midrule\n"]
for p, nme in (("hindsight_posterior", "hindsight (posterior)"), ("hindsight_latent_pf_adapt", "hindsight (particles)"),
               ("online_sgd", "online SGD"), ("oracle_latent", "oracle")):
    lines.append(nme + " & " + " & ".join(f"{ratio(PD, f'delay{d}/both', p)[0]:.3f}" for d in ds) + " \\\\\n")
lines.append("\\bottomrule\n\\end{tabular}\n"); (G / "tab_delay.tex").write_text("".join(lines))
for d, nm in ((0, "Zero"), (20, "Twenty")):
    m(f"dlPost{nm}", ratio(PD, f"delay{d}/both", "hindsight_posterior")[0]); m(f"dlSgd{nm}", ratio(PD, f"delay{d}/both", "online_sgd")[0])
    m(f"dlOracle{nm}", ratio(PD, f"delay{d}/both", "oracle_latent")[0])
    m(f"dlPostVsSgd{nm}", ratio(PD, f"delay{d}/both", "hindsight_posterior", "online_sgd")[0])
m("dlGainZero", pct(1 - ratio(PD, "delay0/both", "hindsight_posterior")[0])); m("dlGainTwenty", pct(1 - ratio(PD, "delay20/both", "hindsight_posterior")[0]))
m("dlCapture", pct((1 - ratio(PD, "delay20/both", "hindsight_posterior")[0]) / (1 - ratio(PD, "delay20/both", "oracle_latent")[0])))

# ---------------- v7: hierarchical planning with and without hindsight (E-H)
def succ_pool(exp):
    runs = sorted(load_results(exp), key=lambda r: r["provenance"]["config"]["seed"]); note(exp, runs)
    keys = runs[0]["metrics"]["regimes"]
    return {k: {p: np.concatenate([np.array(r["metrics"]["regimes"][k][p]["success"]) for r in runs]) for p in keys[k]} for k in keys}


def sdiff(a, b, B=10_000):
    i = rng.integers(0, len(a), (B, len(a))); d = a[i].mean(1) - b[i].mean(1)
    return float(a.mean() - b.mean()), (float(np.quantile(d, .025)), float(np.quantile(d, .975)))


H7 = succ_pool("hier_main")
HP = [("flat:forward", "flat, forward"), ("flat:hindsight_posterior", "flat, hindsight"), ("hier:forward", "hier., forward"),
      ("hier:residual", "hier. + residual"), ("hier:online_sgd", "hier. + online SGD"), ("hier:hindsight_posterior", "hier. + hindsight"),
      ("hier:hindsight_latent_pf_adapt", "hier. + hindsight (particles)"), ("hier:oracle_latent", "hier., oracle $z$")]
lines = ["\\begin{tabular}{l" + "c" * 4 + "}\n\\toprule\nPlanner & no latent & persistent wind & persistent gain + wind & fresh \\\\\n\\midrule\n"]
for p, nm in HP:
    cells = []
    for reg in ("none", "wind", "both", "fresh"):
        v = H7[f"K10/{reg}"][p].mean()
        if p.startswith("hier:") and p != "hier:forward":
            d, ci = sdiff(H7[f"K10/{reg}"][p], H7[f"K10/{reg}"]["hier:forward"])
            mark = "$^{+}$" if ci[0] > 0 else ("$^{-}$" if ci[1] < 0 else "")
            cells.append(f"{v:.2f}{mark}")
        else:
            cells.append(f"{v:.2f}")
    lines.append(nm + " & " + " & ".join(cells) + " \\\\\n")
lines.append("\\bottomrule\n\\end{tabular}\n"); (G / "tab_hier.tex").write_text("".join(lines))
w = lambda reg, p: H7[f"K10/{reg}"][p].mean()
for reg, nm in (("none", "None"), ("wind", "Wind"), ("both", "Both"), ("fresh", "Fresh")):
    m(f"hiFwd{nm}", w(reg, "hier:forward"), "{:.2f}"); m(f"hiHind{nm}", w(reg, "hier:hindsight_posterior"), "{:.2f}")
    m(f"hiOracle{nm}", w(reg, "hier:oracle_latent"), "{:.2f}"); m(f"hiFlat{nm}", w(reg, "flat:forward"), "{:.2f}")
    m(f"hiResid{nm}", w(reg, "hier:residual"), "{:.2f}"); m(f"hiSgd{nm}", w(reg, "hier:online_sgd"), "{:.2f}")
    m(f"hiPf{nm}", w(reg, "hier:hindsight_latent_pf_adapt"), "{:.2f}"); m(f"hiFlatHind{nm}", w(reg, "flat:hindsight_posterior"), "{:.2f}")
    d, ci = sdiff(H7[f"K10/{reg}"]["hier:hindsight_posterior"], H7[f"K10/{reg}"]["hier:forward"])
    m(f"hiDiff{nm}", f"{d:+.2f}"); m(f"hiDiffLo{nm}", f"{ci[0]:+.2f}"); m(f"hiDiffHi{nm}", f"{ci[1]:+.2f}")
m("hiCapture", pct((w("both", "hier:hindsight_posterior") - w("both", "hier:forward")) / (w("both", "hier:oracle_latent") - w("both", "hier:forward"))))
m("hiEpisodes", len(H7["K10/both"]["hier:forward"]), "{:d}")
if load_results("hier_ksweep"):
    KH = succ_pool("hier_ksweep")
    gK = lambda K: (H7 if K == 10 else KH)[f"K{K}/both"]
    for K, nm in ((1, "One"), (5, "Five"), (10, "Ten"), (20, "Twenty")):
        if f"K{K}/both" in (KH if K != 10 else H7):
            m(f"hkGain{nm}", gK(K)["hier:hindsight_posterior"].mean() - gK(K)["hier:forward"].mean(), "{:+.2f}")
            m(f"hkFwd{nm}", gK(K)["hier:forward"].mean(), "{:.2f}"); m(f"hkHind{nm}", gK(K)["hier:hindsight_posterior"].mean(), "{:.2f}")
LT = {r["provenance"]["config"]["threads"]: r for r in load_results("latency")}; note("latency", list(LT.values()))
for th, nm in ((1, "One"), (4, "Four")):
    for p, pn in (("forward", "Fwd"), ("hindsight_posterior", "Post"), ("hindsight_latent_pf_adapt", "Pf")):
        m(f"lat{pn}{nm}", LT[th]["metrics"]["planners"][p]["act_ms"]["p95"], "{:.0f}")
m("latObsPost", LT[1]["metrics"]["planners"]["hindsight_posterior"]["observe_ms"]["p95"], "{:.2f}")
m("latCPU", LT[1]["metrics"]["hardware"]["cpu"].replace("(R)", "").replace("(TM)", ""))

# ---------------- v9: goal-gap correction and relabeling (throw)
T9 = sorted(load_results("throw_v9"), key=lambda r: r["provenance"]["config"]["seed"]); note("throw_v9", T9)
T6 = sorted(TH, key=lambda r: r["provenance"]["config"]["seed"])
mm = {}
for reg in T6[0]["metrics"]["miss"]:
    mm[reg] = {p: np.concatenate([np.array(r["metrics"]["miss"][reg][p]) for r in T6], 1) for p in T6[0]["metrics"]["miss"][reg]}
    for p in T9[0]["metrics"]["miss"][reg]:
        if p != "forward":
            mm[reg][p] = np.concatenate([np.array(r["metrics"]["miss"][reg][p]) for r in T9], 1)
    assert np.array_equal(mm[reg]["forward"], np.concatenate([np.array(r["metrics"]["miss"][reg]["forward"]) for r in T9], 1)), "v9 replication"
s9 = lambda reg, p: mm[reg][p][1:].mean(0)
P9 = [("gap_ilc", "goal-gap correction"), ("residual", "residual reuse"), ("gap_mean", "mean gap"), ("anchor", "relabeling anchor"),
      ("hindsight", "latent inference"), ("hindsight_anchor", "latent inference + relabeling"), ("oracle", "oracle $z$")]
lines = ["\\begin{tabular}{lcccc}\n\\toprule\nPlanner & wind & damping & both & fresh \\\\\n\\midrule\n"]
for p, nm in P9:
    lines.append(nm + " & " + " & ".join(f"{paired_ratio(s9(r, p), s9(r, 'forward'), rng)[0]:.3f}" for r in ("wind", "damping", "both", "fresh")) + " \\\\\n")
lines.append("\\midrule\nfinal-attempt miss / oracle, latent inference & " + " & ".join(f"{mm[r]['hindsight'][9].mean() / mm[r]['oracle'][9].mean():.1f}" for r in ("wind", "damping", "both")) + " & -- \\\\\n")
lines.append("final-attempt miss / oracle, + relabeling & " + " & ".join(f"{mm[r]['hindsight_anchor'][9].mean() / mm[r]['oracle'][9].mean():.1f}" for r in ("wind", "damping", "both")) + " & -- \\\\\n")
lines.append("\\bottomrule\n\\end{tabular}\n"); (G / "tab_v9.tex").write_text("".join(lines))
for reg, nm in (("wind", "Wind"), ("damping", "Damp"), ("both", "Both"), ("fresh", "Fresh")):
    for p, pn in (("gap_ilc", "Gap"), ("anchor", "Anchor"), ("hindsight_anchor", "HA"), ("gap_mean", "GapMean")):
        m(f"vn{pn}{nm}", paired_ratio(s9(reg, p), s9(reg, "forward"), rng)[0])
    if reg != "fresh":
        m(f"vnHAOracle{nm}", mm[reg]["hindsight_anchor"][9].mean() / mm[reg]["oracle"][9].mean(), "{:.1f}")
        m(f"vnHAvsH{nm}", pct(1 - paired_ratio(s9(reg, "hindsight_anchor"), s9(reg, "hindsight"), rng)[0]))

# ---------------- v8: realistic deployment benchmark (sealed seeds)
D8 = succ_pool("realistic_v8")
for key, nm in (("deploy/both", "Both"), ("deploy/none", "None"), ("deploy/fresh", "Fresh")):
    for p, pn in (("hier:forward", "Fwd"), ("hier:residual", "Resid"), ("hier:online_sgd", "Sgd"), ("hier:hindsight_posterior", "Hind"),
                  ("hier:hindsight_latent_pf_adapt", "Pf"), ("hier:oracle_latent", "Oracle")):
        m(f"dp{pn}{nm}", D8[key][p].mean(), "{:.2f}")
    d, ci = sdiff(D8[key]["hier:hindsight_posterior"], D8[key]["hier:forward"])
    m(f"dpDiff{nm}", f"{d:+.2f}"); m(f"dpDiffLo{nm}", f"{ci[0]:+.2f}"); m(f"dpDiffHi{nm}", f"{ci[1]:+.2f}")
    d, ci = sdiff(D8[key]["hier:hindsight_latent_pf_adapt"], D8[key]["hier:forward"])
    m(f"dpPfDiff{nm}", f"{d:+.2f}"); m(f"dpPfDiffLo{nm}", f"{ci[0]:+.2f}")
for f, nm in (("mismatch", "Mismatch"), ("noise", "Noise"), ("delay", "Delay")):
    for p, pn in (("hier:forward", "Fwd"), ("hier:hindsight_posterior", "Hind"), ("hier:oracle_latent", "Oracle")):
        m(f"ab{pn}{nm}", D8[f"only_{f}/both"][p].mean(), "{:.2f}")
g7 = w("both", "hier:hindsight_posterior") - w("both", "hier:forward")
g8 = D8["deploy/both"]["hier:hindsight_posterior"].mean() - D8["deploy/both"]["hier:forward"].mean()
m("dpGainRatio", g8 / g7, "{:.2f}"); m("dpEpisodes", len(D8["deploy/both"]["hier:forward"]), "{:d}")
cs = sorted(load_results("realistic_v8"), key=lambda r: r["provenance"]["config"]["seed"])[0]["provenance"]["config"]["compute_steps"]
m("dpStepsHind", cs["hindsight_posterior"], "{:d}"); m("dpStepsFwd", cs["forward"], "{:d}")
lines = ["\\begin{tabular}{lcccc}\n\\toprule\nPlanner & idealised & deployment & deployment, no latent & deployment, fresh \\\\\n\\midrule\n"]
for p, nm in HP[2:]:
    lines.append(nm + " & " + f"{w('both', p):.2f} & " + " & ".join(f"{D8[k][p].mean():.2f}" for k in ("deploy/both", "deploy/none", "deploy/fresh")) + " \\\\\n")
lines.append("\\bottomrule\n\\end{tabular}\n"); (G / "tab_v8.tex").write_text("".join(lines))
lines = ["\\begin{tabular}{lccc}\n\\toprule\nPlanner & decision p50 (ms) & decision p95 (ms) & belief update p95 (ms) \\\\\n\\midrule\n"]
for p, nm in (("forward", "forward"), ("residual", "residual reuse"), ("online_sgd", "online SGD"), ("hindsight_posterior", "hindsight (posterior)"),
              ("hindsight_latent_pf_adapt", "hindsight (particles)"), ("oracle_latent", "oracle")):
    L = LT[1]["metrics"]["planners"][p]
    lines.append(f"{nm} & {L['act_ms']['p50']:.0f} & {L['act_ms']['p95']:.0f} & {L['observe_ms']['p95']:.2f} \\\\\n")
lines.append("\\bottomrule\n\\end{tabular}\n"); (G / "tab_latency.tex").write_text("".join(lines))
m("dpCapture", pct((D8["deploy/both"]["hier:hindsight_posterior"].mean() - D8["deploy/both"]["hier:forward"].mean()) / (D8["deploy/both"]["hier:oracle_latent"].mean() - D8["deploy/both"]["hier:forward"].mean())))

# ---------------- v10/v10b: subgoal redrawing (gate failed; ceiling of route choice from gate episodes)
FG = json.loads((ROOT / "results" / "fork_gate_v10b.json").read_text())
ceil = lambda d: d["A"] + d["B_only"] - max(d["A"], d["B"])
m("fkCeilFwdMax", max(ceil(r["forward"]) for r in FG["ladder"]), "{:+.2f}")
m("fkCeilFwdMin", min(ceil(r["forward"]) for r in FG["ladder"]), "{:+.2f}")
m("fkCeilOracleMax", max(ceil(r["oracle_latent"]) for r in FG["ladder"]), "{:+.2f}")
ll_gain = [max(r["oracle_latent"]["A"], r["oracle_latent"]["B"]) - max(r["forward"]["A"], r["forward"]["B"]) for r in FG["ladder"]]
m("fkLowGainMin", min(ll_gain), "{:+.2f}"); m("fkLowGainMax", max(ll_gain), "{:+.2f}"); m("fkRungs", len(FG["ladder"]), "{:d}")

body = "% generated by scripts/make_paper1.py -- DO NOT EDIT\n"
body += "".join(f"% provenance {k}: commits {v[0]}, dirty={v[1]}, seeds/runs={v[2]}\n" for k, v in prov.items())
body += "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items()))
(G / "numbers.tex").write_text(body)
print(f"wrote {len(macros)} macros; dirty in any source: {any(v[1] for v in prov.values())}")
