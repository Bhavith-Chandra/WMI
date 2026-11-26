"""Decision rules of v7_preregistration.md (written before results)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmi.harness import load_results  # noqa: E402

rng = np.random.default_rng(0)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])




def pool(exp):
    R = sorted(load_results(exp), key=lambda r: r["provenance"]["config"]["seed"])
    keys = R[0]["metrics"]["regimes"]
    return {k: {p: np.concatenate([np.array(r["metrics"]["regimes"][k][p]["success"]) for r in R]) for p in keys[k]} for k in keys}, R


def diff(a, b, B=10_000):
    n = len(a); i = rng.integers(0, n, (B, n)); d = a[i].mean(1) - b[i].mean(1)
    return float(a.mean() - b.mean()), [float(np.quantile(d, .025)), float(np.quantile(d, .975))]


M, R = pool("hier_main"); out, chk = [f"main: seeds {[r['provenance']['config']['seed'] for r in R]}, "
                                        f"commits {sorted({r['provenance']['git_commit'][:7] for r in R})}"], {}
for k in M:
    out.append(k + " " + str({p: round(float(v.mean()), 3) for p, v in M[k].items()}))
S = lambda reg: M[f"K10/{reg}"]
H, F = "hier:hindsight_posterior", "hier:forward"
d = {}
for reg in ("none", "wind", "both", "fresh"):
    for p in M["K10/none"]:
        d[(reg, p)] = diff(S(reg)[p], S(reg)[F])
        out.append(f"{reg} {p} - hier:forward = {d[(reg, p)][0]:+.3f} {np.round(d[(reg, p)][1], 3).tolist()}")
chk["H1 none: hier:forward > flat:forward"] = diff(S("none")[F], S("none")["flat:forward"])[1][0] > 0
chk["H2 wind/both: posterior and pf_adapt > hier:forward"] = all(
    d[(r, p)][1][0] > 0 for r in ("wind", "both") for p in (H, "hier:hindsight_latent_pf_adapt"))
chk["H3 none/fresh: posterior not significantly > hier:forward"] = all(d[(r, H)][1][0] <= 0 for r in ("none", "fresh"))
chk["H4 wind/both: hier hindsight > flat hindsight"] = all(
    diff(S(r)[H], S(r)["flat:hindsight_posterior"])[1][0] > 0 for r in ("wind", "both"))
chk["H5 both: posterior > residual and > online_sgd"] = all(
    diff(S("both")[H], S("both")[q])[1][0] > 0 for q in ("hier:residual", "hier:online_sgd"))
o, h, f = (S("both")[p].mean() for p in ("hier:oracle_latent", H, F))
out.append(f"H6 capture = {(h - f) / (o - f) if o > f else float('nan'):.3f}")
chk["H6 both: capture >= 0.5"] = o > f and (h - f) / (o - f) >= 0.5
if load_results("hier_ksweep"):
    KS, _ = pool("hier_ksweep")
    gain = {10: h - f}
    for K in (1, 5, 20):
        gain[K] = KS[f"K{K}/both"][H].mean() - KS[f"K{K}/both"][F].mean()
    Ks = [1, 5, 10, 20]; out.append("K-sweep gain " + str({K: round(float(gain[K]), 3) for K in Ks}))
    chk["H7 gain(K=20) > gain(K=1) and Spearman > 0"] = gain[20] > gain[1] and spearman(Ks, [gain[K] for K in Ks]) > 0
out += [("SUPPORTED " if v else "NOT SUPPORTED ") + k for k, v in chk.items()]
print("\n".join(out))
