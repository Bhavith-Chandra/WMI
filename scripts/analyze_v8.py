"""Decision rules of v8_preregistration.md (written before results)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmi.harness import load_results  # noqa: E402

rng = np.random.default_rng(0)


def pool(exp):
    R = sorted(load_results(exp), key=lambda r: r["provenance"]["config"]["seed"])
    keys = R[0]["metrics"]["regimes"]
    return {k: {p: np.concatenate([np.array(r["metrics"]["regimes"][k][p]["success"]) for r in R]) for p in keys[k]} for k in keys}, R


def diff(a, b, B=10_000):
    i = rng.integers(0, len(a), (B, len(a))); d = a[i].mean(1) - b[i].mean(1)
    return float(a.mean() - b.mean()), [float(np.quantile(d, .025)), float(np.quantile(d, .975))]


V, R = pool("realistic_v8"); out, chk = [f"v8 sealed seeds {[r['provenance']['config']['seed'] for r in R]}, "
                                         f"compute_steps {R[0]['provenance']['config']['compute_steps']}"], {}
H, F, PF = "hier:hindsight_posterior", "hier:forward", "hier:hindsight_latent_pf_adapt"
for k in V:
    out.append(k + " " + str({p: round(float(v.mean()), 3) for p, v in V[k].items()}))
    for p in V[k]:
        if p != F:
            dd = diff(V[k][p], V[k][F]); out.append(f"   {p} - forward = {dd[0]:+.3f} {np.round(dd[1], 3).tolist()}")
chk["R1 deploy/both: posterior > forward"] = diff(V["deploy/both"][H], V["deploy/both"][F])[1][0] > 0
chk["R2 deploy/both: particles (with compute delay) > forward"] = diff(V["deploy/both"][PF], V["deploy/both"][F])[1][0] > 0
chk["R3 deploy/none: posterior - forward lower CI > -0.05"] = diff(V["deploy/none"][H], V["deploy/none"][F])[1][0] > -0.05
chk["R4 deploy/fresh: posterior not significantly > forward"] = diff(V["deploy/fresh"][H], V["deploy/fresh"][F])[1][0] <= 0
L = [r for r in load_results("latency") if r["provenance"]["config"]["threads"] == 1][0]["metrics"]["planners"]
out.append("latency (1 thread) " + str({k: (round(v["act_ms"]["p95"], 1), round(v["observe_ms"]["p95"], 2)) for k, v in L.items()}))
chk["R5 posterior p95 act <= 500 ms and observe <= 50 ms"] = L["hindsight_posterior"]["act_ms"]["p95"] <= 500 and \
    L["hindsight_posterior"]["observe_ms"]["p95"] <= 50
M7, _ = pool("hier_main")
g7 = M7["K10/both"][H].mean() - M7["K10/both"][F].mean(); g8 = V["deploy/both"][H].mean() - V["deploy/both"][F].mean()
out.append(f"gain v7 {g7:.3f}, gain v8 {g8:.3f}")
chk["R6 v8 gain >= 0.5 x v7 gain"] = g7 > 0 and g8 >= 0.5 * g7
out += [("SUPPORTED " if v else "NOT SUPPORTED ") + k for k, v in chk.items()]
print("\n".join(out))
