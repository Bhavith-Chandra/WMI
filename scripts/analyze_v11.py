"""Decision rules of v11_preregistration.md (written before results)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wmi.harness import load_results  # noqa: E402

rng = np.random.default_rng(0)
R = sorted(load_results("ramp_main"), key=lambda r: r["provenance"]["config"]["seed"])
get = lambda reg, k, f="score": np.concatenate([np.array(r["metrics"]["regimes"][reg][k][f]) for r in R])


def diff(a, b, B=10_000):
    i = rng.integers(0, len(a), (B, len(a))); d = a[i].mean(1) - b[i].mean(1)
    return float(a.mean() - b.mean()), [float(np.quantile(d, .025)), float(np.quantile(d, .975))]


out, chk = [f"seeds {[r['provenance']['config']['seed'] for r in R]}, task {R[0]['provenance']['config']['task']}"], {}
H = "hindsight_posterior"
for reg in ("none", "wind", "both", "fresh"):
    ks = list(R[0]["metrics"]["regimes"][reg].keys())
    out.append(reg + " score " + str({k: round(float(get(reg, k).mean()), 3) for k in ks}))
    out.append("   success " + str({k: round(float(get(reg, k, "success").mean()), 3) for k in ks}))
    orc = get(reg, f"{H}|oracle", "route")
    out.append("   route B share / agreement with oracle HL: " + str({k: (round(float(np.mean(get(reg, k, 'route') == 'B')), 2),
                                                                     round(float(np.mean(get(reg, k, 'route') == orc)), 2)) for k in ks}))
    out.append(f"   {H}|post - {H}|prior = {diff(get(reg, f'{H}|post'), get(reg, f'{H}|prior'))}")
b = lambda k: get("both", k)
chk["S1 both: hindsight|post > hindsight|prior"] = diff(b(f"{H}|post"), b(f"{H}|prior"))[1][0] > 0
chk["S2 both: forward|post > forward|prior"] = diff(b("forward|post"), b("forward|prior"))[1][0] > 0
best = max((f"{H}|A", f"{H}|B"), key=lambda k: b(k).mean())
d3 = diff(b(f"{H}|post"), b(best)); out.append(f"S3 best fixed = {best}: {d3}")
chk["S3 both: hindsight|post > best fixed route"] = d3[1][0] > 0
den = b(f"{H}|oracle").mean() - b(f"{H}|prior").mean()
cap = (b(f"{H}|post").mean() - b(f"{H}|prior").mean()) / den if den > 0 else float("nan"); out.append(f"S4 capture = {cap:.3f}")
chk["S4 both: capture >= 0.5"] = den > 0 and cap >= 0.5
chk["S5 none: no harm (lower CI > -0.05)"] = diff(get("none", f"{H}|post"), get("none", f"{H}|prior"))[1][0] > -0.05
chk["S6 fresh: not significantly better"] = diff(get("fresh", f"{H}|post"), get("fresh", f"{H}|prior"))[1][0] <= 0
out += [("SUPPORTED " if v else "NOT SUPPORTED ") + k for k, v in chk.items()]
print("\n".join(out))
