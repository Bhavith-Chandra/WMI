"""Decision rules of v9_preregistration.md (written before results)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_mujoco import paired_ratio  # noqa: E402
from wmi.harness import load_results  # noqa: E402

rng = np.random.default_rng(0)
srt = lambda R: sorted(R, key=lambda r: r["provenance"]["config"]["seed"])
V6, V9 = srt(load_results("throw")), srt(load_results("throw_v9"))
assert [r["provenance"]["config"]["seed"] for r in V6] == [r["provenance"]["config"]["seed"] for r in V9]
miss = {}
for reg in V6[0]["metrics"]["miss"]:
    miss[reg] = {p: np.concatenate([np.array(r["metrics"]["miss"][reg][p]) for r in V6], 1) for p in V6[0]["metrics"]["miss"][reg]}
    for p in V9[0]["metrics"]["miss"][reg]:
        miss[reg]["v9_" + p if p == "forward" else p] = np.concatenate([np.array(r["metrics"]["miss"][reg][p]) for r in V9], 1)
out, chk = [], {}
chk["R0 replication: v9 forward == v6 forward"] = all(np.array_equal(miss[r]["forward"], miss[r]["v9_forward"]) for r in miss)
S = lambda reg, p: miss[reg][p][1:].mean(0)
rel = lambda reg, p, q="forward": paired_ratio(S(reg, p), S(reg, q), rng)
for reg in ("wind", "damping", "both", "fresh"):
    out.append(reg + " " + str({p: (round(rel(reg, p)[0], 3), [round(c, 3) for c in rel(reg, p)[1]]) for p in miss[reg] if p != "v9_forward"}))
    out.append("   attempt-10 / oracle: " + str({p: round(float(miss[reg][p][9].mean() / miss[reg]["oracle"][9].mean()), 2)
                                                for p in ("hindsight", "hindsight_anchor", "anchor", "residual", "gap_ilc")}))
chk["G1 gap_ilc: wind < 0.5; loses to hindsight on damping/both; worse than forward on fresh"] = (
    rel("wind", "gap_ilc")[1][1] < 0.5 and all(rel(r, "hindsight", "gap_ilc")[1][1] < 1 for r in ("damping", "both"))
    and rel("fresh", "gap_ilc")[1][0] > 1)
chk["G2 anchor beats residual on damping and both"] = all(rel(r, "anchor", "residual")[1][1] < 1 for r in ("damping", "both"))
chk["G3a hindsight_anchor beats hindsight on wind, damping, both"] = all(
    rel(r, "hindsight_anchor", "hindsight")[1][1] < 1 for r in ("wind", "damping", "both"))
chk["G3b attempt-10 hindsight_anchor <= 2 x oracle in >= 2 of 3 regimes"] = sum(
    miss[r]["hindsight_anchor"][9].mean() <= 2 * miss[r]["oracle"][9].mean() for r in ("wind", "damping", "both")) >= 2
chk["G4 fresh: anchor and hindsight_anchor not better than forward"] = all(rel("fresh", p)[1][0] >= 1 for p in ("anchor", "hindsight_anchor"))
out += [("SUPPORTED " if v else "NOT SUPPORTED ") + k for k, v in chk.items()]
print("\n".join(out))
