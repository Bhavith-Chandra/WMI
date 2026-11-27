"""Fill README.md <!-- AUTO:v3 --> from results/v3_*.json; score every EVAL_v3 criterion (no hand-typed numbers).
evaluate_v2.py labels the two neural models 'MMFM-v2'/'B2-v2' whatever bundle is loaded; here they are the v3 bundle."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_v2 import J, criteria, sim_table, ROOT  # noqa: E402

REL = {"MMFM-v2": "MMFM-v3", "B2-v2": "B2-v3"}


def m1_block(d, title):
    if not d:
        return [f"*{title}: not run*", ""]
    L = [f"**{title}** — {d['n_events']:,} events, {d['n_blocks']} five-minute blocks", "", "| model | bits/event | 95% CI |", "|---|---|---|"]
    L += [f"| {REL.get(k, k)} | {v['bits_per_event']:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |" for k, v in d["models"].items()]
    dd = d["mmfm_minus_b2"]
    return L + ["", f"MMFM − B2: {dd['delta']:+.3f} bits/event, 95% CI [{dd['ci95'][0]:+.3f}, {dd['ci95'][1]:+.3f}]", ""]


def crit_table(C, title):
    if not C:
        return []
    L = [f"**{title}**", "", "| criterion | result | met |", "|---|---|---|"]
    return L + [f"| {c} | {r} | {'✓' if m else ('—' if m is None else '✗')} |" for c, r, m in C] + [""]


def freeclock(name):
    d = J(name)
    if not d:
        return []
    L = [f"**Free-clock diagnostic ({name}, val, no anchoring)** — event ratio median per horizon", "", "| pair | 1s | 5s | 10s |", "|---|---|---|---|"]
    for p, gs in d["per_pair"].items():
        s = gs["MMFM-sim"]
        L.append(f"| {p} | " + " | ".join(f"{s[h]['event_ratio_median']:.2f}" for h in ("1.0", "5.0", "10.0")) + " |")
    return L + [""]


def main():
    fz = J("../FROZEN_v3.json") if (ROOT / "FROZEN_v3.json").exists() else None
    L = []
    if fz:
        L += [f"Frozen bundle: `{fz['run']}` (selection: {fz['selection']['reason']}), freeze_ns {fz['freeze_ns']}, calibration {json.dumps(fz['calib'])}", ""]
    L += m1_block(J("v3_forward_m1.json"), "M1 next-event NLL — v3 FORWARD test")
    L += m1_block(J("v3_forward_m1_heldout.json"), "M1 zero-shot linkusd — v3 FORWARD test")
    L += sim_table(J("v3_forward_sim.json"), "Simulation (anchored clock + online count recal) — v3 FORWARD test")
    L += sim_table(J("v3_forward_sim_heldout.json"), "Zero-shot linkusd (recal) — v3 FORWARD test")
    C = criteria(J("v3_forward_sim.json"), J("v3_forward_m1.json"), J("v3_forward_m1_heldout.json"), J("v3_m5.json"))
    L += crit_table(C, "EVAL_v3 pre-registered criteria (forward test, recal ON)")
    L += crit_table(criteria(J("v3_forward_sim_norecal.json"), None, None, None)[:3], "(info) same window, recal OFF")
    L += crit_table(criteria(J("v3_forward_sim_v2bundle.json"), None, None, None)[:3], "(info) frozen v2 bundle on the v3 window")
    L += sim_table(J("v3_forward_sim_norecal.json"), "(info) v3 bundle, recal OFF — v3 FORWARD")
    L += sim_table(J("v3_forward_sim_v2bundle.json"), "(info) v2 bundle — v3 FORWARD")
    for r in ("v3", "v3_ss"):
        L += m1_block(J(f"v3_val_m1_{r}.json"), f"M1 val (selection) — runs/{r}")
        cal = ROOT / f"runs/{r}/sim_calibration.json"
        if cal.exists():
            c = json.loads(cal.read_text())
            L += [f"runs/{r} calibration (val, recal on): " + ", ".join(f"temp {g['temp']} → loss {g['loss']:.3f}" for g in c["grid"]) + f"; chosen temp {c['chosen']['temp']}", ""]
        L += freeclock(f"v3_val_freeclock_{r}.json")
    m5 = J("v3_m5.json")
    if m5:
        L += ["**Real time (CPU, 4 threads, v3 bundle)**", "", "| pair | direct heads ms | 8 paths (×median rate) | 16 paths (×median) |", "|---|---|---|---|"]
        for p, x in m5["per_pair"].items():
            L.append(f"| {p} | {x['direct_head_ms']:.1f} | {x['rollout']['8']['realtime_factor_median_rate']:.2f}× | {x['rollout']['16']['realtime_factor_median_rate']:.2f}× |")
        L.append("")
    acc = Path("/home/user/data/mmfm_fwd3/test_access.log")
    if acc.exists():
        L += ["v3 forward-test access log:", "", "```"] + acc.read_text().strip().splitlines() + ["```"]
    txt = (ROOT / "README.md").read_text()
    if "<!-- AUTO:v3 -->" not in txt:
        txt += "\n## v3 results (generated)\n<!-- AUTO:v3 -->\n<!-- /AUTO:v3 -->\n"
    txt = re.sub(r"(<!-- AUTO:v3 -->\n).*?(<!-- /AUTO:v3 -->)", lambda m: m.group(1) + "\n".join(L) + "\n" + m.group(2), txt, flags=re.S)
    (ROOT / "README.md").write_text(txt)
    print("\n".join(L))


if __name__ == "__main__":
    main()
