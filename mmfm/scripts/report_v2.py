"""Fill README.md <!-- AUTO:v2 --> from results/v2_*.json and score every EVAL_v2 criterion (no hand-typed numbers)."""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def J(n):
    p = R / n
    return json.loads(p.read_text()) if p.exists() else None


def sim_table(d, title):
    if not d:
        return [f"*{title}: not run*", ""]
    L = [f"**{title}** (n starts per cell shown; targets: event ratio 0.8–1.25, coverage 0.70–0.90, PIT 0.40–0.60)", "",
         "| pair | generator | h | n | event ratio | coverage80 | PIT mean | CRPS bp |", "|---|---|---|---|---|---|---|---|"]
    for pair, gs in d["per_pair"].items():
        for g, s in gs.items():
            for h in ("1.0", "5.0", "10.0"):
                x = s[h]
                L.append(f"| {pair} | {g} | {h}s | {x['n']} | {x.get('event_ratio_median', float('nan')):.2f} | {x['coverage80']:.2f} | "
                         f"{x['pit_mean']:.2f} | {x['crps_bp']:.3f} |")
    return L + [""]


def criteria(sim, m1, m1h, m5):
    C = []
    if sim:
        cells = [(p, h, s["MMFM-sim"][h]) for p, s in sim["per_pair"].items() for h in ("1.0", "5.0", "10.0")]
        ev = [0.8 <= c[2]["event_ratio_median"] <= 1.25 for c in cells]
        cov = [0.70 <= c[2]["coverage80"] <= 0.90 for c in cells]
        pit = [0.40 <= c[2]["pit_mean"] <= 0.60 for c in cells]
        C.append(("1 event ratio in [0.8,1.25] (MMFM-sim)", f"{sum(ev)}/{len(ev)} cells", sum(ev) == len(ev)))
        C.append(("2 coverage80 in [0.70,0.90] (MMFM-sim)", f"{sum(cov)}/{len(cov)} cells", sum(cov) == len(cov)))
        C.append(("3 PIT mean in [0.40,0.60] (MMFM-sim)", f"{sum(pit)}/{len(pit)} cells", sum(pit) == len(pit)))
        wins = 0
        tot = 0
        for p, s in sim["per_pair"].items():
            for h in ("1.0", "5.0", "10.0"):
                tot += 1
                best_b = min(s[g][h]["crps_bp"] for g in ("B1p", "B0p") if g in s)
                wins += s["MMFM-sim"][h]["crps_bp"] <= best_b
        C.append(("4 MMFM-sim CRPS <= best count baseline", f"{wins}/{tot} cells", wins > tot / 2))
        if all("MMFM-direct" in s for s in sim["per_pair"].values()):
            dc = [(s["MMFM-direct"][h]) for s in sim["per_pair"].values() for h in ("1.0", "5.0", "10.0")]
            C.append(("(info) MMFM-direct coverage80 in [0.70,0.90]", f"{sum(0.7 <= x['coverage80'] <= 0.9 for x in dc)}/{len(dc)} cells", None))
    if m1:
        d = m1["mmfm_minus_b2"]
        C.append(("5 NLL: MMFM-v2 beats tuned B2 (CI excludes 0)", f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] bits/event", d["ci95"][1] < 0))
    if m1h:
        d = m1h["mmfm_minus_b2"]
        C.append(("6 zero-shot linkusd NLL gap vs B2 <= 0", f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]", d["delta"] <= 0))
    if m5:
        x = m5["per_pair"]["ethusd"]["rollout"]["8"]["realtime_factor_median_rate"]
        C.append(("7 real time: 8 paths on ethusd >= 1x", f"{x:.2f}x", x >= 1))
    return C


def main():
    L = []
    for name, title in (("v2_forward_m1.json", "M1 next-event NLL — FORWARD test"), ("v2_forward_m1_heldout.json", "M1 zero-shot linkusd — FORWARD test"),
                        ("v2_val_m1.json", "M1 next-event NLL — v2 val (selection)")):
        d = J(name)
        if d:
            L += [f"**{title}** — {d['n_events']:,} events, {d['n_blocks']} five-minute blocks", "", "| model | bits/event | 95% CI |", "|---|---|---|"]
            L += [f"| {k} | {v['bits_per_event']:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |" for k, v in d["models"].items()]
            dd = d["mmfm_minus_b2"]
            L += ["", f"MMFM-v2 − B2-v2: {dd['delta']:+.3f} bits/event, 95% CI [{dd['ci95'][0]:+.3f}, {dd['ci95'][1]:+.3f}]", ""]
    L += sim_table(J("v2_forward_sim.json"), "Simulation & forecast calibration — FORWARD test")
    L += sim_table(J("v2_forward_sim_heldout.json"), "Zero-shot linkusd — FORWARD test")
    L += sim_table(J("v2_val_sim.json"), "Simulation & forecast calibration — v2 val (selection split, not a test)")
    m5 = J("v2_m5.json")
    if m5:
        L += ["**Real time (CPU, 4 threads)**", "", "| pair | direct heads ms | 8 paths: sim events/path/s (×median rate) | 16 paths (×median) |", "|---|---|---|---|"]
        for p, x in m5["per_pair"].items():
            L.append(f"| {p} | {x['direct_head_ms']:.1f} | {x['rollout']['8']['sim_events_per_path_per_s']:.0f} ({x['rollout']['8']['realtime_factor_median_rate']:.2f}×) | "
                     f"{x['rollout']['16']['sim_events_per_path_per_s']:.0f} ({x['rollout']['16']['realtime_factor_median_rate']:.2f}×) |")
        L.append("")
    C = criteria(J("v2_forward_sim.json"), J("v2_forward_m1.json"), J("v2_forward_m1_heldout.json"), m5)
    if C:
        L += ["**EVAL_v2 pre-registered criteria (forward test)**", "", "| criterion | result | met |", "|---|---|---|"]
        L += [f"| {c} | {r} | {'✓' if m else ('—' if m is None else '✗')} |" for c, r, m in C]
        L.append("")
    acc = Path("/home/user/data/mmfm_fwd/test_access.log")
    if acc.exists():
        L += ["Forward-test access log:", "", "```"] + acc.read_text().strip().splitlines() + ["```"]
    txt = (ROOT / "README.md").read_text()
    if "<!-- AUTO:v2 -->" not in txt:
        txt += "\n## v2 results (generated)\n<!-- AUTO:v2 -->\n<!-- /AUTO:v2 -->\n"
    txt = re.sub(r"(<!-- AUTO:v2 -->\n).*?(<!-- /AUTO:v2 -->)", lambda m: m.group(1) + "\n".join(L) + "\n" + m.group(2), txt, flags=re.S)
    (ROOT / "README.md").write_text(txt)
    print("\n".join(L))


if __name__ == "__main__":
    main()
