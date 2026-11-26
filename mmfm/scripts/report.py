"""Fill README.md <!-- AUTO:results --> from results/*.json (no hand-typed numbers)."""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
EV = Path("/home/user/data/mmfm_events")


def J(n):
    p = R / n
    return json.loads(p.read_text()) if p.exists() else None


def m1_table(d, title):
    if not d:
        return [f"*{title}: not run*", ""]
    out = [f"**{title}** — {d['n_events']:,} scored events, {d['n_tiles']} tiles (T={d['T']})", "",
           "| model | bits/event | 95% CI | Δ vs best baseline [95% CI] |", "|---|---|---|---|"]
    for k, v in d["models"].items():
        vs = v.get("vs_best_baseline")
        dl = f"{vs['delta_bits']:+.3f} [{vs['delta_ci95'][0]:+.3f}, {vs['delta_ci95'][1]:+.3f}] vs {vs['baseline'].split(':')[0]}" if vs else ""
        out.append(f"| {k} | {v['bits_per_event']:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {dl} |")
    return out + [""]


def main():
    L = []
    sp = json.loads((EV / "splits.json").read_text()) if (EV / "splits.json").exists() else None
    if sp:
        mins = {k: (sp[k][1] - sp[k][0]) / 6e7 for k in ("train", "val", "test")}
        L += [f"Frozen splits (wall clock): train {mins['train']:.0f} min · val {mins['val']:.0f} min · test {mins['test']:.0f} min · "
              f"embargo {sp['embargo_us'] / 6e7:.0f} min · held-out pair `{sp['heldout_pair']}` · files hash `{sp['files_hash'][:12]}`", ""]
    br = json.loads((EV / "build_report.json").read_text()) if (EV / "build_report.json").exists() else None
    if br:
        v = [x for p in br["pairs"].values() for s in p for x in s["verify"]]
        acc = sum(x["accept"] for x in v)
        n_ev = sum(s["n_events"] for p in br["pairs"].values() for s in p if not s["excluded"])
        L += [f"**M6 data fidelity**: {acc}/{len(v)} snapshot verifications accepted "
              f"(mean order-id Jaccard {np.mean([x['id_jaccard'] for x in v]):.4f}); {n_ev:,} model events in used segments.", ""]
    L += m1_table(J("m1_val.json"), "M1 val (selection split)")
    L += m1_table(J("m1_test.json"), "M1 TEST (sealed, single unseal)")
    L += m1_table(J("m1_heldout_test.json"), "M1 held-out pair, test period (zero-shot)")
    m2 = J("m2_test.json")
    if m2:
        L += ["**M2/M3 simulation realism, test** (start points = exact book checkpoints; lower CRPS/W1 better; PIT mean 0.5 and "
              "event-count ratio 1.0 ideal)", "", "| pair | generator | h (s) | n | PIT mean | in central 80% | CRPS bp | W1 bp | Brier P(up) | event ratio |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for pair, gs in m2["per_pair"].items():
            for g, s in gs.items():
                for h in ("1.0", "5.0", "10.0"):
                    x = s.get(h)
                    if not x:
                        continue
                    er = f"{x['event_count_ratio']:.2f}" if x.get("event_count_ratio") is not None else ""
                    br_ = f"{x['brier']:.3f}" if x.get("brier") is not None else ""
                    L.append(f"| {pair} | {g} | {h} | {x['n_starts']} | {x['pit_mean']:.2f} | {x['pit_in_central_80']:.2f} | {x['crps_bp']:.3f} | {x['w1_bp']:.3f} | {br_} | {er} |")
        L.append("")
    m5 = J("m5.json")
    if m5:
        L += [f"**M5 real-time (CPU, {m5['threads']} threads, {m5['n_params']:,} params)**: live update p50 {m5['update_latency_ms']['p50']:.2f} ms, "
              f"p99 {m5['update_latency_ms']['p99']:.2f} ms.", "", "| paths | ms/step | sim events/path/s | real-time factor vs median rate (per pair) |", "|---|---|---|---|"]
        for n, x in m5["rollout"].items():
            rf = ", ".join(f"{p} {v:.2f}×" for p, v in sorted(x["realtime_factor_vs_median_rate"].items()))
            L.append(f"| {n} | {x['ms_per_step']:.1f} | {x['sim_events_per_path_per_s']:.0f} | {rf} |")
        L.append("")
    acc = EV / "test_access.log"
    if acc.exists():
        L += ["Test access log (every unseal):", "", "```"] + acc.read_text().strip().splitlines() + ["```"]
    txt = (ROOT / "README.md").read_text()
    txt = re.sub(r"(<!-- AUTO:results -->\n).*?(<!-- /AUTO:results -->)", lambda m: m.group(1) + "\n".join(L) + "\n" + m.group(2), txt, flags=re.S)
    (ROOT / "README.md").write_text(txt)
    print("\n".join(L))


if __name__ == "__main__":
    main()
