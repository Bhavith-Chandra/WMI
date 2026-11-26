"""Generate the result-dependent README blocks straight from results/*.json (no hand-typed numbers).

python -I scripts/build_readme_numbers.py
Replaces text between <!-- AUTO:<name> --> and <!-- /AUTO:<name> --> in README.md and writes
results/readme_claims.json (every generated line), which tests/test_claims.py checks against README.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from m3.provenance import stamp  # noqa: E402
R = ROOT / "results"
S3 = ("tiny", "small", "base")


def J(n):
    return json.loads((R / n).read_text())


def f(x, nd=1, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "−")


def block_decode():
    d = J("decode_ablation.json")
    A, B = "A_release_gen_only", "B_context_aware"
    lines = [
        "| | release decode (gen only) | in-context decode |",
        "|---|---|---|",
        f"| mean decoded rel-open price, base | {f(d['base'][A]['rel_open_mean'], 5)} | {f(d['base'][B]['rel_open_mean'], 5)} |",
        f"| cancels hitting an empty level, base | {f(100 * d['base'][A]['cancel_fully_unmatched_frac']['median'])}% | "
        f"{f(100 * d['base'][B]['cancel_fully_unmatched_frac']['median'])}% |",
        "| median return over 512 generated events, from the post-prompt mid, tiny / small / base (bp) | "
        + " / ".join(f(d[s][A]["end_ret_bp"]["median"]) for s in S3) + " | "
        + " / ".join(f(d[s][B]["end_ret_bp"]["median"]) for s in S3) + " |",
    ]
    share = [1 - d[s][B]["end_ret_bp"]["median"] / d[s][A]["end_ret_bp"]["median"] for s in S3]
    lines.append("")
    lines.append("Share of the release-decode median drift that disappears with in-context decoding: "
                 + " / ".join(f"{100 * x:.0f}%" for x in share) + " (tiny / small / base).")
    return lines


def block_rollout():
    r = J("rollout.json")
    lines = ["| | tiny | small | base |", "|---|---|---|---|"]
    lines.append("| simulated time covered by 1024 generated events, median (s) | "
                 + " | ".join(f(r[s]["agg"]["elapsed_s"]["median"], 0) for s in S3) + " |")
    lines.append("| next-token entropy, positions ≤ 1033 (trained range), nats | "
                 + " | ".join(f(r[s]["entropy_in_trained_range"], 2) for s in S3) + " |")
    lines.append("| next-token entropy, positions > 1033, nats | "
                 + " | ".join(f(r[s]["entropy_beyond_trained"], 2) for s in S3) + " |")
    lines.append("| median return over 1024 events from the post-prompt mid (bp) | "
                 + " | ".join(f(r[s]["agg"]["end_ret_bp"]["median"]) for s in S3) + " |")
    e = r["base"]["entropy_by_64"]
    p0 = r["base"]["entropy_bins_abs_start_pos"]
    lines.append("")
    lines.append("Base, entropy per 64-token bin (bin start position: nats): "
                 + ", ".join(f"{p}: {f(x, 2)}" for p, x in zip(p0, e)) + ".")
    pr = r["prompt_replay"]
    lines.append("")
    lines.append(f"Real prompt replayed from its 10-level snapshot: {f(100 * pr['cancel_unmatched_qty_frac'])}% of cancel "
                 f"quantity finds no resting depth; {f(100 * pr['cancel_unmatched_outside_known_frac'])}% of cancels target "
                 f"prices never seen in the book; prompt drift {f(pr['end_ret_bp'], 1, True)} bp over "
                 f"{f(pr['elapsed_s'])} s.")
    return lines


def block_twap():
    lines = []
    runs = [("tiny_h60", "tiny, 3 sizes × 6 reps"), ("small_h60", "small, 3 sizes × 6 reps"),
            ("small_h60_q5r12", "small, 5 sizes × 12 reps")]
    lines.append("| run | baseline drift, mean ± SE (bp), n | stalled (base / buy / sell) | Fisher p buy vs sell | "
                 "buy β [90% CI] | sell β [90% CI] |")
    lines.append("|---|---|---|---|---|---|")
    for key, label in runs:
        p = R / f"twap_{key}_analysis.json"
        if not p.exists():
            lines.append(f"| {label} | not run | | | | |")
            continue
        a = json.loads(p.read_text())
        m = a["main_excluding_stalled"]
        st = a["stall"]["counts"]

        def beta(side):
            x = m[side]
            return f"{f(x['beta'], 2)} [{f(x['beta_ci90'][0], 2)}, {f(x['beta_ci90'][1], 2)}]"

        lines.append(f"| {label} | {f(m['baseline_mean_bp'])} ± {f(m['baseline_se_bp'])}, n={m['n_base']} | "
                     f"{st['base']['stalled']}/{st['base']['n']} / {st['buy']['stalled']}/{st['buy']['n']} / "
                     f"{st['sell']['stalled']}/{st['sell']['n']} | {a['stall']['fisher_p_buy_vs_sell']:.2f} | "
                     f"{beta('buy')} | {beta('sell')} |")
    lines.append("")
    lines.append("Mean impact per size (bp, ± SE incl. baseline uncertainty), stalled paths excluded:")
    lines.append("")
    for key, label in runs:
        p = R / f"twap_{key}_analysis.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())["main_excluding_stalled"]
        for side in ("buy", "sell"):
            cells = m[side]["cells"]
            lines.append(f"* {label}, {side}: " + "; ".join(
                f"{c['Q'] // 1000}k → {f(c['impact_bp_mean'], 0, True)} ± {f(c['impact_bp_se_incl_baseline'], 0)}"
                for c in cells))
    lines.append("")
    lines.append("Sensitivity (stalled paths included at their last reached mid), β buy / sell: ")
    lines.append("")
    for key, label in runs:
        p = R / f"twap_{key}_analysis.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())["sensitivity_including_stalled"]
        lines.append(f"* {label}: {f(s['buy']['beta'], 2)} [{f(s['buy']['beta_ci90'][0], 2)}, {f(s['buy']['beta_ci90'][1], 2)}]"
                     f" / {f(s['sell']['beta'], 2)} [{f(s['sell']['beta_ci90'][0], 2)}, {f(s['sell']['beta_ci90'][1], 2)}]")
    lines.append("")
    for key, label in runs:
        p = R / f"twap_{key}_analysis.json"
        if not p.exists():
            continue
        a = json.loads(p.read_text())
        ch = a["children"]
        lines.append(f"* {label}: {a['stall']['dt_le_1ms_frac_median_reached'] * 100:.0f}% of events ≤ 1 ms in reached paths"
                     + (f" vs {a['stall']['dt_le_1ms_frac_median_stalled'] * 100:.0f}% in stalled paths"
                        if a["stall"]["dt_le_1ms_frac_median_stalled"] is not None else "")
                     + f"; children skipped (empty opposite side) {ch['skipped']}; executed/requested "
                     f"{100 * ch['exec_frac_of_requested']:.0f}%; injected-order round-trip price error median "
                     f"{ch['roundtrip_tick_err_median']:.0f} ticks; stale terminal mids {a['stale_terminal_mid']}.")
    return lines


BLOCKS = {"decode": block_decode, "rollout": block_rollout, "twap": block_twap}


def main():
    txt = (ROOT / "README.md").read_text()
    lits = []
    for name, fn in BLOCKS.items():
        lines = fn()
        body = "\n".join(lines)
        pat = re.compile(rf"(<!-- AUTO:{name} -->\n).*?(<!-- /AUTO:{name} -->)", re.S)
        if not pat.search(txt):
            raise SystemExit(f"README is missing markers for block {name!r}")
        txt = pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), txt)
        lits += [ln for ln in lines if ln.strip() and not ln.startswith("|---")]
    (ROOT / "README.md").write_text(txt)
    (R / "readme_claims.json").write_text(json.dumps({"literals": lits, "_provenance": stamp([])}, indent=1))
    print(f"README updated; {len(lits)} generated lines")


if __name__ == "__main__":
    main()
