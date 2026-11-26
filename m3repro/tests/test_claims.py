"""Every headline number in README.md must be re-derivable from results/*.json (no hand-typed numbers).

Each claim = (README literal, json file, extractor, rounding). The test asserts BOTH that the literal
appears in the README and that the JSON value rounds to it.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
README = (ROOT / "README.md").read_text()


def J(name):
    return json.loads((R / name).read_text())


def scal(size, key):
    rows = J("scaling.json")
    rows = rows["rows"] if isinstance(rows, dict) else rows
    return next(r for r in rows if r["size"] == size)[key]


def twap_cell(model, side, q, key="impact_bp_median"):
    return next(c for c in J(f"twap_{model}_h60.json")["cells"] if c["side"] == side and c["Q"] == q)[key]


CLAIMS = [
    # scaling table
    ("6.228", lambda: scal("tiny", "ckpt_val_loss"), 3), ("6.388", lambda: scal("tiny", "eq2_pred"), 3),
    ("+2.6%", lambda: scal("tiny", "rel_err_pct"), 1), ("6.005", lambda: scal("small", "ckpt_val_loss"), 3),
    ("6.183", lambda: scal("small", "eq2_pred"), 3), ("+3.0%", lambda: scal("small", "rel_err_pct"), 1),
    ("5.886", lambda: scal("base", "ckpt_val_loss"), 3), ("6.112", lambda: scal("base", "eq2_pred"), 3),
    ("+3.8%", lambda: scal("base", "rel_err_pct"), 1), ("14.94", lambda: scal("base", "D_B"), 2),
    # NLL
    ("7.22 / 6.88 / 6.73", lambda: [J("nll.json")[s]["true"]["mean"] for s in ("tiny", "small", "base")], 2),
    # tokenizer
    ("11.6 ticks", lambda: J("tokenizer.json")["context_sensitivity"]["tick_mae"], 1),
    ("47.7%", lambda: 100 * J("tokenizer.json")["reencode_code_agreement"], 1),
    ("14.8%", lambda: 100 * J("tokenizer.json")["negative_dt_frac"], 1),
    # decode ablation
    ("−0.00152", lambda: -J("decode_ablation.json")["base"]["A_release_gen_only"]["rel_open_mean"], 5),
    ("61.7%", lambda: 100 * J("decode_ablation.json")["base"]["A_release_gen_only"]["cancel_fully_unmatched_frac"]["median"], 1),
    ("26.0%", lambda: 100 * J("decode_ablation.json")["base"]["B_context_aware"]["cancel_fully_unmatched_frac"]["median"], 1),
    # rollouts
    ("37 s (tiny), 91 s (small) and 149 s (base)",
     lambda: [J("rollout.json")[s]["agg"]["elapsed_s"]["median"] for s in ("tiny", "small", "base")], 0),
    ("42.5%", lambda: 100 * J("rollout.json")["prompt_replay"]["cancel_unmatched_outside_known_frac"], 1),
    ("68%", lambda: 100 * J("rollout.json")["prompt_replay"]["cancel_unmatched_qty_frac"], 0),
    # TWAP
    ("**−105 bp**", lambda: J("twap_tiny_h60.json")["baseline"]["mean_bp"], 0),
    ("−33 bp", lambda: J("twap_small_h60.json")["baseline"]["mean_bp"], 0),
    ("**7/42**", lambda: J("twap_tiny_h60.json")["stall"]["n_stalled"], 0),
    ("**0.57**", lambda: J("twap_small_h60.json")["fit_buy"]["beta"], 2),
    ("−0.24", lambda: J("twap_small_h60.json")["fit_sell"]["beta"], 2),
    ("35 / 70 / 153", lambda: [twap_cell("small", "buy", q) for q in (10000, 40000, 160000)], 0),
]


def _fmt(v, nd):
    return f"{v:.{nd}f}" if nd else f"{round(v):d}"


@pytest.mark.parametrize("literal,fn,nd", CLAIMS, ids=[c[0] for c in CLAIMS])
def test_readme_claim_matches_results(literal, fn, nd):
    assert literal in README, f"claim text {literal!r} not found in README"
    v = fn()
    vals = v if isinstance(v, list) else [v]
    for x in vals:
        s = _fmt(abs(x), nd)
        assert s in literal.replace("−", "-").replace("+", ""), f"README says {literal!r}, results give {x}"
