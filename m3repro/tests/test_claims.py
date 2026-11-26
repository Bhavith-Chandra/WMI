"""Every headline number in README.md must be re-derivable from results/*.json (no hand-typed numbers).

A claim = (README literal, extractor returning one value or a list, decimals). The test asserts:
  1. the literal appears verbatim in README.md
  2. the numbers inside the literal, parsed IN ORDER WITH SIGN, equal the extractor values rounded
     to `decimals` (exact string equality after formatting; no substring matching)
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
README = (ROOT / "README.md").read_text()
NUM = re.compile(r"[−+-]?\d+(?:\.\d+)?")


def J(name):
    return json.loads((R / name).read_text())


def scal(size, key):
    return next(r for r in J("scaling.json")["rows"] if r["size"] == size)[key]


def an(name):
    return J(f"twap_{name}_analysis.json")["main_excluding_stalled"]


def cell(name, side, q, key="impact_bp_mean"):
    return next(c for c in an(name)[side]["cells"] if c["Q"] == q)[key]


def nllpos(size, lo, hi):
    v = J("nll.json")[size]["per_position_nll_true"][lo:hi]
    return sum(v) / len(v)


S3 = ("tiny", "small", "base")
CLAIMS = [
    # scaling (README §2 table)
    ("| tiny  | 9.66  | 31.46 | 6.228 | 6.388 | +2.6% |",
     lambda: [scal("tiny", "N_M"), scal("tiny", "D_B"), scal("tiny", "ckpt_val_loss"), scal("tiny", "eq2_pred"), scal("tiny", "rel_err_pct")], [2, 2, 3, 3, 1]),
    ("| small | 24.99 | 31.46 | 6.005 | 6.183 | +3.0% |",
     lambda: [scal("small", "N_M"), scal("small", "D_B"), scal("small", "ckpt_val_loss"), scal("small", "eq2_pred"), scal("small", "rel_err_pct")], [2, 2, 3, 3, 1]),
    ("| base  | 74.75 | 14.94 | 5.886 | 6.112 | +3.8% |",
     lambda: [scal("base", "N_M"), scal("base", "D_B"), scal("base", "ckpt_val_loss"), scal("base", "eq2_pred"), scal("base", "rel_err_pct")], [2, 2, 3, 3, 1]),
    # NLL (§4)
    ("Mean NLL over 512 tokens is 7.22 / 6.88 / 6.73", lambda: [512] + [J("nll.json")[s]["true"]["mean"] for s in S3], [0, 2, 2, 2]),
    ("NLL over the last 128 positions is 6.33 / 6.08 / 5.91", lambda: [128] + [nllpos(s, 384, 512) for s in S3], [0, 2, 2, 2]),
    # LOB perturbations (§5)
    ("| flip bid/ask depth (imbalance) | +0.003 (−0.016) | +0.005 (+0.014) | +0.004 (+0.006) |",
     lambda: [x for s in S3 for x in (J("nll.json")[s]["swap_depth"]["delta_vs_true"], J("nll.json")[s]["swap_depth"]["delta_first64"])], 3),
    ("| volume ×10 | +0.015 (+0.082) | +0.040 (+0.174) | +0.067 (+0.229) |",
     lambda: [10] + [x for s in S3 for x in (J("nll.json")[s]["vol_x10"]["delta_vs_true"], J("nll.json")[s]["vol_x10"]["delta_first64"])], [0] + [3] * 6),
    ("| prices +1% | +0.011 (+0.069) | +0.016 (+0.110) | +0.025 (+0.137) |",
     lambda: [1] + [x for s in S3 for x in (J("nll.json")[s]["shift+1"]["delta_vs_true"], J("nll.json")[s]["shift+1"]["delta_first64"])], [0] + [3] * 6),
    ("| prices −1% | +0.029 (+0.179) | +0.043 (+0.196) | +0.038 (+0.186) |",
     lambda: [-1] + [x for s in S3 for x in (J("nll.json")[s]["shift-1"]["delta_vs_true"], J("nll.json")[s]["shift-1"]["delta_first64"])], [0] + [3] * 6),
    # tokenizer (§6)
    ("mean of\n  **11.6 ticks**", lambda: J("tokenizer.json")["context_sensitivity"]["tick_mae"], 1),
    ("only 0.4% land on the same tick", lambda: 100 * J("tokenizer.json")["context_sensitivity"]["tick_exact_same"], 1),
    ("same code only 47.7% of the time", lambda: 100 * J("tokenizer.json")["reencode_code_agreement"], 1),
    ("14.8% of decoded inter-arrival times are negative", lambda: 100 * J("tokenizer.json")["negative_dt_frac"], 1),
]


def _parse(lit):
    return [x.replace("−", "-") for x in NUM.findall(lit)]


def _fmt(v, nd):
    s = f"{v:+.{nd}f}" if nd else f"{round(v):+d}"
    return s


@pytest.mark.parametrize("literal,fn,nd", CLAIMS, ids=[c[0][:40] for c in CLAIMS])
def test_readme_claim_matches_results(literal, fn, nd):
    assert literal in README, f"claim text not found in README: {literal!r}"
    vals = fn()
    vals = vals if isinstance(vals, list) else [vals]
    nds = nd if isinstance(nd, list) else [nd] * len(vals)
    got = _parse(literal)
    assert len(got) == len(vals), f"{literal!r}: {len(got)} numbers in text vs {len(vals)} values"
    for g, v, d in zip(got, vals, nds):
        exp = _fmt(v, d)
        g2 = g if g[0] in "+-" else "+" + g
        assert g2 == exp, f"{literal!r}: README has {g}, results give {exp} (raw {v})"


def test_claims_registry_is_extended_by_generated_claims():
    """Claims that depend on regenerated runs live in results/readme_claims.json, built by
    scripts/build_readme_numbers.py from the result files; every entry must appear verbatim in README."""
    p = R / "readme_claims.json"
    assert p.exists(), "run scripts/build_readme_numbers.py"
    for lit in json.loads(p.read_text())["literals"]:
        assert lit in README, f"generated claim missing from README: {lit!r}"
