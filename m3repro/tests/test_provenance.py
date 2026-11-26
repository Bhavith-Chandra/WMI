"""Every result file must say what produced it, and its recorded input hashes must match the files on disk."""
import json
from pathlib import Path

import pytest

from m3.provenance import INPUTS, sha256

R = Path(__file__).resolve().parents[1] / "results"
RESULTS = sorted(p for p in R.glob("*.json") if not p.name.startswith("."))


@pytest.mark.parametrize("path", RESULTS, ids=[p.name for p in RESULTS])
def test_result_has_valid_provenance(path):
    d = json.loads(path.read_text())
    prov = d.get("_provenance")
    assert prov is not None, f"{path.name}: no provenance (regenerate with current scripts)"
    assert prov["git_commit"], "no git commit recorded"
    for k, v in prov["inputs"].items():
        assert v["sha256"] == sha256(INPUTS[k]), f"{path.name}: input {k} changed since result was produced"
