"""Data-leakage / lookahead audit.

Checks (a) what real data exists and that only it is read, (b) the simulator cannot see the future,
(c) known contamination in the *released checkpoints* is detected and disclosed (we cannot fix it).
"""
import json
import re
from pathlib import Path

import numpy as np
import torch

from m3.provenance import INPUTS, REL, ROOT, load_ckpt_meta



def test_release_contains_no_future_ground_truth():
    """The only real data is the 512-token prompt + its starting snapshot; no realised continuation ships."""
    meta = json.loads(INPUTS["example_metadata"].read_text())
    ids = np.load(INPUTS["example_prompt_ids"])
    assert ids.shape == (1, meta["prompt_len"]) == (1, 512)
    files = sorted(p.name for p in (REL / "examples").iterdir())
    assert files == ["README.md", "conditioning.npz", "example_metadata.json", "minimal_inference.py", "prompt_ids.npy"]
    c = np.load(INPUTS["example_conditioning"])
    assert set(c.files) == {"lob", "initial_lob", "initial_state_float"}
    assert np.array_equal(c["lob"], c["initial_lob"])


def test_conditioning_snapshot_is_at_prompt_start():
    """initial_state must describe the same book as the snapshot (no later information mixed in)."""
    c = np.load(INPUTS["example_conditioning"])
    lob, st = c["lob"][0], c["initial_state_float"][0]
    best_ask, best_bid = lob[0, 0, 0], lob[1, 0, 0]
    assert np.isclose(st[2], best_ask) and np.isclose(st[1], best_bid)
    assert np.isclose(st[0], 0.5 * (best_ask + best_bid), atol=1e-4)
    assert np.all(np.diff(lob[0, :, 0]) > 0) and np.all(np.diff(lob[1, :, 0]) < 0)


def test_scripts_read_only_allowed_inputs():
    """AST scan of our code: (1) no hard-coded data path except the allowed inputs / our own results,
    (2) every torch.load uses weights_only=True (no pickle code execution from downloaded files)."""
    import ast
    allowed_prefixes = (str(REL), str(ROOT))
    bad = []
    for f in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "m3").glob("*.py")):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) > 1 and node.value.startswith("/"):
                if not node.value.startswith(allowed_prefixes) and not node.value.startswith("/home/dataset"):
                    bad.append(f"{f.name}: hard-coded path {node.value!r}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load" \
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
                kw = {k.arg: k.value for k in node.keywords}
                if not (isinstance(kw.get("weights_only"), ast.Constant) and kw["weights_only"].value is True):
                    bad.append(f"{f.name}:{node.lineno}: torch.load without weights_only=True")
    assert not bad, bad


def test_released_checkpoints_selected_on_test_split_is_detected_and_disclosed():
    """Known contamination we cannot undo: best.pt chosen by loss on split='test'. Must be flagged in README."""
    flags = []
    for k in ("tokenizer", "tiny", "small", "base"):
        ta = load_ckpt_meta(INPUTS[k])["train_args"]
        if ta.get("val_split") == "test":
            flags.append(k)
    assert flags == ["tokenizer", "tiny", "small", "base"]
    readme = (ROOT / "README.md").read_text()
    assert 'val_split: "test"' in readme and "leakage" in readme.lower()


def test_example_window_is_from_selection_split():
    """The only real evaluation window comes from the split used for checkpoint selection -> disclosed."""
    meta = json.loads(INPUTS["example_metadata"].read_text())
    assert meta["split"] == "test" and meta["date"] == "20251201"
    assert "selection split" in (ROOT / "README.md").read_text()
