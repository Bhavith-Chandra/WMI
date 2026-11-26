"""MMFM leakage harness (EVAL.md L1-L7) on synthetic data (fast) + static checks on the code."""
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.dataset import SealError, load, raw_chunks  # noqa: E402
from data.tokenizer import Tokenizer, state_features  # noqa: E402
from model.mmfm import MMFM  # noqa: E402

EVD = [("us", "i8"), ("type", "i1"), ("side", "i1"), ("price", "f8"), ("qty", "f8")]


def synth(n=3000, seed=0, t0=0):
    rng = np.random.default_rng(seed)
    ev = np.zeros(n, EVD)
    ev["us"] = t0 + np.cumsum(rng.integers(0, 5000, n))
    ev["type"] = rng.choice(3, n, p=[0.5, 0.48, 0.02])
    ev["side"] = rng.integers(0, 2, n)
    mid = 100 + np.cumsum(rng.normal(0, 0.01, n))
    ev["price"] = mid + rng.normal(0, 0.05, n)
    ev["qty"] = rng.lognormal(0, 1, n)
    st = np.stack([mid - 0.01, mid + 0.01, rng.uniform(1, 5, n), rng.uniform(1, 5, n), rng.uniform(5, 20, n), rng.uniform(5, 20, n)], 1)
    return ev, st


def test_state_features_have_no_lookahead():
    """L6/L7: feats[i] (input when predicting event i+1) must not change if event i+1.. are perturbed."""
    ev, st = synth()
    f0 = state_features(ev, st)
    j = 1500
    ev2, st2 = ev.copy(), st.copy()
    ev2["us"][j:] += 7_000_000  # event j arrives 7 s later: its dt (a target) changes
    ev2["qty"][j:] *= 3
    st2[j + 1:] *= 1.01  # book after event j changes
    f1 = state_features(ev2, st2)
    assert np.allclose(f0[: j - 1], f1[: j - 1]), "features before the perturbation changed"
    assert np.allclose(f0[j - 1], f1[j - 1]), "feats[j-1] (used to predict event j) depends on event j"
    assert not np.allclose(f0[j:], f1[j:]), "vacuous"


def test_model_is_causal():
    ev, st = synth(600)
    tok = Tokenizer.fit([(ev, st)], dict(split="train", files_hash="x", range=[0, 1]))
    t, f, _ = tok.encode(ev, st)
    m = MMFM(tok.vocab, f.shape[1], d=32, n_layers=2, ctx=128).eval()
    T = torch.from_numpy(t[:128])[None]
    F = torch.from_numpy(f[:128])[None]
    a = m.logp(T, F)
    T2, F2 = T.clone(), F.clone()
    T2[0, 90:, 2] = (T2[0, 90:, 2] + 5) % tok.vocab["px"]
    F2[0, 90:] += 1.0
    b = m.logp(T2, F2)
    # logp[t] scores event t+1 from inputs <= t: positions < 89 see only unperturbed inputs
    assert torch.allclose(a[0, :89], b[0, :89], atol=1e-5)
    assert not torch.allclose(a[0, 89:], b[0, 89:])


@pytest.fixture()
def events_dir(tmp_path):
    files = []
    for k, pair in enumerate(("btcusd", "ethusd", "linkusd")):
        ev, st = synth(4000, seed=k, t0=0)
        d = tmp_path / pair
        d.mkdir()
        np.savez(d / "seg_1.npz", **{c: ev[c] for c in ev.dtype.names}, state=st, tick=0.01)
        files.append([f"{pair}/seg_1.npz", "h"])
    T1 = int(ev["us"][-1])
    sp = dict(train=[0, int(0.6 * T1)], val=[int(0.7 * T1), int(0.8 * T1)], test=[int(0.9 * T1), T1],
              heldout_pair="linkusd", files=files, files_hash="fh")
    (tmp_path / "splits.json").write_text(json.dumps(sp))
    return tmp_path


def test_test_split_is_sealed_and_single_unseal(events_dir, monkeypatch):
    monkeypatch.delenv("MMFM_UNSEAL", raising=False)
    with pytest.raises(SealError):
        raw_chunks(events_dir, "test")
    monkeypatch.setenv("MMFM_UNSEAL", "modelhash1")
    with pytest.raises(SealError):
        raw_chunks(events_dir, "test")  # no declared pre-registered metric
    monkeypatch.setenv("MMFM_PURPOSE", "M1")
    raw_chunks(events_dir, "test")
    with pytest.raises(SealError):
        raw_chunks(events_dir, "test")  # second unseal of the same frozen model for the same metric
    log = (events_dir / "test_access.log").read_text().splitlines()
    assert len(log) == 1


def test_heldout_pair_never_in_train_or_val(events_dir):
    for split in ("train", "val"):
        ch, _ = raw_chunks(events_dir, split)
        assert ch and all(c["pair"] != "linkusd" for c in ch)


def test_split_ranges_respected_and_disjoint(events_dir):
    sp = json.loads((events_dir / "splits.json").read_text())
    for split in ("train", "val"):
        ch, _ = raw_chunks(events_dir, split)
        lo, hi = sp[split]
        assert all(c["events"]["us"].min() >= lo and c["events"]["us"].max() <= hi for c in ch)


def test_training_code_never_requests_sealed_splits():
    """L4: AST scan — no call in train.py / model/ passes 'test' or 'heldout_test' as a split."""
    bad = []
    for f in [ROOT / "train.py", *(ROOT / "model").glob("*.py")]:
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Constant) and node.value in ("test", "heldout_test"):
                bad.append(f"{f.name}:{node.lineno}")
    assert not bad, bad


def test_tokenizer_records_fit_split():
    ev, st = synth(500)
    tok = Tokenizer.fit([(ev, st)], dict(split="train", files_hash="fh", range=[0, 1]))
    j = tok.to_json()
    assert j["fit_split"]["split"] == "train" and len(j["hash"]) == 64
    assert Tokenizer.from_json(j).fit_split == tok.fit_split
