"""v2 harness: causality of features, correctness/masking of forecast targets, state-relative size tokens."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.tokenizer_v2 import (CANCEL, TAKE, WHOLE_CANCEL, WHOLE_TAKE, HORIZONS, Tokenizer2,  # noqa: E402
                               forecast_targets, state_features)
from model.mmfm2 import MMFM2  # noqa: E402
from tests.test_mmfm_leakage import synth  # noqa: E402


def synth2(n=3000, seed=0):
    ev, st = synth(n, seed)
    rng = np.random.default_rng(seed)
    lvq = ev["qty"] * rng.choice([1.0, 1.5, 3.0], n)  # level >= qty
    return ev, st, lvq


def test_v2_features_no_lookahead():
    ev, st, _ = synth2()
    f0 = state_features(ev, st)
    j = 1500
    ev2, st2 = ev.copy(), st.copy()
    ev2["us"][j:] += 7_000_000
    st2[j + 1:] *= 1.01
    f1 = state_features(ev2, st2)
    assert np.allclose(f0[: j - 1], f1[: j - 1], atol=1e-6)
    assert np.allclose(f0[j - 1], f1[j - 1], atol=1e-6), "feature used to predict event j depends on event j"
    assert not np.allclose(f0[j:], f1[j:])


def test_forecast_targets_exact_and_masked():
    us = np.array([0, 400_000, 900_000, 1_500_000, 6_200_000, 11_000_000], dtype=np.int64)
    ev = np.zeros(len(us), dtype=[("us", "i8"), ("type", "i1"), ("side", "i1"), ("price", "f8"), ("qty", "f8")])
    ev["us"] = us
    mid = np.array([100.0, 100.0, 100.01, 100.02, 100.03, 100.04])
    st = np.stack([mid - 0.005, mid + 0.005, np.ones(6), np.ones(6), np.ones(6), np.ones(6)], 1)
    R, N = forecast_targets(ev, st)
    # event 0 at t=0: mid after event 0 = pre-state of event 1 = 100.0; mid at t+1s = pre-state of first event
    # after 1.0 s (event 3 at 1.5 s) = 100.02; events in (0, 1s]: 2
    assert N[0, 0] == 2 and np.isclose(R[0, 0], np.log(100.02 / 100.0) * 1e4)
    # 10-s horizon from event 3 (1.5 s -> 11.5 s) runs past the last event -> masked
    assert N[3, 2] == -1 and np.isnan(R[3, 2])


def test_state_relative_size_tokens():
    ev, st, lvq = synth2(500)
    tok = Tokenizer2.fit([(ev, st, lvq)], dict(split="train", files_hash="x", range=[0, 1]))
    t, _, _ = tok.encode(ev, st, lvq)
    whole_c = (ev["type"] == CANCEL) & (np.isclose(ev["qty"], lvq))
    assert (t[whole_c, 3] == WHOLE_CANCEL).all()
    part_c = (ev["type"] == CANCEL) & (lvq > ev["qty"] * 1.2)
    assert (t[part_c, 3] < WHOLE_CANCEL).all()
    take_whole = (ev["type"] == TAKE) & np.isclose(ev["qty"], lvq)
    assert (t[take_whole, 3] == WHOLE_TAKE).all()


def test_v2_model_causal_and_heads():
    ev, st, lvq = synth2(600)
    tok = Tokenizer2.fit([(ev, st, lvq)], dict(split="train", files_hash="x", range=[0, 1]))
    t, f, _ = tok.encode(ev, st, lvq)
    rt, ct = tok.encode_targets(ev, st)
    m = MMFM2(tok.vocab, f.shape[1], [tok.n_ret_bins(k) for k in range(3)], [tok.n_cnt_bins(k) for k in range(3)], d=32, n_layers=2, ctx=64).eval()
    T = torch.from_numpy(t[:64])[None]
    F = torch.from_numpy(f[:64])[None]
    h1 = m.hidden(T, F)
    T2, F2 = T.clone(), F.clone()
    T2[0, 40:, 2] = (T2[0, 40:, 2] + 3) % tok.vocab["px"]
    F2[0, 40:] += 1
    h2 = m.hidden(T2, F2)
    assert torch.allclose(h1[0, :40], h2[0, :40], atol=1e-5)
    ev_bits, lr, lc = m.losses(T, F, torch.from_numpy(rt[:64])[None], torch.from_numpy(ct[:64])[None])
    assert ev_bits.shape == (1, 63, 5) and lr.shape == (len(HORIZONS),) and torch.isfinite(lr).all()


def test_v2_incremental_equals_batch():
    from model.infer2 import IncFeatures2
    ev, st, lvq = synth2(1500, seed=5)
    ev["us"] = np.sort(ev["us"])
    ev["us"][100::37] -= 3000  # non-monotonic exchange stamps, as in real data
    tok = Tokenizer2.fit([(ev, st, lvq)], dict(split="train", files_hash="x", range=[0, 1]))
    tb, fb, _ = tok.encode(ev, st, lvq)
    inc = IncFeatures2(tok)
    inc.seed(st[0])
    prev = None
    for i in range(len(ev) - 1):
        row, dt = inc.tokens_for(int(ev["type"][i]), int(ev["side"][i]), float(ev["price"][i]), float(ev["qty"][i]),
                                 int(ev["us"][i]), prev, st[i], float(lvq[i]))
        assert np.array_equal(row, tb[i]), (i, row, tb[i])
        f = inc.push(int(ev["us"][i]), dt, st[i + 1])
        assert np.allclose(f, fb[i], atol=1e-4), (i, f - fb[i])
        prev = int(ev["us"][i])
