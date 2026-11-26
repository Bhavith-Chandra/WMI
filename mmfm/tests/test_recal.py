"""EVAL_v3 harness: the online count recalibration reads nothing at/after the forecast index, and does react to the past."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.tokenizer_v2 import Tokenizer2  # noqa: E402
from model.mmfm2 import MMFM2  # noqa: E402
from sim.recal import count_bias, shift_count  # noqa: E402
from tests.test_v2 import synth2  # noqa: E402


def _setup(n=6000, seed=3):
    ev, st, lvq = synth2(n, seed=seed)
    rng = np.random.default_rng(seed)
    ev["us"] = 1_000_000_000_000 + np.cumsum(rng.exponential(100_000, n)).astype(np.int64)  # ~10 ev/s, 600 s
    tok = Tokenizer2.fit([(ev, st, lvq)], dict(split="train", files_hash="x", range=[0, 1]))
    torch.manual_seed(0)
    _, f, _ = tok.encode(ev, st, lvq)
    m = MMFM2(tok.vocab, f.shape[1], [tok.n_ret_bins(k) for k in range(3)], [tok.n_cnt_bins(k) for k in range(3)],
              d=32, n_layers=2, ctx=64).eval()
    return ev, st, lvq, tok, m


def test_recal_causal():
    ev, st, lvq, tok, m = _setup()
    idx = 5500
    b0, d0 = count_bias(m, tok, ev, st, lvq, idx)
    assert d0["n"] >= 3 and min(d0["per_h"]) >= 3 and np.any(b0 != 0)
    ev2, st2, lvq2 = ev.copy(), st.copy(), lvq.copy()
    ev2["us"][idx:] = ev2["us"][idx - 1] + 1  # collapse the whole future onto t0+1us
    ev2["type"][idx:] = (ev2["type"][idx:] + 1) % 3
    st2[idx:] *= 1.05
    lvq2[idx:] *= 7
    b1, _ = count_bias(m, tok, ev2, st2, lvq2, idx)
    assert np.array_equal(b0, b1), "recalibration read data at/after the forecast index"
    assert np.array_equal(b0, count_bias(m, tok, ev[:idx], st[:idx], lvq[:idx], idx)[0])


def test_recal_reacts_to_past_activity():
    ev, st, lvq, tok, m = _setup()
    idx = 5500
    b0, _ = count_bias(m, tok, ev, st, lvq, idx)
    ev2 = ev.copy()
    t0 = int(ev["us"][idx - 1])
    # compress the matured past (times only) -> more realised events per horizon -> larger bias
    lo = int(np.searchsorted(ev["us"], t0 - 400_000_000))
    ev2["us"][lo:idx] = t0 - ((t0 - ev["us"][lo:idx]) * 0.5).astype(np.int64)
    b1, _ = count_bias(m, tok, ev2, st, lvq, idx)
    assert np.all(b1 >= b0) and np.any(b1 > b0)


def test_shift_count():
    assert list(shift_count([0, 9, 99], 0.0)) == [0, 9, 99]
    assert list(shift_count([9, 99], np.log(2))) == [19, 199]
    assert shift_count(0, -1.0) == 0
