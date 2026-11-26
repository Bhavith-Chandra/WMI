"""Real-time path == batch path (features, tokens, hidden states)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.tokenizer import Tokenizer, state_features  # noqa: E402
from model.infer import FastMMFM, IncFeatures  # noqa: E402
from model.mmfm import MMFM  # noqa: E402
from tests.test_mmfm_leakage import synth  # noqa: E402


def test_incremental_features_and_tokens_equal_batch():
    ev, st = synth(1500, seed=3)
    tok = Tokenizer.fit([(ev, st)], dict(split="train", files_hash="x", range=[0, 1]))
    t_b, f_b, _ = tok.encode(ev, st)
    inc = IncFeatures(tok)
    inc.seed(st[0])
    prev = None
    for i in range(len(ev) - 1):
        row, dt = inc.tokens_for(int(ev["type"][i]), int(ev["side"][i]), float(ev["price"][i]), float(ev["qty"][i]),
                                 int(ev["us"][i]), prev, st[i])
        assert np.array_equal(row, t_b[i]), f"token mismatch at {i}"
        f = inc.push(dt, st[i + 1])  # post-state of event i == pre-state of i+1
        assert np.allclose(f, f_b[i], atol=1e-4), f"feature mismatch at {i}: {f - f_b[i]}"
        prev = int(ev["us"][i])


def test_kv_cache_equals_full_forward_including_window_rolls():
    torch.manual_seed(0)
    ev, st = synth(900, seed=4)
    tok = Tokenizer.fit([(ev, st)], dict(split="train", files_hash="x", range=[0, 1]))
    t, f, _ = tok.encode(ev, st)
    m = MMFM(tok.vocab, f.shape[1], d=32, n_layers=2, n_heads=4, ctx=64).eval()
    fm = FastMMFM(m, keep=32)
    T = torch.from_numpy(t)[None]
    F_ = torch.from_numpy(f)[None]
    h = fm.prefill(T[:, :40], F_[:, :40])
    ref = m.hidden(T[:, :40], F_[:, :40])[:, -1]
    assert torch.allclose(h, ref, atol=1e-5)
    start = 0  # track the model's current window start to build the reference
    win = list(range(0, 40))
    for i in range(40, 200):
        h = fm.step(T[:, i], F_[:, i])
        win.append(i)
        if len(win) > 64:  # FastMMFM re-prefilled from the last `keep` events
            win = win[-32:]
        ref = m.hidden(T[:, win], F_[:, win])[:, -1]
        assert torch.allclose(h, ref, atol=1e-4), f"step {i}: max diff {(h - ref).abs().max()}"
