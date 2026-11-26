"""EVAL_v3 online count recalibration (causal).

For a forecast made after event idx-1 (time t0 = us[idx-1]) the bias of the count head at horizon h is
    b_h = clip(median_k log((N_h(o_k) + 1) / (m_h(o_k) + 1)), -1, 1)
over origins o_k = t0 - 10 s - k*30 s (k = 0..9). The origin event j_k is the last event with us <= o_k;
m_h is the calibrated count-head median from the context ending at j_k, and N_h is the realised count with
exactly the training-target definition (events after j_k with us <= us[j_k] + h), computed from events[:idx]
only and used only when an event strictly after us[j_k] + h exists before idx. b_h = 0 with < 3 valid origins.
Nothing at or after index idx is read (tests/test_recal.py).
"""
from __future__ import annotations

import numpy as np
import torch

from data.tokenizer_v2 import HORIZONS

N_ORIG, STEP_S, LAG_S, MIN_HIST, CLIP = 10, 30.0, 10.0, 1100, 1.0


def _median_count(p, edges):
    p = np.asarray(p, np.float64)
    p = p / p.sum()
    b = int(np.searchsorted(np.cumsum(p), 0.5))
    e = np.asarray(edges, np.float64)
    lo = 0.0 if b == 0 else e[b - 1]
    hi = e[b] if b < len(e) else e[-1] + 0.5
    return float(np.expm1(0.5 * (lo + hi)))


@torch.no_grad()
def count_bias(model, tok, ev, st, lvq, idx):
    """ev/st/lvq: arrays of history (may extend past idx; only [:idx] is read). Returns (b[3], diag)."""
    us = ev["us"][:idx].astype(np.int64)
    T = model.cfg["ctx"]
    t0 = int(us[-1])
    js = []
    for k in range(N_ORIG):
        o = t0 - int((LAG_S + k * STEP_S) * 1e6)
        j = int(np.searchsorted(us, o, side="right")) - 1
        if j >= max(MIN_HIST, T):
            js.append(j)
    if len(js) < 3:
        return np.zeros(len(HORIZONS)), dict(n=len(js))
    # encode events a..idx-1 with >= MIN_HIST events before every context (windowed activity features need history,
    # as in context_from_history2); feature row r uses st[r+1], and every used row r <= max(js) < idx - 1.
    a = max(0, min(js) - MIN_HIST)
    tokens, feats, _ = tok.encode(ev[a:idx], st[a:idx], lvq[a:idx])
    X = torch.from_numpy(np.stack([tokens[j - a - T + 1:j - a + 1] for j in js]))
    F = torch.from_numpy(np.stack([feats[j - a - T + 1:j - a + 1] for j in js]))
    h = model.hidden(X, F)[:, -1]
    _, c = model.forecast_logits(h)
    b, diag = np.zeros(len(HORIZONS)), dict(n=len(js), per_h=[])
    for k, hz in enumerate(HORIZONS):
        pc = torch.softmax(c[k].float(), -1).numpy()
        lr = []
        for i, j in enumerate(js):
            jj = int(np.searchsorted(us, us[j] + int(hz * 1e6), side="right"))
            if jj >= idx:  # horizon not matured inside the known history
                continue
            n_real = jj - j - 1
            m = _median_count(pc[i], tok.edges_cnt[k])
            lr.append(np.log((n_real + 1.0) / (m + 1.0)))
        if len(lr) >= 3:
            b[k] = float(np.clip(np.median(lr), -CLIP, CLIP))
        diag["per_h"].append(len(lr))
    return b, diag


def shift_count(n, b):
    """N' = floor((N + 1) e^b - 1), elementwise, >= 0."""
    return np.maximum(np.floor((np.asarray(n, np.float64) + 1.0) * np.exp(b) - 1.0 + 1e-9), 0).astype(np.int64)
