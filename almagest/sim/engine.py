"""Batched matching engine for neural rollouts (numba). Each row keeps a VIEW book (what the model sees)
and a REAL book (where prices are measured). Visible/free: injection hits both. Blind: injection hits REAL only;
model events are placed on VIEW (relative to view) and mirrored to REAL at the same absolute price."""
import numpy as np
from numba import njit, prange
from . import world as W

P, D = W.P, W.D
NB = 10


@njit(cache=True)
def world_book_at(par, seed, n_warm, bid, ask):
    ema = np.zeros(W.NSTATE)
    ss = np.zeros(int(par[8]) + 1, np.int64)
    sr = np.zeros(int(par[8]) + 1, np.int64)
    return W.warm_state(par, seed, n_warm, bid, ask, ema, ss, sr)


@njit(cache=True, parallel=True)
def init_rows(par, seeds, warms, vb, va, rb, ra, vbb, vba, rbb, rba):
    for i in prange(seeds.shape[0]):
        bb, ba = world_book_at(par, seeds[i], warms[i], vb[i], va[i])
        rb[i, :] = vb[i, :]
        ra[i, :] = va[i, :]
        vbb[i] = bb
        vba[i] = ba
        rbb[i] = bb
        rba[i] = ba


@njit(cache=True, parallel=True)
def observe(vb, va, vbb, vba, feat, cmask_b, cmask_a):
    """feat: (n, 21) raw volumes top-10 per side + spread (view book). cmask_*: (n, D) cancel validity."""
    for i in prange(vb.shape[0]):
        bb = vbb[i]
        ba = vba[i]
        for k in range(NB):
            feat[i, k] = vb[i, bb - k] if bb - k >= 0 else 0
            feat[i, NB + k] = va[i, ba + k] if ba + k < P else 0
        feat[i, 2 * NB] = ba - bb
        for k in range(D):
            cmask_b[i, k] = 1 if (bb - k >= 0 and vb[i, bb - k] > 0) else 0
            cmask_a[i, k] = 1 if (ba + k < P and va[i, ba + k] > 0) else 0


@njit(cache=True, parallel=True)
def step_events(vb, va, rb, ra, vbb, vba, rbb, rba, typ, side, lvl, size, active):
    """Apply model events (relative to VIEW) to VIEW and mirror to REAL. active[i]=0 -> skip row."""
    for i in prange(vb.shape[0]):
        if active[i] == 0:
            continue
        t, s, l, z = typ[i], side[i], lvl[i], size[i]
        pr = W.abs_price(vbb[i], vba[i], t, s, l)
        vbb[i], vba[i], _ = W.apply_action(vb[i], va[i], vbb[i], vba[i], t, s, l, z)
        vbb[i], vba[i] = W._guard(vb[i], va[i], vbb[i], vba[i])
        if t == 2:
            rbb[i], rba[i], _ = W.apply_action(rb[i], ra[i], rbb[i], rba[i], 2, s, 0, z)
        elif pr >= 0 and pr < P:
            rbb[i], rba[i], _ = W.apply_abs(rb[i], ra[i], rbb[i], rba[i], t, s, pr, z)
        rbb[i], rba[i] = W._guard(rb[i], ra[i], rbb[i], rba[i])


@njit(cache=True, parallel=True)
def inject(vb, va, rb, ra, vbb, vba, rbb, rba, side, size, to_view, active):
    for i in prange(vb.shape[0]):
        if active[i] == 0:
            continue
        rbb[i], rba[i], _ = W.apply_action(rb[i], ra[i], rbb[i], rba[i], 2, side[i], 0, size)
        rbb[i], rba[i] = W._guard(rb[i], ra[i], rbb[i], rba[i])
        if to_view[i] == 1:
            vbb[i], vba[i], _ = W.apply_action(vb[i], va[i], vbb[i], vba[i], 2, side[i], 0, size)
            vbb[i], vba[i] = W._guard(vb[i], va[i], vbb[i], vba[i])


# ---------------- v3 (real-data, replay-exact) observation: top-10 + spread + depth bands; cancel mask width = cm width
BANDS = ((10, 20), (20, 40), (40, 64))
NF3 = 2 * NB + 1 + 2 * len(BANDS)


@njit(cache=True)
def observe_row3(bid, ask, bb, ba, feat, cmb, cma):
    """ONE function used by the tokenizer (training features) and by the generator (rollout features)."""
    n = bid.shape[0]
    for k in range(NB):
        feat[k] = bid[bb - k] if bb - k >= 0 else 0
        feat[NB + k] = ask[ba + k] if ba + k < n else 0
    feat[2 * NB] = ba - bb
    j = 2 * NB + 1
    for s in range(2):
        for q in range(len(BANDS)):
            a, b = BANDS[q]
            tot = 0
            for k in range(a, b):
                if s == 0:
                    tot += bid[bb - k] if bb - k >= 0 else 0
                else:
                    tot += ask[ba + k] if ba + k < n else 0
            feat[j] = tot
            j += 1
    for k in range(cmb.shape[0]):
        cmb[k] = 1 if (bb - k >= 0 and bid[bb - k] > 0) else 0
        cma[k] = 1 if (ba + k < n and ask[ba + k] > 0) else 0


@njit(cache=True, parallel=True)
def observe3(vb, va, vbb, vba, feat, cmask_b, cmask_a):
    for i in prange(vb.shape[0]):
        observe_row3(vb[i], va[i], vbb[i], vba[i], feat[i], cmask_b[i], cmask_a[i])
