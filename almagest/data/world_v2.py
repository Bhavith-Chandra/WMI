"""Convert ground-truth World streams (with tagged interventions) into the real-data v2 token format so ONE model can
be pretrained on synthetic causal data and real Coinbase data together (domain 0 = World, 1 = real).

Alignment (fixed, documented, never tuned on report data):
  * lot scale K: World volumes/sizes are multiplied by K so that the median top-of-book depth matches the real data's
    median (K measured on the real FIT split only, passed in by the caller).
  * size bucket = min(NSB-1, floor(log2(scaled lots))) - same rule as the real tokenizer.
  * time: Poisson arrivals at `rate` events/s, then the public feed's 50 ms batching -> Dt buckets with the same edges.
    World timing is therefore distribution-compatible but carries no information; real data teaches timing.
Token columns: typ, side, lvl, size_b, dt_b, src, domain.
"""
import math
import numpy as np
from almagest.sim import world as W
from almagest.data import coinbase_tokenize as T
from almagest.experiments.e2_train import inj_schedule


def world_v2_stream(par, seed, n, K, rate=240.0, batch_s=0.05, phi=0.11, busy=0.7):
    rng = np.random.default_rng(seed + 7)
    inj, iside, isize = inj_schedule(n, phi, rng, busy=busy)
    ev = np.zeros((n, 4), np.int64)
    feat = np.zeros((n, 21), np.float64)
    mids = np.zeros(n)
    cm = np.zeros((n, 2, W.D), np.int8)
    src = np.zeros(n, np.int64)
    W.gen_stream_inj(par, seed, n, inj, iside, isize, ev, feat, mids, cm, src)
    # synthetic clock with public-feed batching
    t = np.cumsum(rng.exponential(1.0 / rate, n))
    b = np.floor(t / batch_s)
    dt = np.concatenate([[0.0], np.diff(t)])
    dt[1:][b[1:] == b[:-1]] = 0.0
    dtb = np.array([T.dt_bucket(x) for x in dt], np.int64)
    lots = ev[:, 3].astype(float) * K
    sb = np.array([T.size_bucket(x) for x in lots], np.int64)
    tok = np.stack([ev[:, 0], ev[:, 1], ev[:, 2], sb, dtb, src, np.zeros(n, np.int64)], 1)
    f = feat.copy()
    f[:, :20] *= K
    return dict(ev=tok, feat=f.astype(np.float32), mids=mids, cm=cm)


def depth_scale(real_feat, world_feat):
    """K = median real top-of-book depth / median World top-of-book depth (both sides pooled)."""
    r = np.median(np.concatenate([real_feat[:, 0], real_feat[:, 10]]))
    w = np.median(np.concatenate([world_feat[:, 0], world_feat[:, 10]]))
    return float(max(1.0, r / max(w, 1e-9)))
