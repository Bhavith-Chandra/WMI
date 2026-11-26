"""Coinbase public L2 (level2_batch) + matches  ->  Almagest v2 event tokens (real data, physical time).

Approximations (documented, unavoidable with public data):
  * L2 is batched (50 ms): add+cancel pairs inside a batch are invisible; intra-batch order is message order.
  * Coarse price grid kappa = KAPPA_BP * reference price: bid bucket = floor(p/kappa), ask bucket = ceil(p/kappa),
    so the coarse spread is >= 1 bucket. Events deeper than D buckets from the own best are applied to the true book
    but not tokenised (outside the model's action window).
  * Sizes in lots of q0 (per product, fixed a priori below), size bucket = min(11, floor(log2(lots))) for lots >= 1.
  * L2 decreases at a bucket are first attributed to just-printed trades at that bucket (consumed liquidity); only the
    remainder is a cancel. Consecutive matches with the same taker_order_id form ONE market event.
  * Dt = exchange-time gap to the previous emitted event, log-bucketed (DT_EDGES, seconds).
Token = (typ, side, lvl, sizeb, dtb): typ 0 limit / 1 cancel / 2 market; side 0 buy(bid) / 1 sell(ask);
lvl as in the synthetic action space (limit u+1 in [0, D], cancel level in [0, D-1], market 0).
"""
import json, zlib, math, glob, os
from datetime import datetime
import numpy as np

D = 20
KAPPA_BP = 0.5e-4                      # 0.5 bp coarse tick
Q0 = {'BTC-USD': 0.0005, 'ETH-USD': 0.01, 'SOL-USD': 0.25}   # lot sizes (a priori, ~typical small trade)
DT_EDGES = np.array([0, 1e-4, 1e-3, 5e-3, 2e-2, 5e-2, 0.1, 0.3, 1.0, 3.0, 10.0])  # 11 edges -> 11 buckets (>= last)
NDT = len(DT_EDGES)
NSB = 12


def read_lines(path):
    """Truncation-tolerant gzip JSONL reader (live files may have an incomplete tail)."""
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    buf = b''
    with open(path, 'rb') as fh:
        while True:
            ch = fh.read(1 << 20)
            if not ch:
                break
            try:
                buf += d.decompress(ch)
            except Exception:
                break
            *ls, buf = buf.split(b'\n')
            for l in ls:
                try:
                    yield json.loads(l)
                except Exception:
                    return


def ts(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def dt_bucket(dt):
    return int(np.searchsorted(DT_EDGES, dt, side='right') - 1) if dt > 0 else 0


def size_bucket(lots):
    return int(min(NSB - 1, max(0, math.floor(math.log2(max(lots, 1.0))))))


class CoarseBook:
    def __init__(self, kappa, q0):
        self.k = kappa
        self.q0 = q0
        self.fine = {0: {}, 1: {}}          # side -> price -> size (base units)
        self.coarse = {0: {}, 1: {}}        # side -> bucket -> size (base units)

    def bucket(self, side, p):
        return math.floor(p / self.k) if side == 0 else math.ceil(p / self.k)

    def set_level(self, side, p, new):
        old = self.fine[side].get(p, 0.0)
        b = self.bucket(side, p)
        if new > 0:
            self.fine[side][p] = new
        else:
            self.fine[side].pop(p, None)
        c = self.coarse[side].get(b, 0.0) + (new - old)
        if c > 1e-12:
            self.coarse[side][b] = c
        else:
            self.coarse[side].pop(b, None)
        return b, new - old

    def best(self, side):
        cs = self.coarse[side]
        if not cs:
            return None
        return max(cs) if side == 0 else min(cs)

    def lots(self, side, b):
        v = self.coarse[side].get(b, 0.0)
        return 0 if v <= 0 else max(1, int(round(v / self.q0)))

    def features(self):
        bb, ba = self.best(0), self.best(1)
        f = np.zeros(21)
        cm = np.zeros((2, D), np.int8)
        if bb is None or ba is None:
            return f, cm, None
        for k in range(10):
            f[k] = self.lots(0, bb - k)
            f[10 + k] = self.lots(1, ba + k)
        for k in range(D):
            cm[0, k] = 1 if self.lots(0, bb - k) > 0 else 0
            cm[1, k] = 1 if self.lots(1, ba + k) > 0 else 0
        f[20] = max(1, ba - bb)
        return f, cm, 0.5 * (bb + ba)


def tokenize_segment(paths, product):
    """paths: ordered files of ONE connection segment (snapshot at start). Returns dict of arrays."""
    q0 = Q0[product]
    book = None
    ev, feat, mids, cms, times = [], [], [], [], []
    pend_trade = {}                           # (book side, bucket) -> consumed size awaiting the L2 decrease
    cur = None                                # merging market order: [taker_order_id, side, size, time]
    last_t = None
    stats = dict(l2_changes=0, deep_skipped=0, cancels=0, adds=0, markets=0, attributed=0.0)

    def emit(typ, side, lvl, lots, t):
        nonlocal last_t
        f, cm, mid = book.features()
        if mid is None:
            return
        dt = 0.0 if last_t is None else max(0.0, t - last_t)
        last_t = t
        ev.append((typ, side, lvl, size_bucket(lots), dt_bucket(dt)))
        feat.append(f)
        cms.append(cm)
        mids.append(mid)
        times.append(t)

    def flush_market():
        nonlocal cur
        if cur is not None:
            emit(2, cur[1], 0, max(1.0, cur[2] / q0), cur[3])
            stats['markets'] += 1
            cur = None

    for path in paths:
        for m in read_lines(path):
            t = m.get('type')
            if t == 'snapshot':
                mid = 0.5 * (float(m['bids'][0][0]) + float(m['asks'][0][0]))
                book = CoarseBook(KAPPA_BP * mid, q0)
                for p, s in m['bids']:
                    book.set_level(0, float(p), float(s))
                for p, s in m['asks']:
                    book.set_level(1, float(p), float(s))
                continue
            if book is None:
                continue
            if t == 'match':
                tm = ts(m['time'])
                taker_side = 0 if m['side'] == 'sell' else 1     # Coinbase 'side' = maker side
                if cur is not None and cur[0] != m['taker_order_id']:
                    flush_market()
                if cur is None:
                    # feature snapshot must be BEFORE the trade: emit at first fill of this taker
                    cur = [m['taker_order_id'], taker_side, 0.0, tm]
                cur[2] += float(m['size'])
                bside = 1 - taker_side
                b = book.bucket(bside, float(m['price']))
                pend_trade[(bside, b)] = pend_trade.get((bside, b), 0.0) + float(m['size'])
            elif t == 'l2update':
                flush_market()
                tm = ts(m['time'])
                for side_s, p, s in m['changes']:
                    stats['l2_changes'] += 1
                    side = 0 if side_s == 'buy' else 1
                    p, s = float(p), float(s)
                    own = book.best(side)
                    b = book.bucket(side, p)
                    old_best_opp = book.best(1 - side)
                    _, delta = book.set_level(side, p, s)
                    if own is None or abs(delta) < 1e-12:
                        continue
                    off = (own - b) if side == 0 else (b - own)
                    if delta > 0:
                        if off > D - 1:
                            stats['deep_skipped'] += 1
                            continue
                        u = max(-1, off)
                        # undo-apply ordering: features should be pre-event -> re-apply after emit
                        book.set_level(side, p, s - delta)
                        emit(0, side, u + 1, delta / q0, tm)
                        book.set_level(side, p, s)
                        stats['adds'] += 1
                    else:
                        dec = -delta
                        key = (side, b)
                        att = min(dec, pend_trade.get(key, 0.0))
                        if att > 0:
                            pend_trade[key] -= att
                            stats['attributed'] += att
                        rem = dec - att
                        if rem / q0 < 0.5:
                            continue
                        if off < 0 or off > D - 1:
                            stats['deep_skipped'] += 1
                            continue
                        book.set_level(side, p, s + rem)
                        emit(1, side, off, rem / q0, tm)
                        book.set_level(side, p, s)
                        stats['cancels'] += 1
                pend_trade = {k: v for k, v in pend_trade.items() if v > 1e-12}
        flush_market()
    out = dict(ev=np.array(ev, np.int64).reshape(-1, 5), feat=np.array(feat, np.float32).reshape(-1, 21),
               mids=np.array(mids), cm=np.array(cms, np.int8).reshape(-1, 2, D), times=np.array(times))
    out['stats'] = stats
    return out


def segments(root, product):
    """Group files by connection segment id, ordered by hour."""
    fs = sorted(glob.glob(os.path.join(root, product, '*.jsonl.gz')))
    segs = {}
    for f in fs:
        seg = f.rsplit('_seg', 1)[1].split('.')[0]
        segs.setdefault(seg, []).append(f)
    return [sorted(v) for k, v in sorted(segs.items())]
