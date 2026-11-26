"""Harness self-tests. Run: python -m almagest.tests.test_harness  (from repo root)."""
import sys
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest.sim import world as W, metrics as M, campaign as C
from almagest.models import lmm as L


def test_estimator_recovers_planted_law():
    rng = np.random.default_rng(0)
    for delta in (0.4, 0.5, 0.7, 1.0):
        nday, nwin = 20, 16
        V = rng.uniform(15000, 30000, nday)
        q = (np.arange(1, 101)[None, :] * 2) / np.repeat(V, nwin)[:, None]
        did = np.repeat(np.arange(nday), nwin)
        day_shock = rng.normal(0, 0.1, nday)[did][:, None]
        I = 0.7 * q ** delta * q.max() ** (0.5 - delta) * (1 + day_shock) + rng.normal(0, 0.05, q.shape)
        d, Y = M.law_from_pairs(I, q)
        assert abs(d - delta) < 0.12, (delta, d)   # sampling variance with 10% day shocks
        d0, _ = M.law_from_pairs(0.7 * q ** delta, q)
        assert abs(d0 - delta) < 0.01, (delta, d0)   # exact without noise
        ok, _, _ = M.identifiable(q, I)
        assert ok


def test_null_not_identified():
    rng = np.random.default_rng(1)
    q = (np.arange(1, 101)[None, :] * 2) / rng.uniform(15000, 30000, 200)[:, None]
    fails = 0
    for _ in range(50):
        I = rng.normal(0, 0.05, q.shape)
        ok, _, _ = M.identifiable(q, I)
        fails += ok
    assert fails <= 2, fails


def test_world_blind_stream_equals_free():
    par = W.default_params()
    par[26] = 0.1
    par[25] = 0.3
    par[23] = 0.2
    sched = C.schedule(2, 3, 777)
    res, hashes, _ = C.run_arms(C.world_runner(par), sched, 20, 50, 5, 2, modes=(1, 2, 0))
    assert np.all(hashes[2][0] == hashes[0][0]) and np.all(hashes[2][1] == hashes[0][1])
    assert not np.all(hashes[1][0] == hashes[0][0])  # visible must differ


def test_scan_equals_step():
    torch.manual_seed(0)
    for mode in ('learned', 'modebank'):
        m = L.DiagRec(16, 32, mode)
        x = torch.randn(2, 48, 16)
        h0 = torch.randn(2, 32)
        y, h = m.forward_seq(x, h0)
        hh, ys = h0, []
        for t in range(48):
            yt, hh = m.forward_step(x[:, t], hh)
            ys.append(yt)
        assert (torch.stack(ys, 1) - y).abs().max() < 1e-4
        assert (hh - h).abs().max() < 1e-4


def test_slowbank_online_equals_offline():
    rng = np.random.default_rng(2)
    n = 3000
    sv = rng.integers(-3, 4, n).astype(float)
    dm = rng.normal(0, 0.5, n)
    fv, dev = L.slowbank_offline(sv, dm)
    sb = L.SlowBank(1)
    for t in range(n):
        sb.update(torch.tensor([sv[t]], dtype=torch.float32), torch.tensor([dm[t]], dtype=torch.float32))
    ref = L.sb_scale(torch.tensor(fv[-1:], dtype=torch.float32), torch.tensor(dev[-1:], dtype=torch.float32))
    assert (sb.feat() - ref).abs().max() < 1e-2, (sb.feat() - ref).abs().max()


def test_seed_splits_disjoint():
    from almagest import splits as S
    bases = [S.CALIB_BASE, S.DEV_BASE, S.BURNED_BASE, S.TEST_BASE, S.TRAIN_BASE, S.VAL_BASE]
    rng = [set(range(b, b + S.SPAN)) for b in bases]
    for i in range(len(rng)):
        for j in range(i + 1, len(rng)):
            assert not (rng[i] & rng[j]), (bases[i], bases[j])
    assert max(bases) + S.SPAN < S.ROLL_OFFSET   # day/stream seeds < every rollout seed (>= ROLL_OFFSET + 1000*day)
    import inspect
    from almagest.experiments import e2_eval, e2_train
    assert 'S.TEST_BASE' in inspect.getsource(e2_eval) and 'S.DEV_BASE' in inspect.getsource(e2_eval)
    assert 'S.TRAIN_BASE' in inspect.getsource(e2_train) and 'S.VAL_BASE' in inspect.getsource(e2_train)


def test_engine_book_equals_stream_book():
    from almagest.sim import engine as E
    from almagest.experiments.e2_train import build_stream
    par = W.default_params()
    s = build_stream(par, 4242, 3001)
    vb = np.zeros((1, W.P), np.int64); va = np.zeros_like(vb); rb = np.zeros_like(vb); ra = np.zeros_like(vb)
    z = [np.zeros(1, np.int64) for _ in range(4)]
    E.init_rows(par, np.array([4242]), np.array([20000 + 3000]), vb, va, rb, ra, *z)
    feat = np.zeros((1, 21)); cb = np.zeros((1, W.D), np.int64); ca = np.zeros_like(cb)
    E.observe(vb, va, z[0], z[1], feat, cb, ca)
    assert np.allclose(feat[0], s['feat'][3000])


def test_vb_fails_closed():
    assert np.isnan(M.vb_ratio(np.array([1.0]), np.array([-0.1])))
    assert np.isnan(M.vb_ratio(np.array([-1.0]), np.array([0.1])))
    assert abs(M.vb_ratio(np.array([3.0]), np.array([1.0])) - 3) < 1e-12


def test_snapshots_bit_identical():
    par = W.default_params()
    par[25], par[23], par[26], par[28] = 0.3, 0.2, 0.05, 0.2
    days = np.array([4100, 4101], np.int64)
    warms = np.array([30000, 32048], np.int64)
    S = W.Snapshots(par, days, warms)
    ds, ws = np.repeat(days, 2), np.tile(warms, 2)
    rs = np.arange(4, dtype=np.int64) + 777
    for mode in (0, 1, 2):
        a = W.run_batch(par, ds, ws, rs, 10, 40, 3, 1, 2, mode)
        b = S.run(ds, ws, rs, 10, 40, 3, 1, 2, mode)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2])


def test_real_tokenizer_offline_equals_streaming():
    """Training/serving parity: cache builder and live feed use the same StreamTokenizer."""
    from almagest.data import coinbase_tokenize as T
    import json as _j, os as _o, tempfile, gzip
    msgs = [
        {"type": "snapshot", "product_id": "SOL-USD", "bids": [["100.00", "5"], ["99.99", "3"]],
         "asks": [["100.01", "4"], ["100.02", "2"]], "time": "2026-11-26T00:00:00Z"},
        {"type": "l2update", "changes": [["buy", "100.00", "6"]], "time": "2026-11-26T00:00:00.010Z"},
        {"type": "match", "side": "sell", "size": "2", "price": "100.01", "taker_order_id": "a",
         "time": "2026-11-26T00:00:00.020Z"},
        {"type": "l2update", "changes": [["sell", "100.01", "2"], ["sell", "100.03", "1"]],
         "time": "2026-11-26T00:00:00.030Z"},
        {"type": "l2update", "changes": [["buy", "99.99", "0"]], "time": "2026-11-26T00:00:01Z"},
    ]
    d = tempfile.mkdtemp()
    f = _o.path.join(d, 'x.jsonl.gz')
    with gzip.open(f, 'wt') as g:
        for m in msgs:
            g.write(_j.dumps(m) + "\n")
    a = T.tokenize_segment([f], 'SOL-USD')
    st = T.StreamTokenizer('SOL-USD')
    toks = []
    for m in msgs:
        toks += st.feed(m)
    toks += st.flush()
    b = T._pack(toks, st.stats)
    for k in ('ev', 'feat', 'mids', 'cm', 'times'):
        assert np.array_equal(a[k], b[k]), k
    assert len(a['ev']) >= 3 and 2 in a['ev'][:, 0]


def test_mirror_involution():
    from almagest.experiments.e2_train import build_stream, mirror
    par = W.default_params()
    d = build_stream(par, 4343, 2000)
    mm = mirror(mirror(d))
    for k in d:
        assert np.array_equal(np.asarray(d[k]), np.asarray(mm[k])), k
    m = mirror(d)
    assert np.array_equal(m['tgt'][:, 1], 1 - d['tgt'][:, 1])
    assert np.array_equal(m['feat'][:, :10], d['feat'][:, 10:20])
    assert np.allclose(m['dmid'], -d['dmid'])


def _fake_feed(seed=0, n=4000):
    """Self-contained Coinbase-like feed: snapshot, random L2 batches (adds/cancels at many depths, inside-spread
    improvements, crossing-in-batch orderings) and match groups. Prices on a 0.01 grid around 100."""
    rng = np.random.default_rng(seed)
    bids = {round(100 - 0.01 * (k + 1), 2): float(rng.integers(1, 50)) for k in range(400)}
    asks = {round(100 + 0.01 * k, 2): float(rng.integers(1, 50)) for k in range(400)}
    yield dict(type='snapshot', bids=[[str(p), str(s)] for p, s in sorted(bids.items(), reverse=True)],
               asks=[[str(p), str(s)] for p, s in sorted(asks.items())])
    t0 = 1.79e9
    for e in range(n):
        tm = '2026-11-26T00:00:%09.6fZ' % ((e * 0.013) % 59)
        if rng.random() < 0.1 and asks and bids:
            side = 'buy' if rng.random() < 0.5 else 'sell'      # maker side
            book = bids if side == 'buy' else asks
            best = max(book) if side == 'buy' else min(book)
            sz = min(book[best], float(rng.integers(1, 30)))
            yield dict(type='match', side=side, price=str(best), size=str(sz), taker_order_id='t%d' % (e // 2), time=tm)
            book[best] -= sz
            ch = [[side, str(best), str(book[best])]]
            if book[best] <= 0:
                book.pop(best)
            yield dict(type='l2update', changes=ch, time=tm)
            continue
        ch = []
        for _ in range(int(rng.integers(1, 4))):
            side = 'buy' if rng.random() < 0.5 else 'sell'
            book = bids if side == 'buy' else asks
            bb, ba = max(bids), min(asks)
            d = int(min(rng.geometric(0.08), 150))
            if rng.random() < 0.15:
                p = round(bb + 0.01 * int(rng.integers(1, 9)), 2) if side == 'buy' else round(ba - 0.01 * int(rng.integers(1, 9)), 2)
                if not (p < ba if side == 'buy' else p > bb):
                    continue
            else:
                p = round(bb - 0.01 * d, 2) if side == 'buy' else round(ba + 0.01 * d, 2)
            new = 0.0 if (p in book and rng.random() < 0.4) else float(rng.integers(1, 60))
            if new > 0:
                book[p] = new
            else:
                book.pop(p, None)
            ch.append([side, str(p), str(new)])
        if ch:
            yield dict(type='l2update', changes=ch, time=tm)


def test_exact_tokenizer_replay():
    """GATE (EVAL_PROTOCOL B0): replaying v3 tokens (+ hidden updates) through the ENGINE reproduces every recorded
    feature bit-for-bit, random access via checkpoints is exact, and token-only replay (what a perfect model could
    generate) from an exact seed matches when no hidden update falls inside the horizon."""
    from almagest.data import coinbase_exact as X
    from almagest.sim import engine as E
    X.Q0['TEST'] = 1.0
    st = X.ExactTokenizer('TEST')
    for m in _fake_feed():
        st.feed(m)
    st.flush()
    o = st.pack(); z = dict(o); z.update(X.ck_arrays(o['ck']))
    n = len(o['ev'])
    assert n > 2000 and (o['ev'][:, 2] > X.DR).any() and (o['ev'][:, 0] == 2).any()
    bad = [0]

    def obs(bid, ask, bb, ba):
        f = np.zeros(X.NF); cb = np.zeros(X.DR, np.int64); ca = np.zeros(X.DR, np.int64)
        E.observe_row3(bid, ask, bb, ba, f, cb, ca)
        return f, np.stack([cb, ca])

    def on(k, bid, ask, bb, ba):
        f, cm = obs(bid, ask, bb, ba)
        bad[0] += int(not (np.array_equal(f, o['feat'][k]) and np.array_equal(cm, o['cm'][k])))
    bid, ask, bb, ba = X._load_ck(z, 0)
    X.replay(z, int(z['ck_pos'][0]), n, bid, ask, bb, ba, on)
    assert bad[0] == 0, bad[0]
    for k in (n // 3, n - 1):
        f, _ = obs(*X.book_at(z, k))
        assert np.array_equal(f, o['feat'][k])
    hp = o['hid'][:, 0]
    for s in range(100, n - 200, 97):
        e = s + 100
        if ((hp > s) & (hp <= e)).any():
            continue
        bid, ask, bb, ba = X.book_at(z, s)
        for k in range(s, e):
            bb, ba, _ = W.apply_action(bid, ask, bb, ba, int(o['ev'][k, 0]), int(o['ev'][k, 1]),
                                       int(X.eng_lvl(o['ev'][k, 0], o['ev'][k, 2])), int(o['lots'][k]))
        assert np.array_equal(obs(bid, ask, bb, ba)[0], o['feat'][e])


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f()
            print('PASS', name)
