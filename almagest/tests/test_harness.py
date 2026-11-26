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


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f()
            print('PASS', name)
