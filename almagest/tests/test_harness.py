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
    calib = set(range(1000, 1000 + 64))
    evald = set(range(5000, 5000 + 64))
    train = set(range(200000, 200000 + 64))
    val = set(range(300000, 300000 + 64))
    s = [calib, evald, train, val]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (s[i] & s[j])


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f()
            print('PASS', name)
