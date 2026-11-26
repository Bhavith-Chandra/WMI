"""Use case 3: manipulation detection on labelled synthetic manipulation (protocol gate M1, dev split).

Families injected into ground-truth World streams (the market treats them as ordinary flow; labels are never inputs):
  spoof     : K unit limit adds at one level, opposite-side market order, then K cancels at that level
  layering  : unit adds over 3-4 levels, then all cancelled
  ignition  : burst of same-side market orders, then a partial reversal
Detectors:
  rule_spoof           : hand-written template rule for SPOOFING (burst of adds at one price, later burst of cancels)
  model_tail_surprise  : unsupervised; window score = mean of the top-10% per-event NLL
  probe_heldout_family : logistic probe on the frozen model's hidden state, trained on the OTHER two families + clean
                         windows from disjoint streams (protocol M1: unseen injector families)
Metrics per family: window AUC (family-positive windows vs clean windows), precision at a 2% alert budget.
usage: e4_manip.py <ckpt.pt> <out.json> [n_streams]
"""
import sys, os, json
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest import splits as S
from almagest.sim import world as W
from almagest.models import lmm as L
from almagest.experiments.e2_train import make_model
from almagest.experiments.e1_calibrate_world import make

torch.set_num_threads(2)
WINW = 256
FAM = ('spoof', 'layering', 'ignition')


def schedule(n, rng, rate=1 / 4000):
    f_typ = np.full(n, -1, np.int64); f_side = np.zeros(n, np.int64); f_lvl = np.zeros(n, np.int64)
    f_ep = np.zeros(n, np.int64); f_size = np.ones(n, np.int64); lab = np.zeros(n, np.int64)   # 0 clean, 1..3 family
    t = int(rng.integers(2000, 4000)); ep = 0
    while t < n - 2000:
        fam = int(rng.integers(3)); s = int(rng.integers(2))
        if fam == 0:
            K = int(rng.integers(6, 13)); u = int(rng.integers(1, 3)); Wt = int(rng.integers(50, 300))
            for k in range(K):
                f_typ[t + k], f_side[t + k], f_lvl[t + k], f_ep[t + k], lab[t + k] = 0, s, u + 1, ep, 1
            tm = t + K + Wt // 2
            f_typ[tm], f_side[tm], f_size[tm], lab[tm] = 2, 1 - s, int(rng.integers(2, 5)), 1
            tc = t + K + Wt
            for k in range(K):
                f_typ[tc + k], f_side[tc + k], f_ep[tc + k], lab[tc + k] = 1, s, ep, 1
            ep += 1; t = tc + K
        elif fam == 1:
            Lv = int(rng.integers(3, 5)); kp = int(rng.integers(3, 6)); Wt = int(rng.integers(50, 300)); eps = []
            tt = t
            for lv in range(Lv):
                for k in range(kp):
                    f_typ[tt], f_side[tt], f_lvl[tt], f_ep[tt], lab[tt] = 0, s, lv + 2, ep + lv, 2
                    tt += 1
                eps.append(ep + lv)
            tc = tt + Wt
            for e in eps:
                for k in range(kp):
                    f_typ[tc], f_side[tc], f_ep[tc], lab[tc] = 1, s, e, 2
                    tc += 1
            ep += Lv; t = tc
        else:
            M = int(rng.integers(6, 11)); Wt = int(rng.integers(50, 150))
            for k in range(M):
                f_typ[t + 2 * k], f_side[t + 2 * k], f_size[t + 2 * k], lab[t + 2 * k] = 2, s, int(rng.integers(2, 5)), 3
            tr = t + 2 * M + Wt
            for k in range(M // 2):
                f_typ[tr + 2 * k], f_side[tr + 2 * k], f_size[tr + 2 * k], lab[tr + 2 * k] = 2, 1 - s, int(rng.integers(2, 5)), 3
            t = tr + M
        t += int(rng.exponential(1 / rate))
    return f_typ, f_side, f_lvl, f_ep, f_size, lab, ep + 8


def stream(par, seed, n, rng):
    f_typ, f_side, f_lvl, f_ep, f_size, lab, nep = schedule(n, rng)
    ev = np.zeros((n, 4), np.int64); feat = np.zeros((n, 21)); mids = np.zeros(n); cm = np.zeros((n, 2, W.D), np.int8)
    ep_price = np.full(nep, -1, np.int64)
    W.gen_stream_forced(par, seed, n, f_typ, f_side, f_lvl, f_ep, f_size, ev, feat, mids, cm, ep_price)
    return ev, feat, mids, cm, lab


@torch.no_grad()
def surprise(model, ev, feat, mids, cm):
    n = len(ev)
    src = bool(getattr(model, 'src', False)); tag = bool(getattr(model, 'tagch', False))
    e = np.concatenate([ev, np.zeros((n, 1), np.int64)], 1) if src else ev
    prev = np.vstack([[0, 0, 1, 1] + ([0] if src else []), e[:-1]])
    dmid = np.concatenate([[0.0], np.diff(mids)])
    sv = np.where(prev[:, 0] == 2, np.where(prev[:, 1] == 0, 1, -1) * prev[:, 3], 0).astype(float)
    fv, dev = L.slowbank_offline(sv, dmid)
    sb = L.sb_scale(torch.tensor(fv, dtype=torch.float32), torch.tensor(dev, dtype=torch.float32))
    if tag:
        sb = torch.cat([sb, torch.zeros(n, len(L.SB_TAUS))], 1)
    t = lambda a, dt=None: torch.tensor(a[None]) if dt is None else torch.tensor(a[None], dtype=dt)
    x = model.embed(t(prev), t(feat, torch.float32), t(dmid, torch.float32), sb[None])
    z, _ = model.trunk_seq(x[:, :(n // 16) * 16], model.init_state(1))
    m = z.shape[1]
    E = torch.tensor(ev[:m])
    lt, ls, ll, lz = model.heads_logits(z, E[None, :, 0], E[None, :, 1])
    ll = L.mask_lvl(ll, E[None, :, 0], E[None, :, 1], torch.tensor(cm[None, :m]))
    Fn = torch.nn.functional
    nll = Fn.cross_entropy(lt[0], E[:, 0], reduction='none') + Fn.cross_entropy(ls[0], E[:, 1], reduction='none')
    nll = nll + torch.where(E[:, 0] != 2, Fn.cross_entropy(ll[0], E[:, 2], reduction='none'), torch.zeros(m))
    nll = nll + torch.where(E[:, 0] == 2, Fn.cross_entropy(lz[0], (E[:, 3] - 1).clamp(0, 3), reduction='none'), torch.zeros(m))
    return nll.numpy(), z[0].numpy()


def rule_spoof(ev, feat, mids, k=6, burst=8, cwin=12):
    """Spoof-template rule: >= k limit adds at one absolute price within `burst` events, later >= k cancels at that
    price within `cwin` events (inside the next 600) -> flag every event between. (Background unit adds at one price
    are common in this market, so the burst condition is what separates the template.)"""
    n = len(ev); flag = np.zeros(n, bool)
    bb = np.round(mids - feat[:, 20] / 2).astype(int); ba = np.round(mids + feat[:, 20] / 2).astype(int)
    price = np.where(ev[:, 1] == 0, bb - (ev[:, 2] - 1), ba + (ev[:, 2] - 1))
    price_c = np.where(ev[:, 1] == 0, bb - ev[:, 2], ba + ev[:, 2])
    adds = {}
    for i in range(n):
        if ev[i, 0] == 0:
            key = (ev[i, 1], price[i]); adds.setdefault(key, []).append(i)
    cancels = {}
    for i in range(n):
        if ev[i, 0] == 1:
            key = (ev[i, 1], price_c[i]); cancels.setdefault(key, []).append(i)
    for key, idx in adds.items():
        idx = np.array(idx)
        for j in range(len(idx) - (k - 1)):
            if idx[j + k - 1] - idx[j] <= burst:
                c = np.array(cancels.get(key, []))
                c = c[(c > idx[j + k - 1]) & (c <= idx[j + k - 1] + 600)]
                for q in range(len(c) - (k - 1)):
                    if c[q + k - 1] - c[q] <= cwin:
                        flag[idx[j]:c[q + k - 1] + 1] = True
                        break
    return flag


def auc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def windows(model, par, seeds):
    rows = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        ev, feat, mids, cm, lab = stream(par, sd, 80_000, rng)
        nll, z = surprise(model, ev, feat, mids, cm)
        rf = rule_spoof(ev, feat, mids)
        m = len(nll)
        for w0 in range(0, m - WINW + 1, WINW):
            sl = slice(w0, w0 + WINW)
            labs = lab[sl]
            fam = int(np.bincount(labs[labs > 0], minlength=4)[1:].argmax() + 1) if (labs > 0).sum() >= 3 else 0
            tail = np.sort(nll[sl])[-WINW // 10:].mean()
            zf = np.concatenate([z[sl].mean(0), z[sl].max(0), z[sl].std(0), [tail, nll[sl].mean()]])
            rows.append((fam, tail, rf[sl].mean(), zf))
    return rows


def logit_fit(X, y, l2=1.0, iters=40):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xa = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Xa.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xa @ w, -30, 30)))
        g = Xa.T @ (p - y) + l2 * w
        H = (Xa * (p * (1 - p))[:, None]).T @ Xa + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return lambda Z: np.hstack([(Z - mu) / sd, np.ones((len(Z), 1))]) @ w


def main():
    ck = torch.load(sys.argv[1], weights_only=False)
    out = sys.argv[2]; ns = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    assert not os.path.exists(out)
    model, _ = make_model(ck['arm'], cfg=ck.get('cfg')); model.load_state_dict(ck['state']); model.eval()
    par = make(ck['theta'])
    tr = windows(model, par, [S.DEV_BASE + 200 + i for i in range(ns)])     # probe-fitting streams
    te = windows(model, par, [S.DEV_BASE + 300 + i for i in range(ns)])     # disjoint evaluation streams
    ftr = np.array([r[0] for r in tr]); fte = np.array([r[0] for r in te])
    Ztr = np.stack([r[3] for r in tr]); Zte = np.stack([r[3] for r in te])
    res = dict(ckpt=os.path.basename(sys.argv[1]), split='dev', n_train_windows=len(tr), n_test_windows=len(te),
               protocol='leave-one-family-out probe on frozen hidden state; rule = spoof template', families={})
    for f, name in enumerate(FAM, 1):
        trm = ftr != f                                            # held-out family never seen by the probe
        probe = logit_fit(Ztr[trm], (ftr[trm] > 0).astype(float))
        probe_in = logit_fit(Ztr, (ftr == f).astype(float))      # ceiling: family seen in training (disjoint streams)
        sel = (fte == f) | (fte == 0)
        y = (fte[sel] == f).astype(int)
        k = max(1, int(0.02 * sel.sum()))
        d = {}
        for det, sc in (('probe_heldout_family', probe(Zte[sel])), ('probe_in_family', probe_in(Zte[sel])), ('rule_spoof', np.array([r[2] for r in te])[sel]),
                        ('model_tail_surprise', np.array([r[1] for r in te])[sel])):
            top = np.argsort(-sc)[:k]
            d[det] = dict(auc=auc(y, sc), precision_at_2pct=float(y[top].mean()))
        d['n_pos'] = int(y.sum())
        res['families'][name] = d
        print(name, {k_: (round(v['auc'], 3), round(v['precision_at_2pct'], 3)) for k_, v in d.items() if k_ != 'n_pos'},
              'n_pos', d['n_pos'], flush=True)
    json.dump(res, open(out, 'x'), indent=1)


if __name__ == '__main__':
    main()
