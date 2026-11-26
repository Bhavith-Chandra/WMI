"""Track B4: mid-price direction forecasting from the frozen foundation model (linear probe on hidden state) vs
baselines. Fit on FIT hours, select nothing, score on REPORT (or SELECT for dev) hours. AUC with block-bootstrap CI.

usage: e3_forecast.py <cache_dir> <split.json> <model.pt> <out.json> <select|report>
Targets: sign(mid[t+H] - mid[t]) for H in {50, 200, 1000} events, rows with no change are dropped (stated).
Baselines (all fit on FIT): top-of-book imbalance logistic; EWMA(dmid) momentum logistic; book-feature logistic.
"""
import sys, os, json
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest.experiments.e3_real import load_rows, arm_model, WIN

torch.set_num_threads(2)
HS = (50, 200, 1000)


def hidden(model, reset, rows):
    out = []
    with torch.no_grad():
        for r in rows:
            d = {k: torch.tensor(r[k][None]) for k in r}
            B, N = d['tgt'].shape[:2]
            st = model.init_state(1)
            zs = []
            for s in range(0, N - WIN + 1, WIN):
                if reset and s % reset == 0:
                    st = model.init_state(1)
                sl = slice(s, s + WIN)
                x = model.embed(d['prev'][:, sl], d['feat'][:, sl], d['dmid'][:, sl], d['sb'][:, sl])
                z, st = model.trunk_seq(x, st)
                zs.append(z[0].numpy())
            out.append(np.concatenate(zs))
    return out


def feats_base(r):
    f = r['feat']
    imb = (f[:, 0] - f[:, 10]) / (f[:, 0] + f[:, 10] + 1e-9)
    imb5 = (f[:, :5].sum(1) - f[:, 10:15].sum(1)) / (f[:, :5].sum(1) + f[:, 10:15].sum(1) + 1e-9)
    dm = r['dmid'].astype(float)
    ew = np.zeros((len(dm), 3))
    for j, a in enumerate((0.1, 0.01, 0.001)):
        e = 0.0
        for i in range(len(dm)):
            e = (1 - a) * e + a * dm[i]
            ew[i, j] = e
    return dict(imb=np.stack([imb, imb5], 1), ewma=ew, book=np.concatenate([np.log1p(f[:, :20]), f[:, 20:21]], 1))


def targets(r, H):
    m = np.cumsum(r['dmid'].astype(float))
    y = np.full(len(m), np.nan)
    y[:-H] = np.sign(m[H:] - m[:-H])
    return y


def logit_fit(X, y, l2=1e-2, iters=30):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xa = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Xa.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xa @ w, -30, 30)))
        g = Xa.T @ (p - y) + l2 * w
        Hm = (Xa * (p * (1 - p))[:, None]).T @ Xa + l2 * np.eye(len(w))
        w -= np.linalg.solve(Hm, g)
    return lambda Z: 1 / (1 + np.exp(-np.clip(np.hstack([(Z - mu) / sd, np.ones((len(Z), 1))]) @ w, -30, 30)))


def auc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def main():
    cache, split, mp, out, which = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4], sys.argv[5]
    assert which in ('select', 'report') and not os.path.exists(out)
    ck = torch.load(mp, weights_only=False)
    model, reset = arm_model(ck['arm'], cfg=ck.get('cfg'))
    model.load_state_dict(ck['state']); model.eval()
    U = ck['arm'] == 'unified'
    fit = load_rows(cache, split['fit'], unified=U)
    ev = load_rows(cache, split[which], unified=U)
    zf, ze = hidden(model, reset, fit), hidden(model, reset, ev)
    bf = [feats_base(r) for r in fit]; be = [feats_base(r) for r in ev]
    res = dict(which=which, split=split, arm=ck['arm'], params=model.n_params(), horizons={})
    rng = np.random.default_rng(0)
    for H in HS:
        yf = [targets(r, H)[:len(z)] for r, z in zip(fit, zf)]; ye = [targets(r, H)[:len(z)] for r, z in zip(ev, ze)]
        def stack(parts, ys):
            X = np.concatenate([p[:len(y)] for p, y in zip(parts, ys)]); Y = np.concatenate(ys)
            k = np.isfinite(Y) & (Y != 0)
            return X[k], (Y[k] > 0).astype(float)
        sub = slice(None, None, 4)        # thin overlapping targets for fitting speed (stated)
        Rh = {}
        cands = {'model_probe': (zf, ze), 'imbalance': ([b['imb'] for b in bf], [b['imb'] for b in be]),
                 'ewma_momentum': ([b['ewma'] for b in bf], [b['ewma'] for b in be]),
                 'book_logistic': ([b['book'] for b in bf], [b['book'] for b in be])}
        for name, (pf, pe) in cands.items():
            Xf, Yf = stack(pf, yf); Xe, Ye = stack(pe, ye)
            f = logit_fit(Xf[sub], Yf[sub])
            p = f(Xe)
            nb = max(1, len(Ye) // 4096)
            idx = np.array_split(np.arange(len(Ye)), nb)
            bs = []
            for _ in range(200):
                ii = np.concatenate([idx[i] for i in rng.integers(0, nb, nb)])
                bs.append(auc(Ye[ii], p[ii]))
            Rh[name] = dict(auc=auc(Ye, p), auc_ci95=list(np.quantile(bs, [0.025, 0.975])), n=int(len(Ye)))
        res['horizons'][H] = Rh
        print(H, {k: round(v['auc'], 4) for k, v in Rh.items()}, flush=True)
    json.dump(res, open(out, 'x'), indent=1, default=float)


if __name__ == '__main__':
    main()
