"""E3: real-data (Coinbase) v2 model training + Track B1 likelihood benchmark vs baselines.

usage:
  e3_real.py train  <cache_dir> <split.json> <arm> <out.pt> [passes]
  e3_real.py b1     <cache_dir> <split.json> <model.pt> <out.json> <select|report>
  e3_real.py smoke  <raw_root>                       (dev smoke on the LIVE partial hour; prints only, never saved)
Rows = contiguous per (product, segment) event streams restricted to the split's hours, cut into chunks; recurrent
state carried within a chunk (zero at chunk start) - identical at train and eval.
"""
import sys, os, json, glob, time
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest.models import lmm as L
from almagest.data import coinbase_tokenize as T

torch.set_num_threads(4)
CHUNK = 32768
WIN = 256
FIELDS = ('typ', 'side', 'lvl', 'size', 'dt')


def arm_model(arm):
    kw = dict(nsize=T.NSB, ndt=T.NDT, size_all=True)
    if arm == 'short':
        return L.LMM(mode='learned', **kw), 512
    if arm == 'modebank':
        return L.LMM(mode='modebank', **kw), None
    if arm == 'modebank_sb':
        return L.LMM(mode='modebank', slowbank=True, **kw), None
    raise ValueError(arm)


def inputs_from_tokens(ev, feat, mids, cm):
    """Model inputs aligned exactly as in synthetic training: input t = (event t-1, book before t, dmid t, sb t)."""
    prev = np.vstack([[0, 0, 1, 0, 0], ev[:-1]])
    dmid = np.concatenate([[0.0], np.diff(mids)]).astype(np.float32)
    lots = np.power(2.0, prev[:, 3])
    sv = np.where(prev[:, 0] == 2, np.where(prev[:, 1] == 0, 1, -1) * lots, 0.0)
    fv, dev = L.slowbank_offline(sv, dmid.astype(float))
    sb = L.sb_scale(torch.tensor(fv, dtype=torch.float32), torch.tensor(dev, dtype=torch.float32)).numpy()
    return dict(prev=prev, feat=feat.astype(np.float32), dmid=dmid, sb=sb, tgt=ev, cm=cm)


def load_rows(cache_dir, hours):
    hours = set(hours)
    rows = []
    for f in sorted(glob.glob(os.path.join(cache_dir, '*.npz'))):
        z = np.load(f)
        keep = np.isin(z['hours'], list(hours))
        if keep.sum() < WIN * 2:
            continue
        # contiguous runs only (a split hour gap breaks continuity -> separate rows)
        idx = np.nonzero(keep)[0]
        breaks = np.nonzero(np.diff(idx) > 1)[0]
        for run in np.split(idx, breaks + 1):
            if len(run) < WIN * 2:
                continue
            d = inputs_from_tokens(z['ev'][run], z['feat'][run], z['mids'][run], z['cm'][run])
            for s in range(0, len(run) - WIN + 1, CHUNK):
                e = min(len(run), s + CHUNK)
                n = ((e - s) // WIN) * WIN
                if n >= WIN:
                    rows.append({k: v[s:s + n] for k, v in d.items()})
    return rows


def batches(rows, bsz):
    """Group rows of equal length into batches (pad-free: truncate to the batch's min length, multiple of WIN)."""
    rows = sorted(rows, key=lambda r: -len(r['tgt']))
    for i in range(0, len(rows), bsz):
        grp = rows[i:i + bsz]
        n = min(len(r['tgt']) for r in grp)
        yield {k: torch.tensor(np.stack([r[k][:n] for r in grp])) for k in grp[0]}


def run(model, reset, data, opt=None, parts=False):
    B, N = data['tgt'].shape[:2]
    state = model.init_state(B)
    tot, cnt, acc = 0.0, 0, []
    for s in range(0, N - WIN + 1, WIN):
        if reset is not None and s % reset == 0:
            state = model.init_state(B)
        sl = slice(s, s + WIN)
        x = model.embed(data['prev'][:, sl], data['feat'][:, sl], data['dmid'][:, sl], data['sb'][:, sl])
        z, state = model.trunk_seq(x, state)
        if parts:
            acc.append(model.loss_parts(z, data['tgt'][:, sl], data['cm'][:, sl]).detach())
            continue
        loss = model.loss(z, data['tgt'][:, sl], data['cm'][:, sl])
        if opt is not None:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        state = [h.detach() for h in state]
        tot += loss.item(); cnt += 1
    return torch.cat(acc) if parts else tot / max(cnt, 1)


# ------------------------------------------------------------------ baselines (fit on fit split only)
def ctx_key(prev, feat):
    spread = np.minimum(feat[:, 20], 4).astype(int)
    imb = (feat[:, 0] - feat[:, 10]) / (feat[:, 0] + feat[:, 10] + 1e-9)
    ib = np.digitize(imb, [-0.6, -0.2, 0.2, 0.6])
    return prev[:, 0] * 25 + spread * 5 + ib       # previous type x spread x top-of-book imbalance


class TableModel:
    """Empirical factorised model p(typ|c) p(side|typ,c) p(lvl|typ,side,c) p(size|typ,side,c) p(dt|typ,c), Laplace a=1.
    c = context key (None -> unconditional marginal)."""
    def __init__(self, conditional):
        self.cond = conditional

    def _k(self, prev, feat):
        return ctx_key(prev, feat) if self.cond else np.zeros(len(prev), int)

    def fit(self, rows):
        from collections import defaultdict
        self.c = [defaultdict(lambda: None) for _ in range(5)]
        self.cnt = [dict() for _ in range(5)]
        for r in rows:
            k = self._k(r['prev'], r['feat'])
            t = r['tgt']
            keys = [k, k * 3 + t[:, 0], (k * 3 + t[:, 0]) * 2 + t[:, 1], (k * 3 + t[:, 0]) * 2 + t[:, 1],
                    k * 3 + t[:, 0]]
            for f, (kk, nv) in enumerate(zip(keys, [3, 2, L.NLVL, T.NSB, T.NDT])):
                for a, b in zip(kk, t[:, f]):
                    arr = self.cnt[f].get(a)
                    if arr is None:
                        arr = self.cnt[f][a] = np.zeros(nv)
                    arr[b] += 1
        self.nv = [3, 2, L.NLVL, T.NSB, T.NDT]
        return self

    def nll(self, r):
        k = self._k(r['prev'], r['feat'])
        t = r['tgt']
        keys = [k, k * 3 + t[:, 0], (k * 3 + t[:, 0]) * 2 + t[:, 1], (k * 3 + t[:, 0]) * 2 + t[:, 1], k * 3 + t[:, 0]]
        out = np.zeros((len(t), 5))
        for f in range(5):
            for i, (a, b) in enumerate(zip(keys[f], t[:, f])):
                if f == 2 and t[i, 0] == 2:
                    continue
                arr = self.cnt[f].get(a)
                tot = (0 if arr is None else arr.sum()) + self.nv[f]
                out[i, f] = -np.log(((0 if arr is None else arr[b]) + 1.0) / tot)
        return out


def b1(cache_dir, split, model_path, out_path, which):
    assert which in ('select', 'report')
    assert not os.path.exists(out_path)
    fit_rows = load_rows(cache_dir, split['fit'])
    ev_rows = load_rows(cache_dir, split[which])
    ck = torch.load(model_path, weights_only=False)
    model, reset = arm_model(ck['arm'])
    model.load_state_dict(ck['state']); model.eval()
    res = {}
    with torch.no_grad():
        per = [run(model, reset, {k: torch.tensor(r[k][None]) for k in r}, parts=True).numpy() for r in ev_rows]
    # align: model parts cover WIN-multiples from row start
    base = {}
    for name, cond in (('marginal', False), ('table_ctx', True)):
        tm = TableModel(cond).fit(fit_rows)
        base[name] = [tm.nll({k: v[:len(p)] for k, v in r.items()}) for r, p in zip(ev_rows, per)]
    rng = np.random.default_rng(0)

    def summary(arrs):
        tot = np.concatenate([a.sum(1) for a in arrs])
        return float(tot.mean())
    m = np.concatenate([p.sum(1) for p in per])
    res['n_events'] = int(len(m))
    res['model_nll'] = float(m.mean())
    res['model_fields'] = dict(zip(FIELDS, np.concatenate(per).mean(0).tolist()))
    for name, arrs in base.items():
        b = np.concatenate([a.sum(1) for a in arrs])
        dif = m - b
        # block bootstrap over 4096-event blocks (serial dependence)
        nb = max(1, len(dif) // 4096)
        blocks = np.array_split(dif, nb)
        bm = [np.mean(np.concatenate([blocks[i] for i in rng.integers(0, nb, nb)])) for _ in range(300)]
        res[name] = dict(nll=float(b.mean()), diff=float(dif.mean()), diff_ci95=list(np.quantile(bm, [0.025, 0.975])),
                         fields=dict(zip(FIELDS, np.concatenate(arrs).mean(0).tolist())))
    res.update(split=split, which=which, arm=ck['arm'], params=model.n_params())
    json.dump(res, open(out_path, 'x'), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != 'split'}, default=float)[:1500])


def train(cache_dir, split, arm, out, passes=4):
    rows = load_rows(cache_dir, split['fit'])
    vrows = load_rows(cache_dir, split['select'])
    print('fit rows', len(rows), 'events', sum(len(r['tgt']) for r in rows), 'select events',
          sum(len(r['tgt']) for r in vrows), flush=True)
    torch.manual_seed(0)
    model, reset = arm_model(arm)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    hist = []
    t0 = time.time()
    for p in range(passes):
        model.train()
        trl = np.mean([run(model, reset, b, opt) for b in batches(rows, 16)])
        model.eval()
        with torch.no_grad():
            val = np.mean([run(model, reset, b) for b in batches(vrows, 16)]) if vrows else float('nan')
        hist.append((float(trl), float(val)))
        print(f'pass {p} fit {trl:.4f} select {val:.4f} t {time.time()-t0:.0f}s', flush=True)
    torch.save(dict(state=model.state_dict(), cfg=model.cfg, arm=arm, hist=hist, split=split, version='v2-real'), out)


def smoke(raw_root):
    """Dev smoke: tokenise the live partial hour, train 1 short pass, print NLLs. Never saved, never reportable."""
    rows = []
    for p in ('BTC-USD', 'ETH-USD'):
        segs = T.segments(raw_root, p)
        o = T.tokenize_segment(segs[0], p)
        n = (min(len(o['ev']), 60000) // WIN) * WIN
        d = inputs_from_tokens(o['ev'][:n], o['feat'][:n], o['mids'][:n], o['cm'][:n])
        rows.append(d)
    model, reset = arm_model('modebank_sb')
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    half = {k: torch.tensor(np.stack([r[k][:WIN * 160] for r in rows])) for k in rows[0]}
    held = {k: torch.tensor(np.stack([r[k][WIN * 160:WIN * 200] for r in rows])) for k in rows[0]}
    t0 = time.time()
    trl = run(model, reset, half, opt)
    with torch.no_grad():
        v = run(model, reset, held)
    tm = TableModel(True).fit([{k: v[:WIN * 160] for k, v in r.items()} for r in rows])
    tb = np.mean([tm.nll({k: v[WIN * 160:WIN * 200] for k, v in r.items()}).sum(1).mean() for r in rows])
    print(f'[SMOKE, not reportable] params {model.n_params()} fit {trl:.3f} held {v:.3f} table_ctx {tb:.3f} '
          f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'smoke':
        smoke(sys.argv[2])
    elif cmd == 'train':
        train(sys.argv[2], json.load(open(sys.argv[3])), sys.argv[4], sys.argv[5],
              int(sys.argv[6]) if len(sys.argv) > 6 else 4)
    elif cmd == 'b1':
        b1(sys.argv[2], json.load(open(sys.argv[3])), sys.argv[4], sys.argv[5], sys.argv[6])
