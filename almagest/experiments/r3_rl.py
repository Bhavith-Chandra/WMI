"""Closed-loop drift fix: sequence-level fine-tuning on FIT hours (no oracle needed).

Teacher forcing never shows the model its own states, so tiny per-event biases compound (e.g. limit share +0.025 ->
top depth x2 over 3000 events -> fewer level clears -> volatility x0.5). Here the generator is rolled out from real FIT
contexts (exact seed book + teacher-forced state) and trained with REINFORCE on a rollout-level realism reward computed
against the REAL continuation of the same window, with K paired rollouts per window (leave-one-out baseline), plus the
teacher-forced NLL on FIT data as an anchor. Select/report hours are never read.
usage: r3_rl.py <cache_v3> <split.json> <model.pt> <out_prefix> [iters]
Checkpoints every EVAL_EVERY iters are scored on fixed held-out FIT windows (different seed) with the B2 gate score.
"""
import sys, os, json, time
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest.experiments import r3
from almagest.experiments.r3_tune import score as gate_score
from almagest.models import lmm as L
from almagest.data import coinbase_exact as X

T_ROLL = int(os.environ.get('T_ROLL', 500))
POOL = int(os.environ.get('POOL', 48))
REFRESH = int(os.environ.get('REFRESH', 15))
NWIN = int(os.environ.get('NWIN', 12))
K = int(os.environ.get('K', 3))
LR = float(os.environ.get('LR', 1e-4))
LAM_NLL = float(os.environ.get('LAM_NLL', 1.0))
EVAL_EVERY = int(os.environ.get('EVAL_EVERY', 10))


def window_stats(ev, feat, mid):
    """Rollout-level summary used by the reward (per window)."""
    dep = np.log1p(feat[:, 0] + feat[:, 10]).mean()
    band = np.log1p(feat[:, 21:27].sum(1)).mean()
    spread = feat[:, 20].mean()
    mk = ev[:, 0] == 2
    msh = mk.mean()
    s = np.where(ev[mk, 1] == 0, 1.0, -1.0)
    acf = float(np.mean(s[:-1] * s[1:])) if len(s) > 6 else 0.0
    dm = mid[100:] - mid[:-100]
    vol = float(np.var(dm)) if len(dm) > 5 else 0.0
    lim = (ev[:, 0] == 0).mean()
    return np.array([dep, band, spread, msh, acf, vol, lim])


def reward(g, r):
    return -(abs(g[0] - r[0]) + abs(g[1] - r[1]) + 0.5 * abs(g[2] - r[2])
             + abs(np.log((g[3] + 1e-3) / (r[3] + 1e-3))) + abs(g[4] - r[4])
             + 0.5 * min(3.0, abs(np.log((g[5] + 1e-3) / (r[5] + 1e-3)))) + 10 * abs(g[6] - r[6]))


def rollout_logp(model, g, T, gen):
    """Closed-loop rollout on Gen3 `g` WITH gradients through the model; returns per-row summed log-prob, events,
    features, mids (numpy)."""
    lp = torch.zeros(g.B)
    evs, feats, mids = [], [], [g.mid()]
    for t in range(T):
        feats.append(g.feat.copy())
        sbf = torch.cat([g.sb.feat(), g.fb.feat()], -1) if g.use_fb else g.sb.feat()
        x = model.embed(g.prev, g.pfeat, g.pdmid, sbf)
        z, g.state = model.trunk_step(x, g.state)
        u = torch.rand(g.B, 6, generator=gen)
        lt = model.gen_typ_logits(z)
        typ = L.inv_cdf(lt.detach(), u[:, 0])
        ls = model.gen_side_logits(z, typ)
        sd = L.inv_cdf(ls.detach(), u[:, 1])
        ll, lz = model.lvl_size_logits(z, typ, sd)
        cm = torch.stack([torch.tensor(g.cmb), torch.tensor(g.cma)], 1)
        sp = torch.tensor(g.feat[:, 20], dtype=torch.long)
        llm = L.mask_lvl(ll, typ, sd, cm, sp, model.nimp)
        lvl = L.inv_cdf(llm.detach(), u[:, 2])
        lvl = torch.where(typ == 2, torch.zeros_like(lvl), lvl)
        qv = g.queue(torch.stack([typ, sd, lvl, lvl, lvl], 1).numpy().astype(np.int64))
        if model.size_cond:
            lz = model.size_logits(z, typ, sd, lvl, torch.tensor(np.log1p(qv), dtype=torch.float32))
        szb = L.inv_cdf(lz.detach(), u[:, 3])
        zl = z + model.c_side(typ) + model.c_lvl(typ * 2 + sd)
        ld = model.h_dt(zl + model.c_size(szb))
        dtb = L.inv_cdf(ld.detach(), u[:, 4])
        nm = (typ != 2).float()
        step = (torch.log_softmax(lt, -1).gather(1, typ[:, None])[:, 0]
                + torch.log_softmax(ls, -1).gather(1, sd[:, None])[:, 0]
                + nm * torch.log_softmax(llm, -1).gather(1, lvl[:, None])[:, 0]
                + torch.log_softmax(lz, -1).gather(1, szb[:, None])[:, 0]
                + torch.log_softmax(ld, -1).gather(1, dtb[:, None])[:, 0])
        lp = lp + step
        evn = torch.stack([typ, sd, lvl, szb, dtb], 1).numpy().astype(np.int64)
        lots = r3.draw_lots(g.tab, evn, qv, u[:, 5].numpy())
        g.apply(evn, lots)
        g.state = [h for h in g.state]
        evs.append(evn); mids.append(g.mid())
    return lp, np.stack(evs, 1), np.stack(feats, 1), np.stack(mids, 1)


def eval_fit(model, rows, picks, batch, tab, gen=2000):
    model.eval()
    ge, gf, gm = r3.b2_run(model, rows, picks, batch, gen)
    comps, floor = [], []
    for w, (i, s) in enumerate(picks):
        r = rows[i]; a, b = slice(s, s + gen), slice(s + gen, s + 2 * gen)
        real = r3.stats_from(r['tgt'][a], r['feat'][a], np.cumsum(r['dmid'][a]))
        comps.append(r3.compare(r3.stats_from(ge[w], gf[w], gm[w, 1:]), real))
        floor.append(r3.compare(r3.stats_from(r['tgt'][b], r['feat'][b], np.cumsum(r['dmid'][b])), real))
    res = dict(model_vs_real=r3.agg(comps), floor_real_vs_real=r3.agg(floor))
    from almagest.experiments.r3_gate import gate
    return gate_score(res), int(sum(x[-1] for x in gate(res))), res


def main():
    cache, split, mp, outp = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4]
    iters = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 4)))
    ck = torch.load(mp, weights_only=False)
    model = r3.arm_model(ck['cfg']); model.load_state_dict(ck['state'])
    tab = ck['lot_table']
    rows = r3.load_rows(cache, split['fit'], with_meta=True)
    nll_batches = list(r3.batches(rows, 8))
    print('loaded rows', len(rows), flush=True)
    # held-out FIT evaluation windows (fixed, disjoint seed from training draws)
    vpicks = r3.pick_windows(rows, 16, 2000, seed=777)
    tt = time.time(); vbatch, _ = r3.setup(model, rows, vpicks, tab); print('val setup s', round(time.time() - tt), flush=True)
    tt = time.time(); s0, n0, _ = eval_fit(model, rows, vpicks, vbatch, tab); print('val eval s', round(time.time() - tt), flush=True)
    log = [dict(it=0, fit_val_score=s0, fit_val_pass=n0)]
    print(json.dumps(log[-1]), flush=True)
    best = (s0, 0); torch.save(dict(ck), outp + '_it0.pt')
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(1)
    t0 = time.time()
    for it in range(1, iters + 1):
        model.eval()   # no dropout; batchnorm-free -> eval mode is fine for gradients
        if (it - 1) % REFRESH == 0:   # pool of FIT windows; context states recomputed under the current model
            pool = r3.pick_windows(rows, POOL, T_ROLL, seed=int(rng.integers(1 << 30)))
            pbatch, _ = r3.setup(model, rows, pool, tab)
        sel = rng.choice(POOL, NWIN, replace=False)
        picks = [pool[j] for j in sel]
        g, idx = pbatch(K, sel)
        idx = np.repeat(np.arange(NWIN), K)
        lp, ev, fe, mi = rollout_logp(model, g, T_ROLL, torch.Generator().manual_seed(it))
        R = np.zeros(g.B)
        for b in range(g.B):
            i, s = picks[idx[b]]; r = rows[i]; sl = slice(s, s + T_ROLL)
            R[b] = reward(window_stats(ev[b], fe[b], mi[b, 1:]),
                          window_stats(r['tgt'][sl], r['feat'][sl], np.cumsum(r['dmid'][sl])))
        Rg = R.reshape(NWIN, K)
        adv = (Rg - (Rg.sum(1, keepdims=True) - Rg) / (K - 1)).reshape(-1)     # leave-one-out baseline
        adv = adv / (adv.std() + 1e-6)
        pg = -(torch.tensor(adv, dtype=torch.float32) * lp).mean() / T_ROLL
        # teacher-forced anchor on a FIT chunk (first 8 windows of 256)
        bt = nll_batches[it % len(nll_batches)]
        sub = {k: v[:, :2048] for k, v in bt.items()}
        nll = r3.run(model, sub, None) if False else _nll(model, sub)
        loss = pg + LAM_NLL * nll
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        rec = dict(it=it, reward=float(R.mean()), nll=float(nll.item()), sec=round(time.time() - t0))
        if it % EVAL_EVERY == 0:
            vbatch, _ = r3.setup(model, rows, vpicks, tab)
            sc, npass, _ = eval_fit(model, rows, vpicks, vbatch, tab)
            rec.update(fit_val_score=sc, fit_val_pass=npass)
            c = dict(ck); c['state'] = {k: v.detach().clone() for k, v in model.state_dict().items()}
            c['post_real'] = dict(method='REINFORCE rollout realism on FIT hours + NLL anchor', it=it, log=log)
            torch.save(c, outp + f'_it{it}.pt')
            if sc < best[0]:
                best = (sc, it)
        log.append(rec)
        print(json.dumps(rec), flush=True)
    print('best fit-val', best, flush=True)
    json.dump(dict(log=log, best=best), open(outp + '_log.json', 'w'), indent=1)


def _nll(model, data):
    B, N = data['tgt'].shape[:2]
    state = model.init_state(B)
    tot = 0.0; n = 0
    for s in range(0, N - r3.WIN + 1, r3.WIN):
        sl = slice(s, s + r3.WIN)
        x = model.embed(data['prev'][:, sl], data['feat'][:, sl], data['dmid'][:, sl], r3.sbx(model, data['sb'][:, sl]))
        z, state = model.trunk_seq(x, state)
        tot = tot + model.loss(z, data['tgt'][:, sl], data['cm'][:, sl], data['spread'][:, sl], data['lq'][:, sl]); n += 1
    return tot / n


if __name__ == '__main__':
    main()
