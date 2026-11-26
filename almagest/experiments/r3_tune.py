"""FIT-hour realism calibration of generation scalars (v3). No oracle exists on real data; teacher-forced likelihood is
untouched, only sampling scalars (type temperature, market-logit bias, side temperature) are chosen, on FIT-hour
windows, scored by the pre-registered B2 gate (violation-weighted). Select/report hours are never read here.
usage: r3_tune.py <cache_v3> <split.json> <model.pt> <out.pt> [n_windows] [gen_events]
"""
import sys, os, json, itertools
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest.experiments import r3
from almagest.experiments.r3_gate import gate


def score(res):
    v = 0.0
    for k, gv, fv, thr, ok in gate(res):
        if k in ('mshare_ratio', 'vol10_ratio', 'vol100_ratio', 'vol1000_ratio'):
            dev = abs(np.log(max(gv, 1e-9)))
        elif k in ('acf1', 'acf10'):
            dev = abs(gv[0] - gv[1])
        else:
            dev = gv
        v += max(0.0, dev - thr) / max(thr, 1e-6) + 0.1 * dev / max(thr, 1e-6)
    return v


def main():
    cache, split, mp, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4]
    nwin = int(sys.argv[5]) if len(sys.argv) > 5 else 32
    gen = int(sys.argv[6]) if len(sys.argv) > 6 else 2000
    ck = torch.load(mp, weights_only=False)
    model = r3.arm_model(ck['cfg']); model.load_state_dict(ck['state']); model.eval()
    rows = r3.load_rows(cache, split['fit'], with_meta=True)
    picks = r3.pick_windows(rows, nwin, gen, seed=101)
    batch, _ = r3.setup(model, rows, picks, ck['lot_table'])
    floor = []
    for i, s in picks:
        r = rows[i]; a, b = slice(s, s + gen), slice(s + gen, s + 2 * gen)
        floor.append(r3.compare(r3.stats_from(r['tgt'][b], r['feat'][b], np.cumsum(r['dmid'][b])),
                                r3.stats_from(r['tgt'][a], r['feat'][a], np.cumsum(r['dmid'][a]))))
    floor = r3.agg(floor)
    log = []
    grid = list(itertools.product((-0.15, -0.1, -0.05, 0.0), (-0.3, -0.15, 0.0), (0.8, 1.0)))
    for lb, mb, st in grid:
        tt = 1.0
        model.gen_cal.update(typ_temp=tt, mkt_bias=mb, side_temp=st, lim_bias=lb)
        ge, gf, gm = r3.b2_run(model, rows, picks, batch, gen)
        comps = []
        for w, (i, s) in enumerate(picks):
            r = rows[i]; a = slice(s, s + gen)
            comps.append(r3.compare(r3.stats_from(ge[w], gf[w], gm[w, 1:]),
                                    r3.stats_from(r['tgt'][a], r['feat'][a], np.cumsum(r['dmid'][a]))))
        res = dict(model_vs_real=r3.agg(comps), floor_real_vs_real=floor)
        sc = score(res)
        npass = sum(g[-1] for g in gate(res))
        log.append(dict(lim_bias=lb, typ_temp=tt, mkt_bias=mb, side_temp=st, score=sc, n_pass=int(npass)))
        print(json.dumps(log[-1]), flush=True)
    best = min(log, key=lambda x: x['score'])
    cfg = dict(ck['cfg']); cfg['gen_cal'] = dict(typ_temp=best['typ_temp'], mkt_bias=best['mkt_bias'],
                                                 side_temp=best['side_temp'], lim_bias=best['lim_bias'])
    c = dict(ck); c['cfg'] = cfg
    c['post_real'] = dict(method='FIT-hour gate-scored grid over generation scalars', log=log, chosen=cfg['gen_cal'])
    torch.save(c, out)
    print('chosen', cfg['gen_cal'], flush=True)


if __name__ == '__main__':
    main()
