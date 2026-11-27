"""Choose the market-side sampling temperature on FIT windows (n=128) with the pre-registered B2 gate; select/report
hours never read. usage: r3_mtemp.py <cache> <split.json> <model.pt> <out.pt>"""
import sys, os, json
import numpy as np, torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest.experiments import r3
from almagest.experiments.r3_gate import gate
from almagest.experiments.r3_tune import score

torch.set_num_threads(1)
cache, split, mp, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4]
ck = torch.load(mp, weights_only=False)
model = r3.arm_model(ck['cfg']); model.load_state_dict(ck['state']); model.eval()
rows = r3.load_rows(cache, split['fit'], with_meta=True)
gen = 4000
picks = r3.pick_windows(rows, 128, gen, seed=909)
batch, _ = r3.setup(model, rows, picks, ck['lot_table'])
floor = []
for i, s in picks:
    r = rows[i]; a, b = slice(s, s + gen), slice(s + gen, s + 2 * gen)
    floor.append(r3.compare(r3.stats_from(r['tgt'][b], r['feat'][b], np.cumsum(r['dmid'][b])),
                            r3.stats_from(r['tgt'][a], r['feat'][a], np.cumsum(r['dmid'][a]))))
floor = r3.agg(floor)
log = []
for t in (1.0, 0.9, 0.8, 0.7):
    model.gen_cal['mkt_side_temp'] = t
    ge, gf, gm = r3.b2_run(model, rows, picks, batch, gen)
    comps = [r3.compare(r3.stats_from(ge[w], gf[w], gm[w, 1:]),
                        r3.stats_from(rows[i]['tgt'][s:s + gen], rows[i]['feat'][s:s + gen], np.cumsum(rows[i]['dmid'][s:s + gen])))
             for w, (i, s) in enumerate(picks)]
    res = dict(model_vs_real=r3.agg(comps), floor_real_vs_real=floor)
    g = gate(res)
    log.append(dict(mkt_side_temp=t, score=score(res), n_pass=int(sum(x[-1] for x in g)),
                    acf1=res['model_vs_real']['acf1'], fails=[x[0] for x in g if not x[-1]]))
    print(json.dumps(log[-1]), flush=True)
best = min(log, key=lambda x: (-x['n_pass'], x['score']))
c = dict(ck); cfg = dict(ck['cfg']); gc = dict(cfg.get('gen_cal') or {}); gc['mkt_side_temp'] = best['mkt_side_temp']
cfg['gen_cal'] = gc; c['cfg'] = cfg; c['post_mtemp'] = dict(log=log, chosen=best['mkt_side_temp'], windows='fit n=128 seed 909')
torch.save(c, out); print('chosen', best['mkt_side_temp'], flush=True)
