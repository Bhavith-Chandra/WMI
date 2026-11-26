"""Realism post-training on REAL data (no oracle exists): tune generation-time calibration scalars on FIT-hour windows
so closed-loop rollouts match the real continuation; select/report hours are never used here.
usage: e3_tune_real.py <cache_dir> <split.json> <model.pt> <out.pt>
Objective = |acf1_gen - acf1_real| + |log market-share ratio| + |log vol100 ratio| + 0.5 spread_L1 + 0.3 depth_W1
"""
import sys, os, json, itertools
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest.experiments.e3_real import load_rows, arm_model, WIN
from almagest.experiments import e3_realism as R

torch.set_num_threads(1)


def objective(c):
    return (abs(c['acf1'][0] - c['acf1'][1]) + abs(np.log(max(c['mshare_ratio'], 1e-6)))
            + abs(np.log(max(c['vol100_ratio'], 1e-6))) + 0.5 * c['spread_L1'] + 0.3 * c['depth_W1'])


def main():
    cache, split, mp, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4]
    ck = torch.load(mp, weights_only=False)
    model, _ = arm_model(ck['arm'], cfg=ck.get('cfg')); model.load_state_dict(ck['state']); model.eval()
    U = ck['arm'] == 'unified'
    rows = [r for r in load_rows(cache, split['fit'], unified=U) if len(r['tgt']) >= 4 * WIN + 4000]
    picks = R.pick_windows(rows, 16, seed=11)
    batch, _ = R.setup(model, rows, picks)
    log = []
    for st, mb in itertools.product((0.6, 0.8, 1.0), (-0.6, -0.3, 0.0)):
        model.gen_cal.update(side_temp=st, mkt_bias=mb)
        g, fl = R.b2(model, rows, picks, batch, gen_events=2000)
        L = objective(g)
        log.append(dict(side_temp=st, mkt_bias=mb, objective=L, acf1=g['acf1'], mshare_ratio=g['mshare_ratio'],
                        vol100_ratio=g['vol100_ratio'], spread_L1=g['spread_L1'], depth_W1=g['depth_W1'],
                        floor_objective=objective(fl)))
        print(json.dumps(log[-1]), flush=True)
    best = min(log, key=lambda x: x['objective'])
    cfg = dict(ck['cfg']); cfg['gen_cal'] = dict(side_temp=best['side_temp'], mkt_bias=best['mkt_bias'], typ_temp=1.0)
    c = dict(ck); c['cfg'] = cfg
    c['post_real'] = dict(method='grid on FIT-hour closed-loop realism', log=log, chosen=cfg['gen_cal'])
    torch.save(c, out)
    print('chosen', cfg['gen_cal'], flush=True)


if __name__ == '__main__':
    main()
