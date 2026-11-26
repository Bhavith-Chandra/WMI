import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from data.l3 import *  # noqa
from data.tokenizer import Tokenizer
from sim.book import LevelBook
from sim.rollout import Rollout, context_from_history
from evaluate import load_model
torch.set_num_threads(2)
run, pair, tick, H, P = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
root = Path('/home/user/data/bitstamp'); hours = hour_dirs(root); cut = int(open('/home/user/data/dev_cut_ns.txt').read())
conn = [e for e in load_conn(hours) if e['t_ns'] < cut]
msgs = [x for x in pair_messages(hours, pair) if x[0] < cut]
c = [e['t_ns'] for e in conn if e['ev'] == 'connect'][1]
snaps = sorted((e for e in conn if e['ev'] == 'snapshot' and e['pair'] == pair), key=lambda e: e['us'])
a = [e for e in snaps if c + 1e9 <= e['us'] * 1000][0]
r = rebuild(pair, load_snapshot(hours, pair, a['us']), int(a['us']), [x for x in msgs if c <= x[0] < c + 20 * 60e9], [], hours, 1,
            tick=tick, want_state=True, ckpt_every_us=30_000_000)
ev, st = r['events'], r['state']; ck = r['ckpts'][10]; i = ck[0]
b = LevelBook(tick); b.bids = {int(round(p / tick)): q for p, q in ck[2]}; b.asks = {int(round(p / tick)): q for p, q in ck[3]}
m, h = load_model(run); tok = Tokenizer.from_json(json.load(open(f'{run}/tokenizer.json')))
ctx = context_from_history(ev[:i], st[:i], tok, b, m.cfg['ctx'])
R = Rollout(m, tok, ctx, n_paths=P, seed=0)
import cProfile, pstats
t = time.time(); pr = cProfile.Profile(); pr.enable(); R.run(H, max_steps=400); pr.disable(); el = time.time() - t
pstats.Stats(pr).sort_stats("tottime").print_stats(14)
n = np.mean([p['n_ev'] for p in R.paths])
print(f'{pair}: {P} paths, {H}s horizon: {R.n_steps} steps, {el:.1f}s wall, {1000*el/max(R.n_steps,1):.2f} ms/step, {n:.0f} ev/path, alive {sum(p["alive"] for p in R.paths)}')
g = R.mid_grid(np.array([1, 2, H]))
mid0 = 0.5 * (st[i, 0] + st[i, 1])
print('mid0', mid0, 'sim ret bp q10/50/90 at 1,2,H s:', np.round((np.nanpercentile(g, [10, 50, 90], axis=0) / mid0 - 1) * 1e4, 2).tolist())
rt = (ev['us'][i:] - ev['us'][i]) / 1e6
print('real events in horizon:', int((rt <= H).sum()), ' real mid ret at H (bp):', round((0.5*(st[i+int((rt<=H).sum()),0]+st[i+int((rt<=H).sum()),1])/mid0-1)*1e4, 2))
