import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.l3 import segments
TICK = {'btcusd': 0.01, 'ethusd': 0.01, 'xrpusd': 1e-5, 'dogeusd': 1e-6, 'linkusd': 1e-4, 'solusd': 0.01, 'ltcusd': 0.01, 'btceur': 1.0}
for pair in sys.argv[1:] or TICK:
    t = time.time()
    for s in segments(Path('/home/user/data/bitstamp'), pair, tick=TICK[pair]):
        print(pair, 'seg', s['seg'], 'ev', len(s['events']), 'lockstep', {k: v for k, v in s['lockstep'].items()},
              'verify', [(v['accept'], round(v['id_jaccard'], 4)) for v in s['verify']], f'{time.time()-t:.0f}s', flush=True)
