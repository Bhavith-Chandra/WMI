import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.l3 import *  # noqa
pair, tick = sys.argv[1], float(sys.argv[2])
root = Path('/home/user/data/bitstamp'); hours = hour_dirs(root); conn = load_conn(hours); msgs = pair_messages(hours, pair)
c = [e['t_ns'] for e in conn if e['ev'] == 'connect'][int(sys.argv[3]) if len(sys.argv) > 3 else 0]
snaps = sorted((e for e in conn if e['ev'] == 'snapshot' and e['pair'] == pair), key=lambda e: e['us'])
a = [e for e in snaps if c + 1e9 <= e['us'] * 1000][0]
dbg = []
rebuild(pair, load_snapshot(hours, pair, a['us']), int(a['us']), [x for x in msgs if x[0] >= c], [], hours, 0, tick=tick, lockstep_every=1, debug=dbg)
d = dbg[0]
print('first div us', d['us'])
for sd in (1, 0):
    I, E = dict(d['id_levels'][sd]), dict(d['ev_levels'][sd])
    print('side', sd, 'diff:', {k: (I.get(k), E.get(k)) for k in set(I) | set(E) if abs((I.get(k) or 0) - (E.get(k) or 0)) > 1e-6})
for x in d['recent'][-16:]:
    print('  ', x[0], x[1])
for x in d['rows'][-8:]:
    print('  row', x)
