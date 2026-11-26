"""Closed-loop stability diagnostic: N independent free runs of a checkpoint (contexts from DIAG_BASE seeds);
reports the fraction that reach a one-sided book (top-10 depth <= 1 lot on a side) within `steps` events, and when.
usage: e2_collapse.py <ckpt.pt> [runs] [steps]"""
import sys, os, json
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest import splits as S
from almagest.app.live import LiveModel
torch.set_num_threads(1)
ck = sys.argv[1]; runs = int(sys.argv[2]) if len(sys.argv) > 2 else 12; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
res = []
for s in range(runs):
    lm = LiveModel(ck, context_seed=S.DIAG_BASE + s)
    tc = None
    for k in range(steps // 250):
        lm.step(250, seed=S.DIAG_BASE + s * 1000 + k)
        b = lm.book()
        if min(sum(x[1] for x in b['asks']), sum(x[1] for x in b['bids'])) <= 1:
            tc = (k + 1) * 250
            break
    res.append(tc)
rate = float(np.mean([r is not None for r in res]))
print(json.dumps(dict(ckpt=os.path.basename(ck), runs=runs, steps=steps, collapse_rate=rate, t_collapse=res)))
