"""Diagnostic: model vs empirical P(next market order is buy | signed run length), teacher-forced on validation
streams (VAL_BASE). usage: e2_persist.py <ckpt.pt>"""
import sys, json
import numpy as np, torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest import splits as S
from almagest.experiments.e2_train import build_stream, make_model
from almagest.experiments.e1_calibrate_world import make
torch.set_num_threads(2)
ck = torch.load(sys.argv[1], weights_only=False)
m, _ = make_model(ck['arm'], cfg=ck.get('cfg')); m.load_state_dict(ck['state']); m.eval()
par = make(ck['theta'])
emp, mod = {}, {}
for i in range(3):
    d = build_stream(par, S.VAL_BASE + i, 60000)
    t = {k: torch.tensor(v[None]) for k, v in d.items()}
    with torch.no_grad():
        x = m.embed(t['prev'], t['feat'], t['dmid'], t['sb']); z, _ = m.trunk_seq(x, m.init_state(1))
        pb = torch.softmax(m.side_logits(z, torch.full_like(t['tgt'][..., 0], 2)), -1)[0, :, 0].numpy()
    tg = d['tgt']; run = 0; last = None
    for k in range(len(tg)):
        if tg[k, 0] == 2:
            key = 0 if last is None else min(run, 8) * (1 if last == 0 else -1)
            emp.setdefault(key, []).append(float(tg[k, 1] == 0)); mod.setdefault(key, []).append(pb[k])
            run = run + 1 if tg[k, 1] == last else 1; last = tg[k, 1]
rows = {k: (float(np.mean(emp[k])), float(np.mean(mod[k])), len(emp[k])) for k in sorted(emp) if len(emp[k]) > 200}
print(json.dumps(dict(ckpt=sys.argv[1].rsplit('/', 1)[-1], rows=rows)))
