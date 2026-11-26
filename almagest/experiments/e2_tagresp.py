"""Diagnostic: background market-order side vs recent OWN (tagged) signed flow, World vs model, teacher-forced on
interventional validation streams (VAL_BASE). usage: e2_tagresp.py <ckpt.pt>"""
import sys, os, json
import numpy as np, torch
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from scipy.signal import lfilter
from almagest import splits as S
from almagest.experiments.e2_train import build_stream_inj, make_model
from almagest.experiments.e1_calibrate_world import make
torch.set_num_threads(2)
ck = torch.load(sys.argv[1], weights_only=False)
m, _ = make_model(ck['arm'], cfg=ck.get('cfg')); m.load_state_dict(ck['state']); m.eval()
if getattr(m, 'tagch', False):
    os.environ['TAG_CHANNEL'] = '1'
par = make(ck['theta']); X, E, P = [], [], []
for i in range(4):
    d = build_stream_inj(par, S.VAL_BASE + i, 60000)
    dd = d if m.src else {k: (v[:, :4] if k in ('prev', 'tgt') else v) for k, v in d.items()}
    t = {k: torch.tensor(v[None]) for k, v in dd.items()}
    with torch.no_grad():
        x = m.embed(t['prev'], t['feat'], t['dmid'], t['sb']); z, _ = m.trunk_seq(x, m.init_state(1))
        pb = torch.softmax(m.side_logits(z, torch.full_like(t['tgt'][..., 0], 2)), -1)[0, :, 0].numpy()
    tg = d['tgt']
    tagged = np.where(tg[:, 4] == 1, np.where(tg[:, 1] == 0, 1.0, -1.0) * tg[:, 3], 0.0)
    tf = lfilter([1 / 500], [1, -(1 - 1 / 500)], np.concatenate([[0], tagged[:-1]])) * 500
    mk = (tg[:, 0] == 2) & (tg[:, 4] == 0)
    X.append(tf[mk]); E.append((tg[mk, 1] == 0).astype(float)); P.append(pb[mk])
X, E, P = map(np.concatenate, (X, E, P))
rows = {f'({a},{b}]': (float(E[(X > a) & (X <= b)].mean()), float(P[(X > a) & (X <= b)].mean()), int(((X > a) & (X <= b)).sum()))
        for a, b in ((-99, -8), (-8, -2), (-2, 2), (2, 8), (8, 99))}
print(json.dumps(dict(ckpt=sys.argv[1].rsplit('/', 1)[-1], world_vs_model=rows)))
