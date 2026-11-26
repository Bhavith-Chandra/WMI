"""Diagnose rollout drift vs reality on DESIGN data (no test data): what differs between simulated and real
continuations over H seconds from exact book checkpoints."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from data.dataset import raw_chunks
from data.tokenizer import Tokenizer
from evaluate import load_model
from sim.book import LevelBook
from sim.rollout import Rollout, context_from_history, book_state

torch.set_num_threads(2)
events = Path(sys.argv[1]); run = sys.argv[2]; pair = sys.argv[3]; split = sys.argv[4]
H = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0
n_starts = int(sys.argv[6]) if len(sys.argv) > 6 else 6
temp = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
dts = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
model, _ = load_model(run)
tok = Tokenizer.from_json(json.loads((Path(run) / "tokenizer.json").read_text()))
chunks, _ = raw_chunks(events, split)
rng = np.random.default_rng(1)
starts = []
for c in chunks:
    if c["pair"] != pair:
        continue
    z = np.load(events / c["file"])
    lo, hi = c["events"]["us"][0], c["events"]["us"][-1]
    for j, idx in enumerate(z["ck_idx"]):
        if idx > 1200 and lo <= z["us"][idx] <= hi - H * 1e6:
            starts.append((c["file"], j, int(idx)))
pick = [starts[k] for k in rng.choice(len(starts), min(n_starts, len(starts)), replace=False)]
S = {k: [] for k in ("n", "type", "dt_med", "dt_zero", "touch_frac", "mid_moves", "spread_bp", "ret")}
Rl = {k: [] for k in S}


def stats(types, dts, touch, mids, spreads, r):
    return dict(n=len(types), type=np.bincount(types, minlength=3) / max(len(types), 1), dt_med=np.median(dts) if len(dts) else np.nan,
                dt_zero=np.mean(np.array(dts) == 0) if len(dts) else np.nan, touch_frac=np.mean(touch) if len(touch) else np.nan,
                mid_moves=int(np.sum(np.diff(mids) != 0)) if len(mids) > 1 else 0, spread_bp=np.median(spreads) if len(spreads) else np.nan, ret=r)


for f, j, idx in pick:
    z = np.load(events / f)
    tick = float(z["tick"])
    evs = np.zeros(len(z["us"]), dtype=[("us", "i8"), ("type", "i1"), ("side", "i1"), ("price", "f8"), ("qty", "f8")])
    for k in evs.dtype.names:
        evs[k] = z[k]
    st = z["state"]
    b = LevelBook(tick)
    for arr, d in ((z["ck_bids"][j], b.bids), (z["ck_asks"][j], b.asks)):
        for p, q in arr:
            if np.isfinite(p):
                d[int(round(p / tick))] = float(q)
    t0 = evs["us"][idx - 1]
    jend = np.searchsorted(evs["us"], t0 + H * 1e6, side="right")
    re, rs = evs[idx:jend], st[idx:jend + 1]
    mids = 0.5 * (rs[:, 0] + rs[:, 1])
    tok_r, _, _ = tok.encode(evs[idx - 1100:jend], st[idx - 1100:jend])
    tr = tok_r[1100:]
    real = stats(re["type"], np.diff(re["us"], prepend=t0), tr[:, 2] == 8, mids, (rs[:, 1] - rs[:, 0]) / mids * 1e4, np.log(mids[-1] / mids[0]) * 1e4)
    for k in S:
        Rl[k].append(real[k])
    ctx = context_from_history(evs[idx - 1100:idx], st[idx - 1100:idx], tok, b, model.cfg["ctx"])
    R = Rollout(model, tok, ctx, n_paths=8, seed=idx, temperature=temp, dt_scale=dts)
    rec = [dict(types=[], dts=[], touch=[], spreads=[]) for _ in R.paths]
    orig = R._apply

    def wrapped(p, o, _orig=orig, R=R, rec=rec):
        i = R.paths.index(p)
        pre = book_state(p["book"])
        row, f = _orig(p, o)
        if p["us"] - R.t0 <= H * 1e6:
            rec[i]["types"].append(o[0]); rec[i]["dts"].append(o[4]); rec[i]["touch"].append(row[2] == 8)
            if pre:
                rec[i]["spreads"].append((pre[1] - pre[0]) / (0.5 * (pre[0] + pre[1])) * 1e4)
        return row, f

    R._apply = wrapped
    R.run(H, max_steps=40000)
    for i, p in enumerate(R.paths):
        t = (np.array(p["t"]) - R.t0) / 1e6
        m = np.array(p["mid"])[t <= H]
        s = stats(np.array(rec[i]["types"], int), rec[i]["dts"], rec[i]["touch"], m, rec[i]["spreads"], np.log(m[-1] / m[0]) * 1e4)
        for k in S:
            S[k].append(s[k])
fmt = lambda v: np.round(np.nanmean(np.stack(v), 0) if np.ndim(v[0]) else np.nanmedian(v), 3)
print(f"{pair} H={H}s temp={temp} dt_scale={dts} starts={len(pick)}")
for k in S:
    print(f"  {k:11s} real {fmt(Rl[k])}   sim {fmt(S[k])}")
print("  |ret| bp   real median %.2f  sim median %.2f ; sd real %.2f sim %.2f" % (np.median(np.abs(Rl['ret'])), np.median(np.abs(S['ret'])), np.std(Rl['ret']), np.std(S['ret'])))
