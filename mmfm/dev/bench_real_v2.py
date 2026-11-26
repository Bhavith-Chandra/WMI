import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from data.dataset_v2 import raw_chunks
from data.tokenizer_v2 import Tokenizer2
from sim.book import LevelBook
from sim.rollout2 import Rollout2, context_from_history2
from sim_eval_v2 import load_v2
torch.set_num_threads(4)
run, pair = sys.argv[1], sys.argv[2]
m, _ = load_v2(run); tok = Tokenizer2.from_json(json.loads((Path(run) / "tokenizer.json").read_text()))
ev_dir = Path("/home/user/data/mmfm_v2")
ch, _ = raw_chunks(ev_dir, "train")
c = max((c for c in ch if c["pair"] == pair), key=lambda c: len(c["events"]))
z = np.load(ev_dir / c["file"]); j = 20; idx = int(z["ck_idx"][j]); tick = float(z["tick"])
evs = np.zeros(idx, dtype=[("us", "i8"), ("type", "i1"), ("side", "i1"), ("price", "f8"), ("qty", "f8")])
for k in evs.dtype.names: evs[k] = z[k][:idx]
b = LevelBook(tick)
for arr, d in ((z["ck_bids"][j], b.bids), (z["ck_asks"][j], b.asks)):
    for p, q in arr:
        if np.isfinite(p): d[int(round(p / tick))] = float(q)
ctx = context_from_history2(evs[-1100:], z["state"][idx - 1100:idx], z["lvq"][idx - 1100:idx], tok, b, m.cfg["ctx"])
for n in (8, 16, 32):
    R = Rollout2(m, tok, ctx, n_paths=n, seed=0); t = time.time()
    for _ in range(400): R.step()
    el = time.time() - t
    print(f"{pair} {n} paths: {1000*el/400:.2f} ms/step -> {400/el:.0f} sim events/path/s; book levels {len(b.bids)+len(b.asks)}")
