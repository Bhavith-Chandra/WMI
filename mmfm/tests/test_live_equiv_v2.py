"""v2 train/serve skew: recorded messages through LiveFeed2 == batch v2 encoding (design data only)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.l3 import L3Processor, hour_dirs, load_conn, load_snapshot, pair_messages, rebuild  # noqa: E402
from data.tokenizer_v2 import Tokenizer2  # noqa: E402
from serve.live2 import LiveFeed2  # noqa: E402

RAW = Path("/home/user/data/bitstamp")
CUT = Path("/home/user/data/dev_cut_ns.txt")


@pytest.mark.skipif(not CUT.exists(), reason="needs recorded design data")
def test_live2_equals_batch_v2():
    cut = int(CUT.read_text())
    hours = hour_dirs(RAW)[:2]
    conn = [e for e in load_conn(hours) if e["t_ns"] < cut]
    pair = "linkusd"
    msgs = [x for x in pair_messages(hours, pair) if x[0] < cut]
    c = [e["t_ns"] for e in conn if e["ev"] == "connect"][1]
    snaps = sorted((e for e in conn if e["ev"] == "snapshot" and e["pair"] == pair), key=lambda e: e["us"])
    a = [e for e in snaps if c + 1e9 <= e["us"] * 1000][0]
    seg = [x for x in msgs if c <= x[0] < c + 5 * 60e9]
    snap = load_snapshot(hours, pair, a["us"])
    r = rebuild(pair, snap, int(a["us"]), seg, [], hours, 1, tick=1e-4, want_state=True)
    ev, st, lvq = r["events"], r["state"], r["lvq"]
    ok = np.isfinite(st[:, 0])
    tok = Tokenizer2.fit([(ev[ok], st[ok], lvq[ok])], dict(split="train", files_hash="x", range=[0, 1]))
    tb, fb, _ = tok.encode(ev[ok], st[ok], lvq[ok])
    live = LiveFeed2(pair, 1e-4, tok, ctx_len=10**7, hist=10**7)
    live.proc = L3Processor(snap, int(a["us"]))
    for _, m in seg:
        if m["event"] == "trade":
            live.proc.add_trade(m["data"])
    for _, m in seg:
        live._handle(m)
    tl, fl = np.array(live.tokens), np.array(live.feats)
    n = min(len(tl), len(fl), len(tb)) - 1
    assert n > 5000
    bad_t = np.flatnonzero((tl[:n] != tb[:n]).any(1))
    assert not len(bad_t), f"token mismatch rows {bad_t[:5]}"
    bad_f = np.flatnonzero(~np.isclose(fl[:n], fb[:n], atol=1e-4).all(1))
    assert not len(bad_f), f"feature mismatch rows {bad_f[:5]}: {fl[bad_f[0]] - fb[bad_f[0]]}"
