"""Train/serve skew test: recorded messages replayed through the LIVE handler must produce exactly the
tokens and features the batch pipeline (training data) produces. Uses dev-window data only."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.l3 import hour_dirs, load_conn, load_snapshot, pair_messages, rebuild  # noqa: E402
from data.tokenizer import Tokenizer  # noqa: E402
from serve.live import LiveFeed  # noqa: E402
from data.l3 import L3Processor  # noqa: E402

RAW = Path("/home/user/data/bitstamp")
CUT = Path("/home/user/data/dev_cut_ns.txt")


@pytest.mark.skipif(not CUT.exists(), reason="needs recorded dev data")
def test_live_handler_equals_batch_encoding():
    cut = int(CUT.read_text())
    hours = hour_dirs(RAW)
    conn = [e for e in load_conn(hours) if e["t_ns"] < cut]
    pair = "linkusd"
    msgs = [x for x in pair_messages(hours, pair) if x[0] < cut]
    c = [e["t_ns"] for e in conn if e["ev"] == "connect"][1]
    snaps = sorted((e for e in conn if e["ev"] == "snapshot" and e["pair"] == pair), key=lambda e: e["us"])
    a = [e for e in snaps if c + 1e9 <= e["us"] * 1000][0]
    seg = [x for x in msgs if c <= x[0] < c + 5 * 60e9]
    snap = load_snapshot(hours, pair, a["us"])
    r = rebuild(pair, snap, int(a["us"]), seg, [], hours, 1, tick=1e-4, want_state=True)
    ev, st = r["events"], r["state"]
    ok = np.isfinite(st[:, 0])
    tok = Tokenizer.fit([(ev[ok], st[ok])], dict(split="train", files_hash="x", range=[0, 1]))
    tb, fb, _ = tok.encode(ev[ok], st[ok])

    live = LiveFeed(pair, 1e-4, tok, ctx_len=10**7, hist=10**7)
    live.proc = L3Processor(snap, int(a["us"]))
    for _, m in seg:
        if m["event"] == "trade":
            live.proc.add_trade(m["data"])
    for _, m in seg:
        live._handle(m)
    tl = np.array(live.tokens)
    fl = np.array(live.feats)  # finalised rows only (the last row's feature awaits the next event)
    n = min(len(tl), len(tb)) - 1
    assert n > 5000
    assert np.array_equal(tl[:n], tb[:n]), f"token mismatch at {np.flatnonzero((tl[:n] != tb[:n]).any(1))[:5]}"
    bad = np.flatnonzero(~np.isclose(fl[:n], fb[:n], atol=1e-4).all(1))
    if len(bad):
        i = bad[0]
        print("ROW", i, "live", fl[i], "batch", fb[i], "ev", ev[ok][i - 1:i + 2], "st", st[ok][i - 1:i + 3])
    assert np.allclose(fl[:n], fb[:n], atol=1e-4), f"feature mismatch rows {np.flatnonzero(~np.isclose(fl[:n], fb[:n], atol=1e-4).all(1))[:5]}"
