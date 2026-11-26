"""Debug: find the first message where the aggregated model-event book diverges from the id-level book."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.l3 import *  # noqa
from sim.book import LevelBook

pair, tk = sys.argv[1], float(sys.argv[2])
root = Path("/home/user/data/bitstamp")
hours = hour_dirs(root)
conn = load_conn(hours)
s0 = [e for e in conn if e["ev"] == "snapshot" and e["pair"] == pair and e.get("why") == "connect"][0]
snap = load_snapshot(hours, pair, s0["us"])
msgs = [(t, m) for h in hours for f in stream_files(h) for t, m in read_stream(f)
        if m.get("channel", "").endswith("_" + pair) and m.get("event") in ("order_created", "order_changed", "order_deleted", "trade")]
ib = IdBook.from_snapshot(snap)
lb = LevelBook.from_idbook(ib, tk)
trades = {}
for _, m in msgs:
    if m["event"] == "trade":
        trades.setdefault(int(m["data"]["microtimestamp"]), []).append(m["data"])
n = 0
for idx, (_, m) in enumerate(msgs):
    d = m["data"]; us = int(d["microtimestamp"])
    if us <= int(s0["us"]):
        continue
    if m["event"] == "trade":
        print("TRADE", us, d["type"], d["price"], d["amount"], "buy", d["buy_order_id"], "sell", d["sell_order_id"],
              "buy in book", int(d["buy_order_id"]) in ib.orders, "sell in book", int(d["sell_order_id"]) in ib.orders)
        lb.apply(TAKE, BUY if d["type"] == 0 else SELL, int(round(float(d["price"]) / tk)), float(d["amount"]))
    else:
        oid = int(d["id"]); side = BUY if int(d["order_type"]) == 0 else SELL; p = float(d["price"]); a = float(d["amount"])
        prev = ib.orders.get(oid)
        if m["event"] == "order_created":
            ib.orders[oid] = [side, p, a]; lb.apply(ADD, side, int(round(p / tk)), a)
        elif m["event"] == "order_changed":
            ib.orders[oid] = [side, p, a]
            if prev is None:
                lb.apply(ADD, side, int(round(p / tk)), a)
            elif a < prev[2]:
                lb.apply(CANCEL, side, int(round(p / tk)), prev[2] - a) if not any(int(x["buy_order_id"]) == oid or int(x["sell_order_id"]) == oid for x in trades.get(us, [])) else None
            print("CHANGED", us, oid, side, p, "prev", prev and prev[2], "->", a, "traded", d["amount_traded"])
        else:
            ib.orders.pop(oid, None)
            if prev is not None and not any(int(x["buy_order_id"]) == oid or int(x["sell_order_id"]) == oid for x in trades.get(us, [])):
                lb.apply(CANCEL, side, int(round(p / tk)), prev[2])
            elif prev is not None:
                print("DELETED-TRADED", us, oid, side, p, prev[2])
    ref = LevelBook.from_idbook(ib, tk)
    if ref.levels(SELL, 20) != [(k, q) for k, q in lb.levels(SELL, 20)] and any(abs(a[1] - b[1]) > 1e-8 or a[0] != b[0] for a, b in zip(ref.levels(SELL, 20), lb.levels(SELL, 20))):
        print("DIVERGED at msg", idx, m["event"], us, "id-book asks", ref.levels(SELL, 4), "agg asks", lb.levels(SELL, 4))
        n += 1
        if n > 2:
            break
