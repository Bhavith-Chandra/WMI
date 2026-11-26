"""Order-id-level L3 reconstruction from recorder output + verification against REST snapshots.

Segment = one websocket connection. It is anchored on the first REST snapshot whose exchange timestamp is
> 1 s after we subscribed (REST snapshots can be served stale by seconds; anything changed between the
snapshot time and subscription would otherwise be lost). Events up to the anchor are dropped (causal).
Later snapshots in the segment are used ONLY to verify (never to correct) -> fidelity metric M6.

Model-level events (each known at its timestamp):
  ADD     resting quantity added (new resting order, aggressive order's unfilled remainder, size increase)
  CANCEL  resting quantity removed unfilled (delete message carries the unfilled amount; 0 for a full fill)
  TAKE    aggressive execution, aggregated per taker order id per millisecond (qty, worst price)
Maker-side fills are consequences of TAKE and are not modelled separately.

Bitstamp quirks handled (all found by verification):
  * trades arrive on another channel, sometimes before the taker's order_created
  * an aggressive order's order_created carries the full amount; the executed part is a TAKE
  * order_changed with a trade in the same ms reports the execution price, not an amendment
"""
from __future__ import annotations

import gzip
import heapq
import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sortedcontainers import SortedDict

ADD, CANCEL, TAKE = 0, 1, 2
BUY, SELL = 1, 0  # 1 = bid/buy, 0 = ask/sell (Bitstamp order_type 0 = buy, 1 = sell)
EV_DTYPE = [("us", "i8"), ("type", "i1"), ("side", "i1"), ("price", "f8"), ("qty", "f8"), ("oid", "i8")]


# ---------------------------------------------------------------------------------------------- raw io
def _gunzip_salvage(raw: bytes) -> bytes:
    """Concatenated gzip members; a truncated member (killed writer) keeps what decoded before the cut,
    then decoding resumes at the next gzip header."""
    out, pos = [], 0
    while pos < len(raw):
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        parts, i, ok = [], pos, True
        while i < len(raw):
            try:
                parts.append(d.decompress(raw[i:i + 65536]))
            except zlib.error:
                ok = False
                break
            i += 65536
            if d.eof:
                break
        out.append(b"".join(parts))
        if ok and d.eof:
            pos = len(raw) - len(d.unused_data)
            continue
        nxt = raw.find(b"\x1f\x8b\x08", pos + 10)
        if nxt < 0:
            break
        out.append(b"\n")
        pos = nxt
    return b"".join(out)


def read_stream(path: Path):
    for ln in _gunzip_salvage(path.read_bytes()).split(b"\n"):
        if not ln:
            continue
        try:
            t, m = ln.split(b"\t", 1)
            yield int(t), json.loads(m)
        except (ValueError, json.JSONDecodeError):
            continue


def hour_dirs(root: Path):
    return sorted(p for p in Path(root).glob("*/*") if p.is_dir())


def stream_files(h: Path):
    return sorted(h.glob("stream*.jsonl.gz"))


def load_conn(hours):
    ev = []
    for h in hours:
        f = h / "conn.jsonl"
        if f.exists():
            ev += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return sorted(ev, key=lambda e: e["t_ns"])


def load_snapshot(hours, pair, us):
    for h in hours:
        f = h / f"snap_{pair}_{us}.json.gz"
        if f.exists():
            return json.loads(gzip.open(f).read())
    raise FileNotFoundError(f"snapshot {pair} {us}")


def pair_messages(hours, pair):
    out = []
    for h in hours:
        for f in stream_files(h):
            for t, m in read_stream(f):
                if m.get("channel", "").endswith("_" + pair) and m.get("event") in (
                        "order_created", "order_changed", "order_deleted", "trade"):
                    out.append((t, m))
    out.sort(key=lambda x: x[0])
    return out


# ---------------------------------------------------------------------------------------------- books
@dataclass
class IdBook:
    """id -> [side, price, amount] plus an incrementally maintained price-level aggregate per side."""
    orders: dict = field(default_factory=dict)
    agg: tuple = field(default_factory=lambda: (SortedDict(), SortedDict()))  # agg[SELL]=asks, agg[BUY]=bids

    @classmethod
    def from_snapshot(cls, snap):
        b = cls()
        for side, key in ((BUY, "bids"), (SELL, "asks")):
            for p, a, oid in snap[key]:
                b.set(int(oid), side, float(p), float(a))
        return b

    def _agg_add(self, side, p, q, dn):
        """Level = [qty, n_orders]; a level disappears exactly when its last order goes (no float dust)."""
        d = self.agg[side]
        cur = d.get(p)
        if cur is None:
            cur = d[p] = [0.0, 0]
        cur[0] += q
        cur[1] += dn
        if cur[1] <= 0:
            del d[p]

    def set(self, oid, side, p, a):
        prev = self.orders.get(oid)
        if prev is not None:
            self._agg_add(prev[0], prev[1], -prev[2], -1)
        self.orders[oid] = [side, p, a]
        self._agg_add(side, p, a, +1)

    def pop(self, oid):
        prev = self.orders.pop(oid, None)
        if prev is not None:
            self._agg_add(prev[0], prev[1], -prev[2], -1)
        return prev

    def levels(self, side, n=20):
        d = self.agg[side]
        keys = reversed(d.keys()) if side == BUY else iter(d.keys())
        out = []
        for k in keys:
            q = d[k][0]
            if q > 1e-12:  # zero-qty (fully matched, not yet deleted) orders don't make a level
                out.append((k, q))
                if len(out) == n:
                    break
        return out

    def state(self):
        """(best_bid, best_ask, q_bid1, q_ask1, q_bid5, q_ask5) from the current book (NaN if a side is empty)."""
        lb, la = self.levels(BUY, 5), self.levels(SELL, 5)
        if not lb or not la:
            return (np.nan,) * 6
        return (lb[0][0], la[0][0], lb[0][1], la[0][1], sum(q for _, q in lb), sum(q for _, q in la))


def compare_levels(a_levels, b_levels, rel=1e-9):
    if not a_levels:
        return 1.0 if not b_levels else 0.0
    bd = dict(b_levels)
    return sum(1 for p, q in a_levels if p in bd and abs(bd[p] - q) <= rel * max(1.0, q)) / len(a_levels)


def compare(book: IdBook, snap, n=20):
    ref = IdBook.from_snapshot(snap)
    ids_r, ids_b = set(ref.orders), set(book.orders)
    out = dict(n_ref=len(ids_r), n_rec=len(ids_b), id_jaccard=len(ids_r & ids_b) / max(len(ids_r | ids_b), 1))
    for side, nm in ((BUY, "bid"), (SELL, "ask")):
        lr, lb = ref.levels(side, n), book.levels(side, n)
        out[f"best_{nm}_match"] = bool(lr and lb and lr[0][0] == lb[0][0])
        out[f"top{n}_{nm}_level_match"] = compare_levels(lr, lb)
    out["accept"] = bool(out["best_bid_match"] and out["best_ask_match"]
                         and min(out[f"top{n}_bid_level_match"], out[f"top{n}_ask_level_match"]) >= 0.99)
    return out


# ---------------------------------------------------------------------------------------------- segments
def segments(root: Path, pair: str, tick: float | None = None, lockstep_every: int = 50, want_state=False):
    hours = hour_dirs(root)
    conn = load_conn(hours)
    msgs = pair_messages(hours, pair)
    connects = [e["t_ns"] for e in conn if e["ev"] == "connect"]
    ends = [e["t_ns"] for e in conn if e["ev"] in ("disconnect", "server_request_reconnect", "gap_start", "connect")]
    snaps = sorted((e for e in conn if e["ev"] == "snapshot" and e["pair"] == pair), key=lambda e: e["us"])
    for k, c_ns in enumerate(connects):
        t_hi = min([d for d in ends if d > c_ns] + [float("inf")])
        anch = [e for e in snaps if c_ns + 1e9 <= e["us"] * 1000 < t_hi]
        if not anch:
            continue
        snap = load_snapshot(hours, pair, anch[0]["us"])
        seg_msgs = [x for x in msgs if c_ns <= x[0] < t_hi]
        yield rebuild(pair, snap, int(anch[0]["us"]), seg_msgs, anch[1:], hours, k, tick, lockstep_every,
                      t_lo_ns=c_ns, t_hi_ns=t_hi, want_state=want_state)


def _lv_ticks(levels, tick):
    return [(int(round(p / tick)), q) for p, q in levels]


class L3Processor:
    """Message -> model-event logic (single implementation used by batch rebuild AND the live interface).

    Trades must be registered (add_trade) before the order messages of the same matching step are handled;
    batch registers all trades up front, live registers on arrival and handles order messages after a delay.
    """

    def __init__(self, snap, snap_us):
        self.book = IdBook.from_snapshot(snap)
        self.snap_us = int(snap_us)
        self.led = {oid: list(v) for oid, v in self.book.orders.items()}  # event-level resting ledger
        self.trades_by_ms, self.taker_traded, self.taker_first, self.taker_worst, self.traded_ids = {}, {}, {}, {}, {}
        self.taker_done, self.taken = set(), set()

    def add_trade(self, d):
        us = int(d["microtimestamp"])
        self.trades_by_ms.setdefault(us, []).append(d)
        tid = int(d["buy_order_id"] if d["type"] == 0 else d["sell_order_id"])
        self.taker_traded[tid] = self.taker_traded.get(tid, 0.0) + float(d["amount"])
        self.taker_first[tid] = min(self.taker_first.get(tid, us), us)
        px_ = float(d["price"])
        self.taker_worst[tid] = (max if d["type"] == 0 else min)(self.taker_worst.get(tid, px_), px_)
        self.traded_ids.setdefault(us, set()).update((int(d["buy_order_id"]), int(d["sell_order_id"])))

    def prune(self, before_us):
        """Live use: forget trade aggregates older than before_us (bounded memory)."""
        for k in [k for k in self.trades_by_ms if k < before_us]:
            for d in self.trades_by_ms.pop(k):
                tid = int(d["buy_order_id"] if d["type"] == 0 else d["sell_order_id"])
                for dd in (self.taker_traded, self.taker_first, self.taker_worst):
                    dd.pop(tid, None)
                self.taker_done.discard(tid)
            self.traded_ids.pop(k, None)
        self.taken = {x for x in self.taken if x[0] >= before_us}

    def handle(self, m):
        """Process one message (already registered if it is a trade). Returns list of event rows."""
        d = m["data"]
        us = int(d["microtimestamp"])
        rows = []
        if us <= self.snap_us:
            return rows
        if m["event"] == "trade":
            taker = int(d["buy_order_id"] if d["type"] == 0 else d["sell_order_id"])
            if (us, taker) in self.taken or taker in self.taker_done:
                return rows
            same = [x for x in self.trades_by_ms[us] if x["type"] == d["type"]
                    and int(x["buy_order_id"] if x["type"] == 0 else x["sell_order_id"]) == taker]
            qty = sum(float(x["amount"]) for x in same)
            px = max(float(x["price"]) for x in same) if d["type"] == 0 else min(float(x["price"]) for x in same)
            rows.append((us, TAKE, BUY if d["type"] == 0 else SELL, px, qty, taker))
            self.taken.add((us, taker))
            return rows
        book, led = self.book, self.led
        oid = int(d["id"])
        side = BUY if int(d["order_type"]) == 0 else SELL
        price, amt = float(d["price"]), float(d["amount"])
        ev = m["event"]
        traded_now = oid in self.traded_ids.get(us, ())
        prev = book.orders.get(oid)
        if ev == "order_created":
            # aggressive part: this id's taker volume, if its trades fall in the same matching step (<= 50 ms;
            # Bitstamp sometimes stamps the trades 1 ms after the order_created)
            agg_q = self.taker_traded.get(oid, 0.0) if abs(self.taker_first.get(oid, us) - us) <= 50_000 else 0.0
            rest = max(amt - agg_q, 0.0)
            if agg_q > 1e-12 and oid not in self.taker_done and not any(k[1] == oid for k in self.taken):
                rows.append((us, TAKE, side, self.taker_worst[oid], agg_q, oid))  # matching first, then rest
                self.taker_done.add(oid)
            book.set(oid, side, price, amt)  # id-level truth (amount still to match during a sweep)
            led[oid] = [side, price, rest]
            if rest > 1e-12:
                rows.append((us, ADD, side, price, rest, oid))
        elif ev == "order_changed":
            book.set(oid, side, prev[1] if (prev and traded_now) else price, amt)
            L = led.get(oid)
            if L is None:  # first sighting resting (no created seen): remainder starts to rest
                led[oid] = [side, price, amt]
                rows.append((us, ADD, side, price, amt, oid))
            elif traded_now:  # fill: already represented by the TAKE; ledger can only shrink
                L[2] = min(L[2], amt)
            elif price != L[1]:  # genuine price amendment
                if L[2] > 1e-12:
                    rows.append((us, CANCEL, side, L[1], L[2], oid))
                rows.append((us, ADD, side, price, amt, oid))
                led[oid] = [side, price, amt]
            elif amt > L[2] + 1e-12:
                rows.append((us, ADD, side, L[1], amt - L[2], oid))
                L[2] = amt
            elif amt < L[2] - 1e-12:
                rows.append((us, CANCEL, side, L[1], L[2] - amt, oid))
                L[2] = amt
        else:  # order_deleted: message amount = quantity removed unfilled
            book.pop(oid)
            L = led.pop(oid, None)
            q = min(amt, L[2]) if L is not None else 0.0
            if q > 1e-12:
                rows.append((us, CANCEL, side, L[1], q, oid))
        return rows


def rebuild(pair, snap, snap_us, msgs, verify_snaps, hours, seg_id, tick=None, lockstep_every=50, t_lo_ns=None,
            t_hi_ns=None, debug=None, want_state=False, ckpt_every_us=None, ckpt_levels=300):
    from sim.book import LevelBook  # local import: sim.book imports this module's constants

    P = L3Processor(snap, snap_us)
    ckpts, next_ck = [], (snap_us + ckpt_every_us) if ckpt_every_us else None
    book = P.book
    for _, m in msgs:
        if m["event"] == "trade":
            P.add_trade(m["data"])
    rows, verify, states = [], [], []
    pend = sorted(verify_snaps, key=lambda p: p["us"])
    lvl = LevelBook.from_idbook(book, tick) if tick else None
    applied = n_ms = n_checked = n_div = 0
    last_us = None
    for idx, (_, m) in enumerate(msgs):
        us = int(m["data"]["microtimestamp"])
        if us <= snap_us:
            continue
        pre = book.state() if want_state else None
        if next_ck is not None and us >= next_ck and last_us is not None and us != last_us:
            # exact id-book checkpoint BEFORE event index len(rows) (book after all previous events)
            ckpts.append((len(rows), last_us, book.levels(BUY, ckpt_levels), book.levels(SELL, ckpt_levels)))
            next_ck = us + ckpt_every_us
        if last_us is not None and us != last_us:
            n_ms += 1
            if lvl is not None:
                for r in rows[applied:]:
                    lvl.apply(r[1], r[2], int(round(r[3] / tick)), r[4])
                applied = len(rows)
                if n_ms % lockstep_every == 0:
                    n_checked += 1
                    bad = any(compare_levels(_lv_ticks(book.levels(sd, 10), tick), lvl.levels(sd, 10), 1e-7) < 1.0
                              for sd in (BUY, SELL))
                    n_div += bad
                    if bad and debug is not None and not debug:
                        debug.append(dict(us=last_us, id_levels={sd: _lv_ticks(book.levels(sd, 6), tick) for sd in (BUY, SELL)},
                                          ev_levels={sd: lvl.levels(sd, 6) for sd in (BUY, SELL)}, rows=rows[-12:]))
            while pend and pend[0]["us"] < us:
                p = pend.pop(0)
                verify.append(dict(us=p["us"], **compare(book, load_snapshot(hours, pair, p["us"]))))
        last_us = us
        new = P.handle(m)
        rows.extend(new)
        if want_state:
            states.extend([pre] * len(new))
    for p in pend:
        verify.append(dict(us=p["us"], **compare(book, load_snapshot(hours, pair, p["us"]))))
    final = None
    if lvl is not None:
        for r in rows[applied:]:
            lvl.apply(r[1], r[2], int(round(r[3] / tick)), r[4])
        final = dict(bid=compare_levels(_lv_ticks(book.levels(BUY, 50), tick), lvl.levels(BUY, 50), 1e-7),
                     ask=compare_levels(_lv_ticks(book.levels(SELL, 50), tick), lvl.levels(SELL, 50), 1e-7))
    st = np.array(states, dtype=np.float64).reshape(-1, 6) if want_state else None
    return dict(pair=pair, seg=seg_id, snap_us=snap_us, events=np.array(rows, dtype=EV_DTYPE), verify=verify, state=st,
                final_book=book, snapshot=snap, events_book=lvl, t_lo_ns=t_lo_ns, t_hi_ns=t_hi_ns,
                lockstep=dict(ms=n_ms, checked=n_checked, diverged=n_div, final_top50_match=final), ckpts=ckpts)
