"""Bitstamp live L3 recorder (public data, no credentials).

python -I collect/recorder.py --out /home/user/data/bitstamp --pairs btcusd,ethusd,...
Writes, per UTC hour:  <out>/<YYYYmmdd>/<HH>/stream.jsonl.gz   raw websocket messages, one per line,
                                                              prefixed with local receive time (ns)
                       <out>/<YYYYmmdd>/<HH>/snap_<pair>_<us>.json.gz   full L3 REST snapshots (order ids)
                       <out>/<YYYYmmdd>/<HH>/conn.jsonl        connect / disconnect / gap markers
Snapshots are taken right after every (re)connect and then every --snap-every seconds. Downstream code
splits sequences at every disconnect (no fabricated continuity) and verifies the reconstructed book
against the periodic snapshots.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import time
import urllib.request
from pathlib import Path

import websockets

WS = "wss://ws.bitstamp.net"
REST = "https://www.bitstamp.net/api/v2/order_book/{pair}/?group=2"


class HourWriter:
    def __init__(self, root: Path):
        self.root, self.key, self.f = root, None, None

    def _dir(self):
        t = time.gmtime()
        return self.root / time.strftime("%Y%m%d", t) / time.strftime("%H", t)

    def write(self, line: bytes):
        d = self._dir()
        if d != self.key:
            if self.f:
                self.f.close()
            d.mkdir(parents=True, exist_ok=True)
            self.f = gzip.open(d / "stream.jsonl.gz", "ab", compresslevel=6)
            self.key = d
        self.f.write(line)

    def flush(self):
        if self.f:
            self.f.flush()

    def path(self):
        return self._dir()


def conn_log(w: HourWriter, **kv):
    d = w.path()
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "conn.jsonl", "a") as f:
        f.write(json.dumps(dict(t_ns=time.time_ns(), **kv)) + "\n")


def fetch_snapshot(pair: str) -> bytes:
    req = urllib.request.Request(REST.format(pair=pair), headers={"User-Agent": "l3-recorder"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


async def snapshot_all(w: HourWriter, pairs, why: str):
    for p in pairs:
        try:
            raw = await asyncio.to_thread(fetch_snapshot, p)
            us = json.loads(raw)["microtimestamp"]
            d = w.path()
            d.mkdir(parents=True, exist_ok=True)
            with gzip.open(d / f"snap_{p}_{us}.json.gz", "wb") as f:
                f.write(raw)
            conn_log(w, ev="snapshot", pair=p, us=int(us), why=why)
        except Exception as e:  # recorded, never silently ignored
            conn_log(w, ev="snapshot_error", pair=p, err=repr(e)[:300])


async def periodic_snaps(w, pairs, every):
    while True:
        await asyncio.sleep(every)
        await snapshot_all(w, pairs, "periodic")


async def run(a):
    w = HourWriter(Path(a.out))
    pairs = a.pairs.split(",")
    asyncio.create_task(periodic_snaps(w, pairs, a.snap_every))
    backoff = 1
    while True:
        try:
            async with websockets.connect(WS, max_size=2**24, ping_interval=20, ping_timeout=20) as ws:
                for p in pairs:
                    for ch in ("live_orders", "live_trades"):
                        await ws.send(json.dumps({"event": "bts:subscribe", "data": {"channel": f"{ch}_{p}"}}))
                conn_log(w, ev="connect", pairs=pairs)
                await snapshot_all(w, pairs, "connect")  # stream already buffering -> snapshot inside stream
                backoff, n, t_flush = 1, 0, time.time()
                async for msg in ws:
                    w.write(f"{time.time_ns()}\t".encode() + (msg if isinstance(msg, bytes) else msg.encode()) + b"\n")
                    n += 1
                    if '"bts:request_reconnect"' in (msg if isinstance(msg, str) else msg.decode()):
                        conn_log(w, ev="server_request_reconnect")
                        break
                    if time.time() - t_flush > 5:
                        w.flush()
                        t_flush = time.time()
        except Exception as e:
            conn_log(w, ev="disconnect", err=repr(e)[:300])
        w.flush()
        conn_log(w, ev="gap_start")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/user/data/bitstamp")
    ap.add_argument("--pairs", default="btcusd,ethusd,solusd,xrpusd,btceur,ltcusd,dogeusd,linkusd")
    ap.add_argument("--snap-every", type=int, default=900)
    asyncio.run(run(ap.parse_args()))
