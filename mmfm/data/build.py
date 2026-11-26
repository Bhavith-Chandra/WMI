"""Build per-segment event datasets + reconstruction report from recorder output.

python -I data/build.py --raw /home/user/data/bitstamp --out /home/user/data/mmfm_events [--until-ns T]
Each segment -> <out>/<pair>/seg_<snap_us>.npz with
  events: us, type, side, price, qty           (model-level events, data/l3.py)
  state : best_bid, best_ask, q_b1, q_a1, q_b5, q_a5   (id-book state BEFORE each event)
Segments whose verification snapshots fail the EVAL.md §0 rule are written to an exclusion list, not used.
`--until-ns` freezes the data cut (everything received later is ignored), so a build is reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.l3 import hour_dirs, load_conn, load_snapshot, pair_messages, rebuild  # noqa: E402

TICK = {"btcusd": 0.01, "ethusd": 0.01, "solusd": 0.01, "xrpusd": 1e-5, "btceur": 1.0, "ltcusd": 0.01,
        "dogeusd": 1e-6, "linkusd": 1e-4}
MIN_SEG_EVENTS = 1000


def build(raw: Path, out: Path, until_ns: int, pairs, from_ns: int = 0, chunk_s: int = 0):
    hours = hour_dirs(raw)
    conn = [e for e in load_conn(hours) if e["t_ns"] < until_ns]
    report = dict(raw=str(raw), until_ns=until_ns, built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), pairs={})
    connects = [e["t_ns"] for e in conn if e["ev"] == "connect"]
    ends = [e["t_ns"] for e in conn if e["ev"] in ("disconnect", "server_request_reconnect", "gap_start", "connect")]
    lo_read = from_ns - 900 * 10**9 if from_ns else None
    for pair in pairs:
        snaps = sorted((e for e in conn if e["ev"] == "snapshot" and e["pair"] == pair), key=lambda e: e["us"])
        segs = []
        jobs = []  # (seg k, chunk j, c_ns, anchor snapshot, verify snapshots, chunk end ns)
        for k, c_ns in enumerate(connects):
            t_hi = min([d for d in ends if d > c_ns] + [until_ns])
            if t_hi < from_ns:  # segment ends before the window of interest
                continue
            # v2-3: with a window, anchor on the first REST snapshot >= 60 s after (from_ns - 15 min)
            anch = [e for e in snaps if c_ns + 1e9 <= e["us"] * 1000 < t_hi
                    and (lo_read is None or e["us"] * 1000 >= lo_read + 60 * 10**9)]
            if not anch:
                continue
            # v3: long segments are cut into chunks re-anchored on a REST snapshot every ~chunk_s (bounded memory)
            heads = [anch[0]]
            if chunk_s:
                for e in anch[1:]:
                    if e["us"] - heads[-1]["us"] >= chunk_s * 10**6:
                        heads.append(e)
            for j, a in enumerate(heads):
                end = heads[j + 1]["us"] * 1000 if j + 1 < len(heads) else t_hi
                ver = [e for e in anch if a["us"] < e["us"] and e["us"] * 1000 <= end]
                jobs.append((k, j, c_ns, a, ver, end))
        for k, j, c_ns, a, ver, end in jobs:
            msgs = pair_messages(hours, pair, lo_ns=max(c_ns, a["us"] * 1000 - 60 * 10**9), hi_ns=end)
            r = rebuild(pair, load_snapshot(hours, pair, a["us"]), int(a["us"]), msgs, ver, hours, k, TICK[pair], 200,
                        t_lo_ns=c_ns, t_hi_ns=end, want_state=True, ckpt_every_us=30_000_000)
            del msgs
            ev, stt = r["events"], r["state"]
            # v2-1: keep only intervals bounded by two consecutive PASSING verifications (anchor = pass)
            bounds = [(r["snap_us"], True)] + [(v["us"], v["accept"]) for v in r["verify"]] + [(int(ev["us"][-1]) + 1 if len(ev) else r["snap_us"], True)]
            keep = np.zeros(len(ev), bool)
            for (u0, ok0), (u1, ok1) in zip(bounds[:-1], bounds[1:]):
                if ok0 and ok1:
                    keep |= (ev["us"] > u0) & (ev["us"] <= u1)
            # contiguous kept runs become separate parts (a dropped interval is a hard gap)
            runs, i0 = [], None
            for i, kk in enumerate(np.r_[keep, False]):
                if kk and i0 is None:
                    i0 = i
                elif not kk and i0 is not None:
                    runs.append((i0, i))
                    i0 = None
            info = dict(seg=k, chunk=j, snap_us=r["snap_us"], n_events=int(len(ev)), n_kept=int(keep.sum()),
                        verify=r["verify"], lockstep={k2: v for k2, v in r["lockstep"].items() if k2 != "events_replay_book"},
                        parts=[])
            for pi, (a0, a1) in enumerate(runs):
                if a1 - a0 < MIN_SEG_EVENTS:
                    continue
                d = out / pair
                d.mkdir(parents=True, exist_ok=True)
                f = d / f"seg_{r['snap_us']}_p{pi}.npz"
                ck = [(c[0] - a0, c[1], c[2], c[3]) for c in r["ckpts"] if a0 <= c[0] < a1]
                L = 300
                ck_idx = np.array([c[0] for c in ck], np.int64)
                ck_b = np.full((len(ck), L, 2), np.nan)
                ck_a = np.full((len(ck), L, 2), np.nan)
                for jj, c in enumerate(ck):
                    if c[2]:
                        ck_b[jj, :len(c[2])] = c[2]
                    if c[3]:
                        ck_a[jj, :len(c[3])] = c[3]
                sl = slice(a0, a1)
                np.savez_compressed(f, us=ev["us"][sl], type=ev["type"][sl], side=ev["side"][sl], price=ev["price"][sl],
                                    qty=ev["qty"][sl], state=stt[sl].astype(np.float64), lvq=r["lvq"][sl],
                                    tick=np.float64(TICK[pair]), ck_idx=ck_idx, ck_bids=ck_b, ck_asks=ck_a)
                info["parts"].append(dict(file=str(f.relative_to(out)), sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                                          n_events=int(a1 - a0), t0_us=int(ev["us"][a0]), t1_us=int(ev["us"][a1 - 1])))
            info["excluded"] = not info["parts"]
            info["reason"] = None if info["parts"] else "no verified interval long enough"
            print(pair, k, j, info["n_events"], "kept", info["n_kept"], "parts", len(info["parts"]), flush=True)
            segs.append(info)
        report["pairs"][pair] = segs
    (out / "build_report.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/home/user/data/bitstamp")
    ap.add_argument("--out", default="/home/user/data/mmfm_events")
    ap.add_argument("--until-ns", type=int, default=None)
    ap.add_argument("--from-ns", type=int, default=0, help="skip connection segments that end before this time")
    ap.add_argument("--pairs", default=",".join(TICK))
    ap.add_argument("--chunk-s", type=int, default=0, help="re-anchor long segments every ~N s on a REST snapshot")
    a = ap.parse_args()
    until = a.until_ns or time.time_ns()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    build(Path(a.raw), out, until, a.pairs.split(","), from_ns=a.from_ns, chunk_s=a.chunk_s)
