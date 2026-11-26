"""Freeze wall-clock splits (EVAL.md §1). Run ONCE before training; refuses to overwrite.

python -I data/splits.py --events /home/user/data/mmfm_events
Writes <events>/splits.json: boundaries (us), embargo, held-out pair, per-file sha256 list and its hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HELDOUT = "linkusd"
EMBARGO_US = 10 * 60 * 1_000_000
FRAC = (0.70, 0.10, 0.20)


def make(events: Path):
    rep = json.loads((events / "build_report.json").read_text())
    segs = [s for p, ss in rep["pairs"].items() for s in ss if not s["excluded"]]
    t0 = min(s["t0_us"] for s in segs)
    t1 = max(s["t1_us"] for s in segs)
    span = t1 - t0
    b1 = t0 + int(FRAC[0] * span)
    b2 = b1 + int(FRAC[1] * span)
    files = sorted((s["file"], s["sha256"]) for s in segs)
    h = hashlib.sha256(json.dumps(files).encode()).hexdigest()
    return dict(t0_us=t0, t1_us=t1, train=[t0, b1 - EMBARGO_US], val=[b1, b2 - EMBARGO_US], test=[b2, t1],
                embargo_us=EMBARGO_US, heldout_pair=HELDOUT, files=files, files_hash=h,
                build_until_ns=rep["until_ns"], rule="EVAL.md §1: 70/10/20 by wall clock, 10-min embargo before val and test")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="/home/user/data/mmfm_events")
    ap.add_argument("--dev", action="store_true", help="dev build: no dev-cut check")
    a = ap.parse_args()
    p = Path(a.events) / "splits.json"
    if p.exists():
        sys.exit(f"{p} exists: splits are frozen (EVAL.md §1). Delete only with a committed reason.")
    s = make(Path(a.events))
    dev = Path("/home/user/data/dev_cut_ns.txt")
    if dev.exists() and not a.dev:
        dev_us = int(dev.read_text()) // 1000
        if s["train"][1] < dev_us + EMBARGO_US:
            sys.exit(f"train end {s['train'][1]} < dev cut + embargo {dev_us + EMBARGO_US}: data seen during development "
                     "would fall into val/test (EVAL.md addendum 1). Collect longer before freezing.")
        s["dev_cut_us"] = dev_us
    p.write_text(json.dumps(s, indent=1))
    print(json.dumps({k: v for k, v in s.items() if k != "files"}, indent=1))
