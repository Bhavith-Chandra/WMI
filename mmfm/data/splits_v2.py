"""Freeze the v2 design splits (EVAL_v2.md): train <= 07:00, 10-min embargo, val 07:10-08:00 UTC. No test here:
the v2 test is a forward test built later from data recorded after the freeze."""
import hashlib
import json
import sys
from pathlib import Path


def us(hms):
    import datetime as dt
    return int(dt.datetime(2026, 11, 26, *map(int, hms.split(":")), tzinfo=dt.timezone.utc).timestamp() * 1e6)


ev = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/user/data/mmfm_v2")
p = ev / "splits.json"
if p.exists():
    sys.exit("frozen")
rep = json.loads((ev / "build_report.json").read_text())
parts = [p for ss in rep["pairs"].values() for s in ss for p in s.get("parts", [])]
files = sorted((p["file"], p["sha256"]) for p in parts)
t0 = min(p["t0_us"] for p in parts)
sp = dict(train=[t0, us("07:00:00")], val=[us("07:10:00"), us("08:00:00")], test=[us("08:00:00"), us("08:00:01")],
          heldout_pair="linkusd", embargo_us=600_000_000, files=files,
          files_hash=hashlib.sha256(json.dumps(files).encode()).hexdigest(), rule="EVAL_v2.md (test = forward test, separate build)")
p.write_text(json.dumps(sp, indent=1))
print({k: v for k, v in sp.items() if k != "files"})
