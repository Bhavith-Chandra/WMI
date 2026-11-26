"""Freeze the v3 design splits (EVAL_v3.md): cut U from mmfm_v3_until_ns; val = [U-60min, U]; 10-min embargo;
train = everything <= U-70min; held-out pair linkusd. The v3 test is a forward test built after the freeze."""
import hashlib
import json
import sys
from pathlib import Path

ev = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/user/data/mmfm_v3")
p = ev / "splits.json"
if p.exists():
    sys.exit("frozen")
U = int(Path("/home/user/data/mmfm_v3_until_ns").read_text()) // 1000
rep = json.loads((ev / "build_report.json").read_text())
parts = [p for ss in rep["pairs"].values() for s in ss for p in s.get("parts", [])]
files = sorted((p["file"], p["sha256"]) for p in parts)
t0 = min(p["t0_us"] for p in parts)
H = 3_600_000_000
sp = dict(train=[t0, U - H - 600_000_000], val=[U - H, U], test=[U, U + 1], heldout_pair="linkusd", embargo_us=600_000_000,
          files=files, files_hash=hashlib.sha256(json.dumps(files).encode()).hexdigest(),
          rule="EVAL_v3.md (test = forward test, separate build)")
p.write_text(json.dumps(sp, indent=1))
print({k: v for k, v in sp.items() if k != "files"})
