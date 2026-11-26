"""Forward-test split (EVAL_v2): forward = [freeze + 10 min, end], all pairs (linkusd = zero-shot)."""
import hashlib
import json
import sys
from pathlib import Path

ev, start_us, end_us = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
p = ev / "splits.json"
if p.exists():
    sys.exit("frozen")
rep = json.loads((ev / "build_report.json").read_text())
parts = [pp for ss in rep["pairs"].values() for s in ss for pp in s.get("parts", [])]
files = sorted((pp["file"], pp["sha256"]) for pp in parts)
sp = dict(forward=[start_us, end_us], train=[0, 0], val=[0, 0], heldout_pair="linkusd", files=files,
          files_hash=hashlib.sha256(json.dumps(files).encode()).hexdigest(), rule="EVAL_v2.md forward test")
p.write_text(json.dumps(sp, indent=1))
print({k: v for k, v in sp.items() if k != "files"})
