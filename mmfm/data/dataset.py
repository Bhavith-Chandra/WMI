"""Split-guarded data access (EVAL.md L2–L5).

load(split, ...) returns contiguous chunks (tokens, feats) of one split:
  train / val       -> all pairs except the held-out pair, events inside the split's wall-clock range
  test              -> SEALED: needs env MMFM_UNSEAL=<frozen model hash>; each unseal is appended to test_access.log
  heldout_test      -> held-out pair in the test range (same seal)
A chunk never crosses a segment boundary (gaps) or a split boundary.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from data.tokenizer import Tokenizer

SEALED = {"test", "heldout_test"}


class SealError(RuntimeError):
    pass


def _seal_check(events: Path, split: str):
    tok = os.environ.get("MMFM_UNSEAL")
    if not tok:
        raise SealError(f"split '{split}' is sealed (EVAL.md L3). Set MMFM_UNSEAL=<frozen model hash>.")
    purpose = os.environ.get("MMFM_PURPOSE")
    if purpose not in ("M1", "M2", "M1-CI"):
        raise SealError("set MMFM_PURPOSE to the pre-registered metric being computed (M1 or M2)")
    log = events / "test_access.log"
    prev = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    if any(p["model_hash"] == tok and p["split"] == split and p.get("purpose") == purpose for p in prev):
        raise SealError(f"model {tok[:12]} already evaluated on '{split}' for {purpose} (one unseal per model/split/metric).")
    with open(log, "a") as f:
        f.write(json.dumps(dict(model_hash=tok, split=split, purpose=purpose,
                                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))) + "\n")


def raw_chunks(events: Path, split: str):
    sp = json.loads((events / "splits.json").read_text())
    if split in SEALED:
        _seal_check(events, split)
    rng = sp["test" if split in SEALED else split]
    held = sp["heldout_pair"]
    out = []
    for rel, sha in sp["files"]:
        pair = rel.split("/")[0]
        if (split == "heldout_test") != (pair == held):
            continue
        z = np.load(events / rel)
        us = z["us"]
        m = (us >= rng[0]) & (us <= rng[1])
        if m.sum() < 2:
            continue
        idx = np.flatnonzero(m)
        sl = slice(idx[0], idx[-1] + 1)  # contiguous within a segment
        ev = np.zeros(sl.stop - sl.start, dtype=[("us", "i8"), ("type", "i1"), ("side", "i1"), ("price", "f8"), ("qty", "f8")])
        for k in ev.dtype.names:
            ev[k] = z[k][sl]
        out.append(dict(pair=pair, file=rel, events=ev, state=z["state"][sl], tick=float(z["tick"])))
    return out, sp


def load(events: Path, split: str, tok: Tokenizer):
    chunks, sp = raw_chunks(events, split)
    res = []
    for c in chunks:
        tokens, feats, valid = tok.encode(c["events"], c["state"])
        # drop rows with an empty book side (no mid); split chunk there so no window spans the hole
        bad = np.flatnonzero(~valid)
        cuts = [0] + list(bad) + [len(valid)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            a = a + (1 if a in set(bad) else 0)
            if b - a > 64:
                res.append(dict(pair=c["pair"], tokens=tokens[a:b], feats=feats[a:b], us=c["events"]["us"][a:b]))
    return res, sp
