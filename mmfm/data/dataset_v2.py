"""Split-guarded v2 data access (EVAL_v2.md): train / val from the v2 design build only.
The v2 forward test lives in a separate build and is read only by evaluate_v2.py with a logged unseal."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from data.tokenizer_v2 import Tokenizer2

EVD = [("us", "i8"), ("type", "i1"), ("side", "i1"), ("price", "f8"), ("qty", "f8")]


class SealError(RuntimeError):
    pass


def check_forward_seal(events: Path, purpose: str):
    tok = os.environ.get("MMFM_UNSEAL")
    if not tok:
        raise SealError("forward test is sealed: set MMFM_UNSEAL=<frozen v2 model hash>")
    log = events / "test_access.log"
    prev = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    if any(p["model_hash"] == tok and p["purpose"] == purpose for p in prev):
        raise SealError(f"already unsealed for {purpose} with this model")
    with open(log, "a") as f:
        f.write(json.dumps(dict(model_hash=tok, purpose=purpose, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))) + "\n")


def raw_chunks(events: Path, split: str, include_heldout=False, forward=False, purpose=None):
    sp = json.loads((events / "splits.json").read_text())
    if forward:
        check_forward_seal(events, purpose)
        rng = sp["forward"]
    elif split in ("train", "val"):
        rng = sp[split]
    else:
        raise SealError(f"unknown/sealed split {split}")
    held = sp["heldout_pair"]
    out = []
    for rel, _ in sp["files"]:
        pair = rel.split("/")[0]
        if (pair == held) and not include_heldout:
            continue
        if include_heldout == "only" and pair != held:
            continue
        z = np.load(events / rel)
        us = z["us"]
        m = (us >= rng[0]) & (us <= rng[1])
        if m.sum() < 2:
            continue
        idx = np.flatnonzero(m)
        sl = slice(idx[0], idx[-1] + 1)
        ev = np.zeros(sl.stop - sl.start, dtype=EVD)
        for k in ev.dtype.names:
            ev[k] = z[k][sl]
        out.append(dict(pair=pair, file=rel, events=ev, state=z["state"][sl], lvq=z["lvq"][sl], tick=float(z["tick"]),
                        offset=sl.start))
    return out, sp


def load(events: Path, split: str, tok: Tokenizer2, **kw):
    chunks, sp = raw_chunks(events, split, **kw)
    res = []
    for i in range(len(chunks)):
        c = chunks[i]
        chunks[i] = None  # free raw arrays as we go (v3: ~30M train events)
        tokens, feats, valid = tok.encode(c["events"], c["state"], c["lvq"])
        rt, ct = tok.encode_targets(c["events"], c["state"])
        rt, ct = rt.astype(np.int16), ct.astype(np.int16)  # bins < 64, -1 = masked; widened to int64 at batch time
        bad = set(np.flatnonzero(~valid).tolist())
        cuts = [0] + sorted(bad) + [len(valid)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            a = a + (1 if a in bad else 0)
            if b - a > 64:
                res.append(dict(pair=c["pair"], tokens=tokens[a:b], feats=feats[a:b], rt=rt[a:b], ct=ct[a:b],
                                us=c["events"]["us"][a:b].copy(), file=c["file"], offset=c["offset"] + a))
    return res, sp
