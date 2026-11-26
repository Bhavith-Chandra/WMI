"""EVAL_v2 criterion 5/6: next-event NLL, MMFM-v2 vs tuned B2-v2 vs B1/B0 on IDENTICAL tiles,
5-minute wall-clock block bootstrap. Modes: --split val | --forward [--heldout] (forward needs MMFM_UNSEAL).

python -I evaluate_v2.py --events DIR --mmfm runs/v2 --b2 runs/v2_b2 --split val --out results/v2_m1_val.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data.dataset_v2 import load  # noqa: E402
from data.tokenizer_v2 import Tokenizer2  # noqa: E402
from model.baselines import B0, B1  # noqa: E402
from model.mmfm import FeatMLP  # noqa: E402
from sim_eval_v2 import load_v2  # noqa: E402

torch.set_num_threads(4)
BLOCK_US = 300_000_000


def load_b2(run):
    ck = torch.load(Path(run) / "model.pt", map_location="cpu", weights_only=True)
    c = ck["cfg"]
    m = FeatMLP(c["vocab"], c["n_feat"], d=c["d"], K=c["K"])
    m.load_state_dict(ck["state"])
    return m.eval(), hashlib.sha256((Path(run) / "model.pt").read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--mmfm", required=True)
    ap.add_argument("--b2", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--train-events", default=None)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--max-tiles", type=int, default=1500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = Tokenizer2.from_json(json.loads((Path(a.mmfm) / "tokenizer.json").read_text()))
    assert json.loads((Path(a.b2) / "tokenizer.json").read_text())["hash"] == tok.to_json()["hash"]
    m, mh = load_v2(a.mmfm)
    b2, bh = load_b2(a.b2)
    train, _ = load(Path(a.train_events or a.events), "train", tok)
    chunks, _ = load(Path(a.events), a.split, tok, include_heldout=("only" if a.heldout else False), forward=a.forward,
                     purpose="M1" + ("-heldout" if a.heldout else ""))
    T = a.T
    tiles = [(i, s) for i, c in enumerate(chunks) for s in range(0, len(c["tokens"]) - T, T)]
    tiles = tiles[:: max(1, len(tiles) // a.max_tiles)]
    b0, b1 = B0().fit(train, tok.vocab), B1().fit(train, tok.vocab)
    scores = {}
    for name, f in (("B0", lambda tk, ft: -b0.logp(dict(tokens=tk, feats=ft)) / math.log(2)),
                    ("B1", lambda tk, ft: -b1.logp(dict(tokens=tk, feats=ft)) / math.log(2))):
        scores[name] = np.array([f(chunks[i]["tokens"][s:s + T], chunks[i]["feats"][s:s + T]).sum() for i, s in tiles])
    with torch.no_grad():
        for name, model in (("MMFM-v2", m), ("B2-v2", b2)):
            scores[name] = np.array([float((-model.logp(torch.from_numpy(chunks[i]["tokens"][s:s + T])[None],
                                                         torch.from_numpy(chunks[i]["feats"][s:s + T])[None]) / math.log(2)).sum())
                                     for i, s in tiles])
    n_ev = np.full(len(tiles), T - 1)
    groups = np.array([hash((chunks[i]["pair"], int(chunks[i]["us"][s] // BLOCK_US))) for i, s in tiles])
    ug = np.unique(groups)
    rng = np.random.default_rng(0)
    boots = {k: [] for k in scores}
    for _ in range(2000):
        idx = np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(ug, len(ug))])
        for k in scores:
            boots[k].append(scores[k][idx].sum() / n_ev[idx].sum())
    res = dict(split="forward" if a.forward else a.split, heldout=a.heldout, n_tiles=len(tiles), n_events=int(n_ev.sum()),
               n_blocks=int(len(ug)), model_hashes=dict(mmfm=mh, b2=bh), tokenizer_hash=tok.to_json()["hash"], models={})
    for k in scores:
        res["models"][k] = dict(bits_per_event=float(scores[k].sum() / n_ev.sum()), ci95=np.percentile(boots[k], [2.5, 97.5]).tolist())
    d = np.array(boots["MMFM-v2"]) - np.array(boots["B2-v2"])
    res["mmfm_minus_b2"] = dict(delta=res["models"]["MMFM-v2"]["bits_per_event"] - res["models"]["B2-v2"]["bits_per_event"],
                                ci95=np.percentile(d, [2.5, 97.5]).tolist())
    res["_provenance"] = dict(git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                              argv=sys.argv, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k, v in res["models"].items():
        print(f"{k:10s} {v['bits_per_event']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")
    print("MMFM-v2 - B2-v2:", res["mmfm_minus_b2"], "blocks", len(ug))


if __name__ == "__main__":
    main()
