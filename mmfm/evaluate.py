"""M1 next-event NLL: MMFM vs B0 / B1 / B2 on IDENTICAL tiles, with 5-min block-bootstrap CIs.

python -I evaluate.py --events DIR --runs runs/X [runs/Y ...] --split val
       (split=test / heldout_test require MMFM_UNSEAL=<model hash> — see data/dataset.py)
Every model scores exactly the same events (events 2..T of each non-overlapping T-tile).
Writes results/m1_<split>.json with provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data.dataset import load  # noqa: E402
from data.tokenizer import Tokenizer  # noqa: E402
from model.baselines import B0, B1  # noqa: E402
from model.mmfm import MMFM, FeatMLP, bits  # noqa: E402

torch.set_num_threads(4)
BLOCK_US = 5 * 60 * 1_000_000


def tiles(chunks, T, max_tiles):
    t = [(i, s) for i, c in enumerate(chunks) for s in range(0, len(c["tokens"]) - T, T)]
    step = max(1, len(t) // max_tiles)
    return t[::step]


def load_model(run):
    ck = torch.load(Path(run) / "model.pt", map_location="cpu", weights_only=True)
    cfg = ck["cfg"]
    m = MMFM(cfg["vocab"], cfg["n_feat"], d=cfg["d"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"], ctx=cfg["ctx"]) \
        if ck["kind"] == "mmfm" else FeatMLP(cfg["vocab"], cfg["n_feat"], d=cfg["d"], K=cfg["K"])
    m.load_state_dict(ck["state"])
    return m.eval(), hashlib.sha256((Path(run) / "model.pt").read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--max-tiles", type=int, default=600)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    events = Path(a.events)
    tok = Tokenizer.from_json(json.loads((Path(a.runs[0]) / "tokenizer.json").read_text()))
    for r in a.runs[1:]:
        assert json.loads((Path(r) / "tokenizer.json").read_text())["hash"] == tok.to_json()["hash"], "runs use different tokenizers"
    train, sp = load(events, "train", tok)
    ev_chunks, _ = load(events, a.split, tok)
    T = a.T
    til = tiles(ev_chunks, T, a.max_tiles)
    scores = {}
    b0, b1 = B0().fit(train, tok.vocab), B1().fit(train, tok.vocab)
    for name, bl in (("B0_zero_intelligence", b0), ("B1_markov", b1)):
        scores[name] = [bits(torch.from_numpy(bl.logp(dict(tokens=ev_chunks[i]["tokens"][s:s + T], feats=ev_chunks[i]["feats"][s:s + T])))).numpy()
                        for i, s in til]
    hashes = {}
    for r in a.runs:
        m, h = load_model(r)
        name = f"{Path(r).name}:{type(m).__name__}"
        hashes[name] = h
        with torch.no_grad():
            scores[name] = [bits(m.logp(torch.from_numpy(ev_chunks[i]["tokens"][s:s + T])[None],
                                        torch.from_numpy(ev_chunks[i]["feats"][s:s + T])[None]))[0].numpy() for i, s in til]
    # block bootstrap: tiles grouped by (chunk, 5-min block of the tile start position ~ proxy by tile index)
    groups = np.array([hash((ev_chunks[i]["pair"], s // 20000)) for i, s in til])
    ug = np.unique(groups)
    rng = np.random.default_rng(0)
    per_tile = {k: np.array([x.sum() for x in v]) for k, v in scores.items()}
    n_ev = np.array([len(x) for x in scores["B0_zero_intelligence"]])
    res = dict(split=a.split, n_tiles=len(til), n_events=int(n_ev.sum()), T=T, models={}, model_hashes=hashes,
               tokenizer_hash=tok.to_json()["hash"], splits_files_hash=sp["files_hash"])
    best_base = None
    for k in scores:
        allb = np.concatenate(scores[k])
        res["models"][k] = dict(bits_per_event=float(allb.sum() / len(allb)),
                                per_field=dict(zip(("act", "side", "px", "size", "dt"), (allb.mean(0)).tolist())))
    names = list(scores)
    boots = {k: [] for k in names}
    for _ in range(2000):
        gs = rng.choice(ug, len(ug))
        idx = np.concatenate([np.flatnonzero(groups == g) for g in gs])
        for k in names:
            boots[k].append(per_tile[k][idx].sum() / n_ev[idx].sum())
    for k in names:
        res["models"][k]["ci95"] = np.percentile(boots[k], [2.5, 97.5]).tolist()
    base_names = [k for k in names if k.startswith("B") or "FeatMLP" in k]
    for k in names:
        if "MMFM" in k:
            bb = min(base_names, key=lambda x: res["models"][x]["bits_per_event"])
            d = np.array(boots[k]) - np.array(boots[bb])
            res["models"][k]["vs_best_baseline"] = dict(baseline=bb, delta_bits=float(res["models"][k]["bits_per_event"] - res["models"][bb]["bits_per_event"]),
                                                       delta_ci95=np.percentile(d, [2.5, 97.5]).tolist())
    res["_provenance"] = dict(git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                              argv=sys.argv, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out = Path(a.out or ROOT / "results" / f"m1_{a.split}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    for k, v in res["models"].items():
        print(f"{k:40s} {v['bits_per_event']:.4f} bits/event  CI95 {np.round(v['ci95'], 4)}  {v.get('vs_best_baseline', '')}")


if __name__ == "__main__":
    main()
