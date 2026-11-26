"""Train MMFM (or the B2 baseline) on train, select on val only (EVAL.md L2). Never touches test.

python -I train.py --events /home/user/data/mmfm_events --out runs/pilot --model mmfm --minutes 60
Writes <out>/{tokenizer.json, model.pt, selection_log.json, config.json}; model hash = sha256 of model.pt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data.dataset import load, raw_chunks  # noqa: E402   (no import of any test loader: L4)
from data.tokenizer import Tokenizer  # noqa: E402
from model.mmfm import MMFM, FeatMLP, bits  # noqa: E402

torch.set_num_threads(4)


def windows(chunks, T, rng, n):
    lens = np.array([len(c["tokens"]) - T for c in chunks])
    ok = np.flatnonzero(lens > 0)
    w = lens[ok] / lens[ok].sum()
    ci = rng.choice(ok, n, p=w)
    toks, fts = [], []
    for c in ci:
        s = rng.integers(0, lens[c])
        toks.append(chunks[c]["tokens"][s:s + T])
        fts.append(chunks[c]["feats"][s:s + T])
    return torch.from_numpy(np.stack(toks)), torch.from_numpy(np.stack(fts))


@torch.no_grad()
def evaluate(model, chunks, T, max_windows=400):
    """Deterministic tiled evaluation: non-overlapping windows, bits/event per field (scores events 2..T)."""
    model.eval()
    tot, n = np.zeros(5), 0
    tiles = [(c, s) for c in chunks for s in range(0, len(c["tokens"]) - T, T)]
    step = max(1, len(tiles) // max_windows)
    for c, s in tiles[::step]:
        tk = torch.from_numpy(c["tokens"][s:s + T])[None]
        ft = torch.from_numpy(c["feats"][s:s + T])[None]
        b = bits(model.logp(tk, ft))[0].numpy()
        tot += b.sum(0)
        n += len(b)
    model.train()
    return dict(bits_per_event=float(tot.sum() / n), per_field=dict(zip(("act", "side", "px", "size", "dt"), (tot / n).tolist())),
                n_events=int(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="/home/user/data/mmfm_events")
    ap.add_argument("--out", default="runs/pilot")
    ap.add_argument("--model", choices=["mmfm", "b2"], default="mmfm")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=None, help="reuse a tokenizer.json (must be fit on the same train split)")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    events = Path(a.events)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    if a.tokenizer:
        tok = Tokenizer.from_json(json.loads(Path(a.tokenizer).read_text()))
    else:
        raw, sp = raw_chunks(events, "train")
        tok = Tokenizer.fit([(c["events"], c["state"]) for c in raw],
                            dict(split="train", files_hash=sp["files_hash"], range=sp["train"]))
    sp = json.loads((events / "splits.json").read_text())
    assert tok.fit_split["split"] == "train" and tok.fit_split["files_hash"] == sp["files_hash"], "tokenizer not fit on this train split (L1)"
    (out / "tokenizer.json").write_text(json.dumps(tok.to_json()))
    train, _ = load(events, "train", tok)
    val, _ = load(events, "val", tok)
    print(f"train chunks {len(train)} events {sum(len(c['tokens']) for c in train)} | val events {sum(len(c['tokens']) for c in val)}", flush=True)

    nf = train[0]["feats"].shape[1]
    model = (MMFM(tok.vocab, nf, d=a.d, n_layers=a.layers, ctx=a.ctx) if a.model == "mmfm" else FeatMLP(tok.vocab, nf, d=a.d))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    n_par = sum(p.numel() for p in model.parameters())
    log = dict(config=vars(a), n_params=n_par, selection_split="val", evals=[])
    best, t0, step = math.inf, time.time(), 0
    while time.time() - t0 < a.minutes * 60:
        tk, ft = windows(train, a.ctx, rng, a.batch)
        loss = bits(model.logp(tk, ft)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % a.eval_every == 0:
            ev = evaluate(model, val, a.ctx)
            ev.update(step=step, train_bits=float(loss), minutes=(time.time() - t0) / 60)
            log["evals"].append(ev)
            if ev["bits_per_event"] < best:
                best = ev["bits_per_event"]
                torch.save(dict(state=model.state_dict(), cfg=model.cfg, kind=a.model), out / "model.pt")
                log["best"] = ev
            (out / "selection_log.json").write_text(json.dumps(log, indent=1))
            print(f"step {step} train {float(loss):.3f} val {ev['bits_per_event']:.4f} best {best:.4f} {ev['minutes']:.1f}min", flush=True)
    log["model_hash"] = hashlib.sha256((out / "model.pt").read_bytes()).hexdigest()
    log["n_steps"] = step
    (out / "selection_log.json").write_text(json.dumps(log, indent=1))
    print("model hash", log["model_hash"])


if __name__ == "__main__":
    main()
