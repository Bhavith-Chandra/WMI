"""Train MMFM v2 on v2-train, select + calibrate on v2-val only (EVAL_v2.md). Never reads the forward test.

python -I train_v2.py --events /home/user/data/mmfm_v2 --out runs/v2 --minutes 90
Loss (bits) = event NLL (sum of 5 fields) + lam * mean forecast NLL over (R_h, N_h, h in 1/5/10 s).
After training: per-head temperatures fit on val (log-loss), stored in the checkpoint.
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
from data.dataset_v2 import load, raw_chunks  # noqa: E402
from data.tokenizer_v2 import HORIZONS, Tokenizer2  # noqa: E402
from model.mmfm import FeatMLP  # noqa: E402
from model.mmfm2 import MMFM2  # noqa: E402

torch.set_num_threads(4)


def windows(chunks, T, rng, n):
    lens = np.array([len(c["tokens"]) - T for c in chunks])
    ok = np.flatnonzero(lens > 0)
    ci = rng.choice(ok, n, p=lens[ok] / lens[ok].sum())
    out = {k: [] for k in ("tokens", "feats", "rt", "ct")}
    for c in ci:
        s = rng.integers(0, lens[c])
        for k in out:
            out[k].append(chunks[c][k][s:s + T])
    return {k: torch.from_numpy(np.stack(v)) for k, v in out.items()}


@torch.no_grad()
def evaluate(model, chunks, T, max_tiles=500):
    model.eval()
    tiles = [(c, s) for c in chunks for s in range(0, len(c["tokens"]) - T, T)]
    tiles = tiles[:: max(1, len(tiles) // max_tiles)]
    ev, fr, fc, n = 0.0, np.zeros(len(HORIZONS)), np.zeros(len(HORIZONS)), 0
    for c, s in tiles:
        b = {k: torch.from_numpy(c[k][s:s + T])[None] for k in ("tokens", "feats", "rt", "ct")}
        e, lr, lc = model.losses(b["tokens"], b["feats"], b["rt"], b["ct"])
        ev += float(e.sum(-1).mean())
        fr += lr.numpy()
        fc += lc.numpy()
        n += 1
    model.train()
    return dict(event_bits=ev / n, ret_bits=(fr / n).tolist(), cnt_bits=(fc / n).tolist())


@torch.no_grad()
def calibrate(model, chunks, T, max_tiles=600):
    """Fit one temperature per forecast head on val (grid search on log-loss)."""
    model.eval()
    tiles = [(c, s) for c in chunks for s in range(0, len(c["tokens"]) - T, T)]
    tiles = tiles[:: max(1, len(tiles) // max_tiles)]
    H, R, C = [], [], []
    for c, s in tiles:
        H.append(model.hidden(torch.from_numpy(c["tokens"][s:s + T])[None], torch.from_numpy(c["feats"][s:s + T])[None])[0])
        R.append(torch.from_numpy(c["rt"][s:s + T]))
        C.append(torch.from_numpy(c["ct"][s:s + T]))
    H, R, C = torch.cat(H), torch.cat(R), torch.cat(C)
    grid = torch.exp(torch.linspace(math.log(0.5), math.log(3.0), 41))
    res = {}
    for name, heads, tgt, buf in (("ret", model.ret, R, model.t_ret), ("cnt", model.cnt, C, model.t_cnt)):
        for k, head in enumerate(heads):
            m = tgt[:, k] >= 0
            lg = head(H[m]).float()
            y = tgt[m, k]
            nll = [float(-torch.log_softmax(lg / t, -1).gather(-1, y[:, None]).mean()) for t in grid]
            raw = float(-torch.log_softmax(lg, -1).gather(-1, y[:, None]).mean())
            j = int(np.argmin(nll))
            buf[k] = grid[j]
            res[f"{name}_{HORIZONS[k]}"] = dict(T=float(grid[j]), nll_raw_nats=raw, nll_cal_nats=nll[j])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="/home/user/data/mmfm_v2")
    ap.add_argument("--out", default="runs/v2")
    ap.add_argument("--d", type=int, default=160)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--minutes", type=float, default=90)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=["mmfm2", "b2"], default="mmfm2")
    ap.add_argument("--tokenizer", default=None)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    events = Path(a.events)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    if a.tokenizer:
        tok = Tokenizer2.from_json(json.loads(Path(a.tokenizer).read_text()))
        sp = json.loads((events / "splits.json").read_text())
        assert tok.fit_split["files_hash"] == sp["files_hash"] and tok.fit_split["split"] == "train"
    else:
        raw, sp = raw_chunks(events, "train")
        tok = Tokenizer2.fit([(c["events"], c["state"], c["lvq"]) for c in raw], dict(split="train", files_hash=sp["files_hash"], range=sp["train"]))
        del raw
    (out / "tokenizer.json").write_text(json.dumps(tok.to_json()))
    train, _ = load(events, "train", tok)
    val, _ = load(events, "val", tok)
    print(f"train events {sum(len(c['tokens']) for c in train):,} | val events {sum(len(c['tokens']) for c in val):,}", flush=True)
    nret = [tok.n_ret_bins(k) for k in range(len(HORIZONS))]
    ncnt = [tok.n_cnt_bins(k) for k in range(len(HORIZONS))]
    if a.model == "b2":
        model = FeatMLP(tok.vocab, train[0]["feats"].shape[1], d=a.d)
        model.losses = lambda tk, ft, rt, ct: (-model.logp(tk, ft) / math.log(2), torch.zeros(len(HORIZONS)), torch.zeros(len(HORIZONS)))
    else:
        model = MMFM2(tok.vocab, train[0]["feats"].shape[1], nret, ncnt, d=a.d, n_layers=a.layers, ctx=a.ctx)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    def lr_at(frac):  # wall-clock schedule: 3% linear warmup, cosine decay to 5% of max over the time budget
        return a.lr * (frac / 0.03 if frac < 0.03 else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min((frac - 0.03) / 0.97, 1.0))))
    log = dict(config=vars(a), n_params=sum(p.numel() for p in model.parameters()), selection_split="val", evals=[])
    best, t0, step = math.inf, time.time(), 0
    while time.time() - t0 < a.minutes * 60:
        b = windows(train, a.ctx, rng, a.batch)
        ev, lr_, lc = model.losses(b["tokens"], b["feats"], b["rt"], b["ct"])
        loss = ev.sum(-1).mean() + a.lam * (lr_.sum() + lc.sum()) / len(HORIZONS)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g_ in opt.param_groups:
            g_["lr"] = lr_at((time.time() - t0) / (a.minutes * 60))
        opt.step()
        step += 1
        if step % a.eval_every == 0:
            e = evaluate(model, val, a.ctx)
            score = e["event_bits"] + a.lam * (sum(e["ret_bits"]) + sum(e["cnt_bits"])) / len(HORIZONS)
            e.update(step=step, score=score, minutes=(time.time() - t0) / 60, lr=opt.param_groups[0]["lr"])
            log["evals"].append(e)
            if score < best:
                best = score
                torch.save(dict(state=model.state_dict(), cfg=model.cfg, kind=a.model), out / "model_uncal.pt")
                log["best"] = e
            (out / "selection_log.json").write_text(json.dumps(log, indent=1))
            print(f"step {step} ev {e['event_bits']:.3f} ret {np.round(e['ret_bits'], 3)} cnt {np.round(e['cnt_bits'], 3)} best {best:.3f} {e['minutes']:.1f}m", flush=True)
    ck = torch.load(out / "model_uncal.pt", weights_only=True)
    model.load_state_dict(ck["state"])
    log["calibration"] = calibrate(model, val, a.ctx) if a.model == "mmfm2" else None
    torch.save(dict(state=model.state_dict(), cfg=model.cfg, kind=a.model), out / "model.pt")
    log["model_hash"] = hashlib.sha256((out / "model.pt").read_bytes()).hexdigest()
    log["n_steps"] = step
    (out / "selection_log.json").write_text(json.dumps(log, indent=1))
    print("calibration", json.dumps(log["calibration"]), "\nmodel hash", log["model_hash"])


if __name__ == "__main__":
    main()
