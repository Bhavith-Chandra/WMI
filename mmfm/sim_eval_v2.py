"""EVAL_v2 criteria 1-4 + 7: simulation / forecast calibration from exact book checkpoints.

python -I sim_eval_v2.py --events DIR --run runs/v2 --split val --pairs ethusd,xrpusd,dogeusd --starts 20 --paths 16
       (--forward: forward test; needs MMFM_UNSEAL=<frozen hash>, logged; --heldout: zero-shot pair)
Generators (same starts, same engine for simulators):
  MMFM-sim     event-level rollouts of MMFM v2 (dt_scale from --dt-scale, default 1)
  MMFM-direct  calibrated forecast heads (no rollout)
  B1p / B0p    pair-conditioned count generators (Markov on previous act/side; independent marginals)
Metrics per (pair, horizon): event-count ratio (median over starts of sim/real), central-80 coverage,
PIT mean, CRPS (bp), Brier P(up | real move); plus simulated seconds per wall second.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data.dataset_v2 import load, raw_chunks  # noqa: E402
from data.tokenizer_v2 import HORIZONS, Tokenizer2  # noqa: E402
from model.mmfm2 import MMFM2  # noqa: E402
from sim.book import LevelBook  # noqa: E402
from sim.rollout2 import Rollout2, context_from_history2, direct_forecast  # noqa: E402

torch.set_num_threads(4)


def load_v2(run):
    ck = torch.load(Path(run) / "model.pt", map_location="cpu", weights_only=True)
    c = ck["cfg"]
    m = MMFM2(c["vocab"], c["n_feat"], c["n_ret"], c["n_cnt"], d=c["d"], n_layers=c["n_layers"], n_heads=c["n_heads"], ctx=c["ctx"])
    m.load_state_dict(ck["state"])
    import hashlib
    return m.eval(), hashlib.sha256((Path(run) / "model.pt").read_bytes()).hexdigest()


class CountGen:
    """Pair-conditioned count baseline: token sampler fit on train events of that pair."""

    def __init__(self, tokens, vocab, markov, alpha=0.5):
        self.v, self.markov, self.a = vocab, markov, alpha
        t = tokens
        self.c = {}
        prev = t[:-1]
        nxt = t[1:]
        keys = (prev[:, 0] * 2 + prev[:, 1]) if markov else np.zeros(len(nxt), int)
        for kk in np.unique(keys):
            m = keys == kk
            n = nxt[m]
            d = dict(act=np.bincount(n[:, 0], minlength=3) + alpha)
            for a in range(3):
                ma = n[:, 0] == a
                d[("side", a)] = np.bincount(n[ma, 1], minlength=2) + alpha
                d[("dt", a)] = np.bincount(n[ma, 4], minlength=vocab["dt"]) + alpha
                for s in range(2):
                    ms = ma & (n[:, 1] == s)
                    d[("px", a, s)] = np.bincount(n[ms, 2], minlength=vocab["px"]) + alpha
                    d[("size", a)] = np.bincount(n[ma, 3], minlength=vocab["size"]) + alpha
            self.c[kk] = d

    def sample(self, prev, rng):
        kk = int(prev[0] * 2 + prev[1]) if self.markov else 0
        d = self.c.get(kk) or next(iter(self.c.values()))
        draw = lambda w: int(rng.choice(len(w), p=w / w.sum()))
        a = draw(d["act"]); s = draw(d[("side", a)])
        return np.array([a, s, draw(d[("px", a, s)]), draw(d[("size", a)]), draw(d[("dt", a)])])


class CountRollout(Rollout2):
    def __init__(self, gen, tok, ctx, n_paths, seed):
        self.gen = gen
        self.tok, self.temp, self.dt_scale = tok, 1.0, 1.0
        self.rng = np.random.default_rng(seed)
        s0 = __import__("sim.rollout2", fromlist=["book_state"]).book_state(ctx.book)
        import copy
        self.paths = [dict(book=ctx.book.copy(), inc=copy.deepcopy(ctx.inc), us=ctx.last_us, prev_us=ctx.last_us,
                           t=[ctx.last_us], mid=[0.5 * (s0[0] + s0[1])], alive=True, n_ev=0) for _ in range(n_paths)]
        self.prev = [ctx.tokens[-1].copy() for _ in range(n_paths)]
        self.t0 = ctx.last_us
        self.gen_seconds, self.n_steps = 0.0, 0

    def step(self):
        t0 = time.perf_counter()
        for i, p in enumerate(self.paths):
            if not p["alive"]:
                continue
            o = self._concrete(p, self.gen.sample(self.prev[i], self.rng))
            if o is None:
                p["alive"] = False
                continue
            row, _ = self._apply(p, o)
            self.prev[i] = row
        self.gen_seconds += time.perf_counter() - t0
        self.n_steps += 1


def sample_direct(fc, n, rng):
    """Draws of R_h from a categorical over bins (uniform within bin edges, exact 0 bin)."""
    e = np.asarray(fc["ret_edges"])
    lo = np.r_[e[0] - (e[1] - e[0]) * 3, e]
    hi = np.r_[e, e[-1] + (e[-1] - e[-2]) * 3]
    b = rng.choice(len(fc["p_ret"]), n, p=fc["p_ret"] / fc["p_ret"].sum())
    x = lo[b] + rng.random(n) * (hi[b] - lo[b])
    z = np.searchsorted(e, 0.0)
    x[b == z] = 0.0
    return x


def metrics(sims, reals, counts_sim=None, counts_real=None):
    pit = np.array([np.mean(s < r) + np.random.default_rng(0).random() * np.mean(s == r) for s, r in zip(sims, reals)])
    lo = np.array([np.quantile(s, 0.1) for s in sims])
    hi = np.array([np.quantile(s, 0.9) for s in sims])
    cov = float(np.mean((reals >= lo) & (reals <= hi)))
    crps = float(np.mean([np.mean(np.abs(s - r)) - 0.5 * np.mean(np.abs(s[:, None] - s[None])) for s, r in zip(sims, reals)]))
    p_up = np.array([np.mean(s > 0) / max(np.mean(s != 0), 1e-9) for s in sims])
    nz = reals != 0
    out = dict(n=len(reals), coverage80=cov, pit_mean=float(pit.mean()), crps_bp=crps,
               brier_up_given_move=float(np.mean((p_up[nz] - (reals[nz] > 0)) ** 2)) if nz.any() else None,
               real_abs_med=float(np.median(np.abs(reals))), sim_abs_med=float(np.median(np.abs(np.concatenate(sims)))))
    if counts_sim is not None:
        out["event_ratio_median"] = float(np.median([np.median(cs) / max(cr, 1) for cs, cr in zip(counts_sim, counts_real)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--pairs", default="ethusd,xrpusd,dogeusd")
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--paths", type=int, default=16)
    ap.add_argument("--dt-scale", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--gens", default="MMFM-sim,MMFM-direct,B1p,B0p")
    ap.add_argument("--train-events", default=None, help="events dir whose TRAIN split fits the count baselines")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    events = Path(a.events)
    model, mhash = load_v2(a.run)
    tok = Tokenizer2.from_json(json.loads((Path(a.run) / "tokenizer.json").read_text()))
    gens = a.gens.split(",")
    tr_dir = Path(a.train_events or a.events)
    train, _ = load(tr_dir, "train", tok)
    purpose = "sim" + (f"-dt{a.dt_scale}" if a.dt_scale != 1 else "")
    chunks, sp = raw_chunks(events, a.split, include_heldout=("only" if a.heldout else False), forward=a.forward,
                            purpose=purpose + ("-heldout" if a.heldout else ""))
    rng = np.random.default_rng(0)
    res = dict(model_hash=mhash, split="forward" if a.forward else a.split, heldout=a.heldout, horizons=HORIZONS,
               dt_scale=a.dt_scale, temp=a.temp, per_pair={})
    for pair in a.pairs.split(","):
        cs = [c for c in chunks if c["pair"] == pair]
        if not cs:
            continue
        tr_tokens = np.concatenate([c["tokens"] for c in train if c["pair"] == pair]) if any(c["pair"] == pair for c in train) \
            else np.concatenate([c["tokens"] for c in train])
        cg = dict(B1p=CountGen(tr_tokens, tok.vocab, True), B0p=CountGen(tr_tokens, tok.vocab, False))
        starts = []
        for c in cs:
            z = np.load(events / c["file"])
            lo, hi = c["events"]["us"][0], c["events"]["us"][-1]
            for j, idx in enumerate(z["ck_idx"]):
                if idx > 1200 and lo <= z["us"][idx] <= hi - max(HORIZONS) * 1e6:
                    starts.append((c["file"], j, int(idx)))
        if not starts:
            continue
        pick = [starts[k] for k in sorted(rng.choice(len(starts), min(a.starts, len(starts)), replace=False))]
        acc = {g: dict(sims={h: [] for h in HORIZONS}, cnt={h: [] for h in HORIZONS}, wall=0.0) for g in gens}
        reals = {h: [] for h in HORIZONS}
        rcnt = {h: [] for h in HORIZONS}
        for f, j, idx in pick:
            z = np.load(events / f)
            tick = float(z["tick"])
            evs = np.zeros(len(z["us"]), dtype=[("us", "i8"), ("type", "i1"), ("side", "i1"), ("price", "f8"), ("qty", "f8")])
            for k in evs.dtype.names:
                evs[k] = z[k]
            st, lvq = z["state"], z["lvq"]
            us0 = int(evs["us"][idx - 1])
            mid = 0.5 * (st[:, 0] + st[:, 1])
            b = LevelBook(tick)
            for arr, d in ((z["ck_bids"][j], b.bids), (z["ck_asks"][j], b.asks)):
                for p, q in arr:
                    if np.isfinite(p):
                        d[int(round(p / tick))] = float(q)
            mid0 = 0.5 * (max(b.bids) + min(b.asks)) * tick
            ok = True
            rr = {}
            for h in HORIZONS:
                jj = np.searchsorted(evs["us"], us0 + h * 1e6, side="right")
                if jj >= len(evs):
                    ok = False
                    break
                rr[h] = (float(np.log(mid[jj] / mid0) * 1e4), int(jj - idx))
            if not ok:
                continue
            for h in HORIZONS:
                reals[h].append(rr[h][0])
                rcnt[h].append(rr[h][1])
            ctx = context_from_history2(evs[idx - 1100:idx], st[idx - 1100:idx], lvq[idx - 1100:idx], tok, b, model.cfg["ctx"])
            for g in gens:
                t0 = time.perf_counter()
                if g == "MMFM-direct":
                    fc = direct_forecast(model, tok, ctx)
                    for k, h in enumerate(HORIZONS):
                        acc[g]["sims"][h].append(sample_direct(fc[k], 400, rng))
                        pc = fc[k]["p_cnt"] / fc[k]["p_cnt"].sum()
                        acc[g]["cnt"][h].append(np.array([fc[k]["cnt_values"][np.searchsorted(np.cumsum(pc), 0.5)]]))
                else:
                    if g == "MMFM-sim":
                        R = Rollout2(model, tok, ctx, n_paths=a.paths, seed=idx, temperature=a.temp, dt_scale=a.dt_scale)
                    else:
                        R = CountRollout(cg[g], tok, ctx, a.paths, idx)
                    R.run(max(HORIZONS), max_steps=60000)
                    m = R.mid_grid(np.array(HORIZONS))
                    cnts = R.counts(HORIZONS)
                    for k, h in enumerate(HORIZONS):
                        x = np.log(m[:, k] / mid0) * 1e4
                        acc[g]["sims"][h].append(x[np.isfinite(x)] if np.isfinite(x).any() else np.zeros(1))
                        acc[g]["cnt"][h].append(cnts[:, k])
                acc[g]["wall"] += time.perf_counter() - t0
        summ = {}
        for g in gens:
            s = {}
            for h in HORIZONS:
                s[str(h)] = metrics(acc[g]["sims"][h], np.array(reals[h]), acc[g]["cnt"][h], rcnt[h])
            s["sim_seconds_per_wall_second"] = max(HORIZONS) * len(reals[HORIZONS[0]]) / max(acc[g]["wall"], 1e-9)
            summ[g] = s
        res["per_pair"][pair] = summ
        for g in gens:
            print(pair, g, " | ".join(f"{h}s ev{summ[g][str(h)].get('event_ratio_median', float('nan')):.2f} cov{summ[g][str(h)]['coverage80']:.2f} "
                                      f"pit{summ[g][str(h)]['pit_mean']:.2f} crps{summ[g][str(h)]['crps_bp']:.3f}" for h in HORIZONS),
                  f"rt {summ[g]['sim_seconds_per_wall_second']:.2f}", flush=True)
    res["_provenance"] = dict(git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                              argv=sys.argv, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
