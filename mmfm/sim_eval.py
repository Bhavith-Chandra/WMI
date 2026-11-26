"""M2 simulation realism + M3 short-horizon prediction + M5 latency, from exact id-book checkpoints.

python -I sim_eval.py --events DIR --run runs/X --split test --pairs ethusd,xrpusd,dogeusd --starts 20 --paths 8
(test / heldout_test need MMFM_UNSEAL=<model hash>)
For every start checkpoint: context = events before it; book = exact id-book checkpoint; realised future = the
recorded continuation. Generators: MMFM, B1 (Markov), B0 (zero-intelligence) - all decoded by the same
state-aware decoder and matched by the same engine.
Metrics per horizon h in {1,5,10}s: realised return PIT within simulated distribution (calibration),
W1 between pooled simulated and pooled realised returns, event-count ratio, CRPS; P(up) AUC/Brier (M3).
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
from data.dataset import raw_chunks  # noqa: E402
from data.tokenizer import Tokenizer  # noqa: E402
from evaluate import load_model  # noqa: E402
from model.baselines import B0, B1  # noqa: E402
from data.dataset import load  # noqa: E402
from sim.book import LevelBook  # noqa: E402
from sim.rollout import Rollout, context_from_history  # noqa: E402

torch.set_num_threads(4)
H = (1.0, 5.0, 10.0)


class CountRollout(Rollout):
    """Same decoder/engine as Rollout, but tokens come from a count baseline instead of the model."""

    def __init__(self, base, kind, tok, ctx, n_paths, seed):
        self.base, self.kind = base, kind
        super().__init__(DummyModel(), tok, ctx, n_paths=n_paths, seed=seed)
        self.prev = [ctx.tokens[-1].copy() for _ in range(n_paths)]
        self.sb = [float(ctx.feats[-1][0]) for _ in range(n_paths)]

    def _sample_row(self, i):
        v = self.tok.vocab
        rng = self.rng
        if self.kind == "B0":
            return np.array([rng.choice(v[f], p=np.exp(self.base.p[f])) for f in ("act", "side", "px", "size", "dt")])
        b, prev = self.base, self.prev[i]
        cc = int(prev[0] * 10 + prev[1] * 5 + np.digitize(self.sb[i], [-0.5, 0.0, 0.5, 1.0]))

        def draw(key, nv):
            d = b.c.get(key)
            p = (np.zeros(nv) if d is None else d) + b.alpha
            return int(rng.choice(nv, p=p / p.sum()))

        a = draw(("act", cc), v["act"])
        s = draw(("side", cc, a), v["side"])
        px = draw(("px", cc, a, s), v["px"])
        z = draw(("size", a, s, px // 4), v["size"])
        dt = draw(("dt", cc, a), v["dt"])
        return np.array([a, s, px, z, dt])

    def step(self):
        t0 = time.perf_counter()
        for i, p in enumerate(self.paths):
            if not p["alive"]:
                continue
            row = self._sample_row(i)
            o = self._concrete(p, row)
            if o is None:
                p["alive"] = False
                continue
            t, f = self._apply(p, o)
            self.prev[i] = t
            self.sb[i] = float(f[0])
        self.gen_seconds += time.perf_counter() - t0
        self.n_steps += 1


class DummyModel:
    """Placeholder so Rollout.__init__ can run without a network (baseline generators don't use it)."""

    def __init__(self):
        self.cfg = dict(ctx=8)

    def eval(self):
        return self


def _dummy_fm_patch():
    import sim.rollout as R

    class _FM:
        def __init__(self, m):
            pass

        def prefill(self, tk, ft):
            self.hist_tok, self.hist_feat = tk, ft
            return None

    return R, _FM


def realised(ev, st, i, horizons):
    us0 = ev["us"][i - 1]
    mid = 0.5 * (st[:, 0] + st[:, 1])
    out = {}
    for h in horizons:
        j = np.searchsorted(ev["us"], us0 + h * 1e6, side="right")
        if j >= len(ev):
            return None
        out[h] = dict(ret=float(np.log(mid[j] / mid[i]) * 1e4), n=int(j - i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--pairs", default="ethusd,xrpusd,dogeusd")
    ap.add_argument("--starts", type=int, default=20)
    ap.add_argument("--paths", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    events = Path(a.events)
    model, mhash = load_model(a.run)
    tok = Tokenizer.from_json(json.loads((Path(a.run) / "tokenizer.json").read_text()))
    train, _ = load(events, "train", tok)
    b0, b1 = B0().fit(train, tok.vocab), B1().fit(train, tok.vocab)
    chunks, sp = raw_chunks(events, a.split)
    rng = np.random.default_rng(0)
    res = dict(split=a.split, model_hash=mhash, horizons=H, per_pair={}, latency={})
    import sim.rollout as RM
    for pair in a.pairs.split(","):
        cs = [c for c in chunks if c["pair"] == pair]
        if not cs:
            continue
        starts = []
        for c in cs:
            z = np.load(events / c["file"])
            ev, st = z["us"], z["state"]
            lo, hi = c["events"]["us"][0], c["events"]["us"][-1]
            for j, idx in enumerate(z["ck_idx"]):
                if idx > 1200 and lo <= ev[idx] <= hi - max(H) * 1e6:
                    starts.append((c["file"], j, int(idx)))
        if not starts:
            continue
        pick = [starts[k] for k in sorted(rng.choice(len(starts), min(a.starts, len(starts)), replace=False))]
        rows = {g: [] for g in ("MMFM", "B1", "B0")}
        for f, j, idx in pick:
            z = np.load(events / f)
            evs = np.zeros(len(z["us"]), dtype=[("us", "i8"), ("type", "i1"), ("side", "i1"), ("price", "f8"), ("qty", "f8")])
            for k in evs.dtype.names:
                evs[k] = z[k]
            st = z["state"]
            tick = float(z["tick"])
            real = realised(evs, st, idx, H)
            if real is None:
                continue
            b = LevelBook(tick)
            for side_arr, d in ((z["ck_bids"][j], b.bids), (z["ck_asks"][j], b.asks)):
                for p, q in side_arr:
                    if np.isfinite(p):
                        d[int(round(p / tick))] = float(q)
            ctx = context_from_history(evs[idx - 1100:idx], st[idx - 1100:idx], tok, b, model.cfg["ctx"])
            mid0 = 0.5 * (max(b.bids) + min(b.asks)) * tick
            for g in rows:
                t0 = time.perf_counter()
                if g == "MMFM":
                    R = Rollout(model, tok, ctx, n_paths=a.paths, seed=idx).run(max(H), max_steps=200000)
                else:
                    orig = RM.FastMMFM
                    RM.FastMMFM = _dummy_fm_patch()[1]
                    try:
                        R = CountRollout(b1 if g == "B1" else b0, g, tok, ctx, a.paths, idx).run(max(H), max_steps=200000)
                    finally:
                        RM.FastMMFM = orig
                wall = time.perf_counter() - t0
                m = R.mid_grid(np.array(H))
                sim_ret = np.log(m / mid0) * 1e4
                n_ev = np.array([p["n_ev"] for p in R.paths])
                rows[g].append(dict(real={h: real[h] for h in H}, sim=sim_ret.tolist(), n_ev=n_ev.tolist(), wall=wall,
                                    steps=R.n_steps))
        summ = {}
        for g, rr in rows.items():
            if not rr:
                continue
            s = {}
            for k, h in enumerate(H):
                sims = [np.array(r["sim"])[:, k] for r in rr]
                reals = np.array([r["real"][h]["ret"] for r in rr])
                pit = np.array([np.mean(np.nan_to_num(sm, nan=0) < rv) + 0.5 * np.mean(np.nan_to_num(sm, nan=0) == rv) for sm, rv in zip(sims, reals)])
                crps = np.mean([np.mean(np.abs(np.nan_to_num(sm) - rv)) - 0.5 * np.mean(np.abs(np.nan_to_num(sm)[:, None] - np.nan_to_num(sm)[None])) for sm, rv in zip(sims, reals)])
                pooled = np.concatenate([np.nan_to_num(sm) for sm in sims])
                w1 = float(np.mean(np.abs(np.sort(np.quantile(pooled, np.linspace(.01, .99, 99))) - np.sort(np.quantile(reals, np.linspace(.01, .99, 99))))))
                p_up = np.array([np.mean(np.nan_to_num(sm) > 0) for sm in sims])
                y = (reals > 0).astype(float)
                nz = reals != 0
                brier = float(np.mean((p_up[nz] - y[nz]) ** 2)) if nz.any() else None
                ratio = np.median([np.mean(r["n_ev"]) / max(r["real"][h]["n"], 1) for r in rr]) if k == len(H) - 1 else None
                s[str(h)] = dict(n_starts=len(rr), pit_mean=float(pit.mean()), pit_in_central_80=float(np.mean((pit >= .1) & (pit <= .9))),
                                 crps_bp=float(crps), w1_bp=w1, real_abs_ret_med=float(np.median(np.abs(reals))),
                                 sim_abs_ret_med=float(np.median(np.abs(pooled))), brier=brier, event_count_ratio=ratio)
            s["wall_s_per_start"] = float(np.mean([r["wall"] for r in rr]))
            s["sim_seconds_per_wall_second"] = float(max(H) * len(rr) / sum(r["wall"] for r in rr))
            summ[g] = s
        res["per_pair"][pair] = summ
        print(pair, json.dumps({g: {h: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()} if isinstance(d, dict) else round(d, 3) for h, d in s.items()} for g, s in summ.items()}), flush=True)
    res["_provenance"] = dict(git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                              argv=sys.argv, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out = Path(a.out or ROOT / "results" / f"m2_{a.split}.json")
    out.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
