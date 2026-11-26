"""MMFM v2 live interface — wired ONLY to one MMFM v2 checkpoint. Paper trading only (no order routing).

python -I serve/app2.py --run runs/v2 --pair ethusd --port 8765
Forecast = calibrated direct heads (instant) + event-level simulated paths (same model) for comparison.
What-if  = paired event-level simulations (same seeds) with / without the user's market order.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.build import TICK  # noqa: E402
from data.l3 import BUY, SELL, TAKE  # noqa: E402
from data.tokenizer_v2 import HORIZONS, Tokenizer2  # noqa: E402
from serve.live2 import LiveFeed2  # noqa: E402
from serve.paper import Paper  # noqa: E402
from sim.rollout2 import Context2, Rollout2, direct_forecast  # noqa: E402
from sim_eval_v2 import load_v2  # noqa: E402

torch.set_num_threads(4)


def q_from_probs(values, edges, p, qs):
    """Quantiles of a binned distribution (linear within bins)."""
    e = np.asarray(edges, float)
    lo = np.r_[e[0] - (e[1] - e[0]) * 3 if len(e) > 1 else e[0] - 1, e]
    hi = np.r_[e, e[-1] + (e[-1] - e[-2]) * 3 if len(e) > 1 else e[-1] + 1]
    c = np.cumsum(p / p.sum())
    out = []
    for q in qs:
        b = int(np.searchsorted(c, q))
        b = min(b, len(p) - 1)
        prev = c[b - 1] if b > 0 else 0.0
        frac = (q - prev) / max(p[b] / p.sum(), 1e-12)
        out.append(0.0 if lo[b] <= 0 <= hi[b] and abs(hi[b] - lo[b]) < 1e-9 else float(lo[b] + frac * (hi[b] - lo[b])))
    return out


class App:
    def __init__(self, run, pair, dt_scale=None):
        self.model, self.hash = load_v2(run)
        cf = Path(run) / "sim_calibration.json"
        cal = json.loads(cf.read_text())["chosen"] if cf.exists() else dict(dt_scale=1.0, temp=1.0, clock="model")
        if dt_scale is not None:
            cal["dt_scale"] = dt_scale
        self.temp, self.clock = cal["temp"], cal.get("clock", "model")
        dt_scale = cal["dt_scale"]
        self.tok = Tokenizer2.from_json(json.loads((Path(run) / "tokenizer.json").read_text()))
        sel = json.loads((Path(run) / "selection_log.json").read_text())
        self.info = dict(hash=self.hash, run=str(run), params=sel.get("n_params"), kind="MMFM2",
                         val_bits=sel.get("best", {}).get("event_bits"), dt_scale=dt_scale,
                         temp=self.temp, clock=self.clock)
        self.pair, self.dt_scale = pair, dt_scale
        self.feed = LiveFeed2(pair, TICK[pair], self.tok, ctx_len=self.model.cfg["ctx"])
        self.paper = Paper()
        self.feed.listeners.append(self.paper.on_trade)
        self.lock = asyncio.Lock()

    def _ctx(self):
        toks, feats, inc = self.feed.context_arrays()
        return Context2(toks, feats, inc, self.feed.level_book(), int(self.feed.status["last_us"]))

    def forecast(self, n_paths, budget, seed):
        ctx = self._ctx()
        b = ctx.book
        mid0 = 0.5 * (max(b.bids) + min(b.asks)) * b.tick
        t0 = time.perf_counter()
        fc = direct_forecast(self.model, self.tok, ctx)
        direct_ms = (time.perf_counter() - t0) * 1e3
        qs = [0.05, 0.25, 0.5, 0.75, 0.95]
        direct = []
        for k, h in enumerate(HORIZONS):
            f = fc[k]
            p = f["p_ret"] / f["p_ret"].sum()
            z = int(np.searchsorted(f["ret_edges"], 0.0))
            vals = f["ret_values"]
            p_up = float(p[vals > 0].sum())
            p_dn = float(p[vals < 0].sum())
            pc = f["p_cnt"] / f["p_cnt"].sum()
            direct.append(dict(t=h, q=q_from_probs(vals, f["ret_edges"], f["p_ret"], qs), p_up=p_up, p_down=p_dn,
                               p_flat=float(p[z]), events_median=float(f["cnt_values"][np.searchsorted(np.cumsum(pc), 0.5)])))
        t1 = time.perf_counter()
        R = Rollout2(self.model, self.tok, ctx, n_paths=n_paths, seed=seed, dt_scale=self.dt_scale,
                         temperature=self.temp, clock=self.clock).run(max(HORIZONS), wall_budget_s=budget)
        el = time.perf_counter() - t1
        m = R.mid_grid(np.array(HORIZONS))
        r = (m / mid0 - 1) * 1e4
        cnt = R.counts(HORIZONS)
        sim = [dict(t=h, q=[None if not np.isfinite(x) else float(x) for x in np.nanpercentile(r[:, k], [5, 25, 50, 75, 95])],
                    p_up=float(np.nanmean(r[:, k] > 0)), reached=float(np.mean(np.isfinite(r[:, k]))),
                    events_median=float(np.median(cnt[:, k]))) for k, h in enumerate(HORIZONS)]
        sim_s = float(np.median([(R.last_time(p) - R.t0) / 1e6 for p in R.paths]))
        return dict(mid0=mid0, direct=direct, direct_ms=direct_ms, sim=sim, n_paths=n_paths, wall_s=el, steps=R.n_steps,
                    realtime_factor=sim_s / el if el > 0 else None)

    def whatif(self, side, notional, n_paths, budget, seed):
        ctx = self._ctx()
        b = ctx.book
        mid0 = 0.5 * (max(b.bids) + min(b.asks)) * b.tick
        out = {}
        for name, inj in (("baseline", False), ("with_order", True)):
            R = Rollout2(self.model, self.tok, ctx, n_paths=n_paths, seed=seed, dt_scale=self.dt_scale,
                         temperature=self.temp, clock=self.clock)
            if inj:
                qty = notional / mid0
                lim = (min(b.asks) + 10_000) if side == BUY else (max(b.bids) - 10_000)
                R.inject(TAKE, side, lim, qty)
            R.run(max(HORIZONS), wall_budget_s=budget)
            r = (R.mid_grid(np.array(HORIZONS)) / mid0 - 1) * 1e4
            out[name] = [dict(t=h, q=[None if not np.isfinite(x) else float(x) for x in np.nanpercentile(r[:, k], [5, 25, 50, 75, 95])],
                              p_up=float(np.nanmean(r[:, k] > 0))) for k, h in enumerate(HORIZONS)]
        sgn = 1 if side == BUY else -1
        impact = [None if (a["q"][2] is None or w["q"][2] is None) else sgn * (w["q"][2] - a["q"][2])
                  for a, w in zip(out["baseline"], out["with_order"])]
        return dict(mid0=mid0, side="buy" if side == BUY else "sell", notional=notional, impact_bp=impact, **out)


def make_app(args):
    A = App(Path(args.run), args.pair, args.dt_scale)
    app = web.Application()

    async def index(_):
        return web.FileResponse(ROOT / "serve" / "index2.html")

    async def ws(request):
        sock = web.WebSocketResponse(heartbeat=20)
        await sock.prepare(request)
        while not sock.closed:
            f = A.feed
            msg = dict(status=f.status, model=A.info, pair=A.pair, ready=f.ready())
            if f.proc is not None and f.status["last_us"]:
                lad = f.ladder(12)
                msg["ladder"] = lad
                if lad["bids"] and lad["asks"]:
                    mid = 0.5 * (lad["bids"][0][0] + lad["asks"][0][0])
                    msg["mid"] = mid
                    msg["spread_bp"] = (lad["asks"][0][0] - lad["bids"][0][0]) / mid * 1e4
                    last = f.status["last_us"]
                    pts = [(us, m) for us, m in list(f.mids)[-6000:] if last - us <= 120e6]
                    msg["mids"] = [((us - last) / 1e6, m) for us, m in pts[:: max(1, len(pts) // 600)]]
                    msg["events_per_s"] = sum(1 for us, _ in list(f.mids)[-3000:] if last - us <= 10e6) / 10
                    msg["paper"] = A.paper.summary(mid)
            await sock.send_str(json.dumps(msg, default=float))
            await asyncio.sleep(0.5)
        return sock

    async def forecast(request):
        p = await request.json()
        if not A.feed.ready():
            return web.json_response(dict(error="feed warming up"))
        async with A.lock:
            r = await asyncio.to_thread(A.forecast, int(p.get("paths", 16)), float(p.get("budget", 20)), int(p.get("seed", 0)))
        return web.json_response(r, dumps=lambda o: json.dumps(o, default=float))

    async def whatif(request):
        p = await request.json()
        if not A.feed.ready():
            return web.json_response(dict(error="feed warming up"))
        side = BUY if p.get("side", "buy") == "buy" else SELL
        async with A.lock:
            r = await asyncio.to_thread(A.whatif, side, float(p.get("notional", 25000)), int(p.get("paths", 16)),
                                        float(p.get("budget", 20)), int(p.get("seed", 0)))
        return web.json_response(r, dumps=lambda o: json.dumps(o, default=float))

    async def paper_order(request):
        p = await request.json()
        if A.feed.proc is None:
            return web.json_response(dict(error="feed not ready"))
        side = BUY if p.get("side") == "buy" else SELL
        res = A.paper.market(side, float(p["qty"]), A.feed.proc.book) if p.get("type", "market") == "market" \
            else A.paper.limit(side, float(p["qty"]), float(p["price"]), A.feed.proc.book)
        return web.json_response(res, dumps=lambda o: json.dumps(o, default=float))

    async def paper_cancel(request):
        return web.json_response(A.paper.cancel((await request.json())["id"]))

    async def paper_reset(_):
        A.paper.reset()
        return web.json_response(dict(ok=True))

    for path, h in (("/", index), ("/ws", ws)):
        app.router.add_get(path, h)
    for path, h in (("/api/forecast", forecast), ("/api/whatif", whatif), ("/api/paper/order", paper_order),
                    ("/api/paper/cancel", paper_cancel), ("/api/paper/reset", paper_reset)):
        app.router.add_post(path, h)

    async def start_feed(_app):
        _app["feed"] = asyncio.create_task(A.feed.run())

    app.on_startup.append(start_feed)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pair", default="ethusd")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dt-scale", type=float, default=None, help="override; default = frozen runs/<run>/sim_calibration.json")
    a = ap.parse_args()
    web.run_app(make_app(a), host=a.host, port=a.port)
