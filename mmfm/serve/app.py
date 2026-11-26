"""MMFM live interface — wired ONLY to one MMFM checkpoint (hash shown in the UI). Paper trading only.

python -I serve/app.py --run runs/<model> --pair ethusd --port 8765
Endpoints: GET / ; GET /ws (2 Hz state) ; POST /api/forecast ; POST /api/whatif ;
           POST /api/paper/order ; POST /api/paper/cancel ; POST /api/paper/reset ; GET /api/paper
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
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
from data.l3 import ADD, BUY, SELL, TAKE  # noqa: E402
from data.tokenizer import Tokenizer  # noqa: E402
from evaluate import load_model  # noqa: E402
from serve.live import LiveFeed  # noqa: E402
from serve.paper import Paper  # noqa: E402
from sim.rollout import Context, Rollout  # noqa: E402

torch.set_num_threads(4)
GRID = np.array([0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60])


class App:
    def __init__(self, run, pair):
        self.model, self.hash = load_model(run)
        self.tok = Tokenizer.from_json(json.loads((Path(run) / "tokenizer.json").read_text()))
        sel = json.loads((Path(run) / "selection_log.json").read_text())
        self.model_info = dict(hash=self.hash, run=str(run), params=sel.get("n_params"),
                               val_bits=sel.get("best", {}).get("bits_per_event"), kind=type(self.model).__name__)
        self.pair = pair
        self.feed = LiveFeed(pair, TICK[pair], self.tok, ctx_len=self.model.cfg["ctx"])
        self.paper = Paper()
        self.feed.listeners.append(self.paper.on_trade)
        self.lock = asyncio.Lock()

    # ------------------------------------------------------------------ model calls (blocking -> thread)
    def _context(self):
        toks, feats, inc = self.feed.context_arrays()
        return Context(toks, feats, inc, self.feed.level_book(), int(self.feed.status["last_us"]))

    def _fan(self, R, mid0, horizon):
        g = GRID[GRID <= horizon]
        m = R.mid_grid(g)
        r = (m / mid0 - 1) * 1e4
        q = np.nanpercentile(r, [5, 25, 50, 75, 95], axis=0)
        reach = np.mean(np.isfinite(r), 0)
        p_up = np.nanmean(r > 0, 0)
        p_dn = np.nanmean(r < 0, 0)
        return dict(t=g.tolist(), q=np.where(np.isfinite(q), q, None).tolist(), p_up=p_up.tolist(), p_down=p_dn.tolist(),
                    reach=reach.tolist())

    def forecast(self, horizon, n_paths, budget, seed=0):
        ctx = self._context()
        mid0 = 0.5 * (max(ctx.book.bids) + min(ctx.book.asks)) * ctx.book.tick
        t0 = time.perf_counter()
        R = Rollout(self.model, self.tok, ctx, n_paths=n_paths, seed=seed).run(horizon, wall_budget_s=budget)
        el = time.perf_counter() - t0
        sim_s = float(np.median([(p["us"] - R.t0) / 1e6 for p in R.paths]))
        return dict(mid0=mid0, fan=self._fan(R, mid0, horizon), wall_s=el, steps=R.n_steps, n_paths=n_paths,
                    simulated_s_median=sim_s, realtime_factor=sim_s / el if el > 0 else None,
                    ms_per_step=1000 * R.gen_seconds / max(R.n_steps, 1), alive=sum(p["alive"] for p in R.paths))

    def whatif(self, side, notional, horizon, n_paths, budget, seed=0):
        ctx = self._context()
        b = ctx.book
        mid0 = 0.5 * (max(b.bids) + min(b.asks)) * b.tick
        qty = notional / mid0
        out = {}
        for name, inj in (("baseline", False), ("with_order", True)):
            R = Rollout(self.model, self.tok, ctx, n_paths=n_paths, seed=seed)  # same seed: paired comparison
            if inj:
                lim = (min(b.asks) + 10_000) if side == BUY else (max(b.bids) - 10_000)  # marketable sweep
                R.inject(TAKE, side, lim, qty)
            R.run(horizon, wall_budget_s=budget)
            out[name] = self._fan(R, mid0, horizon)
        mb = np.array([np.nan if x is None else x for x in out["baseline"]["q"][2]], float)
        mw = np.array([np.nan if x is None else x for x in out["with_order"]["q"][2]], float)
        sgn = 1 if side == BUY else -1
        impact = (sgn * (mw - mb)).tolist()  # paired (same seed) median difference, bp, direction-adjusted
        return dict(impact_bp=[None if not np.isfinite(x) else x for x in impact],mid0=mid0, qty=qty, side="buy" if side == BUY else "sell", notional=notional, **out)


def json_ok(x):
    return web.json_response(x, dumps=lambda o: json.dumps(o, default=float))


def make_app(args):
    A = App(Path(args.run), args.pair)
    app = web.Application()

    async def index(_):
        return web.FileResponse(ROOT / "serve" / "index.html")

    async def ws(request):
        sock = web.WebSocketResponse(heartbeat=20)
        await sock.prepare(request)
        while not sock.closed:
            f = A.feed
            msg = dict(status=f.status, model=A.model_info, pair=A.pair, ready=f.ready())
            if f.proc is not None and f.status["last_us"]:
                lad = f.ladder(12)
                msg["ladder"] = lad
                if lad["bids"] and lad["asks"]:
                    mid = 0.5 * (lad["bids"][0][0] + lad["asks"][0][0])
                    msg["mid"] = mid
                    msg["spread_bp"] = (lad["asks"][0][0] - lad["bids"][0][0]) / mid * 1e4
                    last = f.status["last_us"]
                    pts = [(us, m) for us, m in list(f.mids)[-6000:] if last - us <= 120e6]
                    step = max(1, len(pts) // 600)
                    msg["mids"] = [((us - last) / 1e6, m) for us, m in pts[::step]]
                    rate = [us for us, _ in list(f.mids)[-3000:] if last - us <= 10e6]
                    msg["events_per_s"] = len(rate) / 10
                    msg["paper"] = A.paper.summary(mid)
            await sock.send_str(json.dumps(msg, default=float))
            await asyncio.sleep(0.5)
        return sock

    async def forecast(request):
        p = await request.json()
        if not A.feed.ready():
            return json_ok(dict(error="feed not ready"))
        async with A.lock:
            return json_ok(await asyncio.to_thread(A.forecast, float(p.get("horizon", 10)), int(p.get("paths", 16)),
                                                   float(p.get("budget", 20)), int(p.get("seed", 0))))

    async def whatif(request):
        p = await request.json()
        if not A.feed.ready():
            return json_ok(dict(error="feed not ready"))
        side = BUY if p.get("side", "buy") == "buy" else SELL
        async with A.lock:
            return json_ok(await asyncio.to_thread(A.whatif, side, float(p.get("notional", 10000)),
                                                   float(p.get("horizon", 10)), int(p.get("paths", 12)),
                                                   float(p.get("budget", 15)), int(p.get("seed", 0))))

    async def paper_order(request):
        p = await request.json()
        if A.feed.proc is None:
            return json_ok(dict(error="feed not ready"))
        side = BUY if p.get("side") == "buy" else SELL
        qty = float(p["qty"])
        if p.get("type", "market") == "market":
            return json_ok(A.paper.market(side, qty, A.feed.proc.book))
        return json_ok(A.paper.limit(side, qty, float(p["price"]), A.feed.proc.book))

    async def paper_cancel(request):
        return json_ok(A.paper.cancel((await request.json())["id"]))

    async def paper_reset(_):
        A.paper.reset()
        return json_ok(dict(ok=True))

    async def paper_get(_):
        lad = A.feed.ladder(1) if A.feed.proc else None
        mid = 0.5 * (lad["bids"][0][0] + lad["asks"][0][0]) if lad and lad["bids"] and lad["asks"] else None
        return json_ok(A.paper.summary(mid))

    app.router.add_get("/", index)
    app.router.add_get("/ws", ws)
    app.router.add_post("/api/forecast", forecast)
    app.router.add_post("/api/whatif", whatif)
    app.router.add_post("/api/paper/order", paper_order)
    app.router.add_post("/api/paper/cancel", paper_cancel)
    app.router.add_post("/api/paper/reset", paper_reset)
    app.router.add_get("/api/paper", paper_get)

    async def start_feed(_app):
        _app["feed_task"] = asyncio.create_task(A.feed.run())

    app.on_startup.append(start_feed)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pair", default="ethusd")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    web.run_app(make_app(a), host=a.host, port=a.port)
