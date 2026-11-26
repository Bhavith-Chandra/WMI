"""Minimal local server for ONE Almagest checkpoint. Standard library only. Binds 127.0.0.1 by default.

usage: python -m almagest.app.server <checkpoint.pt> [port]
  v1 (synthetic) checkpoint -> 'sim' mode (model generates the market; you can trade in it)
  ALMAGEST_MODE=truth + v1 checkpoint -> 'truth' mode (ground-truth World runs the market, model shadows it;
      every what-if shows the model's directional impact next to the EXACT World counterfactual)
  v2-real checkpoint -> 'live' mode (read-only Coinbase feed, env ALMAGEST_PRODUCT=BTC-USD; what-if from live state)
API (JSON):
  GET  /api/state?n=K        advance K model events, return book, recent mids, trades, status
  POST /api/order            {"side":"buy|sell","size":int}  user market order (the model sees it and reacts)
  POST /api/whatif           {"side","children","child_size","gap","tail_blocks","paths"}  paired counterfactual
  POST /api/reset            {"seed":int}  re-boot from a fresh context
"""
import json, os, sys, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
import torch  # noqa: E402
from almagest.app.live import LiveModel  # noqa: E402
from almagest.app.live_real import LiveReal  # noqa: E402
from almagest.app.truth import TruthLive  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = threading.Lock()
LM = None
CKPT = None


MODE = 'sim'


def snapshot(n_mid=600):
    if MODE == 'live':
        LM.pump()
        st = LM.status()
        st['mode'] = 'live'
        return dict(book=LM.book(), mids=LM.mids[-n_mid * 10::10] if len(LM.mids) > n_mid * 10 else LM.mids[-n_mid:],
                    trades=[], status=st)
    b = LM.book()
    st = LM.status()
    st['mode'] = 'truth' if getattr(LM, 'mode', '') == 'truth' else 'sim'
    return dict(book=b, mids=LM.mid_hist[-n_mid:], trades=LM.trade_hist[-60:], status=st)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj, ctype='application/json'):
        body = obj if isinstance(obj, (bytes, bytearray)) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ('/', '/index.html'):
            return self._send(200, open(os.path.join(HERE, 'index.html'), 'rb').read(), 'text/html; charset=utf-8')
        if u.path == '/api/state':
            n = int(parse_qs(u.query).get('n', ['0'])[0])
            n = max(0, min(n, 2000))
            with LOCK:
                if n and MODE == 'sim':
                    LM.step(n)
                return self._send(200, snapshot())
        return self._send(404, {'error': 'not found'})

    def do_POST(self):
        global LM
        u = urlparse(self.path)
        ln = int(self.headers.get('Content-Length', 0) or 0)
        try:
            req = json.loads(self.rfile.read(ln) or b'{}')
        except Exception:
            return self._send(400, {'error': 'bad json'})
        with LOCK:
            try:
                if u.path == '/api/order' and MODE == 'live':
                    return self._send(403, {'error': 'live mode is read-only: orders are never routed'})
                if u.path == '/api/whatif' and MODE == 'live':
                    kw = dict(side=req.get('side', 'buy'), children=max(1, min(int(req.get('children', 20)), 100)),
                              child_lots=max(1, min(int(req.get('child_lots', 8)), 2048)),
                              participation=max(0.01, min(float(req.get('participation', 0.1)), 0.5)),
                              tail_blocks=max(0, min(int(req.get('tail_blocks', 20)), 100)),
                              paths=max(8, min(int(req.get('paths', 48)), 128)))
                    return self._send(200, LM.whatif(**kw))
                if u.path == '/api/order':
                    fill = LM.submit_market(req.get('side', 'buy'), int(req.get('size', 1)))
                    return self._send(200, dict(fill=fill, **snapshot()))
                if u.path == '/api/whatif':
                    kw = {k: req[k] for k in ('side', 'children', 'child_size', 'gap', 'tail_blocks', 'paths')
                          if k in req}
                    for k in ('children', 'child_size', 'gap', 'tail_blocks', 'paths'):
                        if k in kw:
                            kw[k] = int(kw[k])
                    kw['children'] = max(1, min(kw.get('children', 20), 100))
                    kw['paths'] = max(8, min(kw.get('paths', 64), 256))
                    kw['gap'] = max(1, min(kw.get('gap', 40), 200))
                    kw['tail_blocks'] = max(0, min(kw.get('tail_blocks', 20), 100))
                    kw['child_size'] = max(1, min(kw.get('child_size', 2), 4))
                    return self._send(200, LM.whatif(**kw))
                if u.path == '/api/reset':
                    cls = TruthLive if getattr(LM, 'mode', '') == 'truth' else LiveModel
                    LM = cls(CKPT, context_seed=int(req.get('seed', 7_000_001)))
                    return self._send(200, snapshot())
            except Exception as e:
                return self._send(500, {'error': f'{type(e).__name__}: {e}'})
        return self._send(404, {'error': 'not found'})


def main():
    global LM, CKPT, MODE
    CKPT = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    host = os.environ.get('ALMAGEST_HOST', '127.0.0.1')
    ver = torch.load(CKPT, weights_only=False).get('version')
    if str(ver).startswith('v3'):
        from almagest.app.live_real3 import LiveReal3
        MODE = 'live'
        LM = LiveReal3(CKPT, os.environ.get('ALMAGEST_PRODUCT', 'BTC-USD'))
    elif ver == 'v2-real':
        MODE = 'live'
        LM = LiveReal(CKPT, os.environ.get('ALMAGEST_PRODUCT', 'BTC-USD'))
    elif os.environ.get('ALMAGEST_MODE') == 'truth':
        LM = TruthLive(CKPT)          # ground-truth World runs the market; model shadows; what-if vs exact truth
    else:
        LM = LiveModel(CKPT)
    print(f'Almagest live: {CKPT} ({LM.n_params} params, arm={LM.arm}) on http://{host}:{port}', flush=True)
    ThreadingHTTPServer((host, port), H).serve_forever()


if __name__ == '__main__':
    main()
