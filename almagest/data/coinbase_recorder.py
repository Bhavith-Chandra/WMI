"""Coinbase Exchange public-feed recorder (no auth): level2_batch (50-level L2, 50 ms batches) + matches (trades with
aggressor side and maker/taker order IDs) + heartbeat. Raw messages are stored verbatim with a local receive timestamp,
one gzip JSONL file per product-hour. Reconnects with backoff; every (re)connect starts a new segment because the L2
snapshot must be re-applied (segments never straddle a gap). Sequence gaps on 'matches' are logged.

usage: python coinbase_recorder.py <out_dir> BTC-USD ETH-USD SOL-USD
"""
import asyncio, gzip, json, os, ssl, sys, time

URL = 'wss://ws-feed.exchange.coinbase.com'
CA = '/root/.ccr/ca-bundle.crt'


class Sink:
    def __init__(self, root):
        self.root = root
        self.files = {}

    def write(self, product, seg, rec):
        hour = time.strftime('%Y%m%dT%H', time.gmtime(rec['_rx']))
        key = (product, hour, seg)
        f = self.files.get(key)
        if f is None:
            for k in [k for k in self.files if k[0] == product]:
                self.files.pop(k).close()
            d = os.path.join(self.root, product)
            os.makedirs(d, exist_ok=True)
            f = gzip.open(os.path.join(d, f'{hour}_seg{seg}.jsonl.gz'), 'at')
            self.files[key] = f
        f.write(json.dumps(rec, separators=(',', ':')) + '\n')

    def flush(self):
        for f in self.files.values():
            f.flush()


async def run(root, products):
    sink = Sink(root)
    log = open(os.path.join(root, 'recorder.log'), 'a')
    ctx = ssl.create_default_context(cafile=CA) if os.path.exists(CA) else ssl.create_default_context()
    import websockets
    seg = int(time.time())
    backoff = 1
    while True:
        seg += 1
        last_seq = {}
        try:
            async with websockets.connect(URL, ssl=ctx, open_timeout=20, max_size=2 ** 25, ping_interval=20) as ws:
                await ws.send(json.dumps({'type': 'subscribe', 'product_ids': products,
                                          'channels': ['level2_batch', 'matches', 'heartbeat']}))
                log.write(f'{time.time():.3f} connect seg={seg}\n'); log.flush()
                backoff = 1
                n = 0
                while True:
                    m = await asyncio.wait_for(ws.recv(), 30)
                    rec = json.loads(m)
                    rec['_rx'] = time.time()
                    p = rec.get('product_id')
                    if rec.get('type') == 'error':
                        log.write(f'{rec["_rx"]:.3f} error {m[:300]}\n'); log.flush()
                        raise RuntimeError('feed error')
                    if p is None:
                        continue
                    if rec.get('type') == 'match':
                        s = rec.get('sequence')
                        if p in last_seq and s is not None and s <= last_seq[p]:
                            log.write(f'{rec["_rx"]:.3f} nonmonotone {p} {last_seq[p]}->{s}\n')
                        last_seq[p] = s
                    sink.write(p, seg, rec)
                    n += 1
                    if n % 2000 == 0:
                        sink.flush()
        except Exception as e:
            log.write(f'{time.time():.3f} disconnect seg={seg} {type(e).__name__}: {str(e)[:200]}\n'); log.flush()
            sink.flush()
            await asyncio.sleep(backoff)
            backoff = min(60, backoff * 2)


if __name__ == '__main__':
    asyncio.run(run(sys.argv[1], sys.argv[2:]))
