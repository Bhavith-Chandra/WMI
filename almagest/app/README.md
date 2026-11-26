# Almagest Live (minimal, model-bound)

```
python -m almagest.app.server <checkpoint.pt> [port]     # http://127.0.0.1:8765
```
One process = one checkpoint. No other model can be plugged in; nothing is routed to any venue.

- **Market**: the model generates the order flow in real time (adjustable events/s) on its own matching engine.
- **You**: market orders execute on the engine and are fed to the model as events, so the market reacts.
- **What-if**: forks the live state into N paired paths (common random numbers): visible (your metaorder, market
  sees it), blind (fills, market never sees it = mechanical part), free (no order). Shows impact with 95% bands.
- **Monitor**: per-event step latency (p50/p99) and surprise (model NLL of realised events).

Measured on this 4-core CPU container (61k-param checkpoint): 4.0 ms p50 / 8.1 ms p99 per event; what-if with 64
paths x 20 children x 40 gap = 307k events in 6.5 s.

Roadmap to real-time real markets (EVAL_PROTOCOL Track B/C): real-data tokenizer (Coinbase L2+trades, physical time
Δt head, size buckets) -> fine-tune -> condition on live feed (read-only) -> what-if on the live state.
