# 05 — Real-data realism: the representation was the bottleneck (2026-11-26)

All numbers: select split (hour 08 UTC) or fit hours; report hours (15–17) untouched.

## 1. Finding: v2 tokens were not replayable
Replaying the REAL v2 token stream through the engine, starting from the seeded top-10 book:

| events after seed | 10 | 100 | 1000 | 5000 |
|---|---|---|---|---|
| mean abs log-depth error | 0.03 | 0.18 | 1.18 | 1.31 |

Mid-change correlation was 0.31. So B2 was comparing generated books with books the engine could not reproduce even
with perfect events. v2 B2 numbers measure representation loss, not model error.

Causes found by ablation, each verified separately:
1. Features came from a size-rounded coarse book, events from rounded deltas → states drift apart. Fix: the engine
   book is canonical. Every token is the engine-semantics reconciliation of real L2 (`coinbase_exact.py`).
2. Transient crosses inside a 50 ms batch produced spurious in-window changes. Fix: reconcile after the whole batch,
   decreases first.
3. Multi-bucket improvements inside the spread (new best quotes) were untokenisable. Every later relative level was
   then shifted (a cascade). Fix: 8 explicit "improve by k = 2..9" limit classes.
4. Action window D = 20 missed most deep L2 changes. With D = 64, 84% of formerly-hidden changes are tokens.
   Remaining hidden lot share: BTC 11.7%, ETH 5.9%, SOL 32.9%. These are far orders, more than 32 bp away.

Result: replaying the tokens alone (no hidden updates) from an exact seed gives a log-depth error of **0.000 at
3000 events** on BTC, ETH and SOL. This is gated in `tests::test_exact_tokenizer_replay`, and B2's built-in oracle
arm is 12/12 PASS (error ~1e-7).

## 2. Finding: absolute size noise makes queues explode
First v3 model on select:
- B1 −1.28 nats/event vs the book-state table, CI [−1.30, −1.26].
- B2 0/12. Top-of-book depth inflated 527 → 3226 lots within 2000 events. Market share 2.6×.

Ablation with **perfect real events**, replacing only the sizes:

| sizes used | depth @ 2000 events (real 476) |
|---|---|
| exact | 476 |
| drawn within bucket (±0.5 octave) | 2515 |
| lognormal noise σ = 0.02 | 2076 |
| cancels as exact fraction of queue, limits 5% noise | 494 |
| cancels as 12 fraction bins, limits/markets from bucket table | 542 |

Even 2% absolute noise inflates queues 4×. Queues are reflected random walks, and cancels clipped at zero bias them
upward. Cancel size must therefore be expressed relative to the queue it hits. This gives v3.1:
- cancels: 24 queue-fraction classes, the last one being the whole queue (exact);
- limits and markets: half-octave lot classes.

## 3. Harness additions
- B0 representation gate (automated). B2 reports three references:
  - real-vs-real floor;
  - oracle (real events and real lots: must be exact);
  - oracle with sampled sizes (real event classes, generator size rule). This is the representation ceiling for any
    model.
- Pre-registered B2 pass rule in EVAL_PROTOCOL 4.1, written before any v3 training.
