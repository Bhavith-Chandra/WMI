# Almagest LMM — model card (living document)

**What it is.** A lightweight (60–80k parameters) recurrent order-level foundation model for limit order books. One
backbone serves four uses:
1. counterfactual **what-if impact** of your own orders;
2. a **reactive market simulator / RL environment**;
3. **forecasting** heads;
4. **manipulation-detection** features.

It runs on CPU in real time: **0.55 ms p50 / 1.0 ms p99** per event against the live BTC-USD feed (≈ 200 events/s).

All numbers below are tagged with their split. "dev" = selection split. **No test split has been run yet**
(pre-registered protocol: `EVAL_PROTOCOL.md`).

## Architecture (what each piece is for, with the evidence that forced it)
| component | why | evidence |
|---|---|---|
| per-event diagonal linear recurrence; half the channels on fixed log-spaced timescales 1 … 2^17 events | long memory at constant cost | — |
| parameter-free slow bank: EMAs of signed flow and of price − EMA(price), 8 timescales | gives long-horizon state | delays collapse; no reaction effect once collapse is fixed |
| **type-conditional output heads** | a shared linear side head made market-order direction state-independent | val NLL 3.648 → 3.562; free-run collapse 9/12 → 0/12 |
| **source-tagged own orders + own-order slow-bank channel** | the market reacts to *exogenous* flow differently from informed flow; only a persistent memory of "how much recent flow was mine" lets a next-event model learn it | reaction to own buying: World 0.390, model 0.396 (was 0.46); Y 2.9 → 1.2 |
| factored heads type → side → level → size (→ Δt) with validity masks | valid, replayable events | — |
| domain embedding (unified model) | one model for synthetic-causal and real data | — |

## Training recipe
1. **Synthetic causal pretraining.** Ground-truth World streams with random *tagged* metaorders over participation
   2–30%, so the causal reaction is learnable at all. Real data contains no tagged own orders.
2. **Real-data training** on Coinbase L2+trades: 0.5 bp grid, 12 size buckets, Δt buckets. Unified with (1) through
   the domain embedding and a depth-matched lot scale.
3. **DAgger** with the World oracle: closed-loop rollouts with tagged injections. This fixes exposure bias and
   collapse.
4. **Stage-3 post-training:** model soup (base ↔ DAgger) chosen on a training-range tune split against the full
   counterfactual objective.

## Evaluation status (dev)
- **Track A** (World D; that World is itself *uncertified*): best model is the soup, α = 0.25.

  | metric | truth | soup α = 0.25 |
  |---|---|---|
  | Y | 0.61 | 0.95 |
  | visible/blind | 0.28 | 0.54 |
  | δ | 0.60 | 0.72 |
  | R100 end/peak | 0.34 | 0.65 |
  | free-run volatility | 0.45 | 0.39 |
  | collapse | — | 2.7% |

  - Passes: amplitude and attribution are within about 2× of truth.
  - Misses: δ is off by 0.12; collapse is 2.7%, gate 2%.
  - Relaxation is not comparable yet, because World D's truth relaxes unrealistically (0.34 vs ~0.67 in real
    markets).
  - A realistic truth (propagator-anchor World) is in calibration.
- **Track B** (real Coinbase; split v1 pre-registered: fit 02-06, select 08, report 15-17 UTC; numbers are SELECT):

  B1 next-event NLL on 1.03M held-out events:

  | | NLL (nats/event) | model − baseline (95% CI) |
  |---|---|---|
  | unified model | **5.995** | — |
  | book-state table | 6.786 | **−0.79** [−0.82, −0.77] |
  | marginal | 6.956 | −0.96 |

  B4 mid-direction AUC:

  | horizon | model probe | best baseline (book logistic) |
  |---|---|---|
  | 50 events | **0.858** | 0.836 |
  | 200 events | **0.798** | 0.782 |
  | 1000 events | **0.666** | 0.657 |

  B2 closed-loop realism: **NOT yet realistic.**

  | metric | generated | real | real-vs-real floor |
  |---|---|---|---|
  | trade-sign ACF(1) | 0.16–0.21 | 0.37–0.50 | — |
  | volatility (100 events) | 0.5× | — | 0.75–0.85× |
  | depth W1 | 1.3–1.6 | — | 0.3 |

  Scalar generation calibration, deep book seeding and size representation did not close it. A rollout-level
  post-training method is required (no oracle on real data).

  B5 own-order impact on real data: not meaningful until B2 is fixed.
- **Use case 3** (manipulation, synthetic, dev):
  - the hand-written spoof rule wins on its own template (AUC 0.89);
  - a model-state probe detects momentum ignition when trained in-family (AUC 0.80);
  - nothing generalises to unseen families yet. Open.
- **Use case 4** (RL): Gym-style execution environment with model or exact-World market. Sim-to-truth execution-cost
  comparison (gate E2) pending.

## Known limitations (do not deploy for trading decisions until these close)
- No test-split result yet; Track A truth uncertified; real-data results pending.
- Public Coinbase feed is 50 ms-batched (90% of Δt = 0) and has no order IDs. Sub-50 ms dynamics are not learnable
  from it.
- The reaction to *own* orders comes from synthetic causal pretraining. Its real-market transfer is the main open
  risk, and can only be validated with real executions or natural experiments.
- The interface never routes orders.
