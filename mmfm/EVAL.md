# MMFM evaluation & leakage protocol (pre-registered before any model is trained)

MMFM = our market-microstructure foundation model. It is trained on real, public Bitstamp L3 order
flow (`collect/recorder.py`). This file is written and committed **before** training. Any later
change to it must be committed separately with a reason, and `tests/` enforce the mechanical parts.

## 0. Data
* Source: Bitstamp public websocket `live_orders_<pair>` + `live_trades_<pair>`, plus full L3 REST
  snapshots with order ids at every (re)connect and every 15 min.
* Pairs: btcusd, ethusd, solusd, xrpusd, btceur, ltcusd, dogeusd, linkusd.
* **linkusd is a held-out pair**. It is never used for training, fitting or selection, so it measures
  zero-shot transfer.
* Sequences are cut at every disconnect or gap. Continuity is never fabricated.
* The book is reconstructed at full depth from snapshot + events using order ids.
  * The reconstruction is accepted only if it matches the next REST snapshot (§5, M6).
  * Hours that fail the check are excluded **before** splitting, by a rule fixed here: best bid/ask must
    match and the top-20 level quantities must match on ≥ 99% of levels. The exclusion list is logged.

## 1. Splits (by wall-clock time, identical for all pairs)
Let `[T0, T1)` be the collected span after exclusions.
* train = first 70%; embargo 10 min; val = next 10%; embargo 10 min; test = last 20% minus embargo.
* Embargo: no train or val window may contain an event within 10 min of a later split's start, so
  autocorrelated microstructure can't bridge splits.
* No window crosses a split boundary or a gap. A target (future path) must lie fully inside the same split as its context.
* Split boundaries are computed once by `data/splits.py`, written to `splits.json` with the hash of the
  file list, and never recomputed after training starts.

## 2. Leakage rules (each one is enforced by a test)
| rule | enforcement |
|---|---|
| L1 every fitted transform (bins, clip quantiles, vocab, normalisers, baseline parameters) is fit on train only | transforms store the hash of their source split; a test asserts it equals the train hash |
| L2 model selection (early stopping, hyperparameters, checkpoint choice) uses val only | selection log lists the split used; a test asserts it is never `test` |
| L3 test is sealed | the test loader requires `MMFM_UNSEAL=<model_hash>` and appends to `test_access.log`; ≤ 1 unseal per frozen model hash is allowed, and the harness fails otherwise |
| L4 training and selection code never imports the test loader | AST import scan |
| L5 the held-out pair (linkusd) is never in a train or val batch | dataset assertion + test |
| L6 no lookahead inside a window | causality tests (perturb future → past outputs unchanged) for the tokenizer, model and simulator |
| L7 state features at event t use only events ≤ t | the reconstruction engine is the only source of state; a test replays with future events removed and gets an identical state |
| L8 every result is provenance-stamped (commit, args, input hashes) | provenance test |
| L9 README / report numbers are generated from result files | claims test |

## 3. Baselines (fit on train with the same tokenization, so likelihoods are comparable)
* B0 marginal / zero-intelligence: independent per-field marginals.
* B1 Markov: per-field conditionals on the previous event's (type, side) and the current spread bucket.
* B2 strong feature baseline: gradient-boosted or logistic model per field on handcrafted state.
  Features: last k event types, spread, top-5 imbalance, queue sizes, Δt history, time-of-day.
  This is the realistic bar; beating only B0 / B1 means nothing.
* For simulation: ZI and B1 generators driven through the same matching engine.

## 4. Primary metrics (test, per pair, plus held-out pair; 95% CIs by 5-min block bootstrap)
* **M1 next-event NLL** (bits/event, exact factorised likelihood). Promotion: MMFM beats the best
  baseline (normally B2) on val; the gap on test is reported with a CI and must exclude 0 to claim a win.
* **M2 simulation realism**:
  * Start from test states and roll out 60 s, many paths per start.
  * Compare against the realised test continuation on: mid log-return W1 / KS at 1 s / 10 s / 60 s, spread
    distribution, event rate, inter-arrival distribution, and ACF of |r| at 1–60 s lags.
  * Compared with ZI and B1 through the same engine.
* **M3 short-horizon prediction**: P(mid up) at 10 s and 60 s from the Monte-Carlo rollouts.
  * Metrics: AUC, Brier, log-loss.
  * Baselines: constant base rate, order-flow-imbalance logistic regression (fit on train) and B2.
  * Reported "as is"; no claim of tradability without costs.
* **M4 counterfactual plausibility** (no ground truth exists):
  * Impact of TWAP meta-orders is monotone, concave and buy/sell symmetric, with a path-level bootstrap
    exponent CI, using the independent-path simulator from the M3 audit.
  * Reported as diagnostics, not as evidence of correctness.
* **M5 real-time viability**: p50 / p99 CPU latency per event update and per 60 s forecast. Throughput
  must exceed 3× the peak observed event rate of the pair.
* **M6 data fidelity**: reconstruction match rate against snapshots, per hour.

## 5. Reporting rules
* Every number in the report comes from `results/*.json` via a generator; there are no hand-typed metrics.
* Negative results are reported with the same prominence as positive ones.
* A metric computed on test before the model hash is frozen invalidates that model version.

## Addendum 1 (development data rule, committed before any training)
* A **dev build** cut at `dev_cut_ns = 1795663953517049564` (2026-11-26T03:32:33Z) is used to develop the tokenizer and model and to debug.
* The final frozen build is only valid if its train region contains the whole dev window:
  train_end ≥ dev_cut + 10 min. `data/splits.py` refuses otherwise.
  With collection start ≈ 02:27 UTC, the final cut must be ≥ ≈ 04:45 UTC.
* Nothing at or after `dev_cut_ns` is inspected before the final freeze.

## Addendum 2 (seal granularity, committed before any test access)
* L3 is refined to ≤ 1 unseal per (frozen model hash, split, pre-registered metric ∈ {M1, M2}).
  The metric is declared through `MMFM_PURPOSE` and logged, and a repeat for the same triple fails.
* Reason: M1 (NLL) and M2 (simulation realism) are both pre-registered test metrics of the same frozen model.
* Horizons for M2/M3 are 1/5/10 s, not 60 s. Event-level rollouts on CPU can't reach 60 s on fast pairs
  in reasonable time (measured: 8 ms/step for 16 paths). This is reported under M5.
