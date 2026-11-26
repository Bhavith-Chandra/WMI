# Status after the leakage/harness audit (2026-11-26)

Tags: [C] measured here (reproducible from committed code/results); [L] inference; [G] guess.
Protocol: `almagest/EVAL_PROTOCOL.md` (v1.1). No test split has been touched. Everything below is dev/exploratory.

## 1. Audit outcome
Independent adversarial audit found 2 critical, 5 major and 7 minor issues, and no path from World hidden state or
test data into training or model inputs [C]. All of them are fixed:
- the reset arm is evaluated as trained;
- contaminated dev output is quarantined;
- C1 is pooled over identical windows;
- v/b fails closed and gets a CI;
- World burn-in, so days are stationary;
- training rows are contiguous and match evaluation;
- seed constants are centralised (`splits.py`), and outputs are write-once with git revision recorded.

Split 9000+ is retired: the truth was observed there before the fixes. The final test is 13000+.

Harness self-tests: 11 [C].
- estimator recovers planted laws; null gives no law
- blind == free by hash (World and model)
- snapshot runner is bit-identical
- streaming tokenizer == offline tokenizer
- engine book == stream book
- seed ranges disjoint
- v/b fails closed

## 2. Ground-truth World certification: FAILS so far
| World | δ (dev, 95% CI) | Y | R100 end/peak | H | verdict |
|---|---|---|---|---|---|
| B (pre-burn-in) | 0.476 [0.43, 0.54] | 0.62 | 0.04 [-0.15, 0.22] | 0.46 | fail (impact fully transient) |
| C | 0.691 [0.62, 0.81] | 0.72 | 0.44 [0.35, 0.55] | 0.47 | fail (δ > band) |
| D (robust search) | 0.604 [0.54, 0.66] | 0.61 | 0.34 [0.19, 0.45] | 0.49 | fail (δ CI edge 0.657; R100 < 0.45) |

- Winner's curse [C]: calibration scores overstate dev scores (R100 about 0.11 lower on dev for both C and D).
- Within-metaorder δ equals cross-sectional δ in World C (0.61 vs 0.61) [C]. The hypothesis that "LOB-Impact's C4 is
  miscalibrated" is **not supported**.

## 3. E2 (exploratory, World C, dev): observational training learns the wrong reaction
| | truth | short | stateful | modebank | modebank_sb |
|---|---|---|---|---|---|
| Y | 0.72 | 7.7 | 7.5 | 7.4 | **4.7** |
| v/b | **0.34** (market resists) | 171 | 51 | 52 | **6.6** |
| R100 end/peak | 0.44 | 1.25 | 1.23 | 1.32 | **0.91** |
| slope after metaorder (%peak/duration) | -18 | +24 | +26 | +32 | **-6.9** |
| val NLL | n/a | 3.5241 | 3.5236 | 3.5233 | 3.5230 |

- The failure signature of the published panel (2610.27100) is reproduced with exact ground truth [C].
- Fixed long timescales alone (modebank) do not help. Only the parameter-free slow bank of observed flow and price
  deviation helps [C].
- Likelihood is blind to all of this (ΔNLL < 0.001 nats) [C].
- [L] The remaining error is causal: in observational data, buy flow predicts more buy flow (hidden metaorder
  splitting), so an exogenous order is read as informed and amplified. More data or capacity cannot remove this
  confounding.
- Next experiment (running): source-tagged interventional training (own orders tagged; never prediction targets),
  2x2 against observational training on World D, plus A8 held-out participations.

## 4. Real data (Track B) and deployment path (Track C)
- Coinbase public `level2_batch` + `matches` recording since 02:26 UTC, BTC, ETH and SOL; about 240 events/s for
  BTC [C].
  - The authenticated "full" channel is unavailable without keys.
  - 50 ms batching gives 90% zero Δt, a hard resolution limit of the free feed [C].
- Tokenizer v2:
  - coarse 0.5 bp grid;
  - 12 log2 size buckets;
  - Δt buckets;
  - trade-attributed L2 decreases;
  - one `StreamTokenizer` shared by offline and live use, with a parity test [C].
- v2 model: 64.5k parameters.
  - Smoke test only, not reportable: held-out NLL 6.18 vs 6.65 for the book-state table baseline [C].
  - The calendar split needs at least 10 complete hours.
- Live interface (`almagest/app`), bound to one checkpoint:
  - sim mode: the model is the market, and you can trade in it;
  - live mode: read-only BTC-USD, teacher-forced on every real event; 0.55 ms p50 / 1.0 ms p99 per event at
    200 events/s; what-if in bp over seconds; orders refused [C].
  - The current live checkpoint is DEV-only. Its what-if is implausible (a buy pushes the price down), which shows
    why B5 is gated.

## 5. Not yet achieved
A certified truth; any test-split result; an evaluated real-data model; ≥ 50 ms latency/stability over hours (C2).


## 6. CORRECTION (same day): model impact numbers above measure collapsed markets
Free-generation diagnostics [C]:
- Teacher-forced side predictions are calibrated: P(buy) 0.503 vs empirical 0.499.
- Closed-loop generation reaches an absorbing degenerate state: one side of the book empties to the engine's
  1-lot guard; the model, never having seen such states, stops posting on that side; the mid price freezes.
- Rate: 18/24 free runs collapse within 2,500 events. The impact horizon is 16,400 events.

Consequences:
- Every model row in sections 3 and the E2D tables is dominated by collapse. These rows show the slow bank delays
  collapse; they do **not** measure reaction quality.
- The model failures on drift (C1), excess volatility (3x) and huge CIs are explained by this.
- [L] The published panel's drift (up to -879 ticks) may share the mechanism.

Fix in progress:
- synthetic: DAgger with the World as oracle;
- real data (no oracle): rollout-based post-training.
- New gate: collapse rate < 2% over the evaluation horizon before any impact metric is interpreted. Every evaluation
  now reports `collapse_rate` per arm.


## 7. ROOT CAUSE of the collapse: shared linear output heads (fixed)
Ruled out [C]:
- capacity: 295k params, 83% collapse;
- long memory, source tags, mirror augmentation;
- DAgger with oracle: v1 22 -> 55 -> 34% one-sided; v2 2 -> 12 -> 22 -> 45% as beta decays.

Found [C]:
- Teacher-forced, the old model's P(next market order is a buy) was flat at 0.50-0.53 whatever the preceding sign run.
  The World's empirical value is 0.32 after 8 sells and 0.67 after 8 buys.
- A linear probe on the slow-bank inputs beats chance (56% held-out), so the information was available.
- Cause: side/level/size heads were linear in (z + type embedding). Type only shifted a bias, so the state->side
  mapping was shared by limits, cancels and markets, and market sides became state-independent.

Fix: type-conditional heads (`cond_heads`), 61k -> 71k params [C]:
- val NLL 3.648 -> 3.562 (vs < 0.001 for every earlier change);
- persistence curve matches the World (0.28 vs 0.32 after 8 sells, 0.34 vs 0.31 after 4);
- free-run collapse 0/12 within 5k events (was 9/12).

Every E2 model number before this fix is superseded. The real-data v2 model gets the same fix.


## 8. Main result so far (exploratory, World D, dev): own-order memory channel fixes the direction of reaction
Diagnostics [C], teacher-forced on interventional validation streams. P(next background market order is a buy):

| own tagged flow (last ~500 events) | World | no own-order channel | with own-order channel |
|---|---|---|---|
| strong own selling | 0.590 | 0.51 | 0.587 |
| strong own buying | 0.390 | 0.46 | 0.396 |

Dev campaign [C] (identical training data; the only difference is the channel):

| | truth | src (cond heads) | src + own-order channel |
|---|---|---|---|
| δ | 0.60 | 0.98 | 0.78 [0.67, 0.88] |
| Y | 0.61 | 2.9 | 1.22 [0.79, 1.62] |
| visible/blind | 0.28 | 1.37 | 0.64 [0.40, 1.03] |
| R100 end/peak | 0.34 | 1.22 | 0.62 [0.52, 0.69] |
| slope100 | -21 | +12 | -25 |
| R10 end/peak | 0.34 | 1.42 | 0.37 |
| C1 | pass | pass | pass |

Interpretation [L]: in observational data, a price deviation implies an active hidden metaorder, so continuation
follows. After an exogenous order, the market leans against it. Only a persistent memory of how much recent flow was
exogenous lets a next-event model learn the difference; a per-step tag does not.

Still failing the pre-registered gates:
- A3: |Δδ| 0.18
- A4: Y ratio 2.0
- A5: |ΔR100| 0.28
- A6: v/b 2.3x
- collapse 6% > 2%

The truth is uncertified.


## 9. Scale test and remaining gap
2x data and 8 passes change nothing within CIs (Y 1.18 vs 1.22, δ 0.80 vs 0.78) [C]. Not under-training.

Teacher-forced one-step conditionals match the World vs own flow on everything summarised [C]:
- market side and market share;
- limit side and cancel side (all within about 0.01-0.02).

The residual (Y about 2x, R100 0.6 vs 0.34) is therefore closed-loop compounding of unsummarised conditionals [L]:
level placement, book-shape dependence, the hidden-fundamental anchor.

Strategic consequence [L]: real data has no tagged own orders, so the own-order reaction can ONLY be learned from
interventional synthetic data. Stage 1 (synthetic causal pretraining) is necessary, and the synthetic-to-real transfer
of that reaction is the main open risk.
