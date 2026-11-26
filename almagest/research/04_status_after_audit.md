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
