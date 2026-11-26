# E1/E2 progress log (CPU-only, 4 cores, no GPU)

Tags: [C] measured here, reproducible from code in `almagest/`; [L] inference; [G] guess.

## Setup (why a ground-truth simulator)
Real data (LOBSTER) unavailable; real markets have no counterfactual. A known-physics order-level market ("World",
`sim/world.py`) gives *exact* counterfactual impact, so a model trained only on World's observational flow can be scored
against the true causal response. The harness re-implements the LOB-Impact protocol (2610.27100): eta=10%, N=100
aggressive median-size children, gap = 9/phi, buy/sell antisymmetrised, visible vs blind on identical seeds,
binned 18-quantile estimator, Y = a q_edge^(delta-1/2) in Parkinson-sigma units, relaxation 10 children + 100 silent
blocks, day-clustered bootstrap, plus TOST on delta and a day-bootstrap CI on end/peak.

Seed splits (asserted in tests): calibration days 1000+, evaluation days 5000+, training streams 200000+, validation 300000+.

## Harness self-tests [C] (`tests/test_harness.py`, all pass)
- estimator exact on noiseless planted laws (|err| < 0.01 for delta 0.4-1.0); within 0.12 under 10% day shocks
- null (no law): identified in <= 2/50 trials
- World blind arm: agent event stream bit-identical (hash) to the no-injection run; visible differs
- chunked parallel scan == stepwise recurrence (err 1e-6); slow bank online == offline

## E1: can an order-level generator with explicit physics pass the impact ladder at the benchmark operating point?
1. Base World (zero-intelligence + long-memory splitters + contrarian liquidity tethered to multi-timescale EMA
   reference): delta 0.55 but Y 2.7, H 0.36, overshoot relaxation [C].
2. Search over 60 random configs: Y never in [0.5,1] except when delta collapses [C]. [L] natural volatility too low
   relative to child-order reaction.
3. + fundamental random walk F with informed traders: Y down to ~1.2 [C]; but impact fully transient (e/p ~ 0) because
   F is unaffected by flow [C, disjoint-day confirm: delta 0.65, e/p 0.03, v/b 1.2].
4. + flow-learning anchor (Kyle-style: informed reference = F + lambda_K * remembered signed flow) [C mechanism].
   Refinement stalled at loss ~25 with **delta 0.69-0.80 whenever H ~ 0.5** - same band as LOB-Impact's physics
   kernels (0.73-0.84) [C].
5. + latent anchored liquidity (share of limit orders placed uniformly within K ticks of the slow reference: depth grows
   with distance, LLOB-style) -> first candidate with delta 0.47, Y 1.10, e/p 0.62 on calibration days [C].
6. **Calibration noise is the dominant failure mode**: e/p measured on 64-144 rollouts flips sign on disjoint days
   (0.62 -> -0.15; 0.29 -> -0.27) [C]. Fixed with 4x relaxation rollouts and bootstrap CI.
7. World B (refined, `results/world_B.json`), disjoint days 5000-5019, 320 build-up rollouts:
   delta 0.445 [0.26, 0.59], Y 0.62 [0.39, 0.86], v/b 3.9, H 0.46, gamma 0.52, blind stream invariant ->
   C1-C5 pass, TOST fails (CI too wide), e/p -0.27 (overshoot; unrealistic vs real 1/2-2/3) [C].
   Note: refined World B has p_anchor ~ 0, i.e. delta ~0.5 reached via informed/flow-learning balance, not latent
   liquidity [C]; which mechanism is necessary is unresolved [L].

**E1 verdict so far [L]:** C4 (|delta-0.5|<=0.1, Y in [0.5,1]) IS reachable by an order-level generator at the
LOB-Impact operating point - the "C4 unreachable" risk from the synthesis is not confirmed. Realistic relaxation
(e/p 1/2-2/3) with tight CIs is not yet achieved; precision needs >= ~1000 rollouts (per-rollout SNR ~0.2 [estimate]).

## E2: lightweight models trained only on World B observational flow
Four arms, ~60k params each (`models/lmm.py`), 2.56M training events, 4 passes, ~2 min/arm on CPU when uncontended [C]:
| arm | memory | val NLL |
|---|---|---|
| short | learned taus, state reset every 512 events (500-msg-context analogue) | 3.6561 |
| stateful | learned taus, state carried | 3.6561 |
| modebank | half channels fixed log-spaced 2^0..2^17 events | 3.6561 |
| modebank_sb | modebank + parameter-free slow bank | (training) |

All arms reach identical next-event NLL [C] -> long memory buys ~nothing in likelihood. Whether it buys correct
*reaction* is what the 1024-rollout blind-controlled campaign decides (running). This is the LOB-Impact
"mimicry != reaction" finding reproduced in a setting with exact ground truth [L until campaign finishes].

First evaluation at 128 rollouts was noise-dominated even for the truth (delta CI [-0.34, 0.54]) [C] -> rerun at 1024.
