# Almagest LMM — pre-registered evaluation protocol (v1.1, 2026-11-26; still before any test run)

Changelog: v1.1 adds A8 (held-out operating points) after introducing interventional (source-tagged) training;
no test split had been touched.

Any change to this file after a test run invalidates that test split; a new split must be drawn.
Every result file must record: split, seeds/time ranges, git revision (clean tree), config, n_rollouts.

## 0. Principles
1. **Three-way separation everywhere:** fit (train) / select (val or dev) / report (test). Test touched once per frozen model.
2. **No adaptive reuse:** anything observed on a split (even truth-only numbers) burns that split for selection.
3. **Fail closed:** undefined metrics (NaN, ratios with non-positive denominators) count as FAIL.
4. **Same estimator, same windows, same normalisation** for every model and for the reference.
5. **Baselines are mandatory**; a metric without a baseline is not reported as a win.
6. **Uncertainty on everything:** day-clustered bootstrap (days are the independent unit); report 95% CI.

## 1. Splits
| track | fit | select | report | notes |
|---|---|---|---|---|
| Synthetic World (exact counterfactuals) | streams 200000+ | dev days 5000+ | test days 13000+ | 1000+ calibration only; 9000+ burned; see `splits.py` |
| Real crypto (Coinbase L2+trades, recorded) | first 70% of hours | next 10% | last 20% | chronological, 1 h embargo between blocks, ALL products on the same calendar split (cross-asset leakage) |
| Real equities (LOBSTER, if licensed) | months M1..Mk | M(k+1) | M(k+2) | never the LOB-Impact evaluation month (Jan 2026) in fit |

## 2. Leak controls (each is an automated check in `tests/` or an assertion in the evaluator)
- seed/time ranges disjoint (imported constants, not literals)
- model inputs = observable stream only (events, top-10 book, mid change, parameter-free slow features of those);
  no simulator hidden state
- blind arm: injected orders absent from every model input; asserted by hash (blind == free) and visible != free
- truth blind arm == truth no-injection arm (hash)
- engine start book == stream book at window start
- outputs never overwritten; test runs require a clean committed tree
- normalisers (V, sigma, child size, gap) computed from the evaluated day's *reference market* only, never from model output
- calibration / tuning of anything (World physics, slow-feature grid, hyper-parameters) uses fit or select splits only

## 3. Track A — causal what-if correctness (synthetic, exact truth)
Protocol: LOB-Impact operating point (eta 10%, N=100 aggressive median-size children, gap 9/phi, buy/sell antisymmetrised),
visible/blind on identical seeds, R10 (10 children + 100 silent blocks) and R100 (100 children + 100 silent blocks).
Model rollouts: 32 days x 30 windows = 960; truth: 8 replicas per window.

**Reference certification (dev only, before any model is evaluated):** truth must have identified law, delta CI within
[0.35, 0.65], Y in [0.4, 1.2], R100 end/peak in [0.45, 0.85] (real metaorders: ~2/3 after 1-2 durations), H in [0.42, 0.58].

**Model gates (A-pass requires all):**
| gate | criterion |
|---|---|
| A1 balance | |gamma_m - gamma_t| <= 0.10, |H_m - H_t| <= 0.03, |free drift| <= 5 ticks over 100 blocks |
| A2 law exists | identified (L2 profile interior minimum, |Y| >= 0.05) |
| A3 shape | |delta_m - delta_t| <= 0.10 with bootstrap CI of the difference inside [-0.15, 0.15] |
| A4 amplitude | Y_m / Y_t in [0.67, 1.5] |
| A5 relaxation | |ep100_m - ep100_t| <= 0.15 and slope100_m <= 0 + 2 SE |
| A6 attribution | visible/blind ratio of model within factor 2 of truth's ratio (both fail-closed) |
| A7 absolute (LOB-Impact C4) | reported, not gated: |delta - 0.5| <= 0.10, Y in [0.5, 1] |
| A8 operating-point generalisation (v1.1) | A3-A5 also hold vs truth at held-out participations 5% and 20% (models trained with interventions must not be tuned to the 10% point) |

Primary comparison (pre-registered): `modebank_sb` vs `short`. Other arms are ablations (no multiplicity claims).

## 4. Track B — real-data realism and forecasting (held-out real hours)
- **B1 likelihood:** per-event NLL on test hours vs baselines: (i) unconditional marginal, (ii) empirical conditional on
  book state (queue-reactive style, fitted on fit split), (iii) same architecture without slow memory. Win = CI of NLL
  difference below 0.
- **B2 distributional realism (LOB-Bench style):** free-generation rollouts from test-hour contexts; L1/Wasserstein on
  spread, top-of-book depth, order-flow imbalance, inter-arrival (when Δt is modelled), trade sign runs. Baseline: replay
  of an adjacent fit-split hour (the "noise floor").
- **B3 stylised facts:** sign-ACF exponent, Hurst, signature plot, within tolerances of the real test hours.
- **B4 forecasting:** mid direction at 1 s / 10 s / 60 s; baselines no-change, top-of-book imbalance logistic, EWMA,
  gradient boosting on book features (fit split). Report accuracy/AUC with CI; any trading claim must be net of spread.
- **B5 real impact (no truth available):** LOB-Impact phases on test hours; report delta, Y, R10/R100 relaxation, v/b;
  compare with empirical crypto metaorder literature (to be converted to this Y convention before use).

## 5. Track C — real-time deployment readiness
- C1 latency: p50/p99 per-event update and per-query what-if latency on target hardware (CPU baseline here)
- C2 stability: 24 h continuous state update on live feed without divergence (state norm, surprise bounded)
- C3 monitoring: per-event surprise and regime-shift flag; what-if answers carry a confidence flag and conformal interval
- C4 safety: interface is read-only market view + simulated what-if; it never routes orders

## 6. Reporting
One table per track with every gate, the reference/baseline value, the model value, CI, and PASS/FAIL. No cherry-picking:
all four arms and all gates are reported, including failures.
