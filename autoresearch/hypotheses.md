# Pre-registered hypotheses (written before each run; verdict filled after)

## AR-0 baseline: static Bayes (assumes persistence)
Prediction: strong gains (ratio << 1) on persistent/none; ~1.06 on fresh (misspecified); > 1 on switching
and fast drift because it never forgets.
Result: SCORE 0.24911 (commit 0274fa0, estimator 9c009efe3a). Verdict: SUPPORTED. switching 1.64 (worse than
forward), drift_mid 0.94 / drift_fast 0.83 (little gain), fresh 1.07 (misspecified, as predicted by Thm 2).
KEPT as baseline.

## AR-1 interacting multiple model (IMM) over persistence rates
Hypothesis: the gap on switching/drift comes from assuming zero process noise. A bank of Kalman filters with
random-walk process noise q in {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1} (per coefficient, per step), mixed by
predictive likelihood with a sticky Markov switch (p_stay = 0.99), learns how persistent the world is.
q=0 recovers AR-0; q large approaches residual reuse (Kalman gain -> 1).
Prediction: switching and drift ratios fall well below 1; persistent/none slightly worse than AR-0 (mixing
cost); fresh stays ~1.0-1.07; overall SCORE improves by > 0.5%.
Result: SCORE 0.08672. Verdict: PARTLY SUPPORTED. switching 1.64->0.041, drift 0.02-0.03; but fresh
1.07->1.56, none 0.035->0.14, real Silverbox 0.91->1.25 (regressions: high-q filters chase non-persistent noise).
KEPT (score rule), regressions recorded.

## AR-2 add a 'fresh' model (prior re-drawn every step) to the IMM bank, drop q=1e-1
Hypothesis: the regressions are a missing hypothesis, not over-adaptation per se. Adding the correctly
specified model for fresh disturbances (which equals forward, Thm 2) lets the mixture fall back to forward
when outcomes are not persistent.
Prediction: fresh -> ~1.0-1.1, none and Silverbox improve vs AR-1; switching/drift stay << 1; SCORE improves.
Result: SCORE 0.08751 (fresh 1.011, none 0.124, Silverbox 0.967 -- regressions fixed). Verdict on the
hypothesis: SUPPORTED. Verdict under rule v1 (score must improve >0.5%): DISCARDED (score 0.9% worse).

## METRIC REVISION v2 (human decision, made AFTER seeing AR-1/AR-2; logged as such)
Problem: the geometric-mean score rewards huge wins on easy regimes and hides regressions below forward
(AR-1 is 56% WORSE than forward on fresh). A hindsight method that can be worse than ignoring outcomes defeats
the paper's claim. New acceptance rule v2: KEEP iff worst per-task ratio <= 1.05 AND score improves >0.5% over
the best v2-admissible kept score. AR-0 (worst 1.635) and AR-1 (worst 1.560) are NOT v2-admissible, so the
v2 incumbent starts empty. eval.py now also reports WORST.

## AR-3 = re-test of AR-2 code under rule v2
Prediction: worst <= 1.05 (fresh ~1.01), becomes the first v2-admissible incumbent.
Result: SCORE 0.08751, WORST 1.0114. Verdict: SUPPORTED; first v2-admissible incumbent. KEPT.

## SEALED TEST of AR-3 (official real TEST splits, 3 seeds, EMPS excluded; results/realdata_sealed_AR3/)
Verdict: FAILED TO TRANSFER. Geomean ratio vs forward over 15 test records: 1-step 0.984, 10-step 1.055
(10-step worse than forward on 11/15). The a-priori static posterior with forgetting does better on the same
records (1-step 0.971, 10-step 0.916), and online last-layer SGD is best at 1-step (0.903).
Diagnosis: the dev suite is 9/13 synthetic tasks and scores 1-step error only; the IMM's fast-drift filters
track high-frequency model error, which helps 1-step and compounds over a 10-step rollout.
Next (metric v3, to be pre-registered before any further run): score the 10-step planning horizon, weight real
dev tasks equally with synthetic ones, and add online-SGD as a second reference so a method must beat both.

# ===== Metric v3 (eval_v3.py), pre-registered BEFORE running any v3 candidate =====
Candidates (code frozen in autoresearch/candidates/ before running):
- V3-0 static posterior + forgetting, frozen base. Prediction: synthetic fresh ~1.06-1.07 -> WORST > 1.05,
  so NOT admissible; real 10-step ratio vs min(fwd,sgd) ~0.9-1.0.
- V3-1 HYBRID online-SGD base + static-forget posterior. Prediction: best real 10-step ratio (<1 on most
  datasets) because SGD fixes 1-step model error and the posterior fixes persistent gain/offset; synthetic same
  as V3-0 (base irrelevant there) -> WORST likely > 1.05 from fresh.
- V3-2 AR-3 IMM (control). Prediction: v3 must DETECT its sealed failure: real 10-step ratio > 1 somewhere.
- V3-3 IMM with slow rates + fresh only. Prediction: keeps fresh ~1.01 (admissible synthetic) and loses less
  at 10 steps than AR-3; may be the first v3-admissible candidate.

### v3 results (results_v3.tsv): NO candidate admissible (WORST <= 1.05).
- V3-0 SCORE 0.359 WORST 1.410 (switching). Real 10-step vs min(fwd,sgd): Cascaded 0.58, Silverbox 1.28, WH 1.09,
  ParWH 1.12 -> beats forward everywhere (0.47-0.90) but loses to online SGD on 3/4. Prediction partly wrong
  (predicted 0.9-1.0 vs min baseline).
- V3-1 HYBRID SCORE 0.340 WORST 1.410. Real: 0.57, 1.004, 1.038, 0.994 -> roughly ties online SGD except Cascaded
  Tanks. Prediction "<1 on most datasets": REFUTED (only 2/4 below 1, and by <1% on ParWH).
- V3-2 AR-3 control: real 0.60, 1.42, 1.67, 1.93 -> v3 DETECTS the sealed failure on dev data. SUPPORTED.
- V3-3 IMM slow: real 0.67, 1.39, 1.40, 1.42; synthetic worst 1.011. REFUTED (still loses at 10 steps).
Finding: on these real systems, online weight adaptation captures most of what hindsight correction offers; the
posterior adds value on top of it only on Cascaded Tanks.

## V3-4 HYBRID online-SGD base + IMM{static, q=1e-5, fresh}
Hypothesis: switching/fresh synthetic failures need the fresh model; real 10-step losses came from fast-drift
filters, so dropping them while keeping SGD base gives synthetic WORST ~1.01 and real ~V3-1 (0.57-1.04).
Prediction: WORST in [1.01, 1.05] only if real WH/Silverbox stay <= 1.05; borderline.
## V3-5 HYBRID online-SGD base + full AR-3 IMM. Prediction: real worse than V3-4 (fast filters), not admissible.
Results: V3-4 SCORE 0.263 WORST 1.292 (real WH 1.22, ParWH 1.29); V3-5 SCORE 0.233 WORST 1.571.
Both NOT admissible. V3-4 prediction REFUTED; V3-5 SUPPORTED (worse than V3-4). Pattern: any IMM-mixed bank loses
at 10 steps vs online SGD on WH/ParWH -- mixing-induced partial resets make the correction noisy and it compounds.

## V3-6 HYBRID online-SGD + Bayesian model averaging {persistent(lambda=.995), fresh}, no state mixing
Prediction: real 10-step close to V3-1 (<= 1.04) because the gate shrinks the correction when residuals look
fresh; synthetic switching < 1 (faster forgetting than V3-1) but likely > the IMM's 0.04; fresh ~1.0.
Admissible iff real WH <= 1.05 -- uncertain.
Result V3-6: SCORE 0.317, WORST 1.066 (real WH). Synthetic: fresh 1.0003, switching 0.50, none 0.043. Real:
Cascaded 1.000, Silverbox 1.003, WH 1.066, ParWH 1.012 -> the evidence gate selects 'fresh' on real dev data, i.e.
after online SGD the residuals carry ~no exploitable persistent gain/offset. NOT admissible (by 1.6%).

## STOP of v3 tuning (decision logged before any test data is seen)
7 candidates, none admissible. Further tuning on the same dev data = forking paths. EXPLORATORY sealed tests
(not winner-selection) on TWO candidates fixed now: V3-1 (best real dev) and V3-6 (closest to admissible). Both
are reported regardless of outcome; neither is "selected" on test.
