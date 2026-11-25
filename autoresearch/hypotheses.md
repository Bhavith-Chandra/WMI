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
