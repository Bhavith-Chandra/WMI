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
