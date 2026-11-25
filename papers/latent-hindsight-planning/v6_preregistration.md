# v6: diverse domains where theory predicts large gains (+ declared controls) -- PRE-REGISTRATION (before any run)
Principle: choose tasks by the THEORY's conditions (persistent disturbance, weak/absent feedback), not by results;
include control conditions where the theory predicts no gain; report everything.

## E-A Throw-and-retry (one-shot open-loop decisions; MuJoCo puck)
Puck (mass 1) with hidden viscous damping b ~ U[0.5, 2.0] per axis-shared and hidden wind (fx, fy) ~ N(0, 0.5^2) N.
Action = initial velocity v0 in [-4, 4]^2; free motion for 2.0 s; outcome = final position. Each episode has 10
attempts at NEW random targets in [1, 3] x [-1, 1]; the disturbance persists across the 10 attempts (except 'fresh').
World models: 5-member MLP ensembles Mu(v0) -> final position (phi-blind) and Mc(v0, phi) -> final position, trained
on 20k random throws with phi from the same distribution. Planners (CEM over v0, pop 400, 5 iters):
forward (Mu) | residual (Mu + last observed miss vector) | hindsight (particle posterior over phi, 1000 particles,
likelihood sd = adaptive EMA as in v5 with floor 2x val sd, plans with posterior-expected cost over 16 samples) |
oracle (Mc, true phi). Regimes: wind (b fixed 1.0), damping (no wind), both, fresh (phi redrawn every attempt).
200 episodes x 3 seeds. Metric: miss distance per attempt; summary = mean miss over attempts 2-10.
T1 wind, damping, both: hindsight's summary miss <= 0.5 x forward's (paired bootstrap upper CI of ratio < 0.5).
T2 damping and both: hindsight beats residual (a multiplicative effect does not transfer as a fixed offset).
T3 fresh: hindsight not better than forward.
T4 wind, damping, both: hindsight's attempt-10 miss <= 2 x oracle's attempt-10 miss.

## E-B Observation latency (MuJoCo puck, v3 setup, regime 'both', K = 5)
The planner receives the state d steps late, d in {0, 5, 10, 20}; it predicts the current state by rolling its own
model through the known actions taken since (standard delay compensation), then plans. Hindsight estimators receive
transitions with the same delay. Planners: forward, online_sgd, hindsight_posterior, hindsight_latent_pf_adapt,
oracle_latent. 3 seeds x 20 episodes.
D1 hindsight_posterior's gain over forward is larger at d = 20 than at d = 0, and Spearman(d, gain) > 0.
D2 at d = 20, hindsight_posterior and pf_adapt beat forward AND online_sgd.

## Execution log (deviations, not outcome-dependent)
- E-A: the first 3-seed batch hit the 2 h background limit; seed 0 completed, seed 1 was killed in the `fresh` regime,
  seed 2 never started. Seeds 1 and 2 were rerun from scratch (same code, same seeds) in parallel.
- E-B: the sequential seed loop hit the same limit during seed 0 (delay0 and delay5 printed, nothing written, since results
  are saved at the end of a seed). Fix (commit b43e5b1): ensembles are trained once per seed and cached; each
  (seed, delay) runs as its own process. This is numerically equivalent to the original design: the test schedule RNG is
  re-seeded per delay tag (`10_000+seed`) and the planner RNG is `20_000+seed`. `analyze_v6.py` loader changed to merge
  per-delay files; decision rules D1/D2 untouched.

## E-A outcome (3 seeds x 200 episodes, analysis rules from 648ec46)
Summary miss over attempts 2-10, ratio vs forward (paired bootstrap 95% CI):
| regime | hindsight | residual | oracle |
|---|---|---|---|
| wind | 0.189 [0.181, 0.198] | 0.180 [0.173, 0.189] | 0.030 |
| damping | 0.321 [0.304, 0.338] | 0.603 [0.585, 0.622] | 0.138 |
| both | 0.238 [0.224, 0.253] | 0.391 [0.370, 0.411] | 0.125 |
| fresh (control) | 1.269 [1.254, 1.284] | 1.407 | 0.107 |

- T1 SUPPORTED (3-5x lower miss than forward in all persistent regimes).
- T2 SUPPORTED (beats residual reuse where the disturbance is multiplicative).
- T3 SUPPORTED (fresh: hindsight is *worse* than forward, +27%, as Theorem 2 implies it cannot help and the
  inference adds variance).
- T4 NOT SUPPORTED: attempt-10 miss is 1.5x (both), 2.2x (damping), 6x (wind) the oracle. The learned-model
  posterior converges but plateaus above the oracle; on wind, residual reuse ties hindsight (additive disturbance,
  as Theorem 4 predicts consistency there).
- Not pre-registered, observed: attempt 1 hindsight is 9-24% worse than forward (posterior-expected cost under the
  broad prior is more conservative than the point forward plan).
