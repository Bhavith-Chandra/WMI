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
