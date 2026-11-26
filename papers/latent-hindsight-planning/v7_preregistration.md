# v7 pre-registration: hierarchical planning vs hindsight planning (E-H)

Written and committed before the gate and before any outcome of any hindsight planner was seen.
Code: `wmi/hier.py`, `scripts/run_hier.py`; analysis: `scripts/analyze_v7.py` (committed with this file).

## Question
The paper's thesis: a hierarchical planner fails its main goal when a subgoal fails because of a persistent surprise,
and conditioning on outcomes (hindsight) prevents this. Earlier river-crossing tests did not support it (feedback and
execution dominated). This experiment tests it where the theory's conditions hold by construction: persistent latent,
open-loop commitment (K=10), and irreversible subgoal failure.

## Task
MuJoCo puck (same physics and same cached ensemble models as v3/latency). Start (0,0), goal (4,0). Three hazard walls at
x = 1, 2, 3 (half-thickness 0.15) with gaps of width `gap` centred at y = +0.5, -0.5, +0.5; |y| > 1.2 is hazard.
Touching a hazard ends the episode as a failure (irreversible). Success: within 0.25 of the goal within T = 300 steps
(15 s). Hierarchical planners receive subgoals = approach and exit points of each gate (x_w -/+ 0.35, y_c), then the
goal; a subgoal is reached within 0.2. Flat planners plan directly to the goal. All low-level planners: CEM-MPC on the
learned ensemble with a hazard penalty in the rollout, H = 15, open-loop chunks K = 10, beta from the v3 gate.
Latent regimes: none; wind (gain 1, wind N(0,1) per axis, persistent); both (gain U[0.4,1.6] + wind, persistent);
fresh (both, redrawn every 20 steps).

## Planners
flat:forward, flat:hindsight_posterior, hier:forward, hier:residual, hier:online_sgd, hier:hindsight_posterior,
hier:hindsight_latent_pf_adapt, hier:oracle_latent. Seeds 0, 1, 2 (cached models of those seeds), 50 episodes each.
K-sweep (regime both): K in {1, 5, 20} for hier:forward, hier:hindsight_posterior, hier:oracle_latent (K=10 from main).

## Gate (forward and oracle only; seed-0 models, separate test RNG 99999, 20 episodes)
Ladder of gap widths [0.5, 0.7, 0.4], first that passes is used: oracle success in `none` >= 0.8 (task solvable) and
oracle - forward success in `both` >= 0.10 (the latent matters). If none passes, the experiment is reported as invalid.

## Hypotheses and decision rules (95% paired bootstrap over episodes, 150 per cell, difference in success rate)
- H1 (hierarchy is needed): `none`: hier:forward - flat:forward > 0 (lower CI > 0).
- H2 (main): `wind` and `both`: hier:hindsight_posterior - hier:forward > 0 AND hier:hindsight_latent_pf_adapt -
  hier:forward > 0 (lower CI > 0 in all four).
- H3 (control): `none` and `fresh`: hier:hindsight_posterior - hier:forward is NOT significantly > 0 (lower CI <= 0).
- H4 (complementarity, hindsight does not replace hierarchy): `wind` and `both`: hier:hindsight_posterior -
  flat:hindsight_posterior > 0.
- H5 (inference beats cheap corrections): `both`: hier:hindsight_posterior - hier:residual > 0 and
  hier:hindsight_posterior - hier:online_sgd > 0.
- H6 (headroom): `both`: (hind - fwd) / (oracle - fwd) >= 0.5 with hind = hier:hindsight_posterior (point estimate).
- H7 (commitment): `both`: success gain of hier:hindsight_posterior over hier:forward is larger at K=20 than at K=1,
  and Spearman(K, gain) over K in {1,5,10,20} > 0.
Secondary (descriptive, no rule): subgoal index at failure; time to success; trajectories.

## Outcome, main run (3 seeds x 50 episodes, K = 10; analysis rules from 675de01)
Main-goal success (difference to hier:forward, 95% paired bootstrap):
| regime | flat fwd | hier fwd | hier + residual | hier + online SGD | hier + hindsight | hier + hindsight (particles) | hier oracle |
|---|---|---|---|---|---|---|---|
| none | 0.407 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| wind | 0.093 | 0.727 | 0.953 | 0.840 | 0.940 (+0.213 [0.147, 0.280]) | 0.947 | 0.980 |
| both | 0.040 | 0.547 | 0.773 | 0.640 | 0.860 (+0.313 [0.240, 0.393]) | 0.833 | 0.893 |
| fresh | 0.013 | 0.360 | 0.393 | 0.373 | 0.253 (-0.107 [-0.200, -0.013]) | 0.373 | 0.940 |

H1-H6 SUPPORTED (H6 capture 0.904). H7 SUPPORTED: success gain of hindsight over forward by chunk length K = 1, 5, 10, 20: +0.053, +0.120, +0.313, +0.400 (monotone; 3 seeds each).

Unexpected, not pre-registered: flat + hindsight succeeds in 0% of episodes in every regime, including `none` (flat forward 0.407).
Diagnosis (ad hoc, seed-0 models): failures are head-on impacts on the first wall near y = 0.1, about step 27; speeds
are within the training distribution (max 1.2-1.7 m/s vs training p99 4.4 m/s), so it is not extrapolation. With no
latent at all, the posterior acquires a spurious offset (w on vx about -0.025 per step, ~ -0.5 N; theta about +0.15):
it absorbs the world model's own bias. The likely mechanism is that the biased belief predicts the puck stops short of
the wall while it does not. The hierarchical planner's subgoals lie in the open gap row, so it does not depend on precise
stopping in front of a wall. Status: likely, not proven. Lesson: hindsight can absorb model bias into the latent and
hurt when the plan depends on exact stopping distances near hazards.
