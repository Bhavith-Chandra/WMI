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
