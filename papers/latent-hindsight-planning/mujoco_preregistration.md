# MuJoCo interactive hindsight-planning experiment -- PRE-REGISTRATION (written before any run)

## System
MuJoCo puck (mass 1, 2 slide joints, damping 1, two motors gear 5, control dt 0.05 s = 5 x 0.01 s).
Hidden disturbance phi = (g, fx, fy): actuator GAIN g (scales gear) and additive external force (fx, fy) N.
Train distribution: g ~ U[0.4, 1.6], f ~ N(0, 1^2) per axis, fixed within a 50-step training episode.

## World models (learned, never see test episodes)
- M_u(v, a) -> Delta s : unconditioned MLP (cannot know phi).
- M_c(v, a, phi) -> Delta s : conditioned MLP (knows the disturbance form; phi must be supplied/inferred).
Trained on random smooth-action episodes; validation split by episode; early stopping on validation.

## Task
160-step episodes; goal position changes every 40 steps (4 goals in [-1,1]^2); start at rest at origin.
Cost per step ||p - goal||^2 + 0.01 ||a||^2. Metric: mean cost per step (lower better), plus final-10-steps
position error per goal segment ("steady-state error"). 20 episodes per regime per seed, 3 seeds.

## Planners (identical CEM: horizon 10, pop 300, 30 elites, 3 iters, warm-started)
forward (M_u) | residual reuse (M_u + last observed residual, Balestriero/LeCun/Varaku rule applied sequentially) |
online_sgd (M_u, last layer, 1 SGD step per transition, AdaJEPA-style) | hindsight_posterior (M_u + Bayesian
gain-on-action-effect + offset per axis, forgetting 0.98) | hybrid (online_sgd base + hindsight_posterior) |
hindsight_latent (M_c with phi inferred by gradient descent on the last 30 transitions, T2-Z style) |
oracle_latent (M_c with the TRUE phi; upper bound for the model class).

## Regimes (test)
none | gain (persistent g) | wind (persistent f) | both | switching (phi redrawn every 55 steps) |
fresh (iid wind gusts each step, g=1) | ood_gain (g ~ U[0.2, 0.4], outside training range, persistent).

## Predictions (from Theorems 1-4)
P1 persistent regimes (gain, wind, both): every hindsight planner beats forward; hindsight_latent ~ oracle_latent.
P2 wind (additive) : residual reuse beats forward but is worse than hindsight_posterior (Thm 4(i)).
P3 gain / ood_gain : residual reuse degrades (Thm 4(ii)); hindsight_posterior and hindsight_latent do not.
P4 fresh           : no hindsight planner beats forward by more than noise; residual reuse is worse than forward.
P5 none            : residual reuse is worse than forward (Thm 4(iii)); others ~ forward.
P6 switching       : planners with forgetting/windows (posterior, latent, sgd) recover; forward does not adapt.
P7 hybrid vs online_sgd: hybrid better on gain/ood_gain (gain identified through the action channel), ~equal
   elsewhere. (Real-data result predicts ~equal; this is the regime where they should differ.)

## OUTCOME (scripts/analyze_mujoco.py, rules fixed before results; 3 seeds x 20 episodes per regime)
5 of 11 checks supported (generated/mujoco_predictions.txt). Dominant finding, not anticipated in the
pre-registration: HEADROOM IS SMALL. Even oracle_latent (true disturbance) improves on forward by only
0.2% (gain), 2.0% (wind), 4.6% (both), 1.6% (ood_gain), 5.0% (switching). Forward MPC replans from the measured
state every step, which is already feedback and removes most of the disturbance's effect on cost; hindsight can
only recover the remaining offset. Where headroom exists, hindsight_latent tracks the oracle (both 0.961 vs
0.954; wind 0.981 vs 0.980; ood_gain 0.989 vs 0.984) and residual reuse is worse than forward on both/ood_gain.
