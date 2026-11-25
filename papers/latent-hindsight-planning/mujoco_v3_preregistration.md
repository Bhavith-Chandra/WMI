# MuJoCo v3 -- PRE-REGISTRATION (after v2 results, before ANY v3 run)

## Fix for planner exploitation (v2: true-phi oracle 19% WORSE than forward on none/gain)
All world models become ensembles of 5 MLPs (independent init + bootstrap minibatches). Planning uses the ensemble
mean and adds beta * sum_t ||ensemble std of predicted delta-position||^2 / (0.01 m)^2 ... normalised by the
validation residual sd, to the CEM cost (PETS-style pessimism). Online SGD keeps one head per member.
GATE (calibration seed 99, forward vs oracle ONLY, 10 episodes, regimes none/gain/wind/both):
  beta ladder [0.1, 1, 10]; take the first beta where oracle <= forward on ALL four regimes
  (mean ratio <= 1.0). If none passes, report v3 as still confounded and run only the wind/both regimes.

## New estimator
hindsight_latent_pf: particle posterior (256 particles/episode) over phi with the conditioned-ensemble likelihood
N(delta_obs; Mc_mean(v,a,phi), (2 * val_resid_sd)^2); broad prior g ~ U[0.1, 2.0], f ~ N(0, (1.5 wind_sd)^2);
resample when ESS < N/2; each particle re-drawn from the prior w.p. 0.02 per step (handles switches).
Plans with the posterior mean of phi.

## Experiments (3 seeds x 20 episodes each)
E1 main      : K=20, wind_sd=1, all 7 regimes, planners forward, residual, online_sgd, hindsight_posterior, hybrid,
               hindsight_latent, hindsight_latent_pf, oracle_latent.
E2 K-sweep   : regimes wind, both; K in {1,5,10,20,40}; planners forward, online_sgd, hindsight_posterior,
               hindsight_latent_pf, oracle_latent.
E3 rho-sweep : K=20; wind is AR(1) per step f_t = rho f_{t-1} + sqrt(1-rho^2) wind_sd xi_t (same stationary variance),
               g=1; rho in {0, 0.9, 0.99, 1.0}; planners forward, residual, hindsight_posterior,
               hindsight_latent_pf, oracle_latent.

## Predictions (verdicts by paired bootstrap, same rules as analyze_mujoco.py)
R1 E1: hindsight_latent_pf beats forward AND online_sgd on wind, both, gain, ood_gain.
R2 E1: hindsight_latent_pf captures >= 50% of oracle headroom on wind and both.
R3 E2: latent_pf's relative gain over forward is larger at K=40 than at K=1, and Spearman(K, gain) > 0.
R4 E3: latent_pf's gain over forward is monotone in rho (0 < 0.9 < 0.99 < 1.0); no significant gain at rho=0.
R5 E3: residual reuse is worse than forward at rho = 0 and rho = 0.9.
R6 E1: on fresh, no hindsight method beats forward.

## GATE OUTCOME + DIAGNOSIS + logged DEVIATION (before any main/sweep run)
Gate failed at every beta (oracle/forward on none: 1.23, 1.21, 1.06). Diagnosis (scripts/diag_oracle*.py):
(1) single plans from rest: oracle better (4.96 vs 5.74) and its model accurate (pred 4.97 vs true 4.96) -> not
exploitation, not search failure; (2) warm start irrelevant; (3) from the SAME real t=40 state the oracle replans better
(11.73 vs 12.61; 14.58 vs 15.36), but the episodes reach DIFFERENT t=40 states: oracle parks on the old goal
(0.018 m), forward stops 0.145 m off with residual velocity, a better staging point for the unseen next goal.
The loss is a benchmark artefact (chained random goals reward imprecision), not a property of hindsight.
DEVIATION: reset the puck to a fresh random start at each 40-step goal segment; phi and every planner's memory
persist across segments. Gate re-run with the same beta ladder and pass rule.
