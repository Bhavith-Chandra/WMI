# MuJoCo v2 -- PRE-REGISTRATION (written after v1 results, before ANY v2 run)

## Why v2
v1 oracle headroom (true disturbance vs forward) was <= 5%: replanning every step from the measured state is
feedback that cancels most of the disturbance, so no method could show a large gain. A hindsight method can never
beat the oracle, so headroom must be created by the TASK, chosen without looking at hindsight methods.

## Step 1: calibration rule (uses ONLY forward and oracle_latent; calibration seed 99, disjoint from 0-2)
Action chunking: plan every K steps and execute the first K actions of the plan (as in AdaJEPA's action chunks).
Candidate settings, tried in this fixed order; models retrained per setting on its train distribution:
  C1 K=1,  wind_sd=1 (= v1)   C2 K=5, wind_sd=1   C3 K=5, wind_sd=2   C4 K=10, wind_sd=2
CEM horizon H = max(10, K+5). Choose the FIRST setting with oracle headroom >= 20% on regime 'both'
(10 episodes). If none qualifies, use C4 and report the headroom as is.

## Step 2: methods added (designed from theory before any v2 run)
hindsight_latent_cd / hindsight_posterior_cd: as v1 plus CUSUM change detection on the normalised one-step
surprise z_t = mean_d (residual_d / sd_d)^2:  S <- max(0, S + z_t - 1.5); if S > 10: reset estimate to prior
(latent: phi_hat <- prior, window <- last 3 transitions; posterior: precision/info <- prior) and S <- 0.
(kappa=1.5, h=10 fixed a priori; surprise sd from validation residuals.)

## Step 3: predictions for the main run (3 seeds x 20 episodes x 7 regimes), checked by analyze_mujoco.py rules
Q1 in gain/wind/both/ood_gain every hindsight method beats forward, and hindsight_latent captures >= 50% of the
   oracle's headroom.
Q2 switching: the change-detection variants beat their non-detecting versions and beat forward.
Q3 fresh: no hindsight method beats forward; residual worse than forward.
Q4 residual reuse is worse than hindsight_posterior on gain and ood_gain.

## Calibration outcome + logged DEVIATION (before any main run)
Headroom on 'both' (seed 99): C1 4.6%, C2 8.1%, C3 4.0%, C4 10.0% -> none >= 20%. The pre-registered fallback C4
is degenerate: wind_sd=2 exceeds the weak actuator's force (g=0.4 -> 2 N), so forward AND oracle lose control
(costs ~30x v1). Running the main experiment on it cannot inform the question. DEVIATION: extend the grid with
controllable settings and longer open-loop execution, same rule (first with headroom >= 20%):
  C5 K=20, wind_sd=1      C6 K=40, wind_sd=1   (K=40 = one plan per goal segment, the one-shot regime)
If neither qualifies, report the experiment as headroom-limited and do not run the main experiment.

## MAIN OUTCOME (C5: K=20, wind_sd=1; 3 seeds x 20 episodes x 7 regimes; rules from analyze_mujoco.py)
2 of 7 checks supported (generated/mujoco_predictions_v2.txt): Q3, Q3b (fresh). Cost ratio vs forward:
- wind: posterior 0.832, hybrid 0.817, latent 0.865, online_sgd 0.934, oracle 0.370 (all CIs < 1).
  vs online_sgd: posterior 0.891 [0.855,0.922], hybrid 0.875 [0.831,0.909], latent 0.927 -> all significantly better.
- both: posterior 0.873, hybrid 0.864, latent 0.860, online_sgd 0.953, oracle 0.563; hindsight beats online_sgd (0.90-0.92).
- switching: latent 0.930, latent_cd 0.915, residual 0.869, oracle 0.549; vs online_sgd mostly ties.
- fresh: every hindsight method WORSE than forward (posterior 1.33, residual 2.83) -- Thm 2 / Thm 4 confirmed.
- none / gain / ood_gain: every phi-conditioned planner, INCLUDING the true-phi oracle (none 1.186, gain 1.194), is
  WORSE than forward. Diagnostic: Mc with true phi is 17-50x more accurate in 20-step open-loop prediction on
  behaviour-policy actions (none 0.004 m vs 0.065 m; gain 0.004 m vs 0.231 m), so the loss is NOT model inaccuracy on
  training-like actions; likely cause (unverified): CEM exploits Mc's errors on out-of-distribution planned actions.
  Gain-regime conclusions are therefore not valid tests of hindsight until this is resolved.
- change detection (cd) did not help on wind/both (latent_cd 0.961 vs latent 0.865): false alarms.
