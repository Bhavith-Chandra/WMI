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
