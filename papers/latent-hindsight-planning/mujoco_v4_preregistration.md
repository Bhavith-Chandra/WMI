# MuJoCo v4 -- chunk-boundary artefact fix + noise sweep -- PRE-REGISTRATION (before any v4 run)

## A. Artefact (identification R^2 drops every 20 transitions = every chunk boundary)
Diagnosis step (no planner comparison): along latent_pf trajectories, log z_t = mean_d ((dobs - Mc(v,a,phi_TRUE)) / sd_d)^2
and |a_t - a_{t-1}|, and compare boundary steps (t mod 20 == 0) with mid-chunk steps.
Fix (fixed now, applied once, no tuning): particle likelihood becomes Student-t (nu = 4) with per-dimension variance
(2 sd_d)^2 + ensemble variance of Mc's prediction at that particle. Success criterion (identification run, NEW seeds
3-5): mean R^2 over boundary steps t in {20, 40, ..., 140} minus mean R^2 over the 5 steps before each boundary,
averaged over wind_x, wind_y on 'both', moves from its pre-fix value toward 0 by at least half.

## B. Noise sweep (Theorem 4 crossover)
Regime: persistent wind (g = 1) plus i.i.d. gust force N(0, s^2) per axis every step, s in {0, 0.25, 0.5, 1.0} N.
K = 20, 3 seeds x 20 episodes, planners forward, residual, hindsight_posterior, hindsight_latent_pf (fixed version),
oracle_latent (knows the persistent wind, not the gusts).
Predictions:
N1 at s = 0, residual >= hindsight_latent_pf (as in v3 rho = 1).
N2 residual's cost ratio vs forward increases (worsens) monotonically in s.
N3 at s = 1.0, hindsight_posterior AND hindsight_latent_pf beat residual (paired bootstrap).
N4 hindsight_posterior's gain over forward at s = 1.0 is at least half its gain at s = 0.
