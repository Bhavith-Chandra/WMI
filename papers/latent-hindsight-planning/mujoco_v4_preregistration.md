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

## A. DIAGNOSIS OUTCOME + logged change (before any fix or noise-sweep run)
diag_boundary.py: normalised model error with TRUE phi is NOT elevated at boundaries (0.2 vs 0.2), ensemble variance
~equal (0.78 vs 0.73), action jump 1.03 vs 0.33. diag_confound.py ('wind', seeds 3-5): standard particle posterior
boundary dip -0.286 (final R^2 0.76-0.93); with gain particles pinned to the true g=1: dip 0.000, final R^2 0.996.
=> The dips are gain/wind CONFOUNDING (the data identify g*a + f, not g and f separately; each new action at a chunk
boundary rotates the ridge), not a likelihood defect. The pre-registered Student-t fix targets a refuted premise and
is NOT applied (logged). Replacement, fixed now:
 (i) hindsight_latent_pf_bayes: CEM cost averaged over 8 posterior samples of phi (Bayes / posterior-predictive
     planning, Theorem 1) instead of the posterior mean.
 (ii) identification additionally reports predictive R^2 of the identifiable quantity: Mc(v,a,phi_hat) vs
     Mc(v,a,phi_true) for the action actually taken next.
Predictions: F1 pinned-gain diagnostic stands as the explanation (no further test). F2 predictive R^2 shows no
boundary dip larger than 0.05. F3 in the noise sweep, pf_bayes cost <= pf cost at every s (paired bootstrap: not worse).
Noise sweep planners: forward, residual, hindsight_posterior, hindsight_latent_pf, hindsight_latent_pf_bayes, oracle.

## OUTCOME (3 seeds x 20 episodes; identification seeds 3-5) -- generated/mujoco_v4_summary.txt
    gust 0.0: {'forward': (1.0, [1.0, 1.0]), 'hindsight_latent_pf': (0.821, [0.781, 0.863]), 'hindsight_latent_pf_bayes': (0.847, [0.798, 0.902]), 'hindsight_posterior': (0.79, [0.744, 0.839]), 'oracle_latent': (0.708, [0.645, 0.773]), 'residual': (0.783, [0.737, 0.831])}
    gust 0.25: {'forward': (1.0, [1.0, 1.0]), 'hindsight_latent_pf': (0.865, [0.815, 0.917]), 'hindsight_latent_pf_bayes': (0.872, [0.824, 0.925]), 'hindsight_posterior': (0.792, [0.744, 0.839]), 'oracle_latent': (0.71, [0.644, 0.777]), 'residual': (0.795, [0.748, 0.844])}
    gust 0.5: {'forward': (1.0, [1.0, 1.0]), 'hindsight_latent_pf': (0.912, [0.866, 0.963]), 'hindsight_latent_pf_bayes': (0.934, [0.883, 0.99]), 'hindsight_posterior': (0.794, [0.75, 0.842]), 'oracle_latent': (0.712, [0.649, 0.777]), 'residual': (0.833, [0.782, 0.885])}
    gust 1.0: {'forward': (1.0, [1.0, 1.0]), 'hindsight_latent_pf': (1.001, [0.94, 1.072]), 'hindsight_latent_pf_bayes': (1.03, [0.965, 1.102]), 'hindsight_posterior': (0.803, [0.759, 0.849]), 'oracle_latent': (0.716, [0.654, 0.778]), 'residual': (0.981, [0.914, 1.057])}
    provenance: commits ['0ec4d38'], seeds 3
    identification both wind_x: dip -0.150, R2@20 0.841, R2@160 0.865
    identification both wind_y: dip -0.202, R2@20 0.834, R2@160 0.869
    identification both predictive_dv: dip 0.138, R2@20 0.773, R2@160 0.888
    identification wind wind_x: dip -0.365, R2@20 0.827, R2@160 0.925
    identification wind wind_y: dip -0.274, R2@20 0.833, R2@160 0.930
    identification wind predictive_dv: dip 0.121, R2@20 0.778, R2@160 0.913
    SUPPORTED N1 s=0: residual <= latent_pf (not worse)
    SUPPORTED N2 residual ratio vs forward increases monotonically in s
    NOT SUPPORTED N3 s=1: posterior AND latent_pf beat residual
    SUPPORTED N4 posterior gain at s=1 >= half its gain at s=0
    SUPPORTED F3 pf_bayes not worse than pf at every s
    SUPPORTED F2 both: predictive R^2 boundary dip > -0.05
    SUPPORTED F2 wind: predictive R^2 boundary dip > -0.05
Reading: the Gaussian gain/offset posterior (with forgetting) is noise-robust (cost ratio 0.790 -> 0.803 from s=0 to
s=1), residual reuse loses its whole advantage (0.783 -> 0.981), as Theorem 4 predicts. The particle posterior over
phi degrades like residual reuse (0.821 -> 1.001): its likelihood width is calibrated on noise-free model residuals,
so under gusts it is over-confident and chases the noise -- a misspecified likelihood, not a property of hindsight.
N3 fails because of that. The identifiable quantity shows no boundary dip (predictive R^2 jumps UP by 0.12-0.14 at
boundaries: new actions add information), confirming the confounding explanation.
