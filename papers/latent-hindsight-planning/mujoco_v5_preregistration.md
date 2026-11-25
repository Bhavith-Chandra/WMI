# MuJoCo v5 -- adaptive particle likelihood -- PRE-REGISTRATION (before any v5 run)
Problem (v4): the particle posterior's likelihood sd is fixed at 2 x validation residual sd of the noise-free model,
so under gust noise it is over-confident and chases gusts (cost ratio 0.821 -> 1.001 from s=0 to s=1).
Fix (fixed now, no tuning): hindsight_latent_pf_adapt keeps, per episode and state dimension, an exponential moving
average (factor 0.98) of the squared one-step residual e_t = dobs - mean_ensemble Mc(v, a, phi_hat_t), computed BEFORE
the update; the likelihood sd is sqrt(max(EMA, (2 sd_val)^2)). EMA initialised at (2 sd_val)^2. Everything else as
hindsight_latent_pf.
Run: noise sweep identical to v4 (wind regime, gust s in {0, 0.25, 0.5, 1.0}, K=20, seeds 0-2, 20 episodes),
planners forward, residual, hindsight_posterior, hindsight_latent_pf, hindsight_latent_pf_adapt, oracle_latent.
Predictions:
L1 s=1.0: pf_adapt beats residual AND beats pf (paired bootstrap).
L2 s=0: pf_adapt not worse than pf.
L3 s=1.0: |ratio(pf_adapt) - ratio(hindsight_posterior)| <= 0.05.
L4 pf_adapt's ratio vs forward rises by <= 0.05 from s=0 to s=1 (noise-robust like the Gaussian posterior).
