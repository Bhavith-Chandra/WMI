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

## OUTCOME (seeds 0-2, 20 episodes; generated/mujoco_v5_summary.txt)
    gust 0.0: {'forward': (1.0, [1.0, 1.0]), 'hindsight_latent_pf': (0.821, [0.781, 0.863]), 'hindsight_latent_pf_adapt': (0.793, [0.751, 0.84]), 'hindsight_posterior': (0.79, [0.744, 0.839]), 'oracle_latent': (0.708, [0.645, 0.773]), 'residual': (0.783, [0.737, 0.831])}
    gust 0.25: {'forward': (1.0, [1.0, 1.0]), 'hindsight_latent_pf': (0.865, [0.815, 0.917]), 'hindsight_latent_pf_adapt': (0.797, [0.753, 0.842]), 'hindsight_posterior': (0.792, [0.744, 0.839]), 'oracle_latent': (0.71, [0.644, 0.777]), 'residual': (0.795, [0.748, 0.844])}
    gust 0.5: {'forward': (1.0, [1.0, 1.0]), 'hindsight_latent_pf': (0.912, [0.866, 0.963]), 'hindsight_latent_pf_adapt': (0.81, [0.764, 0.858]), 'hindsight_posterior': (0.794, [0.75, 0.842]), 'oracle_latent': (0.712, [0.649, 0.777]), 'residual': (0.833, [0.782, 0.885])}
    gust 1.0: {'forward': (1.0, [1.0, 1.0]), 'hindsight_latent_pf': (1.001, [0.94, 1.072]), 'hindsight_latent_pf_adapt': (0.84, [0.795, 0.886]), 'hindsight_posterior': (0.803, [0.759, 0.849]), 'oracle_latent': (0.716, [0.654, 0.778]), 'residual': (0.981, [0.914, 1.057])}
    provenance: commits ['3c40ae8'], seeds 3
    SUPPORTED L1 s=1: pf_adapt beats residual AND pf
    SUPPORTED L2 s=0: pf_adapt not worse than pf
    SUPPORTED L3 s=1: |pf_adapt - posterior| <= 0.05
    SUPPORTED L4 pf_adapt ratio rises <= 0.05 from s=0 to s=1
    SUPPORTED N1 s=0: residual <= latent_pf (not worse)
    SUPPORTED N2 residual ratio vs forward increases monotonically in s
    NOT SUPPORTED N3 s=1: posterior AND latent_pf beat residual
    SUPPORTED N4 posterior gain at s=1 >= half its gain at s=0
L1-L4 all SUPPORTED. The analysis script first crashed on the absent pf_bayes planner (F3 check unguarded); fixed the guard, data unchanged. (N3 refers to the old pf and is unchanged from v4.)
