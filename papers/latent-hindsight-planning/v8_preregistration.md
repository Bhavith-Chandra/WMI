# v8 pre-registration: realistic deployment benchmark (sealed seeds)

Committed before the latency measurement and before any v8 run. Code: `scripts/run_realistic.py`,
`scripts/measure_latency.py`, `wmi/hier.py` (world options), analysis `scripts/analyze_v8.py`.

## Why
v7 tests hierarchical planning with and without hindsight in an idealised world: perfect state, instant actuation,
free compute, and deployment physics identical to training physics except for the latent z. A deployed controller has
none of these. v8 asks whether the v7 conclusion survives realistic deployment.

## Realism (all fixed here; nothing tuned)
- Unmodelled physics: per-episode mass x U[0.8, 1.25] and joint damping x U[0.7, 1.4]. The world models were trained
  at mass 1, damping 1 and never saw these. The oracle is given z = (gain, wind) only, not mass/damping.
- Sensor noise: Gaussian, 0.01 m on position, 0.05 m/s on velocity, on everything the planner sees.
- Actuation delay: 1 control step (50 ms).
- Compute delay: each planner's decision takes compute_steps = ceil(p95 single-thread wall-clock / 50 ms) control
  steps, measured once by `scripts/measure_latency.py` on this machine (E = 1, production configuration, idle
  machine). While it computes, the previous plan keeps executing; the new plan starts from the state observed when
  planning began. A slower planner pays for its compute in the simulation.
- Hazards and success are judged on the true state.

## Seeds (sealed: never used for any design decision)
Sealed s in {100, 101, 102} paired with the cached models of seeds {0, 1, 2}. Test schedules 100000+s, planner RNG
200000+s, world/noise RNG 300000+s. 50 episodes each, 150 per cell.

## Conditions
deploy/{both, none, fresh}: all realism factors; planners hier:{forward, residual, online_sgd, hindsight_posterior,
hindsight_latent_pf_adapt, oracle_latent}. Ablations only_{mismatch, noise, delay}/both: one factor at a time;
planners hier:{forward, hindsight_posterior, oracle_latent}.

## Decision rules (95% paired bootstrap over episodes, difference in success rate)
- R1: deploy/both: hindsight_posterior - forward > 0 (lower CI > 0).
- R2: deploy/both: hindsight_latent_pf_adapt - forward > 0 (it pays its own measured compute delay).
- R3: deploy/none (no z, only realism): hindsight_posterior - forward lower CI > -0.05 (no meaningful harm).
- R4: deploy/fresh: hindsight_posterior - forward NOT significantly > 0.
- R5 (real time): single-thread p95 decision latency of hindsight_posterior <= K * dt = 500 ms, and p95 belief-update
  latency <= dt = 50 ms.
- R6 (graceful degradation): gain of hindsight_posterior over forward in deploy/both >= 0.5 x its gain in v7 K10/both.
Ablations are descriptive.

## Pre-run correction (before any v8 result existed)
While verifying the simulator against closed-form dynamics for the browser demo, I found that changing
`body_mass` after compilation does not change MuJoCo's mass matrix (it must be followed by `mj_setConst`).
The mass-mismatch factor would have been a silent no-op. Fixed in `wmi/hier.apply_physics`; `tests/test_physics.py`
now checks the simulator against the closed form with perturbed mass and damping (error < 1e-9).
No experiment so far used a mass change (the throw experiment changes damping only, which takes effect).

## Outcome (sealed seeds 100, 101, 102 x 50 episodes; all runs from clean trees; rules from 67738b9). All SUPPORTED.
Measured compute delay (1 thread, p95, 50 ms steps): 2 steps for every planner except the oracle (1).
| condition | hier fwd | + residual | + online SGD | + hindsight | + hindsight (particles) | oracle |
|---|---|---|---|---|---|---|
| idealised v7, gain + wind | 0.547 | 0.773 | 0.640 | 0.860 | 0.833 | 0.893 |
| deployment, gain + wind | 0.053 | 0.000 | 0.153 | 0.413 (+0.360 [0.28, 0.44]) | 0.373 (+0.320) | 0.720 |
| deployment, no latent | 0.980 | 0.000 | 1.000 | 0.993 | 0.913 (-0.067 [-0.12, -0.02]) | 0.980 |
| deployment, fresh latent | 0.027 | 0.000 | 0.033 | 0.033 | 0.060 | 0.760 |
One factor at a time (gain + wind): mismatch only 0.513 / 0.840 / 0.867 (fwd / hindsight / oracle); noise only
0.513 / 0.773 / 0.840; actuation + compute delay only 0.067 / 0.460 / 0.820.
R1-R6 supported (R6: gain 0.360 under deployment vs 0.313 idealised).

Reading: the absolute success of every planner falls under realism, mostly because of delay; forward planning nearly
collapses (0.053), hindsight keeps 0.413, so its gain survives and is slightly larger, but it recovers only about half
of the oracle's headroom (vs 90% idealised). Last-residual reuse fails completely under sensor noise, including with no
latent (Theorem 4). The particle posterior costs 6.7 points of success when there is no latent.
Also: v9 seed-2 rerun on a clean tree (commit 16d9882) is bit-identical to the run flagged dirty.
