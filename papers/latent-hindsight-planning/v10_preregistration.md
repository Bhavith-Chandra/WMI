# v10 pre-registration: subgoal redrawing from the hindsight posterior (fork task)

Committed before the gate, the calibration and any test run. Code: `wmi/fork.py`, `scripts/run_fork.py`,
analysis `scripts/analyze_v10.py`. Tests: `tests/test_planner_integrity.py` (posterior high level is invariant to
the true latent; causality for fork planners; disjoint seed streams).

## Question
v7 showed hindsight helps a hierarchical planner when it corrects the LOW level's model while the subgoals stay fixed.
The original framing says hindsight should also make the HIGH level redraw its subgoals. That only matters when the right
subgoals depend on the latent. This task makes them depend on it.

## Task
Puck, same physics and cached models (seeds 0-2). Start (0,0), goal (3,0), fork point (0.6,0). One hazard wall at
x = 1.5 (half-thickness 0.15), |y| > 1.6 hazard. Route A: narrow gap of width `narrow` at y = 0 (subgoals (1.1,0),
(1.9,0), goal). Route B: wide gap of width 0.7 at y = 1.05 (subgoals (1.1,1.05), (1.9,1.05), goal). Success: reach the
goal within T steps without touching a hazard. The high level chooses the route when the puck reaches the fork point
(or at 3 s). Latents as in v7 (gain U[0.4,1.6], wind N(0,1) per axis); regimes none, wind, both, fresh.
Low level: CEM-MPC, K = 10, H = 15, hazard penalty, same as v7.

## Gate (oracle low level only, seed-0 models, schedule 99998, planner 98, 60 episodes, regime both)
Ladder (narrow, T): (0.35,150), (0.35,120), (0.35,100), (0.30,150), (0.30,120), (0.30,100), (0.40,120), (0.40,100).
First rung where both P(A succeeds and B fails) >= 0.15 and P(B succeeds and A fails) >= 0.15 is used. If none passes,
the experiment is reported as invalid (route choice does not depend on the latent in this task).

## Calibration (route-success models; never sees test schedules)
Per model seed s: schedule 400000+s, planner 410000+s, 100 episodes, regime both. For each low level (forward,
hindsight_posterior, oracle_latent) run forced A and forced B on the same episodes. Fit, per low level, L2 logistic
regressions (quadratic features) of success on (i) the hindsight belief at the fork (posterior mean of theta and w on the
velocity dimensions) and (ii) the true latent. Prior choice = route with higher mean calibration success (A preferred
unless B is higher by more than 0.05). Decision rule at test: A if p_A >= p_B - 0.05, else B.

## Test (schedules 500000+s, planner 600000+s, 50 episodes per seed, 150 per cell)
low|high: forward|prior, forward|post, hindsight|prior, hindsight|post, hindsight|oracle, hindsight|A, hindsight|B,
oracle|oracle.

## Hypotheses (95% paired bootstrap of the difference in success)
- S1 (redrawing adds to low-level hindsight): both: hindsight|post - hindsight|prior > 0.
- S2 (redrawing helps a forward low level): both: forward|post - forward|prior > 0.
- S3 (redrawing beats the best fixed route): both: hindsight|post - max(hindsight|A, hindsight|B) > 0, where the max is
  the fixed route with the higher test success (a conservative comparison).
- S4 (headroom): both: (hindsight|post - hindsight|prior) / (hindsight|oracle - hindsight|prior) >= 0.5 (point estimate;
  void if the denominator <= 0).
- S5 (no harm without a latent): none: hindsight|post - hindsight|prior lower CI > -0.05.
- S6 (control): fresh: hindsight|post - hindsight|prior NOT significantly > 0.
Descriptive: route-choice agreement with the oracle high level; failures by route; time to goal.

## Outcome of the v10 gate (oracle low level): FAILED on every rung -> v10 invalid as designed
With the oracle low level, B succeeded where A failed in 0% of gate episodes at every rung (A-only 0.05-0.17). A
controller that knows the latent executes the narrow route in any wind, so the route choice does not depend on the
latent for it. Reported as a finding: redrawing subgoals can only matter when the low level cannot execute the risky
subgoal.

# v10b (amendment, committed before any v10b run)
Same task, planners, calibration, test and hypotheses S1-S6, with three changes:
1. The gate uses the FORWARD low level (gates may use forward and oracle results only; no hindsight result is used):
   pass if forward A-only >= 0.15 and forward B-only >= 0.15. Oracle numbers are recorded for reference.
2. Ladder: the eight v10 rungs idealised, then the same eight rungs with v8 deployment realism (unmodelled mass and
   damping, sensor noise, 1-step actuation delay, measured compute delay: forward 2, hindsight 2, oracle 1 steps).
   The first passing rung fixes (narrow, T, realism) for calibration and test.
3. Seeds: gate schedule 99997, planner 97, world 99996; calibration world 420000+s; test world 700000+s
   (all disjoint; tested). Results: fork_gate_v10b, fork_calib_v10b, fork_main_v10b.
Belief and fork detection use the observed (noisy) state, never the true state.
