# v11 pre-registration: subgoal redrawing when the latent decides what is possible (ramp task)

Committed before the gate, calibration and any test run. Code: `wmi/ramp.py`, `scripts/run_ramp.py`; analysis
`scripts/analyze_v11.py`; tests in `tests/test_planner_integrity.py` (ramp causality, posterior high level invariant
to the true latent, the ramp pushes back only in its corridor, disjoint seeds).

## Why
v10/v10b showed that in the fork task a perfect route choice adds at most 5 points: the latents never made a route
impossible. Here they do. "A box heavier than it looks": route A climbs a ramp, impossible when the motor is weak.

## Task
Puck, same physics and cached models. Start (0,0), goal (3,0), fork point (0.7,0). Central hazard block
x in [1.1,1.9], y in [-0.8,0.8], with a corridor |y| < 0.25 through it: the RAMP, where a constant force -F_s (x
direction) acts. The ramp is a map feature: the world model does not know it. |y| > 1.5 is hazard.
Route A (short): (0.85,0), (2.15,0), goal. Route B (detour below the block): (0.85,-1.15), (2.15,-1.15), goal.
Success: goal within T steps without touching a hazard. Primary metric: score = success * (1 - t_done / T).
Latents and regimes as v7 (gain U[0.4,1.6], wind N(0,1); none, wind, both, fresh). Low level CEM-MPC, K = 10, H = 15.

## Gate (oracle low level; seed-0 models; schedule 99995, planner 95, 60 episodes, regime both)
Ladder (F_s, T): (3.5,160), (3.5,120), (3.0,160), (4.0,160), (3.0,120), (4.0,120). Pass: with the ORACLE low level,
E[max(score_A, score_B)] - max(E[score_A], E[score_B]) >= 0.10 (a perfect per-episode route choice is worth at least
10 points over the best fixed route). If no rung passes, the experiment is reported invalid.

## Calibration and route models
Per model seed s: schedule 800000+s, planner 810000+s, 100 episodes, regime both; forced A and forced B for each low
level. Per low level, L2 logistic regression (quadratic features) of SCORE (fractional target) on (i) the hindsight
belief at the fork, (ii) the true latent. Prior choice: the route with the higher mean calibration score.
Test decision: the route with the higher predicted score (A on ties).

## Test: schedules 900000+s, planner 910000+s, 50 episodes per seed (150 per cell)
low|high: forward|prior, forward|post, hindsight|prior, hindsight|post, hindsight|oracle, hindsight|A, hindsight|B,
oracle|oracle.

## Hypotheses (score; 95% paired bootstrap of differences)
- S1 both: hindsight|post - hindsight|prior > 0.
- S2 both: forward|post - forward|prior > 0.
- S3 both: hindsight|post - best fixed route (hindsight|A or |B, whichever scores higher at test) > 0.
- S4 both: (hindsight|post - hindsight|prior) / (hindsight|oracle - hindsight|prior) >= 0.5 (void if denominator <= 0).
- S5 none: hindsight|post - hindsight|prior lower CI > -0.05.
- S6 fresh: hindsight|post - hindsight|prior NOT significantly > 0.
Reported alongside: success rates, route shares, agreement with the oracle high level.
