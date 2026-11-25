# River crossing v2 -- PRE-REGISTRATION (after v1 outcome, before any v2 run)

Changes from v1 (motivated by v1 diagnosis):
1. Strong current U[30, 40] N (prior: 80% U[0,3], 20% U[30,40]) so that the direct crossing is genuinely impassable
   and only the bridge works -> hierarchy is necessary.
2. Intermediate waypoints are pass-through (switch at 0.3 m instead of 0.15 m) for ALL hierarchical planners, so
   subgoal structure no longer slows the crossing.
3. The hindsight module is a RE-PLANNER, not a fixed abort rule: on a surprise (belief about c moves by > 2 N since the
   last plan), at start, and on subgoal failure, it simulates four candidate subgoal sequences from the CURRENT state
   under the believed model -- push to far bank then goal; straight to goal; retreat to near bank then bridge;
   bridge -- and executes the fastest one predicted to reach the goal without falling.
Planners: flat_forward, flat_hindsight, hier_cautious (always bridge), hier_replan_forward (same re-planner, belief =
prior mean: re-plans only at start and on subgoal failure), hier_replan_hindsight (belief = posterior mean),
hier_replan_oracle (belief = true c). Calibration rule unchanged (oracle and forward only).

Predictions (3 seeds x 200 episodes):
G1 strong: hier_replan_hindsight success >= 0.8 and >= hier_replan_forward + 0.3; flat planners <= 0.2.
G2 calm: hier_replan_hindsight success 1.0 and mean time < hier_cautious.
G3 mixed: hier_replan_hindsight has the highest success among non-oracle planners and time < hier_cautious.
G4 fresh: hier_replan_hindsight is not better than hier_replan_forward in success.
G5 strong: hier_replan_hindsight success > flat_hindsight (hierarchy needed in addition to hindsight).
Also reported (Theorem 5 condition): in 'strong', fraction of hindsight episodes that entered the current and survived.

## DECLARATION + ADDED EXPERIMENT (before calibration and main runs)
A 6-episode functional smoke test at c = 35 N, waterfall 2.0 was seen for ALL planners: hindsight re-planner 0/6,
forward 0/6, flat 0/6 (all enter at speed and fall), oracle and cautious 6/6. The main run proceeds exactly as
pre-registered (G1 is now expected to fail). Added, theory-driven (Theorem 5: hindsight helps iff detection+abort
precede irreversibility):
G6 waterfall sweep, regime 'strong', distances {2, 3, 4, 6, 8} m, 3 seeds x 200 episodes, planners
hier_replan_forward / hier_replan_hindsight / hier_replan_oracle: hindsight success is non-decreasing in distance and
rises from <= 0.2 at 2 m to >= 0.8 at 8 m; forward stays <= 0.2 at every distance; oracle >= 0.9 at every distance.
