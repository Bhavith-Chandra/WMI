# River crossing v3 -- PRE-REGISTRATION (after v2, before any v3 run)
Change: the low level is CEM-MPC over raw actions (horizon 15, pop 200, 3 iters, replanned EVERY step) on the believed
model (learned linear dynamics + believed current), cost = sum of squared distance to the current waypoint +
1000 * [predicted fall]. High level = v2 route choice (direct vs bridge) at start and on surprise (|c_hat change| > 2).
forward / hindsight / oracle differ ONLY in the believed current (prior mean / posterior mean / true).
Out-of-arena guard moved to |x| > 12. Same regimes, seeds, calibration rule (forward and oracle only).
Predictions: G1 strong: hindsight success >= forward + 0.3. G2 calm: hindsight success 1.0, time < cautious.
G3 mixed: hindsight highest success among non-oracle. G4 fresh: hindsight not better than forward.
G6 sweep (waterfall 2, 3, 4, 6, 8 m, strong): hindsight success non-decreasing in distance; forward <= hindsight at
every distance; oracle >= 0.9.

## CALIBRATION OUTCOME -> main run NOT executed (decision logged)
oracle / forward success (seed 99, 100 episodes): waterfall 2.0: strong 0.54 / 0.89, mixed 0.90 / 0.97;
3.0: strong 0.54 / 1.00; 1.5: strong 0.54 / 0.44. No setting passed. With a CEM-MPC low level replanning every step,
the FORWARD planner pushes through even 30-40 N currents (89-100% at 2-3 m), while the true-current oracle takes the
bridge and its MPC drifts off the 1 m-wide bridge into the current (54%). The oracle is therefore not an upper bound and
a main run could not test the hypothesis; it is not run. Conclusion across river v1-v3: in this task, closed-loop
replanning (feedback) absorbs the surprise; hierarchy + hindsight never beat the closed-loop baseline, and every
failure of the hierarchical planners came from execution (slow waypoint control, limited plan library, bridge
tracking), not from missing information.
