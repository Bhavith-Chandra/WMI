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
