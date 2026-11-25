# River crossing: hierarchical planning with/without hindsight -- PRE-REGISTRATION (before any run)

## Task (MuJoCo puck, mass 1, damping 1, actuator gear 5 => |force| <= 5 N, dt 0.05 s)
Start (0, -1.2) at rest; goal (0, 1.2); success = within 0.15 m of goal before step 400 without falling.
River band |y| < 0.5 carries a hidden current force (c, 0) N; no current on the bridge x in [-3.5, -2.5].
Waterfall: x > 2.0 while in the band = irreversible failure (episode ends).
Hidden current c persists per episode. Prior (what nominal planners assume): mixture 80% U[0,3], 20% U[7,10].

## Learned world model
Linear one-step model dv = A v + B u fitted by least squares on random-action data OUTSIDE the river (no current).
Current effect on dv is kappa * c with kappa = B_x / gear (force-to-dv gain, from the same fit).

## Planners
low level (shared): PD-to-waypoint + feed-forward cancellation of the BELIEVED current c_hat.
- flat_forward: go straight to goal, c_hat = prior mean.        - flat_hindsight: straight, c_hat = posterior mean.
- hier_forward: routes {direct, bridge}; picks by simulating its believed model with c_hat = prior mean; on a
  recoverable subgoal failure re-plans with the same belief.
- hier_cautious: always bridge.        - hier_residual: c_hat = last in-band residual / kappa (residual reuse).
- hier_hindsight: c_hat = posterior mean over a grid c in [0, 12] (prior above, Gaussian residual likelihood);
  re-plans whenever the route decision under the current posterior changes; an abort retreats to the near bank.
- hier_oracle: c_hat = true c.
Route rule for all hier_*: direct iff the believed-model rollout of the crossing stays at x < 2.0 - 0.3 and reaches
the far-bank waypoint.

## Calibration rule (oracle and forward ONLY, seed 99, 100 episodes): keep the task as specified if
hier_oracle success >= 0.9 in every regime AND hier_forward success <= 0.5 in 'strong'. Otherwise scale the waterfall
distance in the fixed order [2.0, 3.0, 1.5] and take the first that satisfies both; else report as is.

## Regimes (3 seeds x 200 episodes): calm (c ~ U[0,3]), strong (c ~ U[7,10]), mixed (prior), fresh (c redrawn from the
prior every step).

## Predictions
H1 strong: hier_hindsight success >> hier_forward (difference >= 0.3) and >= 0.8.
H2 calm: hier_hindsight time-to-goal < hier_cautious (it keeps the fast route) and success equal to hier_forward.
H3 mixed: hier_hindsight has the best success-weighted time among non-oracle planners.
H4 fresh: hier_hindsight is not better than hier_forward (Theorem 2 / 5).
H5 hierarchy matters: hier_hindsight success > flat_hindsight in strong (flat has no bridge route).
H6 residual reuse is worse than hier_hindsight in mixed (noisy single-step estimate triggers wrong decisions).

## DEVIATION (before calibration; based on a smoke test of hier_forward/hier_oracle/hier_cautious only)
(1) The simulator is deterministic, so the linear model's residual sd is ~0 -> likelihood sigma = max(fit sd, 0.002).
(2) With independent 5 N motors per axis, a 7-10 N current never sweeps the puck away (forward and oracle both
succeeded 8/8 in 'strong'), so the task had no surprise to exploit. Strong currents become U[15, 25] N, in the prior
mixture (80% U[0,3], 20% U[15,25]) and in the 'strong' regime. Everything else unchanged; calibration rule unchanged.

## OUTCOME (3 seeds x 200 episodes; success rate, mean steps on success)
    calm {'flat_forward': (np.float64(1.0), np.float64(30.3)), 'flat_hindsight': (np.float64(1.0), np.float64(28.0)), 'hier_cautious': (np.float64(1.0), np.float64(87.0)), 'hier_forward': (np.float64(1.0), np.float64(53.0)), 'hier_hindsight': (np.float64(1.0), np.float64(44.0)), 'hier_oracle': (np.float64(1.0), np.float64(44.0)), 'hier_residual': (np.float64(1.0), np.float64(44.0))}
    strong {'flat_forward': (np.float64(1.0), np.float64(62.1)), 'flat_hindsight': (np.float64(1.0), np.float64(62.1)), 'hier_cautious': (np.float64(1.0), np.float64(87.0)), 'hier_forward': (np.float64(0.577), np.float64(94.0)), 'hier_hindsight': (np.float64(0.363), np.float64(91.2)), 'hier_oracle': (np.float64(1.0), np.float64(88.5)), 'hier_residual': (np.float64(0.363), np.float64(91.2))}
    mixed {'flat_forward': (np.float64(1.0), np.float64(36.4)), 'flat_hindsight': (np.float64(1.0), np.float64(34.6)), 'hier_cautious': (np.float64(1.0), np.float64(87.0)), 'hier_forward': (np.float64(0.917), np.float64(57.7)), 'hier_hindsight': (np.float64(0.868), np.float64(47.3)), 'hier_oracle': (np.float64(1.0), np.float64(52.5)), 'hier_residual': (np.float64(0.868), np.float64(47.3))}
    fresh {'flat_forward': (np.float64(1.0), np.float64(30.8)), 'flat_hindsight': (np.float64(1.0), np.float64(31.5)), 'hier_cautious': (np.float64(1.0), np.float64(87.0)), 'hier_forward': (np.float64(1.0), np.float64(52.2)), 'hier_hindsight': (np.float64(1.0), np.float64(60.3)), 'hier_oracle': (np.float64(0.992), np.float64(113.0)), 'hier_residual': (np.float64(0.997), np.float64(74.0))}
    commits {'a36ff57'}
Verdicts: H1 REFUTED (strong: hier_hindsight 0.36 < hier_forward 0.58); H2 SUPPORTED (calm: hindsight 44 steps vs
cautious 87, success 1.0); H3 REFUTED (flat planners best in mixed); H4 SUPPORTED (fresh: no hindsight gain);
H5 REFUTED (flat_hindsight 1.0 > hier_hindsight 0.36); H6 REFUTED (residual == posterior: noise-free simulator,
one residual identifies c exactly).
Diagnosis: (a) the hard-coded abort (retreat to the nearest bank) is slower than finishing the crossing, so correct
inference led to a worse decision; (b) waypoint PD control slows down near the mid-river subgoal, so hierarchical
crossing is slower than flat crossing and gets swept; (c) 15-25 N is passable at full speed, so hierarchy is not
needed. Hindsight inference was right; the re-derived plan was wrong.
