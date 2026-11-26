# v9 pre-registration: goal-gap correction and hindsight relabeling (throw-and-retry)

Committed before any v9 run. Code: `wmi/throw.py` (kinds gap_ilc, gap_mean, anchor, hindsight_anchor), analysis
`scripts/analyze_v9.py`. Existing kinds are bit-identical to the v6 code (checked against the previous commit).

## Origin
The user's original framing of hindsight planning lists four mechanisms: counterfactual disturbance inference (tested in
v3-v8), goal relabeling ("I missed the bullseye but hit triple-20, so now I know how to hit triple-20"), hindsight
optimisation, and reflection that redraws subgoals. The user then proposed: measure the gap between the previous outcome
and the goal, and plan to close it. v9 tests that proposal literally and in its relabeling form.

## Planners (same task, models, seeds, schedules and planner RNG as v6; 3 seeds x 200 episodes x 10 attempts)
- gap_ilc (the proposal, literal): aim so that predicted outcome + c = goal; after each throw c <- c + (outcome - goal).
  An integral controller on the outcome-to-goal gap.
- gap_mean: c = mean over past attempts of (outcome - model prediction for the executed action).
- anchor (relabeling): pick the past attempt whose achieved outcome is closest to the current goal; predict
  outcome(a) = y* + f(a) - f(a*). The model predicts only the difference from a true example.
- hindsight_anchor: the v6 particle posterior (expected cost over 16 samples), with the anchor correction applied
  under each posterior sample: Mc(a, z_s) + (y* - Mc(a*, z_s)).
- forward is re-run as a replication check: it must equal the stored v6 forward results exactly (same models).
- residual, hindsight, oracle are taken from the stored v6 results (identical code path, verified).

## Metric
Mean miss over attempts 2-10 per episode, ratio to forward (paired bootstrap, 10,000 resamples), and attempt-10 miss.

## Hypotheses (from the theory, before any run)
- G1 (literal gap proposal): wind: gap_ilc ratio upper CI < 0.5. damping and both: gap_ilc worse than hindsight
  (hindsight/gap_ilc upper CI < 1). fresh: gap_ilc worse than forward (lower CI > 1).
- G2 (relabeling beats last-gap): damping and both: anchor beats residual (anchor/residual upper CI < 1).
- G3 (relabeling closes part of the oracle gap): wind, damping and both: hindsight_anchor beats hindsight
  (upper CI < 1); and attempt-10 miss of hindsight_anchor <= 2 x oracle in at least two of the three regimes.
- G4 (control): fresh: anchor and hindsight_anchor are not better than forward (lower CI >= 1).
- R0 (replication): v9 forward equals v6 forward exactly; otherwise v9 comparisons with stored results are void.
