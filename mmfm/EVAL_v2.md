# MMFM v2 evaluation protocol (pre-registered 2026-11-26 ~14:55 UTC, before any v2 design work)

## Why v2 needs a new test
v1's sealed test window (04:29–05:00) was opened six times, and its failures now shape the v2 design.
Using it to evaluate v2 would be test-set selection. v2 is evaluated on a **forward test**: data recorded *after*
the v2 model and its calibration are frozen (hash committed).

## Data
* **Design data** is everything recorded up to 2026-11-26 08:00:00 UTC (the recorder died at 07:59 when the container restarted).
  * v2-train: 02:27–07:00 UTC.
  * 10-min embargo.
  * v2-val: 07:10–08:00 UTC. All model selection, decoder choices and calibration (temperatures etc.) are made here only.
* **Forward test** runs from (freeze + 10 min) to (freeze + ≥ 90 min), recorded live after the freeze commit.
  * Nothing from it is inspected before evaluation: no live UI, no data peeking.
  * Every access is logged with the frozen model hash.
* The held-out pair linkusd is excluded from v2 training, selection and calibration (zero-shot test only).

## Pre-registered success criteria (forward test; 1, 5, 10 s horizons; ethusd, xrpusd, dogeusd + linkusd zero-shot)
A criterion counts as met only if it is met on the forward test. Val results are reported but don't count.
1. **No systematic over- or under-generation**: median event-count ratio (simulated / real) within [0.8, 1.25].
2. **Calibrated spread**: share of realised returns inside the simulated central 80% within [0.70, 0.90].
3. **Unbiased location**: PIT mean within [0.40, 0.60].
4. **Accuracy**: MMFM-v2 CRPS ≤ the best count baseline (B0/B1) on the majority of (pair, horizon) cells.
5. **Next-event NLL**: MMFM-v2 beats tuned B2, with a 95% 5-min-block-bootstrap CI excluding 0.
6. **Zero-shot (linkusd)**: report the NLL gap vs B2; target ≤ 0.
7. **Real-time**: ≥ 1× real time for 8 paths on ethusd at its median event rate.

Every result is reported, including misses.

## Addendum v2-1 (data exclusion granularity, decided on design data before training)
* v1's §0 rule excluded a whole connection segment if any verification snapshot failed. In the v2 design data, two BTC
  segments (2.1M events) were excluded because of **one transient extra order each** (Jaccard 0.9999). All other
  verifications in those segments passed, including the next one.
* v2 rule: keep events only within intervals bounded by two consecutive **passing** verifications (the anchor
  counts as passing). An interval bounded by a failure is dropped, and the segment is split there, so no context
  window spans the hole. The book is never corrected; this only changes what is excluded.
* As in v1, the tail after a segment's last verification is kept (it cannot be verified before the connection ends).

## Addendum v2-2 (simulator clock; decided on v2-VAL, before any forward data is built or unsealed)

Observation (v2-val, 3 starts per pair, `--clock model`): the free-running simulator over-generates events
2.0–3.6× already at the 1 s horizon (ethusd 2.9×, dogeusd 2.7×), while teacher-forced dt sampling on the same
val data sums to 0.75–0.95× (ethusd) of the real elapsed time. Diagnosis: exposure feedback in the model's own
clock, not a decoding bug. A dt_scale grid of 0.8–1.25 cannot remove a ~3× bias, and a larger per-pair scale
would be horizon-dependent curve fitting.

Change: the simulator used for criteria 1–4 runs with `clock="anchored"` (sim/rollout2.py docstring):
MMFM generates event content and its internal clock; per path one u ~ U(0,1) draws N_1 ≤ N_5 ≤ N_10
comonotonically from MMFM's own val-temperature-calibrated count heads (real context only), and the model's
internal times are mapped piecewise-linearly onto those knots. Consequences, stated up front:
- criterion 1 (event ratio) now tests MMFM's count heads + the coupling, not free-running timing;
- criteria 2–4 (return coverage/PIT/CRPS) still test the generated event content per event;
- the free-running clock is reported as a diagnostic (`--clock model`, purpose tag `sim-freeclock`), never as the criterion.
Calibration grid on val changes to temperature ∈ {0.85, 1.0, 1.15} with dt_scale = 1.0, 20 starts per pair;
the chosen config is frozen in runs/v2/sim_calibration.json with the model.

B2 fairness note: the first B2-v2 run shared 4 CPU cores with calibration (load ≈ 8) and was aborted at
16 min (log kept as runs/v2_b2_aborted_contention.log); B2-v2 is retrained alone for the full 60 min budget.

Addendum v2-2a (val only): grid 1 (temp 0.85/1.0/1.15, 20 starts) put event ratio at 0.75–1.19 in all 9 cells
but coverage80 at 0.45–0.75, improving monotonically as temperature falls. Grid 2 extends to
temp ∈ {0.6, 0.7, 0.85}, 30 starts per pair; grid 1 is kept as runs/v2/sim_calibration_grid1.json.
The selection rule (loss in calibrate_sim.py) is unchanged.
