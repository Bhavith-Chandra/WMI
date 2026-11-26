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
