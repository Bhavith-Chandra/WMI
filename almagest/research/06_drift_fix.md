# 06 — Closed-loop drift on real data: what was actually wrong (2026-11-26/27)

**Setup.** Every number below is B2 (EVAL_PROTOCOL 4.1, pre-registered rule) on the select hour of split_v1:
128 windows × 4000 generated events (`B2_SEED=2`). The two references pass 12/12 on every run:
- oracle replay (real events, real lots);
- oracle with sampled sizes.

So every failure below is model error, not representation error.

## 1. Bugs found (in the order found)

| # | bug | effect | fix |
|---|---|---|---|
| 1 | Size head saw only (type, side), not the level or the queue it hits | Whole-queue cancels at the best: generated 6% vs real 21%. Limit sizes at the best ×2. Top depth ×2 within 200 events | Size conditioned on (level, log queue) (v3.4) |
| 2 | Generator slow-bank state at rollout start recomputed from the row start, not from the whole contiguous run | Generation features off by up to 17. Invalidated all earlier B2/RL numbers (including the v3c1 REPORT B2) | Exact inversion from the training features; asserted at runtime |
| 3 | **Training batches never shuffled.** Same length-sorted order every pass, so each pass ended on the same short segments | Model miscalibrated on its **own training data**: P(market) 0.024 vs 0.042. Single-run B2 scores were seed lottery (one config gave 6 / 4 / 2 of 12 across seeds) | Shuffle within length buckets, shuffled batch order, cosine LR (v3.8) |

After fix 3, the model is calibrated:

| | model | real |
|---|---|---|
| P(market), fit | 0.0254 | 0.0259 |
| P(market \| previous was a market order) | 0.284 | 0.283 |

The two seeds now agree (8/12 and 8/12). Select NLL improved 6.03 → 5.79.

## 2. Model levers (each judged on ≥ 2 seeds after fix 3)

| model | select B2 (seed 0 / seed 1) | still failing |
|---|---|---|
| v3.8 (shuffled, 10 passes) | 8 / 8 | depth and band (borderline), volatility at 1000 events, ACF(1) |
| + FlowBank → market-side skip, 15 passes | 9 / 9 | volatility at 100 and 1000 events, ACF(1) |
| + RL 40 iterations (rollout realism, NLL anchor) | 11 / 10 | ACF(1); volatility at 1000 events (seed 1) |
| + market-side temperature (chosen on FIT, n = 128) | **11 (T 0.9)** / 10 (T 1.0) | depth 0.3873 vs threshold 0.385 (seed 0); volatility at 1000 events and ACF(1) (seed 1) |

Dead ends:
- RL from un-calibrated bases;
- RL with a heavy ACF weight;
- the big-batch RL;
- a 30-pass long run with the unshuffled bug (best NLL, B2 1/12: likelihood and closed-loop realism decoupled);
- the level-0 mask (no gain once seeds are controlled).

## 3. What is left

Sign persistence is now learned under teacher forcing: implied ACF(1) 0.40–0.42 vs real 0.46. But it decays in closed loop to about 0.28. The select hour is more persistent than the fit hours (real ACF(1) 0.42 vs 0.33), so a FIT-chosen sampling temperature cannot fully reach the select regime.

The principled remedy is more diverse training hours:
- split_v2 is registered with entirely future select/report hours;
- v1 report hours are excluded from its fit set.

## 4. Status

B2 is **not yet passed**:
- best select result 11/12 (one metric 0.002 over its threshold);
- the second seed reaches 10/12.

No report-hour claim is made for these models.
