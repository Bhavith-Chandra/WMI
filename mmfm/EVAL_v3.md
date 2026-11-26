# EVAL_v3 — pre-registered before any v3 training (committed before the v3 tokenizer is fit)

## Why v3
The v2 forward test (README, EVAL_v2) met criteria 5 and 7 and failed criteria 1–4 and 6. The measured
failure is a regime shift: the forward window was 7–28% quieter than the 5 h of design data, and the count heads
over-predicted by 1.1–1.6×. v3 attacks it three ways, all decided now:
1. **More data / regimes**: every hour recorded up to the v3 cut (02:27–08:00 and 14:51–22:56 UTC, 2026-11-26).
   This includes the burned v2 val and v2 forward windows; they are never used as a test again.
2. **Online count recalibration** (causal, below) on the simulator's anchored clock and on the direct count heads.
3. **Rollout-aware fine-tuning**: two-pass scheduled sampling (Mihaylova & Martins 2019) on top of MMFM-v3.

## Data and splits (frozen in `<v3 dir>/splits.json` before training)
- Build: `data/build.py --chunk-s 3600`. Long recorder segments are re-anchored on a REST order-id snapshot about
  every 60 min, which bounds memory. The v2-1 verification and exclusion rule is unchanged.
- Cut U = 22:56:07 UTC (`/home/user/data/mmfm_v3_until_ns`).
- Split windows:
  - **val** = [U − 60 min, U];
  - **embargo** = 10 min;
  - **train** = everything ≤ U − 70 min.
- **Held-out pair:** linkusd, which is never in train.
- **Forward test v3** = [freeze + 10 min, + 90 min], built only after the window closes, and unsealed once per
  metric with the frozen-bundle hash (as in v2).

## Models (same CPU, each trained ALONE, fixed wall-clock budgets)
- MMFM-v3: same architecture and optimiser as v2 (d=128, 4 layers, ctx 64), from scratch on the v3 tokenizer, 90 min.
- B2-v3: same as B2-v2, 60 min, same tokenizer.
- MMFM-v3-ss: initialised from MMFM-v3.
  - Each step does a no-grad pass and samples a token at every position from the model.
  - Each input token position t ≥ 1 is replaced with probability p by the token sampled from position t−1.
    Book features stay real, and targets stay real.
  - p ramps linearly 0 → 0.25 over the first half of the run, then stays at 0.25.
  - 40 min budget, LR 3e-4 cosine; forecast heads are trained as usual.
  - Val selection and temperature calibration are as in v2.

## Online count recalibration (causal; same code for eval and live UI)
- **Origins.** For a forecast at time t0, the origins are o_k = t0 − 10 s − k·30 s, k = 0..9.
- **Valid origins.** Only origins with ≥ 1100 events of history before them are used.
- **Inputs at each origin.**
  - m_h(o_k): the model's calibrated count-head median, from context ending at o_k.
  - N_h(o_k): the realised count in (o_k, o_k + h]. All of these events are ≤ t0.
- **Bias.** b_h = clip(median_k log((N_h + 1)/(m_h + 1)), −1, 1). It is 0 if there are fewer than 3 valid origins.
- **Anchored clock.** Each path's comonotone draw becomes N'_h = floor((N_h + 1)·e^{b_h} − 1), then cummax over h.
- **Direct count heads.** The same multiplicative shift is applied.
- **Harness.** A test proves that b_h depends on no event after t0.

## Selection (val only)
- **Candidates:** {MMFM-v3, MMFM-v3-ss} × temperature {0.6, 0.7, 0.85}, anchored clock + recalibration.
- **Rule:** pick the candidate minimising the calibrate_sim loss (unchanged from v2) on val, 30 starts per pair.
- **Constraint:** the chosen model's val next-event NLL must beat B2-v3; otherwise take the best candidate that does.
- **Reporting:** both models' val NLL and the free-clock event ratio (diagnostic) are reported.

## Criteria (forward test; identical to EVAL_v2 so v2 → v3 is comparable)
1. **Event ratio:** MMFM-sim event ratio in [0.8, 1.25].
2. **Coverage:** coverage80 in [0.70, 0.90].
3. **PIT:** PIT mean in [0.40, 0.60].

Criteria 1–3 are per cell, all 9 cells (ethusd, xrpusd, dogeusd × 1, 5, 10 s).

4. **CRPS:** MMFM-sim CRPS ≤ best count baseline in a majority of cells.
5. **NLL:** MMFM next-event NLL beats B2-v3, with the 95% block-bootstrap CI excluding 0.
6. **Zero-shot:** linkusd NLL gap vs B2-v3 ≤ 0.
7. **Real time:** 8 paths on ethusd ≥ 1× the median event rate.

Also reported (not criteria): the same table **without** recalibration, the v2 frozen bundle on the v3 forward window,
and MMFM-direct with and without recalibration.
