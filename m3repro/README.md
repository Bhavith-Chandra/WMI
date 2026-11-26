# M3 reproduction & audit (arXiv 2608.19227)

Independent re-implementation of the M3 inference stack plus an audit of what the public release can and
cannot support. Every number below is produced by the scripts in this folder; raw outputs are in `results/`.

## Bottom line

* **Full reproduction of the paper's tables is impossible from the release.** The training/evaluation data
  (CSI 300/500 order-level data, 31.8B events) is not released. Neither are the 366M/1.27B models, the
  training loop, the matching engine, the baselines or the evaluation code. What *is* released: 3 small
  checkpoints (10M/25M/75M), the tokenizer, inference code, and **one** 512-token held-out example window.
* What we could check: architecture, parameter counts, the scaling law against the released checkpoints,
  likelihood on the held-out example, tokenizer behaviour, rollout behaviour, and a closed-loop TWAP
  impact test on that single example.

## Reproduction method

* `m3/model.py`: tokenizer, LOB-prefix encoder and LLaMA AR model, written from the paper and the
  checkpoint layout. **All 4 checkpoints load with `strict=True`**, so the architecture is identical.
  Checkpoints are loaded with `weights_only=True`. None of the released Python is executed.
* `m3/engine.py`: the matching engine, implemented to the paper's appendix spec (aggregate depth,
  price priority, no FIFO; unmatched cancels are recorded but not applied). It also tracks extra
  diagnostics.
* `m3/closedloop.py`: closed-loop simulation with a rolling AR context, in-context decoding and
  encoder-based injection of child orders. The paper describes all three; the release implements none of them.
* `scripts/run_audit.py {scaling,nll,tokenizer,rollout,decode_ablation}` and `scripts/run_twap.py`.

## Findings (n = 1 held-out window unless stated; treat window-level results as case studies)

### 1. Architecture: matches the paper (✓)
Parameter counts are 9,659,776 / 24,989,952 / 74,754,560, which matches the paper's 9.66M / 24.99M / 74.75M exactly.

### 2. Scaling law (Eq. 2) does not predict the released checkpoints (✗)
`results/scaling.json`. D = step × batch × grad_accum × 1024, taken from each checkpoint's `train_args`.

| model | N (M) | D (B tok) | ckpt val loss | Eq. 2 prediction | error |
|---|---|---|---|---|---|
| tiny  | 9.66  | 31.46 | 6.228 | 6.388 | +2.6% |
| small | 24.99 | 31.46 | 6.005 | 6.183 | +3.0% |
| base  | 74.75 | 14.94 | 5.886 | 6.112 | +3.8% |

The paper claims a leave-one-out error of 0.66%. The released runs come out 0.16–0.23 nats *better* than Eq. 2 predicts.
A likely explanation, which can't be verified, is that the scaling grid was fit on different runs (for
example without the LOB prefix; Fig. 10 shows a gap of about 0.1–0.2 nats). The paper does not say which.

### 3. Test data used for model selection (✗, leakage)
Every checkpoint's `train_args` has `val_split: "test"`, both the tokenizer and the AR models.
`best.pt` is chosen by loss on the test split (Dec 2025), the same period the paper calls "strictly
held-out". The released `base` checkpoint is from step 19,000 of 40,000, picked as best on that split.
The scaling-law "validation loss" is therefore test loss. The paper does not disclose this.

### 4. Likelihood on the held-out example matches the reported val loss (✓)
`results/nll.json`. Mean NLL over 512 tokens is 7.22 / 6.88 / 6.73 (tiny / small / base). That is above the
checkpoint val losses only because of the cold start: NLL over the last 128 positions is
6.33 / 6.08 / 5.91, close to 6.23 / 6.01 / 5.89. The ordering tiny > small > base holds.

### 5. The LOB prefix carries weak, short-lived information (⚠)
Effect of a counterfactual book on prompt NLL, mean Δ in nats over 512 tokens (first 64 tokens in brackets):

| perturbation | tiny | small | base |
|---|---|---|---|
| flip bid/ask depth (imbalance) | +0.003 (−0.016) | +0.005 (+0.014) | +0.004 (+0.006) |
| volume ×10 | +0.015 (+0.08) | +0.040 (+0.17) | +0.067 (+0.23) |
| prices +1% | +0.012 (+0.07) | +0.016 (+0.11) | +0.025 (+0.14) |

* Flipping order-book imbalance barely changes the model's likelihood.
* The prefix matters mainly in the first ~64 tokens, then the order context takes over.

This is consistent with finding #3 of my critique: the book is encoded once and goes stale.

### 6. The tokenizer is strongly context-dependent; the release decodes without context (✗)
`results/tokenizer.json`, `results/decode_ablation.json`.
* Decoding the same 256 tokens without the 256 tokens before them changes the decoded price by a mean of
  **11.6 ticks**; only 0.4% land on the same tick. Action and side never change.
* Re-encoding decoded features returns the same code only 47.7% of the time.
* 14.8% of decoded inter-arrival times are negative.
* The released rollout default (`include_prompt=False`) decodes generated ids **alone**, so their
  positions restart at 0. On identical generated tokens (first 512 generated events, 32 rollouts per model):

| | release decode (gen only) | in-context decode |
|---|---|---|
| mean decoded rel-open price, base | −0.00152 | −0.00026 |
| cancels hitting an empty level, base | 61.7% | 26.0% |
| median 512-event return, tiny / small / base (bp) | −24.6 / −25.4 / −49.2 | −14.2 / −16.4 / −30.6 |

The release default biases decoded prices about 0.1% downward. It accounts for most phantom cancels and roughly
40% of the downward drift.

### 7. Released rollout recipe runs past the trained context; generation degrades (✗)
The released sampler has **no rolling window**. The paper says one exists, the code does not. The example run
(512 prompt + 1024 generated) reaches position 1545, while training used 9 + 1024. Mean next-token entropy for base,
by absolute position: 5.7–6.5 nats at 513–1024, then 6.9 → **8.1** nats at 1025–1536 (uniform = 10.4).
Tiny: 4.5 → 6.7, small: 6.0 → 6.8 (inside vs beyond the trained range).

### 8. Horizons in the paper exceed one context window (⚠)
1024 generated events cover a median of 37 s (tiny), 91 s (small) and 149 s (base) of simulated time.
The 300 s TWAP and 5-min prediction experiments therefore need 2–8k events, so they depend on the rolling
context that the release does not include.

### 9. Snapshot-initialised replay cannot honour deep cancels (⚠, affects the paper's own protocol)
Replaying the *real* decoded prompt from its 10-level snapshot: 68% of cancel quantity finds no resting
depth, and 42.5% of cancels target prices never seen in the book (orders resting beyond level 10 before the
window). The paper's matching engine silently drops these, so phantom liquidity is built into every replay.

### 10. Closed-loop TWAP impact (single prompt): square-root law **not** reproduced (✗ sell, ? buy)
`results/twap_{tiny,small}_h60.json`. Setup:
* single released prompt
* 60 s horizon, 10 children at 6 s intervals, each an aggressive limit order 10 ticks through the touch
* Q ∈ {10k, 40k, 160k} shares, 6 rollouts per cell plus a no-trade baseline, 42 paths in total
* rolling context and in-context encoding of the injected orders

This is lighter than the paper's protocol (300 s, 64 prompts × 4 rollouts), which wasn't feasible on CPU.
Daily volume and daily volatility for this stock-day are not released, so Q is in shares. Only the exponent is comparable to the paper's.

| | tiny | small |
|---|---|---|
| baseline drift over 60 s, mean (sd) | **−105 bp** (46), n=4 | −33 bp (37), n=6 |
| stalled paths (clock < 60 s after 4000 events) | **7/42**, 2 of them baseline | 1/42 |
| buy impact median, 10k/40k/160k (bp) | 101 / 98 / 164 | 35 / 70 / 153 |
| sell impact median, 10k/40k/160k (bp) | 93 / −18 / −62 | 123 / 99 / 66 |
| buy β (log-log, 3 bins) [90% CI*] | 0.19 [0.02, 0.38] | **0.57** [0.32, 1.06] |
| sell β | not fittable | −0.24 [−0.80, 0.45] |

\*Approximate CI: cell medians resampled as N(median, SE); draws with non-positive impact dropped.

* **Small, buy side**: β = 0.57 is close to the paper's 0.532. The CI is wide and covers both 0.5 and linear impact (1.0), so it does not confirm a square root.
* **Sell side**: no concave increasing impact for either model. For small, impact *falls* with size.
* **Tiny**: the baseline drifts about −1%/min, against −9.5 bp over 34 s in the real prompt replay. That drift swamps the impact signal.
* **New failure mode, stalling**: in some paths the simulated clock freezes. 47–84% of their events decode to inter-arrival times ≤ 1 ms, against a median of 22–24% in healthy paths.
  It also hits no-trade baseline paths, so the cause is generation under a rolling context, not the intervention.

## Not reproducible from the release

* Table 1: KS / Wasserstein / kurtosis vs ZI and Hawkes.
* Fig. 6: tercile and directional accuracy vs DeepLOB.
* Fig. 7: volatility quintiles.
* The 64-prompt TWAP study.
* Table 2: tokenizer vs bin tokenizers.
* Scaling-grid losses.

All of these need the private test set, or code and models that were not released. Public order-level
alternatives were checked: LOBSTER samples now require proof of purchase, and the HF "lob_bench" set has only
generated messages. No public CSI 300/500 order-level data exists.

## Run
```bash
bash setup.sh
python3 -I scripts/run_audit.py scaling
python3 -I scripts/run_audit.py nll
python3 -I scripts/run_audit.py tokenizer
python3 -I scripts/run_audit.py rollout --rollouts 32 --new 1024
python3 -I scripts/run_audit.py decode_ablation
python3 -I scripts/run_twap.py --model tiny  --horizon 60 --children 10 --reps 6 --qgrid 10000,40000,160000 --max-steps 4000
python3 -I scripts/run_twap.py --model small --horizon 60 --children 10 --reps 6 --qgrid 10000,40000,160000 --max-steps 4000
```
Runs on CPU only (4 cores). Seeds are fixed in the scripts.
