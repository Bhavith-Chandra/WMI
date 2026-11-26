# M3 reproduction & audit (arXiv 2608.19227)

Independent re-implementation of the M3 inference stack plus an audit of what the public release can and
cannot support. Every number below comes from `results/*.json`, produced by the scripts in this folder.
Blocks marked *generated* are written by `scripts/build_readme_numbers.py` directly from those files.
`tests/test_claims.py` fails if any README number drifts from the results.

## Bottom line

* **Full reproduction of the paper's tables is impossible from the release.** The training/evaluation data
  (CSI 300/500 order-level data, 31.8B events) is not released. Neither are the 366M/1.27B models, the
  training loop, the matching engine, the baselines or the evaluation code. What *is* released: 3 small
  checkpoints (10M/25M/75M), the tokenizer, inference code, and **one** 512-token held-out example window.
* What we could check: architecture, parameter counts, the scaling law against the released checkpoints,
  likelihood on the example, tokenizer behaviour, rollout behaviour, and a closed-loop TWAP impact test
  on that single example.

## Reproduction method

* `m3/model.py`: tokenizer, LOB-prefix encoder and LLaMA AR model, written from the paper and the
  checkpoint layout.
  * All 4 checkpoints load with `strict=True`. That proves parameter names and shapes match, not the
    forward semantics.
  * Forward semantics were checked by reading the released code. An independent reviewer found no mismatch:
    `lob_volume_clip` is None in all configs, the feature-clip indices match, and decode uses `time_mode`
    "reconstruction".
  * Checkpoints are loaded with `weights_only=True`. None of the released Python is executed.
* `m3/engine.py`: one matching implementation (`apply_event`), written to the paper's appendix spec:
  aggregate depth, price priority, no FIFO; unmatched cancels are recorded and not applied.
* `m3/closedloop.py`: closed-loop simulator. Batch rows are independent paths: fixed masked
  injection slots, an explicit position id per row, a roll schedule independent of treatment, and per-row
  inverse-CDF sampling. Child orders are immediate-or-cancel (IOC), and each is encoded into the model's
  context as add + cancel.
* `scripts/run_audit.py`, `scripts/run_twap.py`, `scripts/twap_bootstrap.py`, `scripts/regenerate_all.sh`.

## Harness (data-leakage / lookahead / provenance), `pytest tests/`

| test file | guarantees |
|---|---|
| `test_causality.py` | AR model, tokenizer encoder and tokenizer decoder are causal: a future perturbation leaves all earlier outputs unchanged, and is checked to change later ones (non-vacuous). Cached sampling equals a full recompute. Seeded runs are deterministic. |
| `test_closedloop.py` | **A no-trade path is bit-identical whether or not another path trades.** Cache state after masked injections and rolls equals a fresh forward pass. Children execute after their schedule and strictly before the horizon. IOC leaves no resting residual. One path's random stream does not depend on other paths. |
| `test_engine.py` | The book is never crossed. Quantity is conserved. Price priority holds. q ≤ 0 is a no-op. Behaviour with an empty opposite side is checked. One matching implementation only. |
| `test_leakage.py` | The release contains no realised future data, and the conditioning snapshot is consistent with the prompt start. An AST scan allows only permitted data paths and requires `weights_only=True` on every `torch.load`. The checkpoints' test-split selection is detected and disclosed. |
| `test_claims.py` | README numbers equal the result files: exact, signed, in order. |
| `test_provenance.py` | Every result file records the git commit, argv, library versions and the SHA-256 of every input, and those hashes match the files on disk. |

An adversarial reviewer agent audited v1 of the closed loop and found that paths in a batch were coupled.
**All v1 TWAP numbers were retracted.** Every TWAP number below comes from v2, which is covered by the tests above.

## Findings (n = 1 held-out window unless stated; treat window-level results as case studies)

### 1. Parameter counts match the paper (✓)
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
* Every checkpoint's `train_args` has `val_split: "test"`, both the tokenizer and the AR models.
* `best.pt` is therefore chosen by loss on the test split (Dec 2025), the same period the paper calls "strictly held-out".
* The released `base` checkpoint is from step 19,000 of 40,000, picked as best on that split.
* The scaling-law "validation loss" is therefore test loss. The paper does not disclose this.
* The single real evaluation window we have (example, split `test`, 2025-12-01) also comes from this
  **selection split**, so every likelihood number on it is in-sample with respect to checkpoint selection.

### 4. Likelihood on the example is consistent with the reported val loss (✓, in-sample, see #3)
`results/nll.json`. Mean NLL over 512 tokens is 7.22 / 6.88 / 6.73 (tiny / small / base). That is above the checkpoint val losses only because of the cold start.
NLL over the last 128 positions is 6.33 / 6.08 / 5.91, close to the reported val losses. The ordering tiny > small > base holds.

### 5. The LOB prefix carries weak, short-lived information (⚠)
Effect of a counterfactual book on prompt NLL, mean Δ in nats over 512 tokens (first 64 tokens in brackets):

| perturbation | tiny | small | base |
|---|---|---|---|
| flip bid/ask depth (imbalance) | +0.003 (−0.016) | +0.005 (+0.014) | +0.004 (+0.006) |
| volume ×10 | +0.015 (+0.082) | +0.040 (+0.174) | +0.067 (+0.229) |
| prices +1% | +0.011 (+0.069) | +0.016 (+0.110) | +0.025 (+0.137) |
| prices −1% | +0.029 (+0.179) | +0.043 (+0.196) | +0.038 (+0.186) |

* Flipping order-book imbalance barely changes the model's likelihood.
* The prefix matters mainly in the first ~64 tokens, then the order context takes over.

### 6. The tokenizer is strongly context-dependent; the release decodes without context (✗)
`results/tokenizer.json`, `results/decode_ablation.json`.
* Decoding the same 256 tokens without the 256 tokens before them changes the decoded price by a mean of
  **11.6 ticks**; only 0.4% land on the same tick of the in-context decode. Action and side never change.
* Re-encoding decoded features returns the same code only 47.7% of the time.
* 14.8% of decoded inter-arrival times are negative.
* The released rollout default (`include_prompt=False`) decodes generated ids **alone**, so their
  positions restart at 0. On identical generated tokens (first 512 generated events, 32 rollouts per model, *generated*):

<!-- AUTO:decode -->
<!-- /AUTO:decode -->

### 7–9. Rollouts: past the trained context, horizons, snapshot replay (*generated*)
* The released sampler has **no rolling window**; the paper says one exists. The example run (512 prompt + 1024 generated) reaches
  position 1545, while training used 9 + 1024. Entropy is measured over the 32 rollouts.
* The paper's 300 s TWAP and 5-min prediction experiments need more events than one context window covers, so they depend on that missing rolling context.
* Snapshot replay cannot honour cancels of orders resting beyond level 10, so phantom liquidity is built into the paper's own replay protocol.

<!-- AUTO:rollout -->
<!-- /AUTO:rollout -->

### 10. Closed-loop TWAP impact (single prompt, v2 simulator, *generated*)
Setup:
* single released prompt; 60 s horizon
* 10 IOC children at (k − ½)·6 s, each at the touch ± 10 ticks
* rows are independent paths; a no-trade baseline uses the same prompt
* exponent from NLS `I = A·(Q_exec/10⁴)^β` over **all** reached paths, so non-positive impacts are kept
* 90% CI from a path-level bootstrap that resamples baseline paths and paths within each cell

Lighter than the paper's protocol (300 s, 64 prompts × 4 rollouts); that wasn't feasible on CPU. Daily volume and volatility are not
released, so Q is in shares and only β is comparable to the paper's (0.532 / 0.524).

Stalled paths are paths whose simulated clock is still short of 60 s after 4000 generated events. Excluding
them is selection on a post-treatment outcome. The table reports the counts and a Fisher test of stalling
versus side, and a sensitivity analysis refits with stalled paths included.

<!-- AUTO:twap -->
<!-- /AUTO:twap -->

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
scripts/regenerate_all.sh
python3 -I scripts/build_readme_numbers.py
python3 -m pytest -q tests
```
`scripts/regenerate_all.sh` regenerates every result and checks it against the committed version
(`scripts/compare_rerun.py`). Runs on CPU only (4 cores); seeds are fixed.
