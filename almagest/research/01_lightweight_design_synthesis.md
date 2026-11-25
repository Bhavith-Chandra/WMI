# Almagest LMM: lead-scientist cross-check and lightweight design recommendation

**Confidence tags:** **[C]** = read in the source (Certain). **[L]** = strong inference (Likely). **[G]** = Guessing. **[E]** = my own estimate; the arithmetic is shown.

## 0. Spot-checks (5 load-bearing claims, read in full text via alphaXiv)

| # | Claim checked | Verdict | Corrections and new facts |
|---|---|---|---|
| 1 | LOB-Impact numbers (2610.27100 Table 3, Sec 2-4) | **Confirmed** | **(a)** No model passes C4, not even the positive-control Propagator (2/3 + 1/3·ℓ^-1/2 kernel). The replay kernels read δ = 0.73-0.84 at this operating point, and their closed-form value is about 0.9-1.0 under this schedule, because the permanent 2/3 component makes the build-up near-linear [C]. **(b)** Real metaorders relax to about 2/3 of peak within 1-2 execution durations and to about 1/2 within 10 durations; the window spans the latter [C]. **(c)** The C5 bar is the OW kernel's end/peak on the same stock and schedule (AMD 0.81), not 2/3. The closed-form values at this spacing are Propagator 0.94 and OW 0.87 [C]. **(d)** A direct fit of the neural decay kernel gives γ ≈ 0 (permanent impact) [C]. **(e)** QR (90 params) is the only model to pass C5, on EA only (e/p 0.42, slope -1.0), and it fails C4 there (δ 0.74, Y 0.32) [C]. **(f)** The authors say 4k context leaves the memory attribution "untested rather than refuted" [C]. **(g)** Authors' guidance: "learn replenishment rather than unlearn persistence" [C]. |
| 2 | M3 (2608.19227) | **Confirmed** | Slopes 0.532 / 0.524; Y = 1.509 / 1.493. The paper does not say which model size ran the TWAP test. The tiny model is 9.66M (hidden 128, 3 layers). The engine is aggregate-depth with no order IDs and no FIFO. Operating point: participation 0.1-1%, 20 children over 300 s, no relaxation window, no blind control [C]. |
| 3 | FRAC (2609.36314) | **Confirmed** | M = 16 modes, τ from 1 to 2^17, approximation error 6.82e-4 (vs 1.34e-3 at M=8 and 6.45e-4 at M=32). State is 65,536 scalars per layer vs 524,288 for Mamba-2/3. The 7M DNA model has 8 layers, d = 256, expansion 1.5. Fixing α = 1 gives perplexity 61.3 vs 47.2; learned timescales give 54.1. Clips: Δ in [1e-4, 1], λ in [0.25, 4] [C]. It has never been tested on LOB data or generation [C, from the absence of any such experiment]. |
| 4 | Impact-aware QR (2603.24137) | **Confirmed** | G = (1+t/τ)^-1.5 with τ = 50 s; m = 0.036 (0.035 in the appendix). K = 12 exponentials, half-lives 0.01-1000 s, NNLS weights 0.398 at 15.2 s and 0.394 at 43.3 s. m was calibrated by binary search to a *theoretical* √-shape target, not to an empirical Y. 500k metaorders. Code is said to be on GitHub; the licence was not checked [C]. |
| 5 | 8.2M Mamba-3 interpretability (2610.28753) | **Confirmed** | 13 layers, d = 256, 26 tokens/msg. Long-memory heads have half-lives of 28-66 messages; in the random twin no head exceeds 10. The newest message carries 58% of the state, the last 50 carry 77%. Forward R² = 0.04 at H = 25. The model never receives the book as input [C]. |

## 1. Cross-check: conflicts, overstatements, downgrades

1. **Wrong-sign analogy, downgraded to [G] and then contradicted.** Track 2 and Track 4 said the neural models "behave like the no-HFT ablation of 2607.04280."
   - In that ABM, removing replenishment *lowers* δ to 0.386, below 0.5.
   - The neural models sit *above* 0.5 (0.63-1.21).
   - LOB-Impact's own mechanism points the other way: without decay between children, N pushes add up to N instead of √N, so δ moves toward 1 [C].
   - The neural failure looks like "persistent push with a permanent kernel" (Propagator-like, δ about 0.9-1.0), not like the ABM's no-HFT book walk.
   - Consequence: use the 2607.04280 ablations only as synthetic-generator sanity checks. Do not use them as a diagnosis of the neural models.
2. **"Longer context does not help" is confounded.** Track 6 claims S5-4k (0.79) vs S5-120M (0.66) undercuts the memory hypothesis. But S5-4k is 55M vs 120M, and Mamba3-4k is a 2k fine-tune run at 4k [C].
   - Track 1's "55M S5-4k is about as good as 120M" is also overstated: δ is worse on AMD (0.79 vs 0.66), EA (1.05 vs 0.84) and MSFT (0.69 vs 0.63), and similar on NVDA and GOOG [C].
   - Downgrade both to "inconclusive".
3. **The C4 band may be unreachable at this protocol's operating point.** Every baseline with known physics fails C4 [C]. The estimator is biased downward by up to 0.11 at neural amplitudes [C].
   - A design that "has √-law physics" is not guaranteed a C4 pass.
   - This is the main open risk, and Experiment 1 targets it first.
4. **Y targets use different conventions.**
   - LOB-Impact: Y ≈ 0.5, Parkinson range / κ_P = 1.665 [C].
   - Donier-Bonart (MtGox): Y ≈ 0.9 ± 0.35 [Track 4, not re-read; L].
   - 2610.10053: Y ≈ 0.32, end-of-day decayed [Track 2; L].
   - These are not interchangeable. Re-express each in the LOB-Impact convention before any reward or gate uses it.
5. **Not re-verified by me; keep each claim at its track's tag and do not upgrade:**
   - 2607.04280 (ABM; code on request), 2610.10053, 2609.37872 (Itkin), 2610.08816 (log² mode theorem), 2611.14117, 2609.16930, 2501.08822 (MDQR relaxation of about 20%), 2610.20104 (55 GPU-h, K = 256).
   - Tardis prices: Track 4 already flagged the $49 tier as unverified.
   - 2512.23824 "MS-SSM, COLM 2025": an arXiv id from Dec 2025 sits oddly with a COLM 2025 venue. Verify the id before citing it.
6. **An "ABM gives δ = 0.539" result does not give relaxation.** Relaxation was not measured in 2607.04280 [Track 3; C]. Do not count it as a relaxation-capable generator family.
7. **Blind-control honesty with explicit slow features.** Tracks 2 and 6 recommend propagator or LLOB features. The blind arm hides injected orders from the model's inputs [C], so any feature bank computed from engine state that includes the injected fills would leak them and inflate v/b. This is a hard design constraint (see §3.8).

## 2. Completeness gaps (still missing)

- **Positive-control feasibility.** Nobody has shown any generator, neural or parametric, passing C4 under the LOB-Impact schedule (η = 10%, N = 100, aggressive children). This is unknown and decisive.
- **Unread papers that are directly relevant:**
  - ReLOBGen 2609.35867: pointer cancels; impact-tested or not?
  - Causal PFN 2610.12537: shared-noise counterfactuals and level-calibration k.
  - EGGROLL post-training 2610.25604: cost and population numbers.
  - Maitrier-Bouchaud age-clock propagator.
  - DSSM 2607.10244 and CMM 2610.07907.
- **Data access.** LOBSTER price and access for the 5 benchmark names plus the training months are unknown. Without them there is no head-to-head comparison against the 55-120M panel.
- **Modern crypto metaorder Y, δ and relaxation.** No 2020-2026 measurement was found.
- **Operating-point robustness.** LOB-Impact uses one participation rate, one child size and aggressive children only [C, its Limitations section]. A design tuned to that point may not generalise.
- **Hidden/iceberg executions, auctions, open/close.** LOBSTER handling of these is not addressed in any track.
- **Throughput of a factored per-event decoder plus pointer head under a vmapped JAX engine.** Unmeasured.
- **Cheap-proxy validity.** No evidence yet that a short or impulse campaign rank-correlates with the full C4/C5 verdicts.

## 3. Lightweight Almagest design recommendation

### 3.1 Backbone and parameter budget: target about 10M, hard cap 20M

- **Recurrent backbone, one step per event.**
  - 10 blocks, d = 256. Each block is a FRAC-style mode-bank mixer (M = 16, τ log-spaced from 1 to 2^17 *events*, learnable α_t and λ_t, softmax read/write with the −α·log τ prior), a depthwise causal conv of 4, and a SwiGLU MLP.
  - Basis: FRAC's 7M / 8-layer / d=256 setting at 64K [C]; the 8.2M Mamba-3 at d=256 is LOB-Bench-competitive [C].
- **Parameter arithmetic [E]:**
  - Mixer projections: in-proj about 256 × 900 ≈ 0.23M; out-proj 384 × 256 ≈ 0.10M.
  - SwiGLU MLP at hidden 683: 3 × 256 × 683 ≈ 0.52M.
  - Per block about 0.85M, so 10 blocks ≈ 8.5M.
  - Embeddings plus book-feature projection ≈ 0.1M; five factored heads ≈ 0.5M.
  - **Total ≈ 9-10M.**
- **Physical-time decay.** Feed Δ_t = f(log inter-arrival) into the mode bank so it relaxes during silence [L; mechanism from S5 variable steps (2309.00638) and FRAC's ZOH discretisation].
- **Freeze the slow tail.** Make 4 of the 16 modes, those with τ above 10^3 events, input-independent (non-selective retention).
  - Rationale: the less-selective S5 family has the lowest δ and the lowest |drift| on all 5 stocks [C]. That selectivity is the cause is [L].
- **Do not initialise retention Griffin-style ([0.9, 0.999]).** The random twin reaches only ≤ 10 messages of memory [C].
- **Always feed the true L10 book state from the engine as input.**
  - LOBS5 reports a sharp loss drop when the book is added [C, Track 1].
  - Implicit state tracking fails (2608.23706) [Track 6; C].

### 3.2 Impact and relaxation mechanism: hybrid, cheapest physics first

- **Level A (default, about 0 extra params).** An explicit slow-flow bank of K = 12 leaky integrators of ε·√V (signed square-root trade volume), with the half-lives of 2603.24137 Table 1 (0.01-1000 s, wall-clock).
  - Computed from the **model-visible stream only**.
  - Read by the network as input features. Also optionally enters as a bounded logit tilt on aggressive-order side probabilities, exp(m·[φ]+), as in 2603.24137 [C mechanism; transfer to a neural generator is L].
  - This is the planned two-clock memory's slow clock at about zero cost.
- **Level B (only if Experiment 1 shows Level A cannot get the build-up concave enough).** A multi-timescale LLOB latent-liquidity core: K = 3-4 liquidity classes on a 128-bin grid, fewer than 100 scalars.
  - The visible mid is soft-tethered to the latent price.
  - Theory: √Q with Y independent of φ for J_s ≪ m0 ≪ J, and relaxation starting as 1 − √((t−T)/T) [Track 2: 1412.0141, 1710.03734; C per track, not re-read].
  - Reason to keep it in reserve: a plain propagator with a permanent component reads δ ≈ 0.76-0.9 in LOB-Impact [C]. A concave build-up may need the latent-book mechanism, not just a decaying kernel [L].
- **Constraint.** Never pair exponential (OW-type) resilience with √-impact. Gatheral's condition is γ + δ ≥ 1 [Track 2; L, secondary source].

### 3.3 Tokenisation

- **One embedding per event.** Sum the field embeddings: type, side, price offset from touch, log-size bucket, log-Δt bucket, plus a book-feature projection.
  - Equities: price offset in ticks.
  - Crypto: price offset in bps or a dynamic tick (small-tick risk [Track 4; L]).
- **Factored autoregressive heads within the event:** type → side → price → size → Δt → cancel pointer (pointer over resting orders at that level, as in ReLOBGen; unread, [G] on quality).
- **Horizon arithmetic.** At 22-26 tokens per message, a 40k-message metaorder is 0.88-1.04M token steps; one step per event makes it 40k steps, inside τ_max = 131,072 [E: 40,000 × 22 = 880,000; 40,000 × 26 = 1,040,000].

### 3.4 Synthetic generator mix (Stage 1, the "almagest-sim" JAX engine)

- **Engine.** JaxMARL-HFT fork (Apache-2.0 [Track 3; C]) with OrderID and TraderID per order (2308.13289).
- **Families sampled per regime, targeting K ≥ 256 regimes (2610.20104 [Track 3; C, not re-read]):**
  - (a) Poisson / Hawkes / nonstationary heavy-tailed background flow.
  - (b) Institutional splitters plus HFT replenishers, with p ~ U(0.6, 0.95) (2607.04280).
  - (c) Impact-aware QR tilt with τ ∈ [10, 300] s and m ∈ [0.01, 0.1].
  - (d) Spoof/layer agents using the 2508.17086 Table 4 parameters.
  - (e) Optional LLOB latent-liquidity agents.
- **Shared-noise counterfactual pairs** (with vs without the planted metaorder; replenishment on vs off), following the Causal PFN shared-noise idea.
- **Keep only regimes that pass** δ in 0.5 ± 0.1, Y in [0.5, 1], e/p below OW, |drift| ≤ 5 ticks, plus a labelled broken-physics minority. Episodes of at least 100k messages.

### 3.5 Data plan

- **Stage 1.** About 2B synthetic events [E; 2610.20104 used 1.6B tokens per family at 4 tokens/msg, i.e. about 0.4B messages per family].
- **Stage 2.**
  - LOBSTER for the 8-name corpus, Jan 2022 to Dec 2025, **never January 2026** (that is LOB-Impact's evaluation month [C]).
  - Self-recorded Coinbase `level3` (BTC, ETH and 2-4 mid-liquidity pairs) plus Tardis free first-of-month days. Thin day sampling: 100-200 days [Track 4; L].
  - About 2B real events in total [E].
- **Post-training split.** Use the validation month only. Test month and held-out stock are touched once per release.

### 3.6 Training recipe and GPU-hours

- **Anchor.** 14.2M model at 131k tokens/s on a single GPU (2610.20104) [Track 3; C per track].
- **Pretraining plus fine-tune at about 4e9 events:** 4e9 / 1.31e5 ≈ 30,500 s ≈ **8.5 GPU-h** [E].
- **FLOP cross-check:** 6 × 1e7 × 4e9 = 2.4e17 FLOPs. At an assumed 10% MFU on H100 (≈ 989 TFLOP/s dense BF16 peak, vendor spec [L]) that is ≈ 0.7 h; at 1% MFU ≈ 6.7 h [E].
- **Ablation ladder:** 4 arms × 3 seeds × about 8.5 h ≈ **100 GPU-h** [E].
- **For comparison, LOBS5-35M:** 30.4 L40-days ≈ 730 L40-hours (2502.09172) [Track 1; C per track].
- **Evaluation campaign per stock:**
  - About 1024 rollouts × 2 directions × 4 arms; on AMD, 40k sequential steps per arm [C for the protocol].
  - Batched 1024-wide at about 3 ms per step (vmapped engine about 2.6 ms at capacity 100 for 1000 books, 2308.13289 [Track 3]; model step ≈ 0.5 ms [G]): 8 × 40,000 × 3 ms ≈ 960 s ≈ 16 min.
  - GOOG (73k) ≈ 30 min. **About 1.5-2 GPU-h for all 5 stocks [E].**
  - Compare LOB-Impact's 39-54 s per rollout for 78-120M models on H100 [C].
- **Post-training (Stage 3).** ES (EGGROLL-style) on the fewer than 100 physics and coupling scalars (m, kernel weights, tether gain), plus optional low-rank adapters on the output heads.
  - Population 32, 30 generations, about 1 min per proxy evaluation ⇒ 32 × 30 × 1 min ≈ **16 GPU-h** [E].
- **Total programme ≈ 150 GPU-h [E].**

### 3.7 Cheap impact proxy for post-training

- **Proxy campaign:** N = 20 children at the stock's ΔN, plus 10 silent execution-durations, 256 rollouts per direction, antisymmetrised, binned-means δ (18 bins), Y, e/p, drift.
  - Cost ≈ 2 × (20 × 401 + ~10k) messages per rollout on AMD [E].
- **Fitness:** |δ − 0.5| + λ₁·|ln(Y/0.7)| + λ₂·max(0, e/p − e/p_OW) + λ₃·|drift|/5 + λ₄·max(0, H − H_replay − 0.02) [E; weights to be tuned].
- **Caveat.** Shorter metaorders bias δ upward [C, Sec 2.3]. Calibrate the proxy against the full campaign and require rank correlation ≥ 0.7 on the 12 ladder checkpoints before trusting it [E].

### 3.8 Gate specification

- **Base criteria.** Adopt the LOB-Impact C1-C5 verbatim [C]:
  - C1 tolerances: γ_flow 0.10, H 0.02, drift 5 ticks.
  - C3: v/b ≥ 3.
  - C4: |δ − 0.5| ≤ 0.10 and Y in [0.5, 1].
  - C5: e/p ≤ OW, slope ≤ 0, drift < 20 ticks.
  - Sizing: 1,024 rollouts per direction per phase over 20 days; never fewer than 5 days.
- **Additions:**
  - TOST on δ: the 90% day-bootstrap CI must lie inside [0.40, 0.60]. 90% half-widths ≈ 0.95-level half-widths × 0.839, i.e. 0.034-0.117 [E: 1.645 / 1.96 = 0.839].
  - Lower bound of the v/b ratio ≥ 3.
  - One-sided H ≤ H_replay + 0.02.
  - Placebo battery: sign-scrambled, far-touch and zero-size injections, time-shifted contexts.
  - Stretch target: e/p ≤ 0.67.
  - Held-out stock: Y within [0.25, 2] (EA shows OOD Y of 12-50 [C]).
  - **Blind-arm rule:** every engineered feature (slow bank, LLOB source term) is computed from the model-visible stream, so in the blind arm the injected fills are invisible to the features too.
  - Checkpoint selection on the validation month only; PBO via CSCV < 0.5 [Track 5; L].

### 3.9 Leakage controls

- Calendar-block splits with a 1-trading-day embargo (equities) and ≥ 24 h (crypto).
- All crypto assets on the same calendar split, because of BTC→ETH cross-impact (2504.15908) [Track 5; L].
- January 2026 LOBSTER is excluded from every stage, including synthetic-generator and kernel calibration (τ, m, LLOB parameters fitted on training months only).
- Post-training never sees the test month or the held-out stock.
- Every test-month query is logged.

### 3.10 Ordered experiments (cheapest first, each can kill the design)

1. **E1: physics ceiling, CPU only, about 1 day.** Run QR(90) + 2603.24137 feedback tilt (+ HFT replenishment variant, + LLOB variant) through the open LOB-Impact harness on AMD and EA, with m and τ calibrated on 2025 data.
   - **Kill condition:** no variant gets δ ≤ 0.65 with e/p ≤ OW. Then C4 is unreachable with explicit physics at this operating point: re-scope the claim to C3 + C5 + Y and report it.
   - **Confirm condition:** a variant passes C5 (and Y), which fixes the physics module.
2. **E2: small-neural ladder, about 100 GPU-h.** On 2022-2025 LOBSTER (or a 2-stock subset), train 4 arms at about 8M: (a) Mamba-3 at 1 step/event; (b) FRAC mixer; (c) (b) + physical-time Δ + frozen slow modes; (d) (c) + slow-flow bank.
   - Score C1-C5 on AMD/EA plus the proxy.
   - **Kill condition:** (d) does not beat S5-120M's AMD δ = 0.66 and e/p = 1.19 [C].
   - **Confirm condition:** e/p < 1 with slope ≤ 0 and |drift| < 20.
   - This also measures proxy validity.
3. **E3: synthetic prior value, about 20 GPU-h.** Take arm (d) with vs without Stage-1 almagest-sim pretraining (planted metaorders, shared-noise pairs), then fine-tune on the same real data.
   - **Kill condition:** no gain in Y or e/p ⇒ drop Stage 1 and keep only the physics module.

### 3.11 Top risks

1. C4 may be unreachable under this protocol (§1.3) [L].
2. "Baked-in answer" criticism. Mitigation: blind arm honesty, C1 balance, and ablation (d) vs (c) [L].
3. Competition:
   - M3 has public code and weights, including a tiny 9.66M model [C]; anyone can run LOB-Impact on it.
   - Oxford owns the benchmark [C].
   - Claim only a sub-30M C4/C5 pass with blind attribution.
4. The per-event factored decoder may lose within-message conditioning and break cancel references [G].
5. Near-critical modes (τ up to 2^17) may drift over 70k-step rollouts [L, from 2610.08816 via Track 1].
6. Crypto Y and relaxation targets are thin; the only metaorder-level anchor is 2011-13 MtGox [Track 4; C per track].
7. Evaluation power: about 20 effective days ⇒ TOST is underpowered, so plan ≥ 40 test days [Track 5; L].

**Files:** none written. Sources: arXiv 2610.27100, 2608.19227, 2609.36314, 2603.24137 and 2610.28753 (read via alphaXiv in this pass), plus the track-cited ids above.