# World Models, Test-Time Context Inference, and the Limits of Counterfactual Identification (for "Hindsight Planning" with Latent/JEPA World Models)

Scope note: all 2026 world-model papers below were read from their arXiv PDFs via the alphaXiv PDF tool (v1 unless noted). Classic meta-RL and causal-inference results are cited to their canonical arXiv/proceedings pages; those were not re-read in full in this session and are summarized from well-established content (flagged where relevant).

---

## Q1. Counterfactual World Models: JEPAs That Reason About "What If" (Balestriero, LeCun, Varaku; arXiv 2611.24778, 18 Nov 2026)

### Takeaway
This is the anchor paper. It proves that factual (unpaired) world models cannot identify individual counterfactuals. It then derives "residual reuse", ẑ(a′) = z_obs + μ(z,a′) − μ(z,a). The rule is exact in additive worlds, and its error is O(h²) relative to the disturbance variance for short, smooth, separate-channel physics. That rate is minimax-optimal with a sharp leading constant. The rule fails at first order for gain (multiplicative) faults, contacts and folded coordinates. In one-transition hindsight planning, residual reuse beats a true-simulator oracle that does not see the outcome (35–69% vs 14–23% success). The paper gives no regret theory for planning, and its analysis covers a single transition only: there is no multi-step chaining.

### Cited Findings
**Setup and rule**
- Dynamics are s_{t+1} = F_s(s_t, a_t, ε_t), read as a plant, a structural causal equation and a world model at once. Here ε is the exogenous disturbance held fixed in the counterfactual — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Definition 3.1 (Counterfactual world model): ẑ_{t+1}(a′) = z_{t+1} + μ(H_t,a′) − μ(H_t,a_t), where H_t may include past embeddings and actions. A readout version applies π(z_{t+1}) + π(μ(H,a′)) − π(μ(H,a)). The two differ when π is nonlinear. No paired data, extra parameters, disturbance inference or retraining is needed — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Exact error identity (eqs. 9–10): T_R − Y(a′) = E_ν[Δ] − Δ, so the mean squared error equals tr Var_ν(Δ), with Δ = Y(a′) − Y(a). The error is the variance of the individual action contrast across disturbances, which is the gap between the individual treatment effect (ITE) and the conditional average treatment effect (CATE). Residual reuse is exact for an action pair iff the contrast is disturbance-independent — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)

**Nonidentification theorems**
- Proposition 2.1: two structural models can agree on every action-conditional outcome law yet predict different alternatives for the same observed transition — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Theorem 2.2 / H.1 (factual gauge nonidentification): suppose P_{s,a} = (T_{s,a})_#ν. For any family of ν-preserving bijections G_{s,a}, the mechanisms T̃ = T∘G induce identical factual laws. Their counterfactual coupling T_{s,a′}∘G_{s,a′}∘G_{s,a}^{-1}∘T_{s,a}^{-1} differs unless the gauge is action-independent. Therefore likelihood, calibration, or any criterion that depends only on {P_{s,a}} cannot identify the same-disturbance counterfactual. Example: ν = N(0,I₂) with G = R(ka) yields a rotation counterfactual. The converse also holds: every factual-law-preserving reparameterization has this form — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Theorem H.2 (sharp minimax impossibility): take U ~ Unif(0,1), M₊: (Y(0),Y(1)) = (U,U) and M₋: (U,1−U). With unpaired factual data (even n = ∞), inf_δ max_m risk = (y−½)² pointwise, 1/12 on average and ¼ in the worst case, all attained by Ŷ₁ ≡ ½. The paper frames this as an instance of the causal hierarchy separating interventions from counterfactuals (it cites Bareinboim et al.) and of the classical nonidentification of the joint distribution of potential outcomes — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Table 1 (identification map):
  - Interventional mean and distribution: identified under exchangeability and overlap.
  - Individual counterfactual with unrestricted coupling: not identified.
  - Same, with exact factual likelihood: not identified (gauge).
  - Additive response: exact by residual reuse.
  - Monotone scalar response: identified by rank transport.
  - Smooth force-like physics: O(h²) relative error.
  - Paired data or a simulator: identified.
  - Contacts, multiplicative faults, folded coordinates: additional structure needed — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)

**Positive theorems**
- Proposition 3.2 (uniqueness): any rule exact in every additive world F = m(z,a) + q(z,ε), with a common disturbance law and supported queries, must equal y + μ(z,a′) − μ(z,a). The counterfactual is identified even though ε itself is not, since q may be noninjective — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Theorem 3.3 (interaction bound): assume the mixed derivative satisfies ‖∂_a∂_ε F[u,v]‖ ≤ K_z‖u‖‖v‖. Then E‖T_R − Y(a′)‖² ≤ K_z²‖a′−a‖² tr Cov(ε). Under a separation condition κ_z, the relative risk satisfies ρ_z(a,a′) = tr Var(Δ)/tr Var(Y(a′)) ≤ (K_z‖a′−a‖/κ_z)². ρ < 1 iff residual reuse beats the best outcome-blind (exact) predictor — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Theorem 3.4 / Corollary 3.5 (short-time): with separate smooth action and disturbance channels, the interaction is second order in transition duration h, giving ρ_h = O(h²). The window for weak disturbances scales as ∝ κσ. Corollary E.4 extends the result through smooth representations and to hidden velocities (inertial dynamics); the paper says this is specific to inertial dynamics, not general insufficient representations — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Theorem 3.6 (minimax rate-optimality): over the class F_C of worlds satisfying Thm 3.4 with remainder constant C and actions in [−1,1], 4C²h⁴/(1+C²h²) ≤ inf_δ sup E(δ − Y(1))² ≤ sup E(T_R − Y(1))² = 4C²h⁴. This holds even when the estimator knows exact factual laws at every source, action and duration plus the observed outcome Y_h(−1). The lower bound uses a sign-ambiguous rotation of the disturbance and holds within smooth autonomous ODEs. The paper notes this is worst-case: "a known physical mechanism, paired counterfactual observations, or repeated observations of a shared disturbance may permit more accurate estimators" — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Theorem 3.7, Corollary 3.8 and Proposition 3.9: under shared-input and exchangeability conditions, a squared-loss factual predictor supplies the correct (causal) action contrast without propensity weighting or encoder retraining. The paper assumes logged actions depended on nothing the world model does not see — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)

**Stated failure modes and measured evidence**
- Multiplicative or gain faults (ṡ = aε) interact at first order. The error is ∝ h(a−a′)ε and the relative risk need not shrink with h. On the offset-to-gain interpolation σε((1−λ) + λa/a_max), theory predicts ρ = 0, 0.5 and 2 at λ = 0, 0.5 and 1. Measured values were 0.50/0.48 (HandReach/Ball-in-Cup) at λ = 0.5 and 2.01/1.97 at λ = 1 — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- In learned gain-fault experiments at λ = 1, feature-space correction falls to 6.4–11.0% R², below the trained predictor at 41–42%. Readout correction falls to −12.8 to −11.8% — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Contacts, clipping and saturation are nonsmooth and violate the uniform expansion. Push-T block velocities have ρ > 1, and residual reuse with exact simulator means gives R² of 5.6, −15.5 and −44.4%, against outcome-blind references of 20–39% — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Folded coordinates (cos(pitch)) give ρ > 1. On the boat task, MAE-B raw features give ρ = 1.92, so additivity can hold in physical readout coordinates but not in the features themselves — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- On a damped pendulum, an additive force gives ρ decaying like h⁸ and uncertain damping gives h². Gain faults stay near 2.06 and saturation near 0.06 regardless of h. Damping faults sit at 0.74–1.08 because they move the outcome so little — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- The interaction ratio "needs paired outcomes, so it is an evaluation diagnostic, not a quantity one can compute from factual logs". Factual data can flag a large interaction (outcome spread grows with the command) but cannot certify a small one — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)

**Prediction results**
- Physical counterfactual R² rises from 38–46% to 86–96% with readout correction on four environments (DINOv2, MAE), and to 73–94% with feature-space correction. BlueROV2 reaches 90–94% with 16× more factual data. Four further encoders (DINOv3, SigLIP 2, CLIP, EVA-02) give 81–95%. Residual reuse beats an oracle that knows the simulator and full state but not the disturbance — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- With exact simulator means, residual reuse explains 98.3–100% of counterfactual variance on primary coordinates. Most of the remaining gap is therefore learning error in the action contrast — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)

**Hindsight planning (Section 4.4, Table 15)**
- Protocol: after one factual transition, the planner receives a goal that another action reached under the same source state and disturbance, so the goal is attainable by construction. It selects a command by cross-entropy search (CEM: 256 candidates, 32 elites, 4 rounds). The command is scored by replaying the same event in simulation; success means standardized distance below 0.2. The outcome-blind oracle minimizes expected squared distance over the disturbance law (6-node Gauss–Hermite) — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Success rates (MAE-B), listed as trained predictor / oracle without outcome / counterfactual world model:

  | Environment | Trained predictor | Oracle, no outcome | Counterfactual WM |
  |---|---|---|---|
  | HandReach | 11 | 15 | 37 |
  | Ball-in-Cup | 10 | 15 | 60 |
  | FetchReach | 11 | 14 | 69 |
  | BlueROV2 | 18 | 23 | 50 |
  | Liquid transport | 13 | 19 | 62 |

  The true-disturbance oracle succeeds 100% everywhere. The counterfactual-model gain over the outcome-blind oracle is positive in all 10 environment–encoder cells, with bootstrap CIs above zero (e.g. +22 to +55 points with MAE-B). The paper notes the outcome-blind oracle "is a reference for that objective, not a universal upper bound on thresholded success" — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Two more same-event retries raise counterfactual-model success to 51–97%, while the trained predictor stays at 9–24% — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Over three consecutive transitions:
  - A persistent fault improves success from 4–25% to 11–51%.
  - Fresh (i.i.d.) disturbances change success by only −4 to +6 points.
  - With no disturbance, residual correction reduces replanning success by 4–15 points.
  - With fresh disturbances at loose tolerances, the correction lowers success.
  - Command-specific error analysis (BlueROV2): the residual removes 83% of the error for a same-event retry and 75% for the next transition under a persistent fault, but −137% (it adds error) under a fresh disturbance. "The planning benefit therefore depends on whether the disturbance revealed by the outcome also affects the action being evaluated" — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Stated limitations:
  - The rule needs additive coordinates and a readout with labels.
  - It needs a separate disturbance channel, no contacts and short duration.
  - "It reuses one observed residual over one transition. It does not chain counterfactuals over long horizons."
  - The contrast must be causal (no hidden operator confounding).
  - Planning tasks use one-transition search and at most three steps, and all environments are simulated.
  - Two of seven environments failed observability criteria — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Open directions the authors list:
  - Paired trials to certify safety.
  - Training encoders so that actions and disturbances add.
  - Mode-switching or command-rescaled transport for contacts and gain faults.
  - Action-response-targeted model selection.
  - Chaining counterfactuals over several transitions — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)

### Inferences
- The paper's planning result is empirical. No theorem links the counterfactual mean-squared-error bound (Thm 3.3/3.4) to decision quality. A natural bridge is: hindsight-planning suboptimality ≤ Lipschitz(cost) × counterfactual error + search error. Comparing that with the forward planner's irreducible error tr Var(Y(a′)) gives a "hindsight beats forward iff ρ < (something)" result. ρ < 1 is already the prediction-level version of "exactly when".
- The persistence findings, together with the replanning losses when no disturbance is present, imply a clean planning-level criterion: residual transport helps the next decision iff the disturbance autocorrelation exceeds a threshold tied to ρ. The paper reports this pattern but does not prove it.
- The minimax lower bound is per transition with one outcome. Multiple observed outcomes under a persistent disturbance should allow rates beyond O(h²); the paper says so explicitly ("repeated observations of a shared disturbance may permit more accurate estimators"). This is an open theoretical lever.

### Gaps
- Not stated in the paper: any regret or suboptimality bound for hindsight planning; multi-step error propagation (compounding of residuals across transitions); treatment of switching or regime-changing disturbances; and a test of the interaction ratio computable from factual data alone (the authors say this is impossible in general).
- Appendix proofs of Thm 3.4 and 3.7 were not read in full; the statements above come from the main text.

---

## Q2. Recent latent-world-model test-time adaptation and counterfactual papers (2026)

### Takeaway
All six methods (T2-Z, AdaJEPA, JEPA-TTT, sft-JEPA, OpTTA, Look Back/RLB) are empirical test-time adaptation (TTA) or context-inference methods. Only Look Back (RLB) and the CoRL headroom paper prove anything substantive. RLB gives unbiasedness and variance conditions for counterfactual baselines, plus sample-complexity separations between learning a counterfactual model and an ordinary world model. The headroom paper gives an information-theoretic definition (one-step conditional mutual information) of how much context inference can add. None gives regret or planning guarantees for latent TTA. sft-JEPA's empirical finding that adaptation mostly corrects an action-independent offset is directly relevant: a shared offset is exactly what residual reuse captures.

### Cited Findings
**T2-Z (Zhou, Hong, et al., Stanford/CMU/TRI/USC; arXiv 2610.22364)**
- Method:
  - Augment a video world model (BWM on Wan2.2-TI2V-5B) with a 32-D environment latent z_e, with no supervision of which factors vary.
  - Train by alternating flow-matching optimization: θ with z_e fixed (lr 1e-5), then z_e with θ frozen (lr 0.03).
  - At test time, freeze θ, initialize a new z, and optimize it for 40 steps on 1–2 interaction trajectories with random actions.
  - Then predict the outcomes of other actions with p_θ(y | x, a′, z*). This is "analysis by synthesis" (non-amortized), and the authors contrast it with RMA's predefined privileged factors and learned history encoders — [arXiv 2610.22364](https://arxiv.org/abs/2610.22364)
- Results:
  - Simulation (7 settings): average physical error 12.86 → 4.80 and action success 37.9% → 74.2%.
  - Real world (4 tasks): tracked-state error 19.30 → 10.85 px and success 26.47% → 70.80%.
  - Baselines: LoRA TTA, DINOv2 context encoder, TTT-KQV — [arXiv 2610.22364](https://arxiv.org/abs/2610.22364)
- Interpretability: PCA of the latents shows 1-D structure aligned with friction, a 2-D structure for mass × friction, and alignment consistent across 30 visual backgrounds. Ground truth is used only for visualization. The evidence is qualitative; no identifiability claim or theorem is made — [arXiv 2610.22364](https://arxiv.org/abs/2610.22364)
- Ablation: directly optimizing the full 3072-D conditioning token collapses performance (error 94.5, success 24%). A low-dimensional latent is essential — [arXiv 2610.22364](https://arxiv.org/abs/2610.22364)
- No theory. Context is a per-environment constant (episode-level), so this is a population or parameter-level context, not an event-level disturbance — [arXiv 2610.22364](https://arxiv.org/abs/2610.22364)

**AdaJEPA (Wang, Bounou, LeCun, Ren, NYU/AMI; arXiv 2606.32026)**
- Method:
  - Plan–execute–adapt–replan inside MPC.
  - After each executed action chunk, take one gradient step on the latent prediction loss with stop-gradient targets, over a 5-transition recent buffer.
  - Update only the last predictor and encoder layers.
  - Reset to pretrained weights each episode.
  - Claims to be the first to adapt a JEPA world model during planning — [arXiv 2606.32026](https://arxiv.org/abs/2606.32026)
- Results:
  - PushObj unseen shapes: roughly doubles success.
  - PointMaze unseen layouts: 53.3 → 66.0 (GD) and 78.7 (first predictor layer).
  - Gains on dynamics shifts (low mass, high damping) are small (+0.7 to +6) because the frozen model already does well, "which might be due to in context learning on the history of three frames."
  - Latency +0.01–0.03 s per replan.
  - Gains are modest for red-anchor/red-block color shifts — [arXiv 2606.32026](https://arxiv.org/abs/2606.32026)
- Data scale: a single-shape model trained on 1k trajectories, once adapted, beats the frozen 64k-trajectory model on seen shapes (61% vs 54%) — [arXiv 2606.32026](https://arxiv.org/abs/2606.32026)
- No theory — [arXiv 2606.32026](https://arxiv.org/abs/2606.32026)

**JEPA-TTT (Zhang, Ye, et al., Honda RI/JHU; arXiv 2610.00722)**
- Method:
  - Persistent self-supervised updates to the predictor only; the encoder and an offline-trained reward head stay frozen.
  - No goal image is used.
  - Updates use dense replay with windows at every temporal offset.
  - The buffer persists across episodes — [arXiv 2610.00722](https://arxiv.org/abs/2610.00722)
- Results:
  - Eight shifts across PushT, Two-Room, Reacher and OGB-Cube: action rotation, contact rotation, wave and grid fields, joint phase, harmonic, quadratic crosstalk, delayed cyclic actuation.
  - After 500 episodes, autoregressive latent MSE drops 83% and best score improves 153% (0.267 → 0.678).
  - Retaining state across episodes raises best score 0.289 → 0.729 under matched updates.
  - Figure 3 shows the frozen model mis-ranking candidate plans, and adaptation fixing the ranking — [arXiv 2610.00722](https://arxiv.org/abs/2610.00722)
- Stated limitations: only a single fixed shift; no sequences of shifts; no study of forgetting or of returning to the training dynamics. No theory — [arXiv 2610.00722](https://arxiv.org/abs/2610.00722)

**"What Test-Time Training Corrects in Latent World Models? A Slow-Fast Approach" (sft-JEPA; Niu et al.; arXiv 2611.29453)**
- Action-fan probe: reset the simulator to a planner state and execute 9 action chunks. Decompose the prediction error e(a) into an offset ē shared across actions and an action-dependent part e(a) − ē, with bias-corrected energies C and A — [arXiv 2611.29453](https://arxiv.org/abs/2611.29453)
- Findings:
  - The offset is 43–79% of the frozen model's error on PushObj.
  - An offset "effectively shifts the planner's goal": ẑ(a) − g = z⁺(a) + d(a) − (g − b).
  - AdaJEPA mainly tracks the offset; the common shift alone recovers 91% of its planning gain.
  - Part of the offset is episode-specific: reusing the previous episode's shift cuts success by 26 points versus fitting it online (16.0% vs 42.7% on shape I), and a fast learner that is never reset loses 10–12 points.
  - Action-dependent error needs cross-episode learning — [arXiv 2611.29453](https://arxiv.org/abs/2611.29453)
- Method: a fast learner (last block, reset each episode) plus a slow learner (LoRA, kept and consolidated after each episode). Results:
  - Beats AdaJEPA by 4–22 points.
  - PushObj unseen: 69.3/70.7% vs AdaJEPA's 50.0/52.9%.
  - LeWAM on OGBench with arm shifts (rotation, worn arm, offset, gain 0.5 + per-episode offset): average 51.2% (Puzzle) and 68.5% (Scene) vs AdaJEPA's 30.5 and 51.4 — [arXiv 2611.29453](https://arxiv.org/abs/2611.29453)
- The analysis is empirical decomposition, not a theorem. The paper does not cite or connect to counterfactual or residual-reuse theory — [arXiv 2611.29453](https://arxiv.org/abs/2611.29453)

**"Adapt for Planning, Not Just Prediction: Optimistic Test-Time World Model Adaptation" (OpTTA; Mete, Kalathil, Kumar, Texas A&M; arXiv 2610.35636)**
- Claim: prediction-only TTA fits the model only to the outcomes of the agent's own actions, so the model "can remain wrong about the actions that would reach the goal". The authors call this a closed-loop identifiability problem (Borkar & Varaiya 1979) and an instance of objective mismatch (Lambert et al. 2020) — [arXiv 2610.35636](https://arxiv.org/abs/2610.35636)
- Method: L = L_pred + α_k L_plan, inspired by reward-biased maximum likelihood estimation (RBMLE; Kumar & Becker 1982). The planning cost is differentiated with respect to predictor parameters over the final CEM candidates, and α_k is gradient-norm-scaled and grows as √(k/K_max) — [arXiv 2610.35636](https://arxiv.org/abs/2610.35636)
- Results:
  - Over 15 shifted settings and three JEPA world models (LeWM, DINO-WM, temporal straightening), the mean gain over frozen is +13.5 points vs +6.9 for AdaJEPA (predictor-only), and +13.5 vs +7.5 with the encoder also adapted.
  - OpTTA beats AdaJEPA in 14 of 15 settings.
  - Both methods hurt on TwoRoom, and the authors say this task dependence is unexplained — [arXiv 2610.35636](https://arxiv.org/abs/2610.35636)
- Theory is only cited (RBMLE and UCB regret in adaptive control). The paper proves no guarantee for the TTA objective itself — [arXiv 2610.35636](https://arxiv.org/abs/2610.35636)

**"When Does Dynamics Randomization Test Context Inference?" (Smirnov et al., King AI Labs/KTH; CoRL 2026; arXiv 2610.32414v2)**
- Formal definition:
  - Predictive headroom I_emp = E_d KL(T_θ(·|s,a) ‖ T̄_d(·|s,a)) = I_d(θ; s′ | s, a). This is the minimum one-step log-loss gain from revealing the context, beyond (s, a).
  - Two prior-weighted variants are also defined: I_struct (fixed queries μ) and I_match (visited queries).
  - Because the three quantities use different distributions, "those quantities do not constitute general bounds on one another" — [arXiv 2610.32414](https://arxiv.org/abs/2610.32414)
- Findings:
  - MiniGrid Wind (v_max = 1): I_struct 0.871, I_match 0.620, Î_emp 0.077. Visitation-dependent reweighting removes 88% of the headroom, because state already encodes the context.
  - Resistance: 0.062 / 0.059 / 0.055. Headroom is low because the kernels are weakly distinct.
  - Point-Mass-Attractor: 1.006 prior vs 0.109 visited.
  - Resetting state components raises headroom: Wind full reset every 10 steps gives +430% and every 2 steps +833%; Attractor full reset gives +324% and position-only reset +28%. "Context evidence accumulates in state", including velocity — [arXiv 2610.32414](https://arxiv.org/abs/2610.32414)
- Limitations: data collected only under random actions; one-step prediction only; not planning — [arXiv 2610.32414](https://arxiv.org/abs/2610.32414)

**"Look Back: Reinforcement Learning with Counterfactual World Models" (Kim, Abbasov, …, Cho, Ranganath, KAIST/NYU/MIT; arXiv 2611.29259)**
- Defines a counterfactual world model as p_ω(x_{i+1} | x_i, a_i, o_i), conditioned on a factual transition o_i under the same environmental noise u_i (SCM formulation following Buesing et al.). Two ways to build one: training on paired shared-noise data, or inverting a diffusion world model (diffusion counterfactual inference) — [arXiv 2611.29259](https://arxiv.org/abs/2611.29259)
- Propositions:
  - Prop 1: maximum-likelihood training on paired data recovers the true counterfactual.
  - Prop 2: an ML counterfactual model over a finite class Ω has expected squared Hellinger error ≤ (1 + log|Ω|)/N.
  - A WM lower bound (Props 3–5): the counterfactual-model upper bound is smaller than the worst-case WM lower bound when E² > 48A²(3i+4)(1+log|Ω|). That is, when environmental evolution varies more than action-induced change, counterfactual models are provably easier to learn.
  - Prop 6: an invertible diffusion WM-induced counterfactual model gives exact counterfactuals of some SCM consistent with the MDP — [arXiv 2611.29259](https://arxiv.org/abs/2611.29259)
- REINFORCE Look Back (RLB) uses counterfactual returns as RLOO baselines.
  - Prop 8: RLB is unbiased under exact counterfactuals.
  - Corollary 1: it stays unbiased with the counterfactuals of any SCM consistent with the MDP, so identifiability is not needed for unbiasedness; the choice of SCM affects only variance.
  - Further propositions give bias bounded by model error and variance-reduction conditions, with luck and interaction terms σ_luck and σ_inter — [arXiv 2611.29259](https://arxiv.org/abs/2611.29259)
- Results: Chrome Dino with a diffusion-induced counterfactual model. RLB + counterfactual model gives 3.77% collision and 109.0 survival, vs RLOO + WM at 7.89% and 97.5 — [arXiv 2611.29259](https://arxiv.org/abs/2611.29259)
- Related-work position: Sinclair et al.'s hindsight learning "lets a planner see the future inputs, which biases its values towards foresight", whereas RLB is unbiased — [arXiv 2611.29259](https://arxiv.org/abs/2611.29259)

### Inferences
- There is a clean taxonomy of what is inferred at test time:
  - Episode-level context (T2-Z, RMA, PEARL, VariBAD).
  - Slowly varying offsets (AdaJEPA's fast tracking, sft-JEPA's fast learner).
  - Persistent shifts learned across episodes (JEPA-TTT, sft-JEPA's slow learner).
  - Event-level exogenous noise (counterfactual world models, RLB).

  Hindsight planning in the counterfactual-world-model sense targets the last category, which is distinct from, and complementary to, test-time adaptation.
- sft-JEPA's "offset shifts the goal" result and the residual-reuse identity are the same mathematical object viewed differently. An action-independent error b cancels in action contrasts but not in distance-to-goal costs; residual reuse removes exactly b, using the observed outcome. A theory paper could unify AdaJEPA-style TTA (b tracked by gradient steps) with residual reuse (b read off in closed form). The episode-specific part of b corresponds to the "fresh vs persistent disturbance" axis in the counterfactual-world-model paper.
- OpTTA's closed-loop identifiability point (fitting only one's own actions) mirrors the counterfactual-world-model requirement that the action contrast be learned with support at a′. Both point to a coverage or overlap term that any regret bound must contain.
- Predictive headroom I(θ; s′ | s, a) is a candidate "value of hindsight information" quantity. Its event-level analogue would be I(ε; Y(a′) | s, a, Y(a)), the information the observed outcome carries about the counterfactual. Residual reuse's (1 − ρ) is a second-moment surrogate for it.

### Gaps
- None of T2-Z, AdaJEPA, JEPA-TTT, sft-JEPA or OpTTA proves convergence, regret, identifiability of the inferred latent, or planning-performance guarantees.
- T2-Z interpretability is PCA-only, with no identifiability-up-to-transformation result.
- JEPA-TTT and sft-JEPA do not study switching or sequential regimes formally; sft-JEPA has only a mixed-shape stream experiment.
- I did not read the appendices of Look Back (e.g. the exact statement of the variance proposition).

---

## Q3. Meta-RL context inference: PEARL, RMA, VariBAD, and Bayes-optimality theory

### Takeaway
Classic context-inference methods amortize posterior inference over a latent task. PEARL uses posterior sampling, VariBAD approximates Bayes-adaptive behavior, and RMA regresses privileged extrinsics. They inherit Bayes-optimality only in the idealized limit. The Bayes-adaptive MDP framework defines optimality, and posterior sampling (PSRL) and information-directed methods give Bayesian regret bounds for tabular or finite-context settings. No regret theory exists for latent test-time inference in learned (JEPA) world models.

### Cited Findings
- PEARL (Rakelly et al., ICML 2019):
  - Off-policy meta-RL with a probabilistic context encoder q(z|c) over task latents, a permutation-invariant product of Gaussians, trained with an information bottleneck.
  - Exploration by posterior sampling of z (Thompson-style), with the sample held per episode.
  - Empirical; no regret guarantee — [arXiv 1903.08254](https://arxiv.org/abs/1903.08254)
- VariBAD (Zintgraf et al., ICLR 2020):
  - Meta-learns approximate variational inference over a task embedding m, using a recurrent encoder and a decoder that reconstructs past and future transitions and rewards.
  - The policy is conditioned on the posterior belief, approximating Bayes-optimal behavior in the BAMDP.
  - Bayes-optimality is approximate; there is no regret bound — [arXiv 1910.08348](https://arxiv.org/abs/1910.08348)
- RMA (Kumar, Fu, Pathak, Malik, RSS 2021):
  - Phase 1: train a base policy with an environment-factor encoder over privileged simulator parameters (mass, friction, motor strength, etc.), giving an extrinsics vector z.
  - Phase 2: train an adaptation module to regress z from a recent state–action history, enabling real-world deployment without fine-tuning.
  - The latent is a learned embedding of predefined factors. T2-Z criticizes it as "limited to pre-defined environmental factors with ground-truth values" — [arXiv 2107.04034](https://arxiv.org/abs/2107.04034); critique in [arXiv 2610.22364](https://arxiv.org/abs/2610.22364)
- Bayes-optimality of memory-based meta-learning:
  - Ortega et al. (2019) argue that meta-learned sequential strategies (memory-based) approximate Bayes-optimal agents — [arXiv 1905.03030](https://arxiv.org/abs/1905.03030)
  - Mikulik et al. (NeurIPS 2020) give empirical evidence that meta-trained agents implement Bayes-optimal agents — [arXiv 2010.11223](https://arxiv.org/abs/2010.11223)
- Posterior sampling regret: PSRL (Osband, Russo, Van Roy 2013) gives Õ(τS√(AT)) Bayesian regret for finite MDPs. This is the theoretical template for PEARL-style posterior sampling — [arXiv 1306.0940](https://arxiv.org/abs/1306.0940)
- Contextual POMDP theory (2026): C-IDS gives an information-directed objective for latent-context inference, with a sublinear Bayesian regret bound of roughly √K and log|C| (finite context set). Not verified in depth — [arXiv 2602.03939](https://arxiv.org/abs/2602.03939)
- Related meta-bandit regret results: Meta-Thompson Sampling — [arXiv 2102.06129](https://arxiv.org/abs/2102.06129); prior-aligned meta-RL meta-regret — [arXiv 2510.05446](https://arxiv.org/abs/2510.05446). These are bandit or tabular, not latent world models.
- Hindsight learning for Exo-MDPs (Sinclair et al., ICML 2023):
  - When the only unknowns are exogenous inputs, use hindsight-optimal actions computed on the realized exogenous trace as a training signal, with decision-performance guarantees.
  - Look Back notes that hindsight values are biased "towards foresight" — [arXiv 2207.06272](https://arxiv.org/abs/2207.06272); [arXiv 2611.29259](https://arxiv.org/abs/2611.29259)
- Context information can already live in the state, so context encoders may have little to add. This is quantified by predictive headroom — [arXiv 2610.32414](https://arxiv.org/abs/2610.32414)

### Inferences
- "Hindsight planning" (choose a′ for the event that just occurred) differs from Bayes-adaptive forward planning. The forward Bayes-optimal planner conditions on the posterior over context or noise before acting. The hindsight planner conditions on the realized outcome of this event. The value gap between them is exactly the value of information of the realized disturbance. The counterfactual-world-model oracle comparison (outcome-blind vs true-disturbance oracle: 14–23% vs 100%) is an empirical measurement of that gap, and residual reuse recovers part of it.
- The prophet or hindsight-optimal literature (Exo-MDPs) gives the "foresight bias" caveat. Hindsight-optimal values overestimate achievable forward values, so a theorem claiming hindsight planning is "better" must specify better at what. Possible objectives: (a) counterfactual decision accuracy for the same event (diagnosis or retry), or (b) forward performance when the disturbance persists. These are different claims.

### Gaps
- The PEARL, VariBAD and RMA PDFs were not re-read this session; the summaries above are standard characterizations.
- I found no paper giving regret or suboptimality bounds for test-time latent inference in learned (non-tabular, representation-based) world models, nor for JEPA-specific TTA.

---

## Q4. Counterfactual identification theory: ladder of causation, causal hierarchy theorem, nonidentifiability, restoring conditions, counterfactual credit assignment

### Takeaway
Individual (rung-3) counterfactuals are generically not identified from interventional (rung-2) data. This is the causal hierarchy theorem; it also follows from the Fréchet-style nonidentifiability of joint potential outcomes. Identification is restored by structural restrictions:
- additive noise or bijective/monotone mechanisms (1-D monotonicity, counterfactual stability for Gumbel-max);
- paired or shared-noise data;
- known mechanisms or simulators.

The counterfactual-world-model paper adds a physics-derived approximate restriction (separate channels, short h) with a rate. In RL, counterfactual credit assignment uses these ideas for variance reduction, and unbiasedness can hold without identifiability.

### Cited Findings
- Pearl's ladder or hierarchy (association, intervention, counterfactual), formalized with the Causal Hierarchy Theorem (CHT): for almost all SCMs, lower-layer data underdetermine higher layers — Bareinboim, Correa, Ibeling, Icard, "On Pearl's Hierarchy and the Foundations of Causal Inference" (2022) — [causalai.net/r60.pdf](https://causalai.net/r60.pdf). The counterfactual-world-model paper cites this as [8] — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Neural causal models inherit the CHT: no amount of L2 data or expressivity lets a neural SCM answer L3 queries without extra structure — Xia, Lee, Bengio, Bareinboim (NeurIPS 2021) — [arXiv 2107.00793](https://arxiv.org/abs/2107.00793)
- Bounds rather than point identification for probabilities of causation (PN, PS, PNS), from combined experimental and observational data, with point identification under monotonicity — Tian & Pearl (2000) — [arXiv 1301.3898](https://arxiv.org/abs/1301.3898)
- Bijective Generation Mechanisms (nonlinear additive noise, location-scale, post-nonlinear) give counterfactual identifiability in several structures with unobserved confounding, given the causal diagram — Nasr-Esfahany, Alizadeh, Shah (ICML 2023) — [arXiv 2302.02228](https://arxiv.org/abs/2302.02228). Companion paper: "Counterfactual (Non-)identifiability of Learned Structural Causal Models", where multi-dimensional bijective mechanisms are not identified in general (cf. the rotation gauge in the counterfactual-world-model paper) — [arXiv 2301.09031](https://arxiv.org/abs/2301.09031)
- Monotone scalar response gives identification by rank transport, and additive response gives exactness by residual reuse (counterfactual-world-model paper, Table 1, citing [14, 5, 37, 61] and [26, 45]) — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Oberst & Sontag (ICML 2019):
  - MDP transitions do not identify counterfactuals.
  - They propose Gumbel-max SCMs satisfying "counterfactual stability" for counterfactual off-policy evaluation.
  - Look Back cites this as the canonical nonidentifiability of MDP counterfactuals — [arXiv 1905.05824](https://arxiv.org/abs/1905.05824); [arXiv 2611.29259](https://arxiv.org/abs/2611.29259)
- Buesing et al., "Woulda, Coulda, Shoulda: Counterfactually-Guided Policy Search" (ICLR 2019):
  - Write POMDPs as SCMs.
  - Infer scenario noise from logged episodes with a given mechanism.
  - Simulate alternative policies.
  - Motivated by cases where scenario distributions are harder to model than mechanisms — [arXiv 1811.06272](https://arxiv.org/abs/1811.06272); characterization confirmed in [arXiv 2611.29259](https://arxiv.org/abs/2611.29259)
- Counterfactual Credit Assignment (Mesnard et al., ICML 2021): value baselines conditioned on hindsight features, constrained to be independent of the current action, separating skill from luck; model-free — [arXiv 2011.09464](https://arxiv.org/abs/2011.09464); characterization in [arXiv 2611.29259](https://arxiv.org/abs/2611.29259)
- Related RL lines (via Look Back's related work):
  - HCA (Harutyunyan et al. 2019).
  - QCA (Mesnard et al. 2023).
  - Input-dependent baselines (Mao et al. 2018), which condition on exogenous inputs.
  - Exogenous-state removal (Dietterich et al. 2018).
  - Exo-MDP sample complexity (Efroni et al. 2022).
  - rMC (Bazerghi et al. 2025), which uses coupled true counterfactuals.
  - TRPO vine with common random numbers — [arXiv 2611.29259](https://arxiv.org/abs/2611.29259)
- Unbiasedness without identifiability: counterfactual baselines from any SCM consistent with the MDP give unbiased policy gradients; the SCM choice affects only variance — [arXiv 2611.29259](https://arxiv.org/abs/2611.29259)
- Control-theory precedents: the counterfactual-world-model paper identifies disturbance observers, offset-free MPC and iterative learning control as counterfactual estimators "usually analyzed with a known nominal model, often a linear or linearized one" — [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)

### Inferences
- Identification depends on purpose. Credit assignment and policy gradients need only some consistent coupling (RLB, Cor. 1). Hindsight planning, retries and diagnosis need the true coupling, which requires physical structure such as additivity. A theory of "hindsight > forward planning" must therefore assume a structural restriction. The interaction constant K (Thm 3.3) or ρ is the natural quantity to parameterize how far one may depart from it.
- The rotation gauge in Theorem 2.2 is the same obstruction that keeps multi-dimensional bijective SCMs from being identified (2301.09031). Monotonicity in 1-D removes it, which suggests rank-transport hindsight planning for scalar outcomes as a provable alternative when additivity fails (e.g. gain faults with known sign).

### Gaps
- I did not re-read Bareinboim et al., Tian–Pearl, Buesing or Mesnard in this session; their summaries are standard and partly corroborated through the related-work sections of 2611.24778 and 2611.29259.
- I found no existing result giving decision-theoretic (regret) consequences of partial counterfactual identification for planning, as opposed to estimation risk.

---

## Q5. Open gaps a "hindsight planning is provably better than forward planning, and exactly when" paper could fill for latent world models

### Takeaway
The prediction-level ingredients exist: nonidentification, residual-reuse exactness and uniqueness, the interaction bound, the O(h²) minimax rate, and the empirical persistence dependence. The decision-level theory does not. A paper that turns ρ and the persistence structure into planning-regret statements, extends them to multi-step chains and switching disturbances, and gives factually checkable or learnable additivity conditions would fill clear, explicitly acknowledged gaps.

### Cited Findings
- Explicitly acknowledged gaps in the anchor paper:
  - Single transition only ("does not chain counterfactuals over long horizons").
  - Planning has at most three steps.
  - The interaction ratio cannot be certified from factual logs.
  - Gain faults and contacts need new transport rules (mode switching, command rescaling).
  - Additivity is checked, not learned.
  - Model selection is not targeted at the action contrast.
  - Hidden confounding in logs is untreated.

  [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- Empirical regularities that no theory covers:
  - Persistent fault helps (+7 to +26 points).
  - Fresh disturbances are neutral or harmful.
  - No disturbance costs 4–15 points.
  - Residual removes 83% / 75% / −137% of the error (same event / persistent / fresh).

  [arXiv 2611.24778](https://arxiv.org/abs/2611.24778)
- TTA methods are evaluated with no guarantees, and stated open issues include unexplained task dependence (TwoRoom) [arXiv 2610.35636](https://arxiv.org/abs/2610.35636), forgetting and sequential shifts [arXiv 2610.00722](https://arxiv.org/abs/2610.00722), and the episode-specific vs recurring offset split [arXiv 2611.29453](https://arxiv.org/abs/2611.29453)
- The interpretability of inferred latents rests on PCA plots only [arXiv 2610.22364](https://arxiv.org/abs/2610.22364). A headroom-style information measure exists for one-step prediction but not for planning [arXiv 2610.32414](https://arxiv.org/abs/2610.32414)

### Inferences (candidate contributions, each tied to an evidenced gap)
1. **Decision-level theorem (regret or suboptimality).** For a goal cost c that is L-Lipschitz in readout coordinates, bound the gap between the hindsight planner's expected cost and the true-disturbance oracle by roughly L·√(tr Var Δ) + learning error ε_μ(a,a′) + search error. Bound the forward (outcome-blind) planner from below by roughly L·√(tr Var Y(a′)) (minus curvature terms). The result "hindsight beats forward iff ρ < τ(L, goal tolerance, ε_μ)" is the "exactly when" statement. The anchor paper only shows the prediction analogue (ρ < 1).
2. **Thresholded-success version.** The anchor paper notes its oracle minimizes squared distance, not success probability. A theorem for success probability under tolerance δ, for example via anti-concentration of Y(a′) and concentration of Δ, would explain the 14–23% vs 35–69% gap.
3. **Persistence or autocorrelation criterion.** Model ε_t as AR(1) with correlation φ, or as a hidden Markov switching process. Prove that transporting the residual to the next decision helps iff φ exceeds some φ*(ρ, h), with optimal shrinkage φ·r_t (a Kalman or offset-free-MPC gain). This explains why "no disturbance" hurts: you transport pure model error. It also connects to sft-JEPA's episode-specific vs recurring offset decomposition.
4. **Multi-step chaining.** Bound error compounding when residuals are reused across H transitions, each with its own observed outcome. Find the conditions (contraction of ∂F/∂s, bounded interaction) under which error grows linearly rather than exponentially, and compare with forward rollouts that compound an outcome-blind error. Explicitly listed as open direction (v).
5. **Beyond O(h²) with repeated observations.** With k observations of a shared persistent disturbance (retries), give rates that beat the single-outcome minimax bound. The anchor paper says this is possible but gives no bound. Empirically, retries reach 51–97%.
6. **Gain faults and contacts.** Give provable transport rules. For a multiplicative fault, rescale the residual by a′/a (rank or scale transport), which is exact for scalar monotone gain. For contacts, use mode-conditional (piecewise) residual reuse with a misclassification term. Pair each rule with lower bounds showing plain residual reuse is first-order suboptimal there (ρ → 2 at λ = 1, as measured).
7. **Switching disturbances and regime detection.** A hindsight planner must decide whether the disturbance it just observed is still active. Combine change-point detection with residual reuse and derive detection-delay vs regret trade-offs. Not addressed by any paper read here.
8. **Factually checkable sufficient conditions.** The ratio ρ cannot be certified from factual logs (anchor paper), but factual data can flag large interactions (heteroscedasticity in the command). Possible results: one-sided tests, or a minimal number of paired trials that certifies ρ < τ with confidence 1 − δ.
9. **Learning additive coordinates.** Training an encoder or readout so that ∂_a∂_ε F ≈ 0 in latent space, i.e. learned additivity (open direction ii). A theory could link a JEPA regularizer (e.g. penalizing the action–residual interaction) to a bound on ρ in the learned latent space. This directly addresses the MAE-B ρ = 1.92 failure in raw features.
10. **Unification with TTA.** Formalize AdaJEPA, sft-JEPA and OpTTA corrections as estimators of an action-independent offset plus an action-dependent error. Show that residual reuse is the closed-form, zero-step optimal estimator of the event-specific offset, and that gradient TTA is a smoothed or persistent version of it. Give conditions under which each is preferable, in terms of the persistence timescale of the offset. Also add the closed-loop identifiability or coverage term from OpTTA.
11. **Value of hindsight information.** Define an event-level analogue of predictive headroom, I(ε; Y(a′) | s, a, Y(a)) or its second-moment version 1 − ρ. Prove that it upper-bounds the possible hindsight-planning gain, mirroring the CoRL paper's argument that context inference can add only what the inputs miss.
12. **Interpretability of inferred latents.** Give identifiability up to a group (e.g. affine or rotation gauge) for test-time-optimized environment latents (T2-Z-style) or for residuals. Theorem 2.2 suggests the action-dependent gauge is the obstruction, and additivity removes it.

### Gaps
- I found no published 2025–2026 paper that proves hindsight planning (with a learned latent world model) dominates forward planning under stated conditions. The anchor paper is the closest, and it stops at prediction-level guarantees plus empirical planning results. This absence is reported from the papers read here and their related-work sections, not from an exhaustive literature search.
