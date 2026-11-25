# Adaptive control regret theory and disturbance estimation/filtering as foundations of "hindsight planning"

Scope note: "Hindsight planning" here means using observed outcomes to infer hidden disturbances or parameters and then re-plan, i.e. certainty-equivalent (CE) adaptive control or disturbance-observer / offset-free control. "Forward planning" means planning with a fixed prior model (open-loop in the parameters).
Verification legend: **[V]** means checked this session against the primary page (arXiv abstract, PMLR page or paper PDF). **[S]** means checked only against secondary sources or search snippets. **[M]** means from domain knowledge and not re-verified this session, so the writer should check it before quoting. **[D]** means derived here, not quoted from a source.

---

## 1. Least-squares consistency (Lai & Wei 1982), certainty equivalence and its failure modes

### Takeaway
Least squares in stochastic regression with adaptively chosen regressors is strongly consistent whenever the information matrix grows in every direction (λ_min → ∞) faster than log λ_max. The error rate is about log λ_max / λ_min. Certainty equivalence fails exactly when the closed loop stops exciting some parameter direction. This is the identifiability trap that dual control (Feldbaum) addresses by deliberate probing.

### Cited Findings
- Lai, T. L. & Wei, C. Z. (1982), "Least Squares Estimates in Stochastic Regression Models with Applications to Identification and Control of Dynamic Systems," *Annals of Statistics* 10(1):154–166, doi:10.1214/aos/1176345697. The abstract says it establishes strong consistency and asymptotic normality of LS estimates under weak assumptions on the stochastic regressors and errors. — [Project Euclid](https://projecteuclid.org/journals/annals-of-statistics/volume-10/issue-1/Least-Squares-Estimates-in-Stochastic-Regression-Models-with-Applications-to/10.1214/aos/1176345697.full) [S: bibliographic details confirmed via search; the full-text fetch returned empty]
- The consistency condition, as restated by later work: λ_min(Σ_{t≤N} x_t x_tᵀ) → ∞ a.s. and log λ_max(N) = O(λ_min(N)) a.s. Another restatement extends Lai–Wei's Corollary 3 using the form [log λ_max]^ν = o(λ_min) for ν > 1. — [Search summary of Chang & Lin and of arXiv:1711.04990](https://arxiv.org/pdf/1711.04990) [S; the secondary sources disagree on O vs o]
- **[M]** Exact form I believe is in Lai & Wei (1982), Theorem 1. The errors ε_t are a martingale difference sequence with sup_t E(|ε_t|^α | F_{t−1}) < ∞ a.s. for some α > 2, and the regressors x_t are F_{t−1}-measurable. On the event {λ_min(N) → ∞, log λ_max(N) = o(λ_min(N))}, θ̂_N → θ a.s. with ‖θ̂_N − θ‖² = O(log λ_max(N) / λ_min(N)) a.s. Verify o vs O against the PDF.
- Lai–Wei's condition is sufficient but not necessary: Christopeit & Massmann (2013, Tinbergen discussion paper) give a strongly consistent OLS example that violates it. — [tinbergen.nl](https://papers.tinbergen.nl/13151.pdf) [S]
- **[M]** Failure modes of certainty equivalence:
  - **Closed-loop identifiability trap (Borkar & Varaiya 1979):** Borkar & Varaiya, "Adaptive control of Markov chains, I: Finite parameter set," *IEEE TAC* 24(6), 1979. The CE adaptive controller converges to a parameter θ̄ that is indistinguishable from the true θ *under the closed-loop policy induced by θ̄*. That policy need not be optimal for the true θ. This is the canonical "closed-loop identifiability" failure of CE.
  - **Dual control:** Feldbaum, A. A. (1960–61), "Dual control theory I–IV," *Automation and Remote Control* 21–22. Optimal control under parameter uncertainty must trade off "directing" (exploitation) against "investigating" (probing). CE ignores the probing value.
  - **Self-tuning regulator:** Åström & Wittenmark (1973), "On self tuning regulators," *Automatica* 9:185–199. Even when parameter estimates converge to a biased value, the closed loop can still converge to the minimum-variance controller (the "self-tuning property"). So inconsistency of the estimates does not always imply suboptimal control. Textbook: Åström & Wittenmark, *Adaptive Control*, 2nd ed., Addison-Wesley 1995 (Dover reprint 2008).

### Inferences
- **[D]** For a hindsight planner that re-estimates hidden parameters θ by least squares, Lai–Wei supplies the consistency condition: "the history must excite every direction of θ, with λ_min outgrowing log λ_max." Whenever this holds, the per-step estimation error is Õ(1/λ_min(t)). If λ_min(t) grows linearly (persistent excitation), the error is Õ(1/t).
- **[D]** A forward (fixed-prior) planner whose prior is biased by b has an estimation error that stays at b. If the per-step cost gap is at least c·b² (locally quadratic, see Mania et al. below), its regret is Ω(c b² T): linear. A consistent hindsight planner's regret is sublinear. This is the cleanest "hindsight beats forward" statement.
- The Borkar–Varaiya trap characterises when hindsight planning can fail. It fails when the actions chosen by the re-planned policy stop generating information about the parameter directions that matter for the true optimum.

### Gaps
- I could not fetch Lai–Wei's full text, so the exact theorem numbering, the moment condition (α > 2) and o vs O are not verified.
- The Borkar–Varaiya, Feldbaum and Åström–Wittenmark details are from memory [M].

---

## 2. Online LQR regret: which settings give log T vs √T, and why

### Takeaway
With both A and B unknown, Θ̃(√T) regret is optimal (Simchowitz & Foster 2020). Polylog(T) is impossible because the certainty-equivalent input u = Kx is collinear with the state, so exploration must be injected at a cost. If only A is unknown, or only B is unknown with a non-degenerate optimal gain, then process noise alone (passed through the fixed control law) excites every unknown direction, and logarithmic regret is attainable (Cassel, Cohen & Koren 2020). Partially observed LQG also admits polylog(T) regret under the assumptions of Lale et al. (2020).

### Cited Findings
- **Abbasi-Yadkori & Szepesvári (2011)**, "Regret Bounds for the Adaptive Control of Linear Quadratic Systems," *Proc. COLT 2011*, PMLR 19:1–26. The algorithm is optimism in the face of uncertainty (OFU) over a high-probability confidence set, with regret Õ(√T). The paper describes this as the first regret bound for LQ adaptive control. — [PMLR](https://proceedings.mlr.press/v19/abbasi-yadkori11a.html) [V] **[M]** The algorithm is not computationally efficient (the optimistic step is non-convex), and the bound has exponential dependence on dimension in some constants.
- **Dean, Mania, Matni, Recht & Tu (2018)**, "Regret bounds for robust adaptive control of the linear quadratic regulator," NeurIPS 2018, arXiv:1805.09388. Efficient (SDP-based) method with Õ(T^{2/3}) regret. [M: arXiv ID not verified]
- **Cohen, Koren & Mansour (2019)**, "Learning Linear-Quadratic Regulators Efficiently with only √T Regret," ICML 2019, arXiv:1902.06223. First polynomial-time algorithm with Õ(√T) regret, using an SDP relaxation and optimism. [M: arXiv ID not verified]
- **Mania, Tu & Recht (2019)**, "Certainty Equivalence is Efficient for Linear Quadratic Control," NeurIPS 2019, arXiv:1902.07826. The CE controller's sub-optimality gap "enjoys a fast statistical rate, scaling as the square of the parameter error." This holds for fully observed LQR and for partially observed LQG (the first such LQG guarantee), and the analysis relies on perturbation bounds for discrete Riccati equations. — [arXiv:1902.07826](https://arxiv.org/abs/1902.07826) [V] (the NeurIPS 2019 venue is [M]; the arXiv page lists no venue). **[M]** The result is J(K̂) − J* ≤ C·ε² for ε = max(‖Â−A‖, ‖B̂−B‖) ≤ ε₀, a local radius that depends on the instance. Combined with ε-greedy exploration noise, this gives Õ(√T) regret.
- **Simchowitz & Foster (2020)**, "Naive Exploration is Optimal for Online LQR," ICML 2020, arXiv:2001.09576. They show "the optimal regret scales as" Θ̃(√(d_u² d_x T)). Their lower bounds "rule out the possibility of a poly(log T)-regret algorithm," which had been conjectured. The upper bound is attained by "a simple variant of certainty equivalent control" with exploratory noise, and holds for any stabilizable instance. — [arXiv:2001.09576](https://arxiv.org/abs/2001.09576) [V] (ICML 2020 venue [M]).
- **Cassel, Cohen & Koren (2020)**, "Logarithmic Regret for Learning Linear Quadratic Regulators Efficiently," ICML 2020, arXiv:2002.08095. Efficient algorithms with regret "(poly)logarithmic" in T in two cases: (i) only A is unknown (B known), and (ii) only B is unknown (A known) and the optimal policy satisfies "a certain non-degeneracy condition." They also give a lower bound showing that when that condition is violated, "square root regret is unavoidable." — [arXiv:2002.08095](https://arxiv.org/abs/2002.08095) [V] **[M]** I believe the non-degeneracy condition is K* K*ᵀ ⪰ μ I for some μ > 0 (the optimal gain has full row rank). Verify in the paper.
- **Lale, Azizzadenesheli, Hassibi & Anandkumar (2020)**, "Logarithmic Regret Bound in Partially Observable Linear Dynamical Systems," arXiv:2003.11227 (NeurIPS 2020 [M]). The algorithm AdaptOn periodically re-solves a linear regression for the dynamics (closed-loop system identification) and updates the controller by online gradient descent. It "achieves a regret upper bound of polylog(T)" and is presented as the first polylog-regret method for unknown partially observable LDS, including LQG. — [arXiv:2003.11227](https://arxiv.org/abs/2003.11227) [V] **[M]** I believe the explicit rate is O(log² T). It depends on strong convexity of the cost in the controller parameters and on persistent excitation from process and measurement noise. Verify in the paper.
- **[M]** Ziemann & Sandberg (2022), "Regret lower bounds for learning linear quadratic Gaussian systems" (arXiv:2201.01680, ID unverified), give √T-type lower bounds for some LQG classes. This suggests that Lale et al.'s polylog bound depends on their specific assumptions and benchmark class.

### Why log T vs √T (mechanism)
- **[D, heuristic, matching the intuition in Simchowitz & Foster / Cassel et al.]**
  - The regressor for estimating θ = [A B] is z_t = (x_t, u_t). Under CE control u_t = K̂ x_t, every z_t lies in the d_x-dimensional subspace {(x, K̂x)}, so the information matrix Σ z_t z_tᵀ is rank-deficient in the d_u directions orthogonal to that graph.
  - Exciting those directions requires added input noise of variance σ². That costs about σ² per step (Ω(σ² T) in total) and buys information of about σ² t in the missing directions.
  - By Mania et al., the control cost of an estimation error ε is about ε², which is about 1/(σ² t). Balancing σ² T against Σ_t 1/(σ² t) ≈ (log T)/σ² gives σ² ≈ T^{−1/2} up to logs, and so √T regret.
  - The lower bound shows this trade-off cannot be avoided when both A and B are unknown.
- **[D]** **A unknown, B known:** the regressor is x_t alone. Process noise w_t with covariance W ≻ 0 makes λ_min(Σ x_t x_tᵀ) ≳ λ_min(W) t at no cost. The error is then Õ(1/t), the per-step cost gap Õ(1/t), and the regret Σ 1/t = O(log T).
- **[D]** **B unknown, A known:** the regressor is u_t = K x_t, whose covariance is K Σ_x Kᵀ. This grows linearly in t iff K has full row rank, which is the non-degeneracy condition. If it fails, some columns of B are never excited and the √T trade-off returns.
- **Identifiability via the control input** is therefore the dividing line. Logarithmic regret is achievable when free exogenous noise passed through the closed loop identifies every parameter that matters. Square-root regret is required when identification needs the input to leave the CE policy's subspace.

### Elliptical potential lemma and self-normalized bound (Abbasi-Yadkori, Pál & Szepesvári 2011)
- Reference: Abbasi-Yadkori, Y., Pál, D. & Szepesvári, Cs. (2011), "Improved Algorithms for Linear Stochastic Bandits," *Advances in Neural Information Processing Systems 24* (NeurIPS 2011), pp. 2312–2320. — [NeurIPS](https://papers.nips.cc/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html) [V]
- **Theorem 1 (Self-Normalized Bound for Vector-Valued Martingales)**, verified from the PDF text:
  - Setup: {F_t} is a filtration. η_t is F_t-measurable and conditionally R-sub-Gaussian: E[e^{λη_t} | F_{t−1}] ≤ exp(λ²R²/2) for all λ ∈ ℝ. X_t ∈ ℝ^d is F_{t−1}-measurable, and V ≻ 0. Define V̄_t = V + Σ_{s=1}^t X_s X_sᵀ and S_t = Σ_{s=1}^t η_s X_s.
  - Statement: for any δ > 0, with probability ≥ 1−δ, for all t ≥ 0,
    ‖S_t‖²_{V̄_t^{-1}} ≤ 2R² log( det(V̄_t)^{1/2} det(V)^{−1/2} / δ ).
  - [NeurIPS PDF](https://papers.nips.cc/paper_files/paper/2011/file/e1d5be1c7f2f456670de3d53c7b54f4a-Paper.pdf) [V]
- **Theorem 2 (Confidence Ellipsoid)**, verified:
  - Setup: same as Theorem 1, with V = λI, Y_t = ⟨X_t, θ*⟩ + η_t and ‖θ*‖₂ ≤ S. The ridge estimate is θ̂_t = (X_{1:t}ᵀX_{1:t} + λI)^{−1} X_{1:t}ᵀ Y_{1:t}.
  - Statement: with probability ≥ 1−δ, for all t ≥ 0,
    ‖θ̂_t − θ*‖_{V̄_t} ≤ R √(2 log(det(V̄_t)^{1/2} det(λI)^{−1/2}/δ)) + λ^{1/2} S.
  - If also ‖X_t‖₂ ≤ L, the radius becomes R √(d log((1 + tL²/λ)/δ)) + λ^{1/2} S.
  - [NeurIPS PDF](https://papers.nips.cc/paper_files/paper/2011/file/e1d5be1c7f2f456670de3d53c7b54f4a-Paper.pdf) [V]
- **Lemma 11 (elliptical potential)**, in the appendix and quoted via secondary sources [S]:
  - Setup: X_1, X_2, … ∈ ℝ^d with ‖X_t‖₂ ≤ L, V ≻ 0 and V̄_t = V + Σ_{s≤t} X_s X_sᵀ.
  - Main statement: Σ_{t=1}^n min(1, ‖X_t‖²_{V̄_{t−1}^{-1}}) ≤ 2 log(det V̄_n / det V).
  - For V = λI this is ≤ 2d log(1 + nL²/(dλ)).
  - **[M]** The un-truncated version is Σ_t ‖X_t‖²_{V̄_{t−1}^{-1}} ≤ 2 log(det V̄_n / det V) when λ_min(V) ≥ max(1, L²).
  - Proof idea: min(1, u) ≤ 2 log(1+u), together with det V̄_t = det V̄_{t−1}(1 + ‖X_t‖²_{V̄_{t−1}^{-1}}), telescoped. — [arXiv:2002.07066](https://arxiv.org/pdf/2002.07066), [arXiv:2010.10182](https://arxiv.org/pdf/2010.10182.pdf)
  - Secondary sources differ slightly in indexing, so check the APS appendix before quoting.
- **[D]** Use in a hindsight-planning proof:
  - The per-step prediction error of the re-estimated model is |⟨X_t, θ̂_{t−1} − θ*⟩| ≤ β_{t−1} ‖X_t‖_{V̄_{t−1}^{-1}}.
  - By Cauchy–Schwarz and the lemma, Σ_t min(1, error_t) ≤ β_n √(n · 2d log(1 + nL²/(dλ))) = Õ(d√n).
  - The sum of squared errors is Õ(d² log² n), or Õ(d log n · β²).
  - So if the planner's loss is quadratic in model error (Mania et al.), hindsight re-planning incurs only polylog(n) cumulative excess cost from the squared errors, provided the regressors are exogenously excited. Otherwise the cost is Õ(√n).

### Inferences
- **[D]** Mapping to hindsight planning:
  - An **additive hidden disturbance or offset** (y = f(x, u) + d) is identified from residuals regardless of the action taken. This is like the "A unknown" case, so CE / hindsight re-planning should achieve O(log T) excess cost, while a forward planner that ignores d incurs Θ(‖d‖² T).
  - A **multiplicative / gain parameter** on the action (y = f(x) + b·u, b unknown) is identified only through the action. This is like the "B unknown" case: log T if the planned actions are non-degenerate, √T (with deliberate exploration) or even non-convergence (Borkar–Varaiya) otherwise.

### Gaps
- I did not verify the Cassel et al. non-degeneracy condition or the explicit Lale et al. rate (log² T) against the papers.
- The arXiv IDs for Cohen–Koren–Mansour, Dean et al. and Ziemann–Sandberg are from memory.
- The ICML/NeurIPS venues for Mania et al., Simchowitz–Foster and Lale et al. are from memory; the arXiv pages do not list them.

---

## 3. Disturbance observers, offset-free MPC and the internal model principle

### Takeaway
Augmenting the plant model with integrating (random-walk or constant) disturbance states, estimating them with a Kalman filter or observer, and shifting the MPC target by the estimate removes steady-state offset for constant disturbances and constant model mismatch. Two conditions must hold: the augmented system is detectable, and the number of disturbance states equals the number of measured (tracked) outputs. This is the control-theoretic version of "hindsight planning": infer the hidden disturbance from the residual, then re-plan.

### Cited Findings
- Maeder, U., Borrelli, F. & Morari, M. (2009), "Linear offset-free Model Predictive Control," *Automatica* 45(10):2214–2222, doi:10.1016/j.automatica.2009.06.005. The abstract says the paper gives compact and intuitive conditions for offset-free MPC, derived from the internal model principle. The plant model is augmented by only as many states as there are tracked variables, and an algorithm guaranteeing offset-free tracking is given. — [MaRDI portal](https://portal.mardi4nfdi.de/wiki/Publication:1049078) [S]
- Pannocchia, G. & Rawlings, J. B. (2003), "Disturbance models for offset-free model-predictive control," *AIChE Journal* 49(2):426–437. It covers square and non-square, open-loop stable, integrating and unstable systems. Lemma 1 gives the detectability condition for the augmented system, Corollary 1 the maximum number of integrated disturbances, and Theorem 1 says that adding integrated disturbances so that the augmented system is detectable guarantees offset-free control. — [UniPi repository](https://arpi.unipi.it/handle/11568/77644) [S]
- **[M]** The standard statements, consistent with both papers above. Augmented model: x⁺ = Ax + Bu + B_d d, d⁺ = d, y = Cx + C_d d, with n_d disturbance states and p outputs.
  - (i) The augmented pair is detectable iff (A, C) is detectable and rank [[A − I, B_d], [C, C_d]] = n + n_d.
  - (ii) This requires n_d ≤ p.
  - (iii) If n_d = p, the observer/Kalman filter is stable, and the closed loop reaches a steady state, then the controlled outputs track the setpoint with zero offset. This holds despite constant unmeasured disturbances and plant–model mismatch, under constraints that are inactive at steady state.
  - Verify the lemma/corollary numbering against the PDFs.
- Muske, K. R. & Badgwell, T. A. (2002), "Disturbance modeling for offset-free linear model predictive control," *Journal of Process Control* 12(5):617–632. [M: volume/pages from memory] It generalises the industrial "DMC bias" (output disturbance) model. **[M]** DMC-style constant output-bias correction, b = y_meas − y_pred, is the special case C_d = I, B_d = 0 with a deadbeat estimator (filter gain = I).
- A later citation notes that Rajamani et al. (2009) showed the equivalence of different disturbance models. — [search summary of offset-free MPC review](https://arxiv.org/pdf/2012.02753) [S]
- **Internal model principle [M]:** Francis, B. A. & Wonham, W. M. (1976), "The internal model principle of control theory," *Automatica* 12(5):457–465. Robust asymptotic regulation requires the controller to contain a model of the exogenous signal generator; for constant disturbances, an integrator.
- **Disturbance observers [M]:**
  - Ohnishi, K. (1987), "A new servo method in mechatronics," *Trans. Japanese Society of Electrical Engineers* 107-D, 83–86. This is the original DOB: estimate lumped disturbance = (nominal inverse model applied to output) − input, low-pass filtered by a Q-filter.
  - Chen, W.-H., Yang, J., Guo, L. & Li, S. (2016), "Disturbance-observer-based control and related methods—An overview," *IEEE Trans. Industrial Electronics* 63(2):1083–1095.

### Inferences
- **[D]** The offset-free result is an exact, assumption-explicit theorem for "hindsight beats forward planning." Take an MPC with the nominal model and no disturbance estimate (forward planning), facing constant disturbance d ≠ 0 or constant mismatch. It generically leaves a non-zero steady-state offset, so cost grows linearly in T. The disturbance-augmented, filter-based MPC (hindsight planning) gives zero offset when n_d = p and the augmented system is detectable.
- **[D]** The detectability rank condition is the "identifiability" analogue of Section 2. A disturbance direction that does not appear in the outputs cannot be inferred from residuals. Hindsight re-planning cannot remove its effect, so it gives no advantage in that direction.

### Gaps
- I did not access the full text of Pannocchia & Rawlings, Maeder et al. or Muske & Badgwell. The rank condition and n_d = p are stated from domain knowledge [M].
- I did not verify the Ohnishi and Chen et al. bibliographic details this session.

---

## 4. Kalman filter for a random-walk disturbance: gain → 1 (residual reuse) vs gain → 0 (constant disturbance / RLS)

### Takeaway
For the local level model y_t = μ_t + ε_t, μ_{t+1} = μ_t + η_t with signal-to-noise ratio q = σ_η²/σ_ε², the steady-state Kalman gain is K = (−q + √(q² + 4q))/2. It increases monotonically from 0 (at q = 0) to 1 (as q → ∞). Hence "add the last residual" (K = 1, the one-step forecast equals the last observation) is the high-SNR limit. For a constant disturbance (q = 0), the gain decays like 1/t, which is recursive least squares (the running mean).

### Cited Findings
- A search-result summary of state-space teaching material reports the steady-state Kalman gain of the local level model as K = (−q + √(q² + 4q))/2, solving K² + qK − q = 0, with steady-state predicted variance (in units of σ_ε²) P̄ = K + q and K = P̄/(P̄+1). It notes that convergence to steady state is slowest for small q. — search summary citing [Aarhus 2016 state-space course slides](https://econ.au.dk/fileadmin/user_upload/2016Aarhus_Day1.pdf) and [Univ. Antwerp notes](https://medialibrary.uantwerpen.be/files/7228/c5185dc4-4483-43f4-924f-5f9ca0978070.pdf) [S: formula attributed via search summary; not read in the primary textbook]
- The same sources give the reduced-form ARIMA(0,1,1) moving-average parameter θ = ½(√(q² + 4q) − 2 − q), and cite Durbin & Koopman (2012) for the derivations. — [Univ. Antwerp notes](https://medialibrary.uantwerpen.be/files/7228/c5185dc4-4483-43f4-924f-5f9ca0978070.pdf) [S]
- Textbook references [M]:
  - Durbin, J. & Koopman, S. J. (2012), *Time Series Analysis by State Space Methods*, 2nd ed., Oxford University Press, Ch. 2 (local level model; steady state, §2.11 or thereabouts).
  - Harvey, A. C. (1989), *Forecasting, Structural Time Series Models and the Kalman Filter*, Cambridge University Press.
  - Muth, J. F. (1960), "Optimal properties of exponentially weighted forecasts," *JASA* 55:299–306, which shows the exponentially weighted moving average (EWMA) is the optimal forecast for random walk plus noise.

### Derivation [D] (independently derived here; it agrees with the [S] formula above)
- **Setup:** let p_t = P_{t|t−1}/σ_ε² (normalised predicted variance). The Kalman recursion gives:
  - K_t = p_t/(p_t + 1),
  - p_{t+1} = p_t(1 − K_t) + q = p_t/(p_t + 1) + q.
- **Fixed point:** p = p/(p+1) + q, so p² − qp − q = 0, and p̄ = (q + √(q² + 4q))/2. Note that the formula "(q + √(q²+4q))/2" is the steady-state *variance*, not the gain.
- **Gain:** since p̄² = q(p̄ + 1), we get K̄ = p̄/(p̄+1) = q/p̄. Rationalising gives **K̄ = (√(q² + 4q) − q)/2**, equivalently K̄² + qK̄ − q = 0.
- **Check:** q = 1 gives K̄ ≈ 0.618 and p̄ ≈ 1.618.
- **Limits:**
  - q → ∞: √(q² + 4q) = q + 2 − 2/q + O(q⁻²), so K̄ = 1 − 1/q + O(q⁻²) → 1.
  - q → 0: K̄ = √q − q/2 + O(q^{3/2}) → 0.
  - K̄ is strictly increasing in q.
- **Forecast form:** the filter for the one-step forecast is μ̂_{t+1|t} = μ̂_{t|t−1} + K̄(y_t − μ̂_{t|t−1}), i.e. an EWMA with smoothing weight K̄.
  - With K̄ = 1: μ̂_{t+1|t} = y_t ("re-use the last residual / last observed disturbance in full").
  - For a disturbance estimate inside a model, d̂_{t+1} = d̂_t + K̄ r_t, with residual r_t = y_t − ŷ_t(d̂_t). K̄ = 1 sets d̂_{t+1} equal to the full last-observed disturbance.
- **Constant disturbance (q = 0):** p_{t+1} = p_t/(p_t+1), so 1/p_{t+1} = 1/p_t + 1, giving p_t = p_1/(1 + (t−1)p_1) and K_t = p_t/(p_t + 1) = p_1/(1 + t p_1) ~ 1/t.
  - With a diffuse prior (p_1 → ∞), K_t = 1/t and μ̂ is the sample mean of y_1..y_t: recursive least squares with no forgetting.
  - The estimation error variance is σ_ε²/t, which is the Lai–Wei / Section 2 "1/t" rate in the scalar case.
- **[D]** Implication for hindsight planning:
  - "Residual reuse" (K = 1) is optimal only when disturbance innovation dominates measurement noise (q ≫ 1).
  - For persistent (near-constant) disturbances observed through noisy outcomes (q ≪ 1), full residual reuse inflates variance. The steady-state excess mean-squared prediction error of K = 1 vs K̄ can be computed in closed form from the ARIMA(0,1,1) representation [not computed here].
  - The optimal hindsight estimator averages over the history with weight about √q per step for small q, or 1/t when q = 0.

### Gaps
- I did not read the exact equation numbers in Durbin & Koopman or Harvey. The [S] formula comes from search-snippet summaries of lecture notes that cite those books, and it agrees with the independent derivation [D].

---

## 5. Gain (multiplicative) vs additive faults in fault-tolerant control

### Takeaway
The fault-diagnosis literature (Blanke et al.) classifies faults as additive (an exogenous signal entering the dynamics or measurements) or multiplicative (a change in system parameters, e.g. a scaling of the input matrix B). Additive faults can be estimated with standard linear observer tools. Multiplicative faults enter bilinearly with state and input, which makes estimation harder and dependent on excitation. This mirrors the A-known/B-unknown distinction in Section 2.

### Cited Findings
- Blanke, M., Kinnaert, M., Lunze, J. & Staroswiecki, M., *Diagnosis and Fault-Tolerant Control*, Springer, 3rd ed. 2016 (ISBN 978-3-662-47942-1). Section 1.2.1 ("Faults") is the likely location of the formal definitions. — [Springer](https://www.springer.com/gp/book/9783662479421) [S]
- A Springer *Encyclopedia of Systems and Control* entry on fault detection says faults "are defined and classified as additive or multiplicative," with Blanke et al. as its main reference. — [Springer reference entry](https://link.springer.com/referenceworkentry/10.1007/978-3-030-44184-5_223) [S]
- A paper on fault estimation (arXiv:2011.13730) says standard system-theoretic tools can estimate additive faults. When the fault is multiplicative, estimation is "a more challenging task due to the nonlinear impact of the fault." — [arXiv:2011.13730](https://arxiv.org/pdf/2011.13730) [S]
- IFAC World Congress 2020 (LPV systems): multiplicative actuator faults are modelled as B(θ_k) → B(θ_k)·G with G diagonal and entries in [0, 1] (loss of effectiveness). — [IFAC 2020 preprint](https://ifatwww.et.uni-magdeburg.de/ifac2020/media/pdfs/0055.pdf) [S]

### Inferences
- **[D]** Formal correspondence:
  - Additive fault f_t in x⁺ = Ax + Bu + Ef corresponds to an additive disturbance state. It is observable from residuals whenever the rank condition of Section 3 holds, independent of u, so hindsight estimation works without probing.
  - Multiplicative actuator fault B → BG corresponds to "B unknown." The residual carries information about G only in proportion to u, so the information matrix grows like Σ u_t u_tᵀ.
  - A controller that, under the faulty model, drives u → 0 or keeps u in a low-rank subspace cannot identify G. This is the Borkar–Varaiya / Cassel non-degeneracy failure.
- **[D]** For a hindsight-planning paper, the natural theorem structure is:
  - (a) Additive hidden disturbance: hindsight re-planning has O(log T) or zero-offset excess cost, while forward planning has Θ(T).
  - (b) Multiplicative hidden gain: hindsight re-planning has O(log T) excess cost if planned actions are persistently exciting in the affected directions; otherwise excess cost is Θ̃(√T) with forced exploration (Simchowitz–Foster / Cassel lower bounds). Hindsight can fail outright (inconsistent estimates) without exploration.

### Gaps
- I did not obtain the book's own definitions or the arXiv:2011.13730 title and authors (only snippet text).
- I found no source giving regret rates specifically for multiplicative-fault estimation. The mapping to the Cassel et al. "B unknown" regime is my inference [D].
