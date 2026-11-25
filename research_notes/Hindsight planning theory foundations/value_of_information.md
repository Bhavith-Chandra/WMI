# Decision- and information-theoretic foundations: why conditioning on observed outcomes ("hindsight") beats open-loop decisions

Scope note: these notes give citable statements, assumptions, and rates. Each item is labelled as one of three kinds.
- **[PROVED-in-source]**: a theorem in the cited paper, checked against the paper or a reliable restatement.
- **[STANDARD]**: textbook-level, with a short self-contained derivation given here that a paper can reproduce.
- **[FOLKLORE / UNVERIFIED]**: commonly asserted, but I could not confirm it against a primary source in this session.

I do not give theorem or proposition numbers unless I saw them in the source.

Standing model used throughout ("persistent hidden parameter"):
- θ ~ prior P₀ is drawn once and stays fixed.
- Transitions follow s_{t+1} = f(s_t, a_t, θ, ε_t), with ε_t i.i.d. and independent of θ.
- An *outcome-ignoring* (open-loop, or prior-predictive) planner chooses a_t as a function of information available before any outcome. That is the prior and, depending on the variant, possibly the current state but not the past transitions used to infer θ.
- A *hindsight / adaptive* planner chooses a_t as a function of the history H_t = (s_1, a_1, …, s_t).

---

## 1. Blackwell's comparison of experiments and its sequential consequences

### Takeaway
For a single Bayesian decision maker with a correct prior and model, extra information is never harmful. Formally, the value of a Bayes-optimal decision cannot decrease when one experiment is replaced by a Blackwell-more-informative one, i.e. one of which the original is a garbling. Blackwell's theorem also gives the converse: "better for every decision problem" is equivalent to "the worse experiment is a garbling of the better one".

In the sequential setting the cleanest citable argument is simpler than Blackwell. The set of history-dependent policies contains the set of open-loop policies, so the Bayes-optimal adaptive value is at least the optimal open-loop value. This is a feasibility (inclusion) argument.

The guarantee fails when the agent is not a single correctly specified Bayesian:
- in games, more information can hurt a player (Hirshleifer 1971; Bassan, Gossner, Scarsini & Zamir 2003);
- under misspecification, the posterior can concentrate on bad models (Grünwald & van Ommen 2017; Berk–Nash equilibrium, Esponda & Pouzo 2016).

### Cited Findings
- **Blackwell (1951).** D. Blackwell, "Comparison of Experiments," *Proc. Second Berkeley Symposium on Mathematical Statistics and Probability*, 1951, pp. 93–102. The Berkeley proceedings page range is from memory and should be checked. The paper builds on Bohnenblust–Shapley–Sherman: experiment E is "more informative" than F if every risk function attainable with F is attainable with E. — [Project Euclid record of the 1953 companion paper, which describes the 1951 setup](https://projecteuclid.org/euclid.aoms/1177729032); [Wikipedia: Blackwell's informativeness theorem](https://www.wikipedia.org/wiki/Blackwell%27s_informativeness_theorem)
- **Blackwell (1953).** D. Blackwell, "Equivalent Comparisons of Experiments," *Annals of Mathematical Statistics* 24(2): 265–272, 1953, DOI 10.1214/aoms/1177729032. One listing gives 267–272.
  - Sherman and Stein had shown, for finitely many outcomes, that the 1951 criterion is equivalent to the Bohnenblust–Shapley–Sherman one. The 1953 paper gives a new proof and removes the finite-outcome restriction.
  - It also introduces weaker comparisons in "k-decision problems".
  - [Project Euclid](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-24/issue-2/Equivalent-Comparisons-of-Experiments/10.1214/aoms/1177729032.short)
- **[PROVED-in-source] Standard modern statement (Blackwell's informativeness theorem).** For experiments P, Q on a finite state space, the following are equivalent:
  - (i) Q is a garbling of P: there is a Markov kernel (stochastic matrix) M with Q = P·M;
  - (ii) for every finite decision problem and every prior, the Bayes value under P is at least the Bayes value under Q;
  - (iii) P is "sufficient for" Q in the sense of the distribution of posteriors (a mean-preserving spread / convex order).

  The direction (i) ⇒ (ii) is what a paper needs. If the planner sees a finer signal, the Bayes-optimal expected utility is weakly higher. — [Wikipedia: Blackwell's informativeness theorem](https://www.wikipedia.org/wiki/Blackwell%27s_informativeness_theorem); [QuantEcon lecture on Blackwell's theorem](https://python.quantecon.org/blackwell_kihlstrom.html)
- **Games: value of information can be negative.**
  - Bassan, Gossner, Scarsini & Zamir, "Positive value of information in games," *International Journal of Game Theory* 32(1): 17–31, 2003.
    - Main result: if the extended game has a unique Pareto payoff profile, that profile is a Nash payoff profile. It also dominates every Nash payoff profile under any coarser information structure. Blackwell's theorem is the one-agent special case.
    - Their framing: "in one-agent contexts the value of information is always positive, but in strategic situations this is not always the case."
    - [EconPapers](https://econpapers.repec.org/RePEc:spr:jogath:v:32:y:2003:i:1:p:17-31); [HUJI CRIS](https://cris.huji.ac.il/en/publications/positive-value-of-information-in-games/)
  - J. Hirshleifer, "The Private and Social Value of Information and the Reward to Inventive Activity," *American Economic Review* 61(4): 561–574, 1971. The paper distinguishes foreknowledge from discovery and shows that private and social value of information diverge. It is the source of the "Hirshleifer effect": public early information can destroy risk-sharing and lower welfare. — [Duke-hosted PDF](https://people.duke.edu/~qc2/BA532/1971%20AER%20Hirshleifer.pdf); [EconBiz](https://econbiz.de/Record/the-private-and-social-value-of-information-and-the-reward-to-inventive-activity-hirshleifer-jack/10002863722)
- **Misspecification breaks "more data helps".**
  - Grünwald & van Ommen, "Inconsistency of Bayesian Inference for Misspecified Linear Models, and a Proposal for Repairing It," *Bayesian Analysis* 12(4): 1069–1103, 2017, arXiv:1412.3730.
    - Setting: heteroskedastic data fitted with a homoskedastic linear model.
    - Result: as n grows, the posterior puts mass on worse and worse, higher-dimensional models ("hypercompression").
    - Proposed fix: SafeBayes, which uses a learned learning-rate exponent η < 1 on the likelihood.
    - [arXiv](https://www.arxiv.org/abs/1412.3730); [Project Euclid](https://projecteuclid.org/euclid.ba/1510974325)
  - Esponda & Pouzo, "Berk–Nash Equilibrium: A Framework for Modeling Agents With Misspecified Models," *Econometrica* 84(3): 1093–1130, 2016, arXiv:1411.1152, DOI 10.3982/ECTA12609.
    - Agents with misspecified models converge to KL-best-fit beliefs, and those beliefs depend on their own actions.
    - Nash and self-confirming equilibria are the correctly specified special cases.
    - So a learning agent can settle on a suboptimal action permanently.
    - [Econometrica](https://www.econometricsociety.org/doi/10.3982/ECTA12609); [arXiv](https://arxiv.org/abs/1411.1152v4)

### Inferences
- **[STANDARD] Inclusion argument (the citable core for the paper).**
  - Let Π_OL be the set of open-loop policies. These are maps from (t, information available before outcomes) to actions; by convention a constant map for each t.
  - Let Π_H be the set of history-dependent (possibly randomized) policies. Then Π_OL ⊆ Π_H.
  - Hence V*_H := sup_{π∈Π_H} E_{θ~P₀} E^π_θ[Σ_t r_t] ≥ sup_{π∈Π_OL} (same) =: V*_OL.
  - This holds for any prior, any horizon, and any reward, including misspecified *environments*, as long as the expectation is taken under the same law the planner uses.
  - Blackwell is the stronger statement: it also orders two *different* signal structures, not only "signal vs no signal".
- **What Blackwell does NOT guarantee.** It does not guarantee that an *approximate* or heuristic adaptive planner beats a well-tuned open-loop planner. The guarantee is about the *optimum* over the larger class.
- A practical "hindsight" planner that infers θ with a misspecified model can be worse than open-loop. The Grünwald–van Ommen and Esponda–Pouzo results are concrete citations for this caveat.
- **Ordering of signals within the sequential problem.** Observing (s_t, a_t, s_{t+1}) is a Blackwell-finer experiment about θ than observing nothing, and than any garbling of it, such as summary statistics. So the Bayes value is monotone in what is retained from hindsight. This follows by applying (i) ⇒ (ii) to the posterior-state formulation (see §2).

### Gaps
- I did not retrieve the 1951 Berkeley paper's full text. The page range 93–102 is from memory.
- I did not retrieve Neyman (1991), "The positive value of information," *Games and Economic Behavior*. It is relevant to games where information is known to be acquired, but it is unverified here.

---

## 2. Value of information (Howard 1966) and VOI in sequential problems / POMDPs (convexity of the value function in the belief)

### Takeaway
- **Howard's EVPI.** EVPI = E[max_a U(a, θ)] − max_a E[U(a, θ)] ≥ 0 by Jensen's inequality, because max is convex.
- **Expected value of sample information.** EVSI = E_Y[max_a E[U | Y]] − max_a E[U] ≥ 0 by the tower rule.
- **EVSI = 0 condition.** EVSI is zero iff the prior-optimal action stays optimal under almost every posterior.
- **POMDPs.** The optimal finite-horizon value is a piecewise-linear convex function of the belief (Smallwood & Sondik 1973). Convexity in the belief is exactly the sequential version of "information has nonnegative value": a mean-preserving spread of beliefs, i.e. learning, cannot lower the expected value.

### Cited Findings
- **Howard (1966).** R. A. Howard, "Information Value Theory," *IEEE Transactions on Systems Science and Cybernetics* SSC-2(1): 22–26, 1966. Pages are cited inconsistently as 22–26 and 22–34; check IEEE Xplore.
  - The paper founds information value theory on the joint probabilistic and economic factors that affect decisions.
  - It argues that Shannon-information approaches to such problems "have, in the large, come to grief". VOI is decision-relative, not entropy-relative.
  - [Scilit](https://www.scilit.net/publications/6df64dca9e2dcb60dd51e3a0eb6a9aa1); [Synapse listing](https://synapsesocial.com/papers/W2001863269)
- **Smallwood & Sondik (1973).** "The Optimal Control of Partially Observable Markov Processes over a Finite Horizon," *Operations Research* 21(5): 1071–1088. With finitely many control intervals remaining, the optimal payoff is a piecewise-linear, convex function of the current state probabilities (the belief). The dynamic-programming update preserves this structure. — [INFORMS](https://pubsonline.informs.org/doi/fpi/10.1287/opre.21.5.1071); [IDEAS/RePEc](https://ideas.repec.org/a/inm/oropre/v21y1973i5p1071-1088.html)

### Inferences
- **[STANDARD] EVPI/EVSI nonnegativity and the exact zero condition.**
  - Write V(b) = max_a E_{θ~b}[U(a, θ)]. It is a maximum of linear functionals of b, hence convex in b.
  - For any signal Y with posterior b_Y, E[b_Y] = b (martingale property). By Jensen, E[V(b_Y)] ≥ V(E b_Y) = V(b).
  - So EVSI = E[V(b_Y)] − V(b) ≥ 0.
  - Equality holds iff V is affine on the support of b_Y, which happens iff a single action a₀ is optimal for almost every realized posterior b_Y. Typically a₀ is the prior-optimal action.
  - This is the exact zero-advantage condition for one decision.
- **[STANDARD] The Bayes-adaptive / persistent-parameter model as a POMDP.**
  - Augment the state to (s, θ), with θ unobserved and static. The belief over θ is a sufficient statistic.
  - Smallwood–Sondik convexity then applies to the finite-θ case: V_t(s, b) is convex in b.
  - Each observed transition produces a posterior b' with E[b' | b, s, a] = b. Applying Jensen stage by stage gives the result that conditioning on the observed transition (choosing the next action as a function of b') weakly beats freezing b.
  - For general (non-finite) θ, convexity of V in b follows the same way, as a supremum of linear functionals of b (sup over policies of expected return, which is linear in the prior). Piecewise linearity requires finite state and observation spaces.
- **[STANDARD] Strict advantage per step.** For an outcome-ignoring planner, the *per-step* loss relative to a clairvoyant who knows θ is the one-step EVPI:

  Δ_OL = E_θ[max_a μ(a, θ)] − max_a E_θ[μ(a, θ)],

  where μ(a, θ) is the expected one-step reward. It is strictly positive whenever no single action is optimal for P₀-almost every θ.

### Gaps
- I did not confirm the exact page range of Howard (1966).
- Raiffa & Schlaifer (1961), *Applied Statistical Decision Theory*, is usually credited with EVSI/EVPI terminology. I did not verify this here.

---

## 3. Bayes-adaptive MDPs (Duff 2002; Guez et al. BAMCP), Bayes-optimal exploration, and the Gittins / dual-control tradeoff

### Takeaway
The Bayes-adaptive MDP (BAMDP) is the precise object a "hindsight" planner approximates:
- its state is the hyperstate (s, posterior over θ);
- its optimal policy is Bayes-optimal and automatically trades off exploitation and information gathering (dual control);
- by construction it dominates every policy that ignores outcomes.

BAMCP (Guez, Silver & Dayan) is a Monte-Carlo tree search approximation that is proved to converge to the Bayes-optimal policy. In the special case of independent-arm bandits with geometric discounting, the Bayes-optimal policy is an index policy (Gittins). Outside that case, exact Bayes-optimality is generally intractable.

### Cited Findings
- **Duff (2002).** M. O. Duff, *Optimal Learning: Computational Procedures for Bayes-Adaptive Markov Decision Processes*, PhD thesis, University of Massachusetts Amherst, 2002. The approach keeps the full decision model, including a Bayesian model of parameter uncertainty, as an MDP over hyperstates. The thesis gives Monte-Carlo and gradient-based methods for finite-state stochastic-automaton policies. — [ScholarWorks@UMass](https://scholarworks.umass.edu/dissertations/AAI3039353); [catalogue record](https://openaccess.library.uitm.edu.my/Record/ndltd-UMASS-oai-scholarworks.umass.edu-dissertations-3616/Details)
- **Earlier roots.** Duff's thesis is cited together with Martin (1967), *Bayesian Decision Problems and Markov Chains*, as the Bayesian formulation of decisions in unknown MDPs. — [arXiv:1402.1958](https://arxiv.org/pdf/1402.1958)
- **Guez, Silver & Dayan, NIPS 2012.** "Efficient Bayes-Adaptive Reinforcement Learning using Sample-Based Search," *NIPS 25*, pp. 1025–1033, arXiv:1205.3109.
  - BAMCP = BA-UCT + root sampling + lazy sampling + rollout learning.
  - It avoids posterior updates inside simulations by sampling θ at the root.
  - The authors state that it converges to the Bayes-optimal policy, a consistency result that is asymptotic in the number of simulations.
  - [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2012/file/35051070e572e47d2c26c241ab88307f-Paper.pdf); [arXiv/ar5iv](https://ar5iv.arxiv.org/html/1205.3109); [poster](https://www.gatsby.ucl.ac.uk/~aguez/files/guez_nips2012POSTER.pdf)
- **JAIR 2013 extension.** Guez, Silver & Dayan, "Scalable and Efficient Bayes-Adaptive Reinforcement Learning Based on Monte-Carlo Tree Search," *JAIR* 48, 2013. — [JAIR](https://www.jair.org/index.php/jair/article/view/10853)
- **Bayes-adaptive POMDPs** (unknown model plus partial observability). Ross, Pineau, Chaib-draa & Kreitmann, *JMLR* 12, 2011. — [PDF](https://www.cs.mcgill.ca/~jpineau/files/sross-jmlr11.pdf)

### Inferences
- **[STANDARD] The BAMDP optimum dominates every outcome-ignoring planner.** This is the inclusion argument of §1. The gap equals the value of the information in the transitions, and it includes the *active* (dual-control) part: the optimal policy may take informative actions that the prior-optimal action would not.
- **Dual control.** Feldbaum's dual control (1960s) and the Gittins index are the classical names for the exploration–exploitation decomposition.
- **Gittins index citation.** J. C. Gittins, "Bandit processes and dynamic allocation indices," *JRSS-B* 41(2): 148–177, 1979. This is from memory and was not re-verified in this session. The Gittins index policy is Bayes-optimal only for independent arms with geometric discounting and an infinite horizon.
- **Passive vs active learning (useful structure for the paper).**
  - If the planner's actions do not affect how informative the transitions are about θ, the problem is *passive learning*: for example, the disturbance is observed regardless of the action, or (s_t, a_t, s_{t+1}) reveals the same statistic for all a.
  - In that case the posterior evolves exogenously, and the optimal adaptive policy is the myopic certainty-equivalent policy for the current posterior: act optimally for the posterior predictive. Most of the hindsight benefit is captured without exploration.
  - With active learning (the informativeness depends on the action), a greedy "condition on hindsight but never explore" planner can lock in on a wrong action, i.e. incomplete learning.
  - This last claim is folklore-level; a standard example is the two-armed bandit with one known arm.

### Gaps
- I did not retrieve the exact BAMCP convergence theorem statement or its assumptions, beyond the authors' stated claim. My understanding is that the guarantee is convergence in probability of the root action values as the number of simulations goes to infinity, for finite state and action spaces and a discrete horizon or discount. Check the NIPS paper's theorem before citing.

---

## 4. Information-theoretic regret: Russo–Van Roy information ratio; Clarke–Barron; Gaussian linear-regression log-det identity → O(d log T)

### Takeaway
Three tools turn "conditioning on hindsight" into quantitative guarantees:
- **Russo–Van Roy.** Thompson sampling has Bayesian regret ≤ √(Γ̄ · H(A*) · T), with Γ̄ ≤ K/2 for K-armed bandits, ≤ d/2 for linear bandits, and ≤ 1/2 under full information.
- **Clarke–Barron.** The cumulative Bayes log-loss redundancy equals I(θ; Data), which is asymptotically (d/2) log(n/(2πe)) + (1/2) log det I(θ) + log(1/w(θ)) + o(1).
- **Gaussian linear-model identity.** In the Gaussian linear model, I(θ; Y_{1:T} | X) = ½ Σ_t log(1 + x_tᵀ Σ_{t−1} x_t / σ²) = ½ log det(I + σ⁻² Σ₀^{1/2} XᵀX Σ₀^{1/2}) ≤ (d/2) log(1 + Tλ L²/(d σ²)).

  Since the per-step posterior predictive variance is bounded by a constant times log(1 + ·), the cumulative expected squared prediction error of the posterior-mean predictor is O(d log T).

### Cited Findings
- **Russo & Van Roy, JMLR 2016.** "An Information-Theoretic Analysis of Thompson Sampling," *JMLR* 17(68): 1–30, 2016, arXiv:1403.5341. — [JMLR](https://jmlr.org/papers/v17/14-087.html); [arXiv](https://arxiv.org/abs/1403.5341); [ar5iv](https://ar5iv.arxiv.org/html/1403.5341)
  - **Information ratio.** Γ_t := (E_t[R(Y_{t,A*}) − R(Y_{t,A_t})])² / I_t(A*; (A_t, Y_{t,A_t})). The numerator is the squared one-step expected regret under the posterior; the denominator is the mutual information between the optimal action and the new observation.
  - **Proposition 1 (as numbered in the paper).** If Γ_t ≤ Γ̄ almost surely for all t ≤ T, then E[Regret(T, π^TS)] ≤ √(Γ̄ · H(α₁) · T), where α₁ is the prior distribution of A*. H(α₁) = H(A*) ≤ log |A|.
  - **Bounds on Γ.** The proposition numbers below are as reported by my read of the ar5iv text and should be double-checked.
    - Prop. 3: K-armed bandit with rewards in [0,1], Γ ≤ |A|/2, which gives regret ≤ √(½ |A| T log|A|).
    - Prop. 4: full information, Γ ≤ 1/2.
    - Prop. 5: linear bandit in ℝᵈ, Γ ≤ d/2.
    - Prop. 6: combinatorial semi-bandit with |a| ≤ m and rewards in [−½, ½], Γ ≤ d/(2m²).
  - **Caveat.** The H(A*) dependence can blow up for many or continuous actions. Dong & Van Roy address this with rate-distortion arguments. — [arXiv:1805.11845](https://arxiv.org/pdf/1805.11845v2)
- **Clarke & Barron, IEEE TIT 1990.** "Information-Theoretic Asymptotics of Bayes Methods," *IEEE Trans. Inform. Theory* 36(3): 453–471, 1990. — [researchr bibtex](https://researchr.org/publication/ClarkeB90/bibtex)
  - **Assumptions.** A smooth d-dimensional parametric family (i.i.d. data) with positive definite Fisher information I(θ), a prior density w continuous and positive at θ, and regularity conditions.
  - **Result:**

    D(P_θⁿ ‖ M_n) = (d/2) log(n / (2πe)) + ½ log det I(θ) + log(1/w(θ)) + o(1),

    where M_n = ∫ P_θⁿ w(θ) dθ is the Bayes mixture.
  - **Cumulative log-loss interpretation.** D(P_θⁿ ‖ M_n) equals the expected cumulative excess log-loss of the sequential Bayes predictive over the true θ (chain rule). Averaged over θ ~ w, it equals I(Θ; Xⁿ).
  - The formula is confirmed via secondary restatements, not the original PDF. — [Krattenthaler & Slater, arXiv:quant-ph/9612043](https://arxiv.org/pdf/quant-ph/9612043); [arXiv:math/9705212](https://arxiv.org/pdf/math/9705212)
  - **Minimax version.** Over a compact K in the interior of the parameter space, the minimax redundancy is (d/2) log(n/(2πe)) + log ∫_K √det I(θ) dθ + o(1), achieved asymptotically by Jeffreys' prior. The source attributes this to Clarke & Barron 1994, "Jeffreys' prior is asymptotically least favorable under entropy risk," *J. Statist. Plann. Inference* 41: 37–60 (venue and pages from memory). — [arXiv:quant-ph/9612043](https://arxiv.org/pdf/quant-ph/9612043)
- **Srinivas, Krause, Kakade & Seeger.** "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design," ICML 2010, arXiv:0912.3995. Journal version: "Information-Theoretic Regret Bounds for Gaussian Process Optimization in the Bandit Setting," *IEEE Trans. Inform. Theory* 58(5), 2012, DOI 10.1109/TIT.2011.2182033.
  - The information-gain identity (stated as Lemma 5.3 in several secondary sources) is: for a GP with Gaussian noise variance σ², I(y_{1:T}; f) = ½ Σ_t log(1 + σ⁻² σ²_{t−1}(x_t)).
  - The regret bound is O*(√(T β_T γ_T)), where γ_T is the maximal information gain. For d-dimensional linear kernels, γ_T = O(d log T).
  - **Correction to the task prompt.** The identity has a factor ½: Σ_t log(1 + xᵀΣx/σ²) = 2 I(θ; data). That is what the prompt wrote; equivalently, I = ½ Σ log(1 + ·).
  - [arXiv](https://arxiv.org/abs/0912.3995); [ICML PDF](https://icml.cc/imls/conferences/2010/papers/422.pdf); [IEEE TIT version (EPFL)](https://infosciences.epfl.ch/record/177246/files/srinivas_ieeeit2012.pdf)
- **Kakade, Seeger & Foster, NIPS 2005.** "Worst-Case Bounds for Gaussian Process Models," NIPS 18, pp. 619–626. This is a worst-case (individual-sequence) online analysis of Bayesian GP regression under log loss: the cumulative log-loss regret of the Bayesian predictor against any fixed function is bounded by a ½ log det(I + cK)-type complexity term plus a norm term. — [NeurIPS](https://papers.nips.cc/paper/2005/hash/952c3ff98a6acdc36497d839e31aa57c-Abstract.html); [EPFL PDF](https://infoscience.epfl.ch/record/161315/files/nips05-gponline.pdf?version=1)
- **Seeger, Kakade & Foster, IEEE TIT 2008.** "Information Consistency of Nonparametric Gaussian Process Methods" derives consistency rates from a cumulative log-loss regret bound with main term n⁻¹ E[log|I + cK|]. — [PDF](https://sham.seas.harvard.edu/sites/projects.iq.harvard.edu/files/kakade/files/ieeeinf-2008.pdf)

### Inferences
**[STANDARD] Derivation of O(d log T) squared-error regret for Bayesian prediction of a persistent linear-Gaussian disturbance.** This is reproducible in a paper.

1. **Model.** y_t = x_tᵀθ + ε_t, with θ ~ N(0, Σ₀), ε_t ~ N(0, σ²) i.i.d. and independent of θ, and x_t chosen as any function of the past (adaptivity is allowed). Assume ‖x_t‖ ≤ L and Σ₀ ⪯ λI.
2. **Posterior.** Σ_t⁻¹ = Σ₀⁻¹ + σ⁻² Σ_{s≤t} x_s x_sᵀ. The posterior mean θ̂_t is Bayes-optimal for squared loss.
3. **Per-step excess squared error of the hindsight predictor.**

   E[(x_tᵀθ − x_tᵀθ̂_{t−1})² | H_{t−1}] = x_tᵀ Σ_{t−1} x_t =: σ² u_t.
4. **Chain rule for Gaussian mutual information.** I(θ; y_t | H_{t−1}, x_t) = ½ log(1 + u_t). Hence

   I(θ; y_{1:T} | x_{1:T}) = ½ Σ_t log(1 + u_t) = ½ log [det Σ₀ / det Σ_T] = ½ log det(I + σ⁻² Σ₀^{1/2} X_Tᵀ X_T Σ₀^{1/2}).
5. **Determinant bound (AM–GM on eigenvalues).** ≤ (d/2) log(1 + T λ L² / (d σ²)).
6. **Linearizing the log.** Let U := λL²/σ² ≥ u_t. Since u ↦ u / log(1+u) is increasing, u_t ≤ (U / log(1+U)) log(1+u_t).
7. **Conclusion (Bayesian squared-error regret of the hindsight predictor vs an oracle knowing θ):**

   Σ_t E[(x_tᵀθ − x_tᵀθ̂_{t−1})²] ≤ σ² (U/log(1+U)) Σ_t log(1+u_t) = 2σ² (U/log(1+U)) I(θ; data) ≤ σ² (U/log(1+U)) · d · log(1 + TλL²/(dσ²)) = O(d log T).

   This is a Bayesian-expectation bound, valid for any adaptively chosen x_t. A frequentist analogue is the elliptical potential lemma.
8. **Contrast with an outcome-ignoring predictor** (uses the prior mean 0 forever). Per-step excess error is x_tᵀΣ₀x_t, so the total is Θ(T) whenever x_tᵀΣ₀x_t ≥ c > 0. The hindsight advantage is therefore Θ(T) − O(d log T).

**Clarke–Barron link.** Step 4 is the Gaussian instance of "cumulative Bayes log-loss redundancy = I(θ; data)". Clarke–Barron says that, generically, this quantity grows like (d/2) log T. So the O(d log T) rate is the information-theoretic cost of learning a d-dimensional persistent parameter. It is paid once, and it is logarithmic rather than linear in T.

**Elliptical potential lemma (frequentist counterpart).** Σ_t min(1, ‖x_t‖²_{V_{t−1}⁻¹}) ≤ 2 log(det V_T / det V_0) ≤ 2d log(1 + TL²/(dλ₀)). It appears in Abbasi-Yadkori, Pál & Szepesvári (NeurIPS 2011) and in Lattimore & Szepesvári, *Bandit Algorithms* (CUP 2020), linear-bandit chapters. It is cited from memory, and I did not verify the lemma number.

**Planning use.** If the per-step planning loss from using θ̂ instead of θ is Lipschitz-quadratic in the prediction error, for example a smooth optimum with locally quadratic value, then cumulative planning regret is O(d log T) as well. If it is only Lipschitz (non-smooth), Cauchy–Schwarz gives O(√(T d log T)).

### Gaps
- I did not verify the exact lemma numbering in Srinivas et al. Secondary sources call it Lemma 5.3.
- I did not retrieve the Clarke–Barron original to quote its regularity conditions verbatim.

---

## 5. Lower bounds: linear regret of outcome-ignoring planners; log T and √T lower bounds for adaptive planners

### Takeaway
- **Outcome-ignoring planners [STANDARD].** Under a persistent unknown θ, the Bayesian regret of any outcome-ignoring planner against the θ-clairvoyant is exactly T × (one-step EVPI) in the stationary case. That is linear in T unless one action is optimal for almost every θ.
- **Adaptive planners, instance-dependent.** Consistent adaptive policies must incur Ω(log T) regret on each fixed instance (Lai & Robbins 1985).
- **Adaptive planners, minimax.** Worst-case regret is Ω(√(KT)) for K-armed bandits and Ω(d√T) for d-dimensional linear bandits. The worst-case Bayesian regret equals the minimax regret (Lattimore & Szepesvári, COLT 2019), so these √T bounds also bound the Bayesian regret under a least-favourable prior.
- **Passive learning.** When there is no exploration tradeoff, the learning cost is O(d log T) (§4). Under full information with a smooth (strongly convex or quadratic) loss, the log T rate is generally tight, cf. Clarke–Barron's (d/2) log T for log loss.

### Cited Findings
- **Lai & Robbins (1985).** "Asymptotically Efficient Adaptive Allocation Rules," *Advances in Applied Mathematics* 6(1): 4–22, DOI 10.1016/0196-8858(85)90002-8.
  - The first asymptotic frequentist regret lower bound for parametric bandits, derived by change of measure.
  - It applies to "consistent" (uniformly good) rules, whose regret is o(T^α) for every α > 0 on every instance.
  - Such rules have regret growing at least logarithmically. UCB-type rules attain the constant.
  - [UT reading list entry](https://cs.utexas.edu/~shivaram/readings/b2hd-LaiRobbins1985.html); [MaRDI](https://portal.mardi4nfdi.de/wiki/Publication:1060517); [arXiv:2410.02279 discussion](https://arxiv.org/pdf/2410.02279)
  - **Standard form (KL form not verified against the original in this session).** For one-parameter exponential-family arms,

    lim inf_{T→∞} E[R_T] / log T ≥ Σ_{a: Δ_a>0} Δ_a / KL(θ_a, θ*).
- **Lai (1987).** "Adaptive Treatment Allocation and the Multi-Armed Bandit Problem," *Annals of Statistics* 15(3): 1091–1114, DOI 10.1214/aos/1176350495. The paper covers UCB rules and also Bayesian allocation. — [Project Euclid](https://www.projecteuclid.org/euclid.aos/1176350495)
- **Lattimore & Szepesvári (2019).** "An Information-Theoretic Approach to Minimax Regret in Partial Monitoring," COLT 2019, arXiv:1902.00470. The paper shows that minimax regret equals worst-case Bayesian regret, i.e. the sup over priors of the Bayes-optimal regret. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1902.00470); restated in [Shufaro et al., arXiv:2405.16581](https://arxiv.org/pdf/2405.16581)
- **Ω(√(KT)) minimax lower bound.** The construction (Gaussian arms) is in Lattimore & Szepesvári, *Bandit Algorithms*, Cambridge University Press, 2020, as the minimax lower-bound chapter and Exercise 15.2, per a secondary citation. — [arXiv:2608.17841 citing it](https://arxiv.org/pdf/2608.17841)
- **Fano-type Bayesian lower bounds.** Shufaro, Merlis, Weinberger & Mannor, "On Bits and Bandits: Quantifying the Regret-Information Trade-off," arXiv:2405.16581 (2024). The paper uses Fano together with the Yang–Barron construction to lower-bound worst-case Bayesian regret by Ω(√(|Π| T)). It also relates regret to the number of information bits accumulated. — [arXiv](https://arxiv.org/abs/2405.16581)

### Inferences
- **[STANDARD] Exact linear regret of outcome-ignoring planners.**
  - Consider a stationary one-step reward μ(a, θ). An outcome-ignoring planner can do no better than a* ∈ argmax_a E_{P₀}[μ(a, θ)] at every step.
  - Its Bayesian regret against the clairvoyant is

    T · Δ_OL, where Δ_OL = E[max_a μ(a,θ)] − max_a E[μ(a,θ)] = EVPI.
  - This is exactly linear, and it equals the "Bayes risk of the prior predictive" decision.
  - For the squared-error prediction special case, Δ_OL = E[x ᵀΣ₀ x], the prior predictive variance of the mean.
  - With state dynamics, the analogous statement holds with Δ_OL replaced by the average per-step value gap of the best open-loop or prior-predictive policy. Linearity then requires an ergodicity or stationarity assumption.
- **Combined headline for the paper.**
  - Outcome-ignoring: Θ(T) regret, with constant EVPI.
  - Hindsight / Bayes-adaptive:
    - O(d log T) in passive (full-information) settings (§4, Clarke–Barron);
    - O(√(Γ H(A*) T)), e.g. O(d√(T log|A|)), in active (bandit-feedback) settings via Russo–Van Roy;
    - with matching Ω(log T) (instance-dependent, Lai–Robbins) and Ω(√T) (minimax / worst-prior Bayesian) lower bounds.
- **Log T vs √T for adaptive planners.**
  - Ω(log T) applies per fixed instance with positive gaps when feedback is partial (bandit). It also appears in full-information density estimation as (d/2) log T for log loss.
  - Ω(√T) applies minimax, or Bayesian under priors that put mass near zero gaps, when actions must be sacrificed to learn.
  - In the full-information passive case with a strongly convex loss, regret is Θ(log T). In the full-information case with linear (non-smooth) losses, the minimax rate is Θ(√(T log K)) (experts setting; Cesa-Bianchi & Lugosi 2006, from memory).

### Gaps
- **[FOLKLORE / UNVERIFIED]** That the Bayes-optimal regret for K-armed bandits under a smooth prior is of order (log T)², attributed to Lai (1987). The search did not confirm this, so do not cite it without checking the paper.
- I did not verify the Ω(d√T) linear-bandit minimax lower bound source. It is usually attributed to Dani, Hayes & Kakade (COLT 2008) and to Lattimore & Szepesvári (2020).

---

## 6. Precise condition for zero advantage

### Takeaway
Conditioning on past outcomes has exactly zero value iff the decision-relevant posterior is unchanged by the history, or more weakly, iff the prior-optimal action remains optimal under almost every posterior reachable along the optimal trajectory.

A clean sufficient condition has two parts:
- (a) there is no persistent unknown: disturbances are i.i.d., independent of the past, with a known distribution;
- (b) the current state is already observed.

In that case the history beyond the current state is irrelevant (Markov property), and Markov policies are optimal. Note: "open-loop" in the sense of ignoring the *current state* can still lose to state feedback even with i.i.d. noise. The zero-advantage result is about ignoring *past transitions* given the current state.

### Cited Findings
- **Convexity of the value in the belief** (Smallwood & Sondik 1973) gives the "≥" direction; equality requires affinity of V over the reachable posteriors. — [INFORMS](https://pubsonline.informs.org/doi/fpi/10.1287/opre.21.5.1071)
- **Blackwell's theorem (§1).** A garbling of a signal can never help. A signal that is independent of θ (and of future disturbances) is a garbling of "no signal", and vice versa, so it is worth exactly zero. — [Wikipedia: Blackwell's informativeness theorem](https://www.wikipedia.org/wiki/Blackwell%27s_informativeness_theorem)

### Inferences
**[STANDARD] Theorem template (proof sketch suitable for a paper).**

Setting: s_{t+1} = f(s_t, a_t, θ, ε_t), with θ ~ P₀ and ε_t i.i.d., independent of θ and of everything before time t.

1. **(Always ≥.)** V*_H ≥ V*_{Markov-in-s} ≥ V*_OL, by policy-class inclusion.
2. **(Zero advantage of history over current state.)**
   - Suppose P₀ is a point mass (θ known) or, more generally, the conditional law of (θ, ε_{t:∞}) given (H_t) equals its conditional law given s_t.
   - Then the controlled process is an MDP in s_t, and Markov policies achieve V*_H. So V*_H = V*_{Markov-in-s}.
   - This is the standard result that Markov (indeed deterministic Markov) policies are optimal for finite-horizon MDPs. Puterman, *Markov Decision Processes*, Wiley 1994, ch. 4 (from memory; no theorem number given).
   - Hence "hindsight" about past transitions adds nothing when the disturbances are fresh i.i.d. draws and nothing persistent is unknown.
3. **(Positive advantage.)**
   - Suppose θ is non-degenerate and the transitions are informative about θ: the likelihood p(s_{t+1} | s_t, a_t, θ) varies with θ on a set of positive probability.
   - Then b_t is a non-trivial martingale.
   - V*_H > V*_{Markov-in-s} iff, at some reachable (t, s), the action that is optimal for the prior-predictive kernel is suboptimal under a positive-probability set of posteriors. This is the EVSI > 0 condition of §2 applied stagewise.
4. **(Zero advantage despite unknown θ.)** This can happen when:
   - the optimal action is the same for all θ in supp(P₀) (a "θ-robust dominant action"), e.g. a monotone problem where the same action is best regardless; or
   - the observations are uninformative about the decision-relevant functional of θ. Example: they identify only the mean of θ, while the decision depends only on the mean of the prior predictive, which is already known. In the linear-Gaussian case, the decision uses only E[θ], and actions are evaluated on directions x with x ᵀΣ₀ = 0.
5. **Distinguishing "i.i.d. fresh disturbances".**
   - If ε_t is i.i.d. but its *distribution* is unknown (a parameter of the noise law), that unknown distribution *is* a persistent θ. Observed ε's are then informative, and hindsight helps.
   - Zero advantage requires the noise law to be known, or equivalently the disturbance sequence to be independent across time *unconditionally* under the planner's model, not just conditionally on θ.
   - With an unknown persistent θ, disturbances are exchangeable but not independent (de Finetti), and that dependence is precisely what hindsight exploits.
6. **Rate of advantage.** The advantage of hindsight over outcome-ignoring is T·EVPI − O(learning cost):
   - learning cost O(d log T) for passive or full-information learning (§4);
   - O(√(Γ H(A*) T)) with bandit feedback.
   - So the relative advantage tends to 1 − o(1) of the clairvoyant gap as T → ∞, whenever EVPI > 0.

### Gaps
- I found no single primary source that states the "fresh i.i.d. disturbances ⇒ zero value of past outcomes" result in exactly this hindsight-planning language. It is a direct consequence of the Markov property and the optimality of Markov policies (standard; Puterman 1994; Bertsekas, *Dynamic Programming and Optimal Control*), and a paper should present it as a lemma with the short proof above rather than cite it as a named theorem.
- I did not verify a specific Bertsekas section on the open-loop vs closed-loop (open-loop feedback) comparison. Bertsekas discusses open-loop-feedback control and certainty equivalence, and in LQG the separation / certainty-equivalence theorem makes feedback on the estimated state optimal.
