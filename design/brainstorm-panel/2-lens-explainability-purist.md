# tri-boost — Brainstorm from the Explainability Purist (fANOVA / Identifiability) Lens

## 1. Thesis / the bet

My bet is that "perfect explainability" is not a property of the *trees* — it is a property of the **canonical artifact we choose to publish**, and that artifact must be defined *before* we touch accuracy. The depth-3 oblivious ensemble only guarantees that a lossless ≤3rd-order decomposition *exists*; it does not make that decomposition *unique*, *meaningful*, or *auditable*. Uniqueness comes from purification under an **explicitly declared reference measure**, and meaningfulness comes from treating the (measure, grid, purified tables, variance budget) tuple as a single signed contract that an actuary can re-derive by hand. So my core design move is to make the reference measure a first-class, versioned input; make losslessness, mass-conservation, zero-mean purity, and variance-decomposition into *tested build invariants* (not aspirations); and make every downstream "explanation" (SHAP, PDP, H-stat, importances) a **closed-form read of the tables**, never an independent estimator that can disagree with the deployed numbers. The whole game is: one number, derivable three ways (tree sum, table sum, Shapley sum), all bit-equal. If those three ever disagree, we have no product.

## 2. Components

### A. The canonical artifact & the accumulate→purify→tables pipeline

- **Merged-threshold union grid `Ωᵢ` per feature** [BOTH]. Grid = sorted union of all thresholds the ensemble actually used on feature `i`, plus `±∞`. Mechanism: the ensemble is piecewise-constant with breakpoints only at realized cuts, so this grid represents every tree *losslessly* while staying as small as the model's true complexity (§4.2 of research/03). Tradeoff: 3-D tensors cost `|Ωᵢ|·|Ωⱼ|·|Ωₖ|`; the union grid is the *minimal* exact grid, so there's no cheaper lossless option — the only lever is fewer realized triples (see interaction selection).

- **Raw accumulation then high→low purification cascade** [EXPLAINABILITY]. Expand each tree onto its feature-set grid, sum into `T^raw_u`, then `PURIFY` 3-way→2-way→1-way→intercept (Lengerich Alg. 2). Mechanism: mass-moving subtracts a slice mean from `T_u` and adds it to `T_{u\i}`, conserving the prediction sum exactly while driving every slice to zero-mean. Tradeoff: none structurally (linearity, Cor. 2.2, means purify-per-tree-and-sum ≡ sum-then-purify, so it can even stream); the only cost is the measure choice it forces us to make explicit.

- **The decomposition is COMPLETE, not truncated** [EXPLAINABILITY]. Because every tree has ≤3 features, there are *provably no* order-≥4 components to discard. State this loudly: tri-boost's tables are exhaustive, unlike EBM (caps at 2) or any post-hoc fANOVA on a black box (truncates and approximates).

### B. The reference-measure decision (the most consequential call)

- **Three declared measures, `w ∈ {uniform, empirical-product+Laplace, joint-empirical}`** [EXPLAINABILITY]. Default: **Laplace-smoothed empirical product-of-marginals** (`ŵ_lap = ŵ_unif + ŵ_emp`). Mechanism: per-axis empirical marginals respect data density along each feature (dodging Hooker's extrapolation into low-density combos), Laplace keeps weights strictly positive so empty cells don't break the zero-mean conditions or convergence. Tradeoff: a *product* measure still evaluates effects at correlated-feature combinations that rarely co-occur; the **joint-empirical (Hooker hierarchical-orthogonality)** mode fixes that but only guarantees each component orthogonal to its *sub*-components, not full orthogonality — so `σ²(F) ≠ Σσ²(f_u)` and naive equal-split SHAP is no longer exact under it.

- **Recompute-tables-under-a-different-`w`-without-retraining** [EXPLAINABILITY]. Purification is post-hoc and cheap; expose `w` as a config on the existing purify step. Mechanism: leaves stay piecewise-constant, components stay on their ≤3 features, prediction sum is conserved bit-for-bit — only table *values* change. This is mis-scored as `breaks/changes_table_form` in the inventory; it is **`preserves_exactness`** and I'd correct that in the spec. Tradeoff: the joint 3-D density estimate has many empty cells → needs Laplace/copula/factorized smoothing to stay well-conditioned and reproducible (the genuine open research item).

- **Surface `w` on every artifact** [EXPLAINABILITY]. Lengerich: "the choice of distribution can change purified effects dramatically." So every exported table, importance, and SHAP plot carries the `w` that produced it. An actuary must never see a relativity without knowing the measure it was purified against.

### C. Lossless-reconstruction as tested guarantees

These are the heart of "perfect explainability" being *literally provable*. Each is a build-blocking unit-test invariant [all EXPLAINABILITY]:

- **Per-cell equality:** `max_x |F_ensemble(x) − Σ_u f_u(x_u)| < tol`, checked at one interior point per merged-grid cell (exhaustive because piecewise-constant) plus all training rows.
- **Mass conservation:** total signed mass invariant across purification.
- **Zero-mean purity:** every 1-D slice weighted-mean `m(T_u,i,·) ≈ 0` for `|u|≥1` — confirms canonical form, not just losslessness.
- **Variance decomposition:** `σ²(F) ≈ Σ_u σ²(f_u)` under product `w` (under joint `w`, this won't hold — that's a *correct* failure and the test must branch on `w`).
- **SHAP-oracle agreement:** the equal-split table-Shapley must equal an independent interventional-TreeSHAP run (used purely as a test oracle). Three computations of one number, all bit-equal.

### D. Identifiability, importances & exact attributions straight from tables

- **Sobol / Shapley-effects as the canonical importance** [EXPLAINABILITY]. `S_u = σ²(f_u)/σ²(F)` from purified table variances, zero model calls. Mechanism: under product `w` these sum to 1 (variance axiom). Tradeoff: under joint `w`, use **Shapley effects** (efficient by construction) instead of raw Sobol ratios so importances still sum to 1.

- **Exact SHAP / Faith-Shap / n-Shapley as O(1) table reads** [BOTH]. `φᵢ(x) = Σ_{u∋i} f_u(x_u)/|u|` (Bordt–von Luxburg equal-split); InstaSHAP/ANOVA backbone formalizes that a *purified order-n additive model IS* the Shapley computation. Mechanism: at most `#mains + #pairs/2 + #triples/3` lookups, exact not Monte-Carlo. Tradeoff: this is **interventional** SHAP under product `w` — the API must *label it "interventional," never just "SHAP,"* or it silently disagrees with conditional TreeSHAP under correlation.

- **PDP/ICE/H-statistic as thin views over tables** [EXPLAINABILITY]. `PD_i = f₀+f_i`; `H²_jk = σ²(f_ij)/σ²(PD_jk)` exactly (kills H's O(n²) Monte-Carlo instability). Tradeoff: PD=table identity holds exactly only under *unsmoothed* product `w`, approximately under Laplace, not at all under joint — the view must display its measure. Clamp H's known pathologies (can exceed 1; blows up when both mains are weak).

### E. Keeping the table set SMALL & human-readable

- **Complete-set-for-inference vs pruned-set-for-display split** [EXPLAINABILITY]. The deployed model uses *all* realized tables (losslessness is non-negotiable); the *display/filed* view shows top-`k` by Sobol/Shapley-effects variance, hiding near-zero tables. These are two artifacts, tracked separately. Tradeoff: a hidden table still contributes to the prediction — the UI must say "showing 12 of 47 tables explaining 99.3% of variance," never imply the rest are zero.

- **Heredity + FAST as train-time admission, not just explain-time pruning** [BOTH]. Admit a triple only if its sub-pairs cleared a FAST/heredity screen, feeding the interaction-constraint machinery so the booster never *materializes* `C(n,3)`. Mechanism: bounds realized support at training time. Tradeoff: FAST's RSS-on-residual objective ≠ the booster's Newton gain, so use it as a **soft pre-filter**, with *final* selection by exact purified variance — never a hard gate that amputates an interaction the GBM would have found. And prefer **joint boosting over realized supports + one purification pass** over EBM-style two-stage staging, which mis-converges under correlation (the GAMI-Tree lesson).

### F. Per-cell uncertainty on tables

- **Inner/outer bagging → per-cell standard-error bands** [EXPLAINABILITY]. Average multiple oblivious ensembles' purified tables on a *common* merged-union grid; the spread is a credibility annotation. Tradeoff: bags pick different cuts, so the common grid is denser (union of unions); SEs are an annotation, *not* a fANOVA component — compute spread *after* per-bag purification.

- **ANOVA-BART as the principled future head** [EXPLAINABILITY]. A Bayesian sum-of-trees under the *same* identifiability constraint gives full posterior credible intervals on every table cell — the only catalogue method giving genuine per-cell posteriors (SGLB/NGBoost/PGBM give per-row, not per-cell). Flag as research; it's the right long-term answer to "how confident is this relativity in a thin risk cell?"

## 3. Highest-leverage ideas (my lens)

1. **The reference measure is a versioned, signed input — and recomputable post-hoc.** This single decision determines what every "pure" effect *means*. Making it explicit, defaulting to Laplace-product, offering joint-Hooker, and letting actuaries re-purify without retraining is the difference between "tables" and "auditable rating tables." Correct its inventory mis-scoring to `preserves_exactness`.

2. **Three-way-equal-number invariant as a build gate.** Tree-sum = table-sum = Shapley-sum, bit-for-bit, enforced in CI. This is what makes "perfect explainability" *provable* rather than asserted, and it's cheap because the function is piecewise-constant (one check per cell is exhaustive).

3. **Complete-for-inference / pruned-for-display as a hard architectural split**, with variance-budget honesty ("showing N of M, X% of variance"). This is what keeps the promise readable *without* lying by omission.

4. **Exact interventional SHAP/Faith-Shap as O(1) table reads, rigorously labelled** — and stock TreeSHAP demoted to a *test oracle only*. The InstaSHAP/n-Shapley/Möbius equivalence lets us claim the 3rd-order tables *are* the exact Shapley/Faith-Shap up to order 3 — a real theoretical headline.

5. **Per-table balance/calibration diagnostics** (A/E per main-effect level, per interaction cell), exploiting an edge no black-box GBM has. Plus global mean re-anchoring folded into `f₀` (exactness-preserving) as default-on.

## 4. What I would explicitly NOT do (and why)

- **No post-hoc table editing on the audited path.** EBM's `monotonize()` (isotonic on a 1-D term), Whittaker-Henderson graduation, ALE-as-a-separate-artifact — all edit tables *away* from the ensemble and forfeit losslessness. Monotonicity/smoothness must be enforced *in-fit* (split rejection), never bolted on. Any post-hoc smoothing ships as opt-in *deployment* processing tracked separately from the bit-exact artifact.

- **No calibration warp folded into the tables.** Any nonlinear 1-D warp `g` of the aggregate score makes `g(Σf_u) ≠ Σg(f_u)` — local autocalibration, multicalibration, beta/Venn-ABERS on the response scale all break additivity. They live *outside* the decomposition as a declared score→prediction map (exactly where the link sits) and are never re-decomposed. Only affine/intercept adjustments (global re-anchoring; Platt/temperature on the logit scale) fold in cleanly.

- **No ALE as a feature.** It conditions on local neighborhoods — but that faithfulness is *exactly* what joint-`w` purification already provides on the deployable tables. ALE would be a competing artifact whose tables are *not* the lossless ones. Keep it as a research cross-check at most.

- **No raw `C(n,3)` triple enumeration, no Sobol/importance ranking computed *before* purification.** Pre-purification importances are unreliable (the AND/OR/XOR identifiability degeneracy, §1.5) — mass is split arbitrarily across orders until purified. Always purify first, then rank.

- **No soft/neural "oblivious" false-friends** (NODE, NODE-GAM, TEL, GRANDE, GAMI-Lin-Tree). They break hard regions and/or non-constant leaves → no exact decomposition. They *validate the thesis* as prior art; they are not code to import.
