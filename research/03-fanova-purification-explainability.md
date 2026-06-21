# fANOVA, Purification & Explainable Boosting — Research Report

This report covers the functional ANOVA (fANOVA) decomposition of tree ensembles, the purification ("mass-moving") algorithm that recovers a canonical identifiable additive form, EBM/GA2M shape functions and FAST interaction detection, and how to accumulate a depth-3 oblivious-tree ensemble into purified 1D/2D/3D lookup tables. Math is given verbatim from the primary sources where possible.

---

## 1. The functional ANOVA decomposition

### 1.1 Definition

Any square-integrable function `F(X)`, `X = (X_1, ..., X_d)`, can be written as a sum over all feature subsets `u ⊆ [d]`:

```
F(X) = f_0 + Σ_i f_i(X_i) + Σ_{i<j} f_ij(X_i, X_j) + Σ_{i<j<k} f_ijk(...) + ...
```

(Lengerich Eq. 4; Hutter §3.1). For `pattern-boost`, because every tree depends on ≤3 features, all components with `|u| > 3` are identically zero, so the sum **truncates exactly at 3rd order** — this is the structural fact the whole design exploits.

### 1.2 Recursive (centering) formula for components

Under a reference measure `P` (see §1.3), each component is defined recursively by integrating out the complement `−S` and subtracting all strict sub-components (Hutter Eq. 4; Molnar):

```
f_∅            = ∫ F(x) dP(x)                                                   (constant / mean)
f_S(x_S)       = ∫_{X_{−S}} [ F(x) − Σ_{V ⊊ S} f_V(x_V) ] dP(X_{−S})
```

Concretely:

```
f_0      = E[F]
f_i(x_i) = E[F | X_i=x_i] − f_0
f_ij     = E[F | X_i, X_j] − f_i − f_j − f_0
f_ijk    = E[F | X_i,X_j,X_k] − f_ij − f_ik − f_jk − f_i − f_j − f_k − f_0
```

Equivalently (Hutter), define the **marginal** `a_U(θ_U) = E[F | X_U = θ_U]` (average over all other coordinates), then `f_U = a_U − Σ_{W ⊊ U} f_W`. The marginal is the conditional expectation under the reference measure; the component is the marginal minus all lower-order pieces.

### 1.3 The reference measure question

The integral `dP(X_{−S})` requires choosing a measure. Three canonical choices, with very different semantics:

- **Uniform** over each feature's range (`P` = product of uniforms). Used by Hutter–Hoos–Leyton-Brown for hyperparameter importance. Simplest; "double-centering" with uniform weights.
- **Product of marginals** `P(X) = ∏_i p_i(X_i)`. Each feature integrated against its own empirical marginal, features treated as independent. This is the *classical* fANOVA / Sobol setting and what EBM-purification's "empirical" estimator approximates per-axis.
- **Joint distribution** `P(X) = p(X)` (the true data density). This is the *correct* measure (Lengerich: "the correct `w(x)` under which to understand effects is the true data distribution `p(x)`"), but it is hard to estimate and couples the axes.

Lengerich uses three piecewise-constant density estimators for `ŵ`:

```
Uniform:    ŵ_unif(x_{−u}) ∝ 1
Empirical:  ŵ_emp(x_{−u})  ∝ Σ_{x_i ∈ X_train} 1{ x_i_{−u} = x_{−u} }
Laplace:    ŵ_lap         ∝ ŵ_unif + ŵ_emp        (additive smoothing)
```

"The choice of distribution can change (sometimes dramatically) the purified effects."

### 1.4 Orthogonality / identifiability conditions

Under classical fANOVA (independent / product reference), components satisfy three axioms (Hooker 2004/2007; Molnar):

```
Zero means:    ∫ f_S(x_S) dP(X_S) = 0           for every S ≠ ∅
Orthogonality: ∫ f_S(x_S) f_V(x_V) dP(X) = 0     for S ≠ V
Variance dec.: σ²(F) = Σ_S σ²(f_S)
```

The **zero-means / integrate-to-zero** condition is the identifiability constraint. Lengerich states it as: for piecewise-constant `F`, every 1-D slice of each effect tensor has weighted mean zero (Eq. 5c → 6a):

```
∀ u, ∀ i∈u, ∀ x_{u\i}:   Σ_{x_i ∈ Ω_i} f_u(x_{u\i}, x_i) · Σ_{X_{−u}} w(X) = 0
```

This is *equivalent* to the orthogonality of `f_u` to every member operating on a subset of `u` (Lengerich Lemma 4.1, citing Hooker). When these hold, the decomposition is the **unique** orthogonal decomposition with **minimum variance in the higher-order terms** — i.e. mass is pushed maximally down into the lowest-order effects that can carry it.

### 1.5 Why non-unique under correlation, and how it is resolved

**The identifiability problem (Lengerich's motivating example).** With unconstrained additive-plus-interaction models you can freely shuttle mass between orders without changing predictions. For Boolean `X_1, X_2`, AND, OR, and XOR-with-main-effects all produce *identical outputs* but contradictory tables; e.g. `X_1 ∧ X_2 = −0.25(X_1 ⊕ X_2) + 0.5(X_1−0.5) + 0.5(X_2−0.5) + 0.25`. For the multiplicative model `Y ≈ a + bX_1 + cX_2 + dX_1X_2`, the reparametrization `(d)(X_1−α)(X_2−β)` with shifted main/intercept terms gives the *same function* for any `α, β`, so the apparent "interaction strength" and "main effect strength" are arbitrary.

**Hooker's correlation issue.** Under classical orthogonality, when features are *dependent*, integrating against a **product / uniform** measure evaluates `F` at feature combinations that never occur (low-density / extrapolation regions), so the decomposition emphasizes behavior in regions of near-zero probability and is unstable. But integrating against the **joint** measure breaks the clean orthogonality `∫ f_S f_V dP = 0`: components for overlapping-but-non-nested sets (e.g. `{1,2}` and `{2,3}`) cannot all be mutually orthogonal.

**Hooker's resolution — generalized fANOVA with *hierarchical orthogonality*.** Replace full orthogonality with: each component is orthogonal only to its own **sub-components** (lower-order terms whose index set it contains), weighted by `w(x)`:

```
∀ S ⊊ U:   ∫ f_S(x_S) f_U(x_U) w(x) dx = 0
```

This keeps the variance-decomposition / identifiability benefit (a lower-order effect can't be hiding inside a higher-order one) while permitting the joint/weighted measure, so the decomposition respects the data density and avoids extrapolation. Lengerich's purification is exactly the constructive, exact algorithm that enforces these hierarchical zero-mean conditions on piecewise-constant `F` for an arbitrary weight `w`.

---

## 2. Purification: the exact algorithm

### 2.1 Setup

Represent the ensemble as a set of **effect tensors** `{T_u}`, one per feature set `u` (with `T_∅` the scalar intercept). Each `T_u` is a grid of effect sizes over the bins `Ω_i` of the features in `u`. The ensemble prediction at `x` is `Σ_u T_u[x_u]`. "Purifying" means transforming `{T_u}` so each tensor satisfies the zero-mean-slice condition (6a) **without changing the sum** — i.e. recovering the fANOVA.

### 2.2 The mass-moving operator

Define the weighted mean of a 1-D slice of `T_u` along axis `i` (Lengerich Eq. 10):

```
m(T_u, i, x_{u\i}) = Σ_{x_i ∈ Ω_i} f_u(x_{u\i}, x_i) · Σ_{X_{−u}} w(X)
```

Because predictions are a *sum* over tensors, the value `m(T_u, i, x_{u\i})` can be **moved from `T_u` into `T_{u\i}`** (the tensor one order lower, on the remaining axes) without changing any prediction. Subtracting it from the slice makes that slice mean-zero; adding it to `T_{u\i}` conserves total mass. This is "mass-moving."

### 2.3 Purify-Matrix (Algorithm 1) — purify one tensor

```
PURIFY-MATRIX(T, w, u, Ω):           # makes every slice of T_u zero-mean
  repeat until pure:
    pure ← True
    for i in u:                       # for each axis
      for x_{u\i} in Ω_{u\i}:         # for each slice perpendicular to i
        m0 ← m(T_u, i, x_{u\i})        # weighted slice mean, Eq. 10
        if m0 ≠ 0:
          pure ← False
          T_u[x_{u\i}, :] ← T_u[x_{u\i}, :] − m0     # center the slice
          T_{u\i}[x_{u\i}] ← T_{u\i}[x_{u\i}] + m0   # push mass down one order
  return T
```

### 2.4 Purify (Algorithm 2) — the full cascade

```
PURIFY(T, w, Ω):
  order ← sort feature-sets u by DECREASING |u|     # 3-way, then 2-way, then 1-way
  for u in order:
    T ← PURIFY-MATRIX(T, w, u, Ω)
  return T
```

Mass cascades **down the order lattice**: triples → pairs → mains → intercept. Processing strictly from high to low order guarantees that once `T_u` is purified, later operations on its sub-tensors do not re-pollute it. At convergence `{T_u}` is exactly the fANOVA decomposition (unique under non-degenerate `w`).

### 2.5 Key properties (operationally important for `pattern-boost`)

- **Convergence.** Define unpurified mass `M_t = Σ_{i,j} w_{ij}(|r_i| + |c_j|)` where `r_i, c_j` are weighted row/column sums.
  - *Uniform (or any axis-constant) weight:* converges in a **single pass** per axis — `M_t = 0 ∀ t ≥ 2` (Theorem 1). This is ordinary "double-centering."
  - *Generic non-degenerate `w`:* geometric convergence, `M_t ≤ ε` for `t ≥ τ(ε) = log_2(M_0/ε)`, i.e. `O(log(M_0) − log(ε))` iterations (Theorem 2). Empirically "almost all mass is moved in the first iteration."
- **Permutation invariance** (Cor. 2.1): slice order and nominal re-encoding don't change the result.
- **Linearity** (Cor. 2.2): purification is a linear operator in the interaction tensor with `Σα_i = 1`, so **you can purify per-tree and sum, or sum then purify — identical result**. This is the property that makes ensemble accumulation clean (§4).

A reference implementation lives in `microsoft/interpret` (and `LengerichLab/gam_purification`), "under 100 lines of Python."

---

## 3. EBM / GA2M: shape functions, FAST, additivity, cyclic boosting

### 3.1 Model form

GA2M / EBM are GAMs with optional pairwise terms (Lou §2; InterpretML):

```
g(E[y]) = β_0 + Σ_j f_j(x_j) + Σ_{(i,j)} f_ij(x_i, x_j)
```

`g` = link (identity for regression, logit for classification). Each `f_j` is a **1-D shape function** (a per-feature lookup table); each `f_ij` is a **2-D heatmap** lookup table. By construction the model is a sum of ≤2-D pieces, so every term is individually plottable — this is the source of "intelligibility."

### 3.2 How shape functions / heatmaps are produced — cyclic (round-robin) boosting

EBM fits each `f_j` by **cyclic gradient boosting restricted to one feature at a time** (InterpretML):

1. Initialize all terms to 0 (or intercept).
2. Cycle round-robin through features `j = 1..n`. On feature `j`, fit a small tree (often a shallow stump / few leaves) **only on `x_j`** to the current gradient/residual, multiply by a **very low learning rate** `η`, and add it into `f_j`'s table.
3. Repeat for thousands of boosting rounds. Because `η` is tiny and updates are interleaved, **feature order does not matter** and collinearity is mitigated.
4. Bagging over many outer bags + averaging smooths the shape functions.

The accumulated per-feature additive contribution *is* the shape function; binning `x_j` makes `f_j` a lookup table. Pairwise terms `f_ij` are boosted the same way on a 2-D grid, **after** main effects, on the residual.

### 3.3 FAST — ranking which pairwise interactions to include

FAST (Lou §4.1) screens all `C(n,2)` candidate pairs cheaply on the **residual** `R = y − F_mains(x)`:

- For a pair `(x_i, x_j)`, build the *simplest possible* interaction model `T_ij`: place **one cut `c_i` on `x_i` and one cut `c_j` on `x_j`**, splitting the plane into 4 quadrants `{a,b,c,d}`, each predicting the **mean residual** in that quadrant.
- Search all `(c_i, c_j)` cut positions; the interaction **strength** of `(i,j)` is the lowest achievable **RSS** of this 4-quadrant predictor (largest RSS reduction ⇒ strongest interaction).
- **Efficiency trick:** precompute marginal cumulative histograms of target-sums and weight-sums, `CH^t_i(v), CH^w_i(v)`, and a dynamic-programming lookup table `L^t(c_i,c_j)=[a,b,c,d]` filled row-by-row via `a[p][q] = a[p−1][q] + Σ_{k≤q} H^t_ij(v^p_i, v^k_j)`. With these structures, evaluating any cut's quadrant sums and its RSS is `O(1)`, so all pairs are ranked extremely fast.

The GA2M outer loop (Lou Alg. 1) is greedy forward stagewise: fit best additive model in `H_1 + Σ_{u∈S} H_u`, take residual `R`, rank remaining pairs by `F_u = E[R|x_u]`, add the best pair to `S`, **refit**, repeat until no accuracy gain. Refitting after each addition lets spurious correlation-induced pairs shrink away.

### 3.4 Keeping it additive / intelligible

Restrict to ≤2-D terms; cap the number of pairwise terms (user-controlled); use FAST to admit only high-RSS-reduction pairs. Each retained term is a low-D lookup table that can be visualized. **EBM additionally purifies** the learned pairwise terms (it ships Lengerich's algorithm) so that main effects own all the mass they can and the heatmaps show only true residual interaction — fixing the identifiability degeneracy of §1.5.

---

## 4. Accumulating an oblivious-tree ensemble into fANOVA tables

This is the heart of `pattern-boost`. An oblivious (symmetric) depth-`D` tree uses the **same split (feature, threshold) at every node of a given level**, so it is a single multilinear lookup: depth-3 ⇒ at most 3 distinct split features, `2³ = 8` leaves indexed by the bit pattern of the 3 binary tests. Such a tree is *exactly* a piecewise-constant function over its ≤3 features — the ideal input to purification.

### 4.1 Per-tree contribution as a tensor

For tree `t` with split features `u_t = {i,j,k}` and thresholds, the tree is a function `g_t(x_i, x_j, x_k)` taking one of 8 leaf values according to `(1{x_i>θ_i}, 1{x_j>θ_j}, 1{x_k>θ_k})`. Multiply by the learning rate/shrinkage already baked into leaf values. This is a rank-1-grid tensor `T^{(t)}_{u_t}`.

### 4.2 Choosing the grid resolution (the key engineering point)

The natural grid for each feature is the **union of split thresholds actually used by the ensemble for that feature** — *not* all candidate bins, and *not* a fixed dense grid:

```
Ω_i = { all distinct thresholds θ_i across all trees that split on feature i }  ∪ {±∞ endpoints}
```

Reasons:
- Each tree is piecewise-constant with breakpoints only at its own thresholds; the *sum* of trees is piecewise-constant with breakpoints at the **union** of thresholds. Between consecutive thresholds the ensemble is exactly constant, so finer grids add zero information and waste memory.
- Using the union (a "merged" grid) makes every tree's tensor exactly representable on the common grid — a tree's value over an interval `[θ_a, θ_b)` of the merged grid is constant, so it maps losslessly onto every sub-cell.
- This keeps tables as small as the model's true complexity (`|Ω_i|` = number of distinct cuts on feature `i`), which for oblivious GBMs is typically modest.

### 4.3 Summation into per-feature-set tables

Build one **raw** tensor per feature set that appears, on the merged grid:

```
T_u^raw = Σ_{ t : u_t ⊆ u }  expand(g_t)  onto grid Ω_u      for each distinct u of size 1, 2, 3
```

In practice: for each tree `t`, take its `≤3`-feature tensor, **broadcast/expand** it onto the merged grid of its own feature set `u_t`, and add into `T^raw_{u_t}`. (A tree that uses only 2 distinct features contributes to a 2-way raw tensor; a degenerate 1-feature tree to a main-effect tensor.) After this pass you have raw tensors `T^raw_∅, T^raw_i, T^raw_ij, T^raw_ijk` whose **sum reproduces the ensemble exactly**, but which are *not yet identifiable* (mass is split arbitrarily across orders because each tree's leaf values are uncentered).

### 4.4 Purify into canonical 1D/2D/3D effects

Run `PURIFY({T^raw_u}, w, Ω)` from §2: cascade 3-way → 2-way → 1-way → intercept under the chosen reference measure `w`. Output:

```
f_0                        scalar
f_i(x_i)        = T_i      purified 1-D main-effect tables  (each slice mean-zero)
f_ij(x_i,x_j)   = T_ij     purified 2-D pairwise tables     (every row & column mean-zero)
f_ijk(...)      = T_ijk     purified 3-D triple tables       (every axis-slice mean-zero)
```

By linearity (Cor. 2.2) this equals purifying each tree and summing, so you can also stream-purify incrementally. After purification, the tables are the **unique minimum-higher-order-variance fANOVA** of the ensemble.

---

## 5. Verifying losslessness

The defining invariant is **exact reconstruction**: for every `x`,

```
F_ensemble(x)  ==  f_0 + Σ_i f_i(x_i) + Σ_{i<j} f_ij(x_i,x_j) + Σ_{i<j<k} f_ijk(x_i,x_j,x_k)
```

This holds *exactly* (to floating point) because:

1. **Accumulation is exact:** the merged-threshold grid (§4.2) represents each tree's piecewise-constant function with zero approximation error, so `Σ_u T^raw_u(x) = F(x)` identically.
2. **Purification conserves mass:** every mass-move subtracts `m0` from one tensor and adds the *same* `m0` to another (Alg. 1 lines 9–10), so `Σ_u T_u(x)` is invariant under purification. Hence `Σ_u f_u(x) = Σ_u T^raw_u(x) = F(x)`.

**Practical verification protocol for `pattern-boost`:**
- *Algebraic check:* assert `max_x | F_ensemble(x) − Σ_u f_u(x_u) | < tol` over (a) all training rows and (b) a random sample of grid-cell corners (the worst case, since the function is constant within cells). Because the function is piecewise-constant on the merged grid, checking one interior point per cell is exhaustive.
- *Mass-conservation check:* track total signed mass before/after purify; it must be unchanged.
- *Purity check:* assert every 1-D slice weighted mean `m(T_u,i,·) ≈ 0` for all `u` with `|u|≥1` (the fANOVA zero-mean condition) — this confirms canonical form, not just losslessness.
- *Variance check:* `σ²(F) ≈ Σ_u σ²(f_u)` under `w` (the variance-decomposition axiom) — a strong end-to-end test that both losslessness and orthogonality hold.

---

## 6. Practical issues: combinatorics, selection, readability

- **Number of possible tables.** Up to `C(n,1) + C(n,2) + C(n,3)` tables. The `C(n,3)` triples dominate (e.g. `n=100` ⇒ 161 700 possible triples). **But you only ever materialize feature sets that actually appear as a tree's support** — the realized set is bounded by the number of distinct `≤3`-feature combinations the booster chose, which is typically a few hundred, not `C(n,3)`. Build tables lazily from the trees, never enumerate the full lattice.
- **Sparsity / selection.** Combine three levers: (a) FAST-style RSS-reduction ranking to admit only strong pairs/triples; (b) the **hierarchy/heredity restriction** (admit an interaction only if its sub-effects are already present); (c) post-hoc pruning by **purified variance** `σ²(f_u)/σ²(F)` (Sobol-style importance) — drop tables that explain negligible variance *after* purification (pre-purification importances are unreliable per §1.5).
- **Readability.** After purification, main effects carry maximal mass and interaction tables are genuinely "pure" residual structure, so most tables are near-zero and can be hidden. Sort and display by purified variance fraction. Keep per-feature grids coarse (merged thresholds only). Cap displayed triples to top-`k` by variance.
- **Cost note.** Per-tensor purification is `O(#cells × log(1/ε))` and embarrassingly parallel across tensors; the union-grid keeps `#cells` small. The dominant memory cost is 3-D tensors `|Ω_i|·|Ω_j|·|Ω_k|`, another reason to keep grids at realized-threshold resolution.

---

## Design implications for `pattern-boost`

**Recommended decomposition + purification pipeline**

1. **Train** depth-3 oblivious/symmetric trees (CatBoost-style), recording per tree its `≤3` split features + thresholds + 8 shrunk leaf values. The ≤3-feature constraint guarantees the fANOVA truncates at 3rd order by construction — no higher-order tables can ever exist, so the decomposition is *complete*, not approximate.
2. **Merged grid:** for each feature `i`, set `Ω_i` = sorted union of all thresholds used on `i` across the ensemble (+ `±∞`). Tables live on this grid.
3. **Accumulate** raw tensors `T^raw_{u}` by expanding each tree onto its feature set's grid and summing (only realized feature sets `u`, `|u|≤3`).
4. **Purify** with the cascade `PURIFY` (3-way → 2-way → 1-way → intercept), yielding `f_0, {f_i}, {f_ij}, {f_ijk}`. Use Lengerich's mass-moving operator; it converges in one pass for uniform `w`, a handful of passes otherwise; purify-then-sum ≡ sum-then-purify (linearity) so it can be incremental.
5. **Verify losslessness** by the per-cell reconstruction assertion + mass conservation + per-slice zero-mean + variance-sum identity (§5). Make these unit-test invariants of the build.

**Choice of reference measure**

- **Default: empirical product-of-marginals with Laplace smoothing** (`ŵ_lap = ŵ_unif + ŵ_emp`). Rationale: per-axis empirical marginals respect the data density along each feature (avoiding Hooker's extrapolation problem in unlikely regions), Laplace smoothing keeps weights strictly positive (so empty grid cells don't break the zero-mean conditions and convergence stays well-conditioned), and a product (per-axis) reference makes purification a sequence of simple weighted-mean removals.
- **Expose `w` as a first-class choice.** Offer `uniform` (fast, single-pass, effect-coding semantics) and `joint/weighted` (Hooker's hierarchical-orthogonality, most faithful to correlated data) as alternatives. Because "the choice of distribution can change purified effects dramatically," surface the chosen `w` alongside every explanation and let users recompute tables under a different `w` without retraining (purification is post-hoc and cheap).

**How to expose it**

- **Per-feature-set tables as the public artifact:** intercept `f_0`; one 1-D table per feature; 2-D heatmap per retained pair; 3-D table (sliceable / shown as conditioned 2-D heatmaps) per retained triple. Each is a lookup table, so scoring is `F = f_0 + Σ lookups` — fast and auditable, and *bit-for-bit equal* to the tree ensemble.
- **Importances via component variance:** report `FU = σ²(f_u)/σ²(F)` (Hutter's fraction-of-variance / Sobol index) for every table; these sum to ~1 (variance-decomposition axiom) and give a principled, identifiable ranking of mains vs pairs vs triples. Compute *after* purification only.
- **Sparsity controls:** FAST/RSS ranking + heredity restriction at train time; variance-threshold pruning of tables at explain time; top-`k` display by `FU`.
- **Local explanations:** for a single prediction, the additive term contributions `f_u(x_u)` are exact attributions that sum to `F(x) − f_0`; show them sorted by magnitude.

The combination "depth-3 oblivious trees ⇒ exact 3rd-order fANOVA ⇒ purified lookup tables ⇒ provable lossless reconstruction + variance-based importances" is internally consistent and directly supported by the cited literature.

---

## Sources

- [Lengerich, Tan, Chang, Hooker, Caruana (2020), "Purifying Interaction Effects with the Functional ANOVA" — AISTATS / PMLR v108](https://proceedings.mlr.press/v108/lengerich20a/lengerich20a.pdf) · [arXiv:1911.04974](https://arxiv.org/abs/1911.04974)
- [Hutter, Hoos, Leyton-Brown (2014), "An Efficient Approach for Assessing Hyperparameter Importance" — ICML / PMLR v32](http://proceedings.mlr.press/v32/hutter14.pdf)
- [Hooker (2007), "Generalized Functional ANOVA Diagnostics for High-Dimensional Functions of Dependent Variables" — JCGS 16(3):709–732](https://www.tandfonline.com/doi/abs/10.1198/106186007X237892)
- [Lou, Caruana, Gehrke, Hooker (2013), "Accurate Intelligible Models with Pairwise Interactions" (GA2M + FAST) — KDD'13](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf)
- [InterpretML — EBM documentation](https://interpret.ml/docs/ebm.html) · [interpretml/interpret (GitHub, ships the purification algorithm)](https://github.com/interpretml/interpret)
- [Molnar, "Interpretable Machine Learning" — Ch. Functional Decomposition](https://christophm.github.io/interpretable-ml-book/decomposition.html)
- [LengerichLab/gam_purification](https://github.com/LengerichLab/gam_purification)
- [fANOVA library (PyPI) — Hutter et al. implementation](https://pypi.org/project/fanova/)
- Supporting: [GAMI-Net (Yang et al. 2020)](https://arxiv.org/pdf/2003.07132); [Sun et al. (2023), "Interpretable ML based on Functional ANOVA Framework"](https://arxiv.org/pdf/2305.15670)
