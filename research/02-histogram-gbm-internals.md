# Histogram-Based Gradient Boosting Internals — Research Report

This report covers the engineering behind LightGBM/XGBoost/scikit-learn HistGradientBoosting, with exact formulas, and a final section adapting it to `pattern-boost`'s depth-3 oblivious-tree constraint.

---

## 1. Feature binning / quantization

**Why.** Pre-binning each continuous feature once, up front, converts split-finding from an O(n log n) sort-per-node (XGBoost `exact`) into an O(n) histogram scan over a small fixed number of bins. This is the single biggest speedup in modern GBMs. Binning happens **once at the start of training**, not per node.

**How bins are chosen.**
- **Quantile (most common).** Bin edges are sample quantiles so each bin holds roughly equal *count* (or equal *Hessian mass*, see below). LightGBM and scikit-learn HistGradientBoosting default to quantile-style bins. scikit-learn computes `max_bins - 1` quantile thresholds via `np.percentile(..., method="averaged_inverted_cdf")` over `linspace(0,100,max_bins+1)`.
- **Midpoints for low-cardinality features.** If a feature has fewer distinct values than `max_bins`, edges are placed at midpoints of consecutive distinct values: `thresholds = sliding_window_view(distinct_values, 2).mean(axis=1)`. This avoids wasting bins and gives exact splits.
- **Uniform/equal-width** is simpler but worse on skewed data; production libraries use quantile.
- **XGBoost** (`hist`/`approx`) uses a **weighted quantile sketch**: candidate split points are quantiles of the distribution weighted by the Hessian `h_i` (the "rank function" `r_k(z)` in the paper), so bins carry equal *second-order loss mass*, not equal count. This is more accurate for the Newton objective.

**Bin count.** Default `max_bin = 255` (LightGBM) / `max_bins = 255` (sklearn). 255 is chosen so a binned value fits in **one `uint8`** with a slot left over: sklearn reserves bin index `n_bins - 1 = 255` for **missing values** (so `max_bins = n_bins - 1 = 255` data bins + 1 missing bin = 256 = full uint8 range). LightGBM similarly keeps ≤256 bins so training data is stored as `uint8`.

**Large data.** Quantiles are estimated on a **subsample** (sklearn `subsample = 200_000` rows, drawn with `rng.choice`, weighted if sample weights exist). XGBoost uses the distributed weighted quantile sketch with provable ε-approximation so it never needs all data in memory at once. Approximate quantiles cost almost nothing in accuracy at these bin counts.

---

## 2. Gradient/Hessian histograms, the subtraction trick, memory layout

**Accumulation.** For node (leaf) with instance set `I`, for each feature `k`, build a histogram of `nbins` buckets. For every instance `i ∈ I` with binned value `b = bin_k(i)`:

```
Hist_k[b].grad += g_i        // sum of gradients
Hist_k[b].hess += h_i        // sum of hessians
Hist_k[b].count += 1
```

So each bin holds the per-bin partial sums:  `G_{k,v} = Σ_{i: bin=v} g_i`,  `H_{k,v} = Σ_{i: bin=v} h_i`. Cost = O(#instances × #features) (one pass), independent of #bins.

**Split finding from the histogram.** Sort within a feature is implicit (bins are ordered). Sweep `v = 0..nbins-1` accumulating a running prefix:
`G_L = Σ_{u≤v} G_{k,u}`, `H_L = Σ_{u≤v} H_{k,u}`; then `G_R = G_total − G_L`, `H_R = H_total − H_L`. Evaluate the split-gain (Section 3) at each threshold; keep the best `(feature, bin)`. This is a single linear scan per feature.

**Histogram subtraction trick (parent − sibling).** After a parent node with histogram `Hist_P` is split into children `L` and `R`, the two children partition the parent's instances, so for every bin:
```
Hist_L[b] + Hist_R[b] = Hist_P[b]   ⟹   Hist_R = Hist_P − Hist_L
```
Therefore you **only build the histogram for the smaller child** (fewer instances ⇒ cheaper O(#data_small) build), and obtain the larger sibling for **O(#bins)** by subtraction. This roughly halves total histogram-build work and is the key constant-factor win. LightGBM: *"construct histograms for only one leaf (with smaller #data than its neighbor)… get histograms of its neighbor by histogram subtraction with small cost (O(#bins))."* Requires keeping the parent histogram alive while children are processed.

**Memory layout.**
- Binned feature matrix stored as `uint8` (≤256 bins) — ~4–8× smaller than `float32`, cache-friendly, no per-node re-sort.
- Histograms are contiguous arrays of `{grad: f64/f32, hess: f64/f32}` (or struct-of-arrays) indexed by bin, sized `nfeatures × nbins`. Contiguous per-feature layout gives sequential, vectorizable access.
- **Quantized training** (LightGBM 4.x, "Quantized Training of GBDT", arXiv:2207.09682): gradients/Hessians are themselves quantized to low-bit integers (e.g. int16/int8) so histogram accumulation uses **integer adds**, halving/quartering memory and enabling SIMD; reported near-lossless accuracy with substantial speedups. Worth considering but not essential for v1.

---

## 3. Split gain, optimal leaf weight, Newton derivation

**Regularized objective** at boosting round `t`, tree `f_t` with `T` leaves and leaf weights `w`:
```
Obj^(t) = Σ_i l(y_i, ŷ_i^(t-1) + f_t(x_i)) + Ω(f_t),   Ω(f) = γ·T + ½·λ·Σ_j w_j²   (+ α·Σ_j |w_j| for L1)
```

**Second-order Taylor expansion** of the loss around `ŷ^(t-1)`, with
`g_i = ∂_{ŷ} l(y_i, ŷ^(t-1))`, `h_i = ∂²_{ŷ} l(y_i, ŷ^(t-1))`:
```
Obj^(t) ≈ Σ_i [ g_i f_t(x_i) + ½ h_i f_t(x_i)² ] + γT + ½λ Σ_j w_j²   (constant l(y,ŷ^(t-1)) dropped)
```

**Group by leaf.** A tree assigns every instance to exactly one leaf `j` with constant weight `w_j`. Let `G_j = Σ_{i∈I_j} g_i`, `H_j = Σ_{i∈I_j} h_i`. The objective becomes a sum of independent per-leaf quadratics:
```
Obj^(t) = Σ_j [ G_j w_j + ½ (H_j + λ) w_j² ] + γT
```

**Optimal leaf weight** (set ∂/∂w_j = 0 ⟹ `(H_j+λ) w_j + G_j = 0`):
```
            -G_j
w_j*  =  ───────────                (no L1)
          H_j + λ
```
With **L1 (α)** the numerator is soft-thresholded (proximal solution of the |w| term):
```
            -T(G_j, α)                      ⎧ G - α   if G >  α
w_j*  =  ──────────────  ,  T(G,α) =   ⎨ G + α   if G < -α
              H_j + λ                          ⎩  0      if |G| ≤ α
```

**Optimal objective (structure score)** for a fixed tree structure (plug `w*` back):
```
Obj* = -½ Σ_j  G_j² / (H_j + λ)  +  γT
```

**Split gain.** Splitting one leaf (gradients `G = G_L+G_R`, `H = H_L+H_R`) into children L,R changes the structure score by:
```
              1 ⎡  G_L²      G_R²     (G_L+G_R)²  ⎤
Gain  =  ──── ⎢ ───────  +  ───────  −  ──────────── ⎥  −  γ
              2 ⎣ H_L+λ     H_R+λ     H_L+H_R+λ  ⎦
```
`γ` (`min_split_gain` / `gamma`) is the minimum gain required to keep a split; a split is made only if `Gain > 0`. The bracket term is exactly `loss(L) + loss(R) − loss(parent)` measured in structure score. (With L1, replace each `G²` by `T(G,α)²`.)

---

## 4. Parallelism (and mapping to Rust/rayon, SIMD)

**Within a node (single machine — the case that matters for `pattern-boost`):**
- **Feature-parallel histogram build.** Each thread owns a disjoint subset of features and builds those features' histograms over all instances independently — embarrassingly parallel, no shared mutable state. This is the natural rayon mapping: `features.par_iter().for_each(|f| build_hist(f))` or `par_chunks` over the feature block. Histogram subtraction is done per feature after, also parallelizable.
- **Data-parallel histogram build** (alternative): partition rows across threads, each builds partial histograms over all features, then reduce-sum. Needs a per-thread histogram buffer + a final reduction; more memory but better load balance when feature count is low. rayon `fold`/`reduce` fits this.
- LightGBM uses **both**, and across a single machine relies on feature-parallel + the subtraction trick.

**Distributed (not needed for v1, but the design pattern):**
- *Feature-parallel*: workers hold all data, find local best splits, communicate splits to agree on global best. No data shuffle.
- *Data-parallel*: each worker builds local histograms on its row shard, then **Reduce-Scatter** merges non-overlapping feature histograms (cost `O(0.5 · #feature · #bin)`), giving each worker the global histogram for its features.
- *Voting-parallel*: two-stage voting on top features reduces communication to ~constant.

**Rust specifics.**
- Use rayon for feature-parallel histogram construction; the binned matrix as `Vec<u8>` (column-major / per-feature contiguous) is the hot data structure.
- The inner accumulation loop (`hist[bin].g += g[i]`) has a scatter dependency that defeats auto-vectorization; mitigations used in practice: (a) **multiple histogram replicas** unrolled over disjoint bins to break write dependencies, (b) **quantized integer g/h** so accumulation is integer SIMD, (c) gather/scatter AVX2/AVX-512 only on newer ISAs. The prefix-sum split-gain scan *does* vectorize well (sequential reads). For a first version, plain scalar accumulation + rayon over features gets most of the win; SIMD is a later optimization.

---

## 5. LightGBM-specific tricks: GOSS and EFB

**GOSS — Gradient-based One-Side Sampling.** Observation: instances with large `|gradient|` are under-trained and contribute most to information gain; small-gradient instances are well-trained. Procedure each iteration:
1. Sort instances by `|g_i|` descending.
2. Keep the **top `a·n`** (e.g. `a = top_rate = 0.2`) large-gradient instances.
3. **Randomly sample `b·n`** (e.g. `b = other_rate = 0.1`) from the remaining `(1−a)·n` small-gradient instances.
4. When computing histograms/gain, **amplify the sampled small-gradient instances' g and h by the constant `(1−a)/b`** to compensate for under-sampling and keep the data distribution unbiased.

Net: train on ~`(a+b)·n ≈ 30%` of rows per tree with near-unbiased gain estimates; the paper proves the estimation error is bounded and vanishes as `n→∞`. **Worth implementing?** Medium priority. It's a row-subsampling speedup orthogonal to tree shape; plain stochastic `subsample` (uniform) is simpler and captures much of the regularization benefit. Defer GOSS to a perf pass; the `(1−a)/b` reweighting is the only subtle part.

**EFB — Exclusive Feature Bundling.** In sparse/one-hot data many features are mutually exclusive (rarely nonzero together). EFB merges them into one "bundle" feature to cut `#features` (and thus histogram cost) without losing split information:
1. Build a conflict graph (edge weight = #rows where both features are nonzero).
2. Greedy graph-coloring assigns features to bundles, allowing a small **conflict tolerance** (`max_conflict_rate`) of overlapping nonzeros.
3. Merge by **offsetting bin ranges**: feature `j` with `k_j` bins gets bins `[offset_j, offset_j + k_j)` where offsets are cumulative; the bundle's bin value disambiguates which original feature was nonzero. Optimal bundling is NP-hard; greedy gives a good approximation.

**Worth implementing?** Low priority for `pattern-boost` unless you target high-cardinality sparse/categorical data. It only helps when features are sparse and exclusive (one-hot encodings); on dense numeric data it does nothing. Skip for v1.

---

## 6. Missing-value and sparse handling

**Default/learned direction.** XGBoost/LightGBM learn, per split, a **default direction** for missing values. During split evaluation at a node, missing instances are tentatively sent **both** ways; the direction that yields higher gain is stored on the node. At predict time a missing value follows the learned default direction. Concretely, the gain sweep is run twice (missing→left and missing→right) and the better is kept — this is cheap because missing-instance `G_miss, H_miss` are a single known quantity added to whichever side.

**Dedicated missing bin.** scikit-learn HistGradientBoosting puts NaNs in a **reserved bin** (`missing_values_bin_idx_ = n_bins - 1`). During the split scan it tries assigning that bin to left vs right and keeps the better — equivalent to a learned default direction, integrated into the histogram with no special-casing.

**Sparse / zero handling.** Implicit zeros are treated as a value (often binned with the default direction = the zero side), so sparse matrices need not be densified. EFB (Section 5) further exploits sparsity. For `pattern-boost`: implement the **reserved-missing-bin + try-both-directions** approach — it's the simplest, matches sklearn, and composes cleanly with histograms.

---

## 7. Monotonic constraints

Goal: guarantee the model output is monotone (↑ or ↓) in a constrained feature, globally, by enforcing it within each tree (sums of monotone trees are monotone). Two mechanisms work together:

**(a) Split rejection (local).** When splitting on a monotone-increasing feature, require the optimal child weights to respect order: `w_L* ≤ w_R*` (for a `+1` constraint; reverse for `−1`). If a candidate split violates this, set its gain to `−∞` so it's never chosen. (Note: with `hist`, fewer candidate thresholds exist, so constraints can wipe out all candidates → no split; XGBoost docs advise raising `max_bin`.)

**(b) Bound propagation (global, the essential part).** Local rejection alone is insufficient — descendants splitting on *other* features could still break monotonicity across cousin leaves. So each node carries a `[lower, upper]` bound on permissible leaf weights, propagated downward:
- When a node with constrained feature splits into L (lower feature values) and R (higher), compute the midpoint `m = (w_L* + w_R*) / 2`.
- For a `+1` constraint: every descendant in the **left** subtree gets `upper = min(upper, m)` (capped at m); every descendant in the **right** subtree gets `lower = max(lower, m)` (floored at m). (Reverse L/R for `−1`.)
- For splits on **unconstrained** features, both children simply **inherit the parent's `[lower, upper]`** unchanged.
- Every leaf weight is then **clamped** to its `[lower, upper]`: `w_j* = clip(-G_j/(H_j+λ), lower, upper)`.

Because the left subtree is capped at `m` and the right subtree floored at `m`, no left-side leaf can exceed any right-side leaf — monotonicity holds across the whole tree, and tightens as the tree deepens. Bounds start at `(−∞, +∞)` at the root.

---

## 8. Regularization knobs that matter

| Knob (XGBoost / LightGBM name) | Role | Formula touchpoint |
|---|---|---|
| `lambda` / `lambda_l2` (`reg_lambda`) | L2 on leaf weights; shrinks weights, stabilizes denominators | `H+λ` in `w*` and Gain |
| `alpha` / `lambda_l1` (`reg_alpha`) | L1 on leaf weights; soft-thresholds, induces zero leaves | `T(G,α)` numerator |
| `gamma` / `min_split_gain` (`min_gain_to_split`) | Minimum gain to accept a split; complexity penalty per leaf | `−γ` in Gain |
| `min_child_weight` / `min_sum_hessian_in_leaf` | Min total Hessian `H_j` per leaf; prevents tiny/over-confident leaves | reject split if `H_L<` or `H_R<` threshold |
| `min_data_in_leaf` (`min_child_samples`) | Min instance count per leaf | reject split below count |
| `max_bin` / `max_bins` | #histogram bins; accuracy vs speed/memory | binning resolution |
| `learning_rate` (`eta`) | Shrinkage: `ŷ ← ŷ + η·f_t`; the single most important accuracy knob | applied to leaf outputs |
| `subsample` (`bagging_fraction`) | Row sampling per tree; variance reduction / speed | rows used per tree |
| `colsample_bytree`/`bylevel`/`bynode` (`feature_fraction`) | Column sampling; decorrelates trees | features considered |
| `max_depth` / `num_leaves` | Tree capacity (for `pattern-boost`, depth is fixed at 3) | structure size |

Priority for `pattern-boost` v1: `learning_rate`, `lambda`, `min_sum_hessian_in_leaf` (= `min_child_weight`), `min_data_in_leaf`, `max_bin`, plus `subsample`/`colsample`. `gamma` and `alpha` are secondary.

---

## Design implications for `pattern-boost`

`pattern-boost` builds **depth-3 symmetric/oblivious trees**: at each level *all* nodes share **one** `(feature, threshold)` split (CatBoost-style). This changes split-finding fundamentally — the split must be chosen **jointly across all current leaves at that level**, not greedily per node.

**1. Gain is summed over all leaves at the level.** At level `d`, the tree currently has `2^d` leaves (level 0: 1 leaf; level 1: 2; level 2: 4). For a single candidate `(feature k, threshold v)` applied to *every* leaf simultaneously, each leaf `ℓ` splits into `ℓ_L, ℓ_R`. The level gain is the **sum over leaves** of the per-leaf split gains:
```
LevelGain(k, v) = Σ_{ℓ ∈ leaves(d)}  ½ [ G_{ℓ,L}²/(H_{ℓ,L}+λ) + G_{ℓ,R}²/(H_{ℓ,R}+λ) − G_ℓ²/(H_ℓ+λ) ]  − γ
```
Pick the `(k, v)` maximizing `LevelGain`. This is the oblivious analogue of the per-node argmax: you evaluate the *same* threshold against the histograms of *all* leaves and add up the gains. Crucially, you need **one histogram per (leaf, feature)** at each level, and the prefix-sum sweep gives `G_{ℓ,L}, H_{ℓ,L}` per leaf per bin exactly as in Section 3.

Data structure: a histogram tensor `Hist[leaf][feature][bin] → {g, h, count}`. To find the level split: for each feature `k`, for each bin threshold `v`, sum the candidate-split gain across all `2^d` leaves; track the global best `(k, v)`. The outer loop over features parallelizes cleanly with rayon (`features.par_iter`).

**2. Histogram subtraction trick — still works, per leaf.** Symmetry does **not** break subtraction; it actually amplifies its value. When a parent leaf `ℓ` (from level `d−1`, histogram `Hist_ℓ`) is split by the chosen level-`d−1` threshold into `ℓ_L, ℓ_R`, you:
- Build the histogram of the **smaller** child only (over its instances), `O(#data_small × #features)`.
- Get the sibling by subtraction: `Hist_{ℓ_R} = Hist_ℓ − Hist_{ℓ_L}`, `O(#features × #bins)`.

This applies independently to each of the parent's leaves, so at every level you build only ~half the data into histograms. Because depth is fixed at 3, total levels = 3 and total leaves ≤ 8 — the histogram tensor is tiny and fully cache-resident, and the subtraction trick keeps build cost at roughly `O(3 × #data × #features)` worst case (often half that).

**3. Recommended data structures.**
- **Binned matrix**: `Vec<u8>` (≤256 bins), stored **column-major** (per-feature contiguous) so each feature's histogram build is a sequential pass — best cache behaviour and rayon-shardable by feature.
- **Bin edges**: `Vec<Vec<f32>>` (per feature), computed once via quantile-on-subsample (Section 1); reserve bin index 255 for missing.
- **Histogram**: flat `Vec<HistBin>` of length `n_leaves * n_features * n_bins`, `HistBin = { g: f64, h: f64, count: u32 }` (or SoA: three parallel `Vec`s for SIMD-friendly prefix sums). Keep parent-level histograms alive to enable subtraction.
- **Sample→leaf map**: `Vec<u8>` of length `n` giving each row's current leaf index (0..2^d). Updated in place after each level's split is fixed. Because the tree is oblivious, a row's leaf index at depth 3 is just the **3-bit concatenation of the three split outcomes** — you can even compute leaf assignment as a bitfield, which makes prediction extremely fast (3 comparisons → 3-bit index → table lookup of 8 leaf values). This is a structural advantage of oblivious trees: dense, branch-free inference.
- **Leaf values**: `[f64; 8]` per tree (`w_j* = clip(-T(G_j,α)/(H_j+λ), lower, upper)`), with shrinkage `η` applied.

**4. Constraints/missing/monotonic carry over.** Missing → reserved bin + try-both-directions, evaluated against the *summed* level gain. Monotonic constraints: apply bound propagation per leaf-path; with oblivious trees all leaves share the same feature at a level, so the left/right bound split at midpoint `m` must be done per-parent-leaf and the final 8 leaf weights clamped — slightly more bookkeeping than for asymmetric trees but the same principle.

**5. Practical recommendation order.** v1: quantile binning (subsampled) → per-leaf histograms with subtraction → level-summed gain argmax → Newton leaf weights with `λ`, `min_hessian`, `min_data` → shrinkage + row/col subsampling → missing-bin handling. Defer GOSS, EFB, SIMD/quantized-int histograms, and distributed parallelism to a performance pass. The oblivious structure means the histogram tensor is small (≤8 leaves), so memory pressure is low and the dominant cost is the per-level full-data histogram build — exactly what rayon feature-parallelism + the subtraction trick target.

---

## Sources

- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree (Ke et al., NeurIPS 2017, PDF)](https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)
- [LightGBM Features documentation](https://lightgbm.readthedocs.io/en/latest/Features.html)
- [XGBoost — Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
- [XGBoost — Monotonic Constraints](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html)
- [scikit-learn binning.py source (BinMapper)](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_hist_gradient_boosting/binning.py)
- [scikit-learn — Histogram-Based Gradient Boosting (DeepWiki)](https://deepwiki.com/scikit-learn/scikit-learn/4.3-histogram-based-gradient-boosting)
- [scikit-learn — HistGradientBoostingRegressor docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [How XGBoost and LightGBM enforce monotonic constraints (Carson Yan, TDS)](https://medium.com/towards-data-science/how-does-the-popular-xgboost-and-lightgbm-algorithms-enforce-monotonic-constraint-cf8fce797acb)
- [Quantized Training of Gradient Boosting Decision Trees (arXiv:2207.09682)](https://arxiv.org/pdf/2207.09682)
- [L1, L2 Regularization in XGBoost Regression (Albert Um)](https://albertum.medium.com/l1-l2-regularization-in-xgboost-regression-7b2db08a59e0)
- [LightGBM Demystified — the math behind the algorithm](https://www.intelligentmachines.blog/post/lightgbm-demystified-understanding-the-math-behind-the-algorithm)

### Key formulas (quick reference)
- Leaf weight: `w* = -G/(H+λ)` (L1: numerator → soft-threshold `T(G,α)`)
- Split gain: `½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ)] − γ`
- Structure score: `Obj* = -½ Σ_j G_j²/(H_j+λ) + γT`
- Histogram subtraction: `Hist_R = Hist_P − Hist_L`
- GOSS amplification: small-gradient sampled instances scaled by `(1−a)/b`
- **Oblivious level gain (pattern-boost): `Σ_{leaves ℓ} [per-leaf split gain at (k,v)] − γ`, maximized jointly over `(feature k, threshold v)`**
