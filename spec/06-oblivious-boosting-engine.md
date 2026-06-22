## 06 — The oblivious boosting engine

This is the performance-critical core. It owns the trained-model structs (`ObliviousTree`, `Split`, `Model`), the public handle (`Booster`, `FitSpec`, and the config struct), the histogram accumulator type (`Hist`), and the algorithms that turn binned data + a `Loss` into a `Model`: the depth-3 oblivious level-wise Newton split-finder, the histogram engine (quantized integer g/h, subtraction trick, per-thread lock-free accumulation), multi-step Newton leaf estimation with Armijo backtracking, MVS + subsampling, regularization, and the boosting loop. It consumes `Loss` (§05), the binning grids and missing handling (§03), and `MonotoneMap` + the per-level joint leaf-clamp + the credibility floors (§07); it produces a `Model` that §08 purifies and §10 scores/serializes.

It serves all three aims at once: **accurate** (XGBoost/LightGBM-parity Newton gain and exact leaf weights, plus the gap-closers MVS, multi-step Newton, fully-corrective refit hooks); **decomposable** (one shared split per level ⇒ ≤3 distinct raw features per tree ⇒ I1/I2 by construction, never by post-hoc repair); **fast** (row-count-independent split search after binning, a tiny ≤8-leaf histogram tensor, integer SIMD accumulation, subtraction trick, rayon feature-parallelism).

### 6.1 Configuration

The booster is builder-configured; `Config` is the owned config struct, validated once in `Booster::fit` (returning `PbError::InvalidConfig`). Defaults reflect the depth-3 cap's shift toward more, lower-LR trees. The leaf-credibility floors (`min_sum_hessian_in_leaf`, `min_data_in_leaf`, `path_smooth`) are **owned by §07** (`CredibilityFloor`); `Config` references them rather than redefining them (the §03 grid-build floor `min_data_per_bin` is a distinct, separately-owned knob).

```rust
pub struct Config {
    pub n_trees: u32,            // 1000  (single source of truth; §12 sklearn forwards this)
    pub learning_rate: f32,      // 0.05  (auto-scaled by data size later; fixed for now)
    pub lambda: f32,             // 1.0   L2 on leaf weights (the `λ` in w*, gain)
    pub floors: CredibilityFloor, // §07-owned: min_sum_hessian_in_leaf (1e-3), min_data_in_leaf (20), path_smooth (0.0)
    pub max_delta_step: Option<f32>, // None ⇒ fall back to Loss::max_delta_step() (Poisson ⇒ 0.7); Some(v>0) caps |w*|
    pub subsample: f32,          // 1.0   Bernoulli row sampling per tree
    pub colsample_bytree: f32,   // 1.0   feature sampling per tree
    pub colsample_bylevel: f32,  // 1.0   additional per-level feature sampling (rsm); §09 ensemble-diversity path
    pub colsample_bynode: f32,   // 1.0   additional per-node (per-candidate-scan) feature sampling
    pub min_split_gain: f32,     // 0.0   (`gamma`) graceful early-termination when best LevelGain < min_split_gain
    pub l1_leaf: f32,            // 0.0   (`alpha`) L1 leaf reg via soft-thresholding (off by default)
    pub sampling: Sampling,      // Mvs { lambda_mvs: 1.0 } by default; or Bernoulli / Bayesian / None
    pub leaf_newton_steps: u8,   // 1     multi-step Newton (>1 only after benchmark)
    pub hist_precision: HistPrecision, // QuantizedI32 (default) | FullF64
    pub lr_schedule: LrSchedule, // Constant (default); reserved hook, LR×n_trees is CORE (§14)
    pub early_stopping_rounds: Option<u32>, // Some(50) — patience once early stopping is enabled; §12 forwards this default
    pub validation_fraction: Option<f32>, // None ⇒ early stopping DISABLED (default; sklearn-familiar); Some(frac) ⇒ seeded internal holdout
    pub accel: Accel,            // None (Nesterov is v2, benchmark-gated → §09)
}
pub enum Sampling { None, Bernoulli, Mvs { lambda_mvs: f32 }, Bayesian { temperature: f32 } }
pub enum HistPrecision { QuantizedI32, FullF64 }
pub enum Accel { None, Nesterov { momentum: f32 } }
pub enum LrSchedule { Constant, Linear { end_factor: f32 }, Step { gamma: f32, every: u32 } }
```

`FitSpec` (§2.9) carries the per-call data-shaped arguments (`loss`, `weight`, `exposure`, `monotone`, `interaction`, `seed`); `Config` carries the optimizer knobs. The interaction whitelist now travels as `FitSpec.interaction: InteractionPolicy { max_order, groups }` (replacing the bare scalar `max_interaction_order`), threaded into the split-finder's feature-budget guard. The public entry point is `Booster::fit(&self, x: &BinnedMatrix, y: &[f32], spec: &FitSpec) -> Result<Model, PbError>`.

`max_delta_step` is an `Option<f32>` defaulting to `None`; when `None` it falls back to `Loss::max_delta_step()` (so Poisson is stabilized at 0.7 without the user opting in), and `Some(v)` with `v > 0` caps `|w*|` explicitly. Hessian-inflation is **rejected** as a stabilizer (it would perturb the `QuantGradHess` histograms); the clamp is purely leaf-stage.

### 6.2 The split-finder: oblivious level-wise summed Newton gain

**Decision (locked):** Newton/L2 summed gain with exact leaf weights `w* = −G/(H+λ)`. CatBoost's Cosine default is rejected (it forfeits exact leaf weights, which I2's ThreeWayEqual check and the rating-table reading both require). Cosine survives only as a research A/B, never the default.

A tree is grown one level at a time, depth 1→3. At level `d` the tree has `2^d` leaves. A candidate is a `(axis, bin_le)` pair (test `bin ≤ bin_le`), where `axis` indexes a `BinnedMatrix` column with known `AxisProvenance`. The candidate is applied to **every** current leaf simultaneously (the oblivious constraint). The level objective is the sum of per-leaf Newton split gains:

```
LevelGain(axis, v) = ½ · Σ_{ℓ ∈ leaves(d)} [ G_{ℓL}²/(H_{ℓL}+λ) + G_{ℓR}²/(H_{ℓR}+λ) − G_ℓ²/(H_ℓ+λ) ]
```

where, within leaf `ℓ`'s per-axis histogram, the prefix sweep gives `G_{ℓL} = Σ_{b≤v} G_{ℓ,axis,b}`, `H_{ℓL}` likewise, and `G_{ℓR} = G_ℓ − G_{ℓL}`. The level keeps the single `(axis, bin_le)` maximizing `LevelGain` over all axes and borders.

**Feature-budget guard (I1).** A candidate axis is admissible at level `d` only if `provenance[axis].raw` is not already used by an earlier level *and* (with `spec.interaction: InteractionPolicy { max_order, groups }`, §07) the resulting raw-feature set stays within `max_order` and an allowed `groups` whitelist entry. This is checked while scanning, so the chosen split can never push a tree past 3 distinct raw features. Combination-CTR axes (>3 raw features behind one column) are forbidden by §04 and never appear in `provenance`.

**Missing values (§03).** Bin 0 is the reserved missing bin. Each candidate is evaluated twice — missing sent left vs. right — by adding the known `(G_miss, H_miss)` of that leaf to whichever side; the better direction is **learned** and recorded in the explicit `Split.missing_left: bool` field (the canonical missing-direction carrier, cited by §03/§08). No special-casing in the inner loop. The learned direction is then honored **identically** at every per-level test — during this scan, in the sample→leaf update (§6.6), and in §08/§10 scoring — by the single canonical low/left bit:

```
low = if bin == 0 { split.missing_left } else { bin <= split.bin_le };
// the leaf-index bit at that level = low as usize
```

i.e. the missing bin routes per its learned `missing_left`, never silently left; honoring this at accumulation/scoring is **required** for tree/table equality (else missing always routes left, breaking I2).

**Monotone & min-leaf vetoes (§07).** Candidates whose post-split leaf weights `w* = −G/(H+λ)` violate the per-level joint monotone clamp, or whose either side has `H < floors.min_sum_hessian_in_leaf` or `count < floors.min_data_in_leaf`, get `LevelGain = −∞`. If **no** admissible candidate has finite gain at a level, or the best finite `LevelGain < min_split_gain` (the `gamma` floor), the tree **terminates early** at `depth < 3` (a legitimate lower-order fANOVA outcome — `PbError` is *not* raised). `LevelGain ≤ 0` for all candidates also terminates the tree.

**Column subsampling at the scan.** The admissible axis set is intersected with the per-tree (`colsample_bytree`), per-level (`colsample_bylevel`/`rsm`), and per-node (`colsample_bynode`) feature masks before the gain scan; all three draws are deterministically re-seeded (§6.7) so masks are thread-count-independent. `feature_weights` is not a separate knob — it is expressible through the §07 `AdmissionPrior` soft-prior seam.

**Complexity.** Per level: histogram build O(n_used · n_axes_sampled) (halved by subtraction), then a gain scan O(2^d · n_axes_sampled · n_bins) — **independent of row count**. With ≤3 levels and a tiny ≤8-leaf tensor, the dominant cost is the histogram build; total per tree ≈ O(3 · n · F) worst case, typically half that.

### 6.3 The histogram engine

The per-level histogram is a flat tensor indexed `[leaf][axis][bin]`, kept as struct-of-arrays for SIMD-friendly prefix sweeps. The default accumulator is **quantized integer** g/h. `Hist` is the single §06-owned accumulator type; §11 (`FeatureHist`/`LevelHists.arena`) and §02's `Backend` reference *this* type and width (the earlier `Vec<i32>` / `HistogramSet` names are retired).

```rust
/// SoA histogram: one (g,h,count) triple per (leaf, axis, bin), row-major in that order.
/// i64 bin accumulators (the §06-owned width). Counts stay u32.
pub struct Hist { pub g: Vec<i64>, pub h: Vec<i64>, pub count: Vec<u32>,
                  pub n_leaves: usize, pub n_axes: usize, pub n_bins: usize }
```

**Accumulator width (no-overflow proof).** Bin sums are `i64`. Each quantized value satisfies `|g_q| ≤ i32_budget < 2^31`, so a leaf-axis-bin sum over all rows is bounded by `n_rows · 2^31`; this stays below `i64::MAX = 2^63 − 1` for any `n_rows < 2^32` (the `n_rows: u32` cap, §6.5/§2). The bound is asserted once per round and is the boundary test for the accumulation; `i32` accumulators are rejected because they overflow on large `n` and would panic under `overflow-checks` (breaking the no-panic gate).

**Quantized integer g/h (default; the reproducibility *and* ~2× speed mechanism).** Full-precision `GradHess` is quantized once per boosting round into `QuantGradHess { g_q: Vec<i32>, h_q: Vec<i32>, scale }` (§2.3). The scale is `g_scale = (i32_budget) / max_i |g_i|` (h likewise), chosen per round so the largest magnitude maps near the i32 budget without overflow; quantization uses **stochastic rounding** (round up with probability equal to the fractional part, drawn from the deterministically re-seeded PRNG, §6.7) so the integer sums are *unbiased*. Histogram accumulation is then `hist.g[idx] += g_q[i] as i64` — an integer add. **Integer addition is associative**, so the accumulated sums are *identical regardless of thread count or accumulation order* — this is precisely the I2 bit-reproducibility lever (§1 [GATE]) and the speed lever, the same mechanism. Gains and trial leaf weights during the split scan are computed by converting bin sums back to f32 via the scale.

**Mandatory full-precision leaf refit.** Quantized sums are used *only* to choose structure. Once a tree's 3 splits are fixed, the 8 leaf weights are recomputed from the **full-precision** `GradHess` (§6.4). This is mandatory on log-link objectives (Poisson/Gamma/Tweedie), where quantization bias on the leaf value would distort relativities; `HistPrecision::FullF64` (f64 accumulators, no quantization) is offered as a slower exact-reference path and for the determinism gate's cross-check. (§02's `Backend` contract reads "quantized-integer *or* fixed-order float-fold accumulation" to admit this path.)

**Subtraction trick.** When a parent leaf is split, build the histogram only for the smaller child (over its rows), then obtain the sibling by `Hist_R = Hist_parent − Hist_L` in O(n_axes · n_bins). Parent histograms stay alive until both children are built. With integer accumulators the subtraction is exact.

**Per-thread lock-free accumulation (§11).** Rows are partitioned into fixed-size chunks (a fixed `CHUNK_ROWS` constant, never `rayon::current_num_threads()`); each rayon worker accumulates into a private, cache-line-padded `Hist`; partials are reduced in **fixed chunk-index order** (never rayon steal-order `reduce`/`sum`). Integer partials make the reduction order-independent anyway, but fixed-order reduction is retained so the `FullF64` path is also reproducible. No `Mutex` touches a histogram.

**Hot-loop no-panic policy.** The accumulation and prefix-sweep inner loops are on a fallible path, so they must not panic. The policy: either index through `slice.get(i).ok_or(PbError::Internal { what })?`, or confine raw indexing to a `#[allow(clippy::indexing_slicing)]`-scoped helper fn carrying a `// JUSTIFIED:` bounds proof (the bin id is `< n_bins` by construction, the leaf index `< n_leaves`, the row index `< n_rows`) plus a dedicated boundary test. Integer overflow is caught by `overflow-checks = true` in all profiles (not by a crate-blanket `arithmetic_side_effects = "deny"`); clippy `arithmetic_side_effects` is scoped to the modules that need the proof, and float arithmetic is exempt.

### 6.4 Leaf estimation: exact Newton, multi-step, Armijo

For the fixed 8-leaf structure, the base leaf weight is the exact Newton step, then shrunk:

```
w*_j = clip( −SoftThreshold(G_j, l1_leaf) / (H_j + λ),  lower_j, upper_j ),   leaf_j = lr_t · w*_j
```

`G_j, H_j` are **full-precision** leaf sums (§6.3); `[lower_j, upper_j]` come from the §07 monotone bound propagation (defaults `(−∞, +∞)`); `lr_t` is `learning_rate` modulated by `lr_schedule`. With `l1_leaf = 0` (default) `SoftThreshold(G, 0) = G` and the formula is the plain Newton step; `l1_leaf > 0` applies soft-thresholding `SoftThreshold(G, α) = sign(G)·max(|G|−α, 0)` for sparser, more readable tables. `path_smooth > 0` (from §07 `CredibilityFloor`) blends toward the parent: `G_j ← G_j + path_smooth · w_parent · H_j` style parent-shrinkage (exact, value-level). A non-`None` effective `max_delta_step` (config override or `Loss::max_delta_step()` fallback) additionally clamps `|w*_j|` for log-link Newton stability.

**Multi-step Newton (`leaf_newton_steps > 1`).** For non-quadratic losses (Gamma/Tweedie/MAE) the single Newton step is inexact. Each extra step re-derives `(g, h)` at the *current* leaf-adjusted raw scores (only the 8 leaf rows change, structure frozen) and takes another Newton step on the per-leaf aggregated sums. This touches only the 8 leaf values of a fixed structure ⇒ **exactness untouched** (still a constant-cell oblivious tree). **Armijo backtracking** guards ill-conditioned Tweedie: halve the step while the loss fails the sufficient-decrease condition `L(F + α·δ) ≤ L(F) + c₁·α·⟨g, δ⟩` (c₁ = 1e-4). Default `leaf_newton_steps = 1` (the gain is ~nil on squared error); raise only after the §14 benchmark.

Leaves are stored as `[f32; 8]` indexed `b0 | b1<<1 | b2<<2`; unused tail entries (depth < 3) are 0.0. The `ObliviousTree` invariant — distinct raw features across splits equals `depth` — is asserted at construction (`PbError::InvariantViolated { FeatureBudget }`).

**`wht8` screening emit (off the hot path).** Immediately after the 8 leaf values `w* = −G/(H+λ)` are finalized for a tree, the engine invokes **`wht8`** — a frozen O(8) Walsh–Hadamard / Möbius transform that maps the tree's 8 leaf values to its 8 fANOVA coefficients (1 constant + 3 main `c_i` + 3 pairwise `c_ij` + 1 triple `c_123`) under that tree's per-cut `w`-marginals — and feeds the result to §07's online per-support, per-order screening accumulator. This is exact and O(8); it runs **off the hot histogram path** (once per finished tree, not per row/bin/candidate) so it imposes no cost on the split scan. **It reads only the finished leaves**: it does not touch the split-finder (§6.2), the histogram engine (§6.3), the leaf values, or the tree shape — the trained tree is bit-identical with or without the emit, so it is exactness- and determinism-neutral. `wht8` is owned/registered under §07 (where its kernel and the running accumulator live); §06 only invokes it here at leaf estimation.

**Critical caveat (the screening signal is NOT the audited Sobol).** The per-tree coefficients live on each tree's *own* 2-point grid under that tree's `w`-marginals; trees cut different borders, so you **cannot** sum coefficients across trees — the summed per-tree variance drops cross-tree covariance. The `wht8`-derived per-order variance is therefore a **screening signal, not the audited ensemble Sobol**, and must **never** touch the §08 invariant gates (ThreeWayEqual / VarianceSum / Purity / Reconstruction), which stay on the merged-grid purified bank. Under `RefMeasure::Joint` the clean product form degrades to a heuristic. It is a **soft prior** that never hard-gates, hence exactness- and determinism-neutral by construction. The idea that `wht8` could *replace* §08 Lengerich purification is explicitly **rejected** — it cannot cross the merged-grid alignment, and it is a screening front-end, not the purifier.

### 6.5 Sampling: MVS + row/column subsampling

**Decision:** MVS (Minimal Variance Sampling) over GOSS — it provably dominates GOSS on split-score variance. Per row, sampling probability

```
p_i = min( 1,  sqrt( g_i² + lambda_mvs · h_i² ) / μ )
```

(the regularizer weights the **hessian** term `λ·h_i²`). `μ` is set so the expected sampled count equals `subsample · n`; sampled rows are reweighted by `1/p_i` to keep gain estimates unbiased. Draws use the deterministically re-seeded PRNG (§6.7), partitioned by row index so the selection is identical across thread counts. `Bernoulli` (uniform), `Bayesian { temperature }` (Bayesian bootstrap — per-row Gamma(1/temperature) weights), and `None` are the simpler alternatives. Column subsampling (`colsample_bytree`/`bylevel`/`bynode`, §6.2) restricts the admissible axis set, also deterministically re-seeded. Both cut histogram work and decorrelate trees.

### 6.6 The boosting loop

```rust
pub fn fit(&self, x: &BinnedMatrix, y: &[f32], spec: &FitSpec) -> Result<Model, PbError> {
    let cfg = self.config.validate()?;                  // PbError::InvalidConfig
    let offset = spec.exposure.map(log_offset);         // §03; F0 anchored to base level
    let f0 = spec.loss.init_score(y, weight, offset);   // link(weighted mean) — the fANOVA intercept
    let mut raw = vec![f0; x.n_rows as usize];           // + offset per-row if present
    let mut trees = Vec::with_capacity(cfg.n_trees as usize);
    // Early stopping is opt-in (R-EARLYSTOP): Some only when cfg.validation_fraction = Some(frac).
    let mut early_stop = cfg.validation_fraction
        .map(|frac| EarlyStop::carve_holdout(x, y, weight, frac, spec.seed, cfg.early_stopping_rounds));
    // ^ seeded slice via reseed(spec.seed, 0, Stage::Holdout, 0); held-out rows excluded from every tree's sampling.
    for t in 0..cfg.n_trees {
        spec.loss.grad_hess(y, &raw, weight, &mut gh)?;  // one pass w.r.t. raw score; fallible (Result)
        let mut rng = reseed(spec.seed, t, Stage::Sample, 0); // splitmix64_mix → Pcg64::seed_from_u64
        let rows = sample_rows(&gh, &cfg, &mut rng);      // MVS / Bernoulli / Bayesian (reweighted)
        let axes = sample_cols(x, &cfg, t);               // colsample_bytree; re-seeded by (seed,t,Cols,0)
        let qgh  = quantize(&gh, &rows, &mut rng);        // stochastic rounding (if QuantizedI32)
        let tree = grow_oblivious_tree(x, &qgh, &gh, &rows, &axes, spec, &cfg)?; // §6.2–6.4
        update_raw(&mut raw, x, &tree)?;                  // += lr_t · tree.lookup(x_i); panic-free; missing honored (low rule)
        trees.push((1.0, tree));                           // alpha = 1.0 (Nesterov/ensemble adjust later)
        // early stopping only when cfg.validation_fraction = Some(frac); else no-op (R-EARLYSTOP)
        if let Some(es) = early_stop.as_mut() {
            if es.update(spec.loss.deviance(y_hold, &raw_hold, w_hold)?) { break; } // deviance on the seeded holdout; fallible (?)
        }
    }
    // best_iter = es.best_iter+1 when early stopping fired; else all n_trees are kept (R-EARLYSTOP).
    let keep = early_stop.as_ref().map_or(trees.len(), |es| es.best_iter + 1);
    Ok(Model { f0, trees: trees.truncate_to(keep), grids: x.grids.clone(),
               provenance: x.provenance.clone(), link: spec.loss.link(),
               mode: ExactnessMode::Exact, schema_version: SCHEMA })
}
```

`F0 = link(weighted mean)` is computed in **score space** (never `log(mean y)`) and stored as a scalar — it *is* the fANOVA `f₀` term, never a "tree 0." Each `grad_hess` call is fallible (`fn grad_hess(..) -> Result<(), PbError>`, §2.4) so the §12 `PyLoss` path degrades to a typed error rather than `.expect`; the engine threads it with one `?`. The per-round RNG is produced by **deterministic re-seeding** (§6.7), not a persisted stream, so each work unit's draws are position-stable across thread counts.

**Early stopping (R-EARLYSTOP, opt-in).** Early stopping is gated on `Config.validation_fraction: Option<f32>` and is **disabled by default** (`None` — sklearn-familiar; the contradictory "internal holdout by default" is removed). When `validation_fraction = None`, the deviance detector is a no-op and all `n_trees` are kept. When `validation_fraction = Some(frac)` (`0 < frac < 1`, else `PbError::InvalidConfig`), a **deterministic seeded internal holdout** of `frac · n` rows is carved from the training rows by the frozen seed (re-seeded `(seed, 0, Stage::Holdout, 0)`, so the slice is identical across thread counts), held out from every tree's sampling, and the detector tracks deviance **on that holdout**. Early stopping uses the strictly-proper **deviance** (§05), not RMSE: `early_stop.update(dev)` tracks the best holdout deviance seen and its iteration `best_iter`; training stops when `t − best_iter ≥ early_stopping_rounds` (default `Some(50)`, the patience window once stopping is enabled). The exported `Model.trees` is truncated to the exact `best_iteration` prefix — `best_iteration` is load-bearing for the §08 accumulator, which **must** use the same prefix (an enforced contract). The §12 Python `validation_fraction` kwarg defaults to `None` and forwards to this field unchanged. Each tree carries `alpha = 1.0`; the `(alpha, tree)` shape leaves room for Nesterov mixing and ensemble averaging (§09) without changing scoring (`raw = f0 + offset + Σ alpha_t · tree_t.lookup(x)`). The loop emits `ExactnessMode::Exact`; any technique that bends I1/I2 lives in §09 behind the firewall, never here.

**Sample→leaf update (missing honored, R-MISSING).** `update_raw` (and any sample→leaf routing during fit) computes the leaf index using the *same* canonical per-level low/left bit as the split scan (§6.2) and §08/§10 scoring: for each level's `split`, `low = if bin == 0 { split.missing_left } else { bin <= split.bin_le }`, and the leaf-index bit at that level is `low as usize` (assembled `idx = b0 | b1<<1 | b2<<2`). The reserved missing bin (bin 0) therefore routes per its **learned** `missing_left`, never silently left; routing missing identically at fit-time accumulation and at scoring is what makes the tree, the purified tables, and the Shapley sum agree (I2 / ThreeWayEqual). The bin lookup is panic-free (`slice.get(..).ok_or(PbError::Internal { .. })?` or the `// JUSTIFIED`-scoped helper, §6.3).

### 6.7 Determinism

Every randomized decision — binning subsample (§03), MVS/Bernoulli/Bayesian row draws, column draws, stochastic-rounding direction — is driven by the single `seed: u64` through a named, versioned `Pcg64`, **deterministically re-seeded per work unit**: `Pcg64::seed_from_u64(splitmix64_mix(base, round, stage, block))`, where `base = seed`, `round` is the boosting iteration, `stage` is a fixed tag (`Sample`/`Cols`/`Quantize`/`Holdout`/...), and `block` is the chunk/partition index. This is **not** a "splittable" PRNG (which is unimplementable as named); the frozen `splitmix64` mix gives independent, position-stable streams so row `i`'s draw does not depend on thread count. Combined with integer histograms (order-independent sums) and fixed `CHUNK_ROWS`-indexed reductions on the f64 path, this makes the trained `Model` **bit-identical** across `n_threads ∈ {1, 2, 8}` — the §1 [GATE].

### 6.8 Testing

- **Math unit tests:** `LevelGain` and `w* = −G/(H+λ)` checked against closed-form on hand-built histograms; multi-step Newton converges to the analytic leaf optimum on Gamma; Armijo never increases loss; soft-thresholding (`l1_leaf > 0`) matches the closed-form prox.
- **No-overflow boundary test:** the largest-magnitude quantized sum at `n_rows = u32::MAX` (bound check) stays within `i64`; the hot-loop helper's `// JUSTIFIED` bounds are exercised at the boundary bins/leaves.
- **Invariant (I1):** a property test over fitted models asserts every `ObliviousTree` has `depth ∈ 1..=3` and exactly `depth` distinct raw features; early-termination (no valid split, or `LevelGain < min_split_gain`) produces a valid lower-depth tree, not an error.
- **Reproducibility [GATE]:** train at `n_threads ∈ {1, 2, 8}`, assert byte-equality of the serialized model; assert quantized-histogram sums equal across thread counts; assert the `QuantizedI32` and `FullF64` paths agree on chosen splits within tolerance.
- **Subtraction trick:** assert `Hist_parent == Hist_L + Hist_R` exactly (integer) for every split.
- **Equivalence:** the leaf-refit-from-full-precision path matches a brute-force per-leaf Newton solve; `update_raw` via 3-bit lookup equals a reference scan of the splits.

### 6.9 Open fork

**Fully-corrective leaf refit and Nesterov acceleration are default-off and benchmark-gated** (owned by §09; the `accel` and refit hooks live here in `Config`/`Model.trees` alphas). Both are exactness-preserving (linear in leaf values / linear momentum mix) and shrink tree count → smaller tables, but their cost must be measured on pricing data before defaulting. **Recommended default: off** for v1; revisit per §14. No other fork is open in this section — Newton gain, exact `w*`, quantized histograms with mandatory full-precision refit, and MVS are all locked.
