## 08 — The explainability engine

> Owns: `EffectTable`, `FeatureSet`, `TableBank`, `RefMeasure`, the §08-local aliases `Tensor`/`AxisId`; the accumulate→purify→tables pipeline; the default `RefMeasure::ProductMarginals { laplace }`; Sobol importances; exact interventional SHAP / Faith-Shap ≤order-3 as O(1) table reads; the **implementation** of the five `Invariant` checks; complete-for-inference vs pruned-for-display; post-hoc `w` recomputation; the per-cell effective-support annotation and the optional display-only inner-bag SE-band annotation; the `TableBudget`/`OverflowPolicy` table-size firewall (08.10) and the `PbError::TableBudget` failure. Uses (does not own): `Model` + `ModelSchema.cat_encoders` (§06/§04), `ServeBinnedMatrix` (§03; the frozen-encoder serve matrix the bank is audited on), I1/I2 + `ExactnessMode` (§3), `BorderGrid`/`AxisProvenance` (§03), `Split.axis: u32`/`Split.missing_left` (§02/§06), `PbError`/`Invariant` (§2.8).

This is the section where the load-bearing thesis is realized: a trained `Model` is turned into the `TableBank` that **is** the model in a second view, and the equality of the two views is enforced as a build gate. There is no separate "explanation" — there is one tensor bank, raw and purified, with a proof they are equal.

### 08.1 Local aliases & the effect tensor

Tables live on a per-axis **merged grid**: for axis `i`, an **explicit missing cell at index 0** plus the finite-interval cells of the sorted union of every split border realized on `i` across the ensemble (research/03 §4.2), terminated by `±∞`. This is the minimal exact grid — the ensemble is piecewise-constant with breakpoints only at realized borders, so a finer grid adds zero information and a coarser one cannot represent some tree losslessly. The missing cell is **not** an interval endpoint: it is a separate, structurally-disjoint cell mirroring the reserved missing bin (bin 0, §03) so that a tree's learned `Split.missing_left` routing is representable losslessly rather than collapsed into the first finite interval.

```rust
/// A merged-grid axis: `borders` is the sorted union of realized split borders on
/// one raw feature (the FINITE breakpoints). The per-axis tensor extent is
/// `cells == 1 + n_finite_intervals` where `n_finite_intervals == borders.len() + 1`
/// (the half-open finite intervals) — i.e. one EXPLICIT missing cell at index 0
/// PLUS the finite-interval cells. It is NOT `borders.len() + 1`: the leading +1 is
/// the missing cell, mirroring bin 0 of the underlying `BorderGrid` (§03).
pub struct AxisId { pub raw: FeatureId, pub borders: Vec<f32>, pub cells: u32 }
// invariant: cells == borders.len() as u32 + 2  (1 missing + (borders.len()+1) finite intervals)

/// Row-major dense effect tensor over 1..=3 axes (cell counts as extents; each
/// extent counts the explicit missing cell at index 0).
/// `at(idx)` is bounds-checked; no indexing-slicing on any path (§1 panic policy).
pub struct Tensor { pub extents: SmallVec<[u32; 3]>, pub data: Vec<f64> }
```

**Cell-index convention (canonical, R-MERGEDCELL).** Along each axis, cell `0` is the missing cell; cells `1..=borders.len()+1` are the finite half-open intervals (cell `k+1` is the `k`-th interval `[θ_{k-1}, θ_k)`, with `θ_{-1} = −∞`, `θ_{n} = +∞`). A tree contributes to an axis's **missing cell** iff that tree's split on the axis routes missing into the covering leaf (per `Split.missing_left`, §02/§06), and to a **finite cell** iff `bin <= bin_le` selects it. An empty missing cell (no tree routes missing there, or no training row is missing on that axis) is a real, zero-mass cell — flagged by the per-cell support annotation (08.7), never elided.

`Tensor` is `f64` even though the core trains in `f32`: purification accumulates many signed mass-moves and we want the reconstruction residual to sit at `f64` epsilon, not `f32` epsilon, so the reconstruction gate (08.6) has headroom. Leaf values and the scores summed at inference are `f32` (the trained core), but the purified `Tensor` carried by every `EffectTable` is `f64`. `EffectTable.variance` is the `w`-weighted variance `σ²(f_u)` cached at purification time for Sobol importances and the variance-sum gate. `FeatureSet` is the sorted, distinct **raw** id set (size 1..=3) keyed off `AxisProvenance.raw`, so two encoded axes of the same categorical collapse to one feature set — provenance, never column index, defines `u`. Axis lookups index `provenance[s.axis as usize]` through the fixed-width `Split.axis: u32` (§02), so the merged grid is platform-independent.

**Hot-loop panic policy (§1).** Tensor `at(idx)` and the cell-walk in accumulation/purification are the hot paths. They take the agreed no-panic form: either `slice.get(i).ok_or(PbError::Internal { what: "tensor index oob" })?`, or a scoped `#[allow(clippy::indexing_slicing)]` helper carrying a `// JUSTIFIED:` bounds proof (`idx < Π extents`, the strides are built from `extents`) plus a boundary test. There is no bare indexing or unchecked arithmetic on any explainability path; cell-offset arithmetic is integer and runs under `overflow-checks = true` (§1, all profiles), with the clippy `arithmetic_side_effects` lint scoped to those helpers, never crate-blanket, and `f64` mass arithmetic exempt.

### 08.2 Accumulation: ensemble → raw tensors

For each tree `(alpha, t)` in `Model.trees`, its support `u_t` is the set of distinct `provenance[s.axis].raw` over its splits (size = `t.depth`, ≤3 by I1). The tree is a piecewise-constant function of `u_t`; we expand it onto the merged grid of `u_t` (each axis = 1 explicit missing cell + the finite intervals, 08.1) and add `alpha · leaf` into `T_raw[u_t]`, allocating that raw tensor lazily on first touch. The per-level low/left bit is computed by the **canonical routing rule** shared with §06/§10: for a split on axis `a` with threshold `bin_le` and learned default `missing_left`, a cell routes "low" iff `if cell_is_missing { split.missing_left } else { finite_bin <= split.bin_le }`. So a tree contributes to an axis's **explicit missing cell** exactly per `Split.missing_left` and to its finite cells per `bin <= bin_le` — the merged-grid missing cell receives the *same* leaf the engine assigned at scoring, never a silent route-left. This identical honoring of `missing_left` is what makes tree-sum == table-sum hold (I2 / ThreeWayEqual); collapsing missing into the first finite interval would break it.

```rust
fn accumulate(model: &Model, x: &ServeBinnedMatrix, grids: &MergedGrids) -> Result<RawBank, PbError>;
```

**Audit-on-serve (canonical, R-CATSERVE).** Accumulation — and every gate that consumes binned data — runs over a **`ServeBinnedMatrix`**: raw categoricals re-encoded through the model's **frozen full-data `CatEncoder`s** (`ModelSchema.cat_encoders`, §04), never the out-of-fold/prefix `TrainBinnedMatrix` used during fitting. I2 lossless-equivalence is between the **served model function** (frozen encoders) and the `TableBank` accumulated from it, both on a `ServeBinnedMatrix`; reusing the noisy training encodings would audit a model that is never deployed. Numeric binning is fold-independent, so the numeric columns of Train and Serve coincide and only categorical axes differ; the engine therefore must build (or be handed) a `ServeBinnedMatrix` before accumulating. `explain()` (08.8) takes a `ServeBinnedMatrix`.

Expansion is a broadcast: a tree cell spanning border interval `[θ_a, θ_b)` is constant, so it maps losslessly onto every merged sub-cell it covers (the missing cell is handled by the routing rule above, not by interval coverage). A degenerate 2-feature tree (early-terminated at depth 2) contributes to a 2-way raw tensor; a 1-feature tree to a main-effect tensor. After this pass, `f0_raw = model.f0 as f64` and `Σ_u T_raw[u](x) = F_ens(x) − offset` **identically** — this is the exact-accumulation half of losslessness, and it is asserted before any purification runs. Complexity: `O(Σ_t cells(u_t))`; the merged grids keep `cells` at the model's true resolution (a few hundred per axis at most for an oblivious GBM).

We only ever materialize feature sets that **appear** as a tree support — never the `C(n,3)` lattice (n=100 ⇒ 161 700 possible triples). The realized set is bounded by the distinct ≤3-feature supports the booster chose, typically a few hundred.

### 08.3 Purification: the mass-moving cascade

Purification (Lengerich Alg. 1/2) transforms `{T_raw[u]}` into the canonical fANOVA `{f_u}` **without changing the sum**, by moving the `w`-weighted mean of each axis-slice down one order. Define the weighted slice mean along axis `i` of `T_u`:

```
m(T_u, i, x_{u\i}) = Σ_{x_i} f_u(x_{u\i}, x_i) · W_u(i, x_{u\i}, x_i)
```

where `W_u` is the `w`-mass of the slice (08.4). `PURIFY-MATRIX` subtracts `m0` from the slice (centering it) and adds `m0` into `T_{u\i}` (conserving mass). `PURIFY` runs `PURIFY-MATRIX` over feature sets in **decreasing |u|** (3→2→1→intercept), so once `T_u` is pure, later work on its sub-tensors never re-pollutes it.

```rust
/// In-place; cascades 3→2→1→∅. Caches σ²(f_u) into each EffectTable.variance.
fn purify(raw: RawBank, w: &WeightCache, mode: PurifyMode) -> Result<TableBank, PbError>;
pub enum PurifyMode { ToFixpoint { tol: f64, max_iter: u32 }, SinglePass }
```

Convergence (research/03 §2.5): for **axis-constant** `w` (uniform, or any product measure that factorizes per axis — which `ProductMarginals` does) purification is exact in a **single pass** per axis — this is ordinary double-centering, so `ProductMarginals`/`Uniform` use `SinglePass` (`max_iter = 1`, no fixpoint loop). For a **joint** `w` the slice masses couple across axes and we iterate to a fixpoint (`M_t ≤ ε` after `O(log(M_0/ε))` passes; "almost all mass moves in the first iteration"). Default `tol = 1e-9` (`f64`), `max_iter = 64`; exceeding `max_iter` returns `PbError::Internal { what: "purify did not converge" }` rather than shipping an unpurified bank.

By linearity with `Σα = 1` (Lengerich Cor. 2.2), `purify(Σ T_raw) ≡ Σ purify(T_raw)` — so purify-then-sum equals sum-then-purify. This is the property §09 leans on for ensemble averaging and fully-corrective refit: a convex combination of order-≤3 banks is itself an order-≤3 bank, re-purified once.

### 08.4 The reference measure `w`

`w` is the most consequential explainability choice: it sets *which* cells the zero-mean condition centers against, and "the choice of distribution can change purified effects dramatically." It is a versioned, signed input — **recomputing tables under a different `w` without retraining is exactness-preserving** (leaves stay piecewise-constant, the sum is conserved), so it stays `Exact` and never touches the firewall. `w` is stamped on the `TableBank` and every exported table, importance, and SHAP attribution.

```rust
/// Per-axis-cell mass, precomputed once from the data and the chosen RefMeasure.
/// Built over a ServeBinnedMatrix (frozen encoders, R-CATSERVE) so the audited
/// mass matches the deployed model. `per_axis[i][0]` is the explicit missing
/// cell's mass on axis i (08.1) — strictly positive under Laplace smoothing even
/// when no training row is missing, so zero-mean/convergence never break.
struct WeightCache { per_axis: Vec<Vec<f64>>, kind: RefMeasure } // ProductMarginals/Uniform: factorized
fn build_weights(x: &ServeBinnedMatrix, grids: &MergedGrids, w: RefMeasure) -> Result<WeightCache, PbError>;
```

- **`ProductMarginals { laplace }` — DEFAULT.** `ŵ_lap ∝ ŵ_unif + laplace · ŵ_emp` per axis: the Laplace-smoothed empirical marginal (`laplace = 1.0` default, must be `> 0`). Rationale: respects per-axis *marginal* density and stays strictly positive (so empty merged cells don't break zero-mean or convergence), factorizes per axis (single-pass purify), keeps `σ²(F)=Σσ²(f_u)` (so Sobol sums to ~1), and makes equal-split SHAP **exact**. NB: product-of-marginals does **not** avoid Hooker's extrapolation — its weights stay positive on feature combinations that never co-occur, which *is* the extrapolation (Laplace smoothing deepens the reliance); only `Joint` avoids that, at the cost of the variance-sum identity. The default is chosen for clean single-pass purify + the exact variance-sum identity + exact SHAP, **not** for extrapolation-faithfulness.
- **`Uniform`** — fast, single-pass, effect-coding semantics; `ŵ ∝ 1` per cell.
- **`Joint`** (Hooker hierarchical-orthogonality) — faithful under correlated features, but couples axes (iterative purify), **breaks the variance-sum identity**, and makes equal-split SHAP only *interventional*. Under `Joint`, importances switch to Shapley-effects (still sum to 1) and SHAP is labelled accordingly; the VarianceSum gate (08.6) is skipped, not failed.

Default is product-`w`; the joint path is built early so we can **benchmark relativity drift between measures** — that drift is the credibility metric under regulator questioning, an internal diagnostic, not shipped tooling.

### 08.5 Importances & exact local attributions

**Sobol importances.** `S_u = σ²(f_u)/σ²(F)`, read directly from the cached `EffectTable.variance` — zero model calls. Under product/uniform `w` they sum to ~1 (the variance-decomposition axiom) and give the principled mains-vs-pairs-vs-triples ranking that drives heredity admission (§07), display pruning, and export. Computed only *after* purification (pre-purification variances are meaningless under the identifiability degeneracy).

**Exact interventional SHAP / Faith-Shap.** The headline result: a purified order-3 additive model simultaneously **is** the fANOVA decomposition, the exact n-Shapley/Faith-Shap up to order 3, and the Möbius transform (Bordt–von Luxburg; InstaSHAP). So attributions are O(1) table reads, never tree traversals:

```
φ_i(x) = Σ_{u ∋ i} f_u(x_u) / |u|                 (equal-split Shapley value)
Φ_S(x) = f_S(x_S)                                 (Faith-Shap interaction index, |S| ≤ 3)
```

```rust
impl TableBank {
    pub fn shap(&self, x_cells: &[u32]) -> Result<Vec<f64>, PbError>;       // φ_i, sums to F(x)−f0
    pub fn faith_shap(&self, x_cells: &[u32], s: &FeatureSet) -> Result<f64, PbError>; // = f_S(x_S)
    pub fn sobol(&self) -> Vec<(FeatureSet, f64)>;                          // sorted desc
}
```

These are labelled **"interventional"**, never bare "SHAP". Under product-`w` they are exact and `Σ_i φ_i(x) = F(x) − f0` to a derived float tolerance — the ThreeWayEqual gate. Stock TreeSHAP is demoted to a **test oracle only** (§13). H-statistic / PDP-ICE are thin views over the same tables (`H²_jk = σ²(f_ij)/σ²(PD_jk)`, clamp `H > 1`).

### 08.6 The five Invariant checks (build gates)

The explainability guarantee is **enforced, not asserted**. Each check returns `Result<(), PbError>` mapping to `PbError::InvariantViolated { invariant }` (carrying the `Invariant` variant — the single standardized signature §13 reconciles to), and runs as a build-blocking assertion in CI (§13). "If these ever disagree there is no product."

```rust
fn check_reconstruction(model: &Model, bank: &TableBank, tol: f64) -> Result<(), PbError>; // Reconstruction
fn check_mass(raw: &RawBank, bank: &TableBank, tol: f64) -> Result<(), PbError>;            // MassConservation
fn check_purity(bank: &TableBank, w: &WeightCache, tol: f64) -> Result<(), PbError>;        // (Decomposability)
fn check_variance_sum(bank: &TableBank, w: &WeightCache, tol: f64) -> Result<(), PbError>;  // VarianceSum
fn check_three_way(model: &Model, bank: &TableBank, x: &ServeBinnedMatrix) -> Result<(), PbError>; // ThreeWayEqual
```

All five checks walk the **full** merged-grid extents — **including the explicit missing cell** at index 0 of every axis (08.1). Reconstruction enumerates the missing cell as an interior point (a row missing on that axis, finite elsewhere); MassConservation/Purity/VarianceSum sum over the missing cell's `w`-mass like any other; ThreeWayEqual exercises rows that exercise `Split.missing_left`. An empty missing cell still participates (zero mass, flagged by `support`), so a tree whose `missing_left` differs from a naive route-left is caught here rather than slipping through.

1. **Reconstruction** — `max` over one interior point per merged-grid cell (missing cell included) of `|F_ens(x) − (f0 + Σ_u f_u(x_u))| < tol`. Exhaustive (piecewise-constant ⇒ one point per cell is worst-case), not sampled. The tolerance is **a derived float-accumulation bound, not a magic floor**: `recon_tol = 4 · n_trees · f32::EPSILON` in `f32` score space (the worst-case rounding of summing `n_trees` weighted leaves), so the claim is "equal to a derived float tolerance," not "bit-equal." True zero-tolerance bit-equality is reserved for the serialized-model determinism gate (§11/§13), which the fixed-`CHUNK_ROWS` reduction discipline makes achievable.
2. **MassConservation** — total signed `w`-mass identical before/after purify (`|Σ raw − Σ purified| < tol`); each mass-move is a subtract-here/add-there pair, so this is a tautology unless a numerical bug exists.
3. **Purity** — every axis-slice of every `EffectTable` has `w`-weighted mean `< tol` (the zero-mean condition; maps to `Invariant::Decomposability`). This certifies each *individual* table, which the rating export reads, so it runs by default (08.8).
4. **VarianceSum** — `|σ²(F) − Σ_u σ²(f_u)| < tol`. **Branches on `w`**: asserted under `ProductMarginals`/`Uniform`; skipped (not failed) under `Joint`, where hierarchical orthogonality replaces full orthogonality.
5. **ThreeWayEqual** — tree-sum = table-sum = Shapley-sum, equal to a derived float tolerance at sampled rows: `F_ens(x) == f0 + Σ_u f_u(x_u) == f0 + Σ_i φ_i(x)`. The end-to-end "the number deployed is the number audited" check.

### 08.7 The firewall, and complete vs pruned

`build_bank` consults `Model.mode` (§3). An `Exact` model produces a `TableBank` and is gated by all five checks; an `Approximate { reason }` model **refuses** to export an `Exact` bank — it returns `PbError::ExactnessFirewall(reason)` or, where a residual model exists, exports "tables + residual model" explicitly flagged. Recomputing under a different `w`, multi-step Newton leaf refit, fully-corrective refit, Nesterov mixing, and self-ensemble averaging (§09) are all exactness-preserving and stay `Exact`.

**Complete-for-inference vs pruned-for-display** are the two artifacts that run side by side. `TableBank.tables` is the *complete realized support* — LUT-sum scoring (§10) is lossless **only** over this complete set, never a pruned subset. A pruned **view** is a non-destructive selection for human display:

```rust
impl TableBank {
    /// Top-k feature sets by Sobol, honestly labelled. A VIEW: does not mutate the
    /// bank, is NOT inference-valid, carries (shown, total, variance_covered).
    pub fn pruned_view(&self, top_k: usize, min_sobol: f64) -> TableBankView;
}
pub struct TableBankView<'a> { /* refs */ pub shown: usize, pub total: usize, pub variance_covered: f64 }
```

A filed view reads "showing 12 of 47 tables, 99.3% of variance" — the pruning is a display decision, fully decoupled from the admission decision (§07) and from the lossless inference support.

**Inner-bag SE-band annotation (display-only).** When the model is a bagged self-ensemble (§09), each merged cell carries an optional `±SE` band from the spread of the per-bag purified table values at that cell: `se_cell = sqrt( Σ_b (f_u^b − f̄_u)² / (B·(B−1)) )` over the `B` inner bags. This is a **display-only annotation, not an fANOVA component** — it is never summed into `score`, never enters reconstruction/ThreeWayEqual, and is absent (`None`) for single-fit models.

```rust
/// Optional per-cell standard-error band, parallel to EffectTable.values.
/// DISPLAY-ONLY: not an fANOVA term; excluded from all five invariant checks.
pub struct SeBand { pub per_cell: Tensor } // same extents as the table; None unless bagged
impl EffectTable { pub fn se_band(&self) -> Option<&SeBand>; }
```

**Per-cell effective support (display-only, always populated).** Alongside each `EffectTable.values`, the bank carries `support`: the effective `w`-mass (exposure-weighted training-row count) in each *merged-grid* cell. It exists because the credibility floor (§07) binds on a tree's 8-cell leaf, but exported tables live on the finer **merged grid** — so a displayed 3-way cell can stand on far fewer rows than any leaf that fed it (a confident-looking `2.4×` relativity backed by ~11 policies). `support` makes that visible: thin cells are flagged in the export (§10) and explanation views. It is filled in one pass over the binned data during accumulation (the per-axis `w`-mass is already gathered; the joint per-cell count is the same sweep). Like `SeBand` it is **display-only metadata, not an fANOVA component** — never summed into `score`, never entering Reconstruction/VarianceSum/Purity/ThreeWayEqual, and absent from inference (LUT-sum reads `values`, never `support`) — so I2 is untouched.

### 08.8 Public API surface

```rust
impl Model {
    /// Build the complete purified TableBank under `w`. Takes a ServeBinnedMatrix
    /// (R-CATSERVE): the caller re-encodes raw categoricals through this model's
    /// frozen `schema.cat_encoders` (§04) — `explain` MUST NOT be handed a
    /// TrainBinnedMatrix. Runs all five gates by default. The release/debug split
    /// is driven by `cfg!(debug_assertions)`, NOT a config flag: debug/CI run the
    /// full sweep including a derived-tolerance Reconstruction; release runs all
    /// five but with sampled (not exhaustive) Reconstruction rows. Purity always
    /// runs — it certifies the individual tables the rating export reads. Returns
    /// `PbError::TableBudget` (08.10) if a table or the bank exceeds its cell budget
    /// and sparse fallback is disabled.
    pub fn explain(&self, x: &ServeBinnedMatrix, w: RefMeasure) -> Result<TableBank, PbError>;
}
impl TableBank {
    pub fn recompute_under(&self, x: &ServeBinnedMatrix, w: RefMeasure) -> Result<TableBank, PbError>; // exact, no retrain
    pub fn score(&self, x_cells: &[u32]) -> Result<f64, PbError>;     // f0 + Σ lookups, == F_ens
    pub fn reference_measure(&self) -> &RefMeasure;
}
```

### 08.9 Complexity, performance, testing

**Complexity.** Accumulation `O(Σ_t cells(u_t))`; purification `O(#cells · passes)` with `passes = 1` under product/uniform `w`, embarrassingly parallel across feature sets (rayon over independent tensors). Dominant memory cost is 3-D tensors `|Ω_i|·|Ω_j|·|Ω_k|` (each `|Ω|` counts the missing cell) — bounded by realized-threshold resolution, another reason for the union grid, and hard-capped by the table-size budget of 08.10. SHAP/Sobol/Faith-Shap are O(1)–O(#tables) reads with zero model calls. The whole engine runs once post-fit and is cheap relative to training.

**How it serves the three aims.** *Decomposable:* this section is the decomposition, exactness gated in CI. *Fast:* O(1) attributions and LUT-sum scoring (no tree walks); single-pass purify on the default `w`. *Accuracy:* Sobol drives §07's admission funnel, so the engine feeds the learner.

**Testing.** Proptest the purification identities on random small banks: idempotence (`purify(purify) == purify`), linearity (`purify(αA+βB) == α·purify(A)+β·purify(B)`), permutation invariance, and single-pass exactness under product/uniform `w`. Unit-test each of the five gates against hand-computed tiny models (the Boolean AND/OR/XOR degeneracy of research/03 §1.5 is the canonical purification fixture). Cross-check `shap`/`faith_shap` against TreeSHAP-as-oracle on small models (§13). Reconstruction/ThreeWayEqual run on every fitted model in the invariant CI suite. A boundary test pins the hot-loop `at(idx)` helper at its in-bounds and one-past-end edges (08.1 panic policy).

**Gating policy (resolved default, not a fork).** Release-mode gating runs **all five checks by default**, driven by `cfg!(debug_assertions)` rather than a `verify: bool` on `FitSpec`/`explain` (that flag is dropped, not plumbed). Debug/CI run the full exhaustive Reconstruction sweep; release runs all five but samples Reconstruction rows instead of exhausting every cell. Purity is never skipped — it is the per-table certificate the rating export depends on. This keeps the deployed-equals-audited guarantee without the exhaustive purity/variance sweep being a configurable footgun.

### 08.10 Table-size budget (memory firewall)

A 3-way tensor over three border-rich axes is the one place this engine can blow up: with the explicit missing cell, axis `i` has `cells_i = borders_i.len() + 2` (08.1), and a triple costs `Π cells_i`. Three near-`max_bin` axes (≈254 finite bins each ⇒ ≈256 cells) is ≈16.8M `f64` cells ≈ 134 MB for **one** table — and **bagged** ensembles enlarge the merged grid further, since the union grid counts every border realized across **all** bags (a triple's cell count is computed on that bagged union, not on any single bag). So the budget is a hard firewall, not an estimate.

```rust
/// Per-table and whole-bank cell budgets. Defaults are the funnel's safe ceiling;
/// `on_overflow` picks the resolution. Counted on the BAGGED union grid (08.2).
pub struct TableBudget {
    pub max_table_cells: u64,        // per-EffectTable Π cells_i ceiling; default 2_000_000
    pub max_bank_cells:  u64,        // Σ over all tables ceiling; default 32_000_000
    pub on_overflow:     OverflowPolicy,
}
pub enum OverflowPolicy {
    /// Hard error — refuse to build a bank that would exceed the budget.
    Error,                           // ⇒ PbError::TableBudget { u, cells, budget }
    /// EXACT sparse-tensor storage for hot triples (same tensor, sparsely stored).
    SparseFallback { density_threshold: f64 }, // default 0.05
}
```

**Two outcomes, both exactness-preserving.**
- **`Error`** — accumulation/purification refuses to materialize a table (or a bank) whose cell count exceeds the budget and returns the new typed `PbError::TableBudget { u, cells, budget }` (registered in §2.8 alongside the other firewall errors). No silent truncation: a dropped or coarsened table would break Reconstruction, so the build fails loudly instead.
- **`SparseFallback`** — for a triple whose dense cell count exceeds `max_table_cells` but whose **occupied** cells (cells with non-zero `w`-mass or any contributing tree) are below `density_threshold · Π cells_i`, store the tensor **sparsely** (a `(cell_index → f64)` map keyed by the row-major offset). This is **exact** — it is the *same* tensor with the unoccupied (structurally-zero after purification) cells not materialized; the five invariant gates run unchanged over the sparse representation (`at(idx)` returns the stored value or `0.0` for an absent occupied-set member, which is the purified value of a no-mass cell under a strictly-positive `w`). Sparse storage never changes the decomposition, only its memory layout.

**Counting rule (R-TABLEBUDGET).** Cell counts are taken on the **realized bagged union grid** — the same merged grid the bank is built on (08.1/08.2), so a triple that is cheap per-bag but expensive on the union is correctly sized. The budget binds on accumulation (lazy allocation checks the projected `Π cells_i` before touching a raw tensor) so an over-budget triple fails *before* it allocates 100s of MB.

**Funnel coordination (soft, upstream).** The hard budget here is the backstop; §07's admission funnel carries the matching **soft** admission penalty that deprioritizes high-cardinality supports (triples whose merged-grid cell count is large, e.g. near-`max_bin` axes) so they rarely reach this firewall. §14 carries the benchmark gate on adversarial border-rich + bagged fixtures (cell counts, purification time, memory) that proves the budget holds under stress. The split is deliberate: §07 *avoids* the cliff, 08.10 *guarantees* the cliff is never silently crossed.
