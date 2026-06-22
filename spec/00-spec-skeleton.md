# pattern-boost — SPEC Skeleton (the canonical spine)

> 2026-06-21. This is **the contract**, not a brainstorm. It defines the engineering standards, the load-bearing shared types/traits, the hard invariants, and the section-ownership map that all 14 spec sections (01–14) MUST conform to. The brainstorm (`design/01-brainstorm.md`) and gap-closing analysis (`design/02-predictiveness-gap-closing.md`) already resolved the forks; this skeleton turns those resolutions into binding decisions. Where a fork remains it is stated with a recommended default.
>
> **Three aims, in tension-managed balance:** (1) MAXIMUM PREDICTIVENESS — use every gap-closer; (2) FULLY DECOMPOSABLE — the trained model decomposes **losslessly** into constant + 1D + 2D + 3D purified fANOVA tables, because every tree is a depth-3 oblivious tree (≤3 features); (3) FAST — competitive with LightGBM/CatBoost on CPU training, structurally faster at inference (branch-free 8-cell lookup + table-sum scoring).
>
> Naming: types use `PascalCase`, the crate is `pattern-boost-core`, the Python package `pattern_boost`. All inline identifiers below are real Rust to be used verbatim by the owning sections.

---

## 1. Engineering standards — the checklist every section must honor

Every section's decisions MUST satisfy the following. CI gates marked **[GATE]** are build-blocking.

**Error model.**
- A single typed crate error enum `PbError` (§2.10); **every** fallible public function returns `Result<T, PbError>`. No `Box<dyn Error>` in public signatures.
- **Panic policy: no panics in library code.** No `unwrap`, `expect`, `panic!`, `unreachable!`, `unimplemented!`, indexing that can go out of bounds, or arithmetic that can overflow on any fallible path. `unwrap`/`expect` are permitted ONLY in tests, benches, and on a value the surrounding code has just proven non-`None`/in-bounds with a comment justifying it. **[GATE]** clippy lints `unwrap_used`, `expect_used`, `panic`, `indexing_slicing` are `deny` at crate root.
- **No-panic hot loops.** In hot inner loops (e.g. `tree_lookup`'s `row[split.axis]`/`tree.leaves[idx]`, the `fit` accumulation, `bin`'s border indexing) use either the panic-free form `slice.get(i).ok_or(PbError::Internal { what })?` OR a function-scoped `#[allow(clippy::indexing_slicing)]` carrying a `// JUSTIFIED:` bounds proof and a paired boundary test. Indexing is never silently allowed crate-wide.
- **Arithmetic / overflow policy.** Integer overflow is caught by `overflow-checks = true` in **all** cargo profiles (release included), not by a crate-blanket lint; float arithmetic is explicitly exempt. The clippy `arithmetic_side_effects` lint is therefore **scoped** to specific integer-accumulation sites (with a justifying comment), **never** denied at the crate root — a crate-wide deny would make every legitimate accumulation a lint error. Histogram bin accumulators are `i64` (§2.3) with a proven `n_rows·max|g_q| < i64::MAX` bound, so they never overflow in practice.
- Internal invariant violations that "cannot happen" return `PbError::Internal { what }` rather than panicking, so a bug degrades to a typed error, never a crash in a user's Python session.

**Determinism / bit-reproducibility (first-class).**
- Identical inputs + identical config + identical seed ⇒ **bit-identical** model and predictions, **independent of thread count**. **[GATE]** a CI test trains the same model at `n_threads ∈ {1, 2, 8}` and asserts byte-equality of the serialized model.
- Mechanism: all gradient/hessian accumulation uses **quantized integer histograms** (associative ⇒ order-independent sums; §2.3); any unavoidable float reduction uses `fold` over fixed-size `par_chunks` combined in index order, never rayon `reduce`/`sum` in steal order.
- A single `seed: u64` threads through binning subsample, subsampling, MVS, and any randomized split selection. RNG is a named, versioned PRNG (`rand_pcg::Pcg64` via `SeedableRng`); the seeding scheme is documented and frozen. Per-work-unit independent streams are produced by **deterministic re-seeding**, NOT a "splittable" PRNG (which is unimplementable as named for `Pcg64`): each work unit derives its seed as `Pcg64::seed_from_u64(splitmix64_mix(base, round, stage, block))`, where `splitmix64_mix` is a frozen `splitmix64` mixing of the `(base, round, stage, block)` tuple. This is position-stable and thread-count-independent, so the determinism `[GATE]` holds.

**Unsafe policy.**
- `#![forbid(unsafe_code)]` at the crate root of `pattern-boost-core` by default. Any `unsafe` requires: an `unsafe_ok` module-local `#![allow]`, a `// SAFETY:` comment proving each precondition, encapsulation behind a safe API, and a dedicated test (plus a Miri run in CI where feasible). SIMD intrinsics live behind `multiversion`/`pulp`/`wide` (safe wrappers) — hand-written intrinsics are a last resort and must be justified in the section that introduces them.

**Feature-flagging.**
- `pyo3`/`extension-module` exist ONLY in `pattern-boost-py`; `pattern-boost-core` has **zero** pyo3 dependency. Optional capabilities are cargo features: `arrow` (Arrow/PyCapsule ingest), `distill` (CatBoost teacher hooks — data-side only), `nightly` (portable_simd). Default features must compile and pass all invariant gates on stable.

**MSRV & toolchain.** MSRV **1.64** (manylinux2014/glibc 2.17 floor), pinned in `rust-toolchain.toml`; raised only with a changelog entry and CI matrix update.

**Dependency philosophy.** Minimal, audited, widely-used: `rayon`, `ndarray`, `serde`, `serde_json`, `bincode`, `rand`/`rand_pcg`, `thiserror`, `num-traits`; `multiversion`/`pulp`/`wide` for SIMD; `half`/quantization helpers as needed. New deps require justification in the owning section. `cargo deny` (licenses, advisories, bans) is **[GATE]**.
- **`bincode` is pinned to 2.x.** The removed top-level `bincode::serialize`/`deserialize` functions are NOT used; all binary (de)serialization goes through `bincode::serde::encode_to_vec(.., bincode::config::standard())` / `bincode::serde::decode_from_slice(.., bincode::config::standard())`. The `bincode::config::standard()` config is **frozen** (its variant/endian/limit choices are part of the contract) because byte-equality of the serialized model depends on it. §10/§11/§12 use exactly this API and config.

**Doc & test requirements.**
- `#![deny(missing_docs)]` on all public items. Every public type/fn has a doc comment with semantics, error conditions, and a runnable `///` example where non-trivial.
- Tests required per module: unit tests for math (gradients/hessians/gain checked against closed form), property tests (`proptest`) for purification identities, and the invariant gates of §3. **[GATE]** `cargo test`, `cargo clippy -- -D warnings`, `cargo fmt --check`, doctests.

**Naming conventions.** `snake_case` fields/fns; `PascalCase` types; numeric-precision suffixing where ambiguous (`_f32`, `_q` for quantized). Score-space quantities use `raw`/`F`; response-space use `pred`. Reference measure is always `w`. Feature sets are `u` (an `FeatureSet`). No abbreviation that isn't in this skeleton's glossary.

---

## 2. Canonical shared types & traits

These are the single source of truth. **Sections MUST use these signatures verbatim** and may not define divergent versions. (Field-level additions are allowed only via the owning section in §4.) The core is specialized to **`f32`** for features and accumulators where noted.

### 2.1 Feature index & provenance
```rust
/// Index into the user's original feature columns (raw underlying feature).
pub struct FeatureId(pub u32);

/// A model column may be a raw numeric, a reserved-missing-aware bin axis,
/// or a categorical target-statistic axis. Provenance tracks the RAW feature(s)
/// behind an axis so the ≤3-feature invariant is enforced on DISTINCT raw features,
/// never on encoded columns. (See I1, §3.)
pub enum AxisKind { Numeric, CategoricalTS { encoding: TsEncodingId }, Missing }
pub struct AxisProvenance { pub raw: FeatureId, pub kind: AxisKind }
```

### 2.2 Binned matrix & per-feature border grid
```rust
/// One feature's quantile/midpoint borders; bin 0 is the reserved MISSING bin.
/// **Canonical bin/grid cardinality (R-BINS).** `borders[k]` is the upper border
/// of bin k (ascending, strictly). `bin(x)` returns `1 + (count of borders strictly
/// below x)` — i.e. the k-th finite interval maps to bin `k+1` — so data bins are
/// `1..=n_data_bins` where `n_data_bins = borders.len() + 1`. `n_bins = n_data_bins + 1`
/// (data + missing). Constraint: `borders.len() <= max_bin - 1` (default `max_bin = 254`),
/// so `n_data_bins <= 254` and `n_bins <= 255` (fits `u8`, values `0..=254`). §03 owns
/// `bin()`/`build_grid`; §06/§08/§10 reference this exact form.
pub struct BorderGrid { pub borders: Vec<f32>, pub n_bins: u16, pub missing_bin: u8 }

/// Column-major, pre-binned design matrix. `data[f]` is column f as bin ids.
/// f32→u8 binning happens once at ingest; the grid persists in the Model.
pub struct BinnedMatrix {
    pub data: Vec<Vec<u8>>,          // [n_features][n_rows], column-major
    pub n_rows: u32,                 // fixed-width: serialized; `usize` breaks cross-platform byte-equality
    pub grids: Vec<BorderGrid>,
    pub provenance: Vec<AxisProvenance>,
}
```

### 2.3 Gradient / hessian (full-precision + quantized)
```rust
/// Per-row first/second derivatives of the loss w.r.t. the raw score F.
pub struct GradHess { pub g: Vec<f32>, pub h: Vec<f32> }

/// Quantized integer g/h for associative, order-independent histogram sums
/// (the bit-reproducibility AND ~2x-speed mechanism). Stochastic rounding on
/// quantize; LEAVES ARE REFIT FROM FULL-PRECISION g/h (mandatory on log-link).
/// Per-row quantized values are `i32`; the per-bin ACCUMULATORS are `i64`
/// (the `Hist` type, §6/§11) to avoid overflow — see `Hist` below.
pub struct QuantGradHess { pub g_q: Vec<i32>, pub h_q: Vec<i32>, pub scale: GradScale }
pub struct GradScale { pub g_scale: f32, pub h_scale: f32 }

/// The per-bin gradient/hessian histogram accumulator. **`i64` bin accumulators
/// everywhere** (counts stay `u32`): summing per-row `i32` `g_q`/`h_q` over up to
/// `n_rows` bins can exceed `i32` range, and under `overflow-checks = true` an
/// `i32` overflow would panic — breaking the no-panic gate. The bound
/// `n_rows·max|g_q| < i64::MAX` is proven at construction. This is the SINGLE
/// histogram accumulator type; §11's `FeatureHist`/`LevelHists.arena` and §02's
/// `Backend` accumulator both alias `Hist` (no `i32`/`HistogramSet` variants).
/// Owned by §06.
pub struct Hist { pub g: Vec<i64>, pub h: Vec<i64>, pub count: Vec<u32> }
```

### 2.4 The Loss trait
```rust
/// A loss/objective. Fully orthogonal to tree shape (I1/I2 untouched).
/// `grad_hess` is one pass w.r.t. the raw score F (after the exposure offset).
pub trait Loss: Send + Sync {
    /// One full-precision gradient/hessian pass. Returns `Result` (never panics):
    /// a fallible objective — e.g. the §12 `PyLoss` user-callback path — maps its
    /// failure onto `PbError` rather than `.expect(..)`; the engine already returns
    /// `Result`, so the cost is one `?`.
    fn grad_hess(&self, y: &[f32], raw: &[f32], weight: &[f32], out: &mut GradHess) -> Result<(), PbError>;
    /// link(weighted mean) — the mandatory boost_from_average intercept f0. Fallible
    /// (R-LOSSFALLIBLE): returns a typed `PbError` on invalid domains (Gamma `y<=0`,
    /// all-zero weights, all-zero logistic positives, bad/zero exposure, user-loss
    /// failures) rather than panicking or emitting `NaN`. Full-width `f64` (the exact
    /// fANOVA intercept). §06 `init` propagates with `?`.
    fn init_score(&self, y: &[f32], weight: &[f32], offset: Option<&[f32]>) -> Result<f64, PbError>;
    fn link(&self) -> Link;                       // Identity | Log | Logit
    fn pred_from_raw(&self, raw: f32) -> f32;     // inverse link, exp(k*F) not powf
    /// Strictly-proper deviance for early stopping (NOT RMSE on Poisson/Gamma/Tweedie).
    /// Fallible (R-LOSSFALLIBLE): same invalid-domain typed errors as `init_score`;
    /// `f64` fold, reported in `f32`. §08/§13 metric reduction propagate with `?`.
    fn deviance(&self, y: &[f32], raw: &[f32], weight: &[f32]) -> Result<f32, PbError>;
    /// The objective's natural early-stopping metric (deviance by default).
    fn default_metric(&self) -> Metric;
    /// Lower clamp on per-row hessian (numerical floor ε); default `1e-16`
    /// (R-TYPEDRIFT: NaN-guard only — stability is λ + `max_delta_step`'s job; §05 owns).
    fn hessian_floor(&self) -> f32;
    /// Per-objective default leaf-stage |w*|-clamp. `None` = uncapped; Poisson ⇒ `Some(0.7)`.
    /// `Config.max_delta_step: Option<f32>` falls back to THIS when `None` (§06).
    fn max_delta_step(&self) -> Option<f32>;
}
pub enum Link { Identity, Log, Logit }

/// The early-stopping / evaluation metric an objective reports (deviance-based by
/// default; never RMSE on Poisson/Gamma/Tweedie). Owned by §05 (this is §05's
/// canonical form, verbatim — R-TYPEDRIFT).
pub enum Metric { Rmse, LogLoss, PoissonDeviance, GammaDeviance, TweedieDeviance { rho: f32 } }
```
v1 implementors: `SquaredError`, `Logistic`, `Poisson`, `Gamma`, `Tweedie { rho }`. Owned by §05.

### 2.5 The oblivious tree
```rust
/// A depth-3 oblivious tree: one shared (feature, border) test per level,
/// at most 3 DISTINCT raw features, 2^3 = 8 leaf values. Fewer than 3 levels
/// is legal (graceful early-termination, e.g. no monotone-valid split → a
/// lower-order fANOVA outcome, not an error).
pub struct ObliviousTree {
    pub splits: Vec<Split>,   // 1..=3 levels, in test order (bit 0 = level 0)
    pub leaves: [f32; 8],     // index = b0 | b1<<1 | b2<<2; unused tail = 0.0
    pub depth: u8,            // splits.len() as u8, 1..=3
}
/// One shared level test: `bin <= bin_le`. `axis` is `u32` (fixed-width: it is
/// serialized, and `usize` would break cross-platform byte-equality / the wasm32
/// smoke build). `missing_left` is the explicit learned default direction — the
/// reserved missing bin (bin 0) routes left when `true` (the single canonical
/// carrier; §03/§06/§08 cite this field, no `bin_le` sentinel).
pub struct Split { pub axis: u32, pub bin_le: u8, pub missing_left: bool }
```
**Canonical leaf-index bit (R-MISSING).** Missing is a LEARNED default direction
(`Split.missing_left`) and MUST be honored at every scoring/accumulation site —
never silently routed left. For a per-level test on the row's bin value `bin` at
`row[split.axis]`, the canonical low/left bit is:
```rust
// the SINGLE canonical form; used IDENTICALLY in tree_lookup (here), §06 split
// evaluation + sample→leaf update, §10 tree_lookup AND the packed ScoringBank
// kernel. Required for tree/table equality (else missing always routes left → I2 break).
let low = if bin == 0 { split.missing_left } else { bin <= split.bin_le };
let bit = low as usize;            // the leaf-index bit at this level
// idx accumulates: idx |= bit << level; leaves[idx] is the leaf value.
```
`ObliviousTree::lookup(x)` (cited by §2.6 inference) folds the per-level `bit` into
`idx = b0 | b1<<1 | b2<<2` using exactly this formula and returns `leaves[idx]`.

Invariant (checked at construction): `splits.iter().map(|s| provenance[s.axis as usize].raw).collect::<HashSet>().len() == splits.len()` (≤3 distinct raw features). Owned by §06.

### 2.6 The ensemble Model
```rust
/// The trained ensemble: intercept (the f0 term) + weighted oblivious trees,
/// the shared binning grids, provenance, the loss/link, and an exactness flag.
pub struct Model {
    pub f0: f32,                       // link(weighted mean); a scalar, never "tree 0"
    pub trees: Vec<(f32, ObliviousTree)>, // (weight alpha, tree); alphas allow DART/Nesterov/ensemble mixes
    pub grids: Vec<BorderGrid>,
    pub provenance: Vec<AxisProvenance>,
    pub link: Link,
    pub mode: ExactnessMode,           // see §3
    pub schema: ModelSchema,           // R-SCHEMA: metadata for serve/export of cats + classifiers
    pub schema_version: u32,           // covers `schema` too (serialized with the Model)
}

/// Model-level metadata so a `Model` can actually serve and export categoricals +
/// classifiers without the caller re-supplying anything (R-SCHEMA). Serialized with
/// the `Model`; `schema_version` covers it. Skeleton registers; §04 OWNS
/// `CatEncoder`/`CatEncoderStore`; §05 OWNS `LossId`.
pub struct ModelSchema {
    pub feature_names: Vec<String>,
    pub feature_kinds: Vec<AxisKind>,
    pub cat_encoders: CatEncoderStore,
    pub class_labels: Option<Vec<String>>,
    pub objective: ObjectiveTag,
}
/// `TsEncodingId -> frozen CatEncoder` (category -> TS value/bin map + level labels).
/// The FROZEN full-data encoders that back a `ServeBinnedMatrix` (§03/§04); explain()
/// and TableBank accumulation re-encode raw categoricals through THESE. Owned by §04.
pub struct CatEncoderStore { /* TsEncodingId -> frozen CatEncoder */ }
/// The trained objective, recorded so export/predict_proba can reproduce link + loss.
pub struct ObjectiveTag { pub link: Link, pub loss: LossId, pub tweedie_rho: Option<f32> }
```
`AxisKind::CategoricalTS { encoding: TsEncodingId }` resolves to a concrete encoder in
`cat_encoders`. §10 uses `ModelSchema` for export readability + round-trip; §12 uses it
for feature names, class labels, and `predict_proba`.

Inference: `raw(x) = f0 + offset + Σ alpha_t * tree_t.lookup(x)`. Owned by §06 (struct) / §10 (serde + scoring).

### 2.7 fANOVA TableBank & reference measure
```rust
/// One purified effect tensor for feature set u, on the MERGED grid (sorted union
/// of realized borders per axis). 1D/2D/3D ranks share this type.
/// `support` = per-cell effective w-mass (same extents as `values`, always populated):
/// display/credibility metadata flagging thin cells — excluded from the five invariant
/// checks and from inference (scoring reads `values` only).
pub struct EffectTable { pub u: FeatureSet, pub axes: Vec<AxisId>, pub values: Tensor, pub support: Tensor, pub variance: f64 }
pub struct FeatureSet(pub SmallVec<[FeatureId; 3]>); // size 0..=3, sorted, distinct raw ids

/// The complete decomposition: intercept + all purified tables on a shared grid.
/// `complete` is the lossless inference support; display pruning is a VIEW (§08).
pub struct TableBank {
    pub f0: f64,
    pub tables: Vec<EffectTable>,      // every realized u of size 1..=3
    pub merged_grids: Vec<BorderGrid>,
    pub w: RefMeasure,                 // stamped on the bank and every export
}

/// The reference measure for purification. Default = Laplace-smoothed empirical
/// product-of-marginals. Recomputing the bank under a different w WITHOUT
/// retraining is exactness-preserving (§3).
pub enum RefMeasure {
    ProductMarginals { laplace: f32 }, // DEFAULT (laplace > 0)
    Uniform,
    Joint,                              // Hooker hierarchical-orthogonality; importances→Shapley-effects
}
```
Owned by §08. `AxisId`/`Tensor` are §08-local aliases.

### 2.8 Error enum
```rust
#[derive(thiserror::Error, Debug)]
pub enum PbError {
    #[error("invalid input: {what}")]            InvalidInput { what: String },
    #[error("dtype mismatch: expected {expected}")] DtypeMismatch { expected: &'static str },
    #[error("shape mismatch: {what}")]           ShapeMismatch { what: String },
    #[error("invalid config: {what}")]           InvalidConfig { what: String },
    #[error("invariant violated: {invariant}")]  InvariantViolated { invariant: Invariant },
    #[error("exactness firewall: {0}")]          ExactnessFirewall(String),
    #[error("serialization: {0}")]               Serialization(String),
    #[error("internal bug: {what}")]             Internal { what: String },
}
pub enum Invariant { FeatureBudget, Decomposability, MassConservation, Reconstruction, VarianceSum, ThreeWayEqual }
```
Owned by §02 (definition); every section maps its failures onto a variant.

### 2.9 The public booster handle
```rust
/// The public estimator. Builder-configured, fit→Model, sklearn-mirrored in Python.
pub struct Booster { /* config */ }
impl Booster {
    pub fn new() -> Self;
    pub fn fit(&self, x: &BinnedMatrix, y: &[f32], spec: &FitSpec) -> Result<Model, PbError>;
}
pub struct FitSpec<'a> {
    pub loss: &'a dyn Loss,
    pub weight: Option<&'a [f32]>,
    pub exposure: Option<&'a [f32]>,       // offset = log(e); anchors base level = 1.000
    pub monotone: MonotoneMap,             // name→sign, NEVER positional (BTreeMap, §07)
    pub interaction: InteractionPolicy,    // max order + optional groups whitelist
    pub distill: Option<DistillSpec>,      // R-DISTILL: per-row teacher_raw DATA + blend + teacher kind.
                                           // Off-by-default; per-row data belongs with weight/exposure
                                           // here, NOT in BoosterConfig. `DistillSpec` owned by §09.
    pub seed: u64,
}

/// Whole-tree interaction constraint plus the optional feature-group whitelist
/// that §07.9 tests but that scalar `max_interaction_order` had no entry point for.
/// `groups` (when `Some`) restricts each tree's distinct-raw-feature support to lie
/// within one declared group (a soft admission seam; §07 owns the funnel).
/// Owned by §07; threaded through §06 `fit` and the §12 kwarg. (R-TYPEDRIFT:
/// `groups` is `Option<Vec<FeatureSet>>` — §07's canonical form; `None` = unconstrained.)
pub struct InteractionPolicy {
    pub max_order: u8,                     // {1,2,3}, default 3
    pub groups: Option<Vec<FeatureSet>>,   // allowed co-occurrence groups; None = unconstrained
}
```
Config struct (learning rate, n_trees, lambda, max_bin, subsample, MVS, etc.) is owned by §06; the Python mirror by §12.

---

## 3. The invariant contract (I1, I2 + the firewall)

Every section upholds these as **checkable properties**. They are enforced as build-blocking gates, not asserted in prose.

**I1 — Depth-3 oblivious / ≤3 distinct raw features per tree.**
Property: for every `ObliviousTree`, `depth ∈ 1..=3`, splits share one `(axis, bin_le)` per level, and the count of **distinct `provenance[axis].raw`** across splits equals `depth`. Checked at `ObliviousTree` construction and in a CI property test over fitted models. Violation ⇒ `PbError::InvariantViolated { FeatureBudget }`. Non-symmetric growth, linear/soft leaves, and >3-raw-feature encoded axes (e.g. CatBoost combination CTRs) are **forbidden on the exact path**.

**I2 — Exact ≤3rd-order decomposability on a shared grid.**
Property: the ensemble equals a constant + sum of ≤3-feature purified tables on the merged grid, **bit-for-bit** within float tolerance. Enforced by the five `Invariant` checks, run as **[GATE]** assertions:
1. **Reconstruction:** `max over one interior point per merged-grid cell of |F_ens(x) − (f0 + Σ_u f_u(x))| < tol` (exhaustive because piecewise-constant).
2. **MassConservation:** total signed mass invariant across purification.
3. **Purity:** every axis-slice of every `EffectTable` has `w`-weighted mean zero.
4. **VarianceSum:** `σ²(F) = Σ_u σ²(f_u)` (branches on `w`; holds under product/uniform).
5. **ThreeWayEqual:** tree-sum = table-sum = Shapley-sum (exact n-Shapley/Faith-Shap ≤order-3), bit-equal.
"If these ever disagree there is no product." Any technique that bends I1/I2 MUST be flagged in its owning section and gated behind the firewall.

**The Exact/Approximate firewall.**
```rust
pub enum ExactnessMode { Exact, Approximate { reason: String } }
```
A `Model` in `Exact` mode passes all five checks and exports "rating tables." Any operation that cannot pass them (nonlinear calibration warp, continuous-TS axis, linear leaves, a >3-order GLM base-margin) flips the model to `Approximate { reason }` and **refuses** to export an `Exact` TableBank — it returns `PbError::ExactnessFirewall(..)` or exports "tables + residual model." Recomputing tables under a different `RefMeasure`, multi-step Newton leaf refit, fully-corrective refit, Nesterov mixing, and self-ensemble averaging are all **exactness-preserving** and stay `Exact`. This typed wall is the structural defense against death-by-a-thousand-cuts.

---

## 4. Section ownership map

Each of the 14 sections **owns** the listed decisions/types and may not redefine another section's. Cross-references use the shared types of §2.

| § | Title | OWNS (decisions & types) | Uses (does not own) |
|---|---|---|---|
| **01** | Overview / goals / invariants / milestone | The three aims; the TabArena "beat EBM, near-parity, all-exact" milestone (the one success criterion); restatement of I1/I2 for readers; glossary. | §2 types, §3 contract. |
| **02** | Architecture & project layout | Workspace layout (core/py/python/benches); module boundaries; `PbError` + `Invariant` **definition**; `Backend` trait seam (CPU now, GPU-shaped later; its accumulator IS §06's `Hist`, no `HistogramSet` variant); feature-flag matrix; schema_version policy; the `overflow-checks=true`/scoped-`arithmetic_side_effects` policy realization. | All §2 types live across modules per this map; `Hist` (§06). |
| **03** | Data model & binning | `BinnedMatrix`, `TrainBinnedMatrix`/`ServeBinnedMatrix` (fit-only out-of-fold vs frozen-encoder serve matrices, R-CATSERVE/R-SCHEMA), `BorderGrid` (canonical cardinality `n_data_bins = borders.len()+1`, `n_bins = n_data_bins+1`, R-BINS), `AxisKind::{Numeric,Missing}`, `AxisProvenance`, `BinConfig`, `subsample_for_binning` (home); `bin()`/`build_grid`; global per-feature quantile grid (seeded ~200k subsample); `max_bin` default **254** (`borders.len() ≤ max_bin−1` ⇒ `n_bins ≤ 255`, bin 0 = missing); midpoint borders for low-cardinality numerics; merged/union-grid rule; reserved missing bin + **learned default direction** (`Split.missing_left`); exposure offset plumbing (`offset = log(e)`); rare-level collapse; new/unseen-level → base. | `Loss::init_score`, `FitSpec.exposure` (§05/§09 consume). |
| **04** | Categorical handling | `AxisKind::CategoricalTS`, `TsEncodingId`, `TsConfig`, `CatEncoder`, `CatEncoderStore` (frozen `TsEncodingId→CatEncoder`, R-SCHEMA), `CatLevel`, `LeakageScheme`, `Smooth`; leakage-free ordered/cross-fitted **Target Statistics → Fisher sorted-ordinal split** (category stays a row); **empirical-Bayes auto-shrinkage** (`Smooth::Auto` = `within/between`); one-hot for low cardinality; **multi-distinct-categorical-axis trees** with raw-feature provenance; **forbid combination CTRs** (would break I1); the **audit-on-serve** rule (explain()/accumulation re-encode raw cats through the FROZEN `CatEncoderStore` into a `ServeBinnedMatrix`, NEVER the noisy `TrainBinnedMatrix`). | `AxisProvenance` (§03), `ServeBinnedMatrix`/`TrainBinnedMatrix` (§03), I1 enforcement (§03/§06). |
| **05** | Objectives & Loss trait | `Loss` trait (with fallible `init_score`/`deviance`, R-LOSSFALLIBLE) + `Link` + `Metric` + `LossId` + `ObjectiveTag` (link+loss+`tweedie_rho`, R-SCHEMA) + the methods `default_metric`/`hessian_floor` (default `1e-16`, R-TYPEDRIFT)/`max_delta_step`; `SquaredError`, `Logistic`, `Poisson`, `Gamma`, `Tweedie{rho}` with exact (g,h); `GradHess`; `BlendedLoss` distillation adaptor (`blend` = true-label weight, default 0.5, `blend=1.0` reproduces base loss); `max_delta_step` cap; hessian floor ε; deviance early-stop metric; `exp(k·F)` power rule. | `FitSpec`, `BinnedMatrix`, exposure offset (§03). |
| **06** | The oblivious boosting engine | `ObliviousTree`, `Split`, `Model` (struct, incl. the `schema: ModelSchema` block — R-SCHEMA — registered here as a Model sub-struct; §04/§05 own its `CatEncoderStore`/`ObjectiveTag`/`LossId` members), `Booster`, `FitSpec` (incl. `distill: Option<DistillSpec>` — R-DISTILL — threaded to §05/§09), `Config`, `Sampling`, `HistPrecision`, `Accel`, `Hist` (the i64 histogram accumulator); **oblivious level-wise summed Newton-gain split-finder** (`½Σ[G_L²/(H_L+λ)+G_R²/(H_R+λ)−G²/(H+λ)]`, exact `w*=−G/(H+λ)`), with the canonical missing low-bit `low = if bin==0 { split.missing_left } else { bin <= split.bin_le }` honored at split eval + sample→leaf update (R-MISSING); histogram engine + subtraction trick; `QuantGradHess` quantized histograms + stochastic rounding + **leaf refit from full precision**; `Config.max_delta_step: Option<f32>` (`None` ⇒ `Loss::max_delta_step()`); multi-step Newton leaves + Armijo; MVS sampling; per-work-unit re-seeding; LR×n_trees schedule; propagates fallible `Loss::init_score`/`deviance` with `?` (R-LOSSFALLIBLE). | `Loss` (§05), grids (§03), monotone + `InteractionPolicy` (§07), `DistillSpec` (§09). |
| **07** | Interaction selection & constraints | `InteractionPolicy` (`max_order` + `groups`), `MonotoneMap` (`BTreeMap<String, MonoSign>`, name→sign), `MonoSign`, `AdmissionPrior`, `HeredityMode`, `CredibilityFloor`, `LevelDecision`, `FeatureMask`; **`wht8`** (the frozen O(8) Walsh–Hadamard/Möbius transform: 8 leaf values → 8 fANOVA coefficients = 1 const + 3 main `c_i` + 3 pairwise `c_ij` + 1 triple `c_123` under that tree's per-cut `w`-marginals; invoked at §06 leaf estimation) + the **online per-order screening-variance accumulator** (running per-support Parseval sum `Σ_{S≠∅} m_S·c_S²` per `|S|` shell, fed one tree at a time; the `c_123²` triple witness is a soft online triple-detection prior); per-level joint leaf-clamp (→ graceful early-termination); `max_order∈{1,2,3}` whole-tree constraint; the **heredity + FAST-RSS + Sobol admission funnel** (soft prior, never a hard gate); **joint boost over admitted supports, single final purification**; gradient-based triple detector as soft front-end; credibility floors (`min_sum_hessian_in_leaf`, `path_smooth`). | Newton gain (§06), leaf estimation hook (§06), Sobol variance + purification (§08). |
| **08** | Explainability engine | `EffectTable`, `FeatureSet`, `TableBank`, `RefMeasure`, `Tensor`/`AxisId`; **accumulate → PURIFY (3→2→1→intercept, Lengerich mass-moving) → tables**; default `RefMeasure::ProductMarginals{laplace}`; Sobol importances `σ²(f_u)/σ²(F)`; exact interventional SHAP/Faith-Shap ≤order-3 as O(1) table reads; the **five Invariant checks** (implementation); complete-for-inference vs pruned-for-display view; per-cell effective-support annotation (display-only); post-hoc `w` recomputation. | `Model` (§06), I1/I2 (§3). |
| **09** | Predictiveness boosters | `DistillSpec`, `TeacherKind`, `RefitSpec`, `NesterovSpec`, `EnsembleSpec`, `OuterBag`, `DartSpec`; **CatBoost teacher-distillation** training mode (off-by-default via `FitSpec.distill: Option<DistillSpec>`; `blend` = true-label weight, default 0.5, `blend=1.0` = degenerate zero-teacher; soft gradient = base `grad_hess` called with `teacher_raw` as target; firewall-gated, exact to own tables); fully-corrective leaf refit (ridge over #trees×8); Nesterov/accelerated boosting (alpha mixing); **bagged greedy ensemble selection** (Caruana, Σα=1, α≥0, shared `w`, union grid, re-purify); outer-bag table-average on-ramp; refit-leaves; optional knobs (DART, random_strength) — all exactness-preserving. | `Model.trees` alphas (§06), purification linearity (§08), `Loss`/`BlendedLoss` (§05). |
| **10** | Inference & serialization | `Model` scoring (branch-free 8-cell lookup + table-sum); **`ScoringBank`/`PackedTree`** (the load-derived, scoring-only path-A view: `PackedTree` is `#[repr(C)] { feat:[u8;3], thresh:[u8;3], miss:u8, leaf:[f32;8] }` padded to one 64-byte cache line; `ScoringBank(Vec<PackedTree>)` is contiguous in stored tree order, built once in finalize/deserialize as a byte-exact re-encoding of the same f32 leaves and same compares — the serialized `Model` is untouched; the `u8` axis caps the hot path at ≤255 features, wider models take a side-table fallback; layout + kernel detailed in §11); `RatingExport`, `RatingTable`, `AxisExport`, `ModelDoc` (plain nested `{ format_version, schema_version, model }`, no `serde(flatten)`); serde format (**serde_json** canonical/self-describing + **bincode 2.x** fast binary via `bincode::serde::encode_to_vec`/`decode_from_slice` + frozen `bincode::config::standard()`, never pickle); `schema_version` round-trip; `TableBank` export format (JSON); determinism of stored grids/provenance. | `Model`, `TableBank`, `BorderGrid`. |
| **11** | Performance engineering & benchmarking | Column-major cache layout; rayon per-thread padded histograms + fixed-order reduce; SIMD via `multiversion`/`pulp`/`wide` on dense kernels (`target-cpu=x86-64-v3` baseline, AVX-512 at runtime); criterion benches; the bit-reproducibility test harness (the §1 determinism [GATE]). *(Internal yardstick only; no shipped benchmarking tooling.)* | quantized hist (§06), determinism rules (§1). |
| **12** | Python API & interop | PyO3 binding (`_pattern_boost`), maturin layout, **abi3-py310**; sklearn-compatible `PatternBoostRegressor`/`Classifier`; zero-copy numpy (`PyReadonlyArray2<f32>`, `into_pyarray`) + optional Arrow path; `py.detach` GIL release + scoped rayon pool (`n_jobs`); `.pyi` stubs/`py.typed`; `PbError`→Python exception mapping. | `Booster`, `FitSpec`, `Model`, `PbError`. |
| **13** | Testing, quality & engineering standards | Concrete realization of §1 (lints, MSRV CI, `cargo deny`, Miri, fmt); the five Invariant gates as build-blocking tests; proptest suites for purification identities; TreeSHAP as a **test oracle only**; coverage expectations. | §1 checklist, §3 invariants. |
| **14** | Roadmap & build order | Milestone sequencing (v1 spine: §05+§06+§07+§08 core → "beat EBM"; v1.5: Fisher-TS, multi-cat axes, quantized hist, distillation, ensemble; v2: fully-corrective refit, Nesterov, DART); fork-resolution log; non-goals (>3rd order, GPU train, ordered boosting, non-symmetric growth). | All sections; owns no types. |

**No-overlap rule:** if two sections appear to need the same decision, the **owner** in this table decides and the other cites it. Shared types are defined once in §2 and referenced everywhere; a section introducing a new public type must register it here first.

**Registration notes for the schema / serve-vs-train types (R-SCHEMA, R-CATSERVE):** `ModelSchema` is a §2.6 metadata block on `Model` (registered with §06 as a Model sub-struct; serialized with the `Model`, covered by `schema_version`). Its members are owned by their sections: `CatEncoderStore` by §04, `ObjectiveTag`/`LossId` by §05, `feature_kinds: Vec<AxisKind>` reusing §2.1's `AxisKind` (§03/§04). The `TrainBinnedMatrix` (out-of-fold/prefix categorical TS — leakage-free, FIT ONLY) and `ServeBinnedMatrix` (FROZEN full-data `CatEncoderStore` — PREDICT + audited-TableBank accumulation) are both owned by **§03** (skeleton registers here); numeric binning is fold-independent so Train==Serve there, only categoricals differ. I2 lossless-equivalence holds between the SERVED model function (frozen encoders) and the `TableBank`, both on a `ServeBinnedMatrix`; the audit-on-serve rule is stated by §04 and §08.

**Registration notes for the speed/performance additions** (architecture-exploits, `design/04`; all exactness- and determinism-neutral by construction):

- **`wht8` + the online per-order screening-variance accumulator** are owned by **§07** (invoked at the §06 leaf-estimation hook). **Critical caveat:** the per-tree coefficients live on each tree's OWN 2-point grid under that tree's `w`-marginals; trees cut different borders, so you **cannot sum coefficients across trees** (that drops cross-tree covariance). The accumulated per-order variance is therefore a **screening signal, NOT the audited ensemble Sobol** — it MUST NEVER touch the §08 invariant gates (`ThreeWayEqual`/`VarianceSum`/`Purity`/`Reconstruction`), which stay on the merged-grid purified bank (§08). Under `RefMeasure::Joint` the clean product form degrades to a heuristic. It is a **soft prior that never hard-gates** (per §07's funnel design), so it is exactness- and determinism-neutral. The idea "`wht8` replaces §08 Lengerich purification" is **WRONG and explicitly rejected** — `wht8` cannot cross the merged-grid alignment, and a second purification path is an untested I2 hazard.
- **`ScoringBank`/`PackedTree`** are owned by **§10** (a load-derived scoring view; layout + kernel detailed in §11). The leaf-select step over the 8-entry register-resident leaf LUT is an **in-register permute** (AVX2 `vpermps` / AVX-512 `vpermi2ps` / NEON `vqtbl`) via the existing `multiversion`/`pulp`/`wide` safe wrappers — **NOT** a hardware gather (`vgatherdps`), which is microcoded and load-port-bound. Same f32 selected, no reduction reorder; cross-row accumulation stays fixed tree order, so it is bit-reproducible. None of this alters the serialized `Model` (the determinism gate's subject).
