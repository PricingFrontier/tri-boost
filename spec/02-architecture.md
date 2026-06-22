## 02 — Architecture & project layout

> Owner section per the §4 ownership map. This section OWNS: the Cargo workspace layout (core / py / python / benches); the module boundaries inside `tri-boost-core`; the `PbError` + `Invariant` **definition** (the enum bodies live here, every other section maps its failures onto a variant); the `Backend` trait seam (CPU now, GPU-shaped later); the feature-flag matrix; and the `schema_version` policy. It USES every §2 shared type but defines none of them except `PbError`/`Invariant`. Grounded in research/05 (the polars/tokenizers dual-crate shape, abi3-py310, the updated MSRV 1.74 lint-table floor, quantized-reproducibility, defer-GPU-behind-a-`Backend`-trait) and the skeleton's engineering checklist (§1).

### 02.1 — Decision summary

| Decision | Choice | Why (which aim) |
|---|---|---|
| Workspace shape | maturin "separated": pure-Rust `tri-boost-core` + thin `tri-boost-py` (`cdylib`) + `python/` source + `benches/` | core is crates.io-publishable; pyo3 quarantined; FAST/DECOMPOSABLE machinery testable with zero Python in the loop |
| Crate count | two crates in one workspace (not a polars-style 40-crate split) | the core is one cohesive algorithm; over-splitting buys nothing and fragments the invariant gates |
| pyo3 location | ONLY `tri-boost-py`; `tri-boost-core` has zero pyo3 dependency, even optional | crates.io-publishable core; `cargo test` on the core never pulls a Python toolchain |
| Compute seam | a `Backend` trait in the core (CPU impl only in v1), GPU-shaped | keeps the door open without speculative GPU code; the pre-binned `u8` columnar layout is already GPU-friendly |
| Binding ABI | abi3-py310 (one wheel per `(os, arch)`) | matches polars/tokenizers; smaller release matrix |
| MSRV | 1.74, verified in CI | `[workspace.lints]` inheritance requires Rust 1.74; manylinux2014/glibc 2.17 remains compatible |
| Unsafe posture | `#![forbid(unsafe_code)]` on the core; SIMD via safe wrappers | engineering standard; no gratuitous unsafe |
| Error model | one `PbError` enum (defined here), `Result<T, PbError>` on every fallible public fn | typed errors, no panics in library code |
| `schema_version` | a single monotone `u32` on `Model`/`TableBank`, owned here, bumped on any wire-incompatible change | reproducible, auditable, cross-language load |

**Open fork (recommended default):** crate granularity. Recommended default is the **two-crate** workspace above. If, post-v1, the explainability engine (§08) or the distillation data hooks (§09) grow heavy optional dependency trees, split them into `tri-boost-explain` / `tri-boost-distill` sub-crates behind cargo features rather than feature-gating one fat crate. Not v1; flagged so the module boundaries below are drawn to make that split mechanical (each owns a top-level module already).

### 02.2 — Workspace & directory tree

```
tri-boost/
├── Cargo.toml                      # [workspace]; pins pyo3 + shared version + shared lint table at root
├── pyproject.toml                  # build-backend="maturin"; python-source/module-name/manifest-path
├── rust-toolchain.toml             # channel="stable" for contributors; CI verifies MSRV 1.74 with rustfmt, clippy
├── deny.toml                       # cargo-deny: licenses, advisories, bans  [GATE]
├── python/
│   └── tri_boost/
│       ├── __init__.py             # re-export from ._tri_boost; sklearn wrappers (§12)
│       ├── _tri_boost.pyi      # type stubs (§12)
│       ├── py.typed
│       └── sklearn.py              # TriBoostRegressor / Classifier (§12)
├── crates/
│   ├── tri-boost-core/         # PURE Rust. NO pyo3. crates.io-publishable.
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs              # #![forbid(unsafe_code)] #![deny(missing_docs)]; pub re-exports; PbError/Invariant (§02)
│   │       ├── error.rs            # PbError, Invariant  — OWNED HERE
│   │       ├── backend.rs          # Backend trait + CpuBackend  — OWNED HERE
│   │       ├── data/               # §03: BinnedMatrix, BorderGrid, AxisProvenance, binning, missing
│   │       ├── cat/                # §04: TS encoding, Fisher sorted-ordinal axis, provenance
│   │       ├── loss/               # §05: Loss trait, Link, SquaredError/Logistic/Poisson/Gamma/Tweedie
│   │       ├── engine/             # §06: ObliviousTree, Split, Model, Booster, FitSpec, Hist, histogram, split-finder
│   │       ├── constraints/        # §07: MonotoneMap, interaction-order, heredity/FAST/Sobol funnel
│   │       ├── explain/            # §08: EffectTable, TableBank, RefMeasure, purify, the 5 Invariant checks
│   │       ├── boosters/           # §09: distillation target, corrective refit, Nesterov, ensemble selection
│   │       ├── serialize/          # §10: serde_json + bincode, schema_version round-trip, scoring
│   │       └── simd/               # §11: multiversion/pulp/wide dense kernels (safe wrappers only)
│   └── tri-boost-py/           # THIN pyo3 binding. crate-type=["cdylib"].
│       ├── Cargo.toml              # tri-boost-core { path, version } + pyo3 + numpy
│       └── src/lib.rs              # #[pymodule] _tri_boost  (§12)
├── tests/                          # python integration tests (pytest, §13)
└── benches/                        # criterion benches + the bit-reproducibility harness (§11)
```

Module-to-section mapping is 1:1 with the §4 ownership table: each owning section gets exactly one top-level module under `core/src/`, so a section's code, its unit tests, and its slice of the invariant gates live together. `error.rs` and `backend.rs` are the two modules owned by THIS section. The histogram accumulator type (`Hist`, a `Vec<i64>` of per-(axis,bin) g/h sums) is owned by `engine/` (§06); this section's `Backend` trait references it by that single canonical name (see 02.5), never a divergent `HistogramSet`/`FeatureHist` alias.

### 02.3 — Workspace `Cargo.toml` shape

```toml
[workspace]
members = ["crates/tri-boost-core", "crates/tri-boost-py"]
resolver = "2"

[workspace.package]
version = "0.1.0"                 # shared; the wire schema_version is separate (02.8)
edition = "2021"
rust-version = "1.74"            # MSRV  [GATE: CI builds on exactly this]
license = "Apache-2.0"
repository = "https://github.com/.../tri-boost"

[workspace.dependencies]
# pinned ONCE here; each crate opts in. (research/05: pin pyo3 at the root.)
rayon       = "1"
ndarray     = "0.16"
serde       = { version = "1", features = ["derive"] }
serde_json  = "1"
bincode     = "2"                # 2.x; binary path uses bincode::serde::{encode_to_vec, decode_from_slice}
                                 # with a FROZEN bincode::config::standard() (§10) — byte-equality contract
rand        = { version = "0.8", default-features = false }
rand_pcg    = "0.3"              # Pcg64 — the named, versioned PRNG (§1); seeded by deterministic re-seeding, NOT "split"
thiserror   = "1"
num-traits  = "0.2"
smallvec    = { version = "1", features = ["serde"] }   # FeatureSet backing (§2.7)
multiversion = "0.7"             # runtime-dispatched SIMD; safe (§11)
pulp        = "0.18"             # safe SIMD abstraction; safe (§11)
pyo3        = { version = "0.29", default-features = false }
numpy       = "0.29"

# The shared lint table — applied identically in BOTH crates' Cargo.toml via
# `[lints] workspace = true`. This is the §1 panic/missing-docs GATE expressed
# as config. `tri-boost-core` also carries `#![forbid(unsafe_code)]` at crate
# root; this is intentionally not a workspace lint because PyO3 codegen in
# `tri-boost-py` must be locally allowed and `forbid` cannot be relaxed.
# NOTE on integer overflow: the no-overflow guarantee is enforced primarily by
# `overflow-checks = true` in ALL profiles (02.3a), not by a crate-blanket
# `arithmetic_side_effects` deny — which would make every legitimate accumulation a
# lint error. Clippy `arithmetic_side_effects` is therefore SCOPED (denied per-module on
# the index/bin/offset arithmetic that must be proven non-overflowing, not at crate root),
# and FLOAT arithmetic is explicitly exempt (overflow-checks and the lint apply to integers).
[workspace.lints.clippy]
unwrap_used            = "deny"
expect_used            = "deny"
panic                  = "deny"
unreachable            = "deny"
indexing_slicing       = "deny"
# arithmetic_side_effects: NOT crate-root-blanket. Scoped per-module (#![deny(..)] /
# #[allow(..)] with a // JUSTIFIED bounds proof) where integer index/bin math lives;
# floats exempt. See 02.3a and §13.

[workspace.lints.rust]
missing_docs   = "deny"
```

#### 02.3a — Integer-overflow & arithmetic policy (workspace-wide)

The skeleton's "no arithmetic that can overflow" rule (§1) is realized concretely as:

1. **`overflow-checks = true` in every workspace-root profile** — `[profile.dev]`, `[profile.release]`, `[profile.test]`, and `[profile.bench]` in the root `Cargo.toml` all set `overflow-checks = true`. Cargo ignores member-local profile settings in a workspace, so the root placement is part of the gate. Integer overflow traps (a checked panic) in every build, including release, so an overflow can never silently wrap; combined with the no-panic gate this means the only correct code is code that provably cannot overflow.
2. **Clippy `arithmetic_side_effects` is scoped, not crate-blanket** — it is denied module-locally on the hot integer index/bin/offset arithmetic (the `Hist` accumulation, bin id construction, row/axis indexing), where each such site carries a `// JUSTIFIED:` bound or is replaced by a checked form. It is NOT denied at crate root, because that would flag every benign `usize` increment. **Float arithmetic is explicitly exempt** from both the overflow-trap reasoning and this lint (floats saturate to ±inf, handled by the loss/numeric code, not by overflow-checks).
3. **The `Hist` bound is proven, not assumed** — §06 carries the proof that `n_rows · max|g_q| < i64::MAX` for the `i64` bin accumulators (02.5), so histogram accumulation cannot overflow even at large `n`.

This is the reconciliation of the prior "deny `arithmetic_side_effects` crate-wide" line, which was self-defeating: the real mechanism is `overflow-checks`, with the lint as a scoped sharpener.

#### 02.3b — Deterministic re-seeding (frozen)

The single `seed: u64` from `FitSpec` threads through every randomized stage. Per-work-unit independent streams are NOT produced by a "splittable" PRNG (`Pcg64` has no implementable split that yields position-stable, thread-count-independent draws). Instead, each work unit re-seeds deterministically:

```rust
// FROZEN. splitmix64 is the standard 64-bit mixer; this exact function is part of the
// determinism [GATE] contract and must not change without a schema/repro bump.
fn pb_seed(base: u64, round: u32, stage: u32, block: u32) -> u64 {
    let mut z = base
        ^ (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (stage as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (block as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
// then: Pcg64::seed_from_u64(pb_seed(base, round, stage, block))
```

Because the stream for any `(round, stage, block)` is a pure function of the base seed and the work-unit coordinates, draws are position-stable and **independent of thread count** — the basis of the §1 determinism `[GATE]`. The same scheme is referenced by §03.3 (binning subsample), §06.7 (MVS/split sampling), §09.6 (bagging), and §11. (`.wrapping_mul`/`>>`/`^` here are the documented exception to the integer-overflow trap: wrapping is intentional in the mixer.)

`tri-boost-core/Cargo.toml`:

```toml
[package]
name = "tri-boost-core"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[lints]
workspace = true                 # inherits the shared table above

[dependencies]
rayon.workspace = true
ndarray.workspace = true
serde.workspace = true
serde_json.workspace = true
bincode.workspace = true
rand.workspace = true
rand_pcg.workspace = true
thiserror.workspace = true
num-traits.workspace = true
smallvec.workspace = true
multiversion.workspace = true
pulp.workspace = true
# NOTE: no pyo3, no numpy. Verified by CI (02.6).

[profile.dev]
overflow-checks = true           # 02.3a — overflow traps in every profile
[profile.release]
overflow-checks = true
[profile.test]
overflow-checks = true
[profile.bench]
overflow-checks = true

[features]
default = []
arrow   = ["dep:arrow"]          # optional zero-copy Arrow/PyCapsule ingest (§03)
distill = []                     # CatBoost teacher DATA hooks only (§09); no FFI to CatBoost
nightly = []                     # portable_simd path; off the default/shipping wheel (§11)
```

`tri-boost-py/Cargo.toml` adds `pyo3 = { workspace = true, features = ["extension-module", "abi3-py310"] }`, `numpy.workspace = true`, the path+version dep on the core, and `crate-type = ["cdylib"]`. It is the ONLY crate where `extension-module`/`abi3` appear. Its `lib.rs` carries a module-local `#![allow(unsafe_code)]` because the pyo3 procedural macros expand to `unsafe`; this is the single, justified, encapsulated exception to the core-crate unsafe policy (research/05 §2).

### 02.4 — `PbError` + `Invariant` (OWNED HERE — defined verbatim from §2.8)

The error enum is the one defined in the skeleton (§2.8) and is reproduced here as its canonical home; no other section may redefine it. Every fallible public function in the workspace returns `Result<T, PbError>`; no public signature uses `Box<dyn Error>`.

```rust
/// The single crate error type. Defined in `tri-boost-core::error`.
#[derive(thiserror::Error, Debug)]
pub enum PbError {
    #[error("invalid input: {what}")]               InvalidInput { what: String },
    #[error("dtype mismatch: expected {expected}")] DtypeMismatch { expected: &'static str },
    #[error("shape mismatch: {what}")]              ShapeMismatch { what: String },
    #[error("invalid config: {what}")]              InvalidConfig { what: String },
    #[error("invariant violated: {invariant}")]     InvariantViolated { invariant: Invariant },
    #[error("exactness firewall: {0}")]             ExactnessFirewall(String),
    #[error("serialization: {0}")]                  Serialization(String),
    #[error("internal bug: {what}")]                Internal { what: String },
}

/// Which build-blocking invariant (§3) a failure maps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Invariant {
    FeatureBudget,     // I1: >3 distinct raw features / non-oblivious growth
    Decomposability,   // I2 purity: a table axis-slice has non-zero w-mean (the Purity check)
    MassConservation,  // purification did not conserve signed mass
    Reconstruction,    // per-cell |F_ens − Σ f_u| ≥ tol
    VarianceSum,       // σ²(F) != Σ σ²(f_u) under product/uniform w
    ThreeWayEqual,     // tree-sum != table-sum != Shapley-sum
}
```

As in §2.8, there is **no separate `Purity` variant**: the I2 purity check (every `EffectTable` axis-slice has `w`-weighted mean zero) reports its failures as `Invariant::Decomposability` — the implementation (`error.rs`) and §08 carry the same mapping.

Two policy rules this section sets for the whole workspace: (1) an "impossible" internal state returns `PbError::Internal { what }` rather than panicking — a logic bug degrades to a typed error visible at the Python boundary, never an abort in a user's session; (2) the `extension-module` feature being off means `PbError` carries no pyo3 dependency, so the core's error type is usable by pure-Rust consumers. The `PbError → PyErr` mapping (one variant → one Python exception class) is owned by §12; this section only guarantees the variants are stable and exhaustive.

This upholds the engineering standard directly: with the workspace lint table denying `unwrap`/`expect`/`panic`/`indexing_slicing` (and `arithmetic_side_effects` scoped per 02.3a, backed by `overflow-checks = true`), the only way to surface a failure is through this enum, which is the mechanism that keeps "no panics in library code" a compile-time / build `[GATE]`, not a convention.

**No-panic hot-loop policy (workspace-wide).** The `[GATE]` denies `indexing_slicing`, so the hot kernels reachable through `Backend` (histogram build, level argmax, grad/hess, `predict_block`) — and the canonical scoring/`fit` code in §06/§10 — MUST be written in one of exactly two forms, never raw `slice[i]`:

1. **Checked form (default):** `*slice.get(i).ok_or(PbError::Internal { what: "..".into() })?` for any index whose bound is not statically obvious. This is the form used wherever a `Result` is already in scope (the whole engine).
2. **Proven-unchecked form (perf-critical inner loops):** a small `#[allow(clippy::indexing_slicing)]`-scoped `fn` carrying a `// JUSTIFIED:` comment that proves the index in-bounds (e.g. "`idx ∈ 0..8` because it is built from three `bool` bits", "`rows[k] < n_rows` by the caller's contract"), plus a dedicated boundary unit test exercising the extreme indices. The proof and the test are mandatory; an `#[allow]` without both fails review.

The 8-cell leaf lookup (`leaves[idx]`, `idx = b0|b1<<1|b2<<2 ∈ 0..8`) and the per-row bin reads in `predict_block` are the canonical proven-unchecked sites; everything else uses the checked form. `Hist` accumulation likewise uses the proven-unchecked form with the 02.3a/02.5 bound, never wrapping arithmetic.

### 02.5 — The `Backend` trait seam (OWNED HERE)

GPU training is a non-goal for v1 (AIM "Non-goals"), but the data layout and the compute kernels are deliberately shaped so a GPU backend can be added without touching the engine, the invariants, or the public API. The seam is a trait over the four hot kernels — modeled on Burn's swappable-`Backend` design (research/05 §9) — not over the whole training loop. The boosting control flow, the split-finder's argmax, the invariant gates, and serialization stay backend-agnostic in `engine/`/`explain/`. The trait is `pub(crate)`: only `CpuBackend` ships in v1, and the trait references several `engine/`-local types, so it is not part of the public API surface (it is promoted to `pub` only if/when a second backend is shipped).

```rust
/// The compute seam. v1 ships only `CpuBackend`. A backend owns the four kernels that
/// dominate training/inference time; everything else (boosting loop, split argmax,
/// purification, serde) is backend-independent. A backend MUST be bit-reproducible:
/// identical inputs ⇒ identical outputs, independent of internal thread count (§1).
/// `pub(crate)` — internal seam, not a public API contract in v1.
pub(crate) trait Backend: Send + Sync {
    /// Build the integer g/h histogram for one level: per-(axis, bin) i64 sums into `Hist`.
    /// Quantized i64 integer sums are associative ⇒ order-independent ⇒ reproducible.
    /// `Hist` is the single §06-owned accumulator type (Vec<i64>); no `HistogramSet` alias.
    fn build_histograms(&self, x: &BinnedMatrix, gh: &QuantGradHess,
                        rows: &[u32], hist: &mut Hist) -> Result<(), PbError>;

    /// Evaluate the oblivious level-wise summed Newton gain for every candidate
    /// (axis, bin_le) and return the single argmax split for the whole level.
    fn best_level_split(&self, hist: &Hist, lambda: f32,
                        constraints: &LevelConstraints) -> Result<Option<Split>, PbError>;

    /// Accumulate full-precision per-row (g, h) for the current raw scores (§05 Loss).
    /// `Loss::grad_hess` is itself fallible (`-> Result<(), PbError>`, §2.4), so this
    /// kernel propagates that error with `?` — never `.expect`.
    fn grad_hess(&self, loss: &dyn Loss, y: &[f32], raw: &[f32],
                weight: &[f32], out: &mut GradHess) -> Result<(), PbError>;

    /// Branch-free 8-cell leaf lookup + table-sum scoring for a row block (§10).
    /// The 0..8 leaf index and per-row bin reads use the proven-unchecked hot-loop
    /// form of 02.4 (JUSTIFIED bound + boundary test), never raw indexing.
    fn predict_block(&self, model: &Model, x: &BinnedMatrix,
                    rows: &[u32], out: &mut [f32]) -> Result<(), PbError>;
}

/// v1 implementation: rayon per-thread padded `Hist`s + fixed-order reduce,
/// multiversion-dispatched dense kernels. Lives in `backend.rs`; the only Backend in v1.
pub struct CpuBackend { pub n_threads: usize }
```

`Hist` (the `Vec<i64>` per-(axis,bin) accumulator), `LevelConstraints` are `engine/`-local (§06/§07) types; this section only fixes the trait shape and the reproducibility contract, and references `Hist` by its one canonical name. The `i64` accumulator width is mandatory (not `i32`): §06 proves `n_rows · max|g_q| < i64::MAX`, so accumulation cannot overflow even at large `n` (an `i32` accumulator would overflow and, under `overflow-checks = true` (02.3a), trap — breaking the no-panic guarantee). Counts stay `u32`.

The bit-reproducibility `[GATE]` of §1 is a property of the `Backend` impl: `build_histograms` must use **quantized-integer (associative) accumulation OR, on the `FullF64` cross-check path (§06 `HistPrecision::FullF64`), a fixed-order float fold** — `fold` over fixed-size `par_chunks` (a fixed `CHUNK_ROWS` constant, §11) combined in index order, never rayon `reduce`/`sum` in steal order. Any unavoidable float reduction inside `grad_hess`/`predict_block` obeys the same fixed-order-fold rule. A `CpuBackend` constructed with different `n_threads` MUST produce byte-identical `Hist`s and predictions; the CI test trains at `n_threads ∈ {1, 2, 8}` and asserts model byte-equality. Note this trait does NOT expose leaf-value computation: leaves are always refit from FULL-PRECISION `GradHess` on the host (the exact `w* = −G/(H+λ)`, §06), so the quantization that buys reproducibility/speed never touches the values that go into the tables. That separation is what lets a future GPU `build_histograms` change the histogram path without endangering I2.

How the seam serves the three aims: (FAST) the four kernels are exactly the speed-critical ones and the only place a GPU could ever help; (DECOMPOSABLE) keeping leaf-value math and purification off the backend means I1/I2 are enforced by backend-independent code — a backend cannot bend an invariant; (ACCURACY) the seam is orthogonal to the learner, so it adds no accuracy risk. The cost of the abstraction is one trait-object indirection per level (amortized over a whole-level histogram build), which is negligible.

### 02.6 — Feature-flag matrix & the no-pyo3-in-core gate

| Feature | Crate | Default | Effect |
|---|---|---|---|
| (base) | `tri-boost-core` | — | numpy-free, pyo3-free, pure-Rust train/predict/explain on `f32` |
| `arrow` | core | off | optional zero-copy Arrow/PyCapsule column ingest (§03) |
| `distill` | core | off | CatBoost-teacher DATA hooks (soft-target ingestion only; no native FFI) (§09) |
| `nightly` | core | off | `portable_simd` kernels; never on the shipping wheel (§11) |
| `extension-module` | `tri-boost-py` | on for wheels, off for `cargo test` | pyo3 builds a `cdylib` Python extension |
| `abi3-py310` | `tri-boost-py` | on | stable-ABI: one wheel per `(os, arch)` for CPython ≥3.10 |

Rules (build `[GATE]`s): (1) **default features of the core must compile and pass all five Invariant gates on stable Rust** — no optional feature is load-bearing for correctness; (2) `cargo tree -p tri-boost-core | grep -q pyo3` must return non-zero in CI — pyo3 in the core fails the build; (3) the core must build for a non-Python target (e.g. `cargo build -p tri-boost-core --target wasm32-unknown-unknown` smoke-builds the no-Python guarantee — which is exactly why all serialized index fields are fixed-width, never `usize`, see 02.8); (4) `extension-module` is OFF during `cargo test` so the core's unit/invariant tests run without a Python interpreter (research/05 §2). `arrow`/`distill`/`nightly` are additive and independently composable.

### 02.7 — Build, CI matrix & distribution

`pyproject.toml` `[tool.maturin]`: `python-source = "python"`, `module-name = "tri_boost._tri_boost"`, `manifest-path = "crates/tri-boost-py/Cargo.toml"` (essential in a workspace), `features = ["pyo3/extension-module"]`, `strip = true`, plus `include` for `py.typed`/`.pyi`. `[build-system]` is `requires = ["maturin>=1.9,<2"]`, `build-backend = "maturin"`.

CI is two pipelines. **Correctness CI** (every PR, the build `[GATE]`s) runs on the core: `cargo fmt --check`, `cargo clippy -- -D warnings` (with the workspace lint table + the scoped `arithmetic_side_effects` of 02.3a), `cargo test` (unit + the five Invariant gates + proptest purification identities), `cargo deny check`, the bit-reproducibility harness at `n_threads ∈ {1,2,8}`, doctests, and a Miri run over any `unsafe`-adjacent code. Builds use `overflow-checks = true` in every root profile (02.3a). MSRV is verified by building the core on exactly Rust 1.74. **Release CI** (tag-triggered) uses `maturin-action` to build wheels per `(os, arch)`: linux x86_64/aarch64 + musllinux, macOS x86_64/aarch64, Windows x64; with abi3 that is one wheel per platform, not per interpreter. Targets manylinux2014 (x86_64) / manylinux_2_28 (aarch64); wheels build with a portable `-C target-cpu=x86-64-v3` baseline (AVX2), lifting to AVX-512 at runtime via `multiversion` — never `target-cpu=native`. Publishing is OIDC Trusted Publishing to PyPI; `tri-boost-core` is published to crates.io first (`cargo publish -p tri-boost-core`, wait for indexing, then the py crate is not published to crates.io — it ships only as a wheel).

### 02.8 — `schema_version` policy (OWNED HERE)

`Model.schema_version: u32` and the analogous field on `TableBank` (§10) are the single wire-format version, distinct from the crate's semver. Policy: it starts at `1`; it is bumped **only** when the serialized byte layout changes in a way an older reader cannot losslessly parse. Additive, backward-compatible field additions use `#[serde(default)]` (and `#[serde(alias = "old")]` for renames) and do NOT bump it (research/05 §7). On load, `serialize/` checks `schema_version <= CURRENT`; an unknown future version returns `PbError::Serialization`, and a known older version is routed through an explicit `From`-based migration. `deny_unknown_fields` is NOT used on long-lived structs. Because `Model` and its `#[derive(Serialize, Deserialize)]` live in the pure-Rust core, a model trained from Python and one trained in a pure-Rust pipeline serialize and deserialize identically — one serde impl, no format drift.

**Cross-platform byte-equality requires fixed-width serialized index fields.** Any field that is serialized AND is an index/count MUST have a fixed wire width, never the platform-dependent `usize` (the core smoke-builds on `wasm32`, where `usize` is 32-bit, vs 64-bit on the host — a `usize` index would serialize to a different byte length and break the determinism `[GATE]`). The frozen choices, owned jointly with §10/§06/§03: `Split.axis: u32` and `BinnedMatrix.n_rows: u32` (a single `u32` decision — >4B rows is out of scope for v1; if ever needed the bump is to `u64`, chosen once across all index fields). `Split` also carries the explicit `missing_left: bool` missing-direction field (§06/§08 cite it; one byte, the clearest carrier rather than a `bin_le` sentinel). The binary path uses bincode 2.x's `bincode::serde::encode_to_vec(.., bincode::config::standard())` / `decode_from_slice(.., bincode::config::standard())` with the config **frozen** (the removed top-level `bincode::serialize`/`deserialize` are not used); and `ModelDoc` is a plain nested `ModelDoc { format_version, schema_version, model: Model }` — NOT `#[serde(flatten)]`, which is incompatible with the non-self-describing bincode round-trip (§10 owns the format details; this section owns the version-gate and the fixed-width/index-field policy). This is what makes the exported tables an auditable, reproducible contract rather than an opaque blob (serving DECOMPOSABLE: the deployed bytes and the audited bytes are provably the same artifact).

### 02.9 — How this section upholds I1/I2 and serves the three aims

**I1/I2 are protected structurally by the layout, not just by §06/§08.** The module boundaries put all leaf-value math, all tree construction, and all purification in backend-independent code (`engine/`, `explain/`); the `Backend` seam is restricted to histogram/gain/grad-hess/predict kernels that cannot, by their signatures, introduce a fourth feature into a tree or a non-constant leaf. The `ExactnessMode` firewall (§3) lives on `Model` and is checked in `explain/`, which no backend can reach. So "a backend cannot break an invariant" is a property of where the code is allowed to live.

**ACCURACY:** the architecture is deliberately neutral to the learner — losses, sampling, constraints, distillation, and ensembling are separate modules that never touch tree shape, so every gap-closer from §09 composes without architectural friction.

**FAST:** the two-crate split keeps the hot path (`engine/`, `simd/`, `backend.rs`) free of pyo3/Python overhead; the GIL is released at the §12 boundary and a scoped rayon pool runs the `CpuBackend`; the pre-binned `u8` columnar layout is simultaneously the fast-CPU layout and the GPU-ready layout, so the GPU door stays open at zero v1 cost.

**DECOMPOSABLE:** one core crate owns `Model`'s serde and `schema_version`, so the bytes deployed equal the bytes audited; and the invariant gates run in Correctness CI on the pure core with no Python in the loop, so the lossless property is enforced as a build gate (§13) regardless of how the model was produced.

### 02.10 — Testing approach for this section

The architecture's own correctness is testable independently of the learner: (1) **a no-pyo3-in-core test** — a CI step asserting `cargo tree -p tri-boost-core` contains no `pyo3`/`numpy`, plus a non-Python-target (`wasm32`) build of the core; (2) **a `Backend` reproducibility test** — `CpuBackend { n_threads }` for `n ∈ {1,2,8}` over the same fixed inputs must yield byte-identical `Hist` and `predict_block` output (the §1 `[GATE]`); (3) **an error-mapping completeness test** — every `Invariant` variant is reachable and maps to exactly one `PbError::InvariantViolated`, and every public fallible fn in the core returns `Result<_, PbError>` (enforced by clippy + a doc-test convention); (4) **a serde round-trip + `schema_version` test** — `Model`/`TableBank` round-trip through both serde_json and bincode (2.x, frozen `config::standard()`) bit-identically, the `ModelDoc` nested (non-`flatten`) form round-trips through bincode, a bumped-version blob is rejected with `PbError::Serialization`, an aliased old field still loads, and a fixed-width-index model serialized on the host deserializes byte-identically against a `wasm32`-built schema (the `usize`→`u32` guard, §10 owns the format); (5) **a feature-matrix build test** — the core builds and passes the Invariant gates under each combination of `{arrow, distill, nightly}` and with default features only on stable; (6) **a no-panic hot-loop boundary test** — every proven-unchecked `#[allow(clippy::indexing_slicing)]` site of 02.4 (the 8-cell leaf index, `predict_block` bin reads, `Hist` accumulation) has a unit test exercising the extreme indices, and an overflow-trap test confirms `overflow-checks` is live in the test profile (02.3a). These are unit/integration tests in the core plus CI config; none require the Python toolchain, which is itself the point.
