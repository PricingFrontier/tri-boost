# PHASE 0 — THE ENGINEERING FOUNDATION (Implementation Plan)

## How this area's structure makes quality automatic

Phase 0 builds **the engine that makes quality a byproduct of how work is ordered**, not a thing audited later. Its single design principle: *every gate exists before the first line of feature code, so the first feature commit physically cannot merge if it panics, is unformatted, leaks a non-typed error, or breaks reproducibility.*

Three structural mechanisms do the enforcing:

1. **Gates-as-config, committed in PR #1.** The §1/§13.8 lint set (`unwrap_used`/`expect_used`/`panic`/`indexing_slicing` = `deny`, `forbid(unsafe_code)`, `deny(missing_docs)`), `overflow-checks=true` in all profiles (§02.3a), `cargo-deny`, the MSRV-1.64 pin, and doctests all land as `Cargo.toml`/`deny.toml`/`rust-toolchain.toml`/CI-YAML *before any algorithm*. Because they're inherited via `[lints] workspace = true`, a new module is governed the instant it's created — quality is opt-out-impossible, not opt-in.
2. **Illegal states unrepresentable, first.** `PbError` + `Invariant` (§2.8/§02.4) and `ExactnessMode` (§3) are the *first types defined*, so every later fn is forced to return `Result<T, PbError>` and every exactness-bending op is forced through the firewall type. There is no "add error handling later" — the type system already requires it.
3. **The five invariant checks + determinism gate exist as functions/CI jobs (over fixtures) before any tree is fit (Gate G0, §14.2).** When §06/§08 land, they fill in already-wired, already-build-blocking gates — "tables == ensemble" and `{1,2,8}`-thread byte-equality are enforced from the first tree, never retrofitted.

This area owns the §02 (architecture) + §13 (gate machinery) realization for Phase 0; it references the §13 gate definitions and §14 phase-gates rather than re-specifying them. Downstream phases (P1–P7) consume only a green foundation.

---

## Sequenced tasks

> **DoD vocabulary (the Definition-of-Done template, established in F0):** a task is *done = gates-green*, not code-written. Named gates: **Fmt** (`cargo fmt --check`), **Clippy** (`cargo clippy --all-targets --all-features -- -D warnings`, incl. the §13.8 no-panic deny set), **Deny** (`cargo deny check`), **MSRV** (build+test on pinned 1.64), **Doctests** (`cargo test --doc`, + `deny(missing_docs)` as compile gate), **Test** (`cargo test`), **NoPyo3** (`cargo tree -p tri-boost-core | grep pyo3` returns non-zero), **Wasm** (`cargo build -p tri-boost-core --target wasm32-unknown-unknown`), **Determinism** (§13.4 byte-equal model across `n_threads∈{1,2,8}`), **Invariants** (§13.1 five checks via `assert_exact_decomposition`).

---

### F0 — Workspace skeleton + the lint/profile gate table
**Spec:** §02.2, §02.3, §02.3a, §1, §14.2-P0
**Deliverable:** Root `Cargo.toml` (`[workspace]`, `resolver=2`, `[workspace.package]` with `rust-version="1.64"`, `[workspace.dependencies]` pinned per §02.3, `[workspace.lints.clippy]` + `[workspace.lints.rust]` exactly per §02.3); the directory tree of §02.2 (`crates/tri-boost-core`, `crates/tri-boost-py`, `python/`, `tests/`, `benches/`); both crate `Cargo.toml`s inheriting `[lints] workspace = true`; `overflow-checks = true` in `[profile.{dev,release,test,bench}]`; `tri-boost-core/src/lib.rs` with `#![forbid(unsafe_code)] #![deny(missing_docs)]` and an empty-but-documented module tree (`error`, `backend`, `data`, `cat`, `loss`, `engine`, `constraints`, `explain`, `boosters`, `serialize`, `simd`) matching the §4 ownership map 1:1; `tri-boost-py/src/lib.rs` stub with the module-local `#![allow(unsafe_code)]`.
**Dependencies:** none (first PR).
**DoD:** Fmt, Clippy (passes on empty crate — proves the deny-set is wired), MSRV, Doctests (`deny(missing_docs)` compiles), Test (`cargo test` runs, zero tests). Every module file has a `//!` doc citing its owning §.
**Size:** M

---

### F1 — `PbError` + `Invariant` (the typed-error firewall foundation)
**Spec:** §2.8, §02.4, §13.8
**Deliverable:** `error.rs` with `PbError` (8 variants, `thiserror::Error`) and `Invariant` (6 variants, `Copy+Eq`) defined **verbatim** from §02.4; re-exported from `lib.rs`. A unit test asserting every `Invariant` variant maps to exactly one `PbError::InvariantViolated` and that `PbError: std::error::Error + Send + Sync`. A grep-gate script (`xtask check-no-box-dyn` or CI step) asserting no `Box<dyn Error>` in public signatures (§13.8).
**Dependencies:** F0.
**DoD:** Fmt, Clippy, Doctests (each variant doc'd with an `///` example of when it's returned), Test (variant-mapping completeness test green), the `Box<dyn Error>` grep-gate green.
**Size:** S

---

### F2 — `ExactnessMode` firewall type + the canonical §2 type stubs
**Spec:** §3, §2.1–2.6, §2.9, §14.2-P0
**Deliverable:** `ExactnessMode { Exact, Approximate { reason: String } }` (§3) in `engine/` (or `lib.rs` per ownership). Compiling-but-stub canonical types with **verbatim signatures** from §2: `FeatureId`, `AxisKind`, `AxisProvenance` (§2.1); `BorderGrid`, `BinnedMatrix` (`n_rows: u32`) (§2.2); `GradHess`, `QuantGradHess`, `GradScale`, `Hist` (§2.3); `Loss` trait + `Link` + `Metric` (§2.4); `ObliviousTree`, `Split` (`axis: u32`, `missing_left: bool`) (§2.5); `Model` (+ `ModelSchema`, `ObjectiveTag` placeholders), `schema_version: u32` (§2.6); `Booster`, `FitSpec`, `InteractionPolicy` (§2.9). Bodies are `todo!()`-free stubs returning `Err(PbError::Internal{..})` where a fn must compile. **No algorithms** — the goal is the type contract frozen.
**Dependencies:** F1.
**DoD:** Fmt, Clippy (no-panic deny set passes — proves stubs use `Err(..)` not `unimplemented!`/`panic!`), Doctests (`deny(missing_docs)` forces every pub item documented), MSRV, Test. A compile-time test asserting `Split.axis: u32` / `BinnedMatrix.n_rows: u32` (fixed-width, §02.8). **This freezes the §2 contract that all downstream phases build on.**
**Size:** L

---

### F3 — Correctness CI pipeline (all §1 gates active from PR #1)
**Spec:** §02.7, §13.8, §13.9, §13.10, §1
**Deliverable:** `.github/workflows/ci.yml` Correctness pipeline running, on every PR, all build-blocking gates over the core: **Fmt**, **Clippy** (`--all-targets --all-features -- -D warnings`), **Test** (`--all-features` + `--no-default-features`), **Doctests**, **Deny** (`cargo deny check` — `deny.toml` with license allow-list MIT/Apache-2.0/BSD/Unicode-3.0, `RUSTSEC` advisories deny, no `openssl`, crates.io-only, no duplicate-major bans), **MSRV** (job on pinned 1.64 via `rust-toolchain.toml`, channel `1.64` + rustfmt/clippy components), **NoPyo3**, **Wasm** smoke-build. Two grep-gates wired as CI steps: §13.4 `usize`-in-serialized-index-field forbid, and `std::collections::HashMap`-in-serialized-state forbid. `unsafe`/`unsafe_ok` diff-grep audit step (§13.8). Feature-matrix build step (`{arrow,distill,nightly}` each build — even if features are empty stubs now).
**Dependencies:** F0 (needs the workspace + lint table); F1, F2 (so `cargo test` has the contract to compile).
**DoD:** the pipeline itself is green on the F0–F2 skeleton: every named gate above passes on the stub crate. **This is the PR that makes "the first line of feature code already cannot merge if it panics or is unformatted" literally true.**
**Size:** L

---

### F4 — Test-harness layout (§13 tree) + xtask + the DoD/PR/quality doc
**Spec:** §13 (intro tree), §13.7, §13.8 template, §1
**Deliverable:** The §13 directory layout instantiated: `crates/tri-boost-core/tests/` (integration), `tests/invariants/` (determinism + invariant harnesses), `benches/` (Criterion skeleton), `xtask/` crate (dev-only accuracy harness — ships **no** library code, a `xtask accuracy`/`xtask check-*` binary stub). A `CONTRIBUTING.md` quality doc encoding: the **Definition-of-Done template** (DoD = gates-green + spec §-ref, named gates per the F0 vocabulary), small-reviewable-PRs-on-branches + trunk-stays-green discipline, the `// JUSTIFIED:` comment convention for any proven-unchecked `#[allow(clippy::indexing_slicing)]` / scoped `arithmetic_side_effects` (§13.8), and the no-`unwrap`/`expect`-outside-tests rule. A CI grep-gate asserting every non-test `unwrap`/`expect` and every form-(b) `#[allow(clippy::indexing_slicing)]` carries a `// JUSTIFIED:` comment (§13.8).
**Dependencies:** F0, F3 (CI to host the new grep-gates).
**DoD:** Fmt, Clippy, Test (empty harnesses compile + run), the two `// JUSTIFIED:` grep-gates green (vacuously true now — but live, so they fire the instant feature code adds an `unwrap`), Criterion benches compile (`cargo bench --no-run`), `cargo run -p xtask -- --help` works, NoPyo3 still holds (xtask is dev-only).
**Size:** M

---

### F5 — The `Backend` trait seam + `CpuBackend` stub (the I1/I2 structural firewall)
**Spec:** §02.5, §2.3 (`Hist`), §1 (determinism contract)
**Deliverable:** `backend.rs` with the `pub(crate) trait Backend: Send + Sync` defined **verbatim** from §02.5 — the four kernels (`build_histograms`, `best_level_split`, `grad_hess`, `predict_block`), each returning `Result<_, PbError>`, referencing `Hist` by its single canonical name (no `HistogramSet` alias). `CpuBackend { n_threads: usize }` struct + stub impl returning `Err(PbError::Internal{..})`. A doc comment on the trait stating the bit-reproducibility contract (identical outputs independent of internal thread count) and the leaf-values-never-on-backend rule that protects I2. Stub `LevelConstraints` placeholder type (owned later by §06/§07, registered as a stub here so the trait compiles).
**Dependencies:** F2 (needs `Hist`, `BinnedMatrix`, `QuantGradHess`, `Model`, `GradHess`, `Split`).
**DoD:** Fmt, Clippy, Doctests (`deny(missing_docs)` + the reproducibility-contract doc), Test. **Structural DoD:** the trait signatures make it *impossible* for a backend to introduce a 4th feature or a non-constant leaf (no leaf-value method exists) — §02.9's "a backend cannot break an invariant" is enforced by where code is allowed to live.
**Size:** M

---

### F6 — Determinism-gate harness skeleton (the reproducibility engine, pre-wired)
**Spec:** §13.4, §1, §02.3b (`pb_seed`/`splitmix64_mix`), §02.10(2)
**Deliverable:** In `tests/invariants/`: a `determinism` harness with the helper that trains a model at `n_threads ∈ {1,2,8}` (forcing a rayon pool of each size via a scoped `ThreadPoolBuilder`) and asserts byte-equality of `bincode::serde::encode_to_vec(&model, bincode::config::standard())` (frozen config) with tolerance 0 — driven now by a stub/hand-built `Model` fixture (it asserts the *harness* works; it goes live when §06 lands). The frozen `pb_seed(base, round, stage, block)` `splitmix64` mixer (§02.3b) implemented **verbatim**, with a unit test pinning known input→output vectors (so the determinism contract's RNG is frozen from day one). A wired-but-pending CI job invoking the harness.
**Dependencies:** F2 (needs `Model` + the serde stub), F4 (`tests/invariants/` layout), F5 (`CpuBackend { n_threads }` to vary).
**DoD:** Fmt, Clippy, Test (`pb_seed` known-vector test green; the byte-equality harness runs green on the stub `Model` fixture). The CI determinism job exists and is green. **This makes §06's Gate G2 `{1,2,8}`-byte-equality a fill-in-the-fixture, not a build-the-harness, task.**
**Size:** M

---

### F7 — The five `Invariant` checks as functions + Gate G0 wiring (over fixtures)
**Spec:** §13.1, §13.2, §3, §14.2-P0 (Gate G0)
**Deliverable:** In `explain/` (owned by §08, stubbed here per Gate G0) the **signatures verbatim** from §13.1: `assert_exact_decomposition(model, bank, grid_corners) -> Result<(), PbError>`, `ExactTol` (with `recon_tol = 4.0 * n_trees * f32::EPSILON as f64`, `ExactTol::for_model`), and `check_feature_budget(model) -> Result<(), PbError>` (§13.2). Each of the five checks (Reconstruction/MassConservation/Purity/VarianceSum/ThreeWayEqual) is a **separate `#[test]`** so a failure names the broken property — initially run over hand-built fixtures (per Gate G0: "the five checks exist as functions returning `Result<(), PbError>` over hand-built fixtures"). Bodies return `Err(PbError::Internal{ what: "stub" })` or pass trivially on the fixture; the *wiring* and the per-check test scaffolding is the deliverable. The §13.1 negative gate stub (no `wht8` element is read by `assert_exact_decomposition`).
**Dependencies:** F2 (`Model`, `TableBank`/`EffectTable` stubs, `Invariant`), F4 (`tests/invariants/`).
**DoD:** Fmt, Clippy, Doctests, Test — all five named checks present as separate `#[test]`s and green on fixtures; `check_feature_budget` present. **Gate G0 is green: the five invariant checks exist as build-blocking functions and the CI matrix (lints/fmt/deny/MSRV/determinism) is green — the contract is frozen, ready for §06/§08 to fill in real bodies against already-wired gates.**
**Size:** M

---

### F8 — Phase-0 exit verification (G0 green end-to-end) + §01 milestone freeze hook
**Spec:** §14.2 (Gate G0), §14.5 (section→phase map), §01 (authored alongside P0)
**Deliverable:** A top-level CI status check aggregating F3+F4+F6+F7 gates as the **Gate G0 build-blocking set**. Confirm the full §02.10 self-test list passes: NoPyo3 + Wasm build, `Backend` reproducibility harness (on stub), error-mapping completeness, serde round-trip + `schema_version` rejection of a bumped blob (over the §2.6 stub `Model`), feature-matrix build, no-panic hot-loop boundary-test scaffold + an `overflow-checks`-is-live trap test (§02.10(6)). Confirm `CONTRIBUTING.md` DoD template is referenced by a PR template (`.github/pull_request_template.md` requiring spec §-ref + gates-green checklist).
**Dependencies:** F1–F7.
**DoD:** **Gate G0 (§14.2)** green in CI: types frozen, CI matrix green, the five `Invariant` functions + determinism harness + `pb_seed` exist and pass on fixtures, NoPyo3/Wasm/feature-matrix/serde-version/overflow-trap all green. No algorithm exists yet — and *that is the point*: the next phase (P1 binning, §03) builds on a foundation where every standard is already a live, build-blocking gate.
**Size:** S

---

## Critical-path ordering (nothing builds on an unverified foundation)

`F0 → F1 → F2` (workspace+lints → typed errors → frozen §2 contract) is the spine. `F3` (CI) depends on F0–F2 so the gates run against real stubs. `F4` (harness+DoD) and `F5` (Backend seam) parallelize after F2/F3. `F6` (determinism) needs F2+F4+F5; `F7` (invariant checks) needs F2+F4. `F8` aggregates all into **Gate G0**. Total: 3 S, 4 M, 2 L tasks — each an end-to-end *runnable-and-gated* vertical slice, never a pile of untested modules. The deliverable of Phase 0 is not code that does something; it is the **machinery that makes the next person's code unable to be wrong** — gates before features, made structural.
