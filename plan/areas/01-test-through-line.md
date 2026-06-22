# tri-boost — Test / Invariant / Determinism / Bench Through-Line: Implementation Plan

## How this area's structure makes quality automatic

This area is not a test suite written after the code; it is the **scaffolding the code is poured into**. Three mechanisms make high standards a byproduct of sequencing rather than a discipline anyone has to remember:

1. **The gate machinery is the first PR.** Before a single line of algorithm exists, the workspace lint table (§02 `[workspace.lints]`), `clippy -D warnings` with the no-panic set (`unwrap_used`/`expect_used`/`panic`/`indexing_slicing`), `fmt --check`, `cargo-deny`, MSRV-1.74 job, `deny(missing_docs)`, doctests, `overflow-checks=true` in all root profiles, and the grep-gates (`usize`-in-serialized-state, `Box<dyn Error>`, `// JUSTIFIED:` pairing, no-`HashMap`-in-fit-order) are all green in CI. The *first* feature commit therefore physically cannot merge if it panics, is unformatted, or pulls a banned license. Quality is the default state of the trunk, not an aspiration.

2. **The invariant harness exists before the model it audits.** The five §13.1 checks (`assert_exact_decomposition`) and `check_feature_budget` are authored as functions-over-fixtures in Phase 0 (Gate G0), so the moment Phase 3 produces a real `TableBank`, the gate already runs over it. The model is gated from tree #1 — "tables == ensemble" is enforced as the spine is built, never audited afterward.

3. **The ordering rule (binding for every milestone area):** *for each feature, its property test / numerical oracle / determinism assertion is written in the same PR as the feature, and the feature's Definition-of-Done is "those named gates green," not "code written."* Milestone areas (§03–§12) reference the harnesses defined here; they do not re-specify gates. The determinism gate and the I2 gates run on **every fitted model in the corpus**, so any milestone that erodes exactness or reproducibility breaks trunk immediately.

Tasks are ordered so nothing builds on an unverified foundation: gate machinery → invariant/budget skeletons → determinism harness → property-suite scaffolding → per-phase oracle suites → fuzz/golden/bench/accuracy harnesses.

---

## Sequenced tasks

### T0 — Gate machinery & CI skeleton (the first PR)
- **§-ref:** §1, §13.8, §13.9, §13.10, §02.3/02.3a
- **Deliverable:** Root `Cargo.toml` `[workspace.lints]` table (clippy no-panic set as `deny`, `rust` warnings, `missing_docs`), `overflow-checks=true` in *all root* profiles, `rust-toolchain.toml` (stable contributor toolchain), `deny.toml` (license allow-list MIT/Apache-2.0/BSD/Unicode-3.0, RUSTSEC deny with the justified bincode 2.x unmaintained-advisory exception, crates.io-only, no-`openssl` ban). CI workflow `correctness.yml`: jobs for `fmt --check`, `clippy --all-targets --all-features -- -D warnings`, `cargo deny check`, MSRV build, `--doc`, `--no-default-features`, the `wasm32` smoke-build. Grep-gate jobs: `usize` in `#[derive(Serialize)]` index fields; `Box<dyn Error>` in public sigs; every non-test `unwrap`/`expect` and every form-(b) `#[allow(clippy::indexing_slicing)]` carries a `// JUSTIFIED:`; no `std::collections::HashMap` in fit-order/serialized-state paths.
- **Deps:** none (with the §02 scaffold task it gates).
- **DoD:** `correctness.yml` green on the empty/stub workspace; clippy denies a deliberately-planted `unwrap()` (negative test in CI); `cargo deny check` passes; MSRV job builds on 1.74.
- **Size:** M

### T1 — Invariant & budget check signatures + fixture harness (G0)
- **§-ref:** §13.1, §13.2, §3, §2.8, §08.6
- **Deliverable:** In `tests/invariants/`: `assert_exact_decomposition(model, bank, grid_corners) -> Result<(), PbError>` and `check_feature_budget(model) -> Result<(), PbError>` as **five separate `#[test]`-callable functions** plus the budget check, each returning `InvariantViolated { invariant }` naming the broken property. `ExactTol { recon_tol, mass_tol, purity_tol, var_tol }` with `for_model` (`recon_tol = 4.0 * n_trees * f32::EPSILON`). `CellCornerIter` (cartesian product of realized borders). Initially driven over **hand-built fixtures** (the AND/OR/XOR purification degeneracy, a known 2-leaf model). The negative-property test: no element of the `wht8` accumulator is ever read by `assert_exact_decomposition`.
- **Deps:** T0; §02 stub types (`Model`, `TableBank`, `PbError`, `Invariant`).
- **DoD (this is G0):** all five checks + budget check exist returning `Result`, compile under the no-panic lints, pass on hand-built fixtures, and are wired into `correctness.yml`. The `wht8`-exclusion negative test green.
- **Size:** M

### T2 — Determinism harness (thread-invariance + cross-run + grep gates)
- **§-ref:** §13.4, §1, §11.3, §11.7
- **Deliverable:** In `tests/invariants/determinism.rs`: a harness that trains the same `(BinnedMatrix, y, FitSpec, seed)` at `n_threads ∈ {1,2,8}` via the §11 scoped `ThreadPoolBuilder`, encodes each via `bincode::serde::encode_to_vec(.., config::standard())`, and asserts byte-equality (tolerance 0). Companion: predictions bit-equal across thread counts on a held-out matrix. Cross-run reproducibility check (two-process invocation → byte-equal). A helper to force AVX-512 off/on and re-run (the §11.4 SIMD-determinism assertion). The harness is a reusable fn so every later phase plugs its fitted models in.
- **Deps:** T0, T1.
- **DoD:** harness compiles and runs as a no-op over a stub `fit` (or a trivial constant model) green; wired as a **[GATE]** job; the AVX-toggle path exercised. Becomes load-bearing at G2.
- **Size:** M

### T3 — proptest scaffolding + frozen-RNG config + generators
- **§-ref:** §13.3
- **Deliverable:** A shared `proptest` config (`ProptestConfig { cases: 256, max_shrink_iters: 4096, .. }` seeded from a frozen RNG) and reusable `Strategy` generators: random `Vec<f32>` columns, random `BinConfig`/`max_bin`, random small `FitSpec`s (loss, order, monotone map, seed), random order-≤3 effect tensors on random merged grids with random positive `w`. These feed every property suite below so suites stay uniform and reproducible.
- **Deps:** T0; §02 stub types.
- **DoD:** generators compile, shrink deterministically (a planted failing case reproduces from the frozen seed); a smoke property (`bin` total never panics) green.
- **Size:** S

### T4 — Loss-correctness + loss-domain-error property suites (with §05)
- **§-ref:** §13.3 (loss correctness), §13.3 (loss domain errors **[GATE]**), §13.5 (quantization bound), §2.4
- **Deliverable:** Property suite asserting, per `Loss`: finite-difference of `deviance` w.r.t. `raw` matches `out.g` (1e-3 rel), second finite-difference matches `out.h`; `init_score == link(weighted mean)` (1e-6); `pred_from_raw` exact inverse link; `exp(k·F)` vs `powf` reference within f32 tol. All assertions `?`-propagate the fallible `Result`. **Domain-error [GATE] unit suite**: Gamma/Tweedie `y<=0`, all-zero weight, all-zero-positive Logistic, zero/negative exposure each return the *correct* `PbError` variant (never panic/NaN/Inf); valid companions return `Ok(_)` finite; a `PyLoss`-style fallible callback propagates as typed error.
- **Deps:** T0, T3; co-authored with §05 (Phase 4) but the SquaredError arm lands at Phase 2.
- **DoD:** suite green for each landed loss; the domain-error suite is a build-blocking **[GATE]**; "correct-variant-per-fallible-fn" rule (§13.8) satisfied for §05.
- **Size:** M

### T5 — Newton-gain + degenerate-input numerical tests (with §06, Phase 2)
- **§-ref:** §13.5, §06
- **Deliverable:** Hand-built 2-leaf example: summed Newton gain `½Σ[G_L²/(H_L+λ)+G_R²/(H_R+λ)−G²/(H+λ)]` and `w*=−G/(H+λ)` match by-hand constants exactly. Degenerate-input suite: constant target, single-row, single-feature, all-missing column, zero-weight rows, zero exposure → valid model or typed `PbError`, never panic/NaN/Inf (`is_finite()` sweep over all leaves and cells). The boundary test for the canonical missing low-bit and the leaf-index `idx = b0|b1<<1|b2<<2 ∈ 0..8` form-(b) `#[allow]` (§13.8 hot-loop).
- **Deps:** T0, T4 (SquaredError); §06 split-finder.
- **DoD (part of G2):** numerical tests green; the engine recovers a synthetic order-≤3 piecewise-constant target to float tolerance; **the T2 determinism harness now runs on the real `fit` and is byte-equal across `{1,2,8}` threads**; degenerate sweep green.
- **Size:** M

### T6 — Binning property suite (with §03, Phase 1)
- **§-ref:** §13.3 (binning round-trip/monotonicity), §03
- **Deliverable:** Property suite: borders strictly ascending; bin 0 reserved missing; data bins `1..=n_data_bins`, `n_data_bins=borders.len()+1`, `n_bins≤255`; `bin(x)` non-decreasing; all realized values in `0..n_bins`; **only `NaN`→missing**, `±∞`/out-of-range finite → clamp to first/last finite data bin (never missing); no input panics.
- **Deps:** T0, T3; §03 `bin`/`build_grid`.
- **DoD (G1):** suite green; binning byte-reproducible across thread counts (seeded subsample) via the T2 harness; round-trip-to-documented-bin assertion green.
- **Size:** S

### T7 — Purification-identity property suite + the five gates on the real bank (G3, with §08, Phase 3)
- **§-ref:** §13.3 (purification identities — the core suite), §13.1, §08.6
- **Deliverable:** proptest over random small banks: purify **idempotent**, **mass-conserving** (per cell), **linear** (`purify(αA+βB)=α·purify(A)+β·purify(B)`, α+β=1 — the property §09 ensemble/refit/Nesterov rely on), **permutation-invariant**, single-pass-exact under product/uniform `w`. Wire all five checks of T1 over the *real* accumulated `TableBank`: Reconstruction (one interior point per merged-grid cell, exhaustive), MassConservation, Purity (promoted CHECK→**GATE** now §08 lands), VarianceSum (GATE under product/uniform `w`), ThreeWayEqual (tree-sum=table-sum=Faith-Shap to `recon_tol`). Matrix: {SquaredError} × {order 1,2,3} × {ProductMarginals, Uniform} initially, on small synthetic fixtures.
- **Deps:** T1, T3, T5; §08 purify pipeline.
- **DoD (this is G3 — the green spine):** all five checks pass as **build-blocking** assertions on every fitted model in the corpus; purification-identity proptest green; a failing cell flips the model to `Approximate` *and* fails the test (firewall == gate).
- **Size:** L

### T8 — Invariant-matrix expansion across the v1 loss set + early-termination (with Phase 4)
- **§-ref:** §13.1 (matrix), §13.12 (sampled/exhaustive corners)
- **Deliverable:** Extend the T7 matrix to the full v1 set: {SquaredError, Logistic, Poisson, Gamma, Tweedie{1.5}} × {order 1,2,3} × {ProductMarginals, Uniform} × {with/without monotone} × {with/without early-termination trees}. Add the sampled-corner per-commit variant + the **nightly exhaustive** [GATE] (§13.12). VarianceSum→Shapley-sum CHECK downgrade arm under `Joint`.
- **Deps:** T7; §05 loss set (Phase 4).
- **DoD (G4):** five checks green on every link; exposure-offset Poisson fixture anchors base level to 1.000; nightly exhaustive gate configured.
- **Size:** M

### T9 — Monotonicity property suite (with §07, Phase 5)
- **§-ref:** §13.3 (monotonicity), §07, G5
- **Deliverable:** Property: for any fitted model with a `MonotoneMap`, the reconstructed **1-D main-effect table** and the **total score** are monotone in the constrained feature's bin order. Deliberately does *not* assert monotonicity on interaction tensors (documented caveat). A "funnel-is-soft" test: a planted interaction below the FAST threshold is still recoverable. `max_interaction_order=1` ⇒ no realized 2D/3D tables.
- **Deps:** T7, T8; §07.
- **DoD (G5):** monotone holds on total + constrained 1D table (not every 2D/3D cell); funnel-soft test green; G3 invariants still hold.
- **Size:** M

### T10 — Serialization property + golden tests (with §10, Phase 6)
- **§-ref:** §13.3 (serialization), §10, G6
- **Deliverable:** Round-trip proptest over arbitrary fitted models for `serde_json` and `bincode 2.x` (`encode_to_vec`/`decode_from_slice`, frozen `config::standard()`): `model == decode(encode(model))`; `schema_version` survives; scoring the round-tripped model bit-identical; encoded bincode bytes byte-equal across two encodes. **Golden tests**: committed reference JSON + bincode artifacts for a fixed seed/config model; CI asserts the wire form is unchanged (catches accidental format drift). Tree-sum == table-sum runtime echo of ThreeWayEqual. Pickle never exercised.
- **Deps:** T7; §10 serde + scoring.
- **DoD (G6):** round-trip + golden suites green; tree-sum==table-sum; byte-stable round-trip via T2.
- **Size:** M

### T11 — Fuzz targets (nightly [CHECK])
- **§-ref:** §13.6
- **Deliverable:** `cargo-fuzz` targets under `fuzz/`: `fuzz_deserialize` (arbitrary bytes → JSON + bincode `Model` decoders → `Err(PbError::Serialization)` or valid `Model`, never panic/OOM/UB) and `fuzz_binning` (arbitrary `f32` + config → no panic, §13.3 invariants hold). Corpus seeded under `fuzz/corpus/`. Nightly 5-min budget per target.
- **Deps:** T6 (binning), T10 (deserialize).
- **DoD:** both targets build and run clean for the budget; a found panic is a no-panic-policy violation (fixed, not suppressed); nightly job wired.
- **Size:** S

### T12 — Criterion bench harness + baseline (early, [CHECK])
- **§-ref:** §13.7, §11.6, §11.7
- **Deliverable:** `benches/` Criterion benches: histogram build, split search, full-fit (fixed synthetic dataset), branch-free 8-cell inference, table-sum scoring. `--save-baseline`/`--baseline` regression check vs a committed baseline. The bit-reproducibility harness co-lives here per §11. Wired **as early as Phase 2** so perf regressions are visible from the first engine commit.
- **Deps:** T5 (real fit), T0.
- **DoD:** benches compile and run in CI ([CHECK], non-blocking); baseline committed; regression-beyond-threshold reported.
- **Size:** M

### T13 — `wht8` ↔ purification cross-check oracle (with §07/§08, Phase 5)
- **§-ref:** §13.7a, §07, §08.5
- **Deliverable:** `assert_wht8_triple_matches_purified(tree, w, tol)` — proptest over random depth-3 leaf vectors + random positive per-cut `w`-marginals asserting the `wht8` `c_123` equals the per-tree order-3 Faith-Shap from the §08 mass-moving path to a derived `wht8_tol` (NOT bit-equal). All eight coefficients checked against purified terms. **Strictly single-tree** (never summed across trees). Product/Uniform arms **[GATE]**; `Joint` arm **[CHECK]** at looser tol.
- **Deps:** T7; §07 `wht8`, §08 Faith-Shap.
- **DoD:** oracle green as [GATE] on product/uniform; documented [CHECK] on Joint; does NOT promote `wht8` into the audited path (T1 negative-property still green).
- **Size:** M

### T14 — `xtask accuracy` harness + TreeSHAP oracle (dev-only, [CHECK])
- **§-ref:** §13.7, §01 milestone, §14.3
- **Deliverable:** `xtask/` binary (ships no library code): `xtask accuracy` fits tri-boost, records per-objective **deviance/logloss** (strictly-proper §05 metrics, never RMSE on Poisson/Gamma/Tweedie) on TabArena-style fixtures vs EBM and unconstrained incumbents — the "beat EBM, near-parity, all-exact" instrument. TreeSHAP appears here only, as a **test oracle**: stock TreeSHAP attributions compared against exact equal-split `φ_i` from the tables.
- **Deps:** T7, T8.
- **DoD:** `xtask accuracy` runs and emits the per-objective table; TreeSHAP-vs-`φ_i` agreement [CHECK]; not a public API, no library code.
- **Size:** M

### T15 — Python conformance + coverage gates (with §12, Phase 7)
- **§-ref:** §13.10, §13.8 (variant mapping), G7
- **Deliverable:** `pytest` suite over the §12 sklearn contract (`fit→self`, `predict`, `get_params`/`set_params`, `classes_`, `n_features_in_`, `NotFittedError`, Pipeline/`cross_val_score` smoke), `PbError`→exception mapping, `.pyi`/`py.typed` presence, determinism smoke (same seed → same predictions through the binding), **cross-FFI byte-equality** (Python-fit vs Rust-fit model identical). Wheel-smoke [GATE]: install each maturin-built wheel in a clean env, run fit/predict/serialize before any release. `cargo llvm-cov` job: ≥90% line (≥95% math modules); a *drop* on changed files is [GATE].
- **Deps:** T2, T10; §12 binding.
- **DoD (G7):** Python CI green; wheel-smoke green; cross-FFI byte-equality green; coverage gates wired.
- **Size:** M

### T16 — v1.5 gate hardening: re-pass discipline, quantized determinism, table-budget stress
- **§-ref:** §14.3 release gate, §13.4 (quantized path), §13.5 (quantization bound), §08.10
- **Deliverable:** Per-lever re-pass harness: each v1.5 lever (quantized hist, Fisher-TS, multi-cat axes, MVS, ensemble) re-runs the five G3 checks before merge. T2 determinism gate now runs on the **quantized** path. Quantization-error-bound unit test (round-trip `< 0.5·g_scale`; refit-from-full-precision closes the log-link bias). **Table-budget stress benchmark**: border-rich axes (near 254 bins) × deep bagging drive merged-grid supports to worst case; assert `max_table_cells`/total-bank budget respected (admission penalty + sparse fallback or `PbError::TableBudget`); five checks still pass on sparse-stored hot triples; purify time + peak memory bounded. cat×cat×cat budget check (counts as 3 raw; combination-CTR rejected at construction).
- **Deps:** T7, T8, T2; §04/§06/§08/§09 levers.
- **DoD (v1.5 release gate):** every lever re-passes G3; quantized `{1,2,8}`-thread byte-equality green; table-budget stress green (no silent memory/wall-clock inflation); TabArena accuracy yardstick (T14) shows beat-EBM/near-parity, every model `Exact`.
- **Size:** L

---

**Ordering rule restated for milestone areas:** a feature's PR is incomplete until the named gate from this area is green — binning ships with T6, the engine with T5+T2, purification with T7 (G3), losses with T4+T8, interactions with T9+T13, serde with T10, Python with T15. Trunk stays green because the five I2 checks, the I1 budget check, and the determinism harness run over the *entire* model corpus on every PR.
