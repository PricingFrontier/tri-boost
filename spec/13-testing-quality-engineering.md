## 13 — Testing, quality & engineering standards

This section is the build-blocking realization of the §1 checklist and the §3 invariant contract. It introduces no new public types; it specifies the test suites, CI gates, lints, and audits that make every other section's claims *checkable* rather than asserted. Governing principle, from the brainstorm: **"if the tables ever disagree with the ensemble, there is no product"** — so the I2 lossless-equivalence invariants are enforced as `cargo test` gates, not aspirations. Every gate is marked **[GATE]** (build-blocking — failure blocks merge) or **[CHECK]** (reported, non-blocking, promotable per release).

The test tree mirrors the §02 workspace: unit tests live `#[cfg(test)]`-inline beside their math; integration/property suites in `crates/tri-boost-core/tests/`; the determinism and invariant harnesses in `tests/invariants/`; Criterion benches in `benches/`; Python conformance in `tests/` (pytest). The accuracy harness is a dev-only `xtask/` binary that ships no library code.

### 13.1 The lossless-invariant gates (I2 made build-blocking)

The five `Invariant` checks (§3; §2.8 `Invariant` enum; implementation owned by §08) are realized as one deterministic fixture run over a matrix of fitted models — the load-bearing test of the project.

```rust
/// Runs all five I2 checks against a fitted Model + its accumulated TableBank.
/// Returns `Err(PbError::InvariantViolated { invariant })` naming the first
/// violated `Invariant`; `Ok(())` means the model is provably exactly
/// decomposable on its merged grid. Tolerances are absolute, in score space.
/// (Signature standardized with §08: all five checks return `Result<(), PbError>`,
/// carrying the broken `Invariant` in the `InvariantViolated` variant.)
pub(crate) fn assert_exact_decomposition(
    model: &Model,
    bank: &TableBank,
    grid_corners: &CellCornerIter, // one interior point per merged-grid cell
) -> Result<(), PbError>;

/// Float tolerances for the f32 core. recon_tol is a *derived* accumulation
/// bound, not a magic floor: each tree contributes at most one leaf add plus the
/// purify round-off, so the worst-case accumulated score-space rounding over
/// n_trees trees is `4.0 * n_trees * f32::EPSILON` (4 = leaf-add + 3-axis purify
/// fold). No additive `1e-4` floor — the reconstruction checks assert equality
/// "to a derived float tolerance", NOT bit-equality. True bit-equality (tolerance
/// 0) is reserved for the serialized-model determinism gate (§13.4), which the
/// fixed-`CHUNK_ROWS` float-fold discipline (§11) makes achievable.
pub(crate) struct ExactTol { pub recon_tol: f64, pub mass_tol: f64, pub purity_tol: f64, pub var_tol: f64 }
impl ExactTol { pub fn for_model(m: &Model) -> Self; } // recon_tol = 4.0 * n_trees * f32::EPSILON as f64
```

The five checks are each a separate `#[test]` so a failure names the broken property:

1. **Reconstruction [GATE]** — `max over one interior point per merged-grid cell of |raw(x) − (f0 + Σ_u f_u(x_u))| < recon_tol`. The merged grid is piecewise-constant, so one corner per cell is *exhaustive*, not sampled (research/03 §5). `CellCornerIter` is the cartesian product of realized borders across each table's ≤3 axes; for high-cell models the per-commit gate samples corners under the test `seed` and the exhaustive variant runs nightly (§13.12).
2. **MassConservation [GATE]** — total signed mass `Σ_u Σ_cells w(cell)·f_u(cell)` is identical before and after purification, compared as `f64` with `mass_tol = 0.0` where `w` is exactly representable, else `< 1e-9`.
3. **Purity [CHECK→GATE]** — every axis-slice of every order-≥1 `EffectTable` has `w`-weighted mean within `purity_tol` of zero (fANOVA zero-mean, research/03 Eq 6a). Promoted to **[GATE]** once §08 lands.
4. **VarianceSum [GATE under product/uniform `w`]** — `σ²(raw) = Σ_u σ²(f_u)` within `var_tol`. This *branches on `w`*: it holds and is gated under `RefMeasure::ProductMarginals`/`Uniform`; under `RefMeasure::Joint` it is replaced by the Shapley-effect sum and downgraded to **[CHECK]** (the axiom fails under hierarchical orthogonality — research/03 §1.5).
5. **ThreeWayEqual [GATE]** — tree-sum, table-sum and Shapley-sum agree to the derived `recon_tol` (a reconstruction check, so *not* bit-equal — bit-equality is the serialized-model determinism gate's job, §13.4) at each test point: `raw(x)`, `f0 + Σ_u f_u(x_u)`, and `f0 + Σ_i φ_i(x)` (equal-split `φ_i = Σ_{u∋i} f_u(x_u)/|u|`, §08). The strongest end-to-end gate — it catches an error in accumulation, purification, *or* attribution simultaneously.

**Matrix.** `assert_exact_decomposition` runs over {`SquaredError`, `Logistic`, `Poisson`, `Gamma`, `Tweedie{1.5}`} × {`max_interaction_order` 1, 2, 3} × {`ProductMarginals`, `Uniform`} × {with/without monotone constraints} × {with/without early-termination trees}, on small synthetic fixtures (n≤2000, ≤8 features) where exhaustive corner-iteration is cheap. A failing cell flips the model to `ExactnessMode::Approximate` at runtime *and* fails the test — the firewall (§3) and the gate are one wall seen from two sides.

**The `wht8` screening accumulator is NOT a gate input.** The §07 running per-support per-order variance, fed by the frozen O(8) `wht8` per-tree transform (§07/§06 leaf estimation), is a **soft prior** and is **excluded from all five `Invariant` checks above** (ThreeWayEqual / VarianceSum / Purity / Reconstruction / MassConservation). Those five run *only* on the merged-grid purified bank. The reason is load-bearing: `wht8` coefficients live on each tree's OWN 2-point grid under that tree's `w`-marginals, and trees cut different borders, so coefficients CANNOT be summed across trees (that drops cross-tree covariance) — the `wht8`-derived per-order variance is therefore a SCREENING SIGNAL, never the audited ensemble Sobol, and under `RefMeasure::Joint` the clean product form degrades to a heuristic. Because it never hard-gates, it is exactness- and determinism-neutral by construction. The fixture asserts (a **[GATE]** negative property) that no element of the `wht8` accumulator is ever read by `assert_exact_decomposition`, foreclosing the rejected "`wht8` replaces §08 Lengerich purification" shortcut.

These gates uphold **I2** and serve aim 2 (decomposable). I1 is a construction-time invariant, gated next.

### 13.2 The I1 (feature-budget) gate

I1 — depth-3 oblivious, ≤3 distinct *raw* features — is enforced at `ObliviousTree` construction (§06) and re-verified post-fit:

```rust
/// Property over every tree in a fitted Model: depth ∈ 1..=3, splits share one
/// (axis, bin_le) per level, and DISTINCT provenance raw-feature count == depth.
/// Returns `Err(PbError::InvariantViolated { invariant: FeatureBudget })` on
/// breach (signature standardized with §08 / the other four checks).
pub(crate) fn check_feature_budget(model: &Model) -> Result<(), PbError>;
```

**[GATE]** a `proptest` generates random `FitSpec`s (loss, order, monotone map, seed) over generated `BinnedMatrix`es and asserts `check_feature_budget` for every fitted model. The provenance dereference (`provenance[s.axis].raw`) is what binds the budget to raw features, not encoded axes — the §04 multi-categorical-axis path is exercised here to prove a `cat×cat×cat` tree counts as 3, and that a forbidden combination-CTR axis (which would pack >3 raw features into one axis) is rejected at construction with `PbError::InvariantViolated { FeatureBudget }`.

### 13.3 Property-based testing (proptest)

`proptest` (the chosen framework; `quickcheck` is not used) covers the algebraic identities where a single example is unconvincing. Each property uses a fixed `ProptestConfig { cases: 256, max_shrink_iters: 4096, .. }` seeded from a frozen RNG so failures reproduce. Suites:

- **Binning round-trip & monotonicity (§03):** for any `Vec<f32>` column and `max_bin` (default **254**), borders are strictly ascending, bin 0 is the reserved missing bin, the data bins are `1..=n_data_bins` with `n_data_bins = borders.len() + 1` and `n_bins = n_data_bins + 1 ≤ 255` (fits `u8`, values `0..=254`), `bin(x)` is non-decreasing in `x`, and every realized value maps into `0..n_bins`. The missing-vs-extremes rule is asserted **exactly as §03 specifies**: **only `NaN` maps to the missing bin 0**; out-of-range / extreme **finite** values (including `±∞`, which the binner treats as the extreme finite ends) **clamp to the first/last finite data bin** (bin 1 / bin `n_data_bins`), never to missing. None of these inputs panic.
- **Purification identities (§08):** the core suite. For random raw effect tensors on a random merged grid and random positive `w`, purify is (a) **idempotent** (`purify(purify(T)) == purify(T)` to `purity_tol`); (b) **mass-conserving** (`Σ purify(T) == Σ T` per cell); (c) **linear** (`purify(αA + βB) == α·purify(A) + β·purify(B)`, `α+β=1` — Lengerich Cor 2.2, the property the §09 ensemble/refit/Nesterov paths rely on); (d) **permutation-invariant** in slice order. Together these *are* the "purify-then-sum ≡ sum-then-purify" contract that licenses streaming accumulation.
- **Monotonicity (§07):** for any fitted model with a `MonotoneMap`, the reconstructed **1-D main-effect table** and the **total score** are monotone in the constrained feature's bin order. The property deliberately does *not* assert monotonicity on interaction tensors — encoding the documented caveat that individual purified interaction cells need not be monotone, and guarding against a future over-strict test.
- **Loss correctness (§05):** for each `Loss`, the finite-difference of `deviance` w.r.t. `raw` matches the `out.g` written by `grad_hess` to `1e-3` relative, and the analytic `out.h` hessian matches the second finite difference; `init_score == link(weighted mean)` to `1e-6`; `pred_from_raw` is the exact inverse link. Poisson/Gamma/Tweedie use `exp(k·F)`, asserted against a `powf` reference within `f32` tolerance. **All three of `grad_hess`, `init_score`, and `deviance` are fallible** — the canonical §05/§2.4 signatures are `fn grad_hess(..) -> Result<(), PbError>`, `fn init_score(..) -> Result<f64, PbError>`, and `fn deviance(..) -> Result<f32, PbError>` — so every assertion in this suite `?`-propagates the `Result` (the no-panic gate, §1: a bad domain surfaces a typed `PbError`, never a `panic`/`NaN`).
- **Loss domain errors (§05) [GATE]:** a typed-error unit test pins that `init_score` and `deviance` (now fallible — §2.4 / §05: `init_score(..) -> Result<f64, PbError>`, `deviance(..) -> Result<f32, PbError>`) return the *correct* `PbError` variant on each **invalid domain**, never a panic, `NaN`, or `Inf`, matching the §13.8 "correct-variant per fallible public fn" rule: Gamma/Tweedie with any `y <= 0`; all-zero (or all-non-positive) `weight`; the all-zero-positives Logistic case (no positive label mass under `w`); and a bad/zero `exposure` (a non-positive or zero `e` whose `offset = log(e)` is undefined). Each case asserts `Err(PbError::InvalidInput { .. })` (the domain-validity variant; a length mismatch instead yields `Err(PbError::ShapeMismatch { .. })`), and a `PyLoss`-style fallible objective is exercised to confirm its callback failure propagates as a typed `PbError` rather than `.expect(..)`. The valid-domain companions (a positive-`y` Gamma, mixed-label Logistic, positive-`exposure` Poisson) assert `Ok(_)` with a finite, floored result — so the gate distinguishes a real domain error from a spurious one.
- **Serialization (§10):** round-trip over arbitrary fitted models for both `serde_json` and **bincode 2.x** — the latter via `bincode::serde::encode_to_vec(&model, bincode::config::standard())` / `bincode::serde::decode_from_slice(&bytes, bincode::config::standard())` (the frozen `standard()` config; the removed top-level `bincode::serialize`/`deserialize` are not used). `model == decode(encode(model))`; `schema_version` survives; scoring the round-tripped model is **bit-identical** to the original. A determinism sub-check asserts the encoded `bincode` bytes are themselves byte-equal across two encodes of the same model (the config is frozen, so the wire form is stable). Pickle is never exercised because it is never produced.

### 13.4 Determinism & bit-reproducibility (first-class)

The §1 determinism gate is the second load-bearing test after I2.

**[GATE] thread-invariance.** The same `(BinnedMatrix, y, FitSpec, seed)` is trained at `n_threads ∈ {1, 2, 8}` (forcing a rayon pool of each size via the §11 scoped `ThreadPoolBuilder`) and the three serialized `Model`s are asserted **byte-equal** — the `bincode::serde::encode_to_vec(.., bincode::config::standard())` bytes compared with `==` (tolerance 0; this is the *true* bit-equality claim, distinct from the §13.1 derived-tolerance reconstruction checks). This proves the **i64-accumulator** quantized-integer histogram accumulation (the §06-owned `Hist`, `Vec<i64>` bins / `u32` counts; §2.3 `QuantGradHess`) is order-independent and that no float reduction leaked into `reduce`/`sum` work-steal order — the float-fold path uses the fixed `CHUNK_ROWS` constant (§11), never `rayon::current_num_threads()`. It also exercises the deterministic per-work-unit re-seeding (`Pcg64::seed_from_u64(splitmix64_mix(base, round, stage, block))` — a frozen `splitmix64` mix, *not* a "splittable" PRNG), which must yield position-stable draws independent of pool size. A companion **[GATE]** asserts predictions are bit-equal across thread counts on a held-out matrix.

**[GATE] cross-run reproducibility.** Two independent process invocations with identical inputs produce byte-equal models — guarding against ambient nondeterminism. Provenance/grid construction must use a deterministic map or sorted keys; a grep gate forbids `std::collections::HashMap` in serialized-state construction paths and in any config map the iteration of which can reach fit order, steering to `BTreeMap`/sorted `Vec`. Concretely it requires `MonotoneMap = BTreeMap<String, MonoSign>` (§07) — never a `HashMap` — and asserts the §07 `AdmissionPrior` is consumed only via order-independent pure lookups; these maps are config-only and never serialized, so they cannot perturb the wire form.

**[CHECK] platform reproducibility.** Byte-equality *across* architectures (x86-64 vs aarch64) is **not** guaranteed — f32 `exp` differs, and the full-precision leaf-refit path (§06) means cross-arch models may differ in the last ULP. This is a documented non-goal; the gate is intra-platform only. Serialized *index* fields are nonetheless fixed-width (`Split.axis: u32`, `BinnedMatrix.n_rows: u32`, never platform-dependent `usize`), so the wire form is width-stable even where the core smoke-builds on `wasm32`; a CI grep **[GATE]** forbids `usize` in `#[derive(Serialize)]` index fields of serialized state. **Open fork, recommended default off in v1:** offer a `deterministic_cross_platform` feature routing `exp`/`ln` through a vendored bit-reproducible polynomial; revisit if a user needs cross-arch byte-identity for filing.

### 13.5 Numerical tests

Beyond the property suites, targeted unit tests pin the math against closed form:

- **Newton split-gain (§06):** on a hand-built 2-leaf example with known `G, H`, the summed gain `½Σ[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)]` and the exact leaf weight `w* = −G/(H+λ)` match by-hand constants exactly.
- **Quantization error bound:** `QuantGradHess` round-trip error (`dequantize(quantize(g)) − g`) is bounded by `0.5·g_scale` under stochastic rounding, with *expected* error zero over a fixed seed budget; leaves refit from full-precision `g/h` differ from quantized-derived leaves by less than the quantization step — proving the mandatory refit closes the log-link-regression bias the NeurIPS-2022 paper flagged.
- **Degenerate inputs:** constant target, single-row, single-feature, all-missing column, zero-weight rows, and zero `exposure` each produce a valid model or a typed `PbError` — never a panic, NaN, or Inf in any leaf or table cell (an `is_finite()` sweep over all leaves and cells).

### 13.6 Fuzzing

`cargo-fuzz` (libFuzzer) targets the two untrusted-input surfaces, run nightly **[CHECK]** with a 5-minute budget per target and corpus under `fuzz/corpus/`: `fuzz_deserialize` (arbitrary bytes into the `Model` JSON deserializer and the `bincode::serde::decode_from_slice(.., bincode::config::standard())` path — must return `Err(PbError::Serialization(..))` or a structurally-valid `Model`, never panic, OOM, or UB; the highest-value target since deployed models load from disk) and `fuzz_binning` (arbitrary `f32` slices + config into the binner — no panic, §13.3 invariants hold). Since `#![forbid(unsafe_code)]` holds (§13.8), this targets *panic/typed-error discipline and resource bounds*, not memory safety; a found panic is a no-panic-policy violation, fixed not allowed.

### 13.7 Benchmark & accuracy harnesses

**Criterion benches [CHECK]** (`benches/`): histogram build, split search, full-fit on a fixed synthetic dataset, branch-free 8-cell inference, and table-sum scoring. Each asserts *no regression beyond a threshold* against a committed baseline (`criterion --save-baseline`/`--baseline`); reported in CI but non-blocking (wall-clock varies by runner). Serves aim 3 (fast) by making regressions visible. *(Internal yardstick only; no benchmarking tooling ships — AIM non-goals, §11.)*

**Accuracy harness [CHECK], dev-only.** An `xtask accuracy` binary fits tri-boost and records per-objective **deviance/logloss** (the strictly-proper §05 metrics, never RMSE on Poisson/Gamma/Tweedie) on TabArena-style fixtures versus EBM and the unconstrained incumbents — the **"beat EBM, near-parity, all-exact" milestone** instrument (§01; AIM "how we will know it worked"). It is *not* shipped, *not* a public API, and produces no library code. TreeSHAP appears here and only here, as a **test oracle**: stock TreeSHAP attributions are compared against the exact equal-split `φ_i` from the tables — agreement validates §08's exactness claim, and TreeSHAP stays demoted to oracle (never a shipped explainer, per the brainstorm).

### 13.7a The `wht8` ↔ purification cross-check oracle (two routes to one exact quantity)

A correctness/invariant test, **not a performance claim**: for a **single** depth-3 tree, the order-3 (triple) coefficient delivered by the frozen O(8) `wht8` transform must agree, **to a derived float tolerance**, with that tree's per-tree order-3 Faith-Shap / interaction value computed via the §08 mass-moving (Lengerich) purification path. The two routes are independent — `wht8` reads the 8 register-resident leaf values directly and applies the frozen 8×8 transform under that tree's per-cut `w`-marginals (1 constant + 3 main `c_i` + 3 pairwise `c_ij` + 1 triple `c_123`); the §08 route accumulates the same tree's leaves onto its purified grid and runs the mass-moving purify (3→2→1→intercept), then reads the order-3 Faith-Shap as an O(1) table read (§08.5). Agreement of `c_123` with the purified triple is a true cross-check: a single bug touching only one of the two implementations breaks the test.

```rust
/// Oracle [GATE]: for a single ObliviousTree, the `wht8` order-3 coefficient
/// `c_123` equals the per-tree order-3 Faith-Shap from the §08 mass-moving
/// purification path, to a derived float tolerance (NOT bit-equality — these are
/// two independent float reductions). proptest over random depth-3 leaf vectors
/// and random positive per-cut `w`-marginals.
pub(crate) fn assert_wht8_triple_matches_purified(
    tree: &ObliviousTree,
    w: &RefMeasure,
    tol: f64, // `wht8_tol`: a derived accumulation bound, sized like ExactTol (§13.1), not a magic floor
) -> Result<(), PbError>;
```

**Scope — strictly single-tree.** The oracle is asserted **per tree only**. The critical caveat applies in full: `wht8` coefficients live on each tree's OWN 2-point grid under that tree's `w`-marginals; trees cut different borders, so coefficients are NEVER summed across trees in this oracle (that would drop cross-tree covariance — the `wht8` accumulator is a screening signal, not the audited ensemble Sobol). The agreement is exact-in-the-limit only on a **single** tree's own grid, where the two grids coincide. The proptest matrix covers `RefMeasure::ProductMarginals`/`Uniform` as **[GATE]**; under `RefMeasure::Joint` the clean product form degrades to a heuristic, so that arm is **[CHECK]** (asserted to a looser tolerance, non-blocking) and documented as such — matching §07's treatment of the prior. The free byproduct (all eight `wht8` coefficients, not just the triple, can be checked against the corresponding purified main/pairwise/constant terms) is asserted in the same fixture.

This is a sibling to the §13.7 TreeSHAP oracle: like TreeSHAP, `wht8` here is a **test oracle only**, exercising a second independent route to a quantity §08 already computes. It validates that the `wht8` math (the §07 screening primitive) is itself correct, *separately* from the firewall fact that its accumulator never participates in the five §08 gates (§13.1). Crucially, this oracle does NOT promote `wht8` into the audited path — the rejected "`wht8` replaces §08 Lengerich purification" idea stays rejected; §08's accumulate→single-pass-purify remains the sole gated decomposition.

### 13.8 Error-handling, no-panic & unsafe enforcement

**[GATE] lints & overflow policy.** At every crate root: `#![deny(missing_docs)]`, `#![deny(warnings)]` in CI, and the panic-policy clippy set as `deny` — `unwrap_used`, `expect_used`, `panic`, `unreachable`, `indexing_slicing`, `cast_possible_truncation` (where lossy). `cargo clippy --all-targets --all-features -- -D warnings` and `cargo fmt --check` are gates. `unwrap`/`expect` are permitted only in `#[cfg(test)]`/`benches`/`xtask`, or on a value proven non-`None`/in-bounds in the immediately-preceding line with a `// JUSTIFIED:` comment — a CI grep **[GATE]** asserts every non-test `unwrap`/`expect` carries such a comment.

Integer-overflow is caught by **`overflow-checks = true` in *all* Cargo profiles at the workspace root** (dev *and* release — the load-bearing decision, since a wrapped i64 histogram bin would silently corrupt a model), not by a crate-blanket clippy lint. Cargo ignores member-local profile settings in a workspace, so the root placement is part of the gate. `clippy::arithmetic_side_effects` is **scoped, not crate-root-blanket** — applied as a module/fn-level `#![deny]` only where an unchecked integer `+=`/`*` would be a real risk (the histogram accumulators, index math); it is **explicitly NOT applied to float arithmetic** (every `g/h`/score/purify add would otherwise be a lint error, which is the §02.3 self-contradiction this resolves). A `// JUSTIFIED:` comment plus the `overflow-checks` net cover the rest.

**Hot-loop no-panic policy.** The canonical hot paths (`tree_lookup`'s `row[split.axis]` / `tree.leaves[idx]`, the `fit` accumulation `+=`, `bin`'s `(k+1) as u8`) must NOT use raw indexing or unchecked arithmetic that could panic under the gates above. Two permitted forms, both gated: (a) the panic-free form `slice.get(i).ok_or(PbError::Internal { what })?` (or `.get(..)` for ranges); or (b) a `#[allow(clippy::indexing_slicing)]`-scoped `fn` carrying a `// JUSTIFIED:` bounds proof (e.g. `idx = b0 | b1<<1 | b2<<2 ∈ 0..8`, so `leaves[idx]` is in-bounds) *and* a dedicated **boundary test** that exercises the proven extreme indices. The accumulation `+=` is on `i64` bins under `overflow-checks`, with the `n_rows·max|g_q| < i64::MAX` bound proven in §06. A CI grep **[GATE]** asserts every form-(b) `#[allow(clippy::indexing_slicing)]` in non-test code is paired with a `// JUSTIFIED:` comment.

**[GATE] unsafe audit.** `#![forbid(unsafe_code)]` at the `tri-boost-core` root. The audit gate greps the diff for `unsafe` and `unsafe_ok`; any hit requires a reviewer-acknowledged justification, a `// SAFETY:` proof per precondition, a dedicated test, and a **Miri** run over that test (**[GATE]** for the unsafe module). SIMD goes through `multiversion`/`pulp`/`wide` safe wrappers (§11), so steady-state unsafe is *zero* and the gate keeps it that way. The only place `unsafe` may appear is the `tri-boost-py` pyo3 boundary, confined to generated/`#[pymethods]` glue — Miri-exempt (Miri can't run pyo3) but covered by the §13.10 Python suite.

**[GATE] no `Box<dyn Error>`** in any public signature (grep gate); every fallible public fn returns `Result<T, PbError>` (§2.8). Each section maps its failures onto a variant, and a unit test per fallible public fn asserts the *correct variant* for a representative bad input (dtype mismatch → `DtypeMismatch`, >3 raw features → `InvariantViolated { FeatureBudget }`, exact-export after a warp → `ExactnessFirewall`).

### 13.9 Dependency, MSRV & supply-chain gates

- **[GATE] `cargo deny check`** — licenses (allow-list MIT/Apache-2.0/BSD/Unicode-3.0), advisories (`RUSTSEC` deny), bans (no duplicate major versions of core deps, no `openssl`), sources (crates.io only). New deps need a justification in the owning section (§1). The only standing advisory exception is `RUSTSEC-2025-0141` for frozen bincode 2.x: unmaintained-only, explicitly justified, and revisited if a concrete vulnerability lands.
- **[GATE] MSRV 1.74** — a CI job builds and tests on the pinned `1.74` toolchain; raising MSRV requires a changelog entry and matrix bump. The floor is set by `[workspace.lints]` inheritance while remaining compatible with the manylinux2014/glibc-2.17 target (§11).
- **[CHECK] `cargo-semver-checks`** on the core crate before publish, catching accidental breaking changes to the §2 canonical types.

### 13.10 CI matrix, coverage & wheels

**Rust CI [GATE]:** `{stable, MSRV 1.74} × {linux-x86_64, linux-aarch64, macos-aarch64, windows-x64}` running `fmt --check`, `clippy -D warnings`, `test --all-features`, doctests, the determinism gate (§13.4), the invariant gates (§13.1–2), and `cargo deny`. Both default-features and `--no-default-features` must pass (the §1 flag matrix `arrow`/`nightly` each build; `nightly`/`portable_simd` is **[CHECK]** on the nightly toolchain only).

**Python CI [GATE]:** `pytest` over the §12 practical sklearn contract (`fit→self`, `predict`, `get_params`/`set_params`, `classes_`, `n_features_in_`, `NotFittedError`, Pipeline/`cross_val_score` smoke), `PbError`→exception mapping, `.pyi`/`py.typed` presence, and a determinism smoke test (same seed → same predictions through the binding). Wheels build with **maturin-action** (abi3-py310 → one wheel per os/arch) plus an sdist; a **[GATE]** installs each wheel in a clean env and runs a fit/predict/serialize smoke test before any PyPI Trusted-Publishing (OIDC) release.

**Coverage [CHECK]:** `cargo llvm-cov` over the core crate, **target ≥ 90% line coverage**, with the math modules (loss, split-finder, purification, serialization) held to **≥ 95%** since they carry the exactness and reproducibility guarantees. A *drop* below threshold on changed files is **[GATE]**; absolute coverage is **[CHECK]** (avoiding blocks on hard-to-cover error arms). Docs coverage is total — `#![deny(missing_docs)]` makes 100% public-item doc coverage a compile gate, and `--doc` tests every `///` example.

### 13.11 How this section serves the three aims & the invariants

- **Aim 2 (decomposable) / I1, I2:** §13.1–2 turn the invariant contract into build-blocking tests — the firewall and the gate are one wall. The ThreeWayEqual gate, if green, certifies the whole "tables ARE the model" thesis.
- **Aim 1 (predictive):** §13.3 loss-correctness, §13.5 Newton-gain, and §13.7's accuracy harness keep the predictiveness machinery honest against closed form and against EBM/incumbents.
- **Aim 3 (fast):** §13.4's determinism gate validates the quantized-histogram mechanism that is *simultaneously* the speed and reproducibility lever; §13.7's Criterion guards regression.

### 13.12 Open fork & recommended default

One genuine fork: **exhaustive vs sampled grid-corner reconstruction (§13.1) for large models.** Exhaustive corner-iteration is `O(∏|Ω_axis|)` per table and blows up on dense triple grids. **Recommended default:** gate on *exhaustive* corners for the small synthetic matrix (cheap, and a true losslessness proof) and on *seeded-sampled* corners plus all training rows for larger fixtures in the per-commit gate, relegating the exhaustive sweep over all fixtures to a **nightly [GATE]**. This keeps per-commit CI fast while never giving up a provable losslessness check on at least one model per build.
