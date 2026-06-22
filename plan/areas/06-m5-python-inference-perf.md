# Milestone 5 — Python API, Inference & Performance Hardening — Implementation Plan

## How this area's structure makes quality automatic

This milestone is the *consumption* layer: it never constructs trees or tables, it only marshals into the green core and surfaces it. Quality is therefore enforced by **delegation plus equality**. Three structural facts do the work:

1. **Every new code path is a re-encoding of something already gated, so its DoD is a *bit-equality* test against the canonical path, not a fresh correctness argument.** `ScoringBank`/`PackedTree` (§10.2a), the row-blocked kernel (§11.9), `predict_binned`, and the flat path-B arena are all "load-derived views" — they ship *only* with a proptest asserting byte-identity to `score_trees`/`score_row` at tolerance 0 (§10.7). You physically cannot land the fast path without the equality gate green, so "fast" can never silently mean "different."
2. **The determinism `[GATE]` (§13.4) is the spine that runs through every task.** It already exists from the foundation milestone (the `model_is_bit_reproducible_across_thread_counts` harness, §11.8). This milestone *extends its subject set* — serialized bytes, then SIMD-dispatched scoring, then the Python FFI — but never re-derives it. The byte-equality assertion is the same `bincode::serde::encode_to_vec(.., standard())` comparison everywhere.
3. **The firewall and `PbError`→exception funnel make illegal states unrepresentable across the FFI.** No panic crosses the boundary because the core is no-panic (§1) and the single `map_err` funnel (§12.7) turns every `PbError` variant into a typed Python exception. The binding adds *zero* math, so there is nothing in this layer that can break I1/I2 — they are upheld "purely by delegation" (§12 intro).

Sequencing rule honored throughout: serialization (the determinism gate's wire subject) lands before the packed scoring view (which must prove it leaves that wire form untouched); both land before the Python binding (which delegates to them); SIMD/streaming hardening rides last because it is a *constant-factor view* over already-gated kernels. Tasks reference the foundation milestone's gate machinery (clippy `-D warnings` incl. the no-panic set, fmt, `cargo deny`, MSRV 1.64, doctests, `deny(missing_docs)`, the determinism harness, the proptest invariant suite) rather than re-specifying it. These map to the v1 Phase 6/7 exit gates **G6/G7** (§14.5) plus the v1.5 perf-hardening row.

---

## Tasks

### M5-T1 — Serde model wire format (`ModelDoc`, JSON + bincode 2.x)
- **§-ref:** §10.5, §1 (frozen `bincode::config::standard()`), §13.3, §13.4.
- **Deliverable:** `core/src/serialize.rs`: `ModelDoc { format_version: u32, schema_version: u32, model: Model }` (plain nested, **no** `serde(flatten)`); `const CURRENT_FORMAT_VERSION`; `const BINCODE_CFG = bincode::config::standard()`; `Model::{to_json, from_json, to_bincode, from_bincode}` exactly per §10.5 (`serde_json::to_string_pretty`; `bincode::serde::encode_to_vec`/`decode_from_slice`). `from_*` **re-validates on load**: I1 per tree, `grids.len()==provenance.len()`, every `Split.axis < grids.len()`, finite `f0`/leaves, `mode` consistency. CI grep gate: no `usize` in any `#[derive(Serialize)]` index field; no `HashMap` in wire types.
- **Dependencies:** foundation milestone (canonical `Model`/`ObliviousTree`/`Split`/`PbError`/`ExactnessMode` frozen; the five invariant gates exist).
- **DoD (gates):** `cargo test` serialization proptest (§13.3): `model == from_json(to_json(m))` and `from_bincode(to_bincode(m))` over the fitted-model corpus, structural + **bit-identical predictions**; encoded bincode bytes byte-equal across two encodes (frozen config). Cross-version bincode load → `Err(PbError::Serialization)`. The `usize`/`HashMap` grep `[GATE]`. clippy `-D warnings`, fmt, doctests, `deny(missing_docs)` on every new public fn.
- **Size:** M

### M5-T2 — `schema_version` round-trip + forward-only migration
- **§-ref:** §10.5 (evolution rules), §2.6 (`ModelSchema`), §10.6 readability sourcing.
- **Deliverable:** `migrate(doc: serde_json::Value, from: u32) -> Result<Model, PbError>` (forward-only, `From`-based); `#[serde(default=..)]`/`#[serde(alias=..)]` evolution discipline; newer-than-build `format_version` → typed `Serialization` error; schema round-trip validation (`schema.feature_kinds.len()==grids.len()`, every `CategoricalTS{encoding}` resolves to a present `cat_encoders` entry).
- **Dependencies:** M5-T1.
- **DoD (gates):** migration test (§10.7): a fixture corpus of every prior `format_version` JSON loads, migrates, re-validates I1/I2. Schema round-trip test (§12.8 R-SCHEMA mirror at core level): `feature_names`/`class_labels`/`cat_encoders`/`objective` survive `from_json(to_json)` and `from_bincode(to_bincode)`. clippy/fmt/doctests green.
- **Size:** S

### M5-T3 — `RatingExport` rating-table artifact + firewall + reference-level re-basing
- **§-ref:** §10.6, §3 (firewall), §2.7 (`TableBank`/`RefMeasure`).
- **Deliverable:** `RatingExport`, `RatingTable`, `AxisExport`, `RatingBasis { reference: BTreeMap<FeatureSet, Vec<usize>> }`; `TableBank::to_rating_export(link, mode, schema, basis) -> Result<RatingExport, PbError>`. PURE zero-mean form (default); optional rating-view re-basing folding per-axis shift into `f0` (exact basis change, `F(x)` unchanged). Sobol-descending sort; per-cell `support` carried (display-only). Readability resolved through `Model.schema` (names, cat level labels via frozen `cat_encoders`, `objective`/`class_labels`).
- **Dependencies:** M5-T1 (serde infra, `ModelSchema` round-trip), foundation milestone's `TableBank` + the five invariant checks.
- **DoD (gates):** firewall test (§10.7): `to_rating_export` on an `Approximate` model → `Err(PbError::ExactnessFirewall)`. PURE-export zero-mean test: each table `w`-weighted mean within `purity_tol` of zero; asserts it does **not** claim a 1.000 reference row. Re-basing exactness test: `to_rating_export(.., Some(basis))` reconstructs `F(x)` bit-identical to PURE for every row (shift folded into `f0` cancels); each re-centered table reads `1.000` at its `reference`. Round-tripped relativities re-exponentiate to `predict` scores (both forms). clippy/fmt/doctests green.
- **Size:** M

### M5-T4 — Packed path-A scoring view: `ScoringBank` / `PackedTree`
- **§-ref:** §10.2a, §11.9, §2.5 (canonical missing low-bit, R-MISSING).
- **Deliverable:** `core/src/scoring.rs`: `#[repr(C)] PackedTree { feat:[u8;3], thresh:[u8;3], miss:u8, leaf:[f32;8] }` padded to 64 B; `ScoringBank(Vec<PackedTree>)` in stored tree order; built in `finalize`/`deserialize` as a **byte-exact** re-encoding of the same f32 leaves and the same per-level low bit `if bin==0 { (miss>>level)&1!=0 } else { bin<=thresh[level] }`. **Never serialized.** u8-axis cap at ≤255 features with the side-table fallback (parallel `Vec<u32>` axis keying + generic `tree_lookup`), selection recorded once at build. Scalar one-row kernel first (streaming SIMD deferred to M5-T8).
- **Dependencies:** M5-T1 (the canonical `score_trees`/`tree_lookup` it must equal; `finalize`/`deserialize` hooks).
- **DoD (gates):** packed-equivalence proptest (§10.7): `ScoringBank` scoring **bit-identical** (tolerance 0) to `score_trees` walking the `ObliviousTree`s over the model corpus, at `n_threads ∈ {1,2,8}`; missing-bin rows under **both** `missing_left` settings on each axis match (R-MISSING); a >255-feature model exercises the side-table fallback with identical results. Determinism `[GATE]` unchanged: the serialized `ModelDoc` bytes are byte-equal across thread counts (proving `ScoringBank` did not touch the wire form). No-panic hot-loop boundary test (§10.7): `idx==7` and `Split.axis==grids.len()-1` pin the scoped `#[allow(indexing_slicing)]` `// JUSTIFIED:` proof; the `// JUSTIFIED:` grep `[GATE]`. clippy/fmt green.
- **Size:** M

### M5-T5 — `predict_binned` + flat single-arena path-B kernel (digitize-once)
- **§-ref:** §10.2 (digitize-once note), §10.3 (flattened arena note), §10.4.
- **Deliverable:** `Model::predict_binned(&BinnedMatrix, offset)` (no re-bin); flattened path-B kernel digitizing each merged axis **once per row** into a flat `Vec<f32>` tensor arena at compile-known strides, accumulating in fixed table order (load-derived runtime view, never serialized).
- **Dependencies:** M5-T1, foundation `TableBank::score_row`/`score`.
- **DoD (gates):** folded into the §10.7 path-equality `[GATE]`: `predict_binned` and the arena kernel produce **bit-identical** raw scores to `predict`/`score_row` on the same bins; the existing `|score_trees_row − table_bank.score_row| < recon_tol` (`4·n_trees·EPSILON`, §13.1) Reconstruction proptest still green. clippy/fmt/doctests green.
- **Size:** S

### M5-T6 — PyO3 binding skeleton: `tri-boost-py` crate, maturin/abi3, `_tri_boost` module
- **§-ref:** §12.1, §2 (workspace), §11.5.
- **Deliverable:** `crates/tri-boost-py` (`crate-type=["cdylib"]`, `pyo3` features `extension-module`+`abi3-py310`, `numpy`); `#[pymodule] fn _tri_boost`; module-local `#![allow(unsafe_code)]` (the single justified exception for pyo3 macro expansion); `pyproject.toml` (`build-backend="maturin"`, separated layout) + `python/tri_boost/` package skeleton with `py.typed`. Custom exception classes `TriBoostError`(base)/`InvariantError`/`ExactnessError`/`SerializationError`/`InternalError`; the single `map_err(PbError)->PyErr` funnel (§12.7 table). Empty `_Booster`/`_Model`/`_TableBank` shells. **Core carries zero pyo3 dependency** (CI assert).
- **Dependencies:** M5-T1 (serde, since `_Model` will expose it next).
- **DoD (gates):** wheel builds on manylinux2014 via maturin-action (abi3-py310, one wheel per os/arch) — the §13.10 wheel-build `[GATE]`; a clean-env install + import smoke test. `cargo deny`/MSRV/clippy/fmt green on the new crate; `--no-default-features` and default-features both build. Unsafe-audit `[GATE]`: the only `unsafe` is the confined pyo3 glue (Miri-exempt, justified). CI grep confirms no pyo3 in `tri-boost-core`.
- **Size:** M

### M5-T7 — `PyBooster.fit` / `PyModel.predict` with GIL marshal-before-detach + numpy zero-copy
- **§-ref:** §12.2, §12.3, §11.5, §10.2 (scoring), R-PYDETACH, R-PYWEIGHTS, R-SCHEMA, R-EARLYSTOP, R-DISTILL.
- **Deliverable:** `PyBooster { inner: Booster, cfg: Config }` + `PyModel(Arc<Model>)`. `fit` signature per §12.2: marshal **every** input into Rust-owned buffers *while GIL held* (`ServeBinnedMatrix` owned, `.to_vec()` each 1-D `y`/`weight`/`exposure`/`teacher_raw`), build `FitSpec`, then `py.detach(move || pool.install(|| inner.fit_with(..)))` capturing **only `Send`** data; scoped `ThreadPoolBuilder(n_jobs)` (§11.5). Stamp `feature_names`/`class_labels` into `Model.schema`. `predict`/`predict_proba` bin into owned `ServeBinnedMatrix` under GIL, detach, score via §10 path A / `ScoringBank`, return `into_pyarray`; optional `out=` (`PyReadwriteArray1`). `PyLoss` adapter (callable objective) applying `weight[i]` to `(g,h)` (R-PYWEIGHTS); `CallbackHost`/`TrainObserver` re-acquiring GIL only at fire point; built-in `early_stopping` callback gated by `validation_fraction` (default `None` = off). Serde passthrough: `to_json`/`from_json`/`to_bytes`/`from_bytes`/`tables`/getters. `.pyi` stubs.
- **Dependencies:** M5-T6, M5-T1, M5-T4 (scoring view).
- **DoD (gates):** Python pytest `[GATE]` (§12.8): zero-copy assertions (`predict` output shares no buffer with input; `as_array` pointer identity on F-contiguous; native f64 input → `TypeError`); GIL/parallelism test (two concurrent `fit`s overlap; `n_jobs` bounds the pool); **R-PYWEIGHTS** test (a Python `squared_error` callable reproduces native loss bit-for-bit *with non-uniform sample_weight*); callback stop-truncation test; **R-EARLYSTOP** (`validation_fraction=None` fits all `n_estimators`; `Some(frac)` → deterministic seeded holdout, same `(seed,frac)` ⇒ same `best_iteration`, tables built from that prefix); **R-DISTILL** (`teacher_raw=None ≡ blend=1.0` reproduces non-distilled fit bit-for-bit; threaded onto `FitSpec.distill`, stays `Exact`). No panic crosses FFI (Internal-funnel test). clippy/fmt; `mypy --strict` + stubtest on the typed package.
- **Size:** L

### M5-T8 — sklearn estimators + cross-FFI determinism + Python conformance
- **§-ref:** §12.6, §12.7, §12.8, §13.4 (Python mirror), §13.10 (Python CI).
- **Deliverable:** pure-Python `python/tri_boost/sklearn.py`: `TriBoostRegressor(RegressorMixin, BaseEstimator)` and `TriBoostClassifier` wrapping `_Booster`. Defaults **mirror core `Config`** (single source of truth, wrapper forwards never re-decides); `__init__` stores each arg verbatim (powers `get_params`/`set_params`/`clone`); `_validate` does f64→f32 with one-time `PrecisionWarning`, sets `n_features_in_`/`feature_names_in_`; `check_is_fitted`→`NotFittedError`; name-keyed `monotone_constraints` resolved into the §07 `MonotoneMap`; classifier `classes_`/`predict_proba` (column order from `schema.class_labels`)/`decision_function`/`__sklearn_tags__`; `set_params` clears fitted state; `distill=` convenience helper (data-side CatBoost teacher, behind `distill` feature).
- **Dependencies:** M5-T7.
- **DoD (gates):** Python CI `[GATE]` (§13.10/§12.8): `clone`/`get_params`/`set_params` round-trip; `Pipeline`+`GridSearchCV`+`cross_val_score`+stacking smoke; `NotFittedError` on unfitted predict; `predict_proba` column order == `classes_`; a targeted `@parametrize_with_checks` subset (practical contract). **Determinism `[GATE]` mirror:** `fit` at `n_jobs ∈ {1,2,8}` ⇒ byte-equal `to_bytes()` (Python reflection of §13.4). **Schema round-trip (R-SCHEMA):** `from_bytes(to_bytes(m))`/`from_json(to_json(m))` recover `feature_names`+`class_labels`; reloaded classifier `predict_proba` column order matches original `classes_`. f64-input emits exactly one `PrecisionWarning`. `.pyi`/`py.typed` presence + `mypy --strict` + stubtest. The clean-env wheel fit/predict/serialize smoke `[GATE]` before any Trusted-Publishing release.
- **Size:** L

### M5-T9 — SIMD hardening: multiversion dense kernels, in-register permute leaf-select, row-blocked streaming
- **§-ref:** §11.4, §11.9, §10.2 (leaf-select decision), §13.4 (ISA byte-equality).
- **Deliverable:** `#[multiversion(targets("x86_64+avx512f","x86_64+avx2+fma","x86_64+sse2","aarch64+neon"))]` on the dense kernels owned here: SIMD-across-rows `score_trees`, the `exp(k·F)` inverse-link, `f32→u8` binning border-search, and the **in-register permute** leaf-select (`vpermps`/`vpermi2ps`/`vqtbl`) over the register-resident 8-entry leaf LUT — **never** a hardware gather — through the safe `multiversion`/`pulp`/`wide` wrappers (no raw `unsafe`, `forbid(unsafe_code)` holds). `score_tile(bank, tile: RowTile<W>, out)` row-blocked streaming kernel: hold `W≈8–32` rows resident, stream `ScoringBank` once, accumulate `W` lanes per tree in **fixed stored tree order**, `_mm_prefetch` next line. Realizes the `Backend::predict_block` seam.
- **Dependencies:** M5-T4 (the `ScoringBank` it streams), M5-T7 (so the binding immediately benefits).
- **DoD (gates):** tiled-vs-scalar **bit-equality** proptest (§10.7/§11.8): `score_tile` byte-identical to `score_trees`/`tree_lookup` over the corpus at tolerance 0; in-register permute matches scalar leaf read across all 8 index patterns and across **each** multiversion target. **ISA-byte-equality `[GATE]` (§13.4):** the determinism harness run with SIMD dispatch forced low (SSE2) vs high (AVX-512) asserts byte-equal serialized model *and* byte-equal predictions — dispatched ISA must not perturb the bits. No-panic/unsafe-audit gates (zero steady-state `unsafe`). clippy/fmt green.
- **Size:** M

### M5-T10 — Criterion bench baselines + table-size budget stress (perf regression visibility)
- **§-ref:** §11.8, §13.7, §14.3 (v1.5 table-budget stress gate), §10.4.
- **Deliverable:** `benches/` Criterion suite for the inference/perf kernels owned here: branch-free 8-cell `score_trees`, `ScoringBank` row-blocked throughput (rows/s), table-sum `score_row`, the leaf-select permute, and serialize/deserialize — each `--save-baseline`/`--baseline` against a committed baseline (dev-only, never shipped in the wheel). Wire the v1.5 adversarial table-size benchmark fixtures (border-rich axes near the 254-bin cap × deep bagged union grids) that assert per-table/total-bank cell budgets and the path-A `predict` envelope on worst-case grids.
- **Dependencies:** M5-T4, M5-T5, M5-T9 (the kernels being measured).
- **DoD (gates):** Criterion `[CHECK]` (§13.7): no regression beyond threshold vs the committed baseline, reported in CI (non-blocking, runner-variance-tolerant); the inference bench confirms path-B `score_row` cost is `O(|tables|)`, **independent of n_trees** (§10.3). Table-budget stress: the five invariant checks still pass on the worst-case grids; no budget breach silently inflates memory/wall-clock (the v1.5 release-gate component this milestone owns). clippy/fmt on bench code.
- **Size:** M

---

## Sequencing rationale (nothing builds on an unverified foundation)

`T1→T2→T3` establish the **wire contract first** — the determinism gate's subject — so every later "view" can prove it leaves the bytes untouched. `T4` (packed scoring) and `T5` (digitize-once) are gated *bit-equal* against the canonical scorer from `T1`, never landing as unverified modules. `T6` brings up the binding shell with the wheel/exception/`cargo deny` gates green before any Python feature code (gates-before-features at the FFI layer). `T7→T8` deliver the native + sklearn surface, each gated by the Python conformance + cross-FFI determinism mirror of the core gate. `T9` (SIMD/streaming) and `T10` (benches) ride last because they are constant-factor hardening over already-gated kernels — the ISA-byte-equality gate guarantees speed never costs reproducibility, and Criterion baselines make any regression visible from this milestone forward. Foundation/through-line gate machinery (clippy `-D warnings` + no-panic lints, fmt, `cargo deny`, MSRV 1.64, doctests, `deny(missing_docs)`, the `{1,2,8}`-thread determinism harness, the five invariant proptests) is inherited, not re-specified; every task's DoD is *gates-green*, not *code-written*.
