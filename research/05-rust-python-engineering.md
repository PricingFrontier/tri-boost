# Engineering a High-Performance Rust Core + Python-Binding GBM Library — Research Report

Research date: 2026-06-21. Scope: engineering a pure-Rust GBM core with Python (PyO3) bindings, competitive with XGBoost/LightGBM/CatBoost, shipped as wheels. Claims verified against primary sources (PyO3/maturin docs, reference repo Cargo.toml/pyproject.toml files, rust-numpy/rayon docs, SIMD tracking issues, GPU crate pages, existing Rust GBM repos).

---

## 1. Project / Workspace Layout

Reference projects converge on: **a pure-Rust core crate (no Python deps) + a thin PyO3 binding crate (`crate-type=["cdylib"]`) + a separate Python source dir + a maturin `pyproject.toml`.** They differ only in how far they push the separation.

- **pydantic-core** — *single-crate* variant. One crate whose `[lib] name="_pydantic_core"`, `crate-type=["cdylib","rlib"]`; Python wrapper re-exports the compiled `_pydantic_core` submodule. **Does not use abi3** — ships one wheel per Python version for per-version performance.
- **huggingface/tokenizers** — the textbook *(a) core / (b) binding / (c) py-source / (d) maturin* split. Pure-Rust core at `tokenizers/` (published to crates.io standalone); PyO3 binding at `bindings/python/` with `[lib] name="tokenizers" crate-type=["cdylib"]`, a `py_src/` dir, path dependency on the core. **Uses abi3** (`abi3-py310`) → one wheel per platform across CPython ≥3.10.
- **polars** — *large cargo workspace*. ~40 pure-Rust crates under `crates/` + one binding crate `crates/polars-python` (`pyo3` with `abi3-py310`); the Python package + build orchestration live in `py-polars/`. pyo3 version/features are **pinned at the workspace root** and each crate opts in.

**`pyproject.toml` essentials:** `[build-system] requires=["maturin>=1.9,<2"]`, `build-backend="maturin"`, and `[tool.maturin]` keys: `python-source` (importable package dir), `module-name` (`pkg._pkg` private-submodule convention), `manifest-path` (point at the binding crate in a workspace — **essential**), `features` (e.g. `["pyo3/extension-module"]`), `strip`, `compatibility`, `include` (ship `py.typed`/`.pyi`).

**Recommended tree (the maturin "separated" layout):**

```
pattern-boost/
├── Cargo.toml                       # [workspace]; pins pyo3 + shared version at root
├── pyproject.toml                   # build-backend="maturin"; python-source, module-name, manifest-path
├── rust-toolchain.toml              # pin stable toolchain (MSRV >= 1.64 for manylinux)
├── python/
│   └── pattern_boost/
│       ├── __init__.py              # re-export from ._pattern_boost; sklearn wrappers live here
│       ├── _pattern_boost.pyi       # type stubs
│       ├── py.typed
│       └── sklearn.py               # PatternBoostClassifier/Regressor (thin wrappers)
├── crates/
│   ├── pattern-boost-core/          # (a) PURE Rust: model structs, serde, train/predict, histogram
│   │   ├── Cargo.toml               #     NO pyo3 dep; crates.io-publishable; Backend trait lives here
│   │   └── src/{lib,tree,hist,loss,bin,predict,serialize,backend}.rs
│   └── pattern-boost-py/            # (b) thin pyo3 binding, crate-type=["cdylib"]
│       ├── Cargo.toml               #     pattern-boost-core = { path=..., version=... } + pyo3
│       └── src/lib.rs               #     #[pymodule] _pattern_boost
├── tests/                           # python tests (pytest)
└── benches/                         # criterion benches against the core
```

`[tool.maturin]`: `python-source="python"`, `module-name="pattern_boost._pattern_boost"`, `manifest-path="crates/pattern-boost-py/Cargo.toml"`, `features=["pyo3/extension-module"]`.

---

## 2. PyO3 + maturin

**Exposing Rust to Python.** `#[pyfunction]` for free functions; `#[pyclass]` + `#[pymethods]` (with `#[new]`, `#[pyo3(get,set)]`, `#[getter]`/`#[setter]`); `#[pymodule]` for the module. The module fn name **must equal the last segment of `module-name`** (`_pattern_boost`).

**abi3 / stable ABI.** `pyo3 = { features=["abi3-py310"] }` → wheel `cp310-abi3-…` runs on CPython 3.10 **and every later 3.x** → one wheel per platform. polars and tokenizers both do this. **Limitations:** some features gated (class `text_signature` ≥3.10; buffer API ≥3.11; native subclassing ≥3.12); marginally slower than version-specific builds (why pydantic-core opts out). abi3 does **not** cover free-threaded CPython — PEP 803's `abi3t` is separate.

**Dev workflow.** `maturin develop --release` (always `--release` for numeric perf — debug Rust numerics are misleadingly slow). `maturin build --release --strip` for distributables.

**GIL gotcha (the key one).** Release the GIL around heavy compute so other Python threads and your rayon pool run truly in parallel: `py.detach(|| heavy_pure_rust())` (renamed from `py.allow_threads` in PyO3 0.26; alias kept). **You must not move any Python object into the closure** — marshal numpy inputs to Rust types (`ArrayView`) *before* detaching, compute, then re-acquire to build outputs. Note: **PyO3 0.21** introduced the `Bound<'py,T>` API (replacing GIL-Refs); **PyO3 0.26** renamed `with_gil→attach`, `allow_threads→detach`. Pin pyo3 once at the workspace root; keep `extension-module` a feature-driven flag (off for `cargo test`, on for wheels).

---

## 3. NumPy Interop

**Input (zero-copy).** Take `PyReadonlyArray2<'py, f32>` (features) and `PyReadonlyArray1<'py, f32>` (targets/weights); `.as_array()` → `ndarray::ArrayView2<f32>`. `.as_array()` is **always zero-copy regardless of contiguity**. `.as_slice()` is zero-copy **only if contiguous**. `PyReadonlyArray` is a runtime-checked borrow and is **not `Send`/`Sync`** — derive the `ArrayView` *before* `py.detach`.

**Copies happen only on:** (1) dtype mismatch (`PyReadonlyArray2<f32>` *rejects* a float64 array rather than silently casting); (2) demanding `as_slice()` on a non-contiguous array; (3) non-native byte order. For ergonomic f64→f32 acceptance use `PyArrayLike2<'py, f32, AllowTypeChange>` (calls `numpy.asarray`, may copy). Specializing the core to **f32** is the standard memory/cache choice.

**Output.** `vec.into_pyarray(py)` / `from_owned_array` = **zero-copy** ownership transfer (preferred for predictions). `to_pyarray(py)` / `from_array` from a borrow = **copy**. For caller-supplied `out=` buffers, take `PyReadwriteArray1` + `as_array_mut()`.

**Layout for histograms.** Column access (feature-parallel histogram building) is cache-friendly on **Fortran/column-major** `X`; either require/transpose to F-order once at ingest, or pre-bin into an internal column-major `u8`/`u16` store.

**Arrow / polars input (optional path).** Use **`pyo3-arrow`** + the **Arrow PyCapsule protocol** for **zero-copy** ingestion of PyArrow / polars / pandas-arrow data. Arrow primitive buffers are contiguous → view columns as `&[f32]`; the **validity bitmap maps naturally to GBM missing-value handling**. Recommendation: **numpy is the primary path; add the Arrow/dataframe path optionally** behind the same internal column store.

---

## 4. Parallelism (rayon)

**Primitives.** `par_iter`/`par_chunks(_mut)` for **data-parallel**; `(0..n_features).into_par_iter()` for **feature-parallel** histogram building.

**GIL release is mandatory.** Wrap *all* rayon compute in `py.detach(|| …)`. Without releasing the GIL, rayon workers can't run Python and your extension serializes all other Python threads — and can **deadlock**. Derive `ArrayView`s while the GIL is held; the closure must be `Send` and Python-free.

**Thread-pool control.** Don't hijack the user's global pool. Build a **scoped pool per call** and `install()` it inside `detach`:
```rust
let n = if n_jobs <= 0 { num_cpus::get() } else { n_jobs as usize };
let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build()?;
let model = py.detach(|| pool.install(|| train_gbm(x, y)));   // nested rayon ops use THIS pool
```
Expose `n_threads`/`n_jobs` (map `-1`/`None`→all cores, sklearn-style).

**Pitfalls.** (1) **Oversubscription** — `GridSearchCV(n_jobs=8)` over a GBM that itself spawns 8 threads = 64 threads on 8 cores; honor `n_threads`, document `threadpoolctl`/`OMP_NUM_THREADS`. (2) **False sharing + correctness in histogram accumulation** — never share one histogram via `Mutex` (lock contention is rayon's #1 perf killer). Use **per-thread private histograms via `fold(|| zeros(), accumulate).reduce(|| zeros(), merge)`**; pad/align to a cache line. (3) **Non-deterministic fp reductions** — rayon's `reduce`/`sum` combine in work-steal order, so non-associative fp addition gives run-to-run bit differences; for reproducibility, `fold` over **fixed-size `par_chunks`** and combine partials in index order. Integer/quantized histogram sums reproduce exactly — another reason to quantize.

---

## 5. SIMD

**The realistic options (2026):**
- **`std::simd` / `portable_simd`** — still **nightly-only, no stabilization date** (tracking issue rust-lang/rust#86656). **Do not put it in a stable-shipping core.**
- **Stable autovectorization** — **reliable for integers, weak for floats** (Rust honors IEEE-754, no safe `-ffast-math`). Encourage with `chunks_exact(N)`, separate in/out slices, `#[inline]`. Brittle → verify with `cargo-show-asm`.
- **Explicit intrinsics via `std::arch`** — stable; **much safer since Rust 1.87** (`#[target_feature(enable="avx2")]` fns call intrinsics without inner `unsafe`). Runtime dispatch via `is_x86_feature_detected!`.
- **Helper crates:** **`multiversion`** (per-CPU variants + runtime dispatch; lowest friction), **`pulp`** (safe SIMD abstraction, proven in `faer`), **`wide`** (fixed-width portable types).

**For GBM histogram building specifically:** the scatter-add `hist[bin[i]] += g[i]` is **hard to SIMD-vectorize** (data-dependent gather/scatter, same-bin write conflicts, memory-bound). **XGBoost/LightGBM/Tangram do NOT hand-SIMD it** — they win with cache-friendly layout, `_mm_prefetch`, **quantized integer gradients** (autovectorize + halve memory traffic), histogram subtraction, and threads. SIMD **does** help on the dense/regular kernels: gradient/hessian computation, **prediction (leaf lookup)**, loss functions, binning.

**Recommendation:** cache-friendly column-major binned layout + per-thread histograms via **rayon** + **quantized integer** accumulation + **runtime-dispatched intrinsics (`multiversion`/`pulp`/`wide`) for the few dense kernels** + `_mm_prefetch` on the scatter. Build wheels with a **portable baseline** (`-C target-cpu=x86-64-v3` for AVX2), lift to AVX-512 at runtime — **not** `target-cpu=native`. Keep nightly `portable_simd` out of the shipping core (optionally behind a `nightly` feature).

---

## 6. Build & Distribution

**maturin-action** runs any `maturin` subcommand with built-in cross-compilation; for Linux it builds **inside a manylinux/musllinux Docker container** automatically. Key inputs: `command`, `args`, `target`, `manylinux`/`compatibility`, `container`, `before-script-linux`, `rust-toolchain`, `sccache`. Scaffold with `maturin generate-ci github`.

**CI matrix** (tags trigger → build per OS×arch → sdist → OIDC release): linux x86_64/aarch64/armv7 + musllinux_1_2; macOS x86_64 + aarch64; Windows x64. With **abi3** you build **one wheel per (os,arch)** instead of per-interpreter. Free-threaded 3.13t/3.14t needs a separate `abi3t` invocation. Release via **PyPI Trusted Publishing / OIDC**.

**cibuildwheel vs maturin-action.** For a *pure* Rust+PyO3 package, prefer **maturin-action** (compiles Rust once per target, esp. with abi3; cross-compiles cleanly). Use **cibuildwheel** for *mixed* C/C++/Rust packages.

**manylinux ↔ Rust.** **Rust ≥1.64 requires glibc ≥2.17**, so **manylinux2014 (glibc 2.17) is the oldest targetable tag**; aarch64 effectively needs **manylinux_2_28**. Keep **MSRV ≥1.64**, pin the toolchain in CI.

**Keep the core crates.io-publishable.** PyO3 lives **only** in `pattern-boost-py`; `pattern-boost-core` has **zero pyo3 dep**. crates.io rejects path-only deps → use both `path` + `version`. **Publish order:** `cargo publish -p pattern-boost-core`, wait for indexing, then the py crate. Share the version via `[workspace.package]`.

---

## 7. Model Serialization

**Formats:** **bincode** (compact/fast binary, *not* self-describing, brittle for long-term storage); **serde_json** (human-readable, debuggable, self-describing, cross-language); **rmp-serde/MessagePack** (compact self-describing middle ground).

**Versioning/evolution:** embed a `format_version` field; use `#[serde(default)]` (add fields safely), `#[serde(alias="old")]` (rename safely), enum `#[serde(tag="v")]` + `From`-based migrations for big jumps; **avoid `deny_unknown_fields`** on long-lived structs.

**Cross-language load — the architectural win:** because the model struct + its `#[derive(Serialize,Deserialize)]` live in the **pure-Rust core crate**, bytes serialized after training in Python deserialize **identically** in any pure-Rust consumer. One serde impl, no format drift. XGBoost/LightGBM validate this: a **documented stable JSON/UBJSON format** for archival/cross-version IO, with explicit warnings that **pickle/memory-snapshots are unstable**.

**Recommendation:** a **versioned serde model in the core crate**; **default the on-disk format to stable JSON** (the public contract, à la XGBoost), with MessagePack/UBJSON as a compact option and bincode only as an opt-in same-version cache. **Never pickle** the native model.

---

## 8. Python API Conventions

**sklearn estimator contract (the checklist):** `__init__` = explicit keyword args only, **no logic**, store each arg unchanged as `self.<same_name>` (this 1:1 mapping is *why* `get_params`/`set_params`/`clone` work). `fit(X,y)` validates, sets `n_features_in_` (and `feature_names_in_` for DataFrames), **returns `self`**. **Fitted attributes get a trailing underscore** (`classes_`, `trees_`); an unfitted estimator raises `NotFittedError` on predict. Inherit `BaseEstimator` + `ClassifierMixin`/`RegressorMixin`. Classifiers store labels in `classes_`; `predict_proba` columns ordered to match. Override `__sklearn_tags__` (sklearn ≥1.6). Test with `check_estimator` / `@parametrize_with_checks`.

**Dual API (like XGBoost/LightGBM):** native `Booster`-style API (full control: custom objectives, callbacks, early stopping) **plus** thin sklearn wrappers. Value: instant interoperability with `Pipeline`, `GridSearchCV`, stacking.

**How much is worth it for v1:** target the **practical contract, not full `check_estimator`** — `__init__`(mirrored), `fit→self`, `predict`, `score`(mixin), `get_params`/`set_params`, `classes_`, `n_features_in_`, `NotFittedError`. That alone makes it work in Pipeline/GridSearchCV/cross_val_score (95% of the value).

---

## 9. GPU

**Survey:** **cudarc** — mature, production-grade *binding/runtime* (safe CUDA driver API, cuBLAS/cuDNN/NCCL/NVRTC), NVIDIA-only, but you author kernels in CUDA C / NVRTC. **Rust-CUDA/cust** — *revived in 2025* (nightly, CUDA 12.x) but **not production-ready**. **NVIDIA cuda-oxide** — brand-new (May 2026), experimental rustc→PTX backend, too immature. **wgpu** — cross-vendor/portable but **weakest for training perf**. **CubeCL** — the most credible *write-GPU-kernels-in-Rust* option (one `#[cube]` fn → CUDA/ROCm/Metal/Vulkan/WebGPU/CPU-SIMD, autotuning), **alpha but proven in Burn**. **candle** — inference tensor lib, not a GBM building block. **burn** — full DL framework; its **swappable-`Backend`-trait design is the architecture to imitate**.

**Why it's hard:** the histogram/split-finding kernels are exactly the hard part and none of the Rust tools hand them to you. The speedup is **conditional** — ~2x dense, 5–15x only on tens-to-hundreds of millions of rows — and **host↔device transfer overhead actively hurts the common small/medium tabular case** (LightGBM's own docs: GPU is "inefficient" on small data).

**Recommendation: DEFER GPU in v1.** Fast multi-threaded CPU `hist` is the actual competitive baseline. **Keep the door open:** put compute behind a **`Backend` trait** (histogram build, split eval, grad/hess accumulation, predict) modeled on Burn, ship `CpuBackend` only, keep the data layout **already GPU-friendly** (pre-binned `u8`/`u16` columns, `max_bin ≤ 255`, SoA grad/hess, contiguous) — which is *also* the fast-CPU layout. If/when added: **cudarc** (NVIDIA) or **CubeCL** (portable).

---

## 10. State of the Rust ML Ecosystem + Existing GBM

- **linfa** (~4.7k★, maintained) — sklearn-like toolkit, but `linfa-trees` is **single trees only — no GBM, no GPU**. Reuse value: API conventions.
- **smartcore** — RF/Extra Trees; a `smartcore::xgboost` module exists but is **partial (regression only, in-development)**.
- **perpetual** (perpetual-ml, ~696★, actively maintained, Apache-2.0) — **the strongest existing pure-Rust GBM**: Rust core, zero-copy Polars/Arrow, Python+R bindings, a **budget-based hyperparameter-free** philosophy, claims up to 100x vs LightGBM at equal accuracy. **CPU-only.** The incumbent to benchmark against and differentiate from on API/philosophy.
- **forust** (jinlow, ~94★, Apache-2.0) — **XGBoost-algorithm-compatible** pure-Rust GBDT + Python bindings, monotonic constraints, JSON serialization. The **best codebase to learn XGBoost internals from in Rust**.
- **gbdt-rs** (mesalock, ~225★) — **dormant** research artifact; single-threaded, no histogram method. Learn-from only.
- **Tangram** — **dead as an ML project**, but its blog **"Writing the fastest GBDT library in Rust"** remains excellent performance-engineering reading.
- **Bindings** (not pure Rust): `rust-xgboost`, `xgb`, `lightgbm3-rs` — useful only as benchmark baselines.

**Verdict:** a **clean, fast, well-documented, idiomatic pure-Rust GBM with first-class Polars/Arrow, modern histogram method, monotonic constraints, sklearn-compatible API, and a clean backend abstraction is not a saturated niche.** Perpetual is the incumbent (idiosyncratic); forust + Tangram's blog are the best learning references. **The niche pattern-boost targets is real.**

---

## Design Implications for pattern-boost

**Repo layout.** Cargo workspace, maturin "separated" layout: `crates/pattern-boost-core` (pure Rust, **no pyo3**, crates.io-publishable) + `crates/pattern-boost-py` (`cdylib`, thin pyo3 glue) + `python/pattern_boost/` (re-exports compiled `_pattern_boost` + sklearn wrappers + `.pyi`/`py.typed`). pyo3 pinned at the workspace root.

**Binding strategy.** PyO3 with **abi3-py310** (one wheel per platform; follow polars/tokenizers). NumPy is the primary I/O: `PyReadonlyArray2<f32>` + `.as_array()` in, `into_pyarray` out (zero-copy); offer `PyArrayLike2<f32, AllowTypeChange>` for ergonomic f64 acceptance. Add an **optional zero-copy Arrow/polars path via pyo3-arrow**. **Release the GIL with `py.detach`** around every compute path.

**Parallelism + SIMD plan.** rayon with a **per-call scoped `ThreadPoolBuilder`** (`pool.install` inside `py.detach`); expose `n_threads`/`n_jobs`. Histograms: **per-thread private buffers via `fold`/`reduce`** (no `Mutex`), cache-line padded, on a **column-major pre-binned `u8`/`u16` store**. Adopt **quantized integer gradients** (autovectorizes, reproducible, halves memory traffic). SIMD: rely on **stable autovectorization + cache layout + `_mm_prefetch`** for the histogram path; **runtime-dispatched intrinsics via `multiversion`** for prediction/loss/gradient kernels. **No nightly `portable_simd` in the shipping core.** Offer a deterministic mode.

**Build/CI plan.** **maturin-action**, tag-triggered matrix; **abi3 → one wheel per (os,arch)**; sdist job; **OIDC Trusted Publishing**. Target **manylinux2014** (x86_64) / **manylinux_2_28** (aarch64); **MSRV ≥1.64**. Build wheels with a **portable AVX2 baseline (`x86-64-v3`)**, lift to AVX-512 at runtime. Publish `pattern-boost-core` to crates.io independently.

**Serialization choice.** **Versioned serde model in the core crate** (`format_version`, `#[serde(default)]`/`alias`, enum-tagged migrations). **Default on-disk format = stable JSON**, with MessagePack as a compact option. **No pickle.**

**Python API.** Native `Booster`-style API in Rust + thin **sklearn wrappers** (`PatternBoostClassifier`/`Regressor`). Implement the **practical sklearn contract** so it works in Pipeline/GridSearchCV/cross_val_score; defer full `check_estimator`.

**GPU recommendation.** **Defer to post-v1.** Ship a fast CPU `hist` GBM; keep the door open via a **`Backend` trait** and a GPU-ready pre-binned columnar layout.

---

## Sources

**Layout / PyO3 / maturin**
- [Maturin User Guide — Project Layout](https://www.maturin.rs/project_layout.html) · [Configuration](https://www.maturin.rs/config.html) · [Bindings](https://www.maturin.rs/bindings) · [Distribution](https://www.maturin.rs/distribution.html)
- [PyO3 guide — module](https://pyo3.rs/main/module.html) · [classes](https://pyo3.rs/main/class.html) · [parallelism](https://pyo3.rs/main/parallelism.html) · [building & distribution](https://pyo3.rs/main/building-and-distribution.html) · [migration](https://pyo3.rs/main/migration.html)
- [pydantic-core Cargo.toml](https://raw.githubusercontent.com/pydantic/pydantic-core/main/Cargo.toml) · [polars-python Cargo.toml](https://raw.githubusercontent.com/pola-rs/polars/main/crates/polars-python/Cargo.toml) · [tokenizers python Cargo.toml](https://raw.githubusercontent.com/huggingface/tokenizers/main/bindings/python/Cargo.toml) · [PEP 803 (abi3t)](https://peps.python.org/pep-0803/)

**NumPy / Arrow / rayon**
- [PyO3/rust-numpy](https://github.com/PyO3/rust-numpy) · [numpy crate](https://docs.rs/numpy/latest/numpy/) · [PyReadonlyArray](https://docs.rs/numpy/latest/numpy/borrow/struct.PyReadonlyArray.html) · [PyArrayLike](https://docs.rs/numpy/latest/numpy/struct.PyArrayLike.html)
- [PyO3 parallelism guide](https://github.com/PyO3/pyo3/blob/main/guide/src/parallelism.md) · [rayon ThreadPoolBuilder](https://docs.rs/rayon/latest/rayon/struct.ThreadPoolBuilder.html) · [rayon FAQ](https://github.com/rayon-rs/rayon/blob/main/FAQ.md) · [rayon #94 fold/reduce](https://github.com/rayon-rs/rayon/issues/94)
- [Optimization adventures: parallel Rust with/without Rayon](https://gendignoux.com/blog/2024/11/18/rust-rayon-optimized.html) · [pyo3-arrow](https://docs.rs/pyo3-arrow/latest/pyo3_arrow/) · [arro3](https://github.com/kylebarron/arro3) · [sklearn Parallelism](https://scikit-learn.org/stable/computing/parallelism.html)

**SIMD**
- [The state of SIMD in Rust in 2025](https://shnatsel.medium.com/the-state-of-simd-in-rust-in-2025-32c263e5f53d) · [Safe SIMD in Rust (2026)](https://shnatsel.medium.com/safe-simd-in-rust-even-on-the-inside-c6f1ff381828)
- [std::simd](https://doc.rust-lang.org/std/simd/index.html) · [Tracking issue #86656](https://github.com/rust-lang/rust/issues/86656) · [portable-simd #364](https://github.com/rust-lang/portable-simd/issues/364) · [Rust 1.87 blog](https://blog.rust-lang.org/2025/05/15/Rust-1.87.0/)
- [SIMD in stable Rust (wide vs pulp)](https://pythonspeed.com/articles/simd-stable-rust/) · [pulp](https://docs.rs/pulp/latest/pulp/) · [multiversion](https://docs.rs/multiversion/latest/multiversion/) · [target-cpu=native regression #139370](https://github.com/rust-lang/rust/issues/139370)
- [Quantized Training of GBDTs (NeurIPS 2022)](https://arxiv.org/pdf/2207.09682) · [LightGBM Features](https://lightgbm.readthedocs.io/en/latest/Features.html)

**Build / serialization / sklearn**
- [maturin-action](https://github.com/PyO3/maturin-action/blob/main/README.md) · [maturin publish best practices](https://github.com/PyO3/maturin/discussions/1309) · [cibuildwheel](https://cibuildwheel.pypa.io/en/stable/) · [pypa/manylinux](https://github.com/pypa/manylinux) · [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
- [Cargo Workspaces](https://doc.rust-lang.org/book/ch14-03-cargo-workspaces.html) · [bincode v2](https://docs.rs/bincode/2.0.1/bincode/) · [Serde field attributes](https://serde.rs/field-attrs.html) · [XGBoost Model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html)
- [Developing scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html) · [BaseEstimator](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html)

**GPU / Rust ML ecosystem**
- [cudarc](https://docs.rs/cudarc/latest/cudarc/) · [Rust-CUDA update 2025](https://rust-gpu.github.io/blog/2025/08/11/rust-cuda-update/) · [CubeCL](https://github.com/tracel-ai/cubecl) · [candle](https://github.com/huggingface/candle) · [Burn](https://burn.dev/) · [Are We Learning Yet — GPU](https://www.arewelearningyet.com/gpu-computing/)
- [XGBoost GPU Support](https://xgboost.readthedocs.io/en/stable/gpu/index.html) · [LightGBM GPU Performance](https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html)
- [linfa](https://github.com/rust-ml/linfa) · [perpetual](https://github.com/perpetual-ml/perpetual) · [forust](https://github.com/jinlow/forust) · [gbdt-rs](https://github.com/mesalock-linux/gbdt-rs) · ["Writing the fastest GBDT library in Rust" — Tangram blog](https://dev.to/tangram/writing-the-fastest-gbdt-libary-in-rust-197k)
