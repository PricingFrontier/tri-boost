## 12 — Python API & interop

> Owns: the PyO3 binding crate `pattern-boost-py` (compiled module `pattern_boost._pattern_boost`), the maturin layout, the sklearn-compatible estimators, the native `Booster` Python surface (custom objectives, callbacks, early stopping), the explanation/table-export API, NumPy zero-copy + the optional Arrow path, GIL release, `.pyi` stubs, and the `PbError`→Python-exception mapping. Uses (does not own): `Booster`/`FitSpec`/`Model`/`PbError` (§02, §06), `Loss` (§05), `TableBank` (§08), serde (§10), the scoped rayon pool (§11). I1/I2 are upheld here purely by *delegation* — the binding never constructs trees or tables, it only marshals into the core, which enforces the invariants.

### 12.1 Decisions (with defaults)

1. **Binding shape: `cdylib` + abi3-py310.** `pattern-boost-py` is `crate-type=["cdylib"]`, `pyo3 = { features=["extension-module","abi3-py310"] }`, one wheel per `(os,arch)` across CPython ≥3.10 (polars/tokenizers pattern). The `#[pymodule]` fn is named `_pattern_boost`. `pattern-boost-core` carries **zero** pyo3 dependency.
2. **Two-layer API, native-in-Rust + sklearn-in-Python.** The `#[pyclass]` layer (`_Booster`, `_Model`, `_TableBank`) is a thin marshalling shell around the core handles. The sklearn estimators (`PatternBoostRegressor`, `PatternBoostClassifier`) are **pure Python** in `python/pattern_boost/sklearn.py` wrapping `_Booster` — keeping the 1:1 `__init__`↔attribute mirror that `get_params`/`set_params`/`clone` require out of Rust, where it is awkward.
3. **f32 core, f64-tolerant edge.** The native zero-copy path is `PyReadonlyArray2<'py, f32>`. The sklearn layer additionally accepts float64 (the numpy/pandas default) via an explicit, *copying* `f64→f32` cast at the Python boundary with a one-time `RuntimeWarning` (`pattern_boost.PrecisionWarning`), never a silent native-side cast.
4. **NumPy is primary; Arrow is optional.** numpy in/out is always available. The Arrow PyCapsule path (`from_arrow`) is gated behind the core `arrow` cargo feature and the `pyo3-arrow` dep; absent the feature, `from_arrow` raises `NotImplementedError`.
5. **GIL released around all compute** via `py.detach`, with a per-call scoped rayon pool sized by `n_jobs` (sklearn convention: `-1`/`None`→all cores).
6. **sklearn surface for v1 = the practical contract, not full `check_estimator`.** `fit→self`, `predict`, `predict_proba`/`decision_function` (classifier), `score` (mixin), `get_params`/`set_params`, `n_features_in_`, `feature_names_in_`, `classes_`, `NotFittedError`. This makes `Pipeline`/`GridSearchCV`/`cross_val_score`/stacking work — ~95% of the value — without chasing every `check_estimator` corner.
7. **Explanation API is a read-only projection.** `model.tables(...)` returns an immutable `_TableBank` view; export is JSON (§10). The firewall (§3) is surfaced as a Python exception, not a flag a user can override.

### 12.2 The native binding layer (`#[pyclass]` shells)

The binding holds owned core values and never re-implements algorithms. `_Booster` stores a `Booster` plus a parsed config; `fit` marshals arrays, builds a `FitSpec`, calls `Booster::fit`, and wraps the resulting `Model`.

```rust
#[pyclass(name = "Booster", module = "pattern_boost._pattern_boost")]
pub struct PyBooster { inner: pb_core::Booster, cfg: Config }

#[pymethods]
impl PyBooster {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> { /* parse → Config */ }

    /// X: float32 [n_rows, n_features] (C- or F-contiguous); y, weight, exposure: float32 [n_rows].
    /// objective: a registered name OR a Python callable (see §12.4). Returns a fitted PyModel.
    /// interaction = {max_order, groups}: builds the §07 `InteractionPolicy` (groups = optional whitelist).
    /// `validation_fraction` (default `None`): `None` ⇒ early stopping DISABLED (the full
    ///   `n_estimators` are fit; sklearn-familiar). `Some(frac)` ⇒ a deterministic seeded
    ///   internal holdout of that fraction drives the §05 deviance early stopping (mirrors the
    ///   core `Config.validation_fraction`, §06 — there is NO implicit "internal holdout by default").
    /// `teacher_raw` (default `None`): per-row teacher scores in OUR score space F (pre-link); when
    ///   present it is moved into `DistillSpec.teacher_raw` and threaded onto `FitSpec.distill` (§09);
    ///   `blend` is the true-label weight (default 0.5). `None` ⇒ no distillation (train on `y`).
    /// `feature_names` / `class_labels` (default `None`): persisted into `Model.schema` (`ModelSchema`,
    ///   §2.6 / R-SCHEMA) so names, classifier labels, and `predict_proba` column order survive a
    ///   serialize/deserialize round-trip; raw categoricals are re-encoded on serve through
    ///   `schema.cat_encoders` (audit-on-serve, R-CATSERVE).
    #[pyo3(signature = (x, y, *, weight=None, exposure=None, eval_set=None,
                        callbacks=None, monotone=None, max_interaction_order=3,
                        interaction_groups=None, validation_fraction=None,
                        teacher_raw=None, blend=0.5, teacher=None,
                        feature_names=None, class_labels=None, seed=0))]
    fn fit<'py>(
        &self, py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        y: PyReadonlyArray1<'py, f32>,
        weight: Option<PyReadonlyArray1<'py, f32>>,
        exposure: Option<PyReadonlyArray1<'py, f32>>,
        eval_set: Option<EvalSet<'py>>,
        callbacks: Option<Vec<Py<PyAny>>>,
        monotone: Option<&Bound<'py, PyDict>>,   // name -> {-1,0,+1}; marshalled into MonotoneMap (BTreeMap, §07)
        max_interaction_order: u8,
        interaction_groups: Option<Vec<Vec<String>>>,   // optional name-keyed support whitelist (§07)
        validation_fraction: Option<f32>,        // None = early stopping OFF (§06); Some(frac) = seeded holdout
        teacher_raw: Option<PyReadonlyArray1<'py, f32>>, // distillation teacher scores → FitSpec.distill (§09)
        blend: f32,                              // distill true-label weight, default 0.5 (§09)
        teacher: Option<String>,                 // TeacherKind provenance tag for the model card (§09)
        feature_names: Option<Vec<String>>,      // → ModelSchema.feature_names (R-SCHEMA); owned, Send
        class_labels: Option<Vec<String>>,       // → ModelSchema.class_labels (classifier; R-SCHEMA)
        seed: u64,
    ) -> PyResult<PyModel> {
        // ---- All Py borrows are runtime-checked and !Send: marshal EVERY input into Rust-OWNED
        //      buffers WHILE THE GIL IS HELD, then detach with only Send data (R-PYDETACH).
        let xv = x.as_array();                    // zero-copy ArrayView2<f32>, GIL held
        let binned: ServeBinnedMatrix = self.bin_or_reuse(xv)?;   // f32→u8 into an OWNED matrix (§03)
        let y_owned: Vec<f32> = y.as_slice()?.to_vec();           // copy out from the !Send borrow
        let w_owned: Option<Vec<f32>> = match weight { Some(a) => Some(a.as_slice()?.to_vec()), None => None };
        let e_owned: Option<Vec<f32>> = match exposure { Some(a) => Some(a.as_slice()?.to_vec()), None => None };
        // distillation: move the per-row teacher scores into an owned DistillSpec for FitSpec.distill (§09).
        let distill: Option<DistillSpec> = match teacher_raw {
            Some(t) => Some(DistillSpec {
                teacher_raw: t.as_slice()?.to_vec(),              // OWNED Vec<f32>, !Send borrow dropped here
                blend,                                            // §09 clamps to [0,1], rejects NaN
                teacher: parse_teacher_kind(teacher),             // TeacherKind tag (default Other)
            }),
            None => None,
        };
        // interaction = InteractionPolicy { max_order: max_interaction_order, groups } (§07/§2.9)
        let spec = self.build_spec(&y_owned, w_owned, e_owned, monotone,
                                   max_interaction_order, interaction_groups,
                                   validation_fraction, distill, seed)?;  // FitSpec OWNS its buffers
        let host = CallbackHost::new(py, callbacks, eval_set);   // §12.4 (owns Py handles; re-acquires GIL per fire)
        // Now everything crossing the boundary is Send: owned `binned`, `y_owned`, `spec`, `host`.
        let mut model = py.detach(move || self.pool().install(|| {
            self.inner.fit_with(&binned, &y_owned, &spec, &host) // host re-acquires GIL per fire
        })).map_err(map_err)?;
        // Stamp caller-supplied metadata into Model.schema (R-SCHEMA): names + classifier labels;
        // feature_kinds / cat_encoders / objective are filled by core from the fitted axes (frozen
        // encoders, R-CATSERVE). The schema serializes WITH the Model (schema_version covers it, §10).
        model.schema.set_names(feature_names);
        model.schema.set_class_labels(class_labels);
        Ok(PyModel { inner: Arc::new(model) })
    }
}
```

`PyModel` wraps `Arc<Model>` (cheap clone into the sklearn layer; shareable across threads). `predict` derives the view, detaches, scores via the §10 branch-free 8-cell LUT, returns `into_pyarray` (zero-copy ownership transfer):

```rust
#[pymethods]
impl PyModel {
    /// raw=False → response space (inverse link); raw=True → score space F.
    #[pyo3(signature = (x, *, raw=false))]
    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f32>, raw: bool)
        -> PyResult<Bound<'py, PyArray1<f32>>>
    {
        // `x.as_array()` is a runtime-checked, !Send borrow — it must NOT cross `py.detach`.
        // Bin into a Rust-OWNED ServeBinnedMatrix WHILE THE GIL IS HELD (re-encoding raw
        // categoricals through the FROZEN schema encoders, §04/R-CATSERVE), THEN detach
        // with only the owned matrix + the Arc<Model> (R-PYDETACH).
        let binned: ServeBinnedMatrix = self.inner.bin_for_serve(x.as_array())?;  // owned, GIL held
        let m = self.inner.clone();                                   // Arc<Model>, Send
        let out: Vec<f32> = py.detach(move || m.predict_into(&binned, raw));   // owned in, Vec out, no Py objects
        Ok(out.into_pyarray(py))
    }

    /// Binary class probabilities, columns ordered to `schema.class_labels` (§2.6 / R-SCHEMA).
    /// Logistic-only in v1; raises `ExactnessError`-adjacent `ValueError` if `schema.class_labels`
    /// is `None` (model was not fit as a classifier). Marshals into an owned `ServeBinnedMatrix`
    /// under the GIL, then detaches (R-PYDETACH).
    fn predict_proba<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f32>)
        -> PyResult<Bound<'py, PyArray2<f32>>>
    {
        let binned: ServeBinnedMatrix = self.inner.bin_for_serve(x.as_array())?;  // owned, GIL held
        let m = self.inner.clone();
        // p1 = inverse-link(F); column order [P(class0), P(class1)] matches schema.class_labels.
        let proba: Array2<f32> = py.detach(move || m.predict_proba_into(&binned))?;
        Ok(proba.into_pyarray(py))
    }

    fn tables<'py>(&self, py: Python<'py>, w: Option<RefMeasureSpec>) -> PyResult<PyTableBank> { /* §12.5 */ }
    fn to_json(&self) -> PyResult<String>;                 // §10 (emits Model.schema; readable export)
    /// bincode 2.x: `bincode::serde::encode_to_vec(&model, bincode::config::standard())` (frozen config, §10).
    /// `Model.schema` (names, class labels, frozen cat_encoders, objective) is part of the encoded
    /// bytes — `schema_version` covers it — so names/labels/encoders ROUND-TRIP (R-SCHEMA).
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>>;   // §10
    #[staticmethod] fn from_json(s: &str) -> PyResult<PyModel>;
    /// bincode 2.x: `bincode::serde::decode_from_slice(b, bincode::config::standard())` (§10).
    #[staticmethod] fn from_bytes(b: &[u8]) -> PyResult<PyModel>;
    #[getter] fn schema_version(&self) -> u32;
    #[getter] fn link(&self) -> &'static str;
    #[getter] fn n_trees(&self) -> usize;
    /// Feature names from `Model.schema.feature_names` (R-SCHEMA); reconstructs `feature_names_in_`
    /// after a serialize/deserialize round-trip.
    #[getter] fn feature_names(&self) -> Option<Vec<String>>;
    /// Class labels from `Model.schema.class_labels` (classifier; R-SCHEMA); reconstructs `classes_`
    /// and `predict_proba`'s column order after a round-trip.
    #[getter] fn class_labels(&self) -> Option<Vec<String>>;
}
```

**The marshalling rule (load-bearing for correctness, R-PYDETACH):** `PyReadonlyArray`/`ArrayView` is a runtime-checked borrow and is **not `Send`/`Sync`**, so it can **never** be moved into `py.detach`. The canonical pattern is therefore: *while the GIL is held*, marshal every input into Rust-**owned** buffers — bin `X` into an owned `ServeBinnedMatrix` (§03), and `.to_vec()` each 1-D `y`/`weight`/`exposure`/`teacher_raw` slice out from behind its borrow — and only **`Send`** data (the owned matrix, owned `Vec`s, the `FitSpec`/`spec`, the `Arc<Model>`) is then captured by the `move ||` closure passed to `py.detach`. The one-time copy of the 1-D arrays is negligible against training/scoring; `X` is binned (not copied) regardless. No `Py<PyAny>` is ever captured by the detached closure except through `CallbackHost`, which re-acquires the GIL on each fire (§12.4).

### 12.3 NumPy zero-copy & the optional Arrow path

- **Input (zero-copy).** `PyReadonlyArray2<'py, f32>` + `.as_array()` is zero-copy regardless of C/F contiguity. We bin column-major; an F-contiguous `X` therefore costs nothing on ingest, a C-contiguous one is read with strided column access (acceptable; binning is one-time). `.as_slice()` (used for 1-D `y`/`weight`/`exposure`) is zero-copy only when contiguous, else `PbError::InvalidInput` surfaces as `ValueError`.
- **dtype discipline.** The native `PyReadonlyArray2<f32>` *rejects* a float64 array (PyO3 raises `TypeError`) rather than silently casting — the f64→f32 cast lives only in the sklearn layer, explicit and warned (§12.1.3).
- **Output (zero-copy).** Predictions are built as a Rust `Vec<f32>` then `into_pyarray(py)` — ownership transfer, no copy. A caller-supplied `out=` buffer (`PyReadwriteArray1<f32>` + `as_array_mut()`) is supported on `predict` for allocation-free repeated scoring.
- **Arrow (optional, `arrow` feature).** `Booster.fit(..., x=<arrow capsule>)` and `PyModel.predict` accept anything exposing the Arrow PyCapsule protocol via `pyo3-arrow`; primitive `f32` columns view as `&[f32]` zero-copy and the **validity bitmap maps directly onto the reserved missing bin** (§03) — no sentinel dance. Dictionary/string columns route to the categorical-TS axis builder (§04). The Arrow path shares the same internal column store as numpy, so it adds an ingest adapter only, not a second engine.

### 12.4 Native `Booster`: custom objectives, callbacks, early stopping

**Custom objectives.** `objective=` accepts a registered name (`"poisson"`, `"gamma"`, `"tweedie:1.5"`, …, mapping to the §05 `Loss` implementors) **or** a Python callable with the XGBoost-style signature `fn(y_true, raw) -> (grad, hess)` returning two float32 arrays. The callable is wrapped in a `PyLoss` adapter implementing the core `Loss` trait; because `Loss::grad_hess` is called once per boosting round (not per row), the per-call GIL re-acquisition is amortized. `Loss::grad_hess` returns `Result<(), PbError>` (§05/§2.4), so a Python-side error is propagated as a typed `PbError` (never a panic on the train thread — the no-panic [GATE], §1):

```rust
struct PyLoss { f: Py<PyAny>, link: Link, init: f32 }
impl Loss for PyLoss {
    fn grad_hess(&self, y: &[f32], raw: &[f32], w: &[f32], out: &mut GradHess)
        -> Result<(), PbError>
    {
        Python::attach(|py| {                       // re-acquire GIL for this fire
            let (g, h): (Py<PyArray1<f32>>, Py<PyArray1<f32>>) =
                self.f.call1(py, (y.to_pyarray(py), raw.to_pyarray(py)))?.extract(py)?;
            // The Loss contract (§2.4) says the per-row weight SCALES (g,h): the callback returns
            // the UNWEIGHTED derivatives, so the adapter MUST apply weight[i] here (R-PYWEIGHTS).
            let (gs, hs) = (g.bind(py).readonly(), h.bind(py).readonly());
            let (gs, hs) = (gs.as_slice()?, hs.as_slice()?);
            for i in 0..out.g.len() {
                out.g[i] = gs[i] * w[i];
                out.h[i] = hs[i] * w[i];
            }
            Ok::<_, PyErr>(())
        }).map_err(|e| PbError::Internal { what: format!("py loss: {e}") })  // funnel PyErr → PbError; engine returns the Result via one `?`
    }
    // init_score/link/pred_from_raw/deviance: link is user-declared; deviance defaults to the link's canonical deviance.
}
```

A custom-objective model is `Approximate` only if its declared `link` is non-canonical for table reading; with a declared standard link it stays `Exact` (objective is orthogonal to tree shape — I1/I2 untouched, §3).

**Distillation (R-DISTILL).** Distillation is a `FitSpec` field, not a `BoosterConfig` knob — the per-row teacher scores belong with `weight`/`exposure` in `FitSpec`. The native surface exposes it as `fit(..., teacher_raw=<float32[n_rows]>, blend=0.5, teacher=None)`: when `teacher_raw` is present the binding moves it into a `DistillSpec { teacher_raw, blend, teacher }` (§09) and sets `FitSpec.distill = Some(..)`; `None` ⇒ no distillation (train on `y`). `teacher_raw` is in *our* score space `F` (pre-link, caller-aligned), `blend` is the true-label weight (default 0.5; §09 clamps to `[0,1]` and rejects NaN), and `teacher` is a provenance tag (`"catboost"`/`"lightgbm"`/`"xgboost"`/other → `TeacherKind`, default `Other`) for the model card. The sklearn/Python layer offers a convenience `distill=` argument (or the §12 distill helper) that **fits a CatBoost teacher and supplies `teacher_raw` + `blend`** — pattern-boost never links CatBoost (data-side only, behind the `distill` feature). The binding adds no blending math; `BlendedLoss` (§05) does the gradient blend in core. A distilled model stays `Exact` (only the loss target changes — I1/I2 untouched, §3/§09).

**Callbacks + early stopping.** A `callbacks=` list of Python callables receives a per-iteration `CallbackEnv` (`{iteration, train_metric, valid_metrics, model_proxy}`); a callback returning `True` requests a stop. `CallbackHost` owns the `Py` handles and, inside the detached training loop, re-acquires the GIL **only at the callback fire point** (end of each round), so the heavy histogram/split work runs GIL-free:

```rust
impl TrainObserver for CallbackHost {
    fn on_iteration(&self, it: usize, metrics: &Metrics) -> ControlFlow<()> {
        Python::attach(|py| { /* build CallbackEnv; call each cb; OR the bool results */ })
    }
}
```

Early stopping itself is a **built-in callback** (`early_stopping(rounds, metric=deviance, min_delta=0.0)`), evaluating the §05 strictly-proper deviance (never RMSE on Poisson/Gamma/Tweedie). Whether it runs is governed by **`validation_fraction`** (R-EARLYSTOP, mirroring the core `Config.validation_fraction`, §06): the default is **`None` ⇒ early stopping DISABLED** (sklearn-familiar; the full `n_estimators` are fit), and **`Some(frac)` ⇒ a deterministic seeded internal holdout** of that fraction is carved from the training rows and drives deviance early stopping. There is **no implicit "internal holdout by default."** An explicit `eval_set` may also be supplied as the holdout source; absent both an `eval_set` and a `validation_fraction`, early stopping is a no-op. `early_stopping_rounds` (default 50) is only the *patience* once early stopping is enabled — not the on/off switch. On stop, the model is truncated to the `best_iteration` prefix **before** table accumulation — the export-footgun guard from §02-gap-closing: tables must be built from the exact best-iteration tree prefix, enforced in core, not Python.

### 12.5 The explanation / table-export API

`model.tables(w=None)` returns a read-only `_TableBank` (the §08 `TableBank`, default `RefMeasure::ProductMarginals{laplace}`); passing a different `w` recomputes the bank **without retraining** (exactness-preserving, §3). The firewall is surfaced directly: if the model is `Approximate`, `tables()` raises `pattern_boost.ExactnessError` ("model carries non-decomposable structure; export is tables + residual model, not exact rating tables").

```python
bank = model.tables()                      # complete, lossless support
bank.f0                                     # float intercept
bank.importances()                          # {feature_set: sobol_share}, sums≈1 under product-w
bank.table(("age",))                        # 1-D EffectTable as a labelled numpy array + axis borders
bank.table(("age", "region"))               # 2-D; KeyError if that support wasn't realized
bank.to_json(top_k=None)                    # complete; top_k=k → pruned-for-display VIEW (§08),
                                            #   stamped "showing k of N tables, P% of variance" + w
bank.shap(X)                                # interventional Faith-Shap ≤order-3 as O(1) table reads (§08)
```

Key API rules: (1) `to_json()` with no `top_k` emits the **complete** support (the lossless inference set); `top_k` emits a pruned display view that is explicitly *not* sufficient for scoring and is labelled as such. (2) Every export carries the stamped `w` (§08). (3) `bank.shap`/importances are labelled "interventional"; stock TreeSHAP is **never** exposed (it is a test oracle only, §13). The binding adds no explanation math — it projects §08 results to numpy/dict/JSON.

### 12.6 sklearn estimators (pure Python, v1 surface)

```python
class PatternBoostRegressor(RegressorMixin, BaseEstimator):
    # Defaults MIRROR the core Config (§06) — single source of truth; the wrapper forwards, never re-decides.
    def __init__(self, *, objective="squared_error", n_estimators=1000, learning_rate=0.05,
                 max_bin=254, reg_lambda=1.0, max_interaction_order=3, interaction_groups=None,
                 subsample=1.0, monotone_constraints=None, n_jobs=None, random_state=0,
                 validation_fraction=None, early_stopping_rounds=50):  # None = early stopping OFF (§06, R-EARLYSTOP)
        # NO logic: store each arg unchanged as self.<same_name>  (powers get_params/set_params/clone)
        ...
    def fit(self, X, y, sample_weight=None, exposure=None, eval_set=None, teacher=None):
        X = self._validate(X, y)                     # f64→f32 (warn), set n_features_in_, feature_names_in_
        self._booster = _Booster(**self._native_kwargs())
        teacher_raw, blend = self._resolve_teacher(teacher, X, y)  # distill helper: fit CatBoost → (raw, blend), or (None, 0.5) (§09)
        self._model_ = self._booster.fit(X, y, weight=sample_weight, exposure=exposure, eval_set=eval_set,
                                         monotone=self._resolve_monotone(),  # name→sign, never positional
                                         max_interaction_order=self.max_interaction_order,
                                         interaction_groups=self.interaction_groups,  # optional name-keyed whitelist (§07)
                                         validation_fraction=self.validation_fraction,  # None = early stopping OFF (§06)
                                         teacher_raw=teacher_raw, blend=blend,  # distillation → FitSpec.distill (§09)
                                         feature_names=list(self.feature_names_in_),  # → Model.schema (R-SCHEMA)
                                         seed=self.random_state)
        return self                                   # sklearn contract
    def predict(self, X):
        check_is_fitted(self)                         # else NotFittedError
        return self._model_.predict(self._validate(X))
    def tables(self, w=None):  return self._model_.tables(w)   # explanation passthrough
```

The `feature_names_in_` / `n_features_in_` set by `_validate` are also forwarded into the native `fit` so they land in `Model.schema` (`ModelSchema.feature_names`, §2.6 / R-SCHEMA) and **survive serialization** — `from_bytes(to_bytes(m))` round-trips them; on `predict` the schema's frozen `cat_encoders` re-encode raw categoricals (audit-on-serve, R-CATSERVE), so naming is exact and leakage-free. The `distill=` argument feeds the §12 distill helper (R-DISTILL).

`PatternBoostClassifier` adds `classes_` (label encoding from `schema.class_labels`; binary only in v1, Logistic loss), `predict_proba` (columns ordered to `classes_`), `decision_function` (raw score), and `__sklearn_tags__` overrides (sklearn ≥1.6). On fit it passes the encoded label set as `class_labels=` so it persists in `Model.schema.class_labels` (R-SCHEMA); `classes_` and `predict_proba`'s column order are reconstructed from the schema after a serialize/deserialize round-trip (so a reloaded classifier predicts probabilities with the original label ordering, not a re-inferred one). Monotone constraints are **name-keyed** (`{"age": +1}`) and resolved against `feature_names_in_` into the §07 `MonotoneMap` (a `BTreeMap<String, MonoSign>` — ordered, config-only, never serialized, so it cannot perturb the determinism [GATE], §13.4), never positional — the §02 invariant. Fitted attributes carry the trailing underscore (`n_features_in_`, `feature_names_in_`, `classes_`, `_model_`). `set_params` clears any fitted state.

**Determinism contract surfaced in Python:** identical `random_state` + identical config ⇒ bit-identical `model.to_bytes()` regardless of `n_jobs` (the §1 [GATE], delegated to core). `n_jobs` changes only wall-clock, never the result — a property a Python test asserts (§12.8).

### 12.7 Error mapping & typing

`map_err(PbError) -> PyErr`, the single funnel for every fallible call:

| `PbError` variant | Python exception |
|---|---|
| `InvalidInput`, `ShapeMismatch` | `ValueError` |
| `DtypeMismatch` | `TypeError` |
| `InvalidConfig` | `ValueError` (config) |
| `InvariantViolated{..}` | `pattern_boost.InvariantError` |
| `ExactnessFirewall(..)` | `pattern_boost.ExactnessError` |
| `Serialization(..)` | `pattern_boost.SerializationError` |
| `Internal{..}` | `pattern_boost.InternalError` (a bug; degrades to a typed exception, never an interpreter crash) |

All four custom exceptions derive a package base `PatternBoostError`. The compiled module ships `py.typed` and hand-written `_pattern_boost.pyi` stubs (abi3 `text_signature` is available ≥3.10); the sklearn wrappers are typed inline. No panic ever crosses the FFI boundary — the no-panic policy (§1) plus the `Internal` funnel guarantee a Python exception instead of an abort.

### 12.8 Testing

- **sklearn integration (pytest):** `clone`/`get_params`/`set_params` round-trip; `Pipeline`+`GridSearchCV`+`cross_val_score` smoke tests; `NotFittedError` on unfitted predict; `predict_proba` column order == `classes_`; a targeted subset of `@parametrize_with_checks` (the practical-contract checks, not the full suite).
- **Zero-copy assertions:** verify `predict` output and `into_pyarray` share no buffer with input; verify `as_array` does not copy (pointer identity on F-contiguous input); verify f64 input emits exactly one `PrecisionWarning`.
- **GIL/parallelism:** a Python-threads test that runs two `fit`s concurrently and asserts real overlap (proving the GIL is released); an oversubscription guard asserting `n_jobs` bounds the pool.
- **Determinism [GATE] mirror:** `fit` at `n_jobs ∈ {1,2,8}` ⇒ byte-equal `to_bytes()` (Python-side reflection of the §1 core gate).
- **Custom-objective + callback:** a Python `squared_error` callable reproduces the native loss bit-for-bit *with non-uniform `sample_weight`* (asserts the `PyLoss` adapter scales `(g,h)` by `weight[i]`, R-PYWEIGHTS); a stop-requesting callback truncates at the expected iteration.
- **Early stopping (R-EARLYSTOP):** `validation_fraction=None` fits all `n_estimators` (no implicit holdout); `validation_fraction=Some(frac)` carves a deterministic seeded holdout, picks `best_iteration`, and tables are built from that prefix — same `(seed, validation_fraction)` ⇒ same `best_iteration`.
- **Distillation (R-DISTILL):** `teacher_raw=None` ≡ `blend=1.0` reproduces the non-distilled fit bit-for-bit (degenerate zero-teacher oracle); a supplied `teacher_raw` threads onto `FitSpec.distill` and the result stays `Exact`.
- **Schema round-trip (R-SCHEMA):** `from_bytes(to_bytes(m))` / `from_json(to_json(m))` recover `feature_names` and (classifier) `class_labels`; a reloaded classifier's `predict_proba` column order matches the original `classes_`; categoricals re-encode through the frozen `schema.cat_encoders` on serve (R-CATSERVE).
- **Firewall & round-trip:** an `Approximate` model raises `ExactnessError` on `.tables()`; `from_json(to_json(m))` and `from_bytes(to_bytes(m))` reproduce predictions exactly (§10).
- **Stub/lint:** `mypy --strict` over the typed Python package; stubtest against the compiled module.

### 12.9 Open fork (recommended default)

**Free-threaded CPython (3.13t/3.14t).** abi3 does **not** cover the free-threaded ABI (PEP 803 `abi3t` is separate). v1 recommendation: ship abi3-py310 only and **defer** a separate `abi3t` wheel build until free-threaded adoption justifies the extra CI matrix leg. The core is already `Send + Sync` and uses its own rayon pool, so enabling it later is a build-config change, not a redesign — left as a v1.5+ CI item in §14.
