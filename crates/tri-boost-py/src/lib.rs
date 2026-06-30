//! PyO3 bindings for tri-boost (spec §12).
//!
//! `#![allow(unsafe_code)]` is required here (and is the single justified,
//! encapsulated exception to the core's `forbid`): the pyo3/numpy procedural macros
//! expand to `unsafe`. The pure Rust core carries `#![forbid(unsafe_code)]`; this crate
//! remains a thin FFI adapter and owns no model math.
#![allow(unsafe_code)]

use numpy::{
    IntoPyArray, PyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadwriteArray1, PyUntypedArrayMethods,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;
use tri_boost_core::boosters::{
    BoosterConfig, CellRefit, DartSpec, EnsembleSpec, NesterovSpec, RefitSpec,
};
use tri_boost_core::cat::{CatTarget, LeakageScheme, Smooth, TsConfig, TsEncodingId};
use tri_boost_core::constraints::{CredibilityFloor, InteractionPolicy, MonoSign, MonotoneMap};
use tri_boost_core::data::{
    bin, bin_columns, bin_serve_columns, bin_train_columns, BinConfig, BinnedMatrix,
    CategoricalColumn, FeatureId, NumericColumn, ServeCategoricalColumn,
};
use tri_boost_core::engine::{Booster, Config, FitSpec, HistPrecision, Model, Sampling};
use tri_boost_core::error::{Invariant, PbError};
use tri_boost_core::explain::{OverflowPolicy, RefMeasure, TableBank, TableBudget};
use tri_boost_core::loss::{Gamma, Logistic, Loss, Poisson, SquaredError, Tweedie};
use tri_boost_core::serialize::RatingBasis;

create_exception!(
    _tri_boost,
    TriBoostError,
    PyException,
    "Base tri-boost exception."
);
create_exception!(
    _tri_boost,
    InvariantError,
    TriBoostError,
    "A lossless decomposability invariant failed."
);
create_exception!(
    _tri_boost,
    ExactnessError,
    TriBoostError,
    "An exact-only operation was attempted on an approximate model."
);
create_exception!(
    _tri_boost,
    SerializationError,
    TriBoostError,
    "Model serialization or deserialization failed."
);
create_exception!(
    _tri_boost,
    InternalError,
    TriBoostError,
    "An internal tri-boost implementation invariant failed."
);

#[derive(Debug, Clone)]
enum Objective {
    SquaredError,
    Logistic,
    Poisson,
    Gamma,
    Tweedie { rho: f32 },
}

enum LossChoice {
    SquaredError(SquaredError),
    Logistic(Logistic),
    Poisson(Poisson),
    Gamma(Gamma),
    Tweedie(Tweedie),
}

impl LossChoice {
    fn as_loss(&self) -> &dyn Loss {
        match self {
            LossChoice::SquaredError(loss) => loss,
            LossChoice::Logistic(loss) => loss,
            LossChoice::Poisson(loss) => loss,
            LossChoice::Gamma(loss) => loss,
            LossChoice::Tweedie(loss) => loss,
        }
    }
}

impl Objective {
    fn parse(name: Option<String>, tweedie_rho: f32) -> Result<Self, PbError> {
        let normalized = name
            .unwrap_or_else(|| "squared_error".to_owned())
            .replace('-', "_")
            .to_ascii_lowercase();
        match normalized.as_str() {
            "squared_error" | "squarederror" | "l2" | "regression" => Ok(Self::SquaredError),
            "logistic" | "binary_logloss" | "log_loss" | "classifier" => Ok(Self::Logistic),
            "poisson" => Ok(Self::Poisson),
            "gamma" => Ok(Self::Gamma),
            "tweedie" => {
                Tweedie::new(tweedie_rho)?;
                Ok(Self::Tweedie { rho: tweedie_rho })
            }
            other => Err(PbError::InvalidConfig {
                what: format!("unknown objective `{other}`"),
            }),
        }
    }

    fn instantiate(&self) -> Result<LossChoice, PbError> {
        match self {
            Objective::SquaredError => Ok(LossChoice::SquaredError(SquaredError)),
            Objective::Logistic => Ok(LossChoice::Logistic(Logistic)),
            Objective::Poisson => Ok(LossChoice::Poisson(Poisson)),
            Objective::Gamma => Ok(LossChoice::Gamma(Gamma)),
            Objective::Tweedie { rho } => Ok(LossChoice::Tweedie(Tweedie::new(*rho)?)),
        }
    }
}

/// Low-level Python booster wrapper.
#[pyclass(name = "_Booster", skip_from_py_object)]
#[derive(Clone)]
struct PyBooster {
    config: Config,
    bin_config: BinConfig,
    objective: Objective,
    credibility: CredibilityFloor,
    interaction: InteractionPolicy,
    cat_config: TsConfig,
    seed: u64,
    n_jobs: Option<usize>,
}

/// Resolve the `hist_precision` kwarg into the core enum.
fn parse_hist_precision(name: Option<&str>) -> Result<HistPrecision, PbError> {
    match name.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("full") | Some("f64") | Some("fullf64") => Ok(HistPrecision::FullF64),
        Some("quantized") | Some("qhist") | Some("i32") | Some("quantizedi32") => {
            Ok(HistPrecision::QuantizedI32)
        }
        Some(other) => Err(PbError::InvalidConfig {
            what: format!("hist_precision must be 'full' or 'quantized', got {other:?}"),
        }),
    }
}

/// Resolve the `subsample` kwarg into a row-sampling strategy (`None`/`1.0` ⇒ full rows;
/// `0 < r < 1` ⇒ MVS at that rate, with `mvs_min_rows` as the floor).
fn parse_sampling(subsample: Option<f32>, mvs_min_rows: u32) -> Sampling {
    match subsample {
        None => Sampling::Full,
        Some(rate) if rate >= 1.0 => Sampling::Full,
        Some(rate) => Sampling::Mvs {
            rate,
            min_rows: mvs_min_rows.max(1),
        },
    }
}

fn parse_cat_target(name: Option<&str>) -> Result<CatTarget, PbError> {
    match name.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("mean") | Some("rate") => Ok(CatTarget::Mean),
        Some("log_mean") | Some("logmean") | Some("log") => Ok(CatTarget::LogMean),
        Some(other) => Err(PbError::InvalidConfig {
            what: format!("cat_target must be 'mean' or 'log_mean', got {other:?}"),
        }),
    }
}

fn parse_cat_leakage(name: Option<&str>, n_perms: u32, k: u32) -> Result<LeakageScheme, PbError> {
    match name.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        // Default to K-fold cross-fit: lowest-variance leakage-free encoding measured on MTPL
        // (beats both the old Ordered{1} default and leave-one-out on frequency and severity).
        None | Some("kfold") | Some("k_fold") | Some("crossfit") | Some("cross_fit") => {
            Ok(LeakageScheme::KFold { k })
        }
        Some("ordered") => Ok(LeakageScheme::Ordered { n_perms }),
        Some("loo") | Some("leave_one_out") | Some("leaveoneout") => Ok(LeakageScheme::LeaveOneOut),
        Some(other) => Err(PbError::InvalidConfig {
            what: format!("cat_leakage must be 'kfold', 'ordered', or 'loo', got {other:?}"),
        }),
    }
}

#[pymethods]
impl PyBooster {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        n_trees=1000,
        learning_rate=0.05,
        lambda_=1.0,
        l1_leaf=0.0,
        min_split_gain=0.0,
        max_delta_step=None,
        max_bin=254,
        objective=None,
        tweedie_rho=1.5,
        min_data_in_leaf=0,
        min_sum_hessian_in_leaf=0.0,
        min_weight_sum_in_leaf=0.0,
        path_smooth=0.0,
        subsample=None,
        colsample_bytree=1.0,
        learning_rate_decay=0.0,
        validation_fraction=None,
        early_stopping_rounds=50,
        leaf_refine_steps=0,
        leaf_refine_backtracks=4,
        mvs_min_rows=1,
        hist_precision=None,
        n_bags=0,
        bag_subsample=1.0,
        ridge_refit_l2=None,
        ridge_refit_max_iter=5,
        nesterov=false,
        dart_drop_rate=None,
        random_strength=0.0,
        reanchor=false,
        max_interaction_order=3,
        cat_smooth=None,
        cat_target=None,
        cat_leakage=None,
        cat_n_perms=1,
        cat_k=5,
        cat_min_data_per_group=10.0,
        cell_refit_base=None,
        cell_refit_gamma=2.0,
        seed=0,
        n_jobs=None
    ))]
    fn new(
        n_trees: u32,
        learning_rate: f32,
        lambda_: f32,
        l1_leaf: f32,
        min_split_gain: f32,
        max_delta_step: Option<f32>,
        max_bin: u8,
        objective: Option<String>,
        tweedie_rho: f32,
        min_data_in_leaf: u32,
        min_sum_hessian_in_leaf: f32,
        min_weight_sum_in_leaf: f32,
        path_smooth: f32,
        subsample: Option<f32>,
        colsample_bytree: f32,
        learning_rate_decay: f32,
        validation_fraction: Option<f32>,
        early_stopping_rounds: u32,
        leaf_refine_steps: u8,
        leaf_refine_backtracks: u8,
        mvs_min_rows: u32,
        hist_precision: Option<String>,
        n_bags: u16,
        bag_subsample: f32,
        ridge_refit_l2: Option<f32>,
        ridge_refit_max_iter: u8,
        nesterov: bool,
        dart_drop_rate: Option<f32>,
        random_strength: f32,
        reanchor: bool,
        max_interaction_order: u8,
        cat_smooth: Option<f32>,
        cat_target: Option<String>,
        cat_leakage: Option<String>,
        cat_n_perms: u32,
        cat_k: u32,
        cat_min_data_per_group: f32,
        cell_refit_base: Option<f64>,
        cell_refit_gamma: f64,
        seed: u64,
        n_jobs: Option<usize>,
    ) -> PyResult<Self> {
        let cell_refit = cell_refit_base.map(|base| CellRefit {
            base,
            gamma: cell_refit_gamma,
        });
        let ensemble = if n_bags == 0 {
            EnsembleSpec::Off
        } else {
            EnsembleSpec::OuterBag {
                n_bags,
                bag_subsample,
                cell_refit,
            }
        };
        let refit_leaves = match ridge_refit_l2 {
            None => RefitSpec::Off,
            Some(l2) => RefitSpec::Ridge {
                l2,
                max_iter: ridge_refit_max_iter,
                every_k_trees: None,
            },
        };
        if nesterov {
            // AGBM look-ahead currently diverges (the §09.4 momentum-correction step is not
            // implemented), so refuse it loudly rather than silently produce a blown-up model.
            return Err(py_err(PbError::InvalidConfig {
                what: "nesterov/AGBM acceleration is experimental and currently unstable \
                       (it diverges; the momentum-correction step is not yet implemented) — \
                       it is not supported in this release"
                    .into(),
            }));
        }
        let nesterov = NesterovSpec::Off;
        let dart = dart_drop_rate.map(|drop_rate| DartSpec {
            drop_rate,
            normalize: true,
        });
        let boosters = BoosterConfig {
            refit_leaves,
            nesterov,
            ensemble,
            dart,
            random_strength,
            reanchor,
        };
        let config = Config {
            n_trees,
            learning_rate,
            lambda: lambda_,
            l1_leaf,
            min_split_gain,
            max_delta_step,
            sampling: parse_sampling(subsample, mvs_min_rows),
            colsample_bytree,
            learning_rate_decay,
            validation_fraction,
            early_stopping_rounds,
            leaf_refine_steps,
            leaf_refine_backtracks,
            hist_precision: parse_hist_precision(hist_precision.as_deref()).map_err(py_err)?,
            boosters,
        };
        config.validate().map_err(py_err)?;
        let bin_config = BinConfig {
            max_bin,
            ..BinConfig::default()
        };
        bin_config.validate().map_err(py_err)?;
        let credibility = CredibilityFloor {
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            min_weight_sum_in_leaf,
            path_smooth,
        };
        credibility.validate().map_err(py_err)?;
        let interaction = InteractionPolicy {
            max_order: max_interaction_order,
            ..InteractionPolicy::default()
        };
        // Categorical target-statistic config: default to empirical-Bayes Auto smoothing
        // (spec §04), overridable by a fixed pseudo-count via `cat_smooth`.
        let cat_config = TsConfig {
            leakage: parse_cat_leakage(cat_leakage.as_deref(), cat_n_perms, cat_k)
                .map_err(py_err)?,
            smooth: match cat_smooth {
                None => Smooth::Auto,
                Some(m) => Smooth::Fixed { m },
            },
            target: parse_cat_target(cat_target.as_deref()).map_err(py_err)?,
            min_data_per_group: cat_min_data_per_group,
            ..TsConfig::default()
        };
        cat_config.validate().map_err(py_err)?;
        let objective = Objective::parse(objective, tweedie_rho).map_err(py_err)?;
        if matches!(n_jobs, Some(0)) {
            return Err(py_err(PbError::InvalidConfig {
                what: "n_jobs must be >= 1 when set".into(),
            }));
        }
        Ok(Self {
            config,
            bin_config,
            objective,
            interaction,
            credibility,
            cat_config,
            seed,
            n_jobs,
        })
    }

    #[pyo3(signature = (x, y, weight=None, exposure=None, feature_names=None, class_labels=None, monotone=None, cat_x=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray1<'_, f32>,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
        feature_names: Option<Vec<String>>,
        class_labels: Option<Vec<String>>,
        monotone: Option<Vec<i8>>,
        // Native categorical columns as per-row string labels: `cat_x[j]` is one column,
        // appended after the numeric columns of `x` (raw ids assigned sequentially).
        cat_x: Option<Vec<Vec<String>>>,
    ) -> PyResult<PyModel> {
        let columns = raw_columns_from_array(x)?;
        if columns.len() + cat_x.as_ref().map_or(0, Vec::len) == 0 {
            return Err(py_err(PbError::InvalidInput {
                what: "x must contain at least one feature (numeric or categorical)".into(),
            }));
        }
        let y = array1_to_vec(y, "y")?;
        let weight = weight.map(|w| array1_to_vec(w, "weight")).transpose()?;
        let exposure = exposure.map(|e| array1_to_vec(e, "exposure")).transpose()?;
        let state = self.clone();
        let model = py
            .detach(move || {
                fit_owned(
                    state,
                    columns,
                    y,
                    weight,
                    exposure,
                    feature_names,
                    class_labels,
                    monotone,
                    cat_x,
                )
            })
            .map_err(py_err)?;
        Ok(PyModel {
            model: Arc::new(model),
        })
    }
}

/// Low-level Python model wrapper.
#[pyclass(name = "_Model", skip_from_py_object)]
#[derive(Clone)]
struct PyModel {
    model: Arc<Model>,
}

#[pymethods]
impl PyModel {
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        let model = Model::from_json(s).map_err(py_err)?;
        Ok(Self {
            model: Arc::new(model),
        })
    }

    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        let model = Model::from_bincode(bytes).map_err(py_err)?;
        Ok(Self {
            model: Arc::new(model),
        })
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.model.grids.len()
    }

    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.model.schema.feature_names.clone()
    }

    #[getter]
    fn class_labels(&self) -> Option<Vec<String>> {
        self.model.schema.class_labels.clone()
    }

    #[pyo3(signature = (x, out=None, cat_x=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        out: Option<Bound<'py, PyArray1<f32>>>,
        cat_x: Option<Vec<Vec<String>>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let pred = py
            .detach(move || {
                let binned = serve_binned_for_model(&model, columns, cat_x)?;
                model.predict_binned(&binned, None)
            })
            .map_err(py_err)?;
        write_or_return_array1(py, pred, out)
    }

    #[pyo3(signature = (x, out=None, cat_x=None))]
    fn predict_raw<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        out: Option<Bound<'py, PyArray1<f32>>>,
        cat_x: Option<Vec<Vec<String>>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let raw = py
            .detach(move || {
                let binned = serve_binned_for_model(&model, columns, cat_x)?;
                let mut out = vec![0.0_f32; binned.n_rows as usize];
                model.score_trees(&binned, None, &mut out)?;
                Ok::<Vec<f32>, PbError>(out)
            })
            .map_err(py_err)?;
        write_or_return_array1(py, raw, out)
    }

    #[pyo3(signature = (x, cat_x=None))]
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        cat_x: Option<Vec<Vec<String>>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if self.model.link != tri_boost_core::loss::Link::Logit {
            return Err(PyTypeError::new_err(
                "predict_proba is only available for logit-link models",
            ));
        }
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let pred = py
            .detach(move || {
                let binned = serve_binned_for_model(&model, columns, cat_x)?;
                model.predict_binned(&binned, None)
            })
            .map_err(py_err)?;
        if pred.is_empty() {
            // sklearn contract: predict_proba returns (n_samples, n_classes); for an
            // empty design that is (0, 2), not the (0, 0) that from_vec2 of [] yields.
            return Ok(numpy::PyArray2::<f32>::zeros(py, [0usize, 2], false));
        }
        let mut rows = Vec::with_capacity(pred.len());
        for p1 in pred {
            rows.push(vec![1.0 - p1, p1]);
        }
        PyArray::from_vec2(py, &rows).map_err(|err| {
            InternalError::new_err(format!("could not allocate probability array: {err}"))
        })
    }

    #[pyo3(signature = (x, ref_measure=None, laplace=1.0, cat_x=None, overflow=None))]
    fn explain(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        ref_measure: Option<String>,
        laplace: f32,
        cat_x: Option<Vec<Vec<String>>>,
        overflow: Option<String>,
    ) -> PyResult<PyTableBank> {
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let w = parse_ref_measure(ref_measure, laplace).map_err(py_err)?;
        let budget = parse_table_budget(overflow.as_deref()).map_err(py_err)?;
        let bank = py
            .detach(move || {
                let binned = serve_binned_for_model(&model, columns, cat_x)?;
                model.explain_with_budget(
                    &tri_boost_core::data::ServeBinnedMatrix(binned),
                    w,
                    budget,
                )
            })
            .map_err(py_err)?;
        Ok(PyTableBank { bank })
    }

    #[pyo3(signature = (x, ref_measure=None, laplace=1.0, basis_json=None, cat_x=None, overflow=None))]
    #[allow(clippy::too_many_arguments)]
    fn tables(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        ref_measure: Option<String>,
        laplace: f32,
        basis_json: Option<&str>,
        cat_x: Option<Vec<Vec<String>>>,
        overflow: Option<String>,
    ) -> PyResult<String> {
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let w = parse_ref_measure(ref_measure, laplace).map_err(py_err)?;
        let basis = parse_rating_basis(basis_json)?;
        let budget = parse_table_budget(overflow.as_deref()).map_err(py_err)?;
        py.detach(move || {
            let binned = serve_binned_for_model(&model, columns, cat_x)?;
            let bank = model.explain_with_budget(
                &tri_boost_core::data::ServeBinnedMatrix(binned),
                w,
                budget,
            )?;
            let export =
                bank.to_rating_export(model.link, &model.mode, &model.schema, basis.as_ref())?;
            serde_json::to_string_pretty(&export).map_err(|err| {
                PbError::Serialization(format!("could not serialize RatingExport JSON: {err}"))
            })
        })
        .map_err(py_err)
    }

    fn to_json(&self) -> PyResult<String> {
        self.model.to_json().map_err(py_err)
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.model.to_bincode().map_err(py_err)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

/// Low-level Python table-bank wrapper.
#[pyclass(name = "_TableBank", skip_from_py_object)]
#[derive(Clone)]
struct PyTableBank {
    bank: TableBank,
}

#[pymethods]
impl PyTableBank {
    #[getter]
    fn f0(&self) -> f64 {
        self.bank.f0
    }

    #[getter]
    fn n_tables(&self) -> usize {
        // Total effect tables in the decomposition, including factored over-budget order-3
        // effects (§08.10) — which are real f_u terms, just stored per-tree, not densified.
        self.bank.tables.len() + self.bank.factored.len()
    }

    fn score_cells(&self, cells: Vec<u32>) -> PyResult<f64> {
        self.bank.score(&cells).map_err(py_err)
    }

    fn sobol(&self) -> Vec<(Vec<u32>, f64)> {
        self.bank
            .sobol()
            .into_iter()
            .map(|(u, s)| (u.0.iter().map(|f| f.0).collect(), s))
            .collect()
    }
}

/// Build a name-keyed [`MonotoneMap`] from a positional sign vector (`-1`/`0`/`+1` per
/// feature). Keyed `f{i}` to match the fit-time schema names (the user's `feature_names`
/// are applied to the schema only AFTER fit, so monotone resolution runs against `f{i}`).
fn build_monotone_map(signs: Option<&[i8]>, n_features: usize) -> Result<MonotoneMap, PbError> {
    let mut map = MonotoneMap::new();
    let Some(signs) = signs else {
        return Ok(map);
    };
    if signs.len() != n_features {
        return Err(PbError::ShapeMismatch {
            what: format!("monotone len {} != n_features {n_features}", signs.len()),
        });
    }
    for (i, &s) in signs.iter().enumerate() {
        let sign = match s {
            0 => continue,
            1 => MonoSign::Increasing,
            -1 => MonoSign::Decreasing,
            other => {
                return Err(PbError::InvalidConfig {
                    what: format!("monotone[{i}] must be -1, 0, or 1, got {other}"),
                })
            }
        };
        map.insert(format!("f{i}"), sign);
    }
    Ok(map)
}

#[allow(clippy::too_many_arguments)]
fn fit_owned(
    state: PyBooster,
    columns: Vec<Vec<f32>>,
    y: Vec<f32>,
    weight: Option<Vec<f32>>,
    exposure: Option<Vec<f32>>,
    feature_names: Option<Vec<String>>,
    class_labels: Option<Vec<String>>,
    monotone: Option<Vec<i8>>,
    cat_x: Option<Vec<Vec<String>>>,
) -> Result<Model, PbError> {
    let n_numeric = columns.len();
    let n_cat = cat_x.as_ref().map_or(0, Vec::len);
    let monotone_map = build_monotone_map(monotone.as_deref(), n_numeric + n_cat)?;
    // Internal early-stopping needs the validation fold's categorical encodings to exclude
    // each val row's own target. KFold cross-fit (the default) produces exactly that — OOF
    // per-row training encodings (cat.rs `kfold_training_encodings` excludes the row's whole
    // fold) — and the val fold is carved by index from that already-encoded matrix, so it is
    // leakage-free. Ordered/LeaveOneOut have subtler profiles (the LOO target-encoding
    // pathology), so keep them gated with internal early stopping for now.
    if cat_x.is_some()
        && state.config.validation_fraction.is_some()
        && !matches!(state.cat_config.leakage, LeakageScheme::KFold { .. })
    {
        return Err(PbError::InvalidConfig {
            what: "validation_fraction with native categoricals is only leakage-free under \
                   cat_leakage='kfold' (the default); 'ordered'/'loo' are not yet supported \
                   with internal early stopping — use 'kfold' or an external validation split"
                .into(),
        });
    }
    let run = || -> Result<Model, PbError> {
        let loss = state.objective.instantiate()?;
        let spec = FitSpec {
            loss: loss.as_loss(),
            weight: weight.as_deref(),
            exposure: exposure.as_deref(),
            monotone: monotone_map.clone(),
            interaction: state.interaction.clone(),
            credibility: state.credibility,
            seed: state.seed,
        };
        match &cat_x {
            None => {
                let refs: Vec<&[f32]> = columns.iter().map(Vec::as_slice).collect();
                let x = bin_columns(&refs, weight.as_deref(), &state.bin_config, state.seed)?;
                Booster::with_config(state.config.clone()).fit(&x, &y, &spec)
            }
            // Native categorical path: numeric columns keep raw ids `0..n_numeric`, each
            // categorical column gets a sequential raw id after them, so the serve-time
            // `bin_serve_columns` re-aligns by raw without any extra index bookkeeping.
            Some(cats) => {
                let numeric = columns
                    .iter()
                    .enumerate()
                    .map(|(i, values)| {
                        Ok::<_, PbError>(NumericColumn {
                            raw: FeatureId(raw_id(i)?),
                            values,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let categorical = cats
                    .iter()
                    .enumerate()
                    .map(|(j, levels)| {
                        Ok::<_, PbError>(CategoricalColumn {
                            raw: FeatureId(raw_id(n_numeric + j)?),
                            id: TsEncodingId(0),
                            levels,
                            config: &state.cat_config,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let fitted = bin_train_columns(
                    &numeric,
                    &categorical,
                    &y,
                    weight.as_deref(),
                    exposure.as_deref(),
                    &state.bin_config,
                    state.seed,
                )?;
                Booster::with_config(state.config.clone()).fit_train(
                    &fitted.train,
                    &y,
                    &spec,
                    fitted.cat_encoders,
                )
            }
        }
    };
    let mut model = if let Some(n_jobs) = state.n_jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_jobs)
            .build()
            .map_err(|err| PbError::InvalidConfig {
                what: format!("could not build rayon pool: {err}"),
            })?
            .install(run)?
    } else {
        run()?
    };
    if let Some(names) = feature_names {
        if names.len() != model.schema.feature_names.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "feature_names len {} != n_features {}",
                    names.len(),
                    model.schema.feature_names.len()
                ),
            });
        }
        model.schema.feature_names = names;
    }
    model.schema.class_labels = class_labels;
    model.validate()?;
    Ok(model)
}

fn raw_columns_from_array(x: PyReadonlyArray2<'_, f32>) -> PyResult<Vec<Vec<f32>>> {
    let shape = x.shape();
    let n_rows = *shape.first().ok_or_else(|| {
        PyTypeError::new_err("x must be a two-dimensional C-contiguous float32 array")
    })?;
    let n_features = *shape.get(1).ok_or_else(|| {
        PyTypeError::new_err("x must be a two-dimensional C-contiguous float32 array")
    })?;
    // 0 numeric columns is legal when native categorical columns are supplied (all-categorical
    // input); the total-feature (numeric + categorical) check lives in the core fit/serve.
    if n_features == 0 {
        return Ok(Vec::new());
    }
    let slice = x.as_slice().map_err(|_| {
        PyTypeError::new_err("x must be a C-contiguous numpy.ndarray with dtype float32")
    })?;
    let mut columns: Vec<Vec<f32>> = (0..n_features)
        .map(|_| Vec::with_capacity(n_rows))
        .collect();
    for row in slice.chunks_exact(n_features) {
        for (feature, &value) in row.iter().enumerate() {
            let col = columns.get_mut(feature).ok_or_else(|| {
                PyTypeError::new_err("x row width changed while marshaling the array")
            })?;
            col.push(value);
        }
    }
    Ok(columns)
}

fn array1_to_vec(x: PyReadonlyArray1<'_, f32>, name: &str) -> PyResult<Vec<f32>> {
    let slice = x.as_slice().map_err(|_| {
        PyTypeError::new_err(format!(
            "{name} must be a C-contiguous numpy.ndarray with dtype float32"
        ))
    })?;
    Ok(slice.to_vec())
}

fn write_or_return_array1<'py>(
    py: Python<'py>,
    values: Vec<f32>,
    out: Option<Bound<'py, PyArray1<f32>>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let Some(out) = out else {
        return Ok(values.into_pyarray(py));
    };
    {
        // `try_readwrite` (not `readwrite`) so a read-only / borrowed numpy array returns
        // a typed Python error rather than panicking across the FFI boundary.
        let mut borrowed: PyReadwriteArray1<'_, f32> = out
            .try_readwrite()
            .map_err(|_| PyTypeError::new_err("out must be a writable contiguous float32 array"))?;
        let out_slice = borrowed
            .as_slice_mut()
            .map_err(|_| PyTypeError::new_err("out must be a contiguous writable float32 array"))?;
        if out_slice.len() != values.len() {
            return Err(py_err(PbError::ShapeMismatch {
                what: format!(
                    "out len {} != prediction len {}",
                    out_slice.len(),
                    values.len()
                ),
            }));
        }
        out_slice.copy_from_slice(&values);
    }
    Ok(out)
}

/// Convert a feature position into a raw [`FeatureId`], guarding the `u32` cast.
fn raw_id(i: usize) -> Result<u32, PbError> {
    u32::try_from(i).map_err(|_| PbError::InvalidInput {
        what: "more than u32::MAX features is out of scope for v1".into(),
    })
}

/// Rebuild a serve [`BinnedMatrix`] for prediction/explanation. The numeric-only fast
/// path bins through the model grids positionally; the categorical path re-encodes string
/// labels through the frozen [`crate::ModelSchema`] encoders via [`bin_serve_columns`]
/// (matched by raw id, so numeric `0..n_numeric` then sequential categorical ids align
/// with how `fit` laid the axes out).
fn serve_binned_for_model(
    model: &Model,
    numeric_cols: Vec<Vec<f32>>,
    cat_x: Option<Vec<Vec<String>>>,
) -> Result<BinnedMatrix, PbError> {
    let Some(cats) = cat_x else {
        return binned_for_model(model, numeric_cols);
    };
    let n_numeric = numeric_cols.len();
    let numeric = numeric_cols
        .iter()
        .enumerate()
        .map(|(i, values)| {
            Ok::<_, PbError>(NumericColumn {
                raw: FeatureId(raw_id(i)?),
                values,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let categorical = cats
        .iter()
        .enumerate()
        .map(|(j, levels)| {
            Ok::<_, PbError>(ServeCategoricalColumn {
                raw: FeatureId(raw_id(n_numeric + j)?),
                id: TsEncodingId(0),
                levels,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let serve = bin_serve_columns(
        &numeric,
        &categorical,
        &model.grids,
        &model.provenance,
        &model.schema.cat_encoders,
    )?;
    Ok(serve.0)
}

fn binned_for_model(model: &Model, columns: Vec<Vec<f32>>) -> Result<BinnedMatrix, PbError> {
    if columns.len() != model.grids.len() {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "x has {} columns, model has {} features",
                columns.len(),
                model.grids.len()
            ),
        });
    }
    let n_rows = columns.first().map_or(0usize, Vec::len);
    let mut data = Vec::with_capacity(columns.len());
    for (axis, (col, grid)) in columns.iter().zip(&model.grids).enumerate() {
        if col.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("x column {axis} len {} != n_rows {n_rows}", col.len()),
            });
        }
        let mut bins = Vec::with_capacity(n_rows);
        for &value in col {
            bins.push(bin(value, grid)?);
        }
        data.push(bins);
    }
    Ok(BinnedMatrix {
        data,
        n_rows: u32::try_from(n_rows).map_err(|_| PbError::InvalidInput {
            what: "more than u32::MAX rows is out of scope".into(),
        })?,
        grids: model.grids.clone(),
        provenance: model.provenance.clone(),
    })
}

fn parse_ref_measure(name: Option<String>, laplace: f32) -> Result<RefMeasure, PbError> {
    let normalized = name
        .unwrap_or_else(|| "product_marginals".to_owned())
        .replace('-', "_")
        .to_ascii_lowercase();
    match normalized.as_str() {
        "product" | "product_marginals" => Ok(RefMeasure::ProductMarginals { laplace }),
        "uniform" => Ok(RefMeasure::Uniform),
        other => Err(PbError::InvalidConfig {
            what: format!("unknown reference measure `{other}`"),
        }),
    }
}

/// Resolve the table-budget overflow policy for `explain`/`tables` (spec §08.10).
///
/// Defaults to `Error` (the loud, fast fail-fast policy): when a converged model's merged
/// grid pushes an order-3 table over the dense `max_table_cells` budget, the caller gets an
/// immediate, actionable `PbError::TableBudget` rather than a silent failure. `"sparse"` opts
/// into the EXACT `SparseFallback` storage — currently a slow path (accumulation still walks
/// the dense extent), so it is opt-in until the occupancy-driven walk lands. Both policies
/// are exactness-preserving.
fn parse_table_budget(overflow: Option<&str>) -> Result<TableBudget, PbError> {
    let on_overflow = match overflow.map(str::to_ascii_lowercase).as_deref() {
        // DEFAULT: factor over-budget order-3 effects (exact, §08.10) — the decomposition no
        // longer hard-fails at competitive tree counts. `error`/`sparse` remain opt-in.
        None | Some("factored") | Some("factor") => OverflowPolicy::Factored,
        Some("error") | Some("hard") => OverflowPolicy::Error,
        Some("sparse") | Some("sparse_fallback") => OverflowPolicy::SparseFallback {
            density_threshold: 0.05,
        },
        Some(other) => {
            return Err(PbError::InvalidConfig {
                what: format!("overflow must be 'factored', 'error', or 'sparse', got `{other}`"),
            })
        }
    };
    Ok(TableBudget {
        on_overflow,
        ..TableBudget::default()
    })
}

fn parse_rating_basis(value: Option<&str>) -> PyResult<Option<RatingBasis>> {
    value
        .map(|s| {
            serde_json::from_str::<RatingBasis>(s).map_err(|err| {
                py_err(PbError::Serialization(format!(
                    "could not parse RatingBasis JSON: {err}"
                )))
            })
        })
        .transpose()
}

fn py_err(err: PbError) -> PyErr {
    match err {
        PbError::InvariantViolated { invariant } => {
            InvariantError::new_err(invariant_message(invariant))
        }
        PbError::ExactnessFirewall(reason) => ExactnessError::new_err(reason),
        PbError::Serialization(message) => SerializationError::new_err(message),
        PbError::Internal { what } => InternalError::new_err(what),
        other => TriBoostError::new_err(other.to_string()),
    }
}

fn invariant_message(invariant: Invariant) -> String {
    invariant.to_string()
}

/// The compiled extension module `tri_boost._tri_boost` (§02.7).
#[pymodule]
fn _tri_boost(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("TriBoostError", py.get_type::<TriBoostError>())?;
    m.add("InvariantError", py.get_type::<InvariantError>())?;
    m.add("ExactnessError", py.get_type::<ExactnessError>())?;
    m.add("SerializationError", py.get_type::<SerializationError>())?;
    m.add("InternalError", py.get_type::<InternalError>())?;
    m.add_class::<PyBooster>()?;
    m.add_class::<PyModel>()?;
    m.add_class::<PyTableBank>()?;
    Ok(())
}
