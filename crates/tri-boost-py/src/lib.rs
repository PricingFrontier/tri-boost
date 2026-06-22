//! PyO3 bindings for tri-boost (spec §12).
//!
//! `#![allow(unsafe_code)]` is required here (and is the single justified,
//! encapsulated exception to the core's `forbid`): the pyo3/numpy procedural macros
//! expand to `unsafe`. The pure Rust core carries `#![forbid(unsafe_code)]`; this crate
//! remains a thin FFI adapter and owns no model math.
#![allow(unsafe_code)]

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;
use tri_boost_core::constraints::{InteractionPolicy, MonotoneMap};
use tri_boost_core::data::{bin, bin_columns, BinConfig, BinnedMatrix};
use tri_boost_core::engine::{Booster, Config, FitSpec, Model, Sampling};
use tri_boost_core::error::{Invariant, PbError};
use tri_boost_core::explain::{RefMeasure, TableBank};
use tri_boost_core::loss::{Gamma, Logistic, Loss, Poisson, SquaredError, Tweedie};

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
    seed: u64,
    n_jobs: Option<usize>,
}

#[pymethods]
impl PyBooster {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        n_trees=1000,
        learning_rate=0.05,
        lambda_=1.0,
        min_split_gain=0.0,
        max_delta_step=None,
        max_bin=254,
        objective=None,
        tweedie_rho=1.5,
        seed=0,
        n_jobs=None
    ))]
    fn new(
        n_trees: u32,
        learning_rate: f32,
        lambda_: f32,
        min_split_gain: f32,
        max_delta_step: Option<f32>,
        max_bin: u8,
        objective: Option<String>,
        tweedie_rho: f32,
        seed: u64,
        n_jobs: Option<usize>,
    ) -> PyResult<Self> {
        let config = Config {
            n_trees,
            learning_rate,
            lambda: lambda_,
            min_split_gain,
            max_delta_step,
            sampling: Sampling::Full,
            hist_precision: Default::default(),
        };
        config.validate().map_err(py_err)?;
        let bin_config = BinConfig {
            max_bin,
            ..BinConfig::default()
        };
        bin_config.validate().map_err(py_err)?;
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
            seed,
            n_jobs,
        })
    }

    #[pyo3(signature = (x, y, weight=None, exposure=None, feature_names=None, class_labels=None))]
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
    ) -> PyResult<PyModel> {
        let columns = raw_columns_from_array(x)?;
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

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let pred = py
            .detach(move || {
                let binned = binned_for_model(&model, columns)?;
                model.predict_binned(&binned, None)
            })
            .map_err(py_err)?;
        Ok(pred.into_pyarray(py))
    }

    fn predict_raw<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let raw = py
            .detach(move || {
                let binned = binned_for_model(&model, columns)?;
                let mut out = vec![0.0_f32; binned.n_rows as usize];
                model.score_trees(&binned, None, &mut out)?;
                Ok::<Vec<f32>, PbError>(out)
            })
            .map_err(py_err)?;
        Ok(raw.into_pyarray(py))
    }

    #[pyo3(signature = (x, ref_measure=None, laplace=1.0))]
    fn explain(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        ref_measure: Option<String>,
        laplace: f32,
    ) -> PyResult<PyTableBank> {
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let w = parse_ref_measure(ref_measure, laplace).map_err(py_err)?;
        let bank = py
            .detach(move || {
                let binned = binned_for_model(&model, columns)?;
                model.explain(&tri_boost_core::data::ServeBinnedMatrix(binned), w)
            })
            .map_err(py_err)?;
        Ok(PyTableBank { bank })
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
        self.bank.tables.len()
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

fn fit_owned(
    state: PyBooster,
    columns: Vec<Vec<f32>>,
    y: Vec<f32>,
    weight: Option<Vec<f32>>,
    exposure: Option<Vec<f32>>,
    feature_names: Option<Vec<String>>,
    class_labels: Option<Vec<String>>,
) -> Result<Model, PbError> {
    let refs: Vec<&[f32]> = columns.iter().map(Vec::as_slice).collect();
    let x = bin_columns(&refs, weight.as_deref(), &state.bin_config, state.seed)?;
    let run = || {
        let loss = state.objective.instantiate()?;
        let spec = FitSpec {
            loss: loss.as_loss(),
            weight: weight.as_deref(),
            exposure: exposure.as_deref(),
            monotone: MonotoneMap::new(),
            interaction: InteractionPolicy::default(),
            seed: state.seed,
        };
        Booster::with_config(state.config.clone()).fit(&x, &y, &spec)
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
    if n_features == 0 {
        return Err(py_err(PbError::InvalidInput {
            what: "x must contain at least one feature".into(),
        }));
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
