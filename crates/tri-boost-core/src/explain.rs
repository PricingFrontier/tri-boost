//! The explainability engine (spec §2.7 / §08): the trained [`Model`] turned into the
//! [`TableBank`] that **is** the model in a second view, with the equality of the two
//! views enforced as a build gate (the five I2 checks).
//!
//! The pipeline is `accumulate → build_weights → purify → TableBank`, all on a
//! per-raw-feature **merged grid** (the sorted union of every split border realized on
//! that feature across the ensemble, plus an explicit missing cell at index 0 —
//! [`MergedGrids`], R-MERGEDCELL of §08.1). [`Model::explain`] runs the whole pipeline
//! and then the five build-blocking checks:
//!
//! 1. [`check_reconstruction`] — `F_ens == f0 + Σ_u f_u` at every merged-grid cell.
//! 2. `check_mass_conservation` — the purified tables hold zero net `w`-mass (it all
//!    lives in the intercept).
//! 3. `check_purity` — every axis-slice of every table has `w`-weighted mean zero
//!    (maps to [`Invariant::Decomposability`]).
//! 4. `check_variance_sum` — `σ²(F) == Σ_u σ²(f_u)` under product/uniform `w`.
//! 5. [`check_three_way_equal`] — tree-sum = table-sum = Shapley-sum.
//!
//! Plus the I1 [`check_feature_budget`]. The merged-grid missing cell honors each
//! tree's learned `Split.missing_left` exactly, via the SINGLE canonical
//! [`crate::engine`] `low_bit` routing rule — which is what makes tree-sum equal
//! table-sum (and so makes the gates pass rather than merely be asserted).
//!
//! v1 scope: numeric axes only (one axis per raw feature; categoricals arrive with
//! §04), the `ProductMarginals`/`Uniform` reference measures (single-pass purify, exact
//! variance-sum), and exact `Error`/`SparseFallback` table-budget policies. `Joint` is
//! rejected up front rather than silently mishandled.

use crate::data::{BorderGrid, FeatureId, ServeBinnedMatrix};
use crate::engine::{low_bit, Model};
use crate::error::{Invariant, PbError};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

/// Hard ceiling on exhaustive joint-grid enumeration for the reconstruction /
/// variance / three-way checks. Below it the sweep is exhaustive (one interior point
/// per joint cell, spec §08.6); above it the checks sample a deterministic subset
/// (the release-mode behavior of §08.8). MassConservation never enumerates the joint
/// grid (it integrates exactly per tree); VarianceSum self-normalizes and widens its
/// tolerance when sampled; Reconstruction/ThreeWayEqual are per-point max checks, sound
/// under sampling. Most models stay under the cap and run exhaustively.
const JOINT_CAP: usize = 1 << 20;

/// Relative tolerance band for the SAMPLED VarianceSum estimator (a >`JOINT_CAP`-cell
/// joint grid): the self-normalized Monte-Carlo variance carries `O(1/√N)` sampling
/// error, so for such large models VarianceSum is a statistical certification, not a
/// bit-exact one. `5%` is comfortably above the realized sampling error at `N = JOINT_CAP`.
const SAMPLE_VAR_REL: f64 = 0.05;

#[cfg(test)]
thread_local! {
    /// Test-only override for the joint-grid exhaustion cap, so the sampling branch can
    /// be forced on a small model without fitting a multi-million-cell ensemble. `0`
    /// means "use [`JOINT_CAP`]". Thread-local, so parallel tests do not interfere.
    static TEST_JOINT_CAP: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The effective exhaustion cap ([`JOINT_CAP`], or a test override).
fn joint_cap() -> usize {
    #[cfg(test)]
    {
        let t = TEST_JOINT_CAP.with(std::cell::Cell::get);
        if t > 0 {
            return t;
        }
    }
    JOINT_CAP
}

// ===========================================================================
// §08.1 — Local aliases: the merged-grid axis and the effect tensor.
// ===========================================================================

/// A merged-grid axis (spec §08.1). `borders` is the sorted union of realized split
/// borders on one raw feature (the FINITE breakpoints); `cells` is the per-axis tensor
/// extent `== borders.len() + 2` — one EXPLICIT missing cell at index 0 PLUS the
/// `borders.len() + 1` finite half-open interval cells. The missing cell mirrors bin 0
/// of the underlying [`BorderGrid`] (§03) so a tree's learned `missing_left` routing is
/// representable losslessly rather than collapsed into the first finite interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisId {
    /// The raw feature this axis decomposes (keyed off `AxisProvenance.raw`).
    pub raw: FeatureId,
    /// Sorted finite realized split borders (subset of the model grid's borders).
    pub borders: Vec<f32>,
    /// Per-axis tensor extent `== borders.len() + 2` (missing cell + finite cells).
    pub cells: u32,
}

/// A dense row-major n-dimensional tensor of `f64` values (§08-local). Used for an
/// [`EffectTable`]'s purified `values` and its per-cell `support`. `f64` even though the
/// core trains in `f32`: purification accumulates many signed mass-moves and we want the
/// reconstruction residual at `f64` epsilon, not `f32` epsilon (§08.1).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Tensor {
    // Per-axis extents as fixed-width ints: this type is serialized inside an
    // EffectTable, and a serialized `usize` would differ between the host and the
    // wasm32 smoke build, breaking cross-platform byte-equality (spec §02.8).
    // Cell-count dimensions are tiny, so `u32` is ample.
    shape: Vec<u32>,
    data: TensorData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum TensorData {
    Dense(Vec<f64>),
    Sparse(Vec<SparseEntry>),
}

impl Default for TensorData {
    fn default() -> Self {
        TensorData::Dense(Vec::new())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct SparseEntry {
    index: u64,
    value: f64,
}

fn checked_shape(shape: &[usize]) -> Result<(Vec<u32>, usize), PbError> {
    let mut shape_u32 = Vec::with_capacity(shape.len());
    let mut cells = 1usize;
    for (axis, &dim) in shape.iter().enumerate() {
        if dim == 0 {
            return Err(PbError::InvalidInput {
                what: format!("tensor axis {axis} has zero extent"),
            });
        }
        shape_u32.push(u32::try_from(dim).map_err(|_| PbError::InvalidInput {
            what: format!("tensor axis {axis} extent {dim} exceeds u32"),
        })?);
        cells = cells.checked_mul(dim).ok_or_else(|| PbError::Internal {
            what: "tensor shape overflows usize".into(),
        })?;
    }
    Ok((shape_u32, cells))
}

fn checked_shape_cells_u64(shape: &[u32]) -> Result<u64, PbError> {
    let mut cells = 1u64;
    for &dim in shape {
        if dim == 0 {
            return Err(PbError::InvalidInput {
                what: "tensor has a zero extent".into(),
            });
        }
        cells = cells
            .checked_mul(u64::from(dim))
            .ok_or_else(|| PbError::Internal {
                what: "tensor shape overflows u64".into(),
            })?;
    }
    Ok(cells)
}

fn filled_data(cells: usize, value: f64) -> Result<Vec<f64>, PbError> {
    let mut data = Vec::new();
    data.try_reserve_exact(cells)
        .map_err(|_| PbError::Internal {
            what: "tensor allocation failed".into(),
        })?;
    data.resize(cells, value);
    Ok(data)
}

impl Tensor {
    /// Try to build a zero tensor of the given per-axis extents. 0-D — an empty
    /// shape — is one scalar cell.
    ///
    /// # Errors
    /// [`PbError::InvalidInput`] if an extent is zero or exceeds `u32`;
    /// [`PbError::Internal`] if shape arithmetic overflows or allocation fails.
    pub fn try_zeros(shape: Vec<usize>) -> Result<Self, PbError> {
        let (shape, cells) = checked_shape(&shape)?;
        Ok(Self {
            data: TensorData::Dense(filled_data(cells, 0.0)?),
            shape,
        })
    }

    /// Try to build a sparse zero tensor of the given per-axis extents.
    ///
    /// # Errors
    /// [`PbError::InvalidInput`] if an extent is zero or exceeds `u32`;
    /// [`PbError::Internal`] if shape arithmetic overflows.
    pub fn try_sparse_zeros(shape: Vec<usize>) -> Result<Self, PbError> {
        let (shape, _) = checked_shape(&shape)?;
        Ok(Self {
            data: TensorData::Sparse(Vec::new()),
            shape,
        })
    }

    /// Try to build a constant-`value` tensor of the given extents.
    ///
    /// # Errors
    /// [`PbError::InvalidInput`] if an extent is zero or exceeds `u32`;
    /// [`PbError::Internal`] if shape arithmetic overflows or allocation fails.
    pub fn try_filled(shape: Vec<usize>, value: f64) -> Result<Self, PbError> {
        let (shape, cells) = checked_shape(&shape)?;
        Ok(Self {
            data: TensorData::Dense(filled_data(cells, value)?),
            shape,
        })
    }

    /// Build from an explicit row-major buffer.
    ///
    /// # Errors
    /// [`PbError::InvalidInput`] if an extent is zero or exceeds `u32`;
    /// [`PbError::Internal`] if shape arithmetic overflows;
    /// [`PbError::ShapeMismatch`] if `data.len()` does not equal the product of `shape`.
    pub fn from_vec(shape: Vec<usize>, data: Vec<f64>) -> Result<Self, PbError> {
        let (shape, n) = checked_shape(&shape)?;
        if data.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!("tensor data len {} != product(shape) {n}", data.len()),
            });
        }
        Ok(Self {
            shape,
            data: TensorData::Dense(data),
        })
    }

    /// The tensor's per-axis extents (as `usize` for indexing).
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        self.shape.iter().map(|&d| d as usize).collect()
    }

    /// The tensor's per-axis extents as fixed-width serialized dimensions.
    #[must_use]
    pub fn shape_u32(&self) -> &[u32] {
        &self.shape
    }

    /// Total number of cells.
    #[must_use]
    pub fn len(&self) -> usize {
        checked_shape_cells_u64(&self.shape)
            .ok()
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(0)
    }

    /// `true` if the tensor has no cells.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The row-major values. Dense tensors borrow their backing slice; sparse tensors
    /// materialize an owned dense view for display/export compatibility.
    #[must_use]
    pub fn values(&self) -> Cow<'_, [f64]> {
        match &self.data {
            TensorData::Dense(data) => Cow::Borrowed(data),
            TensorData::Sparse(_) => Cow::Owned(self.dense_values().unwrap_or_default()),
        }
    }

    /// `true` if this tensor uses sparse backing storage.
    #[must_use]
    pub fn is_sparse(&self) -> bool {
        matches!(self.data, TensorData::Sparse(_))
    }

    fn sparse_nnz(&self) -> Option<usize> {
        match &self.data {
            TensorData::Dense(_) => None,
            TensorData::Sparse(entries) => Some(entries.len()),
        }
    }

    /// Add a constant to every cell. Used by rating-view re-basing, which moves an
    /// equal and opposite constant into the exported intercept and therefore preserves
    /// every reconstructed score exactly.
    pub fn add_scalar(&mut self, delta: f64) {
        match &mut self.data {
            TensorData::Dense(data) => {
                for v in data {
                    *v += delta;
                }
            }
            TensorData::Sparse(_) => {
                if let Ok(mut dense) = self.dense_values() {
                    for v in &mut dense {
                        *v += delta;
                    }
                    self.data = TensorData::Dense(dense);
                }
            }
        }
    }

    fn offset(&self, coord: &[usize]) -> Option<u64> {
        if coord.len() != self.shape.len() {
            return None;
        }
        let mut off = 0u64;
        for (c, dim) in coord.iter().zip(self.shape.iter()) {
            let dim_u64 = u64::from(*dim);
            let c_u64 = u64::try_from(*c).ok()?;
            if c_u64 >= dim_u64 {
                return None;
            }
            off = off.checked_mul(dim_u64)?.checked_add(c_u64)?;
        }
        Some(off)
    }

    fn dense_values(&self) -> Result<Vec<f64>, PbError> {
        let cells_u64 = checked_shape_cells_u64(&self.shape)?;
        let cells = usize::try_from(cells_u64).map_err(|_| PbError::Internal {
            what: "tensor dense view exceeds usize".into(),
        })?;
        let mut dense = filled_data(cells, 0.0)?;
        match &self.data {
            TensorData::Dense(data) => {
                if data.len() != cells {
                    return Err(PbError::ShapeMismatch {
                        what: "tensor dense backing length does not match shape".into(),
                    });
                }
                dense = data.clone();
            }
            TensorData::Sparse(entries) => {
                for entry in entries {
                    let idx = usize::try_from(entry.index).map_err(|_| PbError::Internal {
                        what: "sparse tensor index exceeds usize".into(),
                    })?;
                    let slot = dense.get_mut(idx).ok_or_else(|| PbError::ShapeMismatch {
                        what: "sparse tensor index outside shape".into(),
                    })?;
                    *slot = entry.value;
                }
            }
        }
        Ok(dense)
    }

    /// Read the value at `coord`, or `None` if out of range / wrong rank.
    #[must_use]
    pub fn at(&self, coord: &[usize]) -> Option<f64> {
        let off = self.offset(coord)?;
        match &self.data {
            TensorData::Dense(data) => usize::try_from(off).ok().and_then(|o| data.get(o).copied()),
            TensorData::Sparse(entries) => entries
                .binary_search_by_key(&off, |entry| entry.index)
                .ok()
                .and_then(|pos| entries.get(pos).map(|entry| entry.value))
                .or(Some(0.0)),
        }
    }

    /// Write `value` at `coord`.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `coord` is out of range or the wrong rank.
    pub fn set(&mut self, coord: &[usize], value: f64) -> Result<(), PbError> {
        let off = self.offset(coord).ok_or_else(|| PbError::ShapeMismatch {
            what: "tensor set coord out of range".into(),
        })?;
        match &mut self.data {
            TensorData::Dense(data) => {
                let off = usize::try_from(off).map_err(|_| PbError::Internal {
                    what: "tensor offset exceeds usize".into(),
                })?;
                let slot = data.get_mut(off).ok_or_else(|| PbError::Internal {
                    what: "tensor offset escaped buffer".into(),
                })?;
                *slot = value;
            }
            TensorData::Sparse(entries) => match entries.binary_search_by_key(&off, |e| e.index) {
                Ok(pos) => {
                    if value == 0.0 {
                        entries.remove(pos);
                    } else if let Some(entry) = entries.get_mut(pos) {
                        entry.value = value;
                    }
                }
                Err(pos) => {
                    if value != 0.0 {
                        entries.try_reserve(1).map_err(|_| PbError::Internal {
                            what: "sparse tensor allocation failed".into(),
                        })?;
                        entries.insert(pos, SparseEntry { index: off, value });
                    }
                }
            },
        }
        Ok(())
    }

    /// Add `delta` to the value at `coord`.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `coord` is out of range or the wrong rank.
    pub fn add(&mut self, coord: &[usize], delta: f64) -> Result<(), PbError> {
        let value = self.at(coord).ok_or_else(|| PbError::ShapeMismatch {
            what: "tensor add coord out of range".into(),
        })?;
        self.set(coord, value + delta)
    }
}

/// A set of 0..=3 distinct, sorted raw feature ids identifying one effect (spec §2.7).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct FeatureSet(pub SmallVec<[FeatureId; 3]>);

impl FeatureSet {
    /// Build a feature set from raw ids (caller ensures distinct/sorted).
    #[must_use]
    pub fn new(ids: &[u32]) -> Self {
        FeatureSet(ids.iter().map(|&i| FeatureId(i)).collect())
    }

    /// The interaction order `|u|` (1 = main effect, 2 = pairwise, 3 = triple).
    #[must_use]
    pub fn order(&self) -> usize {
        self.0.len()
    }

    /// `true` if `f` is a member of this feature set.
    #[must_use]
    pub fn contains(&self, f: FeatureId) -> bool {
        self.0.contains(&f)
    }
}

/// Per-cell standard-error bands for bagged/averaged rating-table displays (§09.5).
/// This is display-only metadata: invariant checks and inference never read it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeBand {
    /// Per-cell standard error, parallel to [`EffectTable::values`].
    pub per_cell: Tensor,
}

/// One purified effect tensor for feature set `u`, on the merged grid (spec §2.7).
/// `support` and `se_band` are display metadata — excluded from the five invariant
/// checks and from inference (scoring reads `values` only).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectTable {
    /// The raw feature set this effect is over.
    pub u: FeatureSet,
    /// The merged-grid axes (parallel to `values`' dimensions).
    pub axes: Vec<AxisId>,
    /// The purified effect values (one cell per merged-grid cell).
    pub values: Tensor,
    /// Per-cell training-row count (display-only; same extents as `values`).
    pub support: Tensor,
    /// Optional per-cell standard-error band, display-only.
    #[serde(default)]
    pub se_band: Option<SeBand>,
    /// `w`-weighted variance of this effect, `σ²(f_u)`.
    pub variance: f64,
}

impl EffectTable {
    /// Evaluate this effect at a row given its per-raw-feature merged-cell ids
    /// (`x_cells[raw] = cell`). The tensor coordinate is read off `axes[k].raw`.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `x_cells` lacks one of this table's axes;
    /// [`PbError::Internal`] if the projected coordinate escapes the tensor.
    pub fn eval(&self, x_cells: &[u32]) -> Result<f64, PbError> {
        let mut coord = Vec::with_capacity(self.axes.len());
        for a in &self.axes {
            let cell = *x_cells
                .get(a.raw.0 as usize)
                .ok_or_else(|| PbError::ShapeMismatch {
                    what: format!("x_cells missing raw feature {} for table eval", a.raw.0),
                })?;
            coord.push(cell as usize);
        }
        self.values.at(&coord).ok_or_else(|| PbError::Internal {
            what: "effect-table coordinate out of range".into(),
        })
    }
}

/// The reference measure for purification (spec §2.7 / §08.4). Default = Laplace-
/// smoothed empirical product-of-marginals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RefMeasure {
    /// Product of per-axis Laplace-smoothed empirical marginals (DEFAULT; `laplace > 0`).
    /// Per cell `ŵ ∝ ŵ_unif + laplace · ŵ_emp` (§08.4), strictly positive (so empty
    /// merged cells never break zero-mean or single-pass convergence).
    ProductMarginals {
        /// Laplace smoothing weight on the empirical marginal.
        laplace: f32,
    },
    /// Uniform over realized cells (`ŵ ∝ 1` per cell).
    Uniform,
    /// Hooker hierarchical-orthogonality joint measure — a v1.5 fork (couples axes,
    /// breaks the variance-sum identity). Rejected by [`Model::explain`] in v1.
    Joint,
}

impl Default for RefMeasure {
    fn default() -> Self {
        RefMeasure::ProductMarginals { laplace: 1.0 }
    }
}

/// The complete decomposition (spec §2.7): intercept + all purified tables on the
/// shared merged grid. `tables` is the lossless inference support; display pruning is a
/// view. `merged_grids` is indexed by raw feature id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableBank {
    /// The intercept term (the `w`-weighted mean of the ensemble).
    pub f0: f64,
    /// Every realized effect `u` of size `1..=3`, plus the lower-order tables the
    /// purification cascade generates (the down-set closure of realized supports).
    pub tables: Vec<EffectTable>,
    /// Per-raw-feature merged grid (sorted union of realized borders + missing cell).
    pub merged_grids: Vec<BorderGrid>,
    /// The reference measure stamped on the bank and every export.
    pub w: RefMeasure,
    /// Over-budget order-3 effects kept in factored (per-tree-box) form — never densified
    /// (§08.10). Empty for budget-fitting banks; each carries the same `f_u` a dense
    /// [`EffectTable`] would, for score/shap/sobol and the five I2 gates.
    #[serde(default)]
    pub factored: Vec<FactoredTriple>,
}

/// Tolerances for the I2 checks (spec §13.1). `recon_tol` is the canonical
/// `4 · n_trees · f32::EPSILON`; the others track it (variance is squared-scale, so it
/// gets headroom).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExactTol {
    /// Per-cell reconstruction tolerance.
    pub recon_tol: f64,
    /// Mass-conservation tolerance.
    pub mass_tol: f64,
    /// Per-slice purity tolerance.
    pub purity_tol: f64,
    /// Variance-sum tolerance (squared-scale).
    pub var_tol: f64,
}

impl ExactTol {
    /// The tolerances for `model`: `recon_tol = 4 · n_trees · f32::EPSILON` (spec §13.1).
    #[must_use]
    pub fn for_model(model: &Model) -> Self {
        let n_trees = model.trees.len().max(1) as f64;
        let base = 4.0 * n_trees * f64::from(f32::EPSILON);
        ExactTol {
            recon_tol: base,
            mass_tol: base,
            purity_tol: base,
            var_tol: 16.0 * base,
        }
    }
}

/// Per-table and whole-bank cell budgets (spec §08.10, the memory firewall). Counted on
/// the realized merged (union) grid (R-TABLEBUDGET), checked at lazy allocation so an
/// over-budget table either fails before dense allocation or uses the explicit sparse
/// policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TableBudget {
    /// Per-[`EffectTable`] `Π cells_i` ceiling.
    pub max_table_cells: u64,
    /// `Σ` over all tables ceiling.
    pub max_bank_cells: u64,
    /// What to do when a table would exceed `max_table_cells`.
    pub on_overflow: OverflowPolicy,
}

impl Default for TableBudget {
    fn default() -> Self {
        Self {
            max_table_cells: 2_000_000,
            max_bank_cells: 32_000_000,
            on_overflow: OverflowPolicy::Factored,
        }
    }
}

/// The resolution when a table would exceed [`TableBudget::max_table_cells`] (§08.10).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverflowPolicy {
    /// Hard error — refuse to build a bank that would exceed the budget
    /// ([`PbError::TableBudget`]). No silent truncation.
    Error,
    /// EXACT sparse-tensor storage for over-budget tables. This preserves the logical
    /// tensor shape and all I2 checks while avoiding dense allocation for cold cells.
    SparseFallback {
        /// Maximum realized nonzero occupancy allowed for sparse storage.
        density_threshold: f64,
    },
    /// Keep an over-budget order-3 effect in FACTORED per-tree-box form (§08.10) — exact,
    /// never densified. Only order-3 supports can exceed the cell budget (order-2 ≤ 254²),
    /// so this is the order-3 escape hatch: the residual stays factored and its order-≤2
    /// mass is shed into the dense lower tables.
    Factored,
}

/// The purification convergence mode (spec §08.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PurifyMode {
    /// Iterate to a fixpoint (for joint `w`, where slice masses couple across axes).
    ToFixpoint {
        /// Mass-move tolerance.
        tol: f64,
        /// Iteration cap before returning a non-convergence error.
        max_iter: u32,
    },
    /// One pass per axis — exact for axis-factorized `w` (product/uniform), §08.3.
    SinglePass,
}

// ===========================================================================
// §08.1 — The merged grids: per-raw-feature sorted union of realized borders.
// ===========================================================================

/// One raw feature's merged axis: the realized split borders (a subset of the model
/// grid's borders, recorded by both value and model-border index) plus the cell maps.
#[derive(Debug, Clone)]
struct MergedAxis {
    axis: usize,
    borders: Vec<f32>,
    model_border_index: Vec<usize>,
    model_n_bins: u16,
}

impl MergedAxis {
    fn cells(&self) -> usize {
        self.borders.len() + 2
    }

    /// Merged cell → a representative model bin that routes identically under `low_bit`
    /// for every split on this feature (cell boundaries ⊇ split borders, §08.1).
    fn rep_model_bin(&self, cell: usize) -> Result<u8, PbError> {
        let bin = match cell {
            0 => 0usize,
            1 => 1usize,
            c => {
                let k = *self
                    .model_border_index
                    .get(c - 2)
                    .ok_or_else(|| PbError::Internal {
                        what: "merged cell escaped border index".into(),
                    })?;
                k + 2
            }
        };
        u8::try_from(bin).map_err(|_| PbError::Internal {
            what: "representative model bin exceeded u8".into(),
        })
    }

    /// Model bin → its merged cell. Uses only the merged border indices (a model bin
    /// `b` sits above merged border index `k` iff `k <= b - 2`).
    fn model_bin_to_cell(&self, bin: u8) -> Result<u32, PbError> {
        if u16::from(bin) >= self.model_n_bins {
            return Err(PbError::InvalidInput {
                what: format!(
                    "model bin {bin} outside grid n_bins {} for axis {}",
                    self.model_n_bins, self.axis
                ),
            });
        }
        if bin == 0 {
            return Ok(0);
        }
        // Count merged borders strictly below bin `b` (i.e. model index <= b-2).
        let threshold = i64::from(bin) - 2;
        let below = self
            .model_border_index
            .iter()
            .filter(|&&k| (k as i64) <= threshold)
            .count();
        u32::try_from(below + 1).map_err(|_| PbError::Internal {
            what: "merged cell id exceeded u32".into(),
        })
    }
}

/// The merged grids of a fitted model: per raw feature, the sorted union of every
/// realized split border, with the §08.1 cell convention (cell 0 = missing).
#[derive(Debug, Clone)]
pub(crate) struct MergedGrids {
    per_raw: Vec<MergedAxis>,
}

impl MergedGrids {
    /// Build the merged grids from a fitted `model`. v1: requires a 1-to-1 raw↔axis
    /// mapping whose raw ids fill `0..n_features` (the green-spine invariant). Both
    /// [`AxisKind::Numeric`] and [`AxisKind::CategoricalTS`] axes are accepted — a
    /// categorical is a Target-Statistic-encoded ordinal axis carrying a normal
    /// [`BorderGrid`] (§04), so the merged-grid logic and the five I2 checks apply to it
    /// identically (the bank is audited on a `ServeBinnedMatrix` re-encoded through the
    /// frozen full-data encoders, R-CATSERVE). The reserved [`AxisKind::Missing`] axis is
    /// not a model feature and is rejected; many-to-one provenance is unsupported in v1.
    pub(crate) fn from_model(model: &Model) -> Result<Self, PbError> {
        use crate::data::AxisKind;
        let n_features = model.provenance.len();
        // axis_of_raw[raw] = the single model axis carrying that raw feature.
        let mut axis_of_raw: Vec<Option<usize>> = vec![None; n_features];
        for (a, prov) in model.provenance.iter().enumerate() {
            if matches!(prov.kind, AxisKind::Missing) {
                return Err(PbError::InvalidConfig {
                    what: "explain does not support a standalone Missing axis kind".into(),
                });
            }
            let r = prov.raw.0 as usize;
            let slot = axis_of_raw
                .get_mut(r)
                .ok_or_else(|| PbError::InvalidConfig {
                    what: format!("raw feature id {r} out of range 0..{n_features} (v1 explain)"),
                })?;
            if slot.is_some() {
                return Err(PbError::InvalidConfig {
                    what: format!("raw feature {r} maps to >1 axis (unsupported in v1 explain)"),
                });
            }
            *slot = Some(a);
        }

        // Gather realized split borders per raw feature (dedup by model-border index).
        let mut border_indices: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n_features];
        for (_, tree) in &model.trees {
            for split in &tree.splits {
                let a = split.axis as usize;
                let prov = model.provenance.get(a).ok_or_else(|| PbError::Internal {
                    what: format!("split axis {a} absent from provenance"),
                })?;
                let r = prov.raw.0 as usize;
                if split.bin_le == 0 {
                    return Err(PbError::Internal {
                        what: "split bin_le 0 has no interior border".into(),
                    });
                }
                let bidx = usize::from(split.bin_le) - 1;
                let grid = model.grids.get(a).ok_or_else(|| PbError::Internal {
                    what: format!("split axis {a} absent from grids"),
                })?;
                if bidx >= grid.borders.len() {
                    return Err(PbError::Internal {
                        what: format!("split bin_le {} escapes grid borders", split.bin_le),
                    });
                }
                let set = border_indices.get_mut(r).ok_or_else(|| PbError::Internal {
                    what: "raw feature escaped border-index table".into(),
                })?;
                set.insert(bidx);
            }
        }

        let mut per_raw = Vec::with_capacity(n_features);
        for r in 0..n_features {
            let axis = axis_of_raw
                .get(r)
                .copied()
                .flatten()
                .ok_or_else(|| PbError::Internal {
                    what: format!("raw feature {r} has no axis"),
                })?;
            let grid = model.grids.get(axis).ok_or_else(|| PbError::Internal {
                what: "merged-grid axis absent from model grids".into(),
            })?;
            if grid.n_bins == 0 {
                return Err(PbError::InvalidInput {
                    what: format!("model grid {axis} has n_bins=0"),
                });
            }
            let idxs = border_indices.get(r).ok_or_else(|| PbError::Internal {
                what: "raw feature escaped border-index table".into(),
            })?;
            let mut model_border_index = Vec::with_capacity(idxs.len());
            let mut borders = Vec::with_capacity(idxs.len());
            for &k in idxs {
                let b = *grid.borders.get(k).ok_or_else(|| PbError::Internal {
                    what: "merged border index escaped grid borders".into(),
                })?;
                model_border_index.push(k);
                borders.push(b);
            }
            per_raw.push(MergedAxis {
                axis,
                borders,
                model_border_index,
                model_n_bins: grid.n_bins,
            });
        }
        Ok(MergedGrids { per_raw })
    }

    fn n_features(&self) -> usize {
        self.per_raw.len()
    }

    fn axis(&self, raw: FeatureId) -> Result<&MergedAxis, PbError> {
        self.per_raw
            .get(raw.0 as usize)
            .ok_or_else(|| PbError::Internal {
                what: format!("merged grid missing raw feature {}", raw.0),
            })
    }

    fn cells(&self, raw: FeatureId) -> Result<usize, PbError> {
        Ok(self.axis(raw)?.cells())
    }

    fn axis_id(&self, raw: FeatureId) -> Result<AxisId, PbError> {
        let ma = self.axis(raw)?;
        Ok(AxisId {
            raw,
            borders: ma.borders.clone(),
            cells: u32::try_from(ma.cells()).map_err(|_| PbError::Internal {
                what: "merged cell count exceeded u32".into(),
            })?,
        })
    }

    /// One [`BorderGrid`] per raw feature (the bank's shared merged grid).
    fn border_grids(&self) -> Result<Vec<BorderGrid>, PbError> {
        let mut out = Vec::with_capacity(self.per_raw.len());
        for ma in &self.per_raw {
            let n_bins = u16::try_from(ma.cells()).map_err(|_| PbError::Internal {
                what: "merged grid n_bins exceeded u16".into(),
            })?;
            out.push(BorderGrid {
                borders: ma.borders.clone(),
                n_bins,
                missing_bin: 0,
            });
        }
        Ok(out)
    }
}

// ===========================================================================
// §08.2 — Accumulation: ensemble → raw tensors.
// ===========================================================================

/// One realized tree-support tensor before purification.
struct RawTable {
    u: FeatureSet,
    axes: Vec<AxisId>,
    values: Tensor,
}

/// The raw (pre-purify) bank: intercept + per-support tensors keyed by support.
struct RawBank {
    f0: f64,
    tables: BTreeMap<FeatureSet, RawTable>,
}

/// One pre-purification effect supplied by another exactness-preserving layer.
pub(crate) struct RawEffect {
    /// The effect support.
    pub u: FeatureSet,
    /// Raw, uncentered score-space values on the supplied merged grids.
    pub values: Tensor,
    /// Display/support counts on the same tensor shape as `values`.
    pub support: Tensor,
}

/// The distinct raw-feature support of a tree (sorted, size = depth by I1).
fn tree_support(model: &Model, tree: &crate::engine::ObliviousTree) -> Result<FeatureSet, PbError> {
    let mut ids: SmallVec<[FeatureId; 3]> = SmallVec::new();
    for split in &tree.splits {
        let prov = model
            .provenance
            .get(split.axis as usize)
            .ok_or_else(|| PbError::Internal {
                what: format!("split axis {} absent from provenance", split.axis),
            })?;
        if !ids.contains(&prov.raw) {
            ids.push(prov.raw);
        }
    }
    ids.sort_unstable();
    Ok(FeatureSet(ids))
}

/// Walk every coordinate tuple of a tensor of `extents` (an in-place odometer; no
/// allocation of the full coordinate list). The empty-extents tensor is one cell.
fn walk_extents(
    extents: &[usize],
    mut f: impl FnMut(&[usize]) -> Result<(), PbError>,
) -> Result<(), PbError> {
    let mut idx = vec![0usize; extents.len()];
    loop {
        f(&idx)?;
        let mut k = extents.len();
        loop {
            if k == 0 {
                return Ok(());
            }
            k -= 1;
            let e = *extents.get(k).ok_or_else(|| PbError::Internal {
                what: "odometer axis escaped extents".into(),
            })?;
            let v = idx.get_mut(k).ok_or_else(|| PbError::Internal {
                what: "odometer index escaped buffer".into(),
            })?;
            *v += 1;
            if *v < e {
                break;
            }
            *v = 0;
        }
    }
}

/// Coordinate with position `p` dropped (no indexing).
fn drop_index(coord: &[usize], p: usize) -> Vec<usize> {
    coord
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| if i == p { None } else { Some(c) })
        .collect()
}

/// `Π extents` as `u64`, or `PbError::Internal` on overflow.
fn product_u64(extents: &[usize]) -> Result<u64, PbError> {
    let mut acc = 1u64;
    for &e in extents {
        acc = acc.checked_mul(e as u64).ok_or_else(|| PbError::Internal {
            what: "merged tensor cell count overflowed u64".into(),
        })?;
    }
    Ok(acc)
}

/// `Π extents` as `u64`, SATURATING to `u64::MAX` on overflow. Used where only the comparison
/// "does this exceed the enumeration cap?" matters — never an exact count. A joint grid too
/// large to represent (e.g. a 130-feature model → `2^130` cells) must route to the sampling
/// branch, not hard-error, so the gate sweep stays bounded on wide models (§08.6/§08.8).
fn saturating_product_u64(extents: &[usize]) -> u64 {
    let mut acc = 1u64;
    for &e in extents {
        acc = acc.saturating_mul(e as u64);
    }
    acc
}

/// Compute the leaf index a tree assigns to a merged cell-tuple of its support, routing
/// each split via the SINGLE canonical `low_bit` on the cell's representative model bin.
fn leaf_index_for_tuple(
    model: &Model,
    tree: &crate::engine::ObliviousTree,
    grids: &MergedGrids,
    u_ids: &[FeatureId],
    tuple: &[usize],
) -> Result<usize, PbError> {
    let mut leaf_idx = 0usize;
    for (level, split) in tree.splits.iter().enumerate() {
        let prov = model
            .provenance
            .get(split.axis as usize)
            .ok_or_else(|| PbError::Internal {
                what: "split axis absent from provenance".into(),
            })?;
        let pos = u_ids
            .iter()
            .position(|r| *r == prov.raw)
            .ok_or_else(|| PbError::Internal {
                what: "split raw feature absent from tree support".into(),
            })?;
        let cell = *tuple.get(pos).ok_or_else(|| PbError::Internal {
            what: "tuple shorter than support".into(),
        })?;
        let rep_bin = grids.axis(prov.raw)?.rep_model_bin(cell)?;
        let bit = usize::from(low_bit(rep_bin, split.bin_le, split.missing_left));
        leaf_idx |= bit << level;
    }
    Ok(leaf_idx)
}

/// Accumulate the ensemble into raw per-support tensors (spec §08.2). Each tree adds
/// `alpha · leaf` into every merged cell of its support; the table-budget firewall is
/// checked at lazy allocation.
fn accumulate(
    model: &Model,
    grids: &MergedGrids,
    budget: &TableBudget,
) -> Result<(RawBank, BTreeSet<FeatureSet>), PbError> {
    if let OverflowPolicy::SparseFallback { density_threshold } = budget.on_overflow {
        if !density_threshold.is_finite() || !(0.0..=1.0).contains(&density_threshold) {
            return Err(PbError::InvalidConfig {
                what: "SparseFallback density_threshold must be finite and in [0, 1]".into(),
            });
        }
    }
    let mut tables: BTreeMap<FeatureSet, RawTable> = BTreeMap::new();
    let mut bank_cells: u64 = 0;
    // Supports whose dense merged cube exceeds the budget under OverflowPolicy::Factored —
    // never materialized here; the pipeline sheds them per tree into the lower tables.
    let mut factored_supports: BTreeSet<FeatureSet> = BTreeSet::new();

    for (alpha, tree) in &model.trees {
        let u = tree_support(model, tree)?;
        if factored_supports.contains(&u) {
            continue;
        }
        let u_ids: Vec<FeatureId> = u.0.iter().copied().collect();
        if !tables.contains_key(&u) {
            let mut extents = Vec::with_capacity(u_ids.len());
            for r in &u_ids {
                extents.push(grids.cells(*r)?);
            }
            let table_cells = product_u64(&extents)?;
            let values = if table_cells > budget.max_table_cells {
                match budget.on_overflow {
                    OverflowPolicy::Error => {
                        return Err(PbError::TableBudget {
                            what: format!("table {u:?}"),
                            cells: table_cells,
                            budget: budget.max_table_cells,
                        });
                    }
                    OverflowPolicy::SparseFallback { .. } => {
                        Tensor::try_sparse_zeros(extents.clone())?
                    }
                    OverflowPolicy::Factored => {
                        if u.order() != 3 {
                            return Err(PbError::TableBudget {
                                what: format!("only order-3 effects can be factored; {u:?}"),
                                cells: table_cells,
                                budget: budget.max_table_cells,
                            });
                        }
                        factored_supports.insert(u.clone());
                        continue;
                    }
                }
            } else {
                bank_cells =
                    bank_cells
                        .checked_add(table_cells)
                        .ok_or_else(|| PbError::Internal {
                            what: "bank cell count overflowed u64".into(),
                        })?;
                if bank_cells > budget.max_bank_cells {
                    return Err(PbError::TableBudget {
                        what: "bank".into(),
                        cells: bank_cells,
                        budget: budget.max_bank_cells,
                    });
                }
                Tensor::try_zeros(extents.clone())?
            };
            let mut axes = Vec::with_capacity(u_ids.len());
            for r in &u_ids {
                axes.push(grids.axis_id(*r)?);
            }
            tables.insert(
                u.clone(),
                RawTable {
                    u: u.clone(),
                    axes,
                    values,
                },
            );
        }
        let table = tables.get_mut(&u).ok_or_else(|| PbError::Internal {
            what: "raw table vanished after insert".into(),
        })?;
        let extents = table.values.shape();
        let alpha = f64::from(*alpha);
        walk_extents(&extents, |tuple| {
            let leaf_idx = leaf_index_for_tuple(model, tree, grids, &u_ids, tuple)?;
            let leaf = *tree.leaves.get(leaf_idx).ok_or_else(|| PbError::Internal {
                what: "oblivious leaf index escaped 0..8".into(),
            })?;
            table.values.add(tuple, alpha * f64::from(leaf))
        })?;
    }

    if let OverflowPolicy::SparseFallback { density_threshold } = budget.on_overflow {
        for (u, table) in &tables {
            if let Some(nnz) = table.values.sparse_nnz() {
                let table_cells = product_u64(&table.values.shape())?;
                let density = nnz as f64 / table_cells as f64;
                if density > density_threshold {
                    return Err(PbError::TableBudget {
                        what: format!(
                            "sparse table {u:?} density {density:.6} exceeds threshold {density_threshold:.6}"
                        ),
                        cells: table_cells,
                        budget: budget.max_table_cells,
                    });
                }
            }
        }
    }

    Ok((
        RawBank {
            f0: f64::from(model.f0),
            tables,
        },
        factored_supports,
    ))
}

// ===========================================================================
// §08.4 — The reference measure `w` (per-axis cell mass).
// ===========================================================================

/// Per-axis merged-cell `w`-mass, precomputed from the data and the chosen
/// [`RefMeasure`]. Indexed by raw feature; each inner vector sums to 1 and is strictly
/// positive (so zero-mean/convergence never break). Built over a [`ServeBinnedMatrix`]
/// (frozen encoders, R-CATSERVE) so the audited mass matches the deployed model.
#[derive(Debug, Clone)]
pub(crate) struct WeightCache {
    per_axis: Vec<Vec<f64>>,
    #[allow(dead_code)] // stamped for provenance; the bank carries the canonical copy.
    kind: RefMeasure,
}

impl WeightCache {
    fn axis(&self, raw: FeatureId) -> Result<&[f64], PbError> {
        self.per_axis
            .get(raw.0 as usize)
            .map(Vec::as_slice)
            .ok_or_else(|| PbError::Internal {
                what: format!("weight cache missing raw feature {}", raw.0),
            })
    }
}

/// Build the per-axis cell weights (spec §08.4). `Joint` is rejected in v1 (it couples
/// axes and breaks the variance-sum identity).
fn build_weights(
    x: &ServeBinnedMatrix,
    grids: &MergedGrids,
    w: &RefMeasure,
) -> Result<WeightCache, PbError> {
    let laplace =
        match w {
            RefMeasure::Uniform => None,
            RefMeasure::ProductMarginals { laplace } => {
                if !laplace.is_finite() || *laplace <= 0.0 {
                    return Err(PbError::InvalidConfig {
                        what: "ProductMarginals laplace must be finite and > 0".into(),
                    });
                }
                Some(f64::from(*laplace))
            }
            RefMeasure::Joint => return Err(PbError::InvalidConfig {
                what:
                    "Joint reference measure is a v1.5 fork; v1 supports ProductMarginals/Uniform"
                        .into(),
            }),
        };

    let n_rows = x.0.n_rows as usize;
    let mut per_axis = Vec::with_capacity(grids.n_features());
    for ma in &grids.per_raw {
        let cells = ma.cells();
        let raw_w = match laplace {
            None => vec![1.0_f64 / cells as f64; cells],
            Some(lap) => {
                // ŵ ∝ ŵ_unif + laplace · ŵ_emp, per merged cell (§08.4).
                let mut counts = vec![0.0_f64; cells];
                let col =
                    x.0.data
                        .get(ma.axis)
                        .ok_or_else(|| PbError::ShapeMismatch {
                            what: format!("serve matrix missing column {} for weights", ma.axis),
                        })?;
                for &bin in col {
                    let cell = ma.model_bin_to_cell(bin)? as usize;
                    let slot = counts.get_mut(cell).ok_or_else(|| PbError::Internal {
                        what: "weight cell escaped counts".into(),
                    })?;
                    *slot += 1.0;
                }
                let unif = 1.0_f64 / cells as f64;
                let inv_n = if n_rows > 0 { 1.0 / n_rows as f64 } else { 0.0 };
                counts
                    .iter()
                    .map(|c| unif + lap * (c * inv_n))
                    .collect::<Vec<f64>>()
            }
        };
        let total: f64 = raw_w.iter().sum();
        if total.is_nan() || total <= 0.0 {
            return Err(PbError::Internal {
                what: "reference-measure axis weights summed to zero".into(),
            });
        }
        per_axis.push(raw_w.iter().map(|x| x / total).collect());
    }
    Ok(WeightCache {
        per_axis,
        kind: w.clone(),
    })
}

// ===========================================================================
// §08.3 — Purification: the mass-moving cascade.
// ===========================================================================

/// Subtract the `axis_w`-weighted slice mean along position `p` from `values`,
/// returning that mean as a tensor over the remaining axes (the mass moved one order
/// down). For an order-1 table the returned tensor is 0-D (a scalar → the intercept).
fn center_along(values: &mut Tensor, p: usize, axis_w: &[f64]) -> Result<Tensor, PbError> {
    let extents = values.shape();
    let sub_extents = drop_index(&extents, p);
    let mut means = Tensor::try_zeros(sub_extents)?;
    // Pass 1: accumulate the weighted slice mean for each lower-order coordinate.
    walk_extents(&extents, |coord| {
        let cell_p = *coord.get(p).ok_or_else(|| PbError::Internal {
            what: "centering position escaped coord".into(),
        })?;
        let wp = *axis_w.get(cell_p).ok_or_else(|| PbError::Internal {
            what: "centering cell escaped axis weights".into(),
        })?;
        let v = values.at(coord).ok_or_else(|| PbError::Internal {
            what: "centering coord out of range".into(),
        })?;
        means.add(&drop_index(coord, p), wp * v)
    })?;
    // Pass 2: subtract the mean from every cell of the slice.
    walk_extents(&extents, |coord| {
        let m = means
            .at(&drop_index(coord, p))
            .ok_or_else(|| PbError::Internal {
                what: "centering mean coord out of range".into(),
            })?;
        values.add(coord, -m)
    })?;
    Ok(means)
}

/// `u` with the raw feature at position `p` removed.
fn support_without(u: &FeatureSet, p: usize) -> FeatureSet {
    FeatureSet(
        u.0.iter()
            .enumerate()
            .filter_map(|(i, &f)| if i == p { None } else { Some(f) })
            .collect(),
    )
}

/// Purify the raw bank into the canonical fANOVA tables (spec §08.3). Single pass per
/// axis in decreasing `|u|` (3→2→1→intercept) — exact for axis-factorized `w`. The
/// cascade lazily creates the lower-order tables it feeds.
fn purify(
    raw: RawBank,
    w: &WeightCache,
    grids: &MergedGrids,
    mode: PurifyMode,
) -> Result<TableBank, PbError> {
    // Work map: support → (axes, values). Seed from the raw realized supports.
    let mut axes_of: BTreeMap<FeatureSet, Vec<AxisId>> = BTreeMap::new();
    let mut values_of: BTreeMap<FeatureSet, Tensor> = BTreeMap::new();
    for (u, rt) in raw.tables {
        axes_of.insert(u.clone(), rt.axes);
        values_of.insert(u, rt.values);
    }
    let mut f0 = raw.f0;

    let single_pass = matches!(mode, PurifyMode::SinglePass);
    if !single_pass {
        return Err(PbError::InvalidConfig {
            what: "v1 purify is SinglePass only (product/uniform w)".into(),
        });
    }

    for order in (1..=3usize).rev() {
        let keys: Vec<FeatureSet> = values_of
            .keys()
            .filter(|u| u.order() == order)
            .cloned()
            .collect();
        for u in keys {
            for p in 0..u.order() {
                let r = *u.0.get(p).ok_or_else(|| PbError::Internal {
                    what: "purify axis position escaped support".into(),
                })?;
                let axis_w = w.axis(r)?.to_vec();
                let means = {
                    let values = values_of.get_mut(&u).ok_or_else(|| PbError::Internal {
                        what: "purify support vanished".into(),
                    })?;
                    center_along(values, p, &axis_w)?
                };
                if order == 1 {
                    let m = means.at(&[]).ok_or_else(|| PbError::Internal {
                        what: "order-1 mean is not a scalar".into(),
                    })?;
                    f0 += m;
                } else {
                    let sub_u = support_without(&u, p);
                    // Ensure the lower-order table exists (lazy cascade allocation).
                    if !values_of.contains_key(&sub_u) {
                        let mut sub_axes = Vec::with_capacity(sub_u.order());
                        let mut sub_extents = Vec::with_capacity(sub_u.order());
                        for sr in &sub_u.0 {
                            sub_axes.push(grids.axis_id(*sr)?);
                            sub_extents.push(grids.cells(*sr)?);
                        }
                        axes_of.insert(sub_u.clone(), sub_axes);
                        values_of.insert(sub_u.clone(), Tensor::try_zeros(sub_extents)?);
                    }
                    let target = values_of.get_mut(&sub_u).ok_or_else(|| PbError::Internal {
                        what: "cascade target vanished".into(),
                    })?;
                    let sub_extents = target.shape();
                    walk_extents(&sub_extents, |coord| {
                        let m = means.at(coord).ok_or_else(|| PbError::Internal {
                            what: "cascade mean coord out of range".into(),
                        })?;
                        target.add(coord, m)
                    })?;
                }
            }
        }
    }

    // Build the EffectTables (variance cached; `support` is display-only metadata
    // filled separately by `fill_support` — it is not an fANOVA component, so purify,
    // which carries the algebra, stays independent of the data matrix).
    let mut tables = Vec::with_capacity(values_of.len());
    for (u, values) in values_of {
        let axes = axes_of.get(&u).cloned().ok_or_else(|| PbError::Internal {
            what: "table lost its axes".into(),
        })?;
        let variance = table_variance(&u, &values, w)?;
        let support = Tensor::try_zeros(values.shape())?;
        tables.push(EffectTable {
            u,
            axes,
            values,
            support,
            se_band: None,
            variance,
        });
    }

    Ok(TableBank {
        f0,
        tables,
        merged_grids: grids.border_grids()?,
        w: w.kind.clone(),
        factored: Vec::new(),
    })
}

/// `σ²(f_u)` under the product measure `w` (the table is pure, so this is `E_w[f²]`).
fn table_variance(u: &FeatureSet, values: &Tensor, w: &WeightCache) -> Result<f64, PbError> {
    let extents = values.shape();
    let mut m1 = 0.0_f64;
    let mut m2 = 0.0_f64;
    walk_extents(&extents, |coord| {
        let mut wprod = 1.0_f64;
        for (k, &cell) in coord.iter().enumerate() {
            let r = *u.0.get(k).ok_or_else(|| PbError::Internal {
                what: "variance axis escaped support".into(),
            })?;
            let wc = *w.axis(r)?.get(cell).ok_or_else(|| PbError::Internal {
                what: "variance cell escaped axis weights".into(),
            })?;
            wprod *= wc;
        }
        let v = values.at(coord).ok_or_else(|| PbError::Internal {
            what: "variance coord out of range".into(),
        })?;
        m1 += wprod * v;
        m2 += wprod * v * v;
        Ok(())
    })?;
    Ok(m2 - m1 * m1)
}

// ===========================================================================
// §08.10 — Factored high-order effects (the over-budget escape hatch).
// ===========================================================================

/// One depth-3 tree's purified order-3 contribution, stored as its `2×2×2` octant box
/// plus the per-axis low-side masks over the merged grid — never the dense union cube.
#[allow(dead_code)] // wired into the over-budget accumulate/purify path in Stage 3.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TripleBox {
    /// Purified octant values; index `= s0 | s1<<1 | s2<<2`, `s_d = 1` on the LOW side of
    /// table axis `d` (matching [`ObliviousTree`] leaf routing). Triple-centered under the
    /// global per-axis `w`, so it is exactly this tree's order-3 fANOVA component.
    p: [f64; 8],
    /// Per-table-axis low-side mask over merged cells (`true` ⇒ cell is on the low side).
    low: [Vec<bool>; 3],
}

/// One per-tree box of a factored order-3 effect in EXPORT form: the per-axis split
/// threshold (merged-grid border value) + missing-left routing, and the 8 purified octant
/// values (index `s0 | s1<<1 | s2<<2`, `s_d = 1` ⇒ the LOW side `x ≤ threshold`). The effect
/// is `Σ_box octant[side(x)]`, so a deployment can score it as a UNION of per-tree CASE
/// expressions without ever materializing the dense merged cube.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactoredBoxExport {
    /// Per-axis split threshold (the realized border value on each of the support's 3 axes).
    pub thresholds: [f32; 3],
    /// Per-axis learned missing direction (the reserved missing cell routes low when true).
    pub missing_left: [bool; 3],
    /// Purified octant values; index `s0 | s1<<1 | s2<<2`, `s_d = 1` ⇒ `x ≤ thresholds[d]`.
    pub octants: [f64; 8],
}

/// A factored order-3 effect for support `u`: the sum of per-tree purified boxes
/// (`f_u = Σ_t p_t`, Lengerich Cor. 2.2 linearity), kept WITHOUT materializing the dense
/// `Π cells` merged cube. Exact prediction ([`eval`](Self::eval), `O(#trees)`) and exact
/// `w`-weighted variance ([`variance`](Self::variance), `Σ_{t,s}⟨p_t,p_s⟩_w` via a per-axis
/// `2×2` weighted-mass contraction) — proven equal to the dense purified table for any
/// axis-factorized `w` (Uniform / ProductMarginals); only non-product `Joint` would break
/// the factorization (rejected in v1).
#[allow(dead_code)] // wired into the over-budget accumulate/purify path in Stage 3.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactoredTriple {
    /// The 3-feature support (table-dim order = sorted raw ids).
    pub u: FeatureSet,
    /// Merged-grid axes, parallel to the box dims.
    pub axes: Vec<AxisId>,
    /// Per-table-axis merged-cell weights (normalized, `Σ = 1`), shared by all boxes.
    per_axis_w: [Vec<f64>; 3],
    /// One purified box per support-`u` tree.
    boxes: Vec<TripleBox>,
    /// Cached `σ²(f_u)` under the bank's `w` (the proven `Σ_{t,s}⟨p_t,p_s⟩` closed form),
    /// so [`TableBank::sobol`]/`check_variance_sum` read it like an [`EffectTable`]'s.
    pub variance: f64,
}

#[allow(dead_code)] // wired into the over-budget accumulate/purify path in Stage 3.
impl FactoredTriple {
    /// Build the factored order-3 effect for support `u` from every tree whose support is
    /// exactly `u`, purifying each tree's box under the global per-axis weights `w`.
    fn from_model(
        model: &Model,
        u: &FeatureSet,
        grids: &MergedGrids,
        w: &WeightCache,
    ) -> Result<Self, PbError> {
        if u.order() != 3 {
            return Err(PbError::Internal {
                what: "FactoredTriple requires an order-3 support".into(),
            });
        }
        let u_ids: Vec<FeatureId> = u.0.iter().copied().collect();
        let mut per_axis_w: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (slot, r) in per_axis_w.iter_mut().zip(u_ids.iter()) {
            *slot = w.axis(*r)?.to_vec();
        }
        let axes: Vec<AxisId> = u_ids
            .iter()
            .map(|r| grids.axis_id(*r))
            .collect::<Result<_, _>>()?;
        let mut boxes = Vec::new();
        for (alpha, tree) in &model.trees {
            if &tree_support(model, tree)? != u {
                continue;
            }
            let (mut p, low) = build_tree_box(model, f64::from(*alpha), tree, &u_ids, grids)?;
            let (wlo, whi) = side_weights(&low, &per_axis_w)?;
            purify_box(&mut p, wlo, whi)?;
            boxes.push(TripleBox { p, low });
        }
        let mut ft = FactoredTriple {
            u: u.clone(),
            axes,
            per_axis_w,
            boxes,
            variance: 0.0,
        };
        ft.variance = ft.compute_variance()?;
        Ok(ft)
    }

    /// `f_u(x_u)` at a row given its per-raw merged-cell ids — the sum of every box's
    /// octant value. `O(#trees)`, no dense cube.
    pub fn eval(&self, x_cells: &[u32]) -> Result<f64, PbError> {
        let cells: Vec<usize> = self
            .axes
            .iter()
            .map(|a| {
                x_cells
                    .get(a.raw.0 as usize)
                    .copied()
                    .map(|c| c as usize)
                    .ok_or_else(|| PbError::ShapeMismatch {
                        what: format!("x_cells missing raw {} for factored eval", a.raw.0),
                    })
            })
            .collect::<Result<_, _>>()?;
        let mut acc = 0.0_f64;
        for b in &self.boxes {
            let mut idx = 0usize;
            for (d_bit, (mask, &cell)) in b.low.iter().zip(cells.iter()).enumerate() {
                let s = *mask.get(cell).ok_or_else(|| PbError::Internal {
                    what: "factored eval cell escaped low mask".into(),
                })?;
                idx |= usize::from(s) << d_bit;
            }
            acc += *b.p.get(idx).ok_or_else(|| PbError::Internal {
                what: "factored eval box index escaped 0..8".into(),
            })?;
        }
        Ok(acc)
    }

    /// Per-tree boxes in export form (thresholds + octants); see [`FactoredBoxExport`]. Lets
    /// the rating export / SQL recipe emit this effect without ever densifying the cube.
    pub fn export_boxes(&self) -> Result<Vec<FactoredBoxExport>, PbError> {
        let mut out = Vec::with_capacity(self.boxes.len());
        for b in &self.boxes {
            let mut thresholds = [0.0_f32; 3];
            let mut missing_left = [false; 3];
            for (((thr, miss), mask), axis) in thresholds
                .iter_mut()
                .zip(missing_left.iter_mut())
                .zip(b.low.iter())
                .zip(self.axes.iter())
            {
                // A tree's threshold θ = borders[j] makes finite cells 1..=j+1 low, so the
                // count of finite low cells is j+1 (always ≥ 1 — the split border is realized).
                let finite_low = mask.iter().skip(1).filter(|&&l| l).count();
                let j = finite_low.checked_sub(1).ok_or_else(|| PbError::Internal {
                    what: "factored box has no low finite cell".into(),
                })?;
                *thr = *axis.borders.get(j).ok_or_else(|| PbError::Internal {
                    what: "factored box threshold escaped borders".into(),
                })?;
                *miss = *mask.first().ok_or_else(|| PbError::Internal {
                    what: "factored box missing cell absent".into(),
                })?;
            }
            out.push(FactoredBoxExport {
                thresholds,
                missing_left,
                octants: b.p,
            });
        }
        Ok(out)
    }

    /// `σ²(f_u) = ⟨Σ_t p_t, Σ_t p_t⟩_w = Σ_{t,s}⟨p_t,p_s⟩_w`. Each pairwise inner product
    /// factorizes as a product over axes of `2×2` weighted-mass matrices (the joint
    /// low/high cell membership of the two trees), so no joint cube is ever built.
    fn compute_variance(&self) -> Result<f64, PbError> {
        let mut var = 0.0_f64;
        for (i, bi) in self.boxes.iter().enumerate() {
            var += self.inner(bi, bi)?;
            for bj in self.boxes.iter().skip(i + 1) {
                var += 2.0 * self.inner(bi, bj)?;
            }
        }
        Ok(var)
    }

    /// `⟨p_i, p_j⟩_w` via per-axis `2×2` joint weighted-mass matrices.
    fn inner(&self, bi: &TripleBox, bj: &TripleBox) -> Result<f64, PbError> {
        // mass[d][a_i][a_j] = Σ_cell w_d[cell] over cells with sides (a_i, a_j); a = 1 ⇒ low.
        let mut mass: Vec<[[f64; 2]; 2]> = Vec::with_capacity(3);
        for ((wd, li), lj) in self.per_axis_w.iter().zip(bi.low.iter()).zip(bj.low.iter()) {
            let mut m = [[0.0_f64; 2]; 2];
            for ((&wc, &a_i), &a_j) in wd.iter().zip(li.iter()).zip(lj.iter()) {
                let row = m
                    .get_mut(usize::from(a_i))
                    .ok_or_else(|| PbError::Internal {
                        what: "factored inner row escaped 0..2".into(),
                    })?;
                let cell = row
                    .get_mut(usize::from(a_j))
                    .ok_or_else(|| PbError::Internal {
                        what: "factored inner col escaped 0..2".into(),
                    })?;
                *cell += wc;
            }
            mass.push(m);
        }
        let (m0, m1, m2) = match mass.as_slice() {
            [a, b, c] => (a, b, c),
            _ => {
                return Err(PbError::Internal {
                    what: "factored inner expects exactly 3 axes".into(),
                })
            }
        };
        let g = |m: &[[f64; 2]; 2], a: usize, b: usize| -> Result<f64, PbError> {
            m.get(a)
                .and_then(|r| r.get(b))
                .copied()
                .ok_or_else(|| PbError::Internal {
                    what: "factored mass index escaped 0..2".into(),
                })
        };
        let pat = |bx: &TripleBox, a0: usize, a1: usize, a2: usize| -> Result<f64, PbError> {
            bx.p.get(a0 | (a1 << 1) | (a2 << 2))
                .copied()
                .ok_or_else(|| PbError::Internal {
                    what: "factored octant index escaped 0..8".into(),
                })
        };
        let mut s = 0.0_f64;
        for a0i in 0..2 {
            for a0j in 0..2 {
                for a1i in 0..2 {
                    for a1j in 0..2 {
                        for a2i in 0..2 {
                            for a2j in 0..2 {
                                let wgt = g(m0, a0i, a0j)? * g(m1, a1i, a1j)? * g(m2, a2i, a2j)?;
                                if wgt != 0.0 {
                                    s += wgt * pat(bi, a0i, a1i, a2i)? * pat(bj, a0j, a1j, a2j)?;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(s)
    }
}

/// Per-axis low/high total `w`-mass for one tree's masks (`whi = 1 − wlo` since `w` sums
/// to 1 per axis, but both are summed explicitly for numerical symmetry).
#[allow(dead_code)] // wired into the over-budget accumulate/purify path in Stage 3.
fn side_weights(
    low: &[Vec<bool>; 3],
    per_axis_w: &[Vec<f64>; 3],
) -> Result<([f64; 3], [f64; 3]), PbError> {
    let mut wlo = [0.0_f64; 3];
    let mut whi = [0.0_f64; 3];
    for (((slot_lo, slot_hi), wd), mask) in wlo
        .iter_mut()
        .zip(whi.iter_mut())
        .zip(per_axis_w.iter())
        .zip(low.iter())
    {
        for (&wc, &is_low) in wd.iter().zip(mask.iter()) {
            if is_low {
                *slot_lo += wc;
            } else {
                *slot_hi += wc;
            }
        }
    }
    Ok((wlo, whi))
}

/// One support-`u` tree's RAW octant box (table-dim side order, `s_d = 1` ⇒ low) plus the
/// per-axis low-side masks over the merged grid. Shared by the factored builder and shed.
#[allow(dead_code)] // wired into the over-budget accumulate/purify path in Stage 3.
fn build_tree_box(
    model: &Model,
    alpha: f64,
    tree: &crate::engine::ObliviousTree,
    u_ids: &[FeatureId],
    grids: &MergedGrids,
) -> Result<([f64; 8], [Vec<bool>; 3]), PbError> {
    let mut low: [Vec<bool>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut level_of: [usize; 3] = [0; 3];
    for (level, split) in tree.splits.iter().enumerate() {
        let prov = model
            .provenance
            .get(split.axis as usize)
            .ok_or_else(|| PbError::Internal {
                what: "factored split axis absent from provenance".into(),
            })?;
        let d = u_ids
            .iter()
            .position(|r| *r == prov.raw)
            .ok_or_else(|| PbError::Internal {
                what: "factored split raw absent from support".into(),
            })?;
        let ma = grids.axis(prov.raw)?;
        let cells = ma.cells();
        let mut mask = Vec::with_capacity(cells);
        for c in 0..cells {
            let rep_bin = ma.rep_model_bin(c)?;
            mask.push(low_bit(rep_bin, split.bin_le, split.missing_left));
        }
        *low.get_mut(d).ok_or_else(|| PbError::Internal {
            what: "factored mask slot escaped 0..3".into(),
        })? = mask;
        *level_of.get_mut(d).ok_or_else(|| PbError::Internal {
            what: "factored level slot escaped 0..3".into(),
        })? = level;
    }
    let mut p = [0.0_f64; 8];
    for s0 in 0..2usize {
        for s1 in 0..2usize {
            for s2 in 0..2usize {
                let leaf_idx = [s0, s1, s2]
                    .iter()
                    .zip(level_of.iter())
                    .fold(0usize, |acc, (&s, &lv)| acc | (s << lv));
                let leaf = *tree.leaves.get(leaf_idx).ok_or_else(|| PbError::Internal {
                    what: "factored leaf index escaped 0..8".into(),
                })?;
                *p.get_mut(s0 | (s1 << 1) | (s2 << 2))
                    .ok_or_else(|| PbError::Internal {
                        what: "factored box index escaped 0..8".into(),
                    })? = alpha * f64::from(leaf);
            }
        }
    }
    Ok((p, low))
}

/// Center a `2×2×2` box along each table axis IN CASCADE ORDER (axis 0, then 1 on the
/// 0-centered box, then 2), returning the per-axis marginals shed downward: `marg[p]` is the
/// `2×2` weighted mean over the OTHER two axes (increasing position order) at step `p` —
/// exactly what the dense [`center_along`] deposits into the order-2 table `u\{p}`. Leaves
/// `p` as the triple-centered residual (the tree's order-3 fANOVA component).
#[allow(dead_code)] // wired into the over-budget accumulate/purify path in Stage 3.
fn center_box_capturing(
    p: &mut [f64; 8],
    wlo: [f64; 3],
    whi: [f64; 3],
) -> Result<[[[f64; 2]; 2]; 3], PbError> {
    let mut marg = [[[0.0_f64; 2]; 2]; 3];
    for (d, ((&wl, &wh), md)) in wlo.iter().zip(whi.iter()).zip(marg.iter_mut()).enumerate() {
        let others: [usize; 2] = match d {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        };
        let o0 = *others.first().ok_or_else(|| PbError::Internal {
            what: "center axis pair a".into(),
        })?;
        let o1 = *others.get(1).ok_or_else(|| PbError::Internal {
            what: "center axis pair b".into(),
        })?;
        for b0 in 0..2usize {
            for b1 in 0..2usize {
                let base = (b0 << o0) | (b1 << o1);
                let i_hi = base; // s_d = 0 (high)
                let i_lo = base | (1 << d); // s_d = 1 (low)
                let hi = *p.get(i_hi).ok_or_else(|| PbError::Internal {
                    what: "center hi index escaped 0..8".into(),
                })?;
                let lo = *p.get(i_lo).ok_or_else(|| PbError::Internal {
                    what: "center lo index escaped 0..8".into(),
                })?;
                let mean = wh * hi + wl * lo;
                let row = md.get_mut(b0).ok_or_else(|| PbError::Internal {
                    what: "marginal row escaped 0..2".into(),
                })?;
                *row.get_mut(b1).ok_or_else(|| PbError::Internal {
                    what: "marginal col escaped 0..2".into(),
                })? = mean;
                *p.get_mut(i_hi).ok_or_else(|| PbError::Internal {
                    what: "center hi write escaped 0..8".into(),
                })? -= mean;
                *p.get_mut(i_lo).ok_or_else(|| PbError::Internal {
                    what: "center lo write escaped 0..8".into(),
                })? -= mean;
            }
        }
    }
    Ok(marg)
}

/// Triple-center a `2×2×2` octant box (residual only); see [`center_box_capturing`].
#[allow(dead_code)] // wired into the over-budget accumulate/purify path in Stage 3.
fn purify_box(p: &mut [f64; 8], wlo: [f64; 3], whi: [f64; 3]) -> Result<(), PbError> {
    center_box_capturing(p, wlo, whi).map(|_| ())
}

/// Run the order-3 purification cascade for an over-budget support `u` WITHOUT the dense
/// cube: per tree, center its box in cascade order and broadcast each axis's marginal into
/// the dense order-2 raw table `u\{p}` (lazily created) — by linearity (`Σ_t`) this deposits
/// exactly what the dense `center_along` would, so the subsequent order-2→1 cascade runs on
/// the augmented lower tables unchanged. Removes `u` from `raw_tables`; returns the
/// triple-centered residual as a [`FactoredTriple`].
#[allow(dead_code)] // wired into the over-budget accumulate/purify path in Stage 3.
fn factored_shed_into_raw(
    model: &Model,
    u: &FeatureSet,
    grids: &MergedGrids,
    w: &WeightCache,
    raw_tables: &mut BTreeMap<FeatureSet, RawTable>,
) -> Result<FactoredTriple, PbError> {
    if u.order() != 3 {
        return Err(PbError::Internal {
            what: "factored_shed_into_raw requires an order-3 support".into(),
        });
    }
    let u_ids: Vec<FeatureId> = u.0.iter().copied().collect();
    let mut per_axis_w: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (slot, r) in per_axis_w.iter_mut().zip(u_ids.iter()) {
        *slot = w.axis(*r)?.to_vec();
    }
    let axes: Vec<AxisId> = u_ids
        .iter()
        .map(|r| grids.axis_id(*r))
        .collect::<Result<_, _>>()?;
    let mut boxes = Vec::new();
    for (alpha, tree) in &model.trees {
        if &tree_support(model, tree)? != u {
            continue;
        }
        let (mut p, low) = build_tree_box(model, f64::from(*alpha), tree, &u_ids, grids)?;
        let (wlo, whi) = side_weights(&low, &per_axis_w)?;
        let marg = center_box_capturing(&mut p, wlo, whi)?;
        for pos in 0..3usize {
            let m_pos = marg.get(pos).ok_or_else(|| PbError::Internal {
                what: "shed marginal axis escaped 0..3".into(),
            })?;
            let others: [usize; 2] = match pos {
                0 => [1, 2],
                1 => [0, 2],
                _ => [0, 1],
            };
            let o0 = *others.first().ok_or_else(|| PbError::Internal {
                what: "shed axis pair a".into(),
            })?;
            let o1 = *others.get(1).ok_or_else(|| PbError::Internal {
                what: "shed axis pair b".into(),
            })?;
            let sub_u = support_without(u, pos);
            if !raw_tables.contains_key(&sub_u) {
                let mut sub_axes = Vec::with_capacity(2);
                let mut sub_extents = Vec::with_capacity(2);
                for sr in &sub_u.0 {
                    sub_axes.push(grids.axis_id(*sr)?);
                    sub_extents.push(grids.cells(*sr)?);
                }
                raw_tables.insert(
                    sub_u.clone(),
                    RawTable {
                        u: sub_u.clone(),
                        axes: sub_axes,
                        values: Tensor::try_zeros(sub_extents)?,
                    },
                );
            }
            let target = raw_tables
                .get_mut(&sub_u)
                .ok_or_else(|| PbError::Internal {
                    what: "factored shed target vanished".into(),
                })?;
            let low_a = low.get(o0).ok_or_else(|| PbError::Internal {
                what: "shed mask a missing".into(),
            })?;
            let low_b = low.get(o1).ok_or_else(|| PbError::Internal {
                what: "shed mask b missing".into(),
            })?;
            let extents = target.values.shape();
            walk_extents(&extents, |coord| {
                let c0 = *coord.first().ok_or_else(|| PbError::Internal {
                    what: "shed coord axis 0 missing".into(),
                })?;
                let c1 = *coord.get(1).ok_or_else(|| PbError::Internal {
                    what: "shed coord axis 1 missing".into(),
                })?;
                let s0 = usize::from(*low_a.get(c0).ok_or_else(|| PbError::Internal {
                    what: "shed cell a escaped low mask".into(),
                })?);
                let s1 = usize::from(*low_b.get(c1).ok_or_else(|| PbError::Internal {
                    what: "shed cell b escaped low mask".into(),
                })?);
                let row = m_pos.get(s0).ok_or_else(|| PbError::Internal {
                    what: "shed marginal row escaped 0..2".into(),
                })?;
                let delta = *row.get(s1).ok_or_else(|| PbError::Internal {
                    what: "shed marginal col escaped 0..2".into(),
                })?;
                target.values.add(coord, delta)
            })?;
        }
        boxes.push(TripleBox { p, low });
    }
    raw_tables.remove(u);
    let mut ft = FactoredTriple {
        u: u.clone(),
        axes,
        per_axis_w,
        boxes,
        variance: 0.0,
    };
    ft.variance = ft.compute_variance()?;
    Ok(ft)
}

/// Fill each table's per-cell training-row count from `x` (display metadata, §08.7).
/// Runs after purify (support is not an fANOVA component, so it never enters the gates).
fn fill_support(
    bank: &mut TableBank,
    grids: &MergedGrids,
    x: &ServeBinnedMatrix,
) -> Result<(), PbError> {
    let n_rows = x.0.n_rows as usize;
    for table in &mut bank.tables {
        let mut coord = vec![0usize; table.u.order()];
        for row in 0..n_rows {
            for (k, r) in table.u.0.iter().enumerate() {
                let ma = grids.axis(*r)?;
                let col =
                    x.0.data
                        .get(ma.axis)
                        .ok_or_else(|| PbError::ShapeMismatch {
                            what: format!("serve matrix missing column {} for support", ma.axis),
                        })?;
                let bin = *col.get(row).ok_or_else(|| PbError::Internal {
                    what: "support row escaped column".into(),
                })?;
                let slot = coord.get_mut(k).ok_or_else(|| PbError::Internal {
                    what: "support coord position escaped".into(),
                })?;
                *slot = ma.model_bin_to_cell(bin)? as usize;
            }
            table.support.add(&coord, 1.0)?;
        }
    }
    Ok(())
}

// ===========================================================================
// §08.6 — The five Invariant checks (build gates), at the real bank.
// ===========================================================================

/// The sorted distinct raw features appearing in any of the bank's tables.
fn bank_features(bank: &TableBank) -> Vec<FeatureId> {
    let mut set: BTreeSet<FeatureId> = BTreeSet::new();
    for t in &bank.tables {
        for r in &t.u.0 {
            set.insert(*r);
        }
    }
    set.into_iter().collect()
}

/// The sorted distinct raw features appearing in any tree split of the model.
fn model_features(model: &Model) -> Result<Vec<FeatureId>, PbError> {
    let mut set: BTreeSet<FeatureId> = BTreeSet::new();
    for (_, tree) in &model.trees {
        for split in &tree.splits {
            let prov =
                model
                    .provenance
                    .get(split.axis as usize)
                    .ok_or_else(|| PbError::Internal {
                        what: format!("split axis {} absent from provenance", split.axis),
                    })?;
            set.insert(prov.raw);
        }
    }
    Ok(set.into_iter().collect())
}

/// The joint-grid features a gate must inspect: every model-realized raw feature AND
/// every table feature. Using only table features would let a malformed bank hide a
/// missing table by shrinking the check domain.
fn gate_features(model: &Model, bank: &TableBank) -> Result<Vec<FeatureId>, PbError> {
    let mut set: BTreeSet<FeatureId> = BTreeSet::new();
    for f in model_features(model)? {
        set.insert(f);
    }
    for f in bank_features(bank) {
        set.insert(f);
    }
    Ok(set.into_iter().collect())
}

/// Visit interior points of the joint merged grid over `feats`. Exhaustive when the
/// product is `<= JOINT_CAP` (the §08.6 worst-case-per-cell sweep); otherwise a
/// deterministic sample (the §08.8 release behavior). Each visit gets `x_cells` (indexed
/// by raw feature) and `rep_bins` (indexed by model axis) for the same point. Returns
/// `true` iff the joint grid was SAMPLED (rather than exhausted) — the integral gates
/// (VarianceSum) self-normalize and widen their tolerance in that case; the per-point
/// gates (Reconstruction/ThreeWayEqual) are sound either way and ignore the flag.
fn enumerate_check_points(
    grids: &MergedGrids,
    feats: &[FeatureId],
    mut visit: impl FnMut(&[u32], &[u8]) -> Result<(), PbError>,
) -> Result<bool, PbError> {
    let n_features = grids.n_features();
    let mut extents = Vec::with_capacity(feats.len());
    for r in feats {
        extents.push(grids.cells(*r)?);
    }
    // SATURATE, don't error: on a wide model the joint grid (Π cells over all gate features)
    // can exceed u64 — that simply means it dwarfs the enumeration cap and must be SAMPLED.
    // (Erroring here was the high-dimensional-categorical decomposition bug, e.g. allstate's
    // 130 features overflowing u64 before the cap check could route to sampling.)
    let total = saturating_product_u64(&extents);

    let mut emit = |tuple: &[usize]| -> Result<(), PbError> {
        let mut x_cells = vec![0u32; n_features];
        let mut rep_bins = vec![0u8; n_features];
        for (k, r) in feats.iter().enumerate() {
            let cell = *tuple.get(k).ok_or_else(|| PbError::Internal {
                what: "check tuple shorter than feats".into(),
            })?;
            let ma = grids.axis(*r)?;
            *x_cells
                .get_mut(r.0 as usize)
                .ok_or_else(|| PbError::Internal {
                    what: "x_cells raw index escaped".into(),
                })? = cell as u32;
            *rep_bins.get_mut(ma.axis).ok_or_else(|| PbError::Internal {
                what: "rep_bins axis index escaped".into(),
            })? = ma.rep_model_bin(cell)?;
        }
        visit(&x_cells, &rep_bins)
    };

    if total <= joint_cap() as u64 {
        walk_extents(&extents, &mut emit)?;
        Ok(false)
    } else {
        // Deterministic strided sample: mix the sample index per axis with a splitmix
        // step so the points spread across the space without RNG state.
        let mut tuple = vec![0usize; feats.len()];
        for s in 0..joint_cap() as u64 {
            for (k, &e) in extents.iter().enumerate() {
                let mut z = s
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((k as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z ^= z >> 27;
                *tuple.get_mut(k).ok_or_else(|| PbError::Internal {
                    what: "sample tuple position escaped".into(),
                })? = (z % e as u64) as usize;
            }
            emit(&tuple)?;
        }
        Ok(true)
    }
}

/// **Reconstruction (I2.1):** the ensemble equals `f0 + Σ_u f_u` at every merged-grid
/// cell (the missing cell included), within `recon_tol`.
///
/// # Errors
/// [`Invariant::Reconstruction`] if any cell's discrepancy exceeds tolerance.
pub fn check_reconstruction(model: &Model, bank: &TableBank) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model).recon_tol;
    let grids = MergedGrids::from_model(model)?;
    let feats = gate_features(model, bank)?;
    enumerate_check_points(&grids, &feats, |x_cells, rep_bins| {
        let ens = model.ensemble_f64(rep_bins)?;
        let tab = bank.score(x_cells)?;
        if (ens - tab).abs() > tol {
            return Err(PbError::invariant(Invariant::Reconstruction));
        }
        Ok(())
    })?;
    Ok(())
}

/// The exact `w`-weighted mean of the ensemble, `E_w[F] = f0 + Σ_t alpha_t·E_w[T_t]`.
/// Computed SEPARABLY per tree over its own ≤3-feature support grid (integrating out the
/// other features, whose product weights sum to 1) — so it never materializes the joint
/// grid and stays exact regardless of the ensemble's total feature count. This is the
/// fix for the prior joint-grid sampler, which produced an un-normalized partial sum and
/// false-failed large exact models on MassConservation.
fn ensemble_w_mean(model: &Model, grids: &MergedGrids, w: &WeightCache) -> Result<f64, PbError> {
    let mut mass = f64::from(model.f0);
    for (alpha, tree) in &model.trees {
        let u = tree_support(model, tree)?;
        let u_ids: Vec<FeatureId> = u.0.iter().copied().collect();
        let mut extents = Vec::with_capacity(u_ids.len());
        for r in &u_ids {
            extents.push(grids.cells(*r)?);
        }
        let mut tree_int = 0.0_f64;
        walk_extents(&extents, |tuple| {
            let leaf_idx = leaf_index_for_tuple(model, tree, grids, &u_ids, tuple)?;
            let leaf = *tree.leaves.get(leaf_idx).ok_or_else(|| PbError::Internal {
                what: "mass: oblivious leaf index escaped 0..8".into(),
            })?;
            let mut wprod = 1.0_f64;
            for (k, r) in u_ids.iter().enumerate() {
                let cell = *tuple.get(k).ok_or_else(|| PbError::Internal {
                    what: "mass tuple shorter than support".into(),
                })?;
                wprod *= *w.axis(*r)?.get(cell).ok_or_else(|| PbError::Internal {
                    what: "mass cell escaped axis weights".into(),
                })?;
            }
            tree_int += wprod * f64::from(leaf);
            Ok(())
        })?;
        mass += f64::from(*alpha) * tree_int;
    }
    // The cell-basis correction is added to `ensemble_f64`, so its w-weighted mass must be
    // accounted here too — otherwise it lands only in `bank.f0` (via the purify cascade) and
    // mass conservation (I2.2: `E_w[F_ens] == f0`) fails by exactly `Σ_cells w·delta`.
    if let Some(corr) = &model.correction {
        for table in &corr.tables {
            let mut raws = Vec::with_capacity(table.axes.len());
            for &axis in &table.axes {
                raws.push(
                    model
                        .provenance
                        .get(axis as usize)
                        .ok_or_else(|| PbError::Internal {
                            what: format!("correction mass axis {axis} absent from provenance"),
                        })?
                        .raw,
                );
            }
            let extents: Vec<usize> = table.shape.iter().map(|&s| s as usize).collect();
            let mut flat = 0usize;
            walk_extents(&extents, |tuple| {
                let mut wprod = 1.0_f64;
                for (k, r) in raws.iter().enumerate() {
                    let cell = *tuple.get(k).ok_or_else(|| PbError::Internal {
                        what: "correction mass tuple shorter than support".into(),
                    })?;
                    wprod *= *w.axis(*r)?.get(cell).ok_or_else(|| PbError::Internal {
                        what: "correction mass cell escaped axis weights".into(),
                    })?;
                }
                let v = *table.values.get(flat).ok_or_else(|| PbError::Internal {
                    what: "correction mass flat index escaped values".into(),
                })?;
                flat += 1;
                mass += wprod * v;
                Ok(())
            })?;
        }
    }
    Ok(mass)
}

/// **MassConservation (I2.2):** all `w`-mass that survives purification sits in the
/// intercept — `E_w[F_ens] == f0` (every table integrates to zero). Computed EXACTLY via
/// the separable per-tree integral ([`ensemble_w_mean`]), so it is correct for arbitrarily
/// large ensembles (no joint-grid enumeration / sampling).
///
/// # Errors
/// [`Invariant::MassConservation`] if the `w`-weighted ensemble mean drifts from `f0`.
pub(crate) fn check_mass_conservation(
    model: &Model,
    bank: &TableBank,
    w: &WeightCache,
) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model).mass_tol;
    let grids = MergedGrids::from_model(model)?;
    let mass = ensemble_w_mean(model, &grids, w)?;
    if (mass - bank.f0).abs() > tol {
        return Err(PbError::invariant(Invariant::MassConservation));
    }
    Ok(())
}

/// The product weight `Π_{i ∈ feats} w_i(x_i)` for a joint cell-tuple.
fn joint_weight(w: &WeightCache, feats: &[FeatureId], x_cells: &[u32]) -> Result<f64, PbError> {
    let mut wprod = 1.0_f64;
    for r in feats {
        let cell = *x_cells.get(r.0 as usize).ok_or_else(|| PbError::Internal {
            what: "joint-weight raw index escaped x_cells".into(),
        })? as usize;
        let wc = *w.axis(*r)?.get(cell).ok_or_else(|| PbError::Internal {
            what: "joint-weight cell escaped axis weights".into(),
        })?;
        wprod *= wc;
    }
    Ok(wprod)
}

/// **Purity (I2.3):** every axis-slice of every [`EffectTable`] has `w`-weighted mean
/// zero. Maps to [`Invariant::Decomposability`] (§2.8 has no separate `Purity`).
///
/// # Errors
/// [`Invariant::Decomposability`] if any conditional axis-slice mean is non-zero.
pub(crate) fn check_purity(
    model: &Model,
    bank: &TableBank,
    w: &WeightCache,
) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model).purity_tol;
    for table in &bank.tables {
        let extents = table.values.shape();
        for p in 0..table.u.order() {
            let r = *table.u.0.get(p).ok_or_else(|| PbError::Internal {
                what: "purity axis escaped support".into(),
            })?;
            let axis_w = w.axis(r)?;
            let mut means: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
            walk_extents(&extents, |coord| {
                let cell_p = *coord.get(p).ok_or_else(|| PbError::Internal {
                    what: "purity position escaped coord".into(),
                })?;
                let wp = *axis_w.get(cell_p).ok_or_else(|| PbError::Internal {
                    what: "purity cell escaped axis weights".into(),
                })?;
                let v = table.values.at(coord).ok_or_else(|| PbError::Internal {
                    what: "purity coord out of range".into(),
                })?;
                *means.entry(drop_index(coord, p)).or_insert(0.0) += wp * v;
                Ok(())
            })?;
            for m in means.values() {
                if m.abs() > tol {
                    return Err(PbError::invariant(Invariant::Decomposability));
                }
            }
        }
    }
    // Factored order-3 effects: each per-tree residual box is triple-centered by
    // construction, so re-centering it must shed ZERO marginals (every axis-slice already
    // has w-weighted mean 0). Σ_t of purified boxes is purified, so this certifies the
    // factored f_u's purity without densifying the merged cube.
    for ft in &bank.factored {
        for b in &ft.boxes {
            let (wlo, whi) = side_weights(&b.low, &ft.per_axis_w)?;
            let mut p = b.p;
            let marg = center_box_capturing(&mut p, wlo, whi)?;
            for plane in &marg {
                for row in plane {
                    for &m in row {
                        if m.abs() > tol {
                            return Err(PbError::invariant(Invariant::Decomposability));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// **VarianceSum (I2.4):** `σ²(F) == Σ_u σ²(f_u)` under product/uniform `w`.
///
/// # Errors
/// [`Invariant::VarianceSum`] if total variance diverges from the sum of per-table
/// variances.
pub(crate) fn check_variance_sum(
    model: &Model,
    bank: &TableBank,
    w: &WeightCache,
) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model).var_tol;
    let grids = MergedGrids::from_model(model)?;
    let feats = gate_features(model, bank)?;
    let (mut m1, mut m2, mut wsum) = (0.0_f64, 0.0_f64, 0.0_f64);
    let sampled = enumerate_check_points(&grids, &feats, |x_cells, rep_bins| {
        let wprod = joint_weight(w, &feats, x_cells)?;
        let e = model.ensemble_f64(rep_bins)?;
        m1 += wprod * e;
        m2 += wprod * e * e;
        wsum += wprod;
        Ok(())
    })?;
    // Self-normalize by the realized total weight. For the EXHAUSTIVE sweep `wsum == 1`
    // (the product of per-axis-normalized weights summed over the full grid), so this is
    // a no-op and the integral stays exact. For the SAMPLED sweep `wsum < 1`, this turns
    // the partial sum into a consistent self-normalized (Horvitz–Thompson) estimator of
    // the true moments — fixing the prior bug where the un-normalized partial sum made a
    // large exact model false-fail VarianceSum.
    if !wsum.is_finite() || wsum <= 0.0 {
        return Err(PbError::Internal {
            what: "variance check accumulated non-positive total weight".into(),
        });
    }
    let var_ens = (m2 / wsum) - (m1 / wsum) * (m1 / wsum);
    let var_tables: f64 = bank.tables.iter().map(|t| t.variance).sum::<f64>()
        + bank.factored.iter().map(|ft| ft.variance).sum::<f64>();
    // Exact tolerance when exhaustive; a relative band when sampled (the sampled estimator
    // carries Monte-Carlo error, so for >JOINT_CAP-cell models VarianceSum is a STATISTICAL
    // certification, not bit-exact — documented FLAG; MassConservation stays exact).
    let eff_tol = if sampled {
        tol + SAMPLE_VAR_REL * (var_tables.abs() + var_ens.abs())
    } else {
        tol
    };
    if (var_ens - var_tables).abs() > eff_tol {
        return Err(PbError::invariant(Invariant::VarianceSum));
    }
    Ok(())
}

/// **ThreeWayEqual (I2.5):** tree-sum = table-sum = Shapley-sum at every cell, within
/// the derived float tolerance. The Shapley leg is an INDEPENDENT path (each `f_u`
/// split equally among its `|u|` features, summed per feature).
///
/// # Errors
/// [`Invariant::ThreeWayEqual`] if any of the three reconstructions disagree.
pub fn check_three_way_equal(model: &Model, bank: &TableBank) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model).recon_tol;
    let grids = MergedGrids::from_model(model)?;
    let feats = gate_features(model, bank)?;
    enumerate_check_points(&grids, &feats, |x_cells, rep_bins| {
        let tree = model.ensemble_f64(rep_bins)?;
        let table = bank.score(x_cells)?;
        let shap = bank.f0 + bank.shap(x_cells)?.iter().sum::<f64>();
        if (tree - table).abs() > tol || (table - shap).abs() > tol {
            return Err(PbError::invariant(Invariant::ThreeWayEqual));
        }
        Ok(())
    })?;
    Ok(())
}

/// Run all five I2 checks against a real fitted model and its purified bank (spec
/// §13.1). Rebuilds the reference weights from `x` (a [`ServeBinnedMatrix`], R-CATSERVE)
/// and `bank.w`. `VarianceSum` is asserted under product/uniform `w` (the v1 measures).
///
/// # Errors
/// The first failing check's [`Invariant`], wrapped in [`PbError::InvariantViolated`];
/// or a propagated weight/grid error.
pub fn assert_exact_decomposition(
    model: &Model,
    bank: &TableBank,
    x: &ServeBinnedMatrix,
) -> Result<(), PbError> {
    let grids = MergedGrids::from_model(model)?;
    let w = build_weights(x, &grids, &bank.w)?;
    check_reconstruction(model, bank)?;
    check_mass_conservation(model, bank, &w)?;
    check_purity(model, bank, &w)?;
    check_variance_sum(model, bank, &w)?;
    check_three_way_equal(model, bank)?;
    Ok(())
}

/// **FeatureBudget (I1, spec §13.2):** every tree is depth `1..=3`, `splits.len() ==
/// depth`, and the count of DISTINCT raw features across its splits equals `depth`.
///
/// # Errors
/// [`Invariant::FeatureBudget`] for any tree that violates the depth-3 / ≤3-distinct
/// contract; [`PbError::Internal`] if a split names an axis absent from provenance.
pub fn check_feature_budget(model: &Model) -> Result<(), PbError> {
    for (_, tree) in &model.trees {
        let depth = usize::from(tree.depth);
        if !(1..=3).contains(&depth) || tree.splits.len() != depth {
            return Err(PbError::invariant(Invariant::FeatureBudget));
        }
        let mut distinct: SmallVec<[u32; 3]> = SmallVec::new();
        for split in &tree.splits {
            let prov =
                model
                    .provenance
                    .get(split.axis as usize)
                    .ok_or_else(|| PbError::Internal {
                        what: format!("split axis {} absent from provenance", split.axis),
                    })?;
            let raw = prov.raw.0;
            if !distinct.contains(&raw) {
                distinct.push(raw);
            }
        }
        if distinct.len() != depth {
            return Err(PbError::invariant(Invariant::FeatureBudget));
        }
    }
    Ok(())
}

// ===========================================================================
// §08.5 / §08.8 — Public API: Model::explain + TableBank reads.
// ===========================================================================

impl Model {
    /// Build the complete purified [`TableBank`] under `w` (spec §08.8). Takes a
    /// [`ServeBinnedMatrix`] (R-CATSERVE): the caller re-encodes raw categoricals
    /// through this model's frozen `schema.cat_encoders` — `explain` MUST NOT be handed a
    /// `TrainBinnedMatrix`. Runs all five build gates by default. An `Approximate` model
    /// refuses to export an `Exact` bank ([`PbError::ExactnessFirewall`]).
    ///
    /// # Errors
    /// [`PbError::ExactnessFirewall`] on an `Approximate` model; [`PbError::InvalidConfig`]
    /// for the unsupported-in-v1 `Joint` measure or a categorical axis; [`PbError::TableBudget`]
    /// if a table or the bank exceeds its cell budget; [`PbError::InvariantViolated`] if any
    /// of the five gates fail; plus propagated shape/grid errors.
    pub fn explain(&self, x: &ServeBinnedMatrix, w: RefMeasure) -> Result<TableBank, PbError> {
        self.explain_with_budget(x, w, TableBudget::default())
    }

    /// Build the complete purified [`TableBank`] with an explicit table budget
    /// (spec §08.10). This is the same exact pipeline as [`Model::explain`], but lets
    /// callers choose [`OverflowPolicy::SparseFallback`] for adversarial or large
    /// merged grids instead of the default hard-error policy.
    ///
    /// # Errors
    /// Same as [`Model::explain`], plus [`PbError::InvalidConfig`] if the sparse
    /// fallback threshold is malformed.
    pub fn explain_with_budget(
        &self,
        x: &ServeBinnedMatrix,
        w: RefMeasure,
        budget: TableBudget,
    ) -> Result<TableBank, PbError> {
        if let crate::engine::ExactnessMode::Approximate { reason } = &self.mode {
            return Err(PbError::ExactnessFirewall(reason.clone()));
        }
        let n_features = self.provenance.len();
        if x.0.data.len() != n_features {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "serve matrix has {} columns, model has {n_features} features",
                    x.0.data.len()
                ),
            });
        }
        if x.0.grids.len() != n_features {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "serve matrix has {} grids, model has {n_features} features",
                    x.0.grids.len()
                ),
            });
        }
        if x.0.provenance.len() != n_features {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "serve matrix has {} provenance entries, model has {n_features} features",
                    x.0.provenance.len()
                ),
            });
        }
        if x.0.grids != self.grids {
            return Err(PbError::ShapeMismatch {
                what: "serve matrix grids do not match model grids".into(),
            });
        }
        if x.0.provenance != self.provenance {
            return Err(PbError::ShapeMismatch {
                what: "serve matrix provenance does not match model provenance".into(),
            });
        }
        let n_rows = x.0.n_rows as usize;
        for (a, col) in x.0.data.iter().enumerate() {
            if col.len() != n_rows {
                return Err(PbError::ShapeMismatch {
                    what: format!("serve column {a} len {} != n_rows {n_rows}", col.len()),
                });
            }
            let grid = self.grids.get(a).ok_or_else(|| PbError::Internal {
                what: "model grid disappeared during serve validation".into(),
            })?;
            for (row, &bin) in col.iter().enumerate() {
                if u16::from(bin) >= grid.n_bins {
                    return Err(PbError::InvalidInput {
                        what: format!(
                            "serve column {a} row {row} bin {bin} outside model grid n_bins {}",
                            grid.n_bins
                        ),
                    });
                }
            }
        }

        let grids = MergedGrids::from_model(self)?;
        let (mut raw, factored_supports) = accumulate(self, &grids, &budget)?;
        fold_correction_into_raw(self, &grids, &mut raw)?;
        let weights = build_weights(x, &grids, &w)?;
        // §08.10: shed each over-budget order-3 support per tree into the dense lower tables,
        // keeping the residual factored (no dense cube). No-op when none are over budget.
        let mut factored = Vec::with_capacity(factored_supports.len());
        for u in &factored_supports {
            factored.push(factored_shed_into_raw(
                self,
                u,
                &grids,
                &weights,
                &mut raw.tables,
            )?);
        }
        verify_raw_accumulation(self, &raw, &factored, &grids)?;
        let mut bank = purify(raw, &weights, &grids, PurifyMode::SinglePass)?;
        bank.factored = factored;
        fill_support(&mut bank, &grids, x)?;

        check_reconstruction(self, &bank)?;
        check_mass_conservation(self, &bank, &weights)?;
        check_purity(self, &bank, &weights)?;
        check_variance_sum(self, &bank, &weights)?;
        check_three_way_equal(self, &bank)?;
        Ok(bank)
    }
}

/// Fold a model's cell-basis correction (the §G1 adaptive fANOVA-cell refit) into the
/// raw per-support tensors, in place, BEFORE purify. Each correction table's raw delta is
/// added cell-for-cell on the merged grid; a zero raw table is inserted first when the
/// support was not a realized tree support (e.g. a pure main whose feature only appeared
/// inside higher-order trees). No-op when the model carries no correction.
///
/// This is the decompose half of the G0-exact symmetry: the SAME raw delta that
/// [`Model::correction_delta`] adds to the tree score is added here to the raw effects, so
/// `purify` re-expresses `trees + delta` losslessly and `bank.score == ensemble_f64`.
fn fold_correction_into_raw(
    model: &Model,
    grids: &MergedGrids,
    raw: &mut RawBank,
) -> Result<(), PbError> {
    let Some(bank) = &model.correction else {
        return Ok(());
    };
    for table in &bank.tables {
        // Support = sorted raw feature ids of the corrected axes (green-spine: raw == axis).
        let mut u_ids: SmallVec<[FeatureId; 3]> = SmallVec::new();
        for &axis in &table.axes {
            let raw_id = model
                .provenance
                .get(axis as usize)
                .ok_or_else(|| PbError::Internal {
                    what: format!("correction axis {axis} absent from provenance"),
                })?
                .raw;
            if !u_ids.contains(&raw_id) {
                u_ids.push(raw_id);
            }
        }
        u_ids.sort_unstable();
        let u = FeatureSet(u_ids.clone());
        // The correction's merged shape must match this model's merged grid for the support.
        let mut extents = Vec::with_capacity(u_ids.len());
        for r in &u_ids {
            extents.push(grids.cells(*r)?);
        }
        if extents.len() != table.shape.len()
            || extents
                .iter()
                .zip(&table.shape)
                .any(|(&e, &s)| e != s as usize)
        {
            return Err(PbError::Internal {
                what: format!(
                    "correction shape {:?} != merged extents {extents:?} for {u:?}",
                    table.shape
                ),
            });
        }
        if !raw.tables.contains_key(&u) {
            let mut axes = Vec::with_capacity(u_ids.len());
            for r in &u_ids {
                axes.push(grids.axis_id(*r)?);
            }
            raw.tables.insert(
                u.clone(),
                RawTable {
                    u: u.clone(),
                    axes,
                    values: Tensor::try_zeros(extents.clone())?,
                },
            );
        }
        let rt = raw.tables.get_mut(&u).ok_or_else(|| PbError::Internal {
            what: "correction raw table vanished after insert".into(),
        })?;
        // Add the raw delta cell-for-cell. `walk_extents` enumerates the merged cells in the
        // same row-major (last-axis-fastest) order the `values` vector was built in.
        let mut flat = 0usize;
        walk_extents(&extents, |tuple| {
            let v = *table.values.get(flat).ok_or_else(|| PbError::Internal {
                what: "correction flat index escaped values during fold".into(),
            })?;
            flat += 1;
            rt.values.add(tuple, v)
        })?;
    }
    Ok(())
}

/// Build a zero-valued cell-basis correction scaffold for `supports` (each a sorted list
/// of model axis ids of size 1..=3), at the model's merged-grid resolution. Fills `shape`
/// (merged cells per axis) and `bin_to_cell` (model bin → merged cell, the exact forward
/// map predict uses); leaves `values` zeroed for the §G1 solver to fill. Centralises the
/// merged-grid logic so predict, the decompose fold, and the solver share one cell layout.
///
/// # Errors
/// [`PbError::Internal`] if an axis is absent from the model or a bin/cell exceeds `u8`/`u32`.
pub(crate) fn correction_scaffold(
    model: &Model,
    supports: &[Vec<u32>],
) -> Result<crate::engine::CorrectionBank, PbError> {
    let grids = MergedGrids::from_model(model)?;
    let mut tables = Vec::with_capacity(supports.len());
    for axes in supports {
        let mut shape = Vec::with_capacity(axes.len());
        let mut bin_to_cell = Vec::with_capacity(axes.len());
        for &axis in axes {
            let raw = model
                .provenance
                .get(axis as usize)
                .ok_or_else(|| PbError::Internal {
                    what: format!("correction scaffold axis {axis} absent from provenance"),
                })?
                .raw;
            let n_cells = grids.cells(raw)?;
            shape.push(u32::try_from(n_cells).map_err(|_| PbError::Internal {
                what: "merged cell count exceeded u32".into(),
            })?);
            let grid = model
                .grids
                .get(axis as usize)
                .ok_or_else(|| PbError::Internal {
                    what: format!("correction scaffold axis {axis} absent from grids"),
                })?;
            let axis_merged = grids.axis(raw)?;
            let mut map = Vec::with_capacity(usize::from(grid.n_bins));
            for bin in 0..grid.n_bins {
                let b = u8::try_from(bin).map_err(|_| PbError::Internal {
                    what: "model bin exceeded u8 in correction scaffold".into(),
                })?;
                map.push(axis_merged.model_bin_to_cell(b)?);
            }
            bin_to_cell.push(map);
        }
        let cells: usize = shape.iter().map(|&s| s as usize).product();
        tables.push(crate::engine::CorrectionTable {
            axes: axes.clone(),
            shape,
            bin_to_cell,
            values: vec![0.0; cells],
        });
    }
    Ok(crate::engine::CorrectionBank { tables })
}

/// The pre-purify exact-accumulation checkpoint (spec §08.2): `f0 + Σ_u T_raw[u](x) ==
/// F_ens(x)` identically at every realized-support cell, before any purification runs.
/// A failure is an accumulation bug, not an invariant violation, so it surfaces as
/// [`PbError::Internal`].
fn verify_raw_accumulation(
    model: &Model,
    raw: &RawBank,
    factored: &[FactoredTriple],
    grids: &MergedGrids,
) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model).recon_tol;
    // Reuse the joint enumerator over the raw bank's realized features (+ factored supports).
    let mut set: BTreeSet<FeatureId> = model_features(model)?.into_iter().collect();
    for u in raw.tables.keys() {
        for r in &u.0 {
            set.insert(*r);
        }
    }
    for ft in factored {
        for r in &ft.u.0 {
            set.insert(*r);
        }
    }
    let feats: Vec<FeatureId> = set.into_iter().collect();
    enumerate_check_points(grids, &feats, |x_cells, rep_bins| {
        let ens = model.ensemble_f64(rep_bins)?;
        let mut acc = raw.f0;
        for rt in raw.tables.values() {
            let mut coord = Vec::with_capacity(rt.u.order());
            for a in &rt.axes {
                let cell = *x_cells
                    .get(a.raw.0 as usize)
                    .ok_or_else(|| PbError::Internal {
                        what: "raw checkpoint x_cells missing raw".into(),
                    })?;
                coord.push(cell as usize);
            }
            acc += rt.values.at(&coord).ok_or_else(|| PbError::Internal {
                what: "raw checkpoint coord out of range".into(),
            })?;
        }
        for ft in factored {
            acc += ft.eval(x_cells)?;
        }
        if (ens - acc).abs() > tol {
            return Err(PbError::Internal {
                what: "raw accumulation does not reconstruct the ensemble".into(),
            });
        }
        Ok(())
    })?;
    Ok(())
}

impl TableBank {
    /// `f0 + Σ_u f_u(x_u)` — the lossless LUT-sum score, equal to `F_ens` (spec §08.8).
    /// `x_cells[raw]` is the row's merged cell id on each raw feature.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`]/[`PbError::Internal`] if a table's axis is missing from
    /// `x_cells` or a coordinate escapes its tensor.
    pub fn score(&self, x_cells: &[u32]) -> Result<f64, PbError> {
        let mut acc = self.f0;
        for t in &self.tables {
            acc += t.eval(x_cells)?;
        }
        for ft in &self.factored {
            acc += ft.eval(x_cells)?;
        }
        Ok(acc)
    }

    /// Exact interventional Shapley values `φ_i(x) = Σ_{u ∋ i} f_u(x_u)/|u|` (spec §08.5),
    /// indexed by raw feature id. Sums to `score(x) − f0`. O(#tables) table reads, zero
    /// model calls.
    ///
    /// # Errors
    /// Propagates any [`EffectTable::eval`] failure.
    pub fn shap(&self, x_cells: &[u32]) -> Result<Vec<f64>, PbError> {
        let mut phi = vec![0.0_f64; self.merged_grids.len()];
        for t in &self.tables {
            let order = t.u.order().max(1) as f64;
            let share = t.eval(x_cells)? / order;
            for r in &t.u.0 {
                *phi.get_mut(r.0 as usize).ok_or_else(|| PbError::Internal {
                    what: "shap raw index escaped phi".into(),
                })? += share;
            }
        }
        for ft in &self.factored {
            let order = ft.u.order().max(1) as f64;
            let share = ft.eval(x_cells)? / order;
            for r in &ft.u.0 {
                *phi.get_mut(r.0 as usize).ok_or_else(|| PbError::Internal {
                    what: "shap raw index escaped phi".into(),
                })? += share;
            }
        }
        Ok(phi)
    }

    /// The exact Faith-Shap interaction index `Φ_S(x) = f_S(x_S)` for `|S| <= 3` (spec
    /// §08.5): the value of the table for support `s`, or `0.0` if `s` is not realized.
    ///
    /// # Errors
    /// Propagates any [`EffectTable::eval`] failure.
    pub fn faith_shap(&self, x_cells: &[u32], s: &FeatureSet) -> Result<f64, PbError> {
        for t in &self.tables {
            if &t.u == s {
                return t.eval(x_cells);
            }
        }
        for ft in &self.factored {
            if &ft.u == s {
                return ft.eval(x_cells);
            }
        }
        Ok(0.0)
    }

    /// Sobol importances `S_u = σ²(f_u)/σ²(F)` from the cached table variances (spec
    /// §08.5), sorted descending. Under product/uniform `w` they sum to ~1.
    #[must_use]
    pub fn sobol(&self) -> Vec<(FeatureSet, f64)> {
        let total: f64 = self.tables.iter().map(|t| t.variance).sum::<f64>()
            + self.factored.iter().map(|ft| ft.variance).sum::<f64>();
        let mut out: Vec<(FeatureSet, f64)> = self
            .tables
            .iter()
            .map(|t| (t.u.clone(), t.variance))
            .chain(self.factored.iter().map(|ft| (ft.u.clone(), ft.variance)))
            .map(|(u, v)| {
                let s = if total > 0.0 { v / total } else { 0.0 };
                (u, s)
            })
            .collect();
        // Sobol-descending with the feature set as an explicit secondary key, so the
        // ranking is total and stable regardless of table insertion order.
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out
    }

    /// The reference measure stamped on this bank (spec §08.8).
    #[must_use]
    pub fn reference_measure(&self) -> &RefMeasure {
        &self.w
    }

    /// Recompute the tables under a different reference measure `w` without retraining
    /// (spec §08.8) — exactness-preserving: the sum is conserved (Lengerich Cor. 2.2), so
    /// the bank still reconstructs `F_ens` and the model stays `Exact`. Re-purifies the
    /// current tables (which sum to `F_ens`) under the new `w`, carrying the per-cell
    /// `support` over unchanged.
    ///
    /// FLAG (spec §08.8 reconciliation): the spec signature passes a `ServeBinnedMatrix`,
    /// but the bank's cached `support` IS the merged-grid empirical marginal already, so
    /// v1 derives `w` from it and needs no serve matrix (and so no model) here.
    ///
    /// # Errors
    /// [`PbError::InvalidConfig`] for the v1-unsupported `Joint` measure; plus propagated
    /// grid/purify errors.
    pub fn recompute_under(&self, w: RefMeasure) -> Result<TableBank, PbError> {
        if !self.factored.is_empty() {
            return Err(PbError::InvalidConfig {
                what:
                    "recompute_under is unsupported for banks with factored high-order tables (v1)"
                        .into(),
            });
        }
        let grids = MergedGrids::from_border_grids(&self.merged_grids);
        let weights = build_weights_from_support(self, &w)?;
        // Seed a RawBank from the current (already-purified) tables: they sum to F_ens,
        // a valid input to purify under the new measure.
        let mut tables: BTreeMap<FeatureSet, RawTable> = BTreeMap::new();
        for t in &self.tables {
            tables.insert(
                t.u.clone(),
                RawTable {
                    u: t.u.clone(),
                    axes: t.axes.clone(),
                    values: t.values.clone(),
                },
            );
        }
        let raw = RawBank {
            f0: self.f0,
            tables,
        };
        let mut bank = purify(raw, &weights, &grids, PurifyMode::SinglePass)?;
        // Support is data-derived (w-independent), so carry it over by support key.
        for t in &mut bank.tables {
            if let Some(src) = self.tables.iter().find(|s| s.u == t.u) {
                t.support = src.support.clone();
            }
        }
        Ok(bank)
    }
}

impl MergedGrids {
    /// Reconstruct merged grids from a bank's stored per-feature [`BorderGrid`]s (the
    /// self-describing grid: cell boundaries ARE the borders, so the model-border index
    /// is `0..borders.len()`). Used by [`TableBank::recompute_under`], where no model is
    /// in hand. Only the borders/cells are consulted downstream (purify + support-derived
    /// weights), so the indices need only be internally consistent.
    pub(crate) fn from_border_grids(grids: &[BorderGrid]) -> Self {
        let per_raw = grids
            .iter()
            .enumerate()
            .map(|(r, g)| MergedAxis {
                axis: r,
                borders: g.borders.clone(),
                model_border_index: (0..g.borders.len()).collect(),
                model_n_bins: g.n_bins,
            })
            .collect();
        MergedGrids { per_raw }
    }
}

fn validate_bank_grid(grid: &BorderGrid, raw: usize) -> Result<(), PbError> {
    if grid.missing_bin != 0 {
        return Err(PbError::InvalidInput {
            what: format!("bank merged grid {raw} missing_bin must be 0"),
        });
    }
    let expected = grid
        .borders
        .len()
        .checked_add(2)
        .ok_or_else(|| PbError::Internal {
            what: "bank merged grid border count overflow".into(),
        })?;
    if usize::from(grid.n_bins) != expected {
        return Err(PbError::InvalidInput {
            what: format!(
                "bank merged grid {raw} n_bins {} inconsistent with {} borders",
                grid.n_bins,
                grid.borders.len()
            ),
        });
    }
    for (i, &border) in grid.borders.iter().enumerate() {
        if !border.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!("bank merged grid {raw} border {i} must be finite"),
            });
        }
    }
    for pair in grid.borders.windows(2) {
        if let [a, b] = pair {
            if a >= b {
                return Err(PbError::InvalidInput {
                    what: format!("bank merged grid {raw} borders must be strictly ascending"),
                });
            }
        }
    }
    Ok(())
}

fn effect_extents(grids: &MergedGrids, u: &FeatureSet) -> Result<Vec<usize>, PbError> {
    let mut extents = Vec::with_capacity(u.order());
    for raw in &u.0 {
        extents.push(grids.cells(*raw)?);
    }
    Ok(extents)
}

fn validate_raw_effect(grids: &MergedGrids, effect: &RawEffect) -> Result<(), PbError> {
    if !(1..=3).contains(&effect.u.order()) {
        return Err(PbError::InvalidInput {
            what: format!("raw effect order {} outside 1..=3", effect.u.order()),
        });
    }
    let extents = effect_extents(grids, &effect.u)?;
    if effect.values.shape() != extents {
        return Err(PbError::ShapeMismatch {
            what: "raw effect values shape does not match merged grid".into(),
        });
    }
    if effect.support.shape() != extents {
        return Err(PbError::ShapeMismatch {
            what: "raw effect support shape does not match merged grid".into(),
        });
    }
    Ok(())
}

fn marginal_counts_from_effect(
    effect: &RawEffect,
    raw: FeatureId,
    cells: usize,
) -> Result<Option<Vec<f64>>, PbError> {
    let Some(pos) = effect.u.0.iter().position(|r| *r == raw) else {
        return Ok(None);
    };
    let mut counts = vec![0.0_f64; cells];
    let extents = effect.support.shape();
    walk_extents(&extents, |coord| {
        let cell = *coord.get(pos).ok_or_else(|| PbError::Internal {
            what: "support marginal position escaped coordinate".into(),
        })?;
        let value = effect.support.at(coord).ok_or_else(|| PbError::Internal {
            what: "support marginal coordinate out of range".into(),
        })?;
        let slot = counts.get_mut(cell).ok_or_else(|| PbError::Internal {
            what: "support marginal cell escaped counts".into(),
        })?;
        *slot += value;
        Ok(())
    })?;
    let total: f64 = counts.iter().sum();
    if total > 0.0 {
        Ok(Some(counts))
    } else {
        Ok(None)
    }
}

fn axis_counts_from_effects(
    effects: &[RawEffect],
    raw: FeatureId,
    cells: usize,
) -> Result<Vec<f64>, PbError> {
    let mut candidates: Vec<&RawEffect> = effects.iter().filter(|e| e.u.contains(raw)).collect();
    candidates.sort_by_key(|e| e.u.order());
    for effect in candidates {
        if let Some(counts) = marginal_counts_from_effect(effect, raw, cells)? {
            return Ok(counts);
        }
    }
    Ok(vec![0.0_f64; cells])
}

fn build_weights_from_effect_support(
    grids: &[BorderGrid],
    effects: &[RawEffect],
    w: &RefMeasure,
) -> Result<WeightCache, PbError> {
    let laplace =
        match w {
            RefMeasure::Uniform => None,
            RefMeasure::ProductMarginals { laplace } => {
                if !laplace.is_finite() || *laplace <= 0.0 {
                    return Err(PbError::InvalidConfig {
                        what: "ProductMarginals laplace must be finite and > 0".into(),
                    });
                }
                Some(f64::from(*laplace))
            }
            RefMeasure::Joint => return Err(PbError::InvalidConfig {
                what:
                    "Joint reference measure is a v1.5 fork; v1 supports ProductMarginals/Uniform"
                        .into(),
            }),
        };

    let mut per_axis = Vec::with_capacity(grids.len());
    for (raw, grid) in grids.iter().enumerate() {
        validate_bank_grid(grid, raw)?;
        let cells = usize::from(grid.n_bins);
        let raw_w = match laplace {
            None => vec![1.0_f64 / cells as f64; cells],
            Some(lap) => {
                let raw_id = FeatureId(u32::try_from(raw).map_err(|_| PbError::InvalidInput {
                    what: "raw feature index exceeds u32".into(),
                })?);
                let counts = axis_counts_from_effects(effects, raw_id, cells)?;
                let n_total: f64 = counts.iter().sum();
                let unif = 1.0_f64 / cells as f64;
                let inv_n = if n_total > 0.0 { 1.0 / n_total } else { 0.0 };
                counts.iter().map(|c| unif + lap * (c * inv_n)).collect()
            }
        };
        let total: f64 = raw_w.iter().sum();
        if total.is_nan() || total <= 0.0 {
            return Err(PbError::Internal {
                what: "reference-measure axis weights summed to zero".into(),
            });
        }
        per_axis.push(raw_w.iter().map(|x| x / total).collect());
    }
    Ok(WeightCache {
        per_axis,
        kind: w.clone(),
    })
}

fn support_for_subset(
    support_by_u: &BTreeMap<FeatureSet, Tensor>,
    target: &FeatureSet,
    grids: &MergedGrids,
) -> Result<Option<Tensor>, PbError> {
    let mut candidates: Vec<(&FeatureSet, &Tensor)> = support_by_u
        .iter()
        .filter(|(u, _)| target.0.iter().all(|raw| u.contains(*raw)))
        .collect();
    candidates.sort_by_key(|(u, _)| u.order());

    if let Some((source_u, support)) = candidates.into_iter().next() {
        if source_u == target {
            return Ok(Some(support.clone()));
        }
        let mut positions = Vec::with_capacity(target.order());
        for raw in &target.0 {
            let pos = source_u
                .0
                .iter()
                .position(|source_raw| source_raw == raw)
                .ok_or_else(|| PbError::Internal {
                    what: "support subset search lost a raw feature".into(),
                })?;
            positions.push(pos);
        }
        let extents = effect_extents(grids, target)?;
        let mut out = Tensor::try_zeros(extents)?;
        let source_extents = support.shape();
        walk_extents(&source_extents, |coord| {
            let mut target_coord = Vec::with_capacity(target.order());
            for pos in &positions {
                target_coord.push(*coord.get(*pos).ok_or_else(|| PbError::Internal {
                    what: "support subset coordinate escaped source".into(),
                })?);
            }
            let v = support.at(coord).ok_or_else(|| PbError::Internal {
                what: "support subset source coordinate out of range".into(),
            })?;
            out.add(&target_coord, v)
        })?;
        return Ok(Some(out));
    }
    Ok(None)
}

/// Purify raw score-space effects on an explicit merged grid, carrying support
/// metadata through for callers that build exact banks without a [`Model`].
pub(crate) fn purify_raw_effects(
    f0: f64,
    merged_grids: Vec<BorderGrid>,
    w: RefMeasure,
    effects: Vec<RawEffect>,
) -> Result<TableBank, PbError> {
    for (raw, grid) in merged_grids.iter().enumerate() {
        validate_bank_grid(grid, raw)?;
    }
    let grids = MergedGrids::from_border_grids(&merged_grids);
    let weights = build_weights_from_effect_support(&merged_grids, &effects, &w)?;
    let mut tables = BTreeMap::new();
    let mut support_by_u = BTreeMap::new();
    for effect in effects {
        validate_raw_effect(&grids, &effect)?;
        let mut axes = Vec::with_capacity(effect.u.order());
        for raw in &effect.u.0 {
            axes.push(grids.axis_id(*raw)?);
        }
        if tables
            .insert(
                effect.u.clone(),
                RawTable {
                    u: effect.u.clone(),
                    axes,
                    values: effect.values,
                },
            )
            .is_some()
        {
            return Err(PbError::InvalidInput {
                what: "duplicate raw effect support".into(),
            });
        }
        support_by_u.insert(effect.u, effect.support);
    }
    let raw = RawBank { f0, tables };
    let mut bank = purify(raw, &weights, &grids, PurifyMode::SinglePass)?;
    for table in &mut bank.tables {
        if let Some(support) = support_for_subset(&support_by_u, &table.u, &grids)? {
            table.support = support;
        }
    }
    Ok(bank)
}

/// Build the per-axis cell weights from a bank's cached `support` tensors — the
/// merged-grid empirical marginal, identical to what [`build_weights`] computes from the
/// serve matrix (both count the same rows into the same cells). Lets `recompute_under`
/// change `w` without a serve matrix or the model.
fn build_weights_from_support(bank: &TableBank, w: &RefMeasure) -> Result<WeightCache, PbError> {
    let laplace =
        match w {
            RefMeasure::Uniform => None,
            RefMeasure::ProductMarginals { laplace } => {
                if !laplace.is_finite() || *laplace <= 0.0 {
                    return Err(PbError::InvalidConfig {
                        what: "ProductMarginals laplace must be finite and > 0".into(),
                    });
                }
                Some(f64::from(*laplace))
            }
            RefMeasure::Joint => return Err(PbError::InvalidConfig {
                what:
                    "Joint reference measure is a v1.5 fork; v1 supports ProductMarginals/Uniform"
                        .into(),
            }),
        };

    let mut per_axis = Vec::with_capacity(bank.merged_grids.len());
    for (r, g) in bank.merged_grids.iter().enumerate() {
        let cells = usize::from(g.n_bins);
        if cells == 0 {
            return Err(PbError::InvalidInput {
                what: format!("bank merged grid {r} has n_bins=0"),
            });
        }
        let raw_w = match laplace {
            None => vec![1.0_f64 / cells as f64; cells],
            Some(lap) => {
                // The main-effect support tensor for {r} is the per-cell row count.
                let main = bank
                    .tables
                    .iter()
                    .find(|t| t.u.0.len() == 1 && t.u.0.first() == Some(&FeatureId(r as u32)));
                let mut counts = vec![0.0_f64; cells];
                let mut n_total = 0.0_f64;
                if let Some(t) = main {
                    for c in 0..cells {
                        let v = t.support.at(&[c]).ok_or_else(|| PbError::Internal {
                            what: "support cell out of range for recompute weights".into(),
                        })?;
                        *counts.get_mut(c).ok_or_else(|| PbError::Internal {
                            what: "recompute weight cell escaped counts".into(),
                        })? = v;
                        n_total += v;
                    }
                }
                let unif = 1.0_f64 / cells as f64;
                let inv_n = if n_total > 0.0 { 1.0 / n_total } else { 0.0 };
                counts.iter().map(|c| unif + lap * (c * inv_n)).collect()
            }
        };
        let total: f64 = raw_w.iter().sum();
        if total.is_nan() || total <= 0.0 {
            return Err(PbError::Internal {
                what: "reference-measure axis weights summed to zero".into(),
            });
        }
        per_axis.push(raw_w.iter().map(|x| x / total).collect());
    }
    Ok(WeightCache {
        per_axis,
        kind: w.clone(),
    })
}

// ===========================================================================
// Hand-built fixtures (doc-hidden public; shared by unit + integration tests).
// ===========================================================================

fn fixture_grid() -> BorderGrid {
    BorderGrid {
        borders: vec![1.5],
        n_bins: 3,
        missing_bin: 0,
    }
}

fn fixture_schema() -> crate::engine::ModelSchema {
    use crate::cat::CatEncoderStore;
    use crate::loss::{Link, LossId, ObjectiveTag};
    crate::engine::ModelSchema {
        feature_names: vec!["x0".into(), "x1".into()],
        feature_kinds: vec![
            crate::data::AxisKind::Numeric,
            crate::data::AxisKind::Numeric,
        ],
        cat_encoders: CatEncoderStore::new(),
        class_labels: None,
        objective: ObjectiveTag {
            link: Link::Identity,
            loss: LossId::SquaredError,
            tweedie_rho: None,
        },
    }
}

/// A tiny exact model whose single depth-2 tree realizes
/// `g(1,1)=6, g(1,2)=2, g(2,1)=2, g(2,2)=0` (a genuine pairwise interaction).
#[doc(hidden)]
#[must_use]
pub fn fixture_model() -> Model {
    use crate::data::{AxisKind, AxisProvenance, FeatureId};
    use crate::engine::{ExactnessMode, ObliviousTree, Split};
    use crate::loss::Link;

    // Leaf index = bit0 | bit1<<1, bit = (bin <= bin_le). With bin_le = 1 and bins
    // {1,2}: bin1 → bit 1, bin2 → bit 0. So (2,2)→0→g=0, (1,2)→1→g=2, (2,1)→2→g=2,
    // (1,1)→3→g=6.
    let leaves = [0.0, 2.0, 2.0, 6.0, 0.0, 0.0, 0.0, 0.0];
    let tree = ObliviousTree {
        splits: vec![
            Split {
                axis: 0,
                bin_le: 1,
                missing_left: false,
            },
            Split {
                axis: 1,
                bin_le: 1,
                missing_left: false,
            },
        ],
        leaves,
        depth: 2,
    };
    Model {
        f0: 0.0,
        trees: vec![(1.0, tree)],
        grids: vec![fixture_grid(), fixture_grid()],
        provenance: vec![
            AxisProvenance {
                raw: FeatureId(0),
                kind: AxisKind::Numeric,
            },
            AxisProvenance {
                raw: FeatureId(1),
                kind: AxisKind::Numeric,
            },
        ],
        link: Link::Identity,
        mode: ExactnessMode::Exact,
        schema: fixture_schema(),
        schema_version: crate::serialize::SCHEMA_VERSION,
        correction: None,
    }
}

/// A 2-axis serve matrix exercising all four `(b0, b1) ∈ {1,2}²` cells of
/// [`fixture_model`] (so the empirical product marginals are balanced).
#[doc(hidden)]
#[must_use]
pub fn fixture_serve() -> ServeBinnedMatrix {
    use crate::data::BinnedMatrix;
    ServeBinnedMatrix(BinnedMatrix {
        data: vec![vec![1, 1, 2, 2], vec![1, 2, 1, 2]],
        n_rows: 4,
        grids: vec![fixture_grid(), fixture_grid()],
        provenance: fixture_model().provenance,
    })
}

/// A model whose tree spans 4 distinct raw features — an I1 violation (`depth = 4 > 3`).
/// Used as the negative `check_feature_budget` fixture.
#[doc(hidden)]
#[must_use]
pub fn fixture_over_budget_model() -> Model {
    use crate::data::{AxisKind, AxisProvenance, FeatureId};
    use crate::engine::{ExactnessMode, ObliviousTree, Split};
    use crate::loss::Link;

    let splits = (0..4u32)
        .map(|a| Split {
            axis: a,
            bin_le: 1,
            missing_left: false,
        })
        .collect();
    let provenance = (0..4u32)
        .map(|a| AxisProvenance {
            raw: FeatureId(a),
            kind: AxisKind::Numeric,
        })
        .collect();
    let tree = ObliviousTree {
        splits,
        leaves: [0.0; 8],
        depth: 4,
    };
    Model {
        f0: 0.0,
        trees: vec![(1.0, tree)],
        grids: vec![],
        provenance,
        link: Link::Identity,
        mode: ExactnessMode::Approximate {
            reason: "deliberately over-budget fixture".into(),
        },
        schema: fixture_schema(),
        schema_version: crate::serialize::SCHEMA_VERSION,
        correction: None,
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::float_cmp
    )]
    use super::*;
    use crate::data::{bin_columns, BinConfig};
    use crate::engine::{Booster, Config, FitSpec};
    use crate::loss::SquaredError;
    use proptest::prelude::*;

    fn fit_spec(loss: &SquaredError) -> FitSpec<'_> {
        FitSpec {
            loss,
            weight: None,
            exposure: None,
            monotone: crate::constraints::MonotoneMap::new(),
            interaction: crate::constraints::InteractionPolicy::default(),
            credibility: crate::constraints::CredibilityFloor::default(),
            seed: 0,
        }
    }

    fn fit(cols: &[Vec<f32>], y: &[f32], cfg: Config) -> (Model, ServeBinnedMatrix) {
        let refs: Vec<&[f32]> = cols.iter().map(Vec::as_slice).collect();
        let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
        let sqe = SquaredError;
        let model = Booster::with_config(cfg)
            .fit(&x, y, &fit_spec(&sqe))
            .unwrap();
        (model, ServeBinnedMatrix(x))
    }

    fn exact_cfg(n_trees: u32) -> Config {
        Config {
            n_trees,
            learning_rate: 1.0,
            lambda: 0.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            boosters: Default::default(),
        }
    }

    fn moderate_model() -> (Model, ServeBinnedMatrix) {
        // A 4-feature additive model with a non-trivial joint grid (each feature
        // realizes several borders), so the integral gates have something to integrate.
        let n = 120usize;
        let cols: Vec<Vec<f32>> = (0..4)
            .map(|f| (0..n).map(|i| ((i * 7 + f * 3) % 11) as f32).collect())
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                cols.iter()
                    .enumerate()
                    .map(|(f, c)| c[i] * (1.0 + f as f32))
                    .sum()
            })
            .collect();
        fit(
            &cols,
            &y,
            Config {
                n_trees: 40,
                learning_rate: 0.3,
                lambda: 1.0,
                ..exact_cfg(40)
            },
        )
    }

    /// Stage 1 (§08.10): the FACTORED order-3 effect reproduces the dense purified table's
    /// variance AND per-cell values bit-for-bit, under BOTH Uniform and the default
    /// ProductMarginals measure — without ever building the dense union cube.
    #[test]
    fn factored_triple_matches_dense_table() {
        // 3 features with a strong 3-way interaction so the booster realizes {0,1,2} trees.
        let n = 240usize;
        let cols: Vec<Vec<f32>> = (0..3)
            .map(|f| (0..n).map(|i| ((i * (f + 2)) % 6) as f32).collect())
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| cols[0][i] * cols[1][i] * cols[2][i] + cols[0][i] - cols[1][i])
            .collect();
        let (model, x) = fit(
            &cols,
            &y,
            Config {
                n_trees: 80,
                learning_rate: 0.3,
                lambda: 1.0,
                ..exact_cfg(80)
            },
        );
        let grids = MergedGrids::from_model(&model).unwrap();
        let n_features = model.provenance.len();
        for w_kind in [RefMeasure::Uniform, RefMeasure::default()] {
            let weights = build_weights(&x, &grids, &w_kind).unwrap();
            let bank = model.explain(&x, w_kind.clone()).unwrap();
            let t3 = bank
                .tables
                .iter()
                .find(|t| t.u.order() == 3)
                .expect("model should realize an order-3 table");
            let ft = FactoredTriple::from_model(&model, &t3.u, &grids, &weights).unwrap();
            let vf = ft.variance;
            assert!(
                (vf - t3.variance).abs() <= 1e-9 * (1.0 + t3.variance.abs()),
                "{w_kind:?}: factored var {vf} != dense var {}",
                t3.variance
            );
            // per-cell values over the full 3-way merged grid
            let cn: Vec<usize> = t3.u.0.iter().map(|r| grids.cells(*r).unwrap()).collect();
            let mut x_cells = vec![0u32; n_features];
            let mut max_abs = 0.0_f64;
            for c0 in 0..cn[0] {
                for c1 in 0..cn[1] {
                    for c2 in 0..cn[2] {
                        x_cells[t3.u.0[0].0 as usize] = c0 as u32;
                        x_cells[t3.u.0[1].0 as usize] = c1 as u32;
                        x_cells[t3.u.0[2].0 as usize] = c2 as u32;
                        let fe = ft.eval(&x_cells).unwrap();
                        let de = t3.eval(&x_cells).unwrap();
                        max_abs = max_abs.max((fe - de).abs());
                    }
                }
            }
            assert!(
                max_abs <= 1e-9,
                "{w_kind:?}: factored eval max abs diff {max_abs}"
            );
        }
    }

    /// Stage 2 (§08.10): shedding the over-budget 3-way per tree into the RAW lower tables,
    /// then running the normal order-2->1 cascade, reproduces the dense bank EXACTLY — every
    /// lower table cell-for-cell, f0, and the 3-way residual (variance + per-cell eval).
    #[test]
    fn factored_shed_reproduces_dense_bank() {
        let n = 240usize;
        let cols: Vec<Vec<f32>> = (0..3)
            .map(|f| (0..n).map(|i| ((i * (f + 2)) % 6) as f32).collect())
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| cols[0][i] * cols[1][i] * cols[2][i] + cols[0][i] - cols[1][i])
            .collect();
        let (model, x) = fit(
            &cols,
            &y,
            Config {
                n_trees: 80,
                learning_rate: 0.3,
                lambda: 1.0,
                ..exact_cfg(80)
            },
        );
        let grids = MergedGrids::from_model(&model).unwrap();
        for w_kind in [RefMeasure::Uniform, RefMeasure::default()] {
            let weights = build_weights(&x, &grids, &w_kind).unwrap();
            let bank_dense = model.explain(&x, w_kind.clone()).unwrap();
            let u3 = bank_dense
                .tables
                .iter()
                .find(|t| t.u.order() == 3)
                .expect("order-3 table")
                .u
                .clone();
            // The 3-way must be fed by >1 tree so the variance cross-terms Σ_{t≠s}⟨p_t,p_s⟩
            // and the multi-tree shed are actually exercised (not an incidental single box).
            let n_u3_trees = model
                .trees
                .iter()
                .filter(|(_, t)| tree_support(&model, t).is_ok_and(|s| s == u3))
                .count();
            assert!(
                n_u3_trees > 1,
                "{w_kind:?}: need >1 tree feeding the 3-way support"
            );

            // Factored path: accumulate raw, shed the 3-way per tree into the raw lower
            // tables, drop the 3-way raw table, then run the standard order-2->1 cascade.
            let (raw, _) = accumulate(&model, &grids, &TableBudget::default()).unwrap();
            let RawBank { f0, mut tables } = raw;
            let ft = factored_shed_into_raw(&model, &u3, &grids, &weights, &mut tables).unwrap();
            assert!(!tables.contains_key(&u3), "3-way raw table must be removed");
            let bank_fac = purify(
                RawBank { f0, tables },
                &weights,
                &grids,
                PurifyMode::SinglePass,
            )
            .unwrap();

            assert!(
                (bank_fac.f0 - bank_dense.f0).abs() <= 1e-9 * (1.0 + bank_dense.f0.abs()),
                "{w_kind:?}: f0 {} != {}",
                bank_fac.f0,
                bank_dense.f0
            );
            assert_eq!(
                bank_fac.tables.len(),
                bank_dense.tables.len() - 1,
                "{w_kind:?}: factored bank should hold every dense table except the 3-way"
            );
            for td in &bank_dense.tables {
                if td.u == u3 {
                    let vf = ft.variance;
                    assert!(
                        (vf - td.variance).abs() <= 1e-9 * (1.0 + td.variance.abs()),
                        "{w_kind:?}: factored 3-way var {vf} != dense {}",
                        td.variance
                    );
                    let cn: Vec<usize> = td.u.0.iter().map(|r| grids.cells(*r).unwrap()).collect();
                    let mut x_cells = vec![0u32; model.provenance.len()];
                    let mut max_abs = 0.0_f64;
                    for c0 in 0..cn[0] {
                        for c1 in 0..cn[1] {
                            for c2 in 0..cn[2] {
                                x_cells[td.u.0[0].0 as usize] = c0 as u32;
                                x_cells[td.u.0[1].0 as usize] = c1 as u32;
                                x_cells[td.u.0[2].0 as usize] = c2 as u32;
                                let d =
                                    (ft.eval(&x_cells).unwrap() - td.eval(&x_cells).unwrap()).abs();
                                max_abs = max_abs.max(d);
                            }
                        }
                    }
                    assert!(
                        max_abs <= 1e-9,
                        "{w_kind:?}: factored 3-way eval diff {max_abs}"
                    );
                    continue;
                }
                let tf = bank_fac
                    .tables
                    .iter()
                    .find(|t| t.u == td.u)
                    .unwrap_or_else(|| panic!("{w_kind:?}: factored bank missing {:?}", td.u));
                assert!(
                    (tf.variance - td.variance).abs() <= 1e-9 * (1.0 + td.variance.abs()),
                    "{w_kind:?}: table {:?} var {} != {}",
                    td.u,
                    tf.variance,
                    td.variance
                );
                let vf = tf.values.values();
                let vd = td.values.values();
                assert_eq!(
                    vf.len(),
                    vd.len(),
                    "{w_kind:?}: table {:?} cell count",
                    td.u
                );
                let mut max_abs = 0.0_f64;
                for (a, b) in vf.iter().zip(vd.iter()) {
                    max_abs = max_abs.max((a - b).abs());
                }
                assert!(
                    max_abs <= 1e-9,
                    "{w_kind:?}: table {:?} cell diff {max_abs}",
                    td.u
                );
            }
        }
    }

    /// Stage 3 (§08.10) END-TO-END: a model whose 3-way table exceeds the cell budget now
    /// produces a VALID bank via the factored path instead of `PbError::TableBudget`. The
    /// `Ok(_)` itself means all five I2 gates passed on the factored bank; we also check it
    /// equals the dense bank (score everywhere + total variance) and is structurally factored.
    #[test]
    fn explain_factors_over_budget_three_way_and_passes_all_gates() {
        let n = 240usize;
        let cols: Vec<Vec<f32>> = (0..3)
            .map(|f| (0..n).map(|i| ((i * (f + 2)) % 6) as f32).collect())
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| cols[0][i] * cols[1][i] * cols[2][i] + cols[0][i] - cols[1][i])
            .collect();
        let (model, x) = fit(
            &cols,
            &y,
            Config {
                n_trees: 80,
                learning_rate: 0.3,
                lambda: 1.0,
                ..exact_cfg(80)
            },
        );
        let grids = MergedGrids::from_model(&model).unwrap();
        let n_features = model.provenance.len();

        // Dense reference (huge budget, hard-error policy — the old behavior).
        let dense_budget = TableBudget {
            max_table_cells: 2_000_000,
            max_bank_cells: 32_000_000,
            on_overflow: OverflowPolicy::Error,
        };
        let bank_dense = model
            .explain_with_budget(&x, RefMeasure::default(), dense_budget)
            .unwrap();
        let u3 = bank_dense
            .tables
            .iter()
            .find(|t| t.u.order() == 3)
            .expect("order-3 table")
            .u
            .clone();

        // Budget that forces ONLY the 3-way over: just above the largest order-2 table.
        let max_2way = bank_dense
            .tables
            .iter()
            .filter(|t| t.u.order() == 2)
            .map(|t| t.values.values().len())
            .max()
            .unwrap_or(1);
        let cells3: usize = u3.0.iter().map(|r| grids.cells(*r).unwrap()).product();
        assert!(
            cells3 > max_2way + 1,
            "3-way ({cells3}) must exceed the budget ({max_2way})"
        );
        let budget = TableBudget {
            max_table_cells: (max_2way + 1) as u64,
            max_bank_cells: 32_000_000,
            on_overflow: OverflowPolicy::Factored,
        };

        // The Ok here means accumulate -> shed -> purify -> ALL FIVE I2 GATES succeeded.
        let bank_fac = model
            .explain_with_budget(&x, RefMeasure::default(), budget)
            .expect("factored explain must pass all five exactness gates");

        // Structurally factored, not dense.
        assert_eq!(
            bank_fac.factored.len(),
            1,
            "exactly the one 3-way is factored"
        );
        assert_eq!(bank_fac.factored[0].u, u3);
        assert!(
            bank_fac.tables.iter().all(|t| t.u.order() < 3),
            "no dense order-3 table should remain"
        );

        // Score equals the dense bank at every merged cell of the 3 features.
        let cells_per: Vec<usize> = (0..n_features)
            .map(|r| grids.cells(FeatureId(r as u32)).unwrap())
            .collect();
        let mut x_cells = vec![0u32; n_features];
        let mut max_score_diff = 0.0_f64;
        for c0 in 0..cells_per[0] {
            for c1 in 0..cells_per[1] {
                for c2 in 0..cells_per[2] {
                    x_cells[0] = c0 as u32;
                    x_cells[1] = c1 as u32;
                    x_cells[2] = c2 as u32;
                    let d = (bank_fac.score(&x_cells).unwrap()
                        - bank_dense.score(&x_cells).unwrap())
                    .abs();
                    max_score_diff = max_score_diff.max(d);
                }
            }
        }
        assert!(
            max_score_diff <= 1e-9,
            "factored score diverges from dense by {max_score_diff}"
        );

        // Total variance (dense tables + factored) matches the dense bank.
        let var_fac: f64 = bank_fac.tables.iter().map(|t| t.variance).sum::<f64>()
            + bank_fac.factored.iter().map(|f| f.variance).sum::<f64>();
        let var_dense: f64 = bank_dense.tables.iter().map(|t| t.variance).sum();
        assert!(
            (var_fac - var_dense).abs() <= 1e-9 * (1.0 + var_dense.abs()),
            "factored total variance {var_fac} != dense {var_dense}"
        );
    }

    /// A factored high-order effect is a SUM of per-tree purified boxes, not a dense table,
    /// so the rebase-a-cell-into-f0 operation is undefined for it. A rating basis that names
    /// a factored support must error loudly (a deployer trying to "anchor" a 3-way), while a
    /// basis naming a still-dense order-2 support in the same factored bank rebases normally.
    #[test]
    fn rating_basis_refuses_to_rebase_factored_effect() {
        use crate::serialize::{RatingBasis, RatingReference};

        let n = 240usize;
        let cols: Vec<Vec<f32>> = (0..3)
            .map(|f| (0..n).map(|i| ((i * (f + 2)) % 6) as f32).collect())
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| cols[0][i] * cols[1][i] * cols[2][i] + cols[0][i] - cols[1][i])
            .collect();
        let (model, x) = fit(
            &cols,
            &y,
            Config {
                n_trees: 80,
                learning_rate: 0.3,
                lambda: 1.0,
                ..exact_cfg(80)
            },
        );
        // Budget just above the largest order-2 table forces ONLY the 3-way to factor.
        let bank_dense = model
            .explain_with_budget(&x, RefMeasure::default(), TableBudget::default())
            .unwrap();
        let max_2way = bank_dense
            .tables
            .iter()
            .filter(|t| t.u.order() == 2)
            .map(|t| t.values.values().len())
            .max()
            .unwrap();
        let bank = model
            .explain_with_budget(
                &x,
                RefMeasure::default(),
                TableBudget {
                    max_table_cells: (max_2way + 1) as u64,
                    max_bank_cells: 32_000_000,
                    on_overflow: OverflowPolicy::Factored,
                },
            )
            .unwrap();
        assert_eq!(bank.factored.len(), 1, "exactly one 3-way is factored");

        // A basis naming the factored 3-way support is rejected.
        let factored_set: Vec<u32> = bank.factored[0].u.0.iter().map(|f| f.0).collect();
        let bad = RatingBasis {
            reference: vec![RatingReference {
                feature_set: factored_set,
                coord: vec![0, 0, 0],
            }],
        };
        assert!(matches!(
            bank.to_rating_export(model.link, &model.mode, &model.schema, Some(&bad)),
            Err(PbError::InvalidConfig { .. })
        ));

        // A basis naming a still-dense order-2 support in the SAME bank rebases fine.
        let pair = bank
            .tables
            .iter()
            .find(|t| t.u.order() == 2)
            .expect("a dense order-2 table survives")
            .u
            .clone();
        let ok = RatingBasis {
            reference: vec![RatingReference {
                feature_set: pair.0.iter().map(|f| f.0).collect(),
                coord: vec![0, 0],
            }],
        };
        bank.to_rating_export(model.link, &model.mode, &model.schema, Some(&ok))
            .unwrap();
    }

    /// Stage 3 (§08.10): MULTIPLE over-budget triples sharing a pair. A non-separable
    /// `x0·x1·x2 − x0·x1·x3` signal forces both {0,1,2} and {0,1,3}, so the shared {0,1}
    /// table receives sheds from BOTH — the case a single-triple test cannot exercise.
    /// All five gates still pass and the bank stays dense-equivalent.
    #[test]
    fn explain_factors_multiple_over_budget_three_ways_sharing_a_pair() {
        let n = 6usize.pow(4);
        let cols: Vec<Vec<f32>> = (0..4)
            .map(|f| {
                (0..n)
                    .map(|i| ((i / 6usize.pow(f as u32)) % 6) as f32)
                    .collect()
            })
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| cols[0][i] * cols[1][i] * cols[2][i] - cols[0][i] * cols[1][i] * cols[3][i])
            .collect();
        let (model, x) = fit(
            &cols,
            &y,
            Config {
                n_trees: 120,
                learning_rate: 0.3,
                lambda: 1.0,
                ..exact_cfg(120)
            },
        );
        let grids = MergedGrids::from_model(&model).unwrap();
        let n_features = model.provenance.len();

        let dense_budget = TableBudget {
            max_table_cells: 2_000_000,
            max_bank_cells: 32_000_000,
            on_overflow: OverflowPolicy::Error,
        };
        let bank_dense = model
            .explain_with_budget(&x, RefMeasure::default(), dense_budget)
            .unwrap();
        let n_triples = bank_dense
            .tables
            .iter()
            .filter(|t| t.u.order() == 3)
            .count();
        assert!(
            n_triples >= 2,
            "need >=2 realized 3-way supports, got {n_triples}"
        );
        let shares_a_pair = {
            let triples: Vec<&FeatureSet> = bank_dense
                .tables
                .iter()
                .filter(|t| t.u.order() == 3)
                .map(|t| &t.u)
                .collect();
            triples.iter().enumerate().any(|(i, a)| {
                triples
                    .iter()
                    .skip(i + 1)
                    .any(|b| a.0.iter().filter(|r| b.0.contains(r)).count() == 2)
            })
        };
        assert!(shares_a_pair, "expected two triples sharing a pair");

        let max_2way = bank_dense
            .tables
            .iter()
            .filter(|t| t.u.order() == 2)
            .map(|t| t.values.values().len())
            .max()
            .unwrap_or(1);
        let budget = TableBudget {
            max_table_cells: (max_2way + 1) as u64,
            max_bank_cells: 32_000_000,
            on_overflow: OverflowPolicy::Factored,
        };
        let bank_fac = model
            .explain_with_budget(&x, RefMeasure::default(), budget)
            .expect("multi-triple factored explain must pass all five gates");
        assert_eq!(
            bank_fac.factored.len(),
            n_triples,
            "every triple should be factored"
        );
        assert!(bank_fac.tables.iter().all(|t| t.u.order() < 3));

        let cn: Vec<usize> = (0..n_features)
            .map(|r| grids.cells(FeatureId(r as u32)).unwrap())
            .collect();
        let mut x_cells = vec![0u32; n_features];
        let mut maxd = 0.0_f64;
        for c0 in 0..cn[0] {
            for c1 in 0..cn[1] {
                for c2 in 0..cn[2] {
                    for c3 in 0..cn[3] {
                        x_cells[0] = c0 as u32;
                        x_cells[1] = c1 as u32;
                        x_cells[2] = c2 as u32;
                        x_cells[3] = c3 as u32;
                        let d = (bank_fac.score(&x_cells).unwrap()
                            - bank_dense.score(&x_cells).unwrap())
                        .abs();
                        maxd = maxd.max(d);
                    }
                }
            }
        }
        assert!(
            maxd <= 1e-9,
            "multi-triple factored score diverges by {maxd}"
        );
    }

    /// Stage 5 (§08.10): the rating export emits factored effects as per-tree boxes, and
    /// evaluating those PUBLISHED boxes (threshold routing, the deployment/SQL form) exactly
    /// reproduces the in-bank factored eval — i.e. the export is a complete, lossless artifact.
    #[test]
    fn factored_rating_export_roundtrips_to_bank_eval() {
        let n = 240usize;
        let cols: Vec<Vec<f32>> = (0..3)
            .map(|f| (0..n).map(|i| ((i * (f + 2)) % 6) as f32).collect())
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| cols[0][i] * cols[1][i] * cols[2][i] + cols[0][i] - cols[1][i])
            .collect();
        let (model, x) = fit(
            &cols,
            &y,
            Config {
                n_trees: 80,
                learning_rate: 0.3,
                lambda: 1.0,
                ..exact_cfg(80)
            },
        );
        let grids = MergedGrids::from_model(&model).unwrap();
        let dense = model
            .explain_with_budget(
                &x,
                RefMeasure::default(),
                TableBudget {
                    max_table_cells: 2_000_000,
                    max_bank_cells: 32_000_000,
                    on_overflow: OverflowPolicy::Error,
                },
            )
            .unwrap();
        let max_2way = dense
            .tables
            .iter()
            .filter(|t| t.u.order() == 2)
            .map(|t| t.values.values().len())
            .max()
            .unwrap_or(1);
        let bank = model
            .explain_with_budget(
                &x,
                RefMeasure::default(),
                TableBudget {
                    max_table_cells: (max_2way + 1) as u64,
                    max_bank_cells: 32_000_000,
                    on_overflow: OverflowPolicy::Factored,
                },
            )
            .unwrap();
        assert!(!bank.factored.is_empty());

        let export = bank
            .to_rating_export(model.link, &model.mode, &model.schema, None)
            .unwrap();
        assert_eq!(export.factored.len(), bank.factored.len());
        assert!(
            export.tables.iter().all(|t| t.feature_set.order() < 3),
            "no dense order-3 table should be exported"
        );

        // Evaluate each EXPORTED factored effect (threshold routing in cell space) and confirm
        // it matches the in-bank eval at every merged cell of its support.
        for (ft, rf) in bank.factored.iter().zip(export.factored.iter()) {
            assert_eq!(rf.feature_set, ft.u);
            let cn: Vec<usize> = ft.u.0.iter().map(|r| grids.cells(*r).unwrap()).collect();
            let mut x_cells = vec![0u32; model.provenance.len()];
            for c0 in 0..cn[0] {
                for c1 in 0..cn[1] {
                    for c2 in 0..cn[2] {
                        let cells = [c0, c1, c2];
                        x_cells[ft.u.0[0].0 as usize] = c0 as u32;
                        x_cells[ft.u.0[1].0 as usize] = c1 as u32;
                        x_cells[ft.u.0[2].0 as usize] = c2 as u32;
                        let mut from_export = 0.0_f64;
                        for bx in &rf.boxes {
                            let mut idx = 0usize;
                            for (d, ((&cell, axis), (&thr, &miss))) in cells
                                .iter()
                                .zip(ft.axes.iter())
                                .zip(bx.thresholds.iter().zip(bx.missing_left.iter()))
                                .enumerate()
                            {
                                let low = if cell == 0 {
                                    miss
                                } else {
                                    // finite cell is low iff cell <= j+1, where borders[j] == thr
                                    let j = axis.borders.iter().position(|&b| b == thr).unwrap();
                                    cell <= j + 1
                                };
                                idx |= usize::from(low) << d;
                            }
                            from_export += bx.octants[idx];
                        }
                        let from_bank = ft.eval(&x_cells).unwrap();
                        assert!(
                            (from_export - from_bank).abs() <= 1e-12,
                            "exported box eval {from_export} != bank eval {from_bank}"
                        );
                    }
                }
            }
        }
    }

    /// F5 (§08.10 deployment): the COMPLETE rating export — intercept + every dense table +
    /// every factored effect together — is a self-sufficient scoring artifact. A scorer that
    /// reads ONLY the export (no bank, no model) reproduces `bank.score` at every cell, and the
    /// reconstruction gate already equates `bank.score` to the ensemble. This closes the chain
    /// the per-component tests only cover in pieces: dense values are copied verbatim and the
    /// factored boxes are checked alone, but their COMPOSITION (f0 + dense + factored summed at
    /// one input) is exercised here. The factored boxes ship raw-value thresholds only, so the
    /// standalone scorer recovers each feature's borders from the dense tables that name it —
    /// proving the export is closed under scoring without re-consulting the model.
    #[test]
    fn rating_export_is_a_complete_standalone_scorer() {
        use std::collections::HashMap;
        let n = 240usize;
        let cols: Vec<Vec<f32>> = (0..3)
            .map(|f| (0..n).map(|i| ((i * (f + 2)) % 6) as f32).collect())
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| cols[0][i] * cols[1][i] * cols[2][i] + cols[0][i] - cols[1][i])
            .collect();
        let (model, x) = fit(
            &cols,
            &y,
            Config {
                n_trees: 80,
                learning_rate: 0.3,
                lambda: 1.0,
                ..exact_cfg(80)
            },
        );
        let grids = MergedGrids::from_model(&model).unwrap();
        let n_features = model.provenance.len();

        let dense = model
            .explain_with_budget(&x, RefMeasure::default(), TableBudget::default())
            .unwrap();
        let max_2way = dense
            .tables
            .iter()
            .filter(|t| t.u.order() == 2)
            .map(|t| t.values.values().len())
            .max()
            .unwrap();
        let bank = model
            .explain_with_budget(
                &x,
                RefMeasure::default(),
                TableBudget {
                    max_table_cells: (max_2way + 1) as u64,
                    max_bank_cells: 32_000_000,
                    on_overflow: OverflowPolicy::Factored,
                },
            )
            .unwrap();
        assert!(
            !bank.factored.is_empty(),
            "the 3-way must factor so composition includes a factored term"
        );

        let export = bank
            .to_rating_export(model.link, &model.mode, &model.schema, None)
            .unwrap();

        // Recover per-feature borders from the dense tables (factored boxes carry only
        // raw-value thresholds). Every feature appears in at least its main-effect table.
        let mut borders_of: HashMap<u32, Vec<f32>> = HashMap::new();
        for t in &export.tables {
            for a in &t.axes {
                borders_of.entry(a.raw).or_insert_with(|| a.borders.clone());
            }
        }

        // A scorer that consults ONLY `export` (+ the borders gathered from it).
        let score_from_export = |x_cells: &[u32]| -> f64 {
            let mut s = export.f0;
            for t in &export.tables {
                let mut idx = 0usize;
                for (d, a) in t.axes.iter().enumerate() {
                    idx = idx * t.shape[d] as usize + x_cells[a.raw as usize] as usize;
                }
                s += t.values[idx];
            }
            for rf in &export.factored {
                for bx in &rf.boxes {
                    let mut oct = 0usize;
                    for (d, raw) in rf.feature_set.0.iter().enumerate() {
                        let cell = x_cells[raw.0 as usize] as usize;
                        let low = if cell == 0 {
                            bx.missing_left[d]
                        } else {
                            let bs = &borders_of[&raw.0];
                            let j = bs.iter().position(|&b| b == bx.thresholds[d]).unwrap();
                            cell <= j + 1
                        };
                        oct |= usize::from(low) << d;
                    }
                    s += bx.octants[oct];
                }
            }
            s
        };

        let cells_per: Vec<usize> = (0..n_features)
            .map(|r| grids.cells(FeatureId(r as u32)).unwrap())
            .collect();
        let mut x_cells = vec![0u32; n_features];
        let mut max_diff = 0.0_f64;
        for c0 in 0..cells_per[0] {
            for c1 in 0..cells_per[1] {
                for c2 in 0..cells_per[2] {
                    x_cells[0] = c0 as u32;
                    x_cells[1] = c1 as u32;
                    x_cells[2] = c2 as u32;
                    let d = (score_from_export(&x_cells) - bank.score(&x_cells).unwrap()).abs();
                    max_diff = max_diff.max(d);
                }
            }
        }
        assert!(
            max_diff <= 1e-12,
            "standalone export scorer diverges from bank by {max_diff}"
        );
    }

    /// F4 (§08.10 + §09.5): an OuterBag model (a tree-soup of all bagged members) whose 3-way
    /// support exceeds the cell budget still produces a VALID factored bank — bagging composes
    /// with the factored path, so the user-facing bagging at competitive fidelity stays
    /// exactly decomposable. (average_banks/recompute_under are internal, not on this path.)
    #[test]
    fn outer_bag_over_budget_three_way_stays_factored_and_exact() {
        use crate::boosters::{BoosterConfig, EnsembleSpec};
        let n = 240usize;
        let cols: Vec<Vec<f32>> = (0..3)
            .map(|f| (0..n).map(|i| ((i * (f + 2)) % 6) as f32).collect())
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| cols[0][i] * cols[1][i] * cols[2][i] + cols[0][i] - cols[1][i])
            .collect();
        let (model, x) = fit(
            &cols,
            &y,
            Config {
                n_trees: 80,
                learning_rate: 0.3,
                lambda: 1.0,
                boosters: BoosterConfig {
                    ensemble: EnsembleSpec::OuterBag {
                        n_bags: 2,
                        bag_subsample: 1.0,
                    },
                    ..Default::default()
                },
                ..exact_cfg(80)
            },
        );
        let grids = MergedGrids::from_model(&model).unwrap();
        let dense = model
            .explain_with_budget(
                &x,
                RefMeasure::default(),
                TableBudget {
                    max_table_cells: 2_000_000,
                    max_bank_cells: 32_000_000,
                    on_overflow: OverflowPolicy::Error,
                },
            )
            .unwrap();
        let u3 = dense
            .tables
            .iter()
            .find(|t| t.u.order() == 3)
            .expect("bagged soup should realize a 3-way")
            .u
            .clone();
        let max_2way = dense
            .tables
            .iter()
            .filter(|t| t.u.order() == 2)
            .map(|t| t.values.values().len())
            .max()
            .unwrap_or(1);
        let cells3: usize = u3.0.iter().map(|r| grids.cells(*r).unwrap()).product();
        assert!(
            cells3 > max_2way + 1,
            "3-way ({cells3}) must exceed the budget"
        );
        // The Ok here = all five I2 gates passed on the BAGGED soup's factored bank.
        let bank = model
            .explain_with_budget(
                &x,
                RefMeasure::default(),
                TableBudget {
                    max_table_cells: (max_2way + 1) as u64,
                    max_bank_cells: 32_000_000,
                    on_overflow: OverflowPolicy::Factored,
                },
            )
            .expect("OuterBag + factored must pass all five exactness gates");
        assert!(
            !bank.factored.is_empty(),
            "the over-budget 3-way should be factored"
        );
        assert!(bank.tables.iter().all(|t| t.u.order() < 3));
    }

    /// MassConservation fix: `ensemble_w_mean` (the new exact per-tree integral) equals
    /// the exhaustive joint-grid `Σ w·F_ens` — so mass is exact with NO joint enumeration.
    #[test]
    fn ensemble_w_mean_equals_exhaustive_joint_integral() {
        let (model, x) = moderate_model();
        let grids = MergedGrids::from_model(&model).unwrap();
        let w = build_weights(&x, &grids, &RefMeasure::default()).unwrap();
        let per_tree = ensemble_w_mean(&model, &grids, &w).unwrap();
        // Independent exhaustive joint-grid integral over all realized features.
        let feats =
            gate_features(&model, &model.explain(&x, RefMeasure::default()).unwrap()).unwrap();
        let mut joint = 0.0_f64;
        enumerate_check_points(&grids, &feats, |x_cells, rep_bins| {
            joint += joint_weight(&w, &feats, x_cells)? * model.ensemble_f64(rep_bins)?;
            Ok(())
        })
        .unwrap();
        assert!(
            (per_tree - joint).abs() < 1e-9 * (1.0 + joint.abs()),
            "per-tree mass {per_tree} != exhaustive joint mass {joint}"
        );
    }

    /// REGRESSION (high-dimensional decomposition): a model wide enough that the joint-grid
    /// cell count `Π(cells)` exceeds `u64` (here 70 binary-cell axes => `2^70`) must SAMPLE the
    /// gate sweep, not hard-error. This was the allstate (130-feature) `tables()` crash —
    /// `enumerate_check_points` computed the product as `u64` and overflowed BEFORE the cap
    /// check could route it to sampling. The fix (`saturating_product_u64`) caps at `u64::MAX`.
    #[test]
    fn wide_model_joint_grid_samples_instead_of_overflowing() {
        let n = 70usize;
        let per_raw = (0..n)
            .map(|i| MergedAxis {
                axis: i,
                borders: vec![], // cells() = 2 per axis => 2^70 total, overflows u64
                model_border_index: vec![],
                model_n_bins: 2,
            })
            .collect();
        let grids = MergedGrids { per_raw };
        let feats: Vec<FeatureId> = (0..n as u32).map(FeatureId).collect();

        TEST_JOINT_CAP.with(|c| c.set(64));
        let mut visited = 0usize;
        let sampled = enumerate_check_points(&grids, &feats, |_x_cells, _rep_bins| {
            visited += 1;
            Ok(())
        });
        TEST_JOINT_CAP.with(|c| c.set(0));

        let sampled = sampled.expect("a >u64 joint grid must sample, not overflow-error");
        assert!(
            sampled,
            "a 2^70-cell joint grid must take the SAMPLING branch"
        );
        assert_eq!(visited, 64, "sampling emits exactly joint_cap points");
    }

    /// The integral gates stay correct when the joint grid is SAMPLED (the >JOINT_CAP
    /// path, forced here with a tiny test cap on a small model). MassConservation is
    /// exact (never samples); the self-normalized sampled mean still recovers f0 (the
    /// un-normalized partial sum — the prior bug — would be a tiny fraction of f0).
    #[test]
    fn integral_gates_correct_under_forced_sampling() {
        let (model, x) = moderate_model();
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        let grids = MergedGrids::from_model(&model).unwrap();
        let feats = gate_features(&model, &bank).unwrap();
        let w = build_weights(&x, &grids, &RefMeasure::default()).unwrap();

        // Force the SAMPLED branch on this small model.
        TEST_JOINT_CAP.with(|c| c.set(64));

        // Mass is exact (per-tree, ignores the cap); Reconstruction/ThreeWayEqual are
        // per-point and sound under sampling — all pass.
        check_mass_conservation(&model, &bank, &w).unwrap();
        check_reconstruction(&model, &bank).unwrap();
        check_three_way_equal(&model, &bank).unwrap();

        // Self-normalization recovers the true mean from the partial sample. Replicate
        // the estimator: the UN-normalized m1 would be ≈ (Σ_sampled wprod)·f0, a small
        // fraction; the self-normalized m1/wsum ≈ f0.
        let (mut m1, mut wsum) = (0.0_f64, 0.0_f64);
        let sampled = enumerate_check_points(&grids, &feats, |x_cells, rep_bins| {
            let wp = joint_weight(&w, &feats, x_cells)?;
            m1 += wp * model.ensemble_f64(rep_bins)?;
            wsum += wp;
            Ok(())
        })
        .unwrap();
        assert!(sampled, "the tiny cap must force sampling");
        assert!(
            wsum < 0.99,
            "a partial sample must under-cover the weight, got {wsum}"
        );
        let self_norm_mean = m1 / wsum;
        assert!(
            (self_norm_mean - bank.f0).abs() < 0.25 * (1.0 + bank.f0.abs()),
            "self-normalized mean {self_norm_mean} should recover f0 {}",
            bank.f0
        );

        // Negatives are still caught under sampling.
        let mut bad_mass = bank.clone();
        bad_mass.f0 += 10.0;
        assert!(matches!(
            check_mass_conservation(&model, &bad_mass, &w),
            Err(PbError::InvariantViolated {
                invariant: Invariant::MassConservation
            })
        ));
        let mut bad_var = bank.clone();
        if let Some(t) = bad_var.tables.first_mut() {
            t.variance += 1000.0;
        }
        assert!(matches!(
            check_variance_sum(&model, &bad_var, &w),
            Err(PbError::InvariantViolated {
                invariant: Invariant::VarianceSum
            })
        ));

        TEST_JOINT_CAP.with(|c| c.set(0));
    }

    fn binary_conditional_mean(
        leaves: &[f64; 8],
        p_low: [f64; 3],
        mask: usize,
        leaf: usize,
    ) -> f64 {
        let mut out = 0.0_f64;
        for (z, &leaf_value) in leaves.iter().enumerate() {
            if (z & mask) != (leaf & mask) {
                continue;
            }
            let mut weight = 1.0_f64;
            for (bit, &p) in p_low.iter().enumerate() {
                if (mask & (1usize << bit)) == 0 {
                    weight *= if (z & (1usize << bit)) != 0 {
                        p
                    } else {
                        1.0 - p
                    };
                }
            }
            out += leaf_value * weight;
        }
        out
    }

    fn binary_fanova_effects(leaves: &[f64; 8], p_low: [f64; 3]) -> [[f64; 8]; 8] {
        let mut effects = [[0.0_f64; 8]; 8];
        for mask in 0usize..8 {
            for leaf in 0usize..8 {
                let mut value = binary_conditional_mean(leaves, p_low, mask, leaf);
                let mut sub = mask;
                while sub > 0 {
                    sub = (sub - 1) & mask;
                    if sub != mask {
                        value -= effects[sub][leaf];
                    }
                }
                effects[mask][leaf] = value;
            }
        }
        effects
    }

    #[test]
    fn fixture_explain_passes_all_gates() {
        let m = fixture_model();
        let x = fixture_serve();
        for w in [RefMeasure::Uniform, RefMeasure::default()] {
            let bank = m.explain(&x, w.clone()).unwrap();
            assert_exact_decomposition(&m, &bank, &x).unwrap();
            check_feature_budget(&m).unwrap();
        }
    }

    #[test]
    fn fixture_uniform_bank_reconstructs_and_centers_intercept() {
        // The merged grid carries the explicit missing cell (cell 0), so the Uniform
        // intercept is the mean over ALL 9 cells of g (incl. the missing-routed leaf),
        // not the 4 finite corners: g sums to 0+2+0 + 2+6+2 + 0+2+0 = 14 ⇒ f0 = 14/9.
        let m = fixture_model();
        let x = fixture_serve();
        let bank = m.explain(&x, RefMeasure::Uniform).unwrap();
        assert!((bank.f0 - 14.0 / 9.0).abs() < 1e-9, "f0 = {}", bank.f0);
        // The bank reconstructs the ensemble at every finite corner (lossless LUT-sum).
        for (cells, want) in [([1, 1], 6.0), ([1, 2], 2.0), ([2, 1], 2.0), ([2, 2], 0.0)] {
            assert!((bank.score(&cells).unwrap() - want).abs() < 1e-6);
        }
        // A genuine pairwise interaction table is realized.
        assert!(bank.tables.iter().any(|t| t.u.order() == 2));
    }

    #[test]
    fn negative_reconstruction_is_caught() {
        let m = fixture_model();
        let x = fixture_serve();
        let mut bank = m.explain(&x, RefMeasure::Uniform).unwrap();
        // Perturb a main-effect finite cell: tables no longer reconstruct the ensemble.
        let main = bank
            .tables
            .iter_mut()
            .find(|t| t.u.order() == 1)
            .expect("a main effect");
        main.values.add(&[1], 1.0).unwrap();
        assert!(matches!(
            check_reconstruction(&m, &bank),
            Err(PbError::InvariantViolated {
                invariant: Invariant::Reconstruction
            })
        ));
    }

    #[test]
    fn reconstruction_enumerates_model_features_even_if_bank_omits_tables() {
        let m = fixture_model();
        let x = fixture_serve();
        let mut bank = m.explain(&x, RefMeasure::Uniform).unwrap();
        // Old bug shape: if the check domain came only from `bank.tables`, clearing
        // tables and setting f0 to the missing/missing score made reconstruction pass
        // vacuously at one point. The gate must inspect the model's realized features.
        bank.f0 = 0.0;
        bank.tables.clear();
        assert!(matches!(
            check_reconstruction(&m, &bank),
            Err(PbError::InvariantViolated {
                invariant: Invariant::Reconstruction
            })
        ));
    }

    #[test]
    fn negative_three_way_is_caught() {
        let m = fixture_model();
        let x = fixture_serve();
        let mut bank = m.explain(&x, RefMeasure::Uniform).unwrap();
        bank.tables[0].values.add(&[2], 0.75).unwrap();
        assert!(matches!(
            check_three_way_equal(&m, &bank),
            Err(PbError::InvariantViolated {
                invariant: Invariant::ThreeWayEqual
            })
        ));
    }

    #[test]
    fn negative_purity_is_caught() {
        let m = fixture_model();
        let x = fixture_serve();
        let w = {
            let grids = MergedGrids::from_model(&m).unwrap();
            build_weights(&x, &grids, &RefMeasure::Uniform).unwrap()
        };
        let mut bank = m.explain(&x, RefMeasure::Uniform).unwrap();
        // Add a constant to a whole b0=1 slice of the pairwise table → its conditional
        // mean is no longer zero (a residual main effect left in the 2-way).
        let pair = bank
            .tables
            .iter_mut()
            .find(|t| t.u.order() == 2)
            .expect("a pairwise table");
        pair.values.add(&[1, 1], 1.0).unwrap();
        pair.values.add(&[1, 2], 1.0).unwrap();
        pair.values.add(&[1, 0], 1.0).unwrap();
        assert!(matches!(
            check_purity(&m, &bank, &w),
            Err(PbError::InvariantViolated {
                invariant: Invariant::Decomposability
            })
        ));
    }

    #[test]
    fn over_budget_model_violates_feature_budget() {
        assert!(matches!(
            check_feature_budget(&fixture_over_budget_model()),
            Err(PbError::InvariantViolated {
                invariant: Invariant::FeatureBudget
            })
        ));
    }

    #[test]
    fn approximate_model_refuses_to_export() {
        let m = fixture_over_budget_model();
        let x = fixture_serve();
        assert!(matches!(
            m.explain(&x, RefMeasure::Uniform),
            Err(PbError::ExactnessFirewall(_))
        ));
    }

    #[test]
    fn joint_measure_is_rejected_in_v1() {
        let m = fixture_model();
        let x = fixture_serve();
        assert!(matches!(
            m.explain(&x, RefMeasure::Joint),
            Err(PbError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn nonfinite_laplace_and_malformed_serve_matrix_are_rejected() {
        let m = fixture_model();
        let x = fixture_serve();
        assert!(matches!(
            m.explain(
                &x,
                RefMeasure::ProductMarginals {
                    laplace: f32::INFINITY
                }
            ),
            Err(PbError::InvalidConfig { .. })
        ));

        let mut bad = x.clone();
        bad.0.data[0][0] = u8::MAX;
        assert!(matches!(
            m.explain(&bad, RefMeasure::Uniform),
            Err(PbError::InvalidInput { .. })
        ));

        let mut bad = x;
        bad.0.grids[0].borders.clear();
        assert!(matches!(
            m.explain(&bad, RefMeasure::Uniform),
            Err(PbError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn tensor_rejects_unrepresentable_shapes_without_panicking() {
        assert!(matches!(
            Tensor::try_zeros(vec![0]),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            Tensor::try_zeros(vec![u32::MAX as usize + 1]),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            Tensor::from_vec(vec![0], Vec::new()),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn sparse_tensor_preserves_logical_dense_values() {
        let mut tensor = Tensor::try_sparse_zeros(vec![2, 3]).unwrap();
        assert!(tensor.is_sparse());
        assert_eq!(tensor.len(), 6);
        assert_eq!(tensor.values().as_ref(), &[0.0; 6]);

        tensor.add(&[1, 2], 1.5).unwrap();
        tensor.add(&[0, 1], -2.0).unwrap();
        tensor.add(&[1, 2], -1.5).unwrap();

        assert_eq!(tensor.at(&[1, 2]), Some(0.0));
        assert_eq!(tensor.at(&[0, 1]), Some(-2.0));
        assert_eq!(tensor.values().as_ref(), &[0.0, -2.0, 0.0, 0.0, 0.0, 0.0]);

        tensor.add_scalar(0.25);
        assert!(!tensor.is_sparse());
        assert_eq!(
            tensor.values().as_ref(),
            &[0.25, -1.75, 0.25, 0.25, 0.25, 0.25]
        );
    }

    #[test]
    fn g3_real_fit_two_feature_additive() {
        // Gate G3: a real fitted model, explained, passes all five checks.
        let n = 64usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 3.0 { 10.0 } else { 20.0 };
                let b = if x1[i] <= 2.0 { 5.0 } else { 0.0 };
                a + b
            })
            .collect();
        let (model, x) = fit(&[x0, x1], &y, exact_cfg(30));
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&model, &bank, &x).unwrap();
        // Sobol importances exist and sum to ~1 (product w).
        let sobol = bank.sobol();
        let s: f64 = sobol.iter().map(|(_, v)| v).sum();
        assert!((s - 1.0).abs() < 1e-6, "sobol sum {s}");
    }

    #[test]
    fn g3_real_fit_three_feature() {
        let n = 64usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 2 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| ((i / 2) % 2 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| ((i / 4) % 2 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 1.0 { 0.0 } else { 4.0 };
                let b = if x1[i] <= 1.0 { 0.0 } else { 2.0 };
                let c = if x2[i] <= 1.0 { 0.0 } else { 1.0 };
                a + b + c - 3.5
            })
            .collect();
        let (model, x) = fit(&[x0, x1, x2], &y, exact_cfg(40));
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&model, &bank, &x).unwrap();
    }

    #[test]
    fn correction_preserves_g0_and_shifts_prediction() {
        // A 2-feature target WITH a {0,1} interaction, so the model realizes the pair.
        let n = 64usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| ((i / 4) % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 2.0 { 10.0 } else { 20.0 };
                let b = if x1[i] <= 2.0 { 5.0 } else { 0.0 };
                let ab = if x0[i] <= 2.0 && x1[i] <= 2.0 { 7.0 } else { 0.0 };
                a + b + ab
            })
            .collect();
        let (mut model, x) = fit(&[x0, x1], &y, exact_cfg(40));

        // Baseline decomposition passes the five gates.
        let base_bank = model.explain(&x, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&model, &base_bank, &x).unwrap();
        let row = [1u8, 1u8];
        let base_score = model.ensemble_f64(&row).unwrap();

        // Attach a raw cell-basis correction: a main on axis 0 AND the {0,1} pair.
        let mut corr = correction_scaffold(&model, &[vec![0], vec![0, 1]]).unwrap();
        for (c, v) in corr.tables[0].values.iter_mut().enumerate() {
            *v = 0.25 + 0.1 * c as f64;
        }
        for (c, v) in corr.tables[1].values.iter_mut().enumerate() {
            *v = -0.2 + 0.05 * c as f64;
        }
        model.correction = Some(corr);
        model.validate().unwrap();

        // Prediction moves by EXACTLY the correction delta at `row`.
        let corr_score = model.ensemble_f64(&row).unwrap();
        let delta = model.correction_delta(&row).unwrap();
        assert!(delta.abs() > 1e-9, "correction must be nonzero");
        assert!((corr_score - base_score - delta).abs() < 1e-12);

        // G0 WITH the correction present: explain re-runs all five exactness gates internally
        // (reconstruction ties ensemble_f64 == bank.score), and they must still pass.
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&model, &bank, &x).unwrap();
        // The correction's marginal flowed into the decomposition (a {0} or {0,1} table exists).
        assert!(!bank.tables.is_empty());
    }

    #[test]
    fn empty_model_explains_to_intercept_only() {
        // A constant target → no trees → the bank is just f0 (== mean), gates trivial.
        let (model, x) = fit(
            &[vec![1.0, 2.0, 3.0, 4.0]],
            &[5.0, 5.0, 5.0, 5.0],
            exact_cfg(10),
        );
        assert!(model.trees.is_empty());
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        assert!(bank.tables.is_empty());
        assert!((bank.f0 - 5.0).abs() < 1e-5);
        assert_exact_decomposition(&model, &bank, &x).unwrap();
    }

    #[test]
    fn shap_sums_to_score_minus_f0() {
        let n = 48usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 3 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| if x0[i] <= 2.0 { -3.0 } else { 4.0 } + x1[i] * 0.5)
            .collect();
        let (model, x) = fit(&[x0, x1], &y, exact_cfg(25));
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        // At every realized cell tuple, Σ φ_i == score − f0.
        let g = MergedGrids::from_model(&model).unwrap();
        for c0 in 0..g.cells(FeatureId(0)).unwrap() {
            for c1 in 0..g.cells(FeatureId(1)).unwrap() {
                let cells = [c0 as u32, c1 as u32];
                let phi: f64 = bank.shap(&cells).unwrap().iter().sum();
                let score = bank.score(&cells).unwrap();
                assert!((phi - (score - bank.f0)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn missing_left_routing_is_honored_end_to_end() {
        // The §08 load-bearing claim: the merged-grid missing cell (cell 0) honors a
        // tree's learned `missing_left` exactly and is NOT collapsed into a finite
        // interval, so tree-sum == table-sum even for rows missing on a split axis. A
        // depth-1 tree with `missing_left: true` routes missing to the LOW leaf; the
        // serve data includes a genuine bin-0 (missing) row.
        use crate::data::{AxisKind, AxisProvenance, BinnedMatrix};
        use crate::engine::{ExactnessMode, ModelSchema, ObliviousTree, Split};
        use crate::loss::{Link, LossId, ObjectiveTag};
        // leaves: idx0 (high, bin2 side) = 3; idx1 (low, bin1 + missing) = 7.
        let tree = ObliviousTree {
            splits: vec![Split {
                axis: 0,
                bin_le: 1,
                missing_left: true,
            }],
            leaves: [3.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            depth: 1,
        };
        let model = Model {
            f0: 0.0,
            trees: vec![(1.0, tree)],
            grids: vec![fixture_grid()],
            provenance: vec![AxisProvenance {
                raw: FeatureId(0),
                kind: AxisKind::Numeric,
            }],
            link: Link::Identity,
            mode: ExactnessMode::Exact,
            schema: ModelSchema {
                feature_names: vec!["x0".into()],
                feature_kinds: vec![AxisKind::Numeric],
                cat_encoders: crate::cat::CatEncoderStore::new(),
                class_labels: None,
                objective: ObjectiveTag {
                    link: Link::Identity,
                    loss: LossId::SquaredError,
                    tweedie_rho: None,
                },
            },
            schema_version: crate::serialize::SCHEMA_VERSION,
            correction: None,
        };
        let x = ServeBinnedMatrix(BinnedMatrix {
            data: vec![vec![0, 1, 2, 1, 2]], // a genuine missing (bin 0) row at index 0
            n_rows: 5,
            grids: vec![fixture_grid()],
            provenance: model.provenance.clone(),
        });
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&model, &bank, &x).unwrap();
        // The missing cell reconstructs the missing-ROUTED leaf (7), exactly as the model
        // scores a missing row — never the bin-2/high leaf (3).
        assert!((bank.score(&[0]).unwrap() - model.ensemble_f64(&[0]).unwrap()).abs() < 1e-6);
        assert!((model.ensemble_f64(&[0]).unwrap() - 7.0).abs() < 1e-6);
        // ...and the missing cell (7) differs from the bin-2 finite cell (3): proof the
        // missing routing is not silently collapsed into the first/last finite interval.
        assert!((bank.score(&[2]).unwrap() - 3.0).abs() < 1e-6);
        assert!((bank.score(&[1]).unwrap() - 7.0).abs() < 1e-6);
    }

    #[test]
    fn wht8_coefficients_match_purified_single_tree_tables() {
        use crate::cat::CatEncoderStore;
        use crate::constraints::wht8_uniform;
        use crate::data::{AxisKind, AxisProvenance};
        use crate::engine::{ExactnessMode, ModelSchema, ObliviousTree, Split};
        use crate::loss::{Link, LossId, ObjectiveTag};

        let leaves = [0.0, 1.0, 3.0, 4.0, 8.0, -2.0, 5.0, 11.0];
        let tree = ObliviousTree {
            splits: vec![
                Split {
                    axis: 0,
                    bin_le: 1,
                    missing_left: false,
                },
                Split {
                    axis: 1,
                    bin_le: 1,
                    missing_left: false,
                },
                Split {
                    axis: 2,
                    bin_le: 1,
                    missing_left: false,
                },
            ],
            leaves,
            depth: 3,
        };
        let provenance: Vec<AxisProvenance> = (0..3u32)
            .map(|raw| AxisProvenance {
                raw: FeatureId(raw),
                kind: AxisKind::Numeric,
            })
            .collect();
        let model = Model {
            f0: 0.0,
            trees: vec![(1.0, tree)],
            grids: vec![fixture_grid(), fixture_grid(), fixture_grid()],
            provenance: provenance.clone(),
            link: Link::Identity,
            mode: ExactnessMode::Exact,
            schema: ModelSchema {
                feature_names: vec!["x0".into(), "x1".into(), "x2".into()],
                feature_kinds: vec![AxisKind::Numeric; 3],
                cat_encoders: CatEncoderStore::new(),
                class_labels: None,
                objective: ObjectiveTag {
                    link: Link::Identity,
                    loss: LossId::SquaredError,
                    tweedie_rho: None,
                },
            },
            schema_version: crate::serialize::SCHEMA_VERSION,
            correction: None,
        };
        let serve = ServeBinnedMatrix(crate::data::BinnedMatrix {
            data: vec![
                vec![2, 1, 2, 1, 2, 1, 2, 1],
                vec![2, 2, 1, 1, 2, 2, 1, 1],
                vec![2, 2, 2, 2, 1, 1, 1, 1],
            ],
            n_rows: 8,
            grids: model.grids.clone(),
            provenance,
        });
        let bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
        let leaves_f64 = leaves.map(f64::from);
        let coeffs = wht8_uniform(leaves_f64).coeffs;
        let half_effects = binary_fanova_effects(&leaves_f64, [0.5; 3]);
        for (mask, values) in half_effects.iter().enumerate() {
            for (leaf, &effect) in values.iter().enumerate() {
                let sign = if ((mask & leaf).count_ones() & 1) == 0 {
                    1.0
                } else {
                    -1.0
                };
                assert!((effect - sign * coeffs[mask]).abs() < 1.0e-10);
            }
        }

        // Under the §08 Uniform measure the missing cell is a third cell. With
        // `missing_left=false`, missing routes with the high side, so the induced
        // binary cut weights are P(low)=1/3 and P(high)=2/3 on every axis.
        let effects = binary_fanova_effects(&leaves_f64, [1.0 / 3.0; 3]);
        assert!((bank.f0 - effects[0][0]).abs() < 1.0e-10);

        for (mask, effect_values) in effects.iter().enumerate().skip(1) {
            let ids: Vec<u32> = (0..3)
                .filter(|bit| (mask & (1usize << bit)) != 0)
                .map(|bit| bit as u32)
                .collect();
            let u = FeatureSet::new(&ids);
            let table = bank
                .tables
                .iter()
                .find(|table| table.u == u)
                .expect("purified table for WHT mask");
            for (leaf, &expected) in effect_values.iter().enumerate() {
                let coord: Vec<usize> = ids
                    .iter()
                    .map(|id| {
                        if (leaf & (1usize << usize::try_from(*id).unwrap())) != 0 {
                            1
                        } else {
                            2
                        }
                    })
                    .collect();
                let got = table.values.at(&coord).unwrap();
                assert!(
                    (got - expected).abs() < 1.0e-9,
                    "mask {mask:03b} leaf {leaf:03b}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn recompute_under_is_exactness_preserving() {
        let n = 48usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| if x0[i] <= 2.0 { 1.0 } else { -2.0 } + if x1[i] <= 2.0 { 0.5 } else { 1.5 })
            .collect();
        let (model, x) = fit(&[x0, x1], &y, exact_cfg(30));
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        // Recompute under Uniform: still reconstructs the ensemble (sum conserved).
        let re = bank.recompute_under(RefMeasure::Uniform).unwrap();
        check_reconstruction(&model, &re).unwrap();
        check_three_way_equal(&model, &re).unwrap();
        // The new bank's stamped measure reflects the recompute.
        assert!(matches!(re.reference_measure(), RefMeasure::Uniform));
    }

    #[test]
    fn table_budget_error_trips_on_tiny_ceiling() {
        // A real triple, with a 1-cell budget → the firewall fires before allocation.
        let n = 32usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 2 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| ((i / 2) % 2 + 1) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (x0[i] + x1[i]) * 0.3).collect();
        let (model, _) = fit(&[x0, x1], &y, exact_cfg(10));
        let grids = MergedGrids::from_model(&model).unwrap();
        let budget = TableBudget {
            max_table_cells: 1,
            max_bank_cells: 1,
            on_overflow: OverflowPolicy::Error,
        };
        assert!(matches!(
            accumulate(&model, &grids, &budget),
            Err(PbError::TableBudget { .. })
        ));
    }

    #[test]
    fn sparse_fallback_is_exact_and_serializable() {
        let model = fixture_model();
        let x = fixture_serve();
        let grids = MergedGrids::from_model(&model).unwrap();
        let budget = TableBudget {
            max_table_cells: 1,
            max_bank_cells: 1,
            on_overflow: OverflowPolicy::SparseFallback {
                density_threshold: 1.0,
            },
        };
        let (raw, _) = accumulate(&model, &grids, &budget).unwrap();
        assert!(raw.tables.values().any(|table| table.values.is_sparse()));
        verify_raw_accumulation(&model, &raw, &[], &grids).unwrap();

        let weights = build_weights(&x, &grids, &RefMeasure::Uniform).unwrap();
        let mut bank = purify(raw, &weights, &grids, PurifyMode::SinglePass).unwrap();
        fill_support(&mut bank, &grids, &x).unwrap();
        assert!(bank.tables.iter().any(|table| table.values.is_sparse()));

        check_reconstruction(&model, &bank).unwrap();
        check_mass_conservation(&model, &bank, &weights).unwrap();
        check_purity(&model, &bank, &weights).unwrap();
        check_variance_sum(&model, &bank, &weights).unwrap();
        check_three_way_equal(&model, &bank).unwrap();

        let encoded = bincode::serde::encode_to_vec(&bank, bincode::config::standard()).unwrap();
        let (decoded, consumed): (TableBank, usize) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, bank);
        check_reconstruction(&model, &decoded).unwrap();
    }

    #[test]
    fn explain_with_budget_activates_sparse_fallback() {
        let model = fixture_model();
        let x = fixture_serve();
        let budget = TableBudget {
            max_table_cells: 1,
            max_bank_cells: 1,
            on_overflow: OverflowPolicy::SparseFallback {
                density_threshold: 1.0,
            },
        };
        let bank = model
            .explain_with_budget(&x, RefMeasure::Uniform, budget)
            .unwrap();
        assert!(bank.tables.iter().any(|table| table.values.is_sparse()));
        assert_exact_decomposition(&model, &bank, &x).unwrap();
    }

    #[test]
    fn sparse_fallback_rejects_invalid_density_threshold() {
        let model = fixture_model();
        let grids = MergedGrids::from_model(&model).unwrap();
        for density_threshold in [f64::NAN, -0.1, 1.1] {
            let budget = TableBudget {
                on_overflow: OverflowPolicy::SparseFallback { density_threshold },
                ..TableBudget::default()
            };
            assert!(matches!(
                accumulate(&model, &grids, &budget),
                Err(PbError::InvalidConfig { .. })
            ));
        }
    }

    #[test]
    fn sparse_fallback_rejects_tables_above_density_threshold() {
        let model = fixture_model();
        let grids = MergedGrids::from_model(&model).unwrap();
        let budget = TableBudget {
            max_table_cells: 1,
            max_bank_cells: 1,
            on_overflow: OverflowPolicy::SparseFallback {
                density_threshold: 0.0,
            },
        };
        assert!(matches!(
            accumulate(&model, &grids, &budget),
            Err(PbError::TableBudget { .. })
        ));
    }

    // --- Purification identity proptests (spec §08.9) -----------------------

    /// A small random raw bank: one pairwise table over a 2×3 merged grid (cells incl.
    /// the missing cell), plus the uniform weight cache, for the purify identities.
    fn small_raw(values: &[f64]) -> (RawBank, WeightCache, MergedGrids) {
        // Two raw features with cells {2, 3}: feature 0 has no realized border (missing
        // + one finite cell); feature 1 has one realized border (missing + two finite).
        let per_raw = vec![
            MergedAxis {
                axis: 0,
                borders: vec![],
                model_border_index: vec![],
                model_n_bins: 2,
            },
            MergedAxis {
                axis: 1,
                borders: vec![1.5],
                model_border_index: vec![0],
                model_n_bins: 3,
            },
        ];
        let grids = MergedGrids { per_raw };
        let axes = vec![
            grids.axis_id(FeatureId(0)).unwrap(),
            grids.axis_id(FeatureId(1)).unwrap(),
        ];
        let mut tens = Tensor::try_zeros(vec![2, 3]).unwrap();
        for (i, &v) in values.iter().enumerate() {
            let c = [i / 3, i % 3];
            tens.set(&c, v).unwrap();
        }
        let mut tables = BTreeMap::new();
        let u = FeatureSet::new(&[0, 1]);
        tables.insert(
            u.clone(),
            RawTable {
                u,
                axes,
                values: tens,
            },
        );
        let raw = RawBank { f0: 0.0, tables };
        let w = WeightCache {
            per_axis: vec![vec![0.5, 0.5], vec![1.0 / 3.0; 3]],
            kind: RefMeasure::Uniform,
        };
        (raw, w, grids)
    }

    fn purified_values(values: &[f64]) -> TableBank {
        let (raw, w, grids) = small_raw(values);
        purify(raw, &w, &grids, PurifyMode::SinglePass).unwrap()
    }

    fn bank_full(bank: &TableBank) -> (f64, BTreeMap<FeatureSet, Vec<f64>>) {
        let mut m = BTreeMap::new();
        for t in &bank.tables {
            let mut v = Vec::new();
            let ext = t.values.shape();
            walk_extents(&ext, |c| {
                v.push(t.values.at(c).unwrap());
                Ok(())
            })
            .unwrap();
            m.insert(t.u.clone(), v);
        }
        (bank.f0, m)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Purity: every purified bank passes the slice-mean-zero check under its w.
        #[test]
        fn prop_purify_is_pure(values in prop::collection::vec(-5.0f64..5.0, 6)) {
            let bank = purified_values(&values);
            let m = fixture_model();
            let (_, w, _) = small_raw(&values);
            prop_assert!(check_purity(&m, &bank, &w).is_ok());
        }

        /// Idempotence: purify(purify(T)) == purify(T) (already-pure stays put).
        #[test]
        fn prop_purify_idempotent(values in prop::collection::vec(-5.0f64..5.0, 6)) {
            let once = purified_values(&values);
            let twice = once.recompute_under(RefMeasure::Uniform).unwrap();
            let (f0a, a) = bank_full(&once);
            let (f0b, b) = bank_full(&twice);
            prop_assert!((f0a - f0b).abs() < 1e-9);
            for (u, va) in &a {
                let vb = b.get(u).unwrap();
                for (x, y) in va.iter().zip(vb) {
                    prop_assert!((x - y).abs() < 1e-9, "u={:?} {} vs {}", u, x, y);
                }
            }
        }

        /// Linearity: purify(αA) == α·purify(A) (cellwise, including the intercept).
        #[test]
        fn prop_purify_linear(
            values in prop::collection::vec(-5.0f64..5.0, 6),
            alpha in -3.0f64..3.0,
        ) {
            let base = purified_values(&values);
            let scaled_in: Vec<f64> = values.iter().map(|v| alpha * v).collect();
            let scaled = purified_values(&scaled_in);
            let (f0a, a) = bank_full(&base);
            let (f0b, b) = bank_full(&scaled);
            prop_assert!((alpha * f0a - f0b).abs() < 1e-7);
            for (u, va) in &a {
                let vb = b.get(u).unwrap();
                for (x, y) in va.iter().zip(vb) {
                    prop_assert!((alpha * x - y).abs() < 1e-7);
                }
            }
        }
    }
}
