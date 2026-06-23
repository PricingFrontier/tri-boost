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
//! variance-sum), and the `Error` table-budget policy. `Joint` and `SparseFallback` are
//! rejected up front rather than silently mishandled.

use crate::data::{BorderGrid, FeatureId, ServeBinnedMatrix};
use crate::engine::{low_bit, Model};
use crate::error::{Invariant, PbError};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};

/// Hard ceiling on exhaustive joint-grid enumeration for the reconstruction /
/// variance / three-way checks. Below it the sweep is exhaustive (one interior point
/// per joint cell, spec §08.6); above it the checks sample a deterministic subset
/// (the release-mode behavior of §08.8). v1 green-spine models have a handful of
/// realized features, so the gate path is exhaustive in practice.
const JOINT_CAP: usize = 1 << 20;

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
    data: Vec<f64>,
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
            data: filled_data(cells, 0.0)?,
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
            data: filled_data(cells, value)?,
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
        Ok(Self { shape, data })
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
        self.data.len()
    }

    /// `true` if the tensor has no cells.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The dense row-major backing values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.data
    }

    /// Add a constant to every cell. Used by rating-view re-basing, which moves an
    /// equal and opposite constant into the exported intercept and therefore preserves
    /// every reconstructed score exactly.
    pub fn add_scalar(&mut self, delta: f64) {
        for v in &mut self.data {
            *v += delta;
        }
    }

    fn offset(&self, coord: &[usize]) -> Option<usize> {
        if coord.len() != self.shape.len() {
            return None;
        }
        let mut off = 0usize;
        for (c, dim) in coord.iter().zip(self.shape.iter()) {
            let dim = *dim as usize;
            if *c >= dim {
                return None;
            }
            off = off.checked_mul(dim)?.checked_add(*c)?;
        }
        Some(off)
    }

    /// Read the value at `coord`, or `None` if out of range / wrong rank.
    #[must_use]
    pub fn at(&self, coord: &[usize]) -> Option<f64> {
        self.offset(coord).and_then(|o| self.data.get(o).copied())
    }

    /// Write `value` at `coord`.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `coord` is out of range or the wrong rank.
    pub fn set(&mut self, coord: &[usize], value: f64) -> Result<(), PbError> {
        let off = self.offset(coord).ok_or_else(|| PbError::ShapeMismatch {
            what: "tensor set coord out of range".into(),
        })?;
        let slot = self.data.get_mut(off).ok_or_else(|| PbError::Internal {
            what: "tensor offset escaped buffer".into(),
        })?;
        *slot = value;
        Ok(())
    }

    /// Add `delta` to the value at `coord`.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `coord` is out of range or the wrong rank.
    pub fn add(&mut self, coord: &[usize], delta: f64) -> Result<(), PbError> {
        let off = self.offset(coord).ok_or_else(|| PbError::ShapeMismatch {
            what: "tensor add coord out of range".into(),
        })?;
        let slot = self.data.get_mut(off).ok_or_else(|| PbError::Internal {
            what: "tensor offset escaped buffer".into(),
        })?;
        *slot += delta;
        Ok(())
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
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
/// over-budget triple fails *before* it materializes 100s of MB.
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
            on_overflow: OverflowPolicy::Error,
        }
    }
}

/// The resolution when a table would exceed [`TableBudget::max_table_cells`] (§08.10).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverflowPolicy {
    /// Hard error — refuse to build a bank that would exceed the budget
    /// ([`PbError::TableBudget`]). No silent truncation.
    Error,
    /// EXACT sparse-tensor storage for hot triples — a v1.5 optimization, rejected up
    /// front in v1 (so a caller never silently receives `Error` behavior under it).
    SparseFallback {
        /// Occupancy below which a triple would be stored sparsely.
        density_threshold: f64,
    },
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
    /// Build the merged grids from a fitted `model`. v1: requires numeric axes with a
    /// 1-to-1 raw↔axis mapping whose raw ids fill `0..n_features` (the green-spine
    /// invariant); categorical or many-to-one provenance is rejected (`InvalidConfig`).
    pub(crate) fn from_model(model: &Model) -> Result<Self, PbError> {
        use crate::data::AxisKind;
        let n_features = model.provenance.len();
        // axis_of_raw[raw] = the single model axis carrying that raw feature.
        let mut axis_of_raw: Vec<Option<usize>> = vec![None; n_features];
        for (a, prov) in model.provenance.iter().enumerate() {
            if !matches!(prov.kind, AxisKind::Numeric) {
                return Err(PbError::InvalidConfig {
                    what: "v1 explain supports numeric axes only (categoricals arrive with §04)"
                        .into(),
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
) -> Result<RawBank, PbError> {
    if let OverflowPolicy::SparseFallback { .. } = budget.on_overflow {
        return Err(PbError::InvalidConfig {
            what: "SparseFallback is a v1.5 optimization; v1 uses OverflowPolicy::Error".into(),
        });
    }
    let mut tables: BTreeMap<FeatureSet, RawTable> = BTreeMap::new();
    let mut bank_cells: u64 = 0;

    for (alpha, tree) in &model.trees {
        let u = tree_support(model, tree)?;
        let u_ids: Vec<FeatureId> = u.0.iter().copied().collect();
        if !tables.contains_key(&u) {
            let mut extents = Vec::with_capacity(u_ids.len());
            for r in &u_ids {
                extents.push(grids.cells(*r)?);
            }
            let table_cells = product_u64(&extents)?;
            if table_cells > budget.max_table_cells {
                return Err(PbError::TableBudget {
                    what: format!("table {u:?}"),
                    cells: table_cells,
                    budget: budget.max_table_cells,
                });
            }
            bank_cells = bank_cells
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
            let mut axes = Vec::with_capacity(u_ids.len());
            for r in &u_ids {
                axes.push(grids.axis_id(*r)?);
            }
            tables.insert(
                u.clone(),
                RawTable {
                    u: u.clone(),
                    axes,
                    values: Tensor::try_zeros(extents)?,
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

    Ok(RawBank {
        f0: f64::from(model.f0),
        tables,
    })
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
/// by raw feature) and `rep_bins` (indexed by model axis) for the same point.
fn enumerate_check_points(
    grids: &MergedGrids,
    feats: &[FeatureId],
    mut visit: impl FnMut(&[u32], &[u8]) -> Result<(), PbError>,
) -> Result<(), PbError> {
    let n_features = grids.n_features();
    let mut extents = Vec::with_capacity(feats.len());
    for r in feats {
        extents.push(grids.cells(*r)?);
    }
    let total = product_u64(&extents)?;

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

    if total <= JOINT_CAP as u64 {
        walk_extents(&extents, &mut emit)?;
    } else {
        // Deterministic strided sample: mix the sample index per axis with a splitmix
        // step so the points spread across the space without RNG state.
        let mut tuple = vec![0usize; feats.len()];
        for s in 0..JOINT_CAP as u64 {
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
    }
    Ok(())
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
    })
}

/// **MassConservation (I2.2):** all `w`-mass that survives purification sits in the
/// intercept — `Σ_x w(x)·F_ens(x) == f0` (every table integrates to zero).
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
    let feats = gate_features(model, bank)?;
    let mut mass = 0.0_f64;
    enumerate_check_points(&grids, &feats, |x_cells, rep_bins| {
        let wprod = joint_weight(w, &feats, x_cells)?;
        mass += wprod * model.ensemble_f64(rep_bins)?;
        Ok(())
    })?;
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
    let (mut m1, mut m2) = (0.0_f64, 0.0_f64);
    enumerate_check_points(&grids, &feats, |x_cells, rep_bins| {
        let wprod = joint_weight(w, &feats, x_cells)?;
        let e = model.ensemble_f64(rep_bins)?;
        m1 += wprod * e;
        m2 += wprod * e * e;
        Ok(())
    })?;
    let var_ens = m2 - m1 * m1;
    let var_tables: f64 = bank.tables.iter().map(|t| t.variance).sum();
    if (var_ens - var_tables).abs() > tol {
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
    })
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
        let budget = TableBudget::default();
        let raw = accumulate(self, &grids, &budget)?;
        verify_raw_accumulation(self, &raw, &grids)?;
        let weights = build_weights(x, &grids, &w)?;
        let mut bank = purify(raw, &weights, &grids, PurifyMode::SinglePass)?;
        fill_support(&mut bank, &grids, x)?;

        check_reconstruction(self, &bank)?;
        check_mass_conservation(self, &bank, &weights)?;
        check_purity(self, &bank, &weights)?;
        check_variance_sum(self, &bank, &weights)?;
        check_three_way_equal(self, &bank)?;
        Ok(bank)
    }
}

/// The pre-purify exact-accumulation checkpoint (spec §08.2): `f0 + Σ_u T_raw[u](x) ==
/// F_ens(x)` identically at every realized-support cell, before any purification runs.
/// A failure is an accumulation bug, not an invariant violation, so it surfaces as
/// [`PbError::Internal`].
fn verify_raw_accumulation(
    model: &Model,
    raw: &RawBank,
    grids: &MergedGrids,
) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model).recon_tol;
    // Reuse the joint enumerator over the raw bank's realized features.
    let mut set: BTreeSet<FeatureId> = model_features(model)?.into_iter().collect();
    for u in raw.tables.keys() {
        for r in &u.0 {
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
        if (ens - acc).abs() > tol {
            return Err(PbError::Internal {
                what: "raw accumulation does not reconstruct the ensemble".into(),
            });
        }
        Ok(())
    })
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
        Ok(0.0)
    }

    /// Sobol importances `S_u = σ²(f_u)/σ²(F)` from the cached table variances (spec
    /// §08.5), sorted descending. Under product/uniform `w` they sum to ~1.
    #[must_use]
    pub fn sobol(&self) -> Vec<(FeatureSet, f64)> {
        let total: f64 = self.tables.iter().map(|t| t.variance).sum();
        let mut out: Vec<(FeatureSet, f64)> = self
            .tables
            .iter()
            .map(|t| {
                let s = if total > 0.0 { t.variance / total } else { 0.0 };
                (t.u.clone(), s)
            })
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
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
            boosters: Default::default(),
        }
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
    fn sparse_fallback_is_rejected_in_v1() {
        let model = fixture_model();
        let grids = MergedGrids::from_model(&model).unwrap();
        let budget = TableBudget {
            on_overflow: OverflowPolicy::SparseFallback {
                density_threshold: 0.05,
            },
            ..TableBudget::default()
        };
        assert!(matches!(
            accumulate(&model, &grids, &budget),
            Err(PbError::InvalidConfig { .. })
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
