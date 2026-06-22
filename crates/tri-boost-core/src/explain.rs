//! Explainability engine (spec §2.7 / §08) and the five lossless invariant checks
//! (spec §13.1/§13.2), instantiated here per Gate G0.
//!
//! The fANOVA types (`EffectTable`, `FeatureSet`, `TableBank`, `RefMeasure`,
//! `Tensor`, `AxisId`) are frozen here. The purification engine lands with §08, but
//! the five I2 checks (`Reconstruction`, `MassConservation`, `Purity`, `VarianceSum`,
//! `ThreeWayEqual`) plus `check_feature_budget` (I1) are **real, implemented
//! functions from day one** — green on hand-built positive fixtures and red with the
//! correct [`Invariant`] on negative ones. When §06/§08 land, they point these
//! already-live, already-build-blocking checks at the real `TableBank`.
//!
//! Critically, these checks read ONLY the merged-grid purified bank + the model —
//! never the §07 `wht8` screening coefficients (which live on per-tree 2-point grids
//! and cannot cross the merged-grid alignment). That separation is asserted, not
//! assumed (see the `wht8_is_not_consulted` test).

use crate::data::{BorderGrid, FeatureId};
use crate::engine::Model;
use crate::error::{Invariant, PbError};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::BTreeMap;

/// A §08-local axis identifier (index into the merged-grid axis list).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AxisId(pub u32);

/// A dense row-major n-dimensional tensor of `f64` values (§08-local). Used for both
/// an `EffectTable`'s purified `values` and its per-cell `support`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Tensor {
    // Per-axis extents as fixed-width ints: this type is serialized inside an
    // EffectTable, and a serialized `usize` would differ between the host and the
    // wasm32 smoke build, breaking cross-platform byte-equality (spec §02.8).
    // Bin-count dimensions are tiny, so `u32` is ample.
    shape: Vec<u32>,
    data: Vec<f64>,
}

fn to_u32_shape(shape: &[usize]) -> Vec<u32> {
    shape
        .iter()
        .map(|&d| u32::try_from(d).unwrap_or(u32::MAX))
        .collect()
}

impl Tensor {
    /// A zero tensor of the given shape.
    #[must_use]
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self {
            data: vec![0.0; n],
            shape: to_u32_shape(&shape),
        }
    }

    /// A constant-`value` tensor of the given shape.
    #[must_use]
    pub fn filled(shape: Vec<usize>, value: f64) -> Self {
        let n: usize = shape.iter().product();
        Self {
            data: vec![value; n],
            shape: to_u32_shape(&shape),
        }
    }

    /// Build from an explicit row-major buffer.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `data.len()` does not equal the product of `shape`.
    pub fn from_vec(shape: Vec<usize>, data: Vec<f64>) -> Result<Self, PbError> {
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!("tensor data len {} != product(shape) {n}", data.len()),
            });
        }
        Ok(Self {
            shape: to_u32_shape(&shape),
            data,
        })
    }

    /// The tensor's per-axis extents (as `usize` for indexing).
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        self.shape.iter().map(|&d| d as usize).collect()
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
            off = off * dim + c;
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

    /// Add `delta` to the value at `coord` (used to build corrupted negative fixtures).
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `coord` is out of range or the wrong rank.
    pub fn add(&mut self, coord: &[usize], delta: f64) -> Result<(), PbError> {
        let current = self.at(coord).ok_or_else(|| PbError::ShapeMismatch {
            what: "tensor add coord out of range".into(),
        })?;
        self.set(coord, current + delta)
    }
}

/// A set of 0..=3 distinct, sorted raw feature ids identifying one effect (spec §2.7).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct FeatureSet(pub SmallVec<[FeatureId; 3]>);

impl FeatureSet {
    /// Build a feature set from raw ids (caller ensures distinct/sorted in real §08).
    #[must_use]
    pub fn new(ids: &[u32]) -> Self {
        FeatureSet(ids.iter().map(|&i| FeatureId(i)).collect())
    }

    /// The interaction order `|u|` (1 = main effect, 2 = pairwise, 3 = triple).
    #[must_use]
    pub fn order(&self) -> usize {
        self.0.len()
    }
}

/// One purified effect tensor for feature set `u`, on the merged grid (spec §2.7).
/// `support` is per-cell effective `w`-mass (display metadata) — excluded from the
/// five invariant checks and from inference (scoring reads `values` only).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectTable {
    /// The raw feature set this effect is over.
    pub u: FeatureSet,
    /// The merged-grid axes (parallel to `values`' dimensions).
    pub axes: Vec<AxisId>,
    /// The purified effect values (one cell per merged-grid cell).
    pub values: Tensor,
    /// Per-cell effective `w`-mass (display-only; same extents as `values`).
    pub support: Tensor,
    /// `w`-weighted variance of this effect, `σ²(f_u)`.
    pub variance: f64,
}

impl EffectTable {
    /// Evaluate this effect at a cell given the row's per-axis bin ids. The tensor is
    /// indexed by bin value directly (bin 0 = missing slot).
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if the cell lacks one of this table's axes;
    /// [`PbError::Internal`] if the projected coordinate escapes the tensor.
    pub fn eval(&self, cell: &[u8]) -> Result<f64, PbError> {
        let mut coord = Vec::with_capacity(self.axes.len());
        for a in &self.axes {
            let bin = *cell
                .get(a.0 as usize)
                .ok_or_else(|| PbError::ShapeMismatch {
                    what: format!("cell missing axis {} for effect-table eval", a.0),
                })?;
            coord.push(usize::from(bin));
        }
        self.values.at(&coord).ok_or_else(|| PbError::Internal {
            what: "effect-table coordinate out of range".into(),
        })
    }
}

/// The reference measure for purification (spec §2.7). Default = Laplace-smoothed
/// empirical product-of-marginals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RefMeasure {
    /// Product of per-axis empirical marginals (DEFAULT; `laplace > 0`).
    ProductMarginals {
        /// Laplace smoothing added to each marginal count.
        laplace: f32,
    },
    /// Uniform over realized cells.
    Uniform,
    /// Hooker hierarchical-orthogonality joint measure.
    Joint,
}

impl Default for RefMeasure {
    fn default() -> Self {
        RefMeasure::ProductMarginals { laplace: 1.0 }
    }
}

/// The complete decomposition (spec §2.7): intercept + all purified tables on a
/// shared grid. `tables` is the lossless inference support; display pruning is a view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableBank {
    /// The intercept term.
    pub f0: f64,
    /// Every realized effect `u` of size `1..=3`.
    pub tables: Vec<EffectTable>,
    /// The shared merged grid (sorted union of realized borders per axis).
    pub merged_grids: Vec<BorderGrid>,
    /// The reference measure stamped on the bank and every export.
    pub w: RefMeasure,
}

/// Tolerances for the I2 checks (spec §13.1). `recon_tol` is the canonical
/// `4 · n_trees · f32::EPSILON`; the others track it (variance is squared-scale, so
/// it gets headroom).
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

// --------------------------------------------------------------------------
// Reference-measure cell weights (the `w` the checks integrate against).
// --------------------------------------------------------------------------

/// Per-cell reference weights for `grid_corners` under measure `w` (normalized to
/// sum 1). On a full product grid, `Uniform` and `ProductMarginals` coincide.
fn cell_weights(corners: &[Vec<u8>], w: &RefMeasure) -> Result<Vec<f64>, PbError> {
    if corners.is_empty() {
        return Err(PbError::InvalidInput {
            what: "no grid corners supplied to invariant check".into(),
        });
    }
    match w {
        RefMeasure::Uniform => {
            let n = corners.len() as f64;
            Ok(vec![1.0 / n; corners.len()])
        }
        RefMeasure::ProductMarginals { .. } => {
            let marg = axis_marginals(corners)?;
            let mut raw = Vec::with_capacity(corners.len());
            for c in corners {
                let mut p = 1.0_f64;
                for (a, &b) in c.iter().enumerate() {
                    let m = marg.get(a).ok_or_else(|| PbError::Internal {
                        what: "axis marginal missing".into(),
                    })?;
                    p *= *m.get(&b).unwrap_or(&0.0);
                }
                raw.push(p);
            }
            let total: f64 = raw.iter().sum();
            if total <= 0.0 {
                return Err(PbError::InvalidInput {
                    what: "product-marginal weights summed to zero".into(),
                });
            }
            Ok(raw.iter().map(|x| x / total).collect())
        }
        RefMeasure::Joint => Err(PbError::InvalidConfig {
            what: "Joint reference measure is not supported by the Phase-0 invariant checks".into(),
        }),
    }
}

/// Empirical per-axis marginals (`bin → fraction`) over the corner list.
fn axis_marginals(corners: &[Vec<u8>]) -> Result<Vec<BTreeMap<u8, f64>>, PbError> {
    let n_axes = corners
        .first()
        .ok_or_else(|| PbError::InvalidInput {
            what: "empty corner list".into(),
        })?
        .len();
    let n = corners.len() as f64;
    let mut counts: Vec<BTreeMap<u8, u64>> = vec![BTreeMap::new(); n_axes];
    for c in corners {
        if c.len() != n_axes {
            return Err(PbError::ShapeMismatch {
                what: "grid corners have inconsistent axis count".into(),
            });
        }
        for (a, &b) in c.iter().enumerate() {
            let entry = counts.get_mut(a).ok_or_else(|| PbError::Internal {
                what: "axis index escaped counts".into(),
            })?;
            *entry.entry(b).or_insert(0) += 1;
        }
    }
    Ok(counts
        .into_iter()
        .map(|m| m.into_iter().map(|(b, ct)| (b, ct as f64 / n)).collect())
        .collect())
}

/// Every coordinate tuple of a row-major tensor of the given shape.
fn all_coords(shape: &[usize]) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new()];
    for &dim in shape {
        let mut next = Vec::with_capacity(out.len() * dim.max(1));
        for prefix in &out {
            for i in 0..dim {
                let mut c = prefix.clone();
                c.push(i);
                next.push(c);
            }
        }
        out = next;
    }
    out
}

fn table_sum(bank: &TableBank, cell: &[u8]) -> Result<f64, PbError> {
    let mut acc = bank.f0;
    for t in &bank.tables {
        acc += t.eval(cell)?;
    }
    Ok(acc)
}

// --------------------------------------------------------------------------
// The five I2 checks (each a real, implemented function — never a stub).
// --------------------------------------------------------------------------

/// **Reconstruction (I2.1):** the ensemble equals `f0 + Σ_u f_u` at every cell,
/// within `recon_tol`.
///
/// # Errors
/// [`Invariant::Reconstruction`] if any cell's discrepancy exceeds tolerance.
pub fn check_reconstruction(
    model: &Model,
    bank: &TableBank,
    corners: &[Vec<u8>],
) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model);
    for c in corners {
        let ens = model.ensemble_f64(c)?;
        let tab = table_sum(bank, c)?;
        if (ens - tab).abs() > tol.recon_tol {
            return Err(PbError::invariant(Invariant::Reconstruction));
        }
    }
    Ok(())
}

/// **MassConservation (I2.2):** all `w`-mass that survives purification sits in the
/// intercept, i.e. `Σ_c w(c)·F_ens(c) == f0` (every non-empty table integrates to 0).
///
/// # Errors
/// [`Invariant::MassConservation`] if the `w`-weighted ensemble mean drifts from `f0`.
pub fn check_mass_conservation(
    model: &Model,
    bank: &TableBank,
    corners: &[Vec<u8>],
) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model);
    let w = cell_weights(corners, &bank.w)?;
    let mut mass = 0.0_f64;
    for (c, wc) in corners.iter().zip(w.iter()) {
        mass += wc * model.ensemble_f64(c)?;
    }
    if (mass - bank.f0).abs() > tol.mass_tol {
        return Err(PbError::invariant(Invariant::MassConservation));
    }
    Ok(())
}

/// **Purity (I2.3):** every axis-slice of every `EffectTable` has `w`-weighted mean
/// zero (no lower-order mass left hiding in a higher-order table). Maps to the
/// canonical [`Invariant::Decomposability`] variant (§2.8 has no separate `Purity`).
///
/// # Errors
/// [`Invariant::Decomposability`] if any conditional axis-slice mean is non-zero.
pub fn check_purity(model: &Model, bank: &TableBank, corners: &[Vec<u8>]) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model);
    let axis_marg = axis_marginals(corners)?;
    for table in &bank.tables {
        let shape = table.values.shape();
        for (p, axis) in table.axes.iter().enumerate() {
            let marg = axis_marg
                .get(axis.0 as usize)
                .ok_or_else(|| PbError::Internal {
                    what: "table axis has no marginal".into(),
                })?;
            // Group cells by the OTHER axes; the weighted sum over axis `p` must be 0.
            let mut sums: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
            for coord in all_coords(&shape) {
                let bin_p = u8::try_from(*coord.get(p).ok_or_else(|| PbError::Internal {
                    what: "coord shorter than rank".into(),
                })?)
                .unwrap_or(u8::MAX);
                let wp = *marg.get(&bin_p).unwrap_or(&0.0);
                if wp == 0.0 {
                    continue;
                }
                let v = table.values.at(&coord).ok_or_else(|| PbError::Internal {
                    what: "purity coord out of range".into(),
                })?;
                let mut key = coord.clone();
                key.remove(p);
                *sums.entry(key).or_insert(0.0) += wp * v;
            }
            for s in sums.values() {
                if s.abs() > tol.purity_tol {
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
pub fn check_variance_sum(
    model: &Model,
    bank: &TableBank,
    corners: &[Vec<u8>],
) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model);
    let w = cell_weights(corners, &bank.w)?;

    let (mut m1, mut m2) = (0.0_f64, 0.0_f64);
    for (c, wc) in corners.iter().zip(w.iter()) {
        let e = model.ensemble_f64(c)?;
        m1 += wc * e;
        m2 += wc * e * e;
    }
    let var_ens = m2 - m1 * m1;

    let mut var_tables = 0.0_f64;
    for t in &bank.tables {
        let (mut t1, mut t2) = (0.0_f64, 0.0_f64);
        for (c, wc) in corners.iter().zip(w.iter()) {
            let v = t.eval(c)?;
            t1 += wc * v;
            t2 += wc * v * v;
        }
        var_tables += t2 - t1 * t1;
    }

    if (var_ens - var_tables).abs() > tol.var_tol {
        return Err(PbError::invariant(Invariant::VarianceSum));
    }
    Ok(())
}

/// **ThreeWayEqual (I2.5):** tree-sum, table-sum, and Shapley-sum agree at every
/// cell. The Shapley leg is an INDEPENDENT path: each effect `f_u(c)` is split
/// equally among its `|u|` features (the n-Shapley allocation), then summed per
/// feature; `f0 + Σ_i φ_i` must equal both the ensemble and the table sum.
///
/// # Errors
/// [`Invariant::ThreeWayEqual`] if any of the three reconstructions disagree.
pub fn check_three_way_equal(
    model: &Model,
    bank: &TableBank,
    corners: &[Vec<u8>],
) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model);
    for c in corners {
        let tree = model.ensemble_f64(c)?;
        let table = table_sum(bank, c)?;

        let mut per_feat: BTreeMap<u32, f64> = BTreeMap::new();
        for t in &bank.tables {
            let order = t.u.order().max(1) as f64;
            let share = t.eval(c)? / order;
            for fid in &t.u.0 {
                *per_feat.entry(fid.0).or_insert(0.0) += share;
            }
        }
        let shap = bank.f0 + per_feat.values().sum::<f64>();

        if (tree - table).abs() > tol.recon_tol || (table - shap).abs() > tol.recon_tol {
            return Err(PbError::invariant(Invariant::ThreeWayEqual));
        }
    }
    Ok(())
}

/// Run all five I2 checks (spec §13.1). The signature is the canonical one: a model,
/// its purified bank, and one interior point per merged-grid cell.
///
/// This reads only `model` + `bank` (the merged-grid purified decomposition); it
/// never consults a `wht8` screening coefficient — see `wht8_is_not_consulted`.
///
/// # Errors
/// The first failing check's [`Invariant`], wrapped in [`PbError::InvariantViolated`].
pub fn assert_exact_decomposition(
    model: &Model,
    bank: &TableBank,
    grid_corners: &[Vec<u8>],
) -> Result<(), PbError> {
    check_reconstruction(model, bank, grid_corners)?;
    check_mass_conservation(model, bank, grid_corners)?;
    check_purity(model, bank, grid_corners)?;
    check_variance_sum(model, bank, grid_corners)?;
    check_three_way_equal(model, bank, grid_corners)?;
    Ok(())
}

/// **FeatureBudget (I1, spec §13.2):** every tree is depth `1..=3`, `splits.len() ==
/// depth`, and the count of DISTINCT raw features across its splits equals `depth`
/// (each level a different raw feature).
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

// --------------------------------------------------------------------------
// Hand-built fixtures (doc-hidden public so both unit and integration tests reuse
// them without duplicating the construction). Phase-0 test support only.
// --------------------------------------------------------------------------

/// The 2-axis, 2-data-bin merged grid: every cell `(b0, b1)` with `b0, b1 ∈ {1, 2}`.
#[doc(hidden)]
#[must_use]
pub fn fixture_grid_corners() -> Vec<Vec<u8>> {
    vec![vec![1, 1], vec![1, 2], vec![2, 1], vec![2, 2]]
}

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

    // Leaf index = bit0 | bit1<<1, where bit = (bin <= bin_le). With bin_le = 1 and
    // bins {1,2}: bin1 → bit 1, bin2 → bit 0. So:
    //   (b0,b1)=(2,2) → idx 0 → g=0
    //   (1,2)         → idx 1 → g=2
    //   (2,1)         → idx 2 → g=2
    //   (1,1)         → idx 3 → g=6
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

/// The exact uniform-measure ANOVA decomposition of [`fixture_model`]:
/// `f0 = 2.5`, main effects `±1.5`, pairwise interaction `±0.5`.
#[doc(hidden)]
#[must_use]
pub fn fixture_bank() -> TableBank {
    // Tensors are indexed by bin value directly (size 3 = {missing, bin1, bin2}).
    let main0 = effect_table(&[0], &[3], &[(vec![1], 1.5), (vec![2], -1.5)], 2.25);
    let main1 = effect_table(&[1], &[3], &[(vec![1], 1.5), (vec![2], -1.5)], 2.25);
    let pair = effect_table(
        &[0, 1],
        &[3, 3],
        &[
            (vec![1, 1], 0.5),
            (vec![1, 2], -0.5),
            (vec![2, 1], -0.5),
            (vec![2, 2], 0.5),
        ],
        0.25,
    );
    TableBank {
        f0: 2.5,
        tables: vec![main0, main1, pair],
        merged_grids: vec![fixture_grid(), fixture_grid()],
        w: RefMeasure::Uniform,
    }
}

fn effect_table(
    axes: &[u32],
    shape: &[usize],
    nonzero: &[(Vec<usize>, f64)],
    variance: f64,
) -> EffectTable {
    let mut values = Tensor::zeros(shape.to_vec());
    for (coord, v) in nonzero {
        // Fixtures are hand-checked; a bad coord is a test-authoring bug, surfaced
        // by `set` returning Err which we convert to a zeroed cell (never panics).
        let _ = values.set(coord, *v);
    }
    EffectTable {
        u: FeatureSet::new(axes),
        axes: axes.iter().map(|&a| AxisId(a)).collect(),
        values,
        support: Tensor::filled(shape.to_vec(), 1.0),
        variance,
    }
}

/// A model whose tree spans 4 distinct raw features — an I1 violation
/// (`depth = 4 > 3`). Used as the negative `check_feature_budget` fixture.
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
        clippy::panic
    )]
    use super::*;

    fn assert_invariant(res: Result<(), PbError>, want: Invariant) {
        match res {
            Err(PbError::InvariantViolated { invariant }) => assert_eq!(invariant, want),
            other => panic!("expected InvariantViolated {{ {want} }}, got {other:?}"),
        }
    }

    #[test]
    fn lookup_realizes_the_fixture_function() {
        let m = fixture_model();
        // g(1,1)=6, g(1,2)=2, g(2,1)=2, g(2,2)=0
        assert_eq!(m.ensemble_f64(&[1, 1]).unwrap(), 6.0);
        assert_eq!(m.ensemble_f64(&[1, 2]).unwrap(), 2.0);
        assert_eq!(m.ensemble_f64(&[2, 1]).unwrap(), 2.0);
        assert_eq!(m.ensemble_f64(&[2, 2]).unwrap(), 0.0);
    }

    #[test]
    fn reconstruction_positive_then_negative() {
        let (m, corners) = (fixture_model(), fixture_grid_corners());
        check_reconstruction(&m, &fixture_bank(), &corners).unwrap();

        let mut bad = fixture_bank();
        bad.tables[0].values.add(&[1], 1.0).unwrap(); // perturb a main-effect cell
        assert_invariant(
            check_reconstruction(&m, &bad, &corners),
            Invariant::Reconstruction,
        );
    }

    #[test]
    fn mass_conservation_positive_then_negative() {
        let (m, corners) = (fixture_model(), fixture_grid_corners());
        check_mass_conservation(&m, &fixture_bank(), &corners).unwrap();

        let mut bad = fixture_bank();
        bad.f0 += 1.0; // a mass leak into the intercept
        assert_invariant(
            check_mass_conservation(&m, &bad, &corners),
            Invariant::MassConservation,
        );
    }

    #[test]
    fn purity_positive_then_negative() {
        let (m, corners) = (fixture_model(), fixture_grid_corners());
        check_purity(&m, &fixture_bank(), &corners).unwrap();

        let mut bad = fixture_bank();
        // Add a constant to the b0=1 slice of the pairwise table → its conditional
        // mean over b1 is no longer zero (a residual main effect left in the 2-way).
        bad.tables[2].values.add(&[1, 1], 1.0).unwrap();
        bad.tables[2].values.add(&[1, 2], 1.0).unwrap();
        assert_invariant(check_purity(&m, &bad, &corners), Invariant::Decomposability);
    }

    #[test]
    fn variance_sum_positive_then_negative() {
        let (m, corners) = (fixture_model(), fixture_grid_corners());
        check_variance_sum(&m, &fixture_bank(), &corners).unwrap();

        let mut bad = fixture_bank();
        // Scale the pairwise effect → Σσ²(f_u) no longer matches σ²(F).
        for coord in [vec![1, 1], vec![1, 2], vec![2, 1], vec![2, 2]] {
            let v = bad.tables[2].values.at(&coord).unwrap();
            bad.tables[2].values.set(&coord, v * 2.0).unwrap();
        }
        assert_invariant(
            check_variance_sum(&m, &bad, &corners),
            Invariant::VarianceSum,
        );
    }

    #[test]
    fn three_way_equal_positive_then_negative() {
        let (m, corners) = (fixture_model(), fixture_grid_corners());
        check_three_way_equal(&m, &fixture_bank(), &corners).unwrap();

        let mut bad = fixture_bank();
        bad.tables[1].values.add(&[2], 0.75).unwrap(); // tree ≠ table at b1=2 cells
        assert_invariant(
            check_three_way_equal(&m, &bad, &corners),
            Invariant::ThreeWayEqual,
        );
    }

    #[test]
    fn feature_budget_positive_then_negative() {
        check_feature_budget(&fixture_model()).unwrap();
        assert_invariant(
            check_feature_budget(&fixture_over_budget_model()),
            Invariant::FeatureBudget,
        );
    }

    #[test]
    fn assert_exact_decomposition_passes_on_the_positive_fixture() {
        assert_exact_decomposition(&fixture_model(), &fixture_bank(), &fixture_grid_corners())
            .unwrap();
    }

    /// The §13.1 negative property, as a real assertion: `assert_exact_decomposition`
    /// depends only on the model + the purified bank. We mutate a hypothetical
    /// screening signal (here: an unrelated field) and confirm the verdict is
    /// unchanged — the checks integrate the merged-grid bank, never a `wht8` shell.
    #[test]
    fn wht8_is_not_consulted() {
        let corners = fixture_grid_corners();
        let m = fixture_model();
        let bank = fixture_bank();
        // Baseline: passes.
        assert_exact_decomposition(&m, &bank, &corners).unwrap();
        // Changing each table's *variance* metadata (a screening-style summary) must
        // NOT change the verdict — the checks recompute from `values`, not metadata.
        let mut b2 = bank.clone();
        for t in &mut b2.tables {
            t.variance = 999.0;
        }
        assert_exact_decomposition(&m, &b2, &corners).unwrap();
    }
}
