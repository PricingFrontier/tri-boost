//! The oblivious boosting engine (spec §2.3, §2.5, §2.6, §2.9 / §06). Owns the
//! trained-model types, the histogram accumulator, the split-finder, and the
//! boosting loop. Phase 1 (this milestone, M1.3) lands the full-precision histogram
//! engine ([`hist`]); the split-finder and `fit` loop land in M1.4/M1.5.

use crate::cat::CatEncoderStore;
use crate::constraints::{InteractionPolicy, MonotoneMap};
use crate::data::{AxisKind, AxisProvenance, BinnedMatrix, BorderGrid};
use crate::error::{Invariant, PbError};
use crate::loss::{Link, Loss, ObjectiveTag};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

pub mod boost;
pub mod hist;
pub mod split;

/// The SINGLE canonical missing low/left bit (spec §2.5 / §06.2, R-MISSING). The
/// reserved missing bin (bin 0) routes by its learned `missing_left`; every other
/// bin routes `bin <= bin_le`. Written ONCE and used identically at split evaluation,
/// the sample→leaf update ([`split::grow_oblivious_tree`]), tree scoring
/// ([`ObliviousTree::lookup`]), and table accumulation (§08) — agreement here is what
/// makes the tree, the purified tables, and the Shapley sum equal (I2 / ThreeWayEqual).
#[must_use]
pub(crate) fn low_bit(bin: u8, bin_le: u8, missing_left: bool) -> bool {
    if bin == 0 {
        missing_left
    } else {
        bin <= bin_le
    }
}

/// The Exact / Approximate firewall (spec §3). An `Exact` model passes all five
/// I2 checks and may export rating tables; any operation that cannot preserve them
/// flips the model to `Approximate { reason }` and refuses an `Exact` export. This
/// typed wall is the structural defense against death-by-a-thousand-cuts.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ExactnessMode {
    /// Passes all five invariant checks; exports an exact `TableBank`.
    #[default]
    Exact,
    /// Cannot pass the checks; `reason` explains why (e.g. nonlinear calibration).
    Approximate {
        /// Why the model is not exactly decomposable.
        reason: String,
    },
}

/// One shared level test of an oblivious tree (spec §2.5): `bin <= bin_le`.
///
/// `axis` is `u32` (fixed-width: serialized; `usize` would break cross-platform
/// byte-equality / the `wasm32` smoke build). `missing_left` is the explicit learned
/// default direction — the reserved missing bin (bin 0) routes left when `true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Split {
    /// Index of the axis this level tests.
    pub axis: u32,
    /// Inclusive upper bin for the "low" (left) child.
    pub bin_le: u8,
    /// Learned default direction for the reserved missing bin (bin 0).
    pub missing_left: bool,
}

/// A depth-3 oblivious tree (spec §2.5): one shared `(axis, bin_le)` test per level,
/// at most 3 DISTINCT raw features, `2^3 = 8` leaf values. Fewer than 3 levels is
/// legal (graceful early-termination → a lower-order fANOVA outcome, not an error).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObliviousTree {
    /// `1..=3` level tests, in test order (bit 0 = level 0).
    pub splits: Vec<Split>,
    /// Leaf values; index = `b0 | b1<<1 | b2<<2`; unused tail entries are `0.0`.
    pub leaves: [f32; 8],
    /// `splits.len()` as `u8`, in `1..=3`.
    pub depth: u8,
}

impl ObliviousTree {
    /// Score one row given its per-axis bin ids, returning the leaf value (spec §2.5).
    ///
    /// Uses the SINGLE canonical missing low-bit:
    /// `low = if bin == 0 { missing_left } else { bin <= bin_le }`. This exact form
    /// is shared by split evaluation, the packed scoring kernel, and these gates —
    /// the basis of tree/table equality (I2).
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if a split's `axis` is absent from `row_bins`;
    /// [`PbError::Internal`] if the folded leaf index escapes `0..8` (impossible —
    /// it is built from at most three bits — but checked rather than indexed raw).
    pub fn lookup(&self, row_bins: &[u8]) -> Result<f32, PbError> {
        let mut idx = 0usize;
        for (level, split) in self.splits.iter().enumerate() {
            let bin = *row_bins
                .get(split.axis as usize)
                .ok_or_else(|| PbError::ShapeMismatch {
                    what: format!("row has no axis {} for tree lookup", split.axis),
                })?;
            let bit = usize::from(low_bit(bin, split.bin_le, split.missing_left));
            idx |= bit << level;
        }
        self.leaves
            .get(idx)
            .copied()
            .ok_or_else(|| PbError::Internal {
                what: "oblivious leaf index escaped 0..8".into(),
            })
    }

    /// Construct a tree, enforcing I1 at the type boundary (spec §2.5 / §3): `depth`
    /// (= `splits.len()`) must be in `1..=3`, and the count of DISTINCT raw features
    /// across the splits (via `provenance`) must equal `depth` — i.e. each level
    /// tests a different raw feature. `leaves[depth.pow2()..]` are the unused tail.
    ///
    /// # Errors
    /// [`Invariant::FeatureBudget`] (as [`PbError::InvariantViolated`]) if the depth
    /// or distinct-raw-feature budget is violated; [`PbError::Internal`] if a split
    /// names an axis absent from `provenance`.
    pub fn try_new(
        splits: Vec<Split>,
        leaves: [f32; 8],
        provenance: &[AxisProvenance],
    ) -> Result<Self, PbError> {
        let depth = splits.len();
        if !(1..=3).contains(&depth) {
            return Err(PbError::invariant(Invariant::FeatureBudget));
        }
        let mut distinct: SmallVec<[u32; 3]> = SmallVec::new();
        for s in &splits {
            let raw = provenance
                .get(s.axis as usize)
                .ok_or_else(|| PbError::Internal {
                    what: format!("split axis {} absent from provenance", s.axis),
                })?
                .raw
                .0;
            if !distinct.contains(&raw) {
                distinct.push(raw);
            }
        }
        if distinct.len() != depth {
            return Err(PbError::invariant(Invariant::FeatureBudget));
        }
        Ok(Self {
            splits,
            leaves,
            depth: depth as u8,
        })
    }
}

/// The per-`(leaf, axis, bin)` gradient/hessian histogram accumulator (spec §06.3),
/// struct-of-arrays in `[leaf][axis][bin]` row-major order with a **uniform `n_bins`
/// stride** (the max grid bins over the built axes; shorter axes leave their high
/// bins zeroed). `count` stays `u32` (a bin holds at most `n_rows <= u32::MAX` rows).
///
/// **v1 uses `f64` accumulators** and earns determinism from a FIXED-ORDER fold
/// (feature-parallel, sequential within each axis — [`hist::build_histogram`]).
/// FLAG (spec reconciliation): §2.3/§06.3 specify `i64` accumulators, but that is the
/// *quantized* path; §14 ships full-precision `GradHess` only in v1 and defers
/// quantized integer histograms to v1.5 (M5-QHIST), where integer associativity
/// replaces the fixed-order fold. So the v1 accumulator is `f64`; the `i64` quantized
/// form returns with M5-QHIST.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Hist {
    /// Per-cell gradient sums (`[leaf][axis][bin]`, row-major).
    pub g: Vec<f64>,
    /// Per-cell hessian sums.
    pub h: Vec<f64>,
    /// Per-cell row counts.
    pub count: Vec<u32>,
    /// Number of leaves at this level (`2^depth`).
    pub n_leaves: usize,
    /// Number of axes (built features) in this histogram.
    pub n_axes: usize,
    /// Uniform per-axis bin stride (max grid bins over the built axes).
    pub n_bins: usize,
}

impl Hist {
    /// Try to allocate a zeroed histogram with the given shape. `g`/`h`/`count`
    /// each hold `n_leaves · n_axes · n_bins` cells.
    ///
    /// # Errors
    /// [`PbError::Internal`] if the shape arithmetic overflows or if the backing
    /// buffers cannot be reserved.
    pub fn try_zeros(n_leaves: usize, n_axes: usize, n_bins: usize) -> Result<Self, PbError> {
        let cells = Self::checked_cell_count(n_leaves, n_axes, n_bins)?;
        Ok(Self {
            g: Self::try_zeroed_vec(cells, "histogram g")?,
            h: Self::try_zeroed_vec(cells, "histogram h")?,
            count: Self::try_zeroed_vec(cells, "histogram count")?,
            n_leaves,
            n_axes,
            n_bins,
        })
    }

    /// The flat row-major offset of cell `(leaf, axis, bin)`, or `None` if any index
    /// is out of range (so callers stay panic-free without raw indexing).
    #[must_use]
    pub fn offset(&self, leaf: usize, axis: usize, bin: usize) -> Option<usize> {
        if leaf >= self.n_leaves || axis >= self.n_axes || bin >= self.n_bins {
            return None;
        }
        leaf.checked_mul(self.n_axes)?
            .checked_add(axis)?
            .checked_mul(self.n_bins)?
            .checked_add(bin)
    }

    /// `(n_leaves, n_axes, n_bins)` — the shape triple, for equality checks.
    #[must_use]
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.n_leaves, self.n_axes, self.n_bins)
    }

    /// Total number of cells (`n_leaves · n_axes · n_bins`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.g.len()
    }

    /// `true` if the histogram has no cells.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.g.is_empty()
    }

    pub(crate) fn checked_cell_count(
        n_leaves: usize,
        n_axes: usize,
        n_bins: usize,
    ) -> Result<usize, PbError> {
        n_leaves
            .checked_mul(n_axes)
            .and_then(|cells| cells.checked_mul(n_bins))
            .ok_or_else(|| PbError::Internal {
                what: "histogram shape overflows usize".into(),
            })
    }

    pub(crate) fn try_zeroed_vec<T>(cells: usize, what: &'static str) -> Result<Vec<T>, PbError>
    where
        T: Clone + Default,
    {
        let mut out = Vec::new();
        out.try_reserve_exact(cells)
            .map_err(|_| PbError::Internal {
                what: format!("{what} allocation failed"),
            })?;
        out.resize(cells, T::default());
        Ok(out)
    }
}

/// The scale factors mapping full-precision g/h onto quantized integers (spec §2.3).
/// Registered for M5-QHIST (v1.5); unused on the v1 full-precision path.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GradScale {
    /// Multiplier applied to gradients before integer rounding.
    pub g_scale: f32,
    /// Multiplier applied to hessians before integer rounding.
    pub h_scale: f32,
}

/// Quantized integer g/h for associative, order-independent histogram sums (spec
/// §2.3). Registered as the M5-QHIST (v1.5) future type — the v1 green spine
/// accumulates full-precision [`crate::loss::GradHess`] directly, so this is unused
/// in v1.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct QuantGradHess {
    /// Quantized per-row gradients.
    pub g_q: Vec<i32>,
    /// Quantized per-row hessians.
    pub h_q: Vec<i32>,
    /// The scale factors used to quantize.
    pub scale: GradScale,
}

/// Histogram accumulator precision (§06/§11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistPrecision {
    /// Full-precision f64 histogram sums (v1 green-spine default).
    FullF64,
    /// Quantized i32 per-row gradients/hessians accumulated as i64, then
    /// dequantized for the existing split scanner. Leaves are still refit from full
    /// precision.
    QuantizedI32,
}

impl Default for HistPrecision {
    fn default() -> Self {
        Self::FullF64
    }
}

/// Model-level metadata so a `Model` can serve and export categoricals + classifiers
/// without the caller re-supplying anything (spec §2.6, R-SCHEMA). Serialized with
/// the `Model`; `schema_version` covers it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSchema {
    /// Human-readable feature names (parallel to provenance).
    pub feature_names: Vec<String>,
    /// Per-axis kinds (reuses `AxisKind`).
    pub feature_kinds: Vec<AxisKind>,
    /// Frozen full-data categorical encoders (owned by §04).
    pub cat_encoders: CatEncoderStore,
    /// Class labels for a classifier; `None` for regression.
    pub class_labels: Option<Vec<String>>,
    /// The trained objective (link + loss + Tweedie power).
    pub objective: ObjectiveTag,
}

/// The trained ensemble (spec §2.6): intercept + weighted oblivious trees, the
/// shared binning grids, provenance, the loss/link, an exactness flag, and the
/// serve/export schema.
///
/// Inference: `raw(x) = f0 + offset + Σ alpha_t · tree_t.lookup(x)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    /// `link(weighted mean)` — a scalar intercept, never "tree 0".
    pub f0: f32,
    /// `(weight alpha, tree)` pairs; alphas allow DART/Nesterov/ensemble mixes.
    pub trees: Vec<(f32, ObliviousTree)>,
    /// Shared per-axis binning grids.
    pub grids: Vec<BorderGrid>,
    /// Per-axis provenance (maps each axis to its raw feature — drives I1).
    pub provenance: Vec<AxisProvenance>,
    /// The inverse-link family.
    pub link: Link,
    /// Exact / Approximate firewall state (§3).
    pub mode: ExactnessMode,
    /// Serve/export metadata (cats + classifier labels + objective).
    pub schema: ModelSchema,
    /// Monotone wire-version covering `Model` AND `schema`.
    pub schema_version: u32,
}

impl Model {
    /// Validate model structure after construction or deserialization.
    ///
    /// This is the §10 load gate: it re-checks fixed-width schema consistency, split
    /// axis bounds, finite scalar payloads, and the I1 feature-budget shape before a
    /// decoded model can be scored or exported.
    ///
    /// # Errors
    /// [`PbError::Serialization`] for a schema-version mismatch;
    /// [`PbError::ShapeMismatch`] for inconsistent parallel metadata;
    /// [`PbError::InvalidInput`] for malformed grids or non-finite scalars;
    /// [`PbError::InvariantViolated`] for an I1 feature-budget violation.
    pub fn validate(&self) -> Result<(), PbError> {
        if self.schema_version != crate::serialize::SCHEMA_VERSION {
            return Err(PbError::Serialization(format!(
                "model schema_version {} != build schema_version {}",
                self.schema_version,
                crate::serialize::SCHEMA_VERSION
            )));
        }
        if !self.f0.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!("model f0 must be finite, got {}", self.f0),
            });
        }
        if self.grids.len() != self.provenance.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "model grids len {} != provenance len {}",
                    self.grids.len(),
                    self.provenance.len()
                ),
            });
        }
        if self.schema.feature_names.len() != self.grids.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "schema feature_names len {} != grid count {}",
                    self.schema.feature_names.len(),
                    self.grids.len()
                ),
            });
        }
        if self.schema.feature_kinds.len() != self.grids.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "schema feature_kinds len {} != grid count {}",
                    self.schema.feature_kinds.len(),
                    self.grids.len()
                ),
            });
        }
        if self.schema.objective.link != self.link {
            return Err(PbError::InvalidInput {
                what: "schema objective link does not match model link".into(),
            });
        }
        for (axis, grid) in self.grids.iter().enumerate() {
            if grid.missing_bin != 0 {
                return Err(PbError::InvalidInput {
                    what: format!(
                        "grid {axis} missing_bin must be 0, got {}",
                        grid.missing_bin
                    ),
                });
            }
            if grid.n_bins == 0 || grid.n_bins > 255 {
                return Err(PbError::InvalidInput {
                    what: format!("grid {axis} n_bins must be in 1..=255, got {}", grid.n_bins),
                });
            }
            let expected_bins =
                u16::try_from(grid.borders.len().checked_add(2).ok_or_else(|| {
                    PbError::Internal {
                        what: "grid border count overflow".into(),
                    }
                })?)
                .map_err(|_| PbError::InvalidInput {
                    what: format!("grid {axis} has too many borders"),
                })?;
            if grid.n_bins != expected_bins && !(grid.n_bins == 1 && grid.borders.is_empty()) {
                return Err(PbError::InvalidInput {
                    what: format!(
                        "grid {axis} n_bins {} inconsistent with {} borders",
                        grid.n_bins,
                        grid.borders.len()
                    ),
                });
            }
            for (i, &border) in grid.borders.iter().enumerate() {
                if !border.is_finite() {
                    return Err(PbError::InvalidInput {
                        what: format!("grid {axis} border {i} must be finite"),
                    });
                }
            }
            for pair in grid.borders.windows(2) {
                if let [a, b] = pair {
                    if a >= b {
                        return Err(PbError::InvalidInput {
                            what: format!("grid {axis} borders must be strictly ascending"),
                        });
                    }
                }
            }
        }
        for (tree_idx, (alpha, tree)) in self.trees.iter().enumerate() {
            if !alpha.is_finite() {
                return Err(PbError::InvalidInput {
                    what: format!("tree {tree_idx} alpha must be finite, got {alpha}"),
                });
            }
            let depth = usize::from(tree.depth);
            if !(1..=3).contains(&depth) || tree.splits.len() != depth {
                return Err(PbError::invariant(Invariant::FeatureBudget));
            }
            let mut distinct: SmallVec<[u32; 3]> = SmallVec::new();
            for split in &tree.splits {
                let axis = split.axis as usize;
                let prov = self
                    .provenance
                    .get(axis)
                    .ok_or_else(|| PbError::ShapeMismatch {
                        what: format!("tree {tree_idx} split axis {axis} absent from provenance"),
                    })?;
                let grid = self.grids.get(axis).ok_or_else(|| PbError::ShapeMismatch {
                    what: format!("tree {tree_idx} split axis {axis} absent from grids"),
                })?;
                if u16::from(split.bin_le) >= grid.n_bins {
                    return Err(PbError::InvalidInput {
                        what: format!(
                            "tree {tree_idx} split bin_le {} outside grid {axis} n_bins {}",
                            split.bin_le, grid.n_bins
                        ),
                    });
                }
                if !distinct.contains(&prov.raw.0) {
                    distinct.push(prov.raw.0);
                }
            }
            if distinct.len() != depth {
                return Err(PbError::invariant(Invariant::FeatureBudget));
            }
            for (leaf, &value) in tree.leaves.iter().enumerate() {
                if !value.is_finite() {
                    return Err(PbError::InvalidInput {
                        what: format!("tree {tree_idx} leaf {leaf} must be finite, got {value}"),
                    });
                }
            }
        }
        Ok(())
    }

    /// The ensemble raw score for one row's bin ids, in full `f64`
    /// (`f0 + Σ alpha_t · tree_t.lookup(x)`), used by the §08 reconstruction gate.
    ///
    /// # Errors
    /// Propagates any [`ObliviousTree::lookup`] failure.
    pub fn ensemble_f64(&self, row_bins: &[u8]) -> Result<f64, PbError> {
        let mut acc = f64::from(self.f0);
        for (alpha, tree) in &self.trees {
            acc += f64::from(*alpha) * f64::from(tree.lookup(row_bins)?);
        }
        Ok(acc)
    }

    /// Raw score for one already-binned row, accumulated in the production `f32`
    /// scoring width (spec §10 path A).
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `row_bins` does not match this model's width;
    /// plus propagated tree lookup errors.
    pub fn score_trees_row(&self, row_bins: &[u8], offset: f32) -> Result<f32, PbError> {
        if row_bins.len() != self.grids.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "row width {} != model grid count {}",
                    row_bins.len(),
                    self.grids.len()
                ),
            });
        }
        let mut acc = self.f0 + offset;
        for (alpha, tree) in &self.trees {
            acc += *alpha * tree.lookup(row_bins)?;
        }
        Ok(acc)
    }

    /// Batch raw scores over a column-major [`BinnedMatrix`] into `out`.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if matrix shape or grids/provenance do not match
    /// the model; [`PbError::InvalidInput`] if any bin escapes its grid.
    pub fn score_trees(
        &self,
        x: &BinnedMatrix,
        offset: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<(), PbError> {
        self.validate_binned_matrix(x)?;
        let n_rows = x.n_rows as usize;
        if out.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("out len {} != n_rows {n_rows}", out.len()),
            });
        }
        if let Some(off) = offset {
            if off.len() != n_rows {
                return Err(PbError::ShapeMismatch {
                    what: format!("offset len {} != n_rows {n_rows}", off.len()),
                });
            }
        }
        let mut row = vec![0u8; self.grids.len()];
        for r in 0..n_rows {
            for (slot, col) in row.iter_mut().zip(&x.data) {
                *slot = *col.get(r).ok_or_else(|| PbError::Internal {
                    what: "validated binned column lost a row".into(),
                })?;
            }
            let off = offset.and_then(|o| o.get(r).copied()).unwrap_or(0.0);
            let score = self.score_trees_row(&row, off)?;
            let dst = out.get_mut(r).ok_or_else(|| PbError::Internal {
                what: "score_trees output row escaped buffer".into(),
            })?;
            *dst = score;
        }
        Ok(())
    }

    /// Response-space predictions from an already-binned design.
    ///
    /// # Errors
    /// Propagates [`Model::score_trees`] validation/scoring failures.
    pub fn predict_binned(
        &self,
        x: &BinnedMatrix,
        offset: Option<&[f32]>,
    ) -> Result<Vec<f32>, PbError> {
        let mut raw = vec![0.0_f32; x.n_rows as usize];
        self.score_trees(x, offset, &mut raw)?;
        for v in &mut raw {
            *v = inverse_link(self.link, *v);
        }
        Ok(raw)
    }

    /// Response-space predictions from a binned design. Alias for
    /// [`Model::predict_binned`] until raw-data ingest is exposed at this layer.
    ///
    /// # Errors
    /// Propagates [`Model::predict_binned`] failures.
    pub fn predict(&self, x: &BinnedMatrix, offset: Option<&[f32]>) -> Result<Vec<f32>, PbError> {
        self.predict_binned(x, offset)
    }

    fn validate_binned_matrix(&self, x: &BinnedMatrix) -> Result<(), PbError> {
        let n_rows = x.n_rows as usize;
        if x.data.len() != self.grids.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "matrix has {} columns, model has {} features",
                    x.data.len(),
                    self.grids.len()
                ),
            });
        }
        if x.grids != self.grids {
            return Err(PbError::ShapeMismatch {
                what: "matrix grids do not match model grids".into(),
            });
        }
        if x.provenance != self.provenance {
            return Err(PbError::ShapeMismatch {
                what: "matrix provenance does not match model provenance".into(),
            });
        }
        for (axis, col) in x.data.iter().enumerate() {
            if col.len() != n_rows {
                return Err(PbError::ShapeMismatch {
                    what: format!("column {axis} len {} != n_rows {n_rows}", col.len()),
                });
            }
            let grid = self.grids.get(axis).ok_or_else(|| PbError::Internal {
                what: "model grid disappeared during score validation".into(),
            })?;
            for (row, &bin) in col.iter().enumerate() {
                if u16::from(bin) >= grid.n_bins {
                    return Err(PbError::InvalidInput {
                        what: format!(
                            "column {axis} row {row} bin {bin} outside grid n_bins {}",
                            grid.n_bins
                        ),
                    });
                }
            }
        }
        Ok(())
    }
}

fn inverse_link(link: Link, raw: f32) -> f32 {
    match link {
        Link::Identity => raw,
        Link::Log => raw.clamp(-30.0, 30.0).exp(),
        Link::Logit => {
            if raw >= 0.0 {
                let z = (-raw).clamp(-30.0, 30.0).exp();
                1.0 / (1.0 + z)
            } else {
                let z = raw.clamp(-30.0, 30.0).exp();
                z / (1.0 + z)
            }
        }
    }
}

/// Row-sampling strategy for split search (§06 / M5-MVS).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sampling {
    /// Use every row for split search (the v1 green-spine default).
    Full,
    /// Minimal-variance-style probability-proportional-to-gradient row sampling.
    ///
    /// `rate` is the target sample fraction and `min_rows` is a lower bound on the
    /// selected row count. The final tree leaves are still refit from all rows.
    Mvs {
        /// Target row fraction, `0 < rate <= 1`.
        rate: f32,
        /// Minimum sampled row count.
        min_rows: u32,
    },
}

impl Default for Sampling {
    fn default() -> Self {
        Self::Full
    }
}

/// The optimizer configuration (spec §06.1). v1 green-spine subset: the full §06.1
/// knob set (`colsample_*`, `hist_precision`, credibility floors, `accel`, LR schedule,
/// early stopping) lands with its features — v1.5 adds row sampling via
/// [`Sampling::Mvs`]. FLAG: `Config` remains a subset of the full §06.1 type for now.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Number of boosting rounds (upper bound; growth also stops if a round can't split).
    pub n_trees: u32,
    /// Learning rate applied to each tree's leaf values.
    pub learning_rate: f32,
    /// L2 leaf regularizer `λ` (in `w* = −G/(H+λ)` and the gain).
    pub lambda: f32,
    /// `gamma` floor: a level terminates if the best gain is `<= min_split_gain`.
    pub min_split_gain: f32,
    /// Leaf-stage `|w*|`-clamp (LightGBM `max_delta_step`, §05.6/§06.4). `None` falls
    /// back to `Loss::max_delta_step()` (Poisson ⇒ `Some(0.7)`); a non-`None` value wins.
    /// Applied on the full-precision aggregated Newton step, never per-row `h`.
    pub max_delta_step: Option<f32>,
    /// Row sampler used for split search. [`Sampling::Full`] is the inert default.
    pub sampling: Sampling,
    /// Histogram precision used for split search.
    pub hist_precision: HistPrecision,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_trees: 1000,
            learning_rate: 0.05,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Sampling::Full,
            hist_precision: HistPrecision::FullF64,
        }
    }
}

impl Config {
    /// Validate the configuration.
    ///
    /// # Errors
    /// [`PbError::InvalidConfig`] if `n_trees == 0`, `learning_rate` is non-finite or
    /// `<= 0`, `lambda` is non-finite or `< 0`, or `min_split_gain` is non-finite or `< 0`.
    pub fn validate(&self) -> Result<(), PbError> {
        if self.n_trees == 0 {
            return Err(PbError::InvalidConfig {
                what: "n_trees must be > 0".into(),
            });
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "learning_rate must be finite and > 0, got {}",
                    self.learning_rate
                ),
            });
        }
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(PbError::InvalidConfig {
                what: format!("lambda must be finite and >= 0, got {}", self.lambda),
            });
        }
        if !self.min_split_gain.is_finite() || self.min_split_gain < 0.0 {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "min_split_gain must be finite and >= 0, got {}",
                    self.min_split_gain
                ),
            });
        }
        if let Some(d) = self.max_delta_step {
            if !d.is_finite() || d <= 0.0 {
                return Err(PbError::InvalidConfig {
                    what: format!("max_delta_step must be finite and > 0 when set, got {d}"),
                });
            }
        }
        match self.sampling {
            Sampling::Full => {}
            Sampling::Mvs { rate, min_rows } => {
                if !rate.is_finite() || rate <= 0.0 || rate > 1.0 {
                    return Err(PbError::InvalidConfig {
                        what: format!("MVS rate must be finite and in (0, 1], got {rate}"),
                    });
                }
                if min_rows == 0 {
                    return Err(PbError::InvalidConfig {
                        what: "MVS min_rows must be > 0".into(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// The public estimator (spec §2.9). Builder-configured, `fit → Model`,
/// sklearn-mirrored in Python.
#[derive(Debug, Clone, Default)]
pub struct Booster {
    config: Config,
}

impl Booster {
    /// A fresh booster with the default [`Config`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }

    /// A booster with an explicit [`Config`].
    #[must_use]
    pub fn with_config(config: Config) -> Self {
        Self { config }
    }

    /// The booster's configuration.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Fit an ensemble (spec §06.6): `f0 = link(weighted mean)`, then per round a
    /// full-precision `grad_hess` pass → `grow_oblivious_tree` → `update_raw`, until
    /// `n_trees` rounds or a round cannot split. Emits an `Exact` [`Model`].
    ///
    /// # Errors
    /// [`PbError::InvalidConfig`] on a bad config; [`PbError::ShapeMismatch`] on a
    /// length mismatch; plus any propagated [`Loss`]/binning/grow error.
    pub fn fit(&self, x: &BinnedMatrix, y: &[f32], spec: &FitSpec) -> Result<Model, PbError> {
        boost::fit(&self.config, x, y, spec)
    }
}

/// The per-fit specification (spec §2.9): objective + per-row data + constraints +
/// the deterministic seed.
pub struct FitSpec<'a> {
    /// The objective.
    pub loss: &'a dyn Loss,
    /// Optional per-row weights.
    pub weight: Option<&'a [f32]>,
    /// Optional per-row exposure (offset = `log(e)`; anchors base level = 1.000).
    pub exposure: Option<&'a [f32]>,
    /// Monotone constraints keyed by feature name.
    pub monotone: MonotoneMap,
    /// Interaction-order limit + optional group whitelist.
    pub interaction: InteractionPolicy,
    /// The deterministic base seed threaded through every randomized stage.
    pub seed: u64,
}
