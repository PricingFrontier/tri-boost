//! The oblivious boosting engine (spec §2.3, §2.5, §2.6, §2.9 / §06). Phase-0
//! stubs: the tree/model/booster types are frozen here (with the *real* scoring
//! `lookup` the invariant gates need), plus the `ExactnessMode` firewall. The
//! summed-Newton split-finder, histogram engine, and leaf refit land with §06.

use crate::boosters::DistillSpec;
use crate::cat::CatEncoderStore;
use crate::constraints::{InteractionPolicy, MonotoneMap};
use crate::data::{AxisKind, AxisProvenance, BinnedMatrix, BorderGrid};
use crate::error::PbError;
use crate::loss::{Link, Loss, ObjectiveTag};
use serde::{Deserialize, Serialize};

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
            let low = if bin == 0 {
                split.missing_left
            } else {
                bin <= split.bin_le
            };
            let bit = usize::from(low);
            idx |= bit << level;
        }
        self.leaves
            .get(idx)
            .copied()
            .ok_or_else(|| PbError::Internal {
                what: "oblivious leaf index escaped 0..8".into(),
            })
    }
}

/// The per-bin gradient/hessian histogram accumulator (spec §2.3). **`i64` bin
/// accumulators everywhere** (counts stay `u32`): summing per-row `i32` `g_q`/`h_q`
/// over up to `n_rows` bins can exceed `i32` range, which would trap under
/// `overflow-checks = true`. This is the SINGLE histogram accumulator type.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Hist {
    /// Per-(axis, bin) gradient sums.
    pub g: Vec<i64>,
    /// Per-(axis, bin) hessian sums.
    pub h: Vec<i64>,
    /// Per-(axis, bin) row counts.
    pub count: Vec<u32>,
}

/// The scale factors mapping full-precision g/h onto quantized integers (spec §2.3).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GradScale {
    /// Multiplier applied to gradients before integer rounding.
    pub g_scale: f32,
    /// Multiplier applied to hessians before integer rounding.
    pub h_scale: f32,
}

/// Quantized integer g/h for associative, order-independent histogram sums (spec
/// §2.3) — the bit-reproducibility AND ~2×-speed mechanism. Per-row values are
/// `i32`; the per-bin accumulators are the `i64` [`Hist`]. Leaves are always refit
/// from full-precision g/h, so quantization never touches table values.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct QuantGradHess {
    /// Quantized per-row gradients.
    pub g_q: Vec<i32>,
    /// Quantized per-row hessians.
    pub h_q: Vec<i32>,
    /// The scale factors used to quantize.
    pub scale: GradScale,
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
}

/// The public estimator (spec §2.9). Builder-configured, `fit → Model`,
/// sklearn-mirrored in Python. Phase-0 stub: `fit` is not yet implemented.
#[derive(Debug, Clone, Default)]
pub struct Booster {}

impl Booster {
    /// A fresh booster with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }

    /// Fit an ensemble. Phase-0 stub: returns [`PbError::Internal`] until §06 lands.
    ///
    /// # Errors
    /// Always [`PbError::Internal`] in Phase 0 (no learner yet).
    pub fn fit(&self, x: &BinnedMatrix, y: &[f32], spec: &FitSpec) -> Result<Model, PbError> {
        let _ = (x, y, spec);
        Err(PbError::Internal {
            what: "Booster::fit is not implemented in Phase 0 (§06 lands the learner)".into(),
        })
    }
}

/// The per-fit specification (spec §2.9): objective + per-row data + constraints +
/// optional distillation + the deterministic seed.
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
    /// Optional teacher distillation (off by default).
    pub distill: Option<DistillSpec>,
    /// The deterministic base seed threaded through every randomized stage.
    pub seed: u64,
}
