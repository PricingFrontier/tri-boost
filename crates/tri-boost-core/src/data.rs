//! Data model & binning (spec §2.1–2.2 / §03). Phase-0 stubs: the binned-matrix
//! and border-grid types are frozen here; `bin()`/`build_grid`, the train-vs-serve
//! matrices, exposure plumbing, and rare-level collapse land with §03.

use crate::cat::TsEncodingId;
use serde::{Deserialize, Serialize};

/// Index into the user's original feature columns — the RAW underlying feature
/// (spec §2.1). `u32` because it appears in serialized provenance (no `usize` on
/// the wire, §02.8). `Ord + Hash` so the ≤3-distinct-raw-feature budget (I1) can be
/// counted over a set of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FeatureId(pub u32);

/// What kind of model column an axis is (spec §2.1). A model column may be a raw
/// numeric, a categorical Target-Statistic axis, or the reserved missing axis.
/// Provenance (below) tracks the RAW feature behind an axis so I1 is enforced on
/// distinct raw features, never on encoded columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AxisKind {
    /// A raw numeric feature, binned by quantile/midpoint borders.
    Numeric,
    /// A categorical feature encoded as a Target-Statistic ordinal axis.
    CategoricalTS {
        /// Which frozen TS encoding produced this axis.
        encoding: TsEncodingId,
    },
    /// The reserved missing-indicator axis (bin 0 semantics).
    Missing,
}

/// Tracks the RAW feature(s) behind an axis (spec §2.1) so the ≤3-feature invariant
/// (I1) is enforced on DISTINCT raw features, not on encoded columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxisProvenance {
    /// The raw feature this axis was derived from.
    pub raw: FeatureId,
    /// How the axis was produced from that raw feature.
    pub kind: AxisKind,
}

/// One feature's quantile/midpoint borders (spec §2.2). Bin 0 is the reserved
/// MISSING bin.
///
/// **Canonical cardinality (R-BINS).** `borders[k]` is the upper border of bin `k`
/// (strictly ascending). `bin(x) = 1 + (count of borders strictly below x)`, so data
/// bins are `1..=n_data_bins` where `n_data_bins = borders.len() + 1`, and
/// `n_bins = n_data_bins + 1` (data + missing). Constraint:
/// `borders.len() <= max_bin - 1` (default `max_bin = 254`), so `n_bins <= 255`
/// (fits `u8`, values `0..=254`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BorderGrid {
    /// Strictly-ascending upper borders; `borders[k]` is the top of data bin `k+1`.
    pub borders: Vec<f32>,
    /// Total bin count including the missing bin (`borders.len() + 2`).
    pub n_bins: u16,
    /// The reserved missing bin id (always `0` in v1).
    pub missing_bin: u8,
}

/// Column-major, pre-binned design matrix (spec §2.2). `data[f]` is column `f` as
/// bin ids. `f32 → u8` binning happens once at ingest; the grid persists in the
/// `Model`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinnedMatrix {
    /// `[n_features][n_rows]`, column-major; each entry is a `u8` bin id.
    pub data: Vec<Vec<u8>>,
    /// Row count. Fixed-width `u32`: it is serialized, and `usize` would break
    /// cross-platform byte-equality / the `wasm32` smoke build (§02.8).
    pub n_rows: u32,
    /// Per-feature border grids (parallel to `data`).
    pub grids: Vec<BorderGrid>,
    /// Per-axis provenance (parallel to `data`).
    pub provenance: Vec<AxisProvenance>,
}
