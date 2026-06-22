//! The `f32 → u8` map (spec §03.4) and the column binner (§03.4 hot path). The hot
//! map is a branchless, panic-free binary search: `partition_point` does the work,
//! and the single `+ 1` is bounded by the cardinality invariant, so no indexing can
//! go out of bounds and no arithmetic can overflow.

use crate::data::grid::build_grid;
use crate::data::{AxisKind, AxisProvenance, BinConfig, BinnedMatrix, BorderGrid, FeatureId};
use crate::error::PbError;
use rayon::prelude::*;

/// Map a finite-or-missing value to its bin id (spec §03.4).
///
/// `NaN` (and Arrow nulls, mapped to `NaN` at ingest) is the ONLY missing case and
/// returns bin `0`. Every finite value — including out-of-range and `±inf` (which are
/// non-NaN) — clamps to the first/last finite data bin via `partition_point`:
/// `bin(v) = 1 + (count of borders strictly below v)`.
///
/// # Errors
/// [`PbError::Internal`] only on the impossible case `bin index > u8` (the
/// cardinality invariant `borders.len() <= 253` rules it out); never panics.
pub fn bin(v: f32, g: &BorderGrid) -> Result<u8, PbError> {
    if v.is_nan() {
        return Ok(g.missing_bin); // == 0
    }
    // k in 0..=borders.len() (<= 253 by the R-BINS cardinality invariant).
    let k = g.borders.partition_point(|&b| b < v);
    // data bin = k + 1, in 1..=n_data_bins <= 254. The `+ 1` cannot overflow usize
    // (k <= 253). u8::try_from never fails here, but maps the impossible case to a
    // typed error rather than panicking under overflow-checks.
    u8::try_from(k + 1).map_err(|_| PbError::Internal {
        what: "bin index exceeded u8 (borders.len() > 253?)".into(),
    })
}

/// Bin a set of column-major raw `f32` feature columns into a [`BinnedMatrix`]
/// (spec §03.4). Each column gets its own frozen [`BorderGrid`] (built once) and is
/// then mapped to `u8` bin ids; the per-feature work is rayon-parallel and
/// order-independent, so the result is byte-identical across thread counts (§1 `[GATE]`).
/// Every axis is `AxisKind::Numeric` with `raw = FeatureId(column index)`.
///
/// # Errors
/// [`PbError::InvalidConfig`] if `cfg` is invalid; [`PbError::ShapeMismatch`] if the
/// columns or `weight` disagree on length; [`PbError::InvalidInput`] if there are
/// more than `u32::MAX` rows; plus any [`build_grid`] error.
pub fn bin_columns(
    columns: &[&[f32]],
    weight: Option<&[f32]>,
    cfg: &BinConfig,
    seed: u64,
) -> Result<BinnedMatrix, PbError> {
    cfg.validate()?;
    let n_rows = columns.first().map_or(0, |c| c.len());
    for c in columns {
        if c.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: "feature columns have unequal lengths".into(),
            });
        }
    }
    if let Some(w) = weight {
        if w.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("weight len {} != n_rows {n_rows}", w.len()),
            });
        }
    }
    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| PbError::InvalidInput {
        what: "more than u32::MAX rows is out of scope for v1".into(),
    })?;

    // Per-feature: build grid + bin the column. Order-preserving collect ⇒ the
    // assembled matrix is independent of how rayon scheduled the features.
    let per_feature: Result<Vec<(BorderGrid, Vec<u8>)>, PbError> = columns
        .par_iter()
        .enumerate()
        .map(|(f, &col)| {
            let feat = FeatureId(u32::try_from(f).map_err(|_| PbError::Internal {
                what: "feature index exceeded u32".into(),
            })?);
            let grid = build_grid(col, weight, cfg, seed, feat)?;
            let mut binned = Vec::with_capacity(col.len());
            for &v in col {
                binned.push(bin(v, &grid)?);
            }
            Ok((grid, binned))
        })
        .collect();
    let per_feature = per_feature?;

    let mut data = Vec::with_capacity(per_feature.len());
    let mut grids = Vec::with_capacity(per_feature.len());
    let mut provenance = Vec::with_capacity(per_feature.len());
    for (f, (grid, binned)) in per_feature.into_iter().enumerate() {
        let raw = FeatureId(u32::try_from(f).map_err(|_| PbError::Internal {
            what: "feature index exceeded u32".into(),
        })?);
        data.push(binned);
        grids.push(grid);
        provenance.push(AxisProvenance {
            raw,
            kind: AxisKind::Numeric,
        });
    }
    Ok(BinnedMatrix {
        data,
        n_rows: n_rows_u32,
        grids,
        provenance,
    })
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

    fn grid(borders: Vec<f32>) -> BorderGrid {
        let n_bins = u16::try_from(borders.len() + 2).unwrap();
        BorderGrid {
            borders,
            n_bins,
            missing_bin: 0,
        }
    }

    #[test]
    fn bin_semantics_nan_inf_and_borders() {
        let g = grid(vec![1.5, 2.5]); // 3 data bins: (-inf,1.5]→1, (1.5,2.5]→2, (2.5,inf)→3
        assert_eq!(bin(f32::NAN, &g).unwrap(), 0); // only NaN is missing
        assert_eq!(bin(1.0, &g).unwrap(), 1);
        assert_eq!(bin(1.5, &g).unwrap(), 1); // upper-inclusive
        assert_eq!(bin(2.0, &g).unwrap(), 2);
        assert_eq!(bin(3.0, &g).unwrap(), 3);
        // ±inf are non-NaN ⇒ clamp to first/last finite bin, NOT missing.
        assert_eq!(bin(f32::NEG_INFINITY, &g).unwrap(), 1);
        assert_eq!(bin(f32::INFINITY, &g).unwrap(), 3);
    }

    #[test]
    fn bin_is_non_decreasing_in_value() {
        let g = grid(vec![0.0, 1.0, 2.0, 3.0]);
        let mut prev = 0;
        for step in 0..=40 {
            let v = -1.0 + step as f32 * 0.1;
            let b = bin(v, &g).unwrap();
            assert!(b >= prev, "bin must be non-decreasing");
            prev = b;
        }
    }

    #[test]
    fn max_cardinality_grid_never_overflows_u8() {
        // 253 borders ⇒ 254 data bins ⇒ max bin id 254 (fits u8, no panic).
        let borders: Vec<f32> = (0..253).map(|i| i as f32).collect();
        let g = grid(borders);
        assert_eq!(g.n_bins, 255);
        // A value above the top border lands in the 254th data bin.
        assert_eq!(bin(1.0e9, &g).unwrap(), 254);
        assert_eq!(bin(f32::INFINITY, &g).unwrap(), 254);
        // Every finite probe stays in 1..=254, missing stays 0.
        for step in 0..300 {
            let b = bin(step as f32 - 10.0, &g).unwrap();
            assert!((1..=254).contains(&b));
        }
        assert_eq!(bin(f32::NAN, &g).unwrap(), 0);
    }

    #[test]
    fn bin_columns_populates_provenance_and_shapes() {
        let c0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let c1: Vec<f32> = vec![10.0, 10.0, 20.0, f32::NAN];
        let cols: Vec<&[f32]> = vec![&c0, &c1];
        let m = bin_columns(&cols, None, &BinConfig::default(), 0).unwrap();
        assert_eq!(m.n_rows, 4);
        assert_eq!(m.data.len(), 2);
        assert_eq!(m.grids.len(), 2);
        assert_eq!(m.provenance[0].raw, FeatureId(0));
        assert_eq!(m.provenance[1].raw, FeatureId(1));
        assert!(matches!(m.provenance[0].kind, AxisKind::Numeric));
        // The NaN in c1 must bin to 0 (missing).
        assert_eq!(m.data[1][3], 0);
    }

    #[test]
    fn bin_columns_rejects_unequal_columns() {
        let c0: Vec<f32> = vec![1.0, 2.0];
        let c1: Vec<f32> = vec![1.0];
        let cols: Vec<&[f32]> = vec![&c0, &c1];
        assert!(matches!(
            bin_columns(&cols, None, &BinConfig::default(), 0),
            Err(PbError::ShapeMismatch { .. })
        ));
    }
}
