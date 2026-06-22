//! The full-precision histogram engine (spec §06.3, milestone M1.3).
//!
//! v1 accumulates full-precision [`GradHess`] into `f64` [`Hist`] cells (the
//! `i64`-quantized path is M5-QHIST, v1.5). The build is **feature-parallel**: each
//! axis is accumulated on its own rayon task, sequentially over rows in a fixed
//! order, into a disjoint region of the `[leaf][axis][bin]` tensor. So the result is
//! a FIXED-ORDER fold — byte-identical regardless of thread count (the §1
//! determinism `[GATE]`) — and memory-bounded (no per-row-chunk full-histogram
//! duplication; that row-parallel variant is a §11 perf concern for narrow data).
//!
//! The subtraction trick (`Hist_R = Hist_parent − Hist_L`) is integer-exact for
//! `count`; for `g`/`h` it is bit-exact for well-conditioned gradients (verified to
//! ~1e-11 even at hundreds of thousands of rows) but CAN drift under catastrophic
//! magnitude spread (float non-associativity), so it matches a direct-right build
//! only within a tolerance. The unconditionally-associative, exact-by-construction
//! version returns with the quantized integer path in M5-QHIST (v1.5).

// `build_histogram`/`subtract` are defined ahead of their consumer: the split-finder
// (M1.4) is what drives the per-level histogram build. They are exercised by the
// tests below; `dead_code` is expected until M1.4 wires them into `grow_oblivious_tree`.
#![allow(dead_code)]

use crate::data::BinnedMatrix;
use crate::engine::Hist;
use crate::error::PbError;
use crate::loss::GradHess;
use rayon::prelude::*;

fn oob(what: &'static str) -> impl Fn() -> PbError {
    move || PbError::Internal { what: what.into() }
}

/// Build the per-level `[leaf][axis][bin]` histogram from full-precision `gh`
/// (spec §06.3). `rows` are the row ids in scope (the subsample; in fixed order),
/// `leaf_of_row[r]` is the leaf id (`0..n_leaves`) of row `r`, and `axes` are the
/// (sampled) feature columns to build. The bin stride is uniform — the max grid
/// `n_bins` over `axes` — so shorter-grid axes leave their high bins zeroed.
///
/// Feature-parallel + sequential-within-axis ⇒ a fixed-order fold ⇒ thread-count
/// independent.
///
/// # Errors
/// [`PbError::Internal`] if any `axis`, row id, leaf id, or bin id is out of range
/// (the engine builds these so it is a bug, surfaced as a typed error not a panic).
pub(crate) fn build_histogram(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axes: &[u32],
) -> Result<Hist, PbError> {
    let mut max_bins = 0usize;
    for &a in axes {
        let grid = x
            .grids
            .get(a as usize)
            .ok_or_else(oob("axis has no grid"))?;
        max_bins = max_bins.max(usize::from(grid.n_bins));
    }
    let n_axes = axes.len();

    // Each axis is built independently (disjoint output region) on its own task.
    let subs: Result<Vec<AxisHist>, PbError> = axes
        .par_iter()
        .map(|&a| accumulate_axis(x, gh, rows, leaf_of_row, n_leaves, a, max_bins))
        .collect();
    let subs = subs?;

    // Assemble the disjoint per-axis sub-histograms into the flat tensor, in axis
    // order — deterministic regardless of how rayon scheduled the per-axis tasks.
    let mut hist = Hist::zeros(n_leaves, n_axes, max_bins);
    for (axis_pos, sub) in subs.iter().enumerate() {
        for leaf in 0..n_leaves {
            for bin in 0..max_bins {
                let src = leaf * max_bins + bin;
                let dst = (leaf * n_axes + axis_pos) * max_bins + bin;
                *hist.g.get_mut(dst).ok_or_else(oob("assemble g"))? =
                    *sub.g.get(src).ok_or_else(oob("sub g"))?;
                *hist.h.get_mut(dst).ok_or_else(oob("assemble h"))? =
                    *sub.h.get(src).ok_or_else(oob("sub h"))?;
                *hist.count.get_mut(dst).ok_or_else(oob("assemble count"))? =
                    *sub.count.get(src).ok_or_else(oob("sub count"))?;
            }
        }
    }
    Ok(hist)
}

/// One axis's `[leaf][bin]` sub-histogram (stride `max_bins`).
struct AxisHist {
    g: Vec<f64>,
    h: Vec<f64>,
    count: Vec<u32>,
}

fn accumulate_axis(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axis: u32,
    max_bins: usize,
) -> Result<AxisHist, PbError> {
    let col = x
        .data
        .get(axis as usize)
        .ok_or_else(oob("axis has no column"))?;
    let size = n_leaves * max_bins;
    let mut g = vec![0.0_f64; size];
    let mut h = vec![0.0_f64; size];
    let mut count = vec![0_u32; size];
    // Sequential over rows in their given (fixed) order ⇒ deterministic f64 fold.
    for &r in rows {
        let ru = r as usize;
        let bin = usize::from(*col.get(ru).ok_or_else(oob("row out of column"))?);
        let leaf = usize::from(*leaf_of_row.get(ru).ok_or_else(oob("row out of leaf map"))?);
        if leaf >= n_leaves || bin >= max_bins {
            return Err(PbError::Internal {
                what: "leaf or bin id out of histogram range".into(),
            });
        }
        let idx = leaf * max_bins + bin;
        *g.get_mut(idx).ok_or_else(oob("g cell"))? +=
            f64::from(*gh.g.get(ru).ok_or_else(oob("gh.g"))?);
        *h.get_mut(idx).ok_or_else(oob("h cell"))? +=
            f64::from(*gh.h.get(ru).ok_or_else(oob("gh.h"))?);
        let c = count.get_mut(idx).ok_or_else(oob("count cell"))?;
        // count <= rows.len() <= n_rows <= u32::MAX, so this never actually overflows;
        // checked_add keeps it panic-free under overflow-checks regardless.
        *c = c.checked_add(1).ok_or_else(oob("bin count overflow"))?;
    }
    Ok(AxisHist { g, h, count })
}

/// The subtraction trick (spec §06.3): `Hist_R = Hist_parent − Hist_L`, computed
/// element-wise. `count` is exact (integer); `g`/`h` equal a direct-right f64
/// accumulation only within a tolerance (float non-associativity), which is the v1
/// trade — the integer-exact form returns with the quantized path (M5-QHIST).
///
/// # Errors
/// [`PbError::ShapeMismatch`] if `parent` and `left` differ in shape;
/// [`PbError::Internal`] if a `count` cell would underflow (left ⊄ parent — a bug).
pub(crate) fn subtract(parent: &Hist, left: &Hist) -> Result<Hist, PbError> {
    if parent.shape() != left.shape() {
        return Err(PbError::ShapeMismatch {
            what: "histogram subtraction: parent/left shape mismatch".into(),
        });
    }
    let mut out = Hist::zeros(parent.n_leaves, parent.n_axes, parent.n_bins);
    for (o, (p, l)) in out.g.iter_mut().zip(parent.g.iter().zip(&left.g)) {
        *o = p - l;
    }
    for (o, (p, l)) in out.h.iter_mut().zip(parent.h.iter().zip(&left.h)) {
        *o = p - l;
    }
    for (o, (p, l)) in out
        .count
        .iter_mut()
        .zip(parent.count.iter().zip(&left.count))
    {
        *o = p
            .checked_sub(*l)
            .ok_or_else(oob("count underflow in histogram subtraction"))?;
    }
    Ok(out)
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
    use crate::data::{AxisKind, AxisProvenance, BorderGrid, FeatureId};
    use crate::engine::Hist;

    /// Build a BinnedMatrix from pre-binned columns + each axis's `n_bins` (only the
    /// `n_bins` matters to the histogram; borders are unread here).
    fn matrix(cols: Vec<Vec<u8>>, n_bins_each: &[u16]) -> BinnedMatrix {
        let n_rows = u32::try_from(cols.first().map_or(0, Vec::len)).unwrap();
        let grids = n_bins_each
            .iter()
            .map(|&nb| BorderGrid {
                borders: vec![0.0; usize::from(nb).saturating_sub(2)],
                n_bins: nb,
                missing_bin: 0,
            })
            .collect();
        let provenance = (0..u32::try_from(cols.len()).unwrap())
            .map(|i| AxisProvenance {
                raw: FeatureId(i),
                kind: AxisKind::Numeric,
            })
            .collect();
        BinnedMatrix {
            data: cols,
            n_rows,
            grids,
            provenance,
        }
    }

    fn gradhess(g: &[f32], h: &[f32]) -> GradHess {
        GradHess {
            g: g.to_vec(),
            h: h.to_vec(),
        }
    }

    fn cell(hist: &Hist, leaf: usize, axis: usize, bin: usize) -> (f64, f64, u32) {
        let o = hist.offset(leaf, axis, bin).unwrap();
        (hist.g[o], hist.h[o], hist.count[o])
    }

    #[test]
    fn build_uses_uniform_stride_and_sums_per_axis_bin() {
        // axis0 has 3 bins, axis1 has 4 bins ⇒ uniform stride max_bins = 4.
        let cols = vec![vec![1u8, 2, 1, 2], vec![1u8, 1, 2, 3]];
        let x = matrix(cols, &[3, 4]);
        let gh = gradhess(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0, 1.0, 1.0]);
        let rows: Vec<u32> = (0..4).collect();
        let leaf_of_row = vec![0u8; 4];
        let hist = build_histogram(&x, &gh, &rows, &leaf_of_row, 1, &[0, 1]).unwrap();

        assert_eq!(hist.shape(), (1, 2, 4)); // uniform stride 4
                                             // axis0: bin1 = rows 0,2 (g 1+3=4, count 2); bin2 = rows 1,3 (g 2+4=6, count 2).
        assert_eq!(cell(&hist, 0, 0, 1), (4.0, 2.0, 2));
        assert_eq!(cell(&hist, 0, 0, 2), (6.0, 2.0, 2));
        // axis0's high bin (3, beyond its 3-bin grid) and missing bin are zero.
        assert_eq!(cell(&hist, 0, 0, 0), (0.0, 0.0, 0));
        assert_eq!(cell(&hist, 0, 0, 3), (0.0, 0.0, 0));
        // axis1: bin1 = rows 0,1 (3, count 2); bin2 = row 2 (3, count 1); bin3 = row 3 (4, count 1).
        assert_eq!(cell(&hist, 0, 1, 1), (3.0, 2.0, 2));
        assert_eq!(cell(&hist, 0, 1, 2), (3.0, 1.0, 1));
        assert_eq!(cell(&hist, 0, 1, 3), (4.0, 1.0, 1));
    }

    #[test]
    fn build_partitions_rows_by_leaf() {
        let cols = vec![vec![1u8, 1, 1, 1]]; // all in bin 1
        let x = matrix(cols, &[3]);
        let gh = gradhess(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0, 1.0, 1.0]);
        let rows: Vec<u32> = (0..4).collect();
        let leaf_of_row = vec![0u8, 0, 1, 1]; // rows 0,1 -> leaf 0; rows 2,3 -> leaf 1
        let hist = build_histogram(&x, &gh, &rows, &leaf_of_row, 2, &[0]).unwrap();
        assert_eq!(cell(&hist, 0, 0, 1), (3.0, 2.0, 2)); // 1+2
        assert_eq!(cell(&hist, 1, 0, 1), (7.0, 2.0, 2)); // 3+4
    }

    fn fixture() -> (BinnedMatrix, GradHess, Vec<u32>, Vec<u8>) {
        let n = 200usize;
        let c0: Vec<u8> = (0..n).map(|i| ((i % 5) + 1) as u8).collect();
        let c1: Vec<u8> = (0..n).map(|i| ((i % 9) + 1) as u8).collect();
        let c2: Vec<u8> = (0..n).map(|i| ((i * 7 % 11) + 1) as u8).collect();
        let x = matrix(vec![c0, c1, c2], &[6, 10, 12]);
        let g: Vec<f32> = (0..n).map(|i| (i as f32 % 13.0) - 6.0).collect();
        let h: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 % 3.0)).collect();
        let gh = gradhess(&g, &h);
        let rows: Vec<u32> = (0..n as u32).collect();
        let leaf_of_row: Vec<u8> = (0..n).map(|i| (i % 4) as u8).collect();
        (x, gh, rows, leaf_of_row)
    }

    #[test]
    fn histogram_is_byte_identical_across_thread_counts() {
        let (x, gh, rows, leaf_of_row) = fixture();
        let axes = [0u32, 1, 2];
        let run = |nt: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| build_histogram(&x, &gh, &rows, &leaf_of_row, 4, &axes).unwrap())
        };
        let h1 = run(1);
        let h2 = run(2);
        let h8 = run(8);
        // Byte-identical f64 cells across thread counts (fixed-order fold).
        let bits = |h: &Hist| -> (Vec<u64>, Vec<u64>, Vec<u32>) {
            (
                h.g.iter().map(|v| v.to_bits()).collect(),
                h.h.iter().map(|v| v.to_bits()).collect(),
                h.count.clone(),
            )
        };
        assert_eq!(bits(&h1), bits(&h2));
        assert_eq!(bits(&h1), bits(&h8));
    }

    #[test]
    fn subtraction_matches_direct_right_for_well_conditioned_data() {
        // NON-INTEGER, well-conditioned f32 gradients (representative of real
        // SquaredError gradients), so the test genuinely exercises the f64 fold (not
        // the degenerate integer-exact case). count is integer-exact; g/h match a
        // direct-right build within a NAMED tolerance. (For well-conditioned data the
        // f64 fold is bit-exact to ~1e-11 even at hundreds of thousands of rows; the
        // tolerance guards the catastrophic-cancellation case — see the next test —
        // which is why the integer-associative M5-QHIST path is the v1.5 alternative.)
        const SUBTRACTION_TOL: f64 = 1e-9;
        let n = 600usize;
        let c0: Vec<u8> = (0..n).map(|i| u8::try_from(i % 5 + 1).unwrap()).collect();
        let c1: Vec<u8> = (0..n).map(|i| u8::try_from(i % 9 + 1).unwrap()).collect();
        let x = matrix(vec![c0, c1], &[6, 10]);
        let g: Vec<f32> = (0..n).map(|i| 0.1 * (i as f32) + 0.0333).collect();
        let h: Vec<f32> = (0..n).map(|i| 0.5 + 0.137 * (i % 7) as f32).collect();
        let gh = gradhess(&g, &h);
        let all: Vec<u32> = (0..n as u32).collect();
        let one_leaf = vec![0u8; n];
        let axes = [0u32, 1];

        let parent = build_histogram(&x, &gh, &all, &one_leaf, 1, &axes).unwrap();
        let k = n / 3;
        let left = build_histogram(&x, &gh, &all[..k], &one_leaf, 1, &axes).unwrap();
        let right_direct = build_histogram(&x, &gh, &all[k..], &one_leaf, 1, &axes).unwrap();
        let right_sub = subtract(&parent, &left).unwrap();

        assert_eq!(right_sub.shape(), right_direct.shape());
        assert_eq!(right_sub.count, right_direct.count); // integer-exact
        for (a, b) in right_sub.g.iter().zip(&right_direct.g) {
            assert!((a - b).abs() <= SUBTRACTION_TOL, "g drift {a} vs {b}");
        }
        for (a, b) in right_sub.h.iter().zip(&right_direct.h) {
            assert!((a - b).abs() <= SUBTRACTION_TOL, "h drift {a} vs {b}");
        }
    }

    #[test]
    fn subtraction_drifts_under_catastrophic_cancellation() {
        // One huge gradient + ten 1.0s in the same cell: the huge value swamps the
        // smalls in the f64 running sum, so `parent − left` loses them while a direct
        // build keeps them. count stays integer-exact, but g drifts FAR past any
        // reasonable tolerance — concrete proof the subtraction trick is NOT
        // unconditionally bit-exact (R3), justifying the named tolerance above and the
        // integer-associative M5-QHIST path as the robust v1.5 alternative.
        let n = 11usize;
        let x = matrix(vec![vec![1u8; n]], &[3]); // all in bin 1
        let mut g = vec![1.0_f32; n];
        g[0] = 1e16; // catastrophic magnitude spread
        let gh = gradhess(&g, &vec![1.0_f32; n]);
        let all: Vec<u32> = (0..n as u32).collect();
        let one_leaf = vec![0u8; n];

        let parent = build_histogram(&x, &gh, &all, &one_leaf, 1, &[0]).unwrap();
        let k = 6;
        let left = build_histogram(&x, &gh, &all[..k], &one_leaf, 1, &[0]).unwrap();
        let right_direct = build_histogram(&x, &gh, &all[k..], &one_leaf, 1, &[0]).unwrap();
        let right_sub = subtract(&parent, &left).unwrap();

        let o = right_direct.offset(0, 0, 1).unwrap();
        assert_eq!(right_sub.count, right_direct.count); // count still integer-exact
        assert!(
            (right_direct.g[o] - 5.0).abs() < 1e-6,
            "direct keeps the five 1.0s"
        );
        assert!(
            right_sub.g[o].abs() < 1e-6,
            "subtraction lost them to cancellation"
        );
        assert!(
            (right_sub.g[o] - right_direct.g[o]).abs() > 1.0,
            "drift {} must exceed the well-conditioned tolerance",
            (right_sub.g[o] - right_direct.g[o]).abs()
        );
    }

    #[test]
    fn assembly_layout_distinguishes_leaf_from_axis() {
        // n_leaves=2 AND n_axes=2: a leaf/axis transposition in the flat offset would
        // swap the off-diagonal cells (leaf1,axis0) <-> (leaf0,axis1). Distinct values
        // at those cells catch it (the build's `dst` and `Hist::offset` must agree).
        let c0 = vec![1u8, 1, 1, 1]; // axis0: all bin 1
        let c1 = vec![2u8, 2, 2, 2]; // axis1: all bin 2
        let x = matrix(vec![c0, c1], &[3, 3]);
        let gh = gradhess(&[10.0, 20.0, 100.0, 200.0], &[1.0, 1.0, 1.0, 1.0]);
        let rows: Vec<u32> = (0..4).collect();
        let leaf_of_row = vec![0u8, 0, 1, 1]; // leaf0: rows 0,1 (Σg 30); leaf1: rows 2,3 (Σg 300)
        let hist = build_histogram(&x, &gh, &rows, &leaf_of_row, 2, &[0, 1]).unwrap();

        assert_eq!(cell(&hist, 0, 0, 1), (30.0, 2.0, 2)); // leaf0, axis0
        assert_eq!(cell(&hist, 1, 0, 1), (300.0, 2.0, 2)); // leaf1, axis0  (off-diagonal)
        assert_eq!(cell(&hist, 0, 1, 2), (30.0, 2.0, 2)); // leaf0, axis1  (off-diagonal)
        assert_eq!(cell(&hist, 1, 1, 2), (300.0, 2.0, 2)); // leaf1, axis1
                                                           // The cells a transposition would read instead are empty in the correct layout.
        assert_eq!(cell(&hist, 1, 0, 2), (0.0, 0.0, 0));
        assert_eq!(cell(&hist, 0, 1, 1), (0.0, 0.0, 0));
    }

    #[test]
    fn subtraction_shape_mismatch_errors() {
        let a = Hist::zeros(1, 2, 3);
        let b = Hist::zeros(2, 2, 3);
        assert!(matches!(
            subtract(&a, &b),
            Err(PbError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn out_of_range_inputs_return_internal_not_panic() {
        let x = matrix(vec![vec![1u8, 1]], &[3]);
        let gh = gradhess(&[1.0, 2.0], &[1.0, 1.0]);
        let rows = [0u32, 1];
        // Axis with no grid/column.
        assert!(matches!(
            build_histogram(&x, &gh, &rows, &[0, 0], 1, &[5]),
            Err(PbError::Internal { .. })
        ));
        // A leaf_of_row entry >= n_leaves.
        assert!(matches!(
            build_histogram(&x, &gh, &rows, &[0, 3], 1, &[0]),
            Err(PbError::Internal { .. })
        ));
        // A bin id >= max_bins (malformed column value 5 against a 3-bin grid).
        let bad = matrix(vec![vec![5u8, 1]], &[3]);
        assert!(matches!(
            build_histogram(&bad, &gh, &rows, &[0, 0], 1, &[0]),
            Err(PbError::Internal { .. })
        ));
        // A row id outside the column.
        assert!(matches!(
            build_histogram(&x, &gh, &[0, 9], &[0, 0], 1, &[0]),
            Err(PbError::Internal { .. })
        ));
    }

    #[test]
    fn subtract_count_underflow_is_internal() {
        let parent = Hist::zeros(1, 1, 2);
        let mut left = Hist::zeros(1, 1, 2);
        left.count[0] = 5; // left ⊄ parent ⇒ count underflow
        assert!(matches!(
            subtract(&parent, &left),
            Err(PbError::Internal { .. })
        ));
    }

    #[test]
    fn counts_are_bounded_by_rows_and_total_matches() {
        // Every cell's count <= n_rows, and the total count over a single axis equals
        // the number of accumulated rows (the u32 count can never exceed n_rows).
        let (x, gh, rows, leaf_of_row) = fixture();
        let hist = build_histogram(&x, &gh, &rows, &leaf_of_row, 4, &[0]).unwrap();
        let total: u64 = hist.count.iter().map(|&c| u64::from(c)).sum();
        assert_eq!(total, rows.len() as u64);
        assert!(hist
            .count
            .iter()
            .all(|&c| u64::from(c) <= u64::from(x.n_rows)));
    }
}
