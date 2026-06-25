//! The full-precision histogram engine (spec §06.3, milestone M1.3).
//!
//! v1 accumulates full-precision [`GradHess`] into `f64` [`Hist`] cells. The build is
//! **feature-parallel** and, for large row sets, **row-chunk parallel** within each
//! axis: chunks accumulate sequentially, then reduce in deterministic chunk order into
//! a disjoint region of the `[leaf][axis][bin]` tensor. So the result is a FIXED-ORDER
//! fold — byte-identical regardless of thread count (the §1 determinism `[GATE]`) —
//! while narrow/small data keeps the lower-overhead sequential axis path.
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

use crate::backend::{pb_seed, Stage};
use crate::data::BinnedMatrix;
use crate::engine::{GradScale, Hist, QuantGradHess};
use crate::error::PbError;
use crate::loss::GradHess;
use rayon::prelude::*;

const ROW_PAR_MIN_ROWS: usize = 32_768;
const ROW_PAR_CHUNK_ROWS: usize = 8_192;

/// Deterministic quantization re-seed coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QuantizeContext {
    /// Base model seed.
    pub seed: u64,
    /// Boosting round.
    pub round: u32,
}

fn oob(what: &'static str) -> impl Fn() -> PbError {
    move || PbError::Internal { what: what.into() }
}

fn offset2(
    first: usize,
    second: usize,
    stride: usize,
    what: &'static str,
) -> Result<usize, PbError> {
    first
        .checked_mul(stride)
        .and_then(|base| base.checked_add(second))
        .ok_or_else(oob(what))
}

fn offset3(
    first: usize,
    second: usize,
    third: usize,
    second_stride: usize,
    third_stride: usize,
    what: &'static str,
) -> Result<usize, PbError> {
    first
        .checked_mul(second_stride)
        .and_then(|base| base.checked_add(second))
        .and_then(|base| base.checked_mul(third_stride))
        .and_then(|base| base.checked_add(third))
        .ok_or_else(oob(what))
}

/// Build the per-level `[leaf][axis][bin]` histogram from full-precision `gh`
/// (spec §06.3). `rows` are the row ids in scope (the subsample; in fixed order),
/// `leaf_of_row[r]` is the leaf id (`0..n_leaves`) of row `r`, and `axes` are the
/// (sampled) feature columns to build. The bin stride is uniform — the max grid
/// `n_bins` over `axes` — so shorter-grid axes leave their high bins zeroed.
///
/// Feature-parallel + fixed-order row-chunk reduction ⇒ thread-count independent.
///
/// # Errors
/// [`PbError::Internal`] if any `axis`, row id, leaf id, or bin id is out of range
/// (the engine builds these so it is a bug, surfaced as a typed error not a panic).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_histogram(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axes: &[u32],
    weight: &[f32],
    unit_weight: bool,
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
        .map(|&a| {
            accumulate_axis(
                x,
                gh,
                rows,
                leaf_of_row,
                n_leaves,
                a,
                max_bins,
                weight,
                unit_weight,
            )
        })
        .collect();
    let subs = subs?;

    // Assemble the disjoint per-axis sub-histograms into the flat tensor, in axis
    // order — deterministic regardless of how rayon scheduled the per-axis tasks.
    let mut hist = Hist::try_zeros(n_leaves, n_axes, max_bins)?;
    for (axis_pos, sub) in subs.iter().enumerate() {
        for leaf in 0..n_leaves {
            for bin in 0..max_bins {
                let src = offset2(leaf, bin, max_bins, "axis histogram offset overflow")?;
                let dst = offset3(
                    leaf,
                    axis_pos,
                    bin,
                    n_axes,
                    max_bins,
                    "histogram assembly offset overflow",
                )?;
                *hist.g.get_mut(dst).ok_or_else(oob("assemble g"))? =
                    *sub.g.get(src).ok_or_else(oob("sub g"))?;
                *hist.h.get_mut(dst).ok_or_else(oob("assemble h"))? =
                    *sub.h.get(src).ok_or_else(oob("sub h"))?;
                *hist.wsum.get_mut(dst).ok_or_else(oob("assemble wsum"))? =
                    *sub.wsum.get(src).ok_or_else(oob("sub wsum"))?;
                *hist.count.get_mut(dst).ok_or_else(oob("assemble count"))? =
                    *sub.count.get(src).ok_or_else(oob("sub count"))?;
            }
        }
    }
    Ok(hist)
}

/// Quantize full-precision gradients/hessians for the integer histogram path
/// (§06/§11 M5-QHIST).
///
/// Scale factors map the maximum absolute value to the i32 range. FLAG (spec
/// reconciliation): M5-QHIST asks for stochastic rounding AND a `< 0.5 / scale` error
/// bound; adjacent stochastic rounding can miss by nearly one step, so this path uses
/// nearest rounding with deterministic randomized tie-breaking keyed by the frozen
/// `Stage::Quantize` stream. The result is deterministic by row position, independent
/// of thread count, and bounded by half a quantization step.
///
/// # Errors
/// [`PbError::ShapeMismatch`] if `g`/`h` lengths differ; [`PbError::InvalidInput`] if
/// any value is non-finite or if a row index exceeds the frozen re-seed coordinate.
pub fn quantize_grad_hess(gh: &GradHess, seed: u64, round: u32) -> Result<QuantGradHess, PbError> {
    if gh.g.len() != gh.h.len() {
        return Err(PbError::ShapeMismatch {
            what: format!("GradHess g len {} != h len {}", gh.g.len(), gh.h.len()),
        });
    }
    let mut max_g = 0.0_f64;
    let mut max_h = 0.0_f64;
    for (i, (&g, &h)) in gh.g.iter().zip(&gh.h).enumerate() {
        if !g.is_finite() || !h.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!("GradHess row {i} must be finite before quantization"),
            });
        }
        max_g = max_g.max(f64::from(g).abs());
        max_h = max_h.max(f64::from(h).abs());
    }
    let g_scale = scale_for(max_g)?;
    let h_scale = scale_for(max_h)?;
    let mut g_q = Vec::with_capacity(gh.g.len());
    let mut h_q = Vec::with_capacity(gh.h.len());
    for (i, (&g, &h)) in gh.g.iter().zip(&gh.h).enumerate() {
        let base = u32::try_from(i).map_err(|_| PbError::InvalidInput {
            what: "quantization supports at most u32::MAX rows".into(),
        })?;
        let block_g = base.checked_mul(2).ok_or_else(|| PbError::InvalidInput {
            what: "quantization row coordinate overflowed".into(),
        })?;
        let block_h = block_g
            .checked_add(1)
            .ok_or_else(|| PbError::InvalidInput {
                what: "quantization row coordinate overflowed".into(),
            })?;
        g_q.push(stochastic_round(
            f64::from(g) * f64::from(g_scale),
            seed,
            round,
            block_g,
        )?);
        h_q.push(stochastic_round(
            f64::from(h) * f64::from(h_scale),
            seed,
            round,
            block_h,
        )?);
    }
    Ok(QuantGradHess {
        g_q,
        h_q,
        scale: GradScale { g_scale, h_scale },
    })
}

fn scale_for(max_abs: f64) -> Result<f32, PbError> {
    let scale = if max_abs > 0.0 {
        (f64::from(i32::MAX) * 0.5) / max_abs
    } else {
        1.0
    };
    let out = scale as f32;
    if out.is_finite() && out > 0.0 {
        Ok(out)
    } else {
        Err(PbError::InvalidInput {
            what: format!("quantization scale is not finite: {scale}"),
        })
    }
}

fn stochastic_round(value: f64, seed: u64, round: u32, block: u32) -> Result<i32, PbError> {
    let lo = value.floor();
    let frac = value - lo;
    let bits = pb_seed(seed, round, Stage::Quantize as u32, block);
    let unit = ((bits >> 11) as f64 + 1.0) / ((1_u64 << 53) as f64 + 1.0);
    let rounded = if frac > 0.5 {
        lo + 1.0
    } else if frac < 0.5 {
        lo
    } else if unit < 0.5 {
        lo + 1.0
    } else {
        lo
    };
    if rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
        return Err(PbError::InvalidInput {
            what: format!("quantized value {rounded} escaped i32"),
        });
    }
    Ok(rounded as i32)
}

/// Build a histogram via quantized integer accumulation, then dequantize into the
/// canonical [`Hist`] view for the existing split scanner.
///
/// # Errors
/// Propagates quantization and shape/index errors.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_quantized_histogram(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axes: &[u32],
    ctx: QuantizeContext,
    weight: &[f32],
) -> Result<Hist, PbError> {
    let qgh = quantize_grad_hess(gh, ctx.seed, ctx.round)?;
    let mut max_bins = 0usize;
    for &a in axes {
        let grid = x
            .grids
            .get(a as usize)
            .ok_or_else(oob("axis has no grid"))?;
        max_bins = max_bins.max(usize::from(grid.n_bins));
    }
    let n_axes = axes.len();
    let subs: Result<Vec<AxisQHist>, PbError> = axes
        .par_iter()
        .map(|&a| {
            accumulate_axis_quantized(x, &qgh, rows, leaf_of_row, n_leaves, a, max_bins, weight)
        })
        .collect();
    let subs = subs?;
    let mut hist = Hist::try_zeros(n_leaves, n_axes, max_bins)?;
    let inv_g = 1.0_f64 / f64::from(qgh.scale.g_scale);
    let inv_h = 1.0_f64 / f64::from(qgh.scale.h_scale);
    for (axis_pos, sub) in subs.iter().enumerate() {
        for leaf in 0..n_leaves {
            for bin in 0..max_bins {
                let src = offset2(leaf, bin, max_bins, "quant axis histogram offset overflow")?;
                let dst = offset3(
                    leaf,
                    axis_pos,
                    bin,
                    n_axes,
                    max_bins,
                    "quant histogram assembly offset overflow",
                )?;
                *hist.g.get_mut(dst).ok_or_else(oob("assemble quant g"))? =
                    *sub.g.get(src).ok_or_else(oob("sub quant g"))? as f64 * inv_g;
                *hist.h.get_mut(dst).ok_or_else(oob("assemble quant h"))? =
                    *sub.h.get(src).ok_or_else(oob("sub quant h"))? as f64 * inv_h;
                *hist
                    .wsum
                    .get_mut(dst)
                    .ok_or_else(oob("assemble quant wsum"))? =
                    *sub.wsum.get(src).ok_or_else(oob("sub quant wsum"))?;
                *hist
                    .count
                    .get_mut(dst)
                    .ok_or_else(oob("assemble quant count"))? =
                    *sub.count.get(src).ok_or_else(oob("sub quant count"))?;
            }
        }
    }
    Ok(hist)
}

/// One axis's `[leaf][bin]` sub-histogram (stride `max_bins`).
struct AxisHist {
    g: Vec<f64>,
    h: Vec<f64>,
    wsum: Vec<f64>,
    count: Vec<u32>,
}

struct AxisQHist {
    g: Vec<i64>,
    h: Vec<i64>,
    /// Sample-weight sums stay full-precision `f64` — weights are not gradients, so they
    /// are never quantized (the credibility floor reads exact Σw on both hist paths).
    wsum: Vec<f64>,
    count: Vec<u32>,
}

impl AxisQHist {
    fn try_zeros(n_leaves: usize, max_bins: usize) -> Result<Self, PbError> {
        let size = Hist::checked_cell_count(n_leaves, 1, max_bins)?;
        Ok(Self {
            g: Hist::try_zeroed_vec(size, "axis quant histogram g")?,
            h: Hist::try_zeroed_vec(size, "axis quant histogram h")?,
            wsum: Hist::try_zeroed_vec(size, "axis quant histogram wsum")?,
            count: Hist::try_zeroed_vec(size, "axis quant histogram count")?,
        })
    }
}

impl AxisHist {
    fn try_zeros(n_leaves: usize, max_bins: usize) -> Result<Self, PbError> {
        let size = Hist::checked_cell_count(n_leaves, 1, max_bins)?;
        Ok(Self {
            g: Hist::try_zeroed_vec(size, "axis histogram g")?,
            h: Hist::try_zeroed_vec(size, "axis histogram h")?,
            wsum: Hist::try_zeroed_vec(size, "axis histogram wsum")?,
            count: Hist::try_zeroed_vec(size, "axis histogram count")?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn accumulate_axis(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axis: u32,
    max_bins: usize,
    weight: &[f32],
    unit_weight: bool,
) -> Result<AxisHist, PbError> {
    if rows.len() < ROW_PAR_MIN_ROWS {
        return accumulate_axis_sequential(
            x,
            gh,
            rows,
            leaf_of_row,
            n_leaves,
            axis,
            max_bins,
            weight,
            unit_weight,
        );
    }
    let chunks: Result<Vec<AxisHist>, PbError> = rows
        .par_chunks(ROW_PAR_CHUNK_ROWS)
        .map(|chunk| {
            accumulate_axis_sequential(
                x,
                gh,
                chunk,
                leaf_of_row,
                n_leaves,
                axis,
                max_bins,
                weight,
                unit_weight,
            )
        })
        .collect();
    let chunks = chunks?;
    let mut out = AxisHist::try_zeros(n_leaves, max_bins)?;
    for chunk in &chunks {
        add_axis_hist(&mut out, chunk)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn accumulate_axis_sequential(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axis: u32,
    max_bins: usize,
    weight: &[f32],
    unit_weight: bool,
) -> Result<AxisHist, PbError> {
    let col = x
        .data
        .get(axis as usize)
        .ok_or_else(oob("axis has no column"))?;
    let mut out = AxisHist::try_zeros(n_leaves, max_bins)?;
    // Sequential over rows in their given (fixed) order ⇒ deterministic f64 fold. When
    // `unit_weight`, the per-row weight read + `Σw` add are skipped (the loop-invariant branch
    // is hoisted by LLVM into two loop variants), and `wsum` is set from `count` afterwards —
    // bit-exact, since for unit weights `Σ 1.0` over a bin equals its (integer, <2^53) count.
    for &r in rows {
        let ru = r as usize;
        let bin = usize::from(*col.get(ru).ok_or_else(oob("row out of column"))?);
        let leaf = usize::from(*leaf_of_row.get(ru).ok_or_else(oob("row out of leaf map"))?);
        if leaf >= n_leaves || bin >= max_bins {
            return Err(PbError::Internal {
                what: "leaf or bin id out of histogram range".into(),
            });
        }
        let idx = offset2(leaf, bin, max_bins, "axis histogram row offset overflow")?;
        *out.g.get_mut(idx).ok_or_else(oob("g cell"))? +=
            f64::from(*gh.g.get(ru).ok_or_else(oob("gh.g"))?);
        *out.h.get_mut(idx).ok_or_else(oob("h cell"))? +=
            f64::from(*gh.h.get(ru).ok_or_else(oob("gh.h"))?);
        if !unit_weight {
            *out.wsum.get_mut(idx).ok_or_else(oob("wsum cell"))? +=
                f64::from(*weight.get(ru).ok_or_else(oob("weight"))?);
        }
        let c = out.count.get_mut(idx).ok_or_else(oob("count cell"))?;
        // count <= rows.len() <= n_rows <= u32::MAX, so this never actually overflows;
        // checked_add keeps it panic-free under overflow-checks regardless.
        *c = c.checked_add(1).ok_or_else(oob("bin count overflow"))?;
    }
    if unit_weight {
        // wsum == count for unit weights (Σ 1.0 == count, exact in f64). Per-chunk this gives
        // each chunk's count; `add_axis_hist` then sums them, matching the per-row Σw exactly.
        for (w, &c) in out.wsum.iter_mut().zip(&out.count) {
            *w = f64::from(c);
        }
    }
    Ok(out)
}

fn add_axis_hist(dst: &mut AxisHist, src: &AxisHist) -> Result<(), PbError> {
    if dst.g.len() != src.g.len()
        || dst.h.len() != src.h.len()
        || dst.wsum.len() != src.wsum.len()
        || dst.count.len() != src.count.len()
    {
        return Err(PbError::Internal {
            what: "axis histogram chunk shape mismatch".into(),
        });
    }
    for (d, s) in dst.g.iter_mut().zip(&src.g) {
        *d += *s;
    }
    for (d, s) in dst.h.iter_mut().zip(&src.h) {
        *d += *s;
    }
    for (d, s) in dst.wsum.iter_mut().zip(&src.wsum) {
        *d += *s;
    }
    for (d, s) in dst.count.iter_mut().zip(&src.count) {
        *d = d
            .checked_add(*s)
            .ok_or_else(oob("axis histogram chunk count overflow"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn accumulate_axis_quantized(
    x: &BinnedMatrix,
    qgh: &QuantGradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axis: u32,
    max_bins: usize,
    weight: &[f32],
) -> Result<AxisQHist, PbError> {
    if rows.len() < ROW_PAR_MIN_ROWS {
        return accumulate_axis_quantized_sequential(
            x,
            qgh,
            rows,
            leaf_of_row,
            n_leaves,
            axis,
            max_bins,
            weight,
        );
    }
    let chunks: Result<Vec<AxisQHist>, PbError> = rows
        .par_chunks(ROW_PAR_CHUNK_ROWS)
        .map(|chunk| {
            accumulate_axis_quantized_sequential(
                x,
                qgh,
                chunk,
                leaf_of_row,
                n_leaves,
                axis,
                max_bins,
                weight,
            )
        })
        .collect();
    let chunks = chunks?;
    let mut out = AxisQHist::try_zeros(n_leaves, max_bins)?;
    for chunk in &chunks {
        add_axis_qhist(&mut out, chunk)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn accumulate_axis_quantized_sequential(
    x: &BinnedMatrix,
    qgh: &QuantGradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axis: u32,
    max_bins: usize,
    weight: &[f32],
) -> Result<AxisQHist, PbError> {
    let col = x
        .data
        .get(axis as usize)
        .ok_or_else(oob("axis has no column"))?;
    let mut out = AxisQHist::try_zeros(n_leaves, max_bins)?;
    for &r in rows {
        let ru = r as usize;
        let bin = usize::from(*col.get(ru).ok_or_else(oob("quant row out of column"))?);
        let leaf = usize::from(
            *leaf_of_row
                .get(ru)
                .ok_or_else(oob("quant row out of leaf map"))?,
        );
        if leaf >= n_leaves || bin >= max_bins {
            return Err(PbError::Internal {
                what: "leaf or bin id out of quant histogram range".into(),
            });
        }
        let idx = offset2(
            leaf,
            bin,
            max_bins,
            "quant axis histogram row offset overflow",
        )?;
        *out.g.get_mut(idx).ok_or_else(oob("quant g cell"))? +=
            i64::from(*qgh.g_q.get(ru).ok_or_else(oob("qgh.g"))?);
        *out.h.get_mut(idx).ok_or_else(oob("quant h cell"))? +=
            i64::from(*qgh.h_q.get(ru).ok_or_else(oob("qgh.h"))?);
        *out.wsum.get_mut(idx).ok_or_else(oob("quant wsum cell"))? +=
            f64::from(*weight.get(ru).ok_or_else(oob("quant weight"))?);
        let c = out.count.get_mut(idx).ok_or_else(oob("quant count cell"))?;
        *c = c
            .checked_add(1)
            .ok_or_else(oob("quant bin count overflow"))?;
    }
    Ok(out)
}

fn add_axis_qhist(dst: &mut AxisQHist, src: &AxisQHist) -> Result<(), PbError> {
    if dst.g.len() != src.g.len()
        || dst.h.len() != src.h.len()
        || dst.wsum.len() != src.wsum.len()
        || dst.count.len() != src.count.len()
    {
        return Err(PbError::Internal {
            what: "axis quant histogram chunk shape mismatch".into(),
        });
    }
    for (d, s) in dst.g.iter_mut().zip(&src.g) {
        *d = d
            .checked_add(*s)
            .ok_or_else(oob("axis quant histogram chunk g overflow"))?;
    }
    for (d, s) in dst.h.iter_mut().zip(&src.h) {
        *d = d
            .checked_add(*s)
            .ok_or_else(oob("axis quant histogram chunk h overflow"))?;
    }
    for (d, s) in dst.wsum.iter_mut().zip(&src.wsum) {
        *d += *s;
    }
    for (d, s) in dst.count.iter_mut().zip(&src.count) {
        *d = d
            .checked_add(*s)
            .ok_or_else(oob("axis quant histogram chunk count overflow"))?;
    }
    Ok(())
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
    let mut out = Hist::try_zeros(parent.n_leaves, parent.n_axes, parent.n_bins)?;
    for (o, (p, l)) in out.g.iter_mut().zip(parent.g.iter().zip(&left.g)) {
        *o = p - l;
    }
    for (o, (p, l)) in out.h.iter_mut().zip(parent.h.iter().zip(&left.h)) {
        *o = p - l;
    }
    for (o, (p, l)) in out.wsum.iter_mut().zip(parent.wsum.iter().zip(&left.wsum)) {
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

/// Fill each LARGER sibling-child leaf of a level-`L` histogram by subtracting the already-built
/// SMALLER child from its level-`(L-1)` parent leaf — the histogram-subtraction trick wired to the
/// oblivious grower. `child` has `2·parent.n_leaves` leaves (each parent leaf split into a smaller +
/// larger child); `pairing[i] = (parent_leaf, smaller_child_leaf, larger_child_leaf)`; `axis_map[a2]`
/// is the column position in `parent` of `child`'s axis `a2` (the parent's axis set is a superset, so
/// every child axis maps). On entry the smaller-child leaves are populated and the larger are zero;
/// on return `child[larger] = parent[p] − child[smaller]` cellwise (g/h/wsum plain f64, count
/// `checked_sub`). Because the SMALLER child is the subtrahend, `larger` is the bigger remainder, so
/// the f64 subtraction is NOT catastrophic cancellation (drift stays ~1e-11 for well-conditioned
/// gradients); `count` is integer-exact and, under unit weights, `wsum == count` stays exact. Used
/// only on the FullF64 path; bin range `0..child.n_bins ≤ parent.n_bins` (a dropped axis may have
/// held the max stride), and every access goes through [`Hist::offset`] so strides are honored.
pub(crate) fn subtract_sibling_into(
    child: &mut Hist,
    parent: &Hist,
    pairing: &[(usize, usize, usize)],
    axis_map: &[usize],
) -> Result<(), PbError> {
    if child.n_leaves != 2 * parent.n_leaves {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "subtract_sibling_into: child n_leaves {} != 2·parent {}",
                child.n_leaves, parent.n_leaves
            ),
        });
    }
    if child.n_axes != axis_map.len() {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "subtract_sibling_into: child n_axes {} != axis_map len {}",
                child.n_axes,
                axis_map.len()
            ),
        });
    }
    if child.n_bins > parent.n_bins {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "subtract_sibling_into: child n_bins {} > parent n_bins {}",
                child.n_bins, parent.n_bins
            ),
        });
    }
    for &(p, sm, lg) in pairing {
        for (a2, &a1) in axis_map.iter().enumerate() {
            for b in 0..child.n_bins {
                let po = parent
                    .offset(p, a1, b)
                    .ok_or_else(oob("subtract_sibling parent offset"))?;
                let so = child
                    .offset(sm, a2, b)
                    .ok_or_else(oob("subtract_sibling smaller offset"))?;
                let lo = child
                    .offset(lg, a2, b)
                    .ok_or_else(oob("subtract_sibling larger offset"))?;
                // Read parent (p) and smaller (sm) first, then write larger (lg) — lg differs
                // from sm so there is no aliasing; copies keep the borrows sequential.
                let pg = *parent
                    .g
                    .get(po)
                    .ok_or_else(oob("subtract_sibling parent g"))?;
                let ph = *parent
                    .h
                    .get(po)
                    .ok_or_else(oob("subtract_sibling parent h"))?;
                let pw = *parent
                    .wsum
                    .get(po)
                    .ok_or_else(oob("subtract_sibling parent wsum"))?;
                let pc = *parent
                    .count
                    .get(po)
                    .ok_or_else(oob("subtract_sibling parent count"))?;
                let sg = *child
                    .g
                    .get(so)
                    .ok_or_else(oob("subtract_sibling small g"))?;
                let sh = *child
                    .h
                    .get(so)
                    .ok_or_else(oob("subtract_sibling small h"))?;
                let sw = *child
                    .wsum
                    .get(so)
                    .ok_or_else(oob("subtract_sibling small wsum"))?;
                let sc = *child
                    .count
                    .get(so)
                    .ok_or_else(oob("subtract_sibling small count"))?;
                *child
                    .g
                    .get_mut(lo)
                    .ok_or_else(oob("subtract_sibling g cell"))? = pg - sg;
                *child
                    .h
                    .get_mut(lo)
                    .ok_or_else(oob("subtract_sibling h cell"))? = ph - sh;
                *child
                    .wsum
                    .get_mut(lo)
                    .ok_or_else(oob("subtract_sibling wsum cell"))? = pw - sw;
                *child
                    .count
                    .get_mut(lo)
                    .ok_or_else(oob("subtract_sibling count cell"))? = pc
                    .checked_sub(sc)
                    .ok_or_else(oob("subtract_sibling count underflow"))?;
            }
        }
    }
    Ok(())
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

    /// Most existing tests don't exercise weighted Σw, so they build with unit weights
    /// (then `wsum == count` as `f64`). The dedicated `wsum_*` tests below pass real
    /// weights to `build_histogram`/`build_quantized_histogram` directly.
    fn build_hist(
        x: &BinnedMatrix,
        gh: &GradHess,
        rows: &[u32],
        leaf_of_row: &[u8],
        n_leaves: usize,
        axes: &[u32],
    ) -> Result<Hist, PbError> {
        build_histogram(
            x,
            gh,
            rows,
            leaf_of_row,
            n_leaves,
            axes,
            &vec![1.0_f32; gh.g.len()],
            false,
        )
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
        let hist = build_hist(&x, &gh, &rows, &leaf_of_row, 1, &[0, 1]).unwrap();

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
        let hist = build_hist(&x, &gh, &rows, &leaf_of_row, 2, &[0]).unwrap();
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
            pool.install(|| build_hist(&x, &gh, &rows, &leaf_of_row, 4, &axes).unwrap())
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
    fn row_parallel_histogram_is_byte_identical_across_thread_counts() {
        let n = ROW_PAR_MIN_ROWS + 1024;
        let c0: Vec<u8> = (0..n).map(|i| ((i % 13) + 1) as u8).collect();
        let c1: Vec<u8> = (0..n).map(|i| ((i * 7 % 17) + 1) as u8).collect();
        let x = matrix(vec![c0, c1], &[14, 18]);
        let g: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 - 9.0) * 0.125).collect();
        let h: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32).collect();
        let gh = gradhess(&g, &h);
        let rows: Vec<u32> = (0..n as u32).collect();
        let leaf_of_row: Vec<u8> = (0..n).map(|i| (i % 4) as u8).collect();
        let axes = [0u32, 1];
        let run = |nt: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| build_hist(&x, &gh, &rows, &leaf_of_row, 4, &axes).unwrap())
        };
        let bits = |h: &Hist| -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u32>) {
            (
                h.g.iter().map(|v| v.to_bits()).collect(),
                h.h.iter().map(|v| v.to_bits()).collect(),
                h.wsum.iter().map(|v| v.to_bits()).collect(),
                h.count.clone(),
            )
        };
        let h1 = run(1);
        assert_eq!(bits(&h1), bits(&run(2)));
        assert_eq!(bits(&h1), bits(&run(8)));
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

        let parent = build_hist(&x, &gh, &all, &one_leaf, 1, &axes).unwrap();
        let k = n / 3;
        let left = build_hist(&x, &gh, &all[..k], &one_leaf, 1, &axes).unwrap();
        let right_direct = build_hist(&x, &gh, &all[k..], &one_leaf, 1, &axes).unwrap();
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

        let parent = build_hist(&x, &gh, &all, &one_leaf, 1, &[0]).unwrap();
        let k = 6;
        let left = build_hist(&x, &gh, &all[..k], &one_leaf, 1, &[0]).unwrap();
        let right_direct = build_hist(&x, &gh, &all[k..], &one_leaf, 1, &[0]).unwrap();
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
        let hist = build_hist(&x, &gh, &rows, &leaf_of_row, 2, &[0, 1]).unwrap();

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
        let a = Hist::try_zeros(1, 2, 3).unwrap();
        let b = Hist::try_zeros(2, 2, 3).unwrap();
        assert!(matches!(
            subtract(&a, &b),
            Err(PbError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn oversized_hist_shapes_error_without_overflowing() {
        assert!(matches!(
            Hist::try_zeros(usize::MAX, 2, 2),
            Err(PbError::Internal { .. })
        ));

        let x = matrix(vec![Vec::new()], &[3]);
        let gh = gradhess(&[], &[]);
        assert!(matches!(
            build_hist(&x, &gh, &[], &[], usize::MAX, &[0]),
            Err(PbError::Internal { .. })
        ));
    }

    #[test]
    fn malformed_hist_offset_overflow_returns_none() {
        let hist = Hist {
            n_leaves: usize::MAX,
            n_axes: usize::MAX,
            n_bins: usize::MAX,
            ..Hist::default()
        };
        assert_eq!(hist.offset(2, 0, 0), None);
    }

    #[test]
    fn out_of_range_inputs_return_internal_not_panic() {
        let x = matrix(vec![vec![1u8, 1]], &[3]);
        let gh = gradhess(&[1.0, 2.0], &[1.0, 1.0]);
        let rows = [0u32, 1];
        // Axis with no grid/column.
        assert!(matches!(
            build_hist(&x, &gh, &rows, &[0, 0], 1, &[5]),
            Err(PbError::Internal { .. })
        ));
        // A leaf_of_row entry >= n_leaves.
        assert!(matches!(
            build_hist(&x, &gh, &rows, &[0, 3], 1, &[0]),
            Err(PbError::Internal { .. })
        ));
        // A bin id >= max_bins (malformed column value 5 against a 3-bin grid).
        let bad = matrix(vec![vec![5u8, 1]], &[3]);
        assert!(matches!(
            build_hist(&bad, &gh, &rows, &[0, 0], 1, &[0]),
            Err(PbError::Internal { .. })
        ));
        // A row id outside the column.
        assert!(matches!(
            build_hist(&x, &gh, &[0, 9], &[0, 0], 1, &[0]),
            Err(PbError::Internal { .. })
        ));
    }

    #[test]
    fn subtract_count_underflow_is_internal() {
        let parent = Hist::try_zeros(1, 1, 2).unwrap();
        let mut left = Hist::try_zeros(1, 1, 2).unwrap();
        left.count[0] = 5; // left ⊄ parent ⇒ count underflow
        assert!(matches!(
            subtract(&parent, &left),
            Err(PbError::Internal { .. })
        ));
    }

    #[test]
    fn subtract_sibling_into_matches_hand_computed_larger_child() {
        // parent: 2 leaves × 2 axes × 3 bins; child: 4 leaves × 1 axis (axis_map=[1], so the
        // child's only axis is the parent's axis 1 — a NON-identity remap, the A_2⊂A_1 case)
        // × 3 bins. Smaller children (leaves 0,1) are populated; the larger (2,3) are derived.
        let mut parent = Hist::try_zeros(2, 2, 3).unwrap();
        let mut child = Hist::try_zeros(4, 1, 3).unwrap();
        let set = |hh: &mut Hist, leaf: usize, axis: usize, bin: usize, g: f64, c: u32| {
            let o = hh.offset(leaf, axis, bin).unwrap();
            hh.g[o] = g;
            hh.h[o] = g * 0.1;
            hh.wsum[o] = f64::from(c);
            hh.count[o] = c;
        };
        // parent axis 1 (axis 0 left zero — unused by the map): leaf 0 then leaf 1.
        set(&mut parent, 0, 1, 0, 10.0, 10);
        set(&mut parent, 0, 1, 1, 20.0, 20);
        set(&mut parent, 0, 1, 2, 30.0, 30);
        set(&mut parent, 1, 1, 0, 5.0, 5);
        set(&mut parent, 1, 1, 1, 15.0, 15);
        set(&mut parent, 1, 1, 2, 25.0, 25);
        // smaller children on child axis 0: leaf 0 (smaller of parent 0), leaf 1 (of parent 1).
        set(&mut child, 0, 0, 0, 3.0, 3);
        set(&mut child, 0, 0, 1, 8.0, 8);
        set(&mut child, 0, 0, 2, 12.0, 12);
        set(&mut child, 1, 0, 0, 2.0, 2);
        set(&mut child, 1, 0, 1, 5.0, 5);
        set(&mut child, 1, 0, 2, 10.0, 10);
        // parent p → (smaller, larger): 0 → (0, 2); 1 → (1, 3).
        subtract_sibling_into(&mut child, &parent, &[(0, 0, 2), (1, 1, 3)], &[1]).unwrap();
        let chk = |leaf: usize, bin: usize, g: f64, c: u32| {
            let o = child.offset(leaf, 0, bin).unwrap();
            assert!((child.g[o] - g).abs() < 1e-12, "g leaf {leaf} bin {bin}");
            assert!(
                (child.wsum[o] - f64::from(c)).abs() < 1e-12,
                "wsum leaf {leaf} bin {bin}"
            );
            assert_eq!(child.count[o], c, "count leaf {leaf} bin {bin}");
        };
        // larger children = parent − smaller.
        chk(2, 0, 7.0, 7);
        chk(2, 1, 12.0, 12);
        chk(2, 2, 18.0, 18);
        chk(3, 0, 3.0, 3);
        chk(3, 1, 10.0, 10);
        chk(3, 2, 15.0, 15);
        // smaller children untouched.
        chk(0, 0, 3.0, 3);
        chk(1, 2, 10.0, 10);
    }

    #[test]
    fn subtract_sibling_into_count_underflow_is_internal() {
        let mut parent = Hist::try_zeros(1, 1, 1).unwrap();
        parent.count[0] = 2;
        let mut child = Hist::try_zeros(2, 1, 1).unwrap();
        let o = child.offset(0, 0, 0).unwrap();
        child.count[o] = 5; // smaller (5) > parent leaf (2) ⇒ underflow on the larger child
        assert!(matches!(
            subtract_sibling_into(&mut child, &parent, &[(0, 0, 1)], &[0]),
            Err(PbError::Internal { .. })
        ));
    }

    #[test]
    fn subtract_sibling_into_shape_mismatch_errors() {
        // child.n_leaves must be 2× parent.
        let parent = Hist::try_zeros(2, 1, 1).unwrap();
        let mut child = Hist::try_zeros(2, 1, 1).unwrap();
        assert!(matches!(
            subtract_sibling_into(&mut child, &parent, &[(0, 0, 1)], &[0]),
            Err(PbError::ShapeMismatch { .. })
        ));
        // axis_map.len() must equal child.n_axes.
        let parent2 = Hist::try_zeros(2, 2, 1).unwrap();
        let mut child2 = Hist::try_zeros(4, 2, 1).unwrap();
        assert!(matches!(
            subtract_sibling_into(&mut child2, &parent2, &[(0, 0, 2)], &[0]),
            Err(PbError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn counts_are_bounded_by_rows_and_total_matches() {
        // Every cell's count <= n_rows, and the total count over a single axis equals
        // the number of accumulated rows (the u32 count can never exceed n_rows).
        let (x, gh, rows, leaf_of_row) = fixture();
        let hist = build_hist(&x, &gh, &rows, &leaf_of_row, 4, &[0]).unwrap();
        let total: u64 = hist.count.iter().map(|&c| u64::from(c)).sum();
        assert_eq!(total, rows.len() as u64);
        assert!(hist
            .count
            .iter()
            .all(|&c| u64::from(c) <= u64::from(x.n_rows)));
    }

    #[test]
    fn quantize_grad_hess_is_deterministic_and_half_step_bounded() {
        let gh = gradhess(&[0.0, 0.125, -0.75, 2.5, -3.25], &[1.0, 0.5, 2.0, 4.0, 8.0]);
        let q1 = quantize_grad_hess(&gh, 77, 3).unwrap();
        let q2 = quantize_grad_hess(&gh, 77, 3).unwrap();
        assert_eq!(q1, q2);
        let g_step = 1.0_f64 / f64::from(q1.scale.g_scale);
        let h_step = 1.0_f64 / f64::from(q1.scale.h_scale);
        for ((&g, &gq), (&h, &hq)) in gh.g.iter().zip(&q1.g_q).zip(gh.h.iter().zip(&q1.h_q)) {
            let dg = (f64::from(gq) * g_step - f64::from(g)).abs();
            let dh = (f64::from(hq) * h_step - f64::from(h)).abs();
            assert!(dg <= 0.5 * g_step + f64::EPSILON, "dg={dg}, step={g_step}");
            assert!(dh <= 0.5 * h_step + f64::EPSILON, "dh={dh}, step={h_step}");
        }
    }

    #[test]
    fn quantized_histogram_is_thread_count_independent() {
        let (x, gh, rows, leaf_of_row) = fixture();
        let build = |threads: usize| -> Hist {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                build_quantized_histogram(
                    &x,
                    &gh,
                    &rows,
                    &leaf_of_row,
                    4,
                    &[0, 1],
                    QuantizeContext { seed: 99, round: 4 },
                    &vec![1.0_f32; gh.g.len()],
                )
                .unwrap()
            })
        };
        let h1 = build(1);
        assert_eq!(h1, build(2));
        assert_eq!(h1, build(8));
    }

    #[test]
    fn wsum_tracks_weighted_mass_per_cell_and_subtracts_exactly() {
        // Two cells (bin 1 / bin 2) on one axis; weights differ from counts so Σw must be
        // distinct from `count` (the credibility `min_weight_sum_in_leaf` path).
        let cols = vec![vec![1u8, 2, 1, 2]];
        let x = matrix(cols, &[3]);
        let gh = gradhess(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0, 1.0, 1.0]);
        let weight = [0.5_f32, 2.0, 1.5, 4.0];
        let rows: Vec<u32> = (0..4).collect();
        let one_leaf = vec![0u8; 4];
        let hist = build_histogram(&x, &gh, &rows, &one_leaf, 1, &[0], &weight, false).unwrap();
        let o1 = hist.offset(0, 0, 1).unwrap();
        let o2 = hist.offset(0, 0, 2).unwrap();
        // bin1 = rows 0,2 (w 0.5+1.5=2.0); bin2 = rows 1,3 (w 2.0+4.0=6.0).
        assert!((hist.wsum[o1] - 2.0).abs() < 1e-12);
        assert!((hist.wsum[o2] - 6.0).abs() < 1e-12);
        // Σw obeys the subtraction trick exactly for this well-conditioned data.
        let parent = build_histogram(&x, &gh, &rows, &one_leaf, 1, &[0], &weight, false).unwrap();
        let left =
            build_histogram(&x, &gh, &rows[..2], &one_leaf, 1, &[0], &weight, false).unwrap();
        let right = subtract(&parent, &left).unwrap();
        let direct =
            build_histogram(&x, &gh, &rows[2..], &one_leaf, 1, &[0], &weight, false).unwrap();
        for (a, b) in right.wsum.iter().zip(&direct.wsum) {
            assert!((a - b).abs() < 1e-9, "wsum subtraction drift {a} vs {b}");
        }
    }

    #[test]
    fn unit_weights_make_wsum_equal_count() {
        let (x, gh, rows, leaf_of_row) = fixture();
        let hist = build_hist(&x, &gh, &rows, &leaf_of_row, 4, &[0, 1]).unwrap();
        for (w, &c) in hist.wsum.iter().zip(&hist.count) {
            assert!((w - f64::from(c)).abs() < 1e-12);
        }
    }

    #[test]
    fn unit_weight_fast_path_is_bit_identical_to_full_sigma_w() {
        // The `unit_weight=true` fast path (skip per-row Σw, set wsum=count) must produce a
        // histogram bit-for-bit equal to the full Σw path fed all-ones — the byte-identity
        // the engine relies on whenever the caller supplied no weights. Exercise both the
        // sequential and the row-chunk-parallel branches (rows >= ROW_PAR_MIN_ROWS).
        let (xs, gh_s, rows_s, leaf_s) = fixture();
        let ones_s = vec![1.0_f32; gh_s.g.len()];
        let big_n = ROW_PAR_MIN_ROWS + 257;
        let cols: Vec<Vec<u8>> = vec![
            (0..big_n).map(|i| (i % 7) as u8 + 1).collect(),
            (0..big_n).map(|i| (i % 5) as u8 + 1).collect(),
        ];
        let xb = matrix(cols, &[8, 6]);
        let gh_b = gradhess(
            &(0..big_n)
                .map(|i| (i % 11) as f32 - 5.0)
                .collect::<Vec<_>>(),
            &(0..big_n).map(|i| 1.0 + (i % 3) as f32).collect::<Vec<_>>(),
        );
        let rows_b: Vec<u32> = (0..big_n as u32).collect();
        let leaf_b = vec![0u8; big_n];
        let ones_b = vec![1.0_f32; big_n];
        for (x, gh, rows, leaf, ones, nl, axes) in [
            (
                &xs,
                &gh_s,
                &rows_s,
                &leaf_s,
                &ones_s,
                4usize,
                &[0u32, 1][..],
            ),
            (
                &xb,
                &gh_b,
                &rows_b,
                &leaf_b,
                &ones_b,
                1usize,
                &[0u32, 1][..],
            ),
        ] {
            let slow = build_histogram(x, gh, rows, leaf, nl, axes, ones, false).unwrap();
            let fast = build_histogram(x, gh, rows, leaf, nl, axes, ones, true).unwrap();
            assert_eq!(slow.g.len(), fast.g.len());
            for i in 0..slow.g.len() {
                assert_eq!(slow.g[i].to_bits(), fast.g[i].to_bits(), "g cell {i}");
                assert_eq!(slow.h[i].to_bits(), fast.h[i].to_bits(), "h cell {i}");
                assert_eq!(
                    slow.wsum[i].to_bits(),
                    fast.wsum[i].to_bits(),
                    "wsum cell {i}"
                );
                assert_eq!(slow.count[i], fast.count[i], "count cell {i}");
            }
        }
    }
}
