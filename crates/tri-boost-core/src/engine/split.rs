//! The oblivious Newton split-finder (spec §06.2 / §06.4, milestone M1.4).
//!
//! A tree is grown one shared split per level (depth 1→3). At each level the level
//! histogram (over the admissible axes) is scanned for the `(axis, bin_le)` that
//! maximizes the SUMMED Newton gain across all current leaves; the reserved missing
//! bin is tried both sides and the better direction is recorded as the learned
//! `Split.missing_left`. The feature-budget guard (I1) keeps each level on a fresh
//! raw feature, so a tree never exceeds 3 distinct raw features. Growth terminates
//! early (a valid lower-depth tree, not an error) when no admissible candidate clears
//! the `min_split_gain` floor. Leaves are the exact Newton step `w* = −G/(H+λ)·lr`
//! from FULL-PRECISION sums.
//!
//! The split scan is sequential (cheap — O(leaves·axes·bins), independent of row
//! count) with a deterministic first-wins argmax; the parallelism lives in the
//! histogram build (`engine::hist`). Determinism is therefore structural.

use crate::constraints::MonoSign;
use crate::data::BinnedMatrix;
use crate::engine::hist::{build_histogram, build_quantized_histogram, QuantizeContext};
use crate::engine::{low_bit, Hist, HistPrecision, ObliviousTree, Split};
use crate::error::PbError;
use crate::explain::FeatureSet;
use crate::loss::GradHess;

fn internal(what: &'static str) -> impl Fn() -> PbError {
    move || PbError::Internal { what: what.into() }
}

/// The split-finder's parameters (the slice of the §06 `Config` the green-spine
/// grower needs; M1.5's `Config` produces one). Credibility floors
/// (`min_sum_hessian_in_leaf`, `min_data_in_leaf`) and monotone bounds are §07/M1.5
/// and land then.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GrowConfig<'a> {
    /// L2 leaf regularizer `λ` (in `w* = −G/(H+λ)` and the gain).
    pub lambda: f64,
    /// Learning rate applied to each leaf value.
    pub lr: f64,
    /// `gamma` floor: a level terminates if the best gain is `<= min_split_gain`.
    pub min_split_gain: f64,
    /// Whole-tree interaction-order cap (`1..=3`); each level uses a fresh raw feature.
    pub max_order: u8,
    /// Leaf-stage `|w*|`-clamp resolved from `Config.max_delta_step` ∨ `Loss::max_delta_step()`
    /// (§05.6). `None` = uncapped; applied on the full-precision aggregated Newton step.
    pub max_delta_step: Option<f64>,
    /// Histogram precision for split search.
    pub hist_precision: HistPrecision,
    /// Base deterministic seed for quantized stochastic rounding.
    pub quant_seed: u64,
    /// Boosting round, used as the quantization re-seed coordinate.
    pub round: u32,
    /// Optional whole-tree feature-group whitelist (§07): every realized tree support
    /// must be a subset of at least one group.
    pub groups: Option<&'a [FeatureSet]>,
    /// Optional per-axis monotone signs resolved at fit entry (§07).
    pub monotone: Option<&'a [Option<MonoSign>]>,
}

/// A candidate level split with its summed Newton gain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Candidate {
    /// Global axis (feature column) index.
    pub axis: u32,
    /// Inclusive upper bin for the low (left) child.
    pub bin_le: u8,
    /// Learned missing-bin direction (bin 0 routes low when `true`).
    pub missing_left: bool,
    /// The summed Newton gain `½ Σ_leaf [G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)]`.
    pub gain: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MonotoneScan<'a> {
    level: usize,
    chosen: &'a [Option<MonoSign>],
    candidate_axis_signs: &'a [Option<MonoSign>],
    lr: f64,
    max_delta_step: Option<f64>,
}

/// The Newton term `G²/(H+λ)`, guarded so a non-positive denominator contributes 0
/// rather than `inf`/`NaN` (with `λ>0` and `H≥0` it never binds, but stays safe for
/// general losses).
fn newton_term(g: f64, h: f64, lambda: f64) -> f64 {
    let denom = h + lambda;
    if denom > 0.0 {
        g * g / denom
    } else {
        0.0
    }
}

fn newton_leaf(g: f64, h: f64, lambda: f64, lr: f64, max_delta_step: Option<f64>) -> f64 {
    let denom = h + lambda;
    let w = if denom > 0.0 { -g / denom } else { 0.0 };
    let w = match max_delta_step {
        Some(d) => w.clamp(-d, d),
        None => w,
    };
    lr * w
}

fn candidate_monotone_ok(
    values: &[f64],
    depth: usize,
    signs: &[Option<MonoSign>],
) -> Result<bool, PbError> {
    let n_leaves = 1usize << depth;
    if values.len() != n_leaves || signs.len() != depth {
        return Err(PbError::Internal {
            what: "monotone candidate shape mismatch".into(),
        });
    }
    for (level, sign) in signs.iter().enumerate() {
        let Some(sign) = sign else {
            continue;
        };
        if matches!(sign, MonoSign::None) {
            continue;
        }
        let bit = 1usize << level;
        for high in 0..n_leaves {
            if high & bit != 0 {
                continue;
            }
            let low = high | bit;
            let low_v = *values.get(low).ok_or_else(internal("monotone low"))?;
            let high_v = *values.get(high).ok_or_else(internal("monotone high"))?;
            let ok = match sign {
                MonoSign::Increasing => low_v <= high_v,
                MonoSign::Decreasing => low_v >= high_v,
                MonoSign::None => true,
            };
            if !ok {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Scan one level's histogram for the best shared `(axis, bin_le, missing_left)`
/// split (spec §06.2). `axes[p]` is the global axis of histogram column `p`, and
/// `n_data_bins[p]` is that axis's data-bin count (candidate `bin_le ∈ 1..=ndb-1`).
/// Returns the gain-maximizing candidate that clears `min_split_gain`, or `None`
/// (graceful early-termination). Ties break deterministically: lowest axis, then
/// lowest `bin_le`, then `missing_left = false` (sequential first-wins, strict `>`).
///
/// # Errors
/// [`PbError::Internal`] on an out-of-range histogram offset (a build/shape bug).
pub(crate) fn best_level_split(
    hist: &Hist,
    axes: &[u32],
    n_data_bins: &[usize],
    lambda: f64,
    min_split_gain: f64,
    monotone: Option<MonotoneScan<'_>>,
) -> Result<Option<Candidate>, PbError> {
    let nl = hist.n_leaves;
    let mut best: Option<Candidate> = None;

    for p in 0..hist.n_axes {
        let axis = *axes.get(p).ok_or_else(internal("axis index"))?;
        let ndb = *n_data_bins
            .get(p)
            .ok_or_else(internal("n_data_bins index"))?;
        if ndb < 2 {
            continue; // need >=2 data bins for a non-trivial split
        }

        // Per-leaf totals (all bins) and the missing-bin (bin 0) mass.
        let mut total_g = vec![0.0_f64; nl];
        let mut total_h = vec![0.0_f64; nl];
        let mut miss_g = vec![0.0_f64; nl];
        let mut miss_h = vec![0.0_f64; nl];
        for leaf in 0..nl {
            let mut tg = 0.0_f64;
            let mut th = 0.0_f64;
            for b in 0..hist.n_bins {
                let o = hist
                    .offset(leaf, p, b)
                    .ok_or_else(internal("scan offset"))?;
                tg += *hist.g.get(o).ok_or_else(internal("scan g"))?;
                th += *hist.h.get(o).ok_or_else(internal("scan h"))?;
            }
            let leaf_g = total_g.get_mut(leaf).ok_or_else(internal("total_g"))?;
            *leaf_g = tg;
            let leaf_h = total_h.get_mut(leaf).ok_or_else(internal("total_h"))?;
            *leaf_h = th;
            let o0 = hist
                .offset(leaf, p, 0)
                .ok_or_else(internal("miss offset"))?;
            *miss_g.get_mut(leaf).ok_or_else(internal("miss_g"))? =
                *hist.g.get(o0).ok_or_else(internal("miss g"))?;
            *miss_h.get_mut(leaf).ok_or_else(internal("miss_h"))? =
                *hist.h.get(o0).ok_or_else(internal("miss h"))?;
        }
        let parent: f64 = (0..nl)
            .map(|l| {
                newton_term(
                    *total_g.get(l).unwrap_or(&0.0),
                    *total_h.get(l).unwrap_or(&0.0),
                    lambda,
                )
            })
            .sum();

        // Prefix the data bins 1..=v as v advances; evaluate both missing directions.
        let mut data_l_g = vec![0.0_f64; nl];
        let mut data_l_h = vec![0.0_f64; nl];
        for v in 1..ndb {
            for leaf in 0..nl {
                let o = hist
                    .offset(leaf, p, v)
                    .ok_or_else(internal("prefix offset"))?;
                *data_l_g.get_mut(leaf).ok_or_else(internal("data_l_g"))? +=
                    *hist.g.get(o).ok_or_else(internal("prefix g"))?;
                *data_l_h.get_mut(leaf).ok_or_else(internal("data_l_h"))? +=
                    *hist.h.get(o).ok_or_else(internal("prefix h"))?;
            }
            // NOTE: this is the AGGREGATE dual of the canonical `low_bit` rule —
            // `ml=true` routes the missing bin into the left sum exactly as
            // `low_bit(0, _, true) = true` routes a missing row left. The two
            // encodings (this set-partition over histogram bins vs. the per-row
            // `low_bit`) must stay in agreement; the grow→lookup round-trip proptest
            // guards that. A change to the routing rule must touch both.
            for &ml in &[false, true] {
                let mut acc = 0.0_f64;
                for leaf in 0..nl {
                    let dlg = *data_l_g.get(leaf).ok_or_else(internal("dlg"))?;
                    let dlh = *data_l_h.get(leaf).ok_or_else(internal("dlh"))?;
                    let (lg, lh) = if ml {
                        (
                            dlg + *miss_g.get(leaf).ok_or_else(internal("mg"))?,
                            dlh + *miss_h.get(leaf).ok_or_else(internal("mh"))?,
                        )
                    } else {
                        (dlg, dlh)
                    };
                    let tg = *total_g.get(leaf).ok_or_else(internal("tg"))?;
                    let th = *total_h.get(leaf).ok_or_else(internal("th"))?;
                    acc += newton_term(lg, lh, lambda) + newton_term(tg - lg, th - lh, lambda);
                }
                let gain = 0.5 * (acc - parent);
                if let Some(scan) = monotone {
                    let mut signs = Vec::with_capacity(scan.level + 1);
                    signs.extend_from_slice(scan.chosen);
                    signs.push(
                        *scan
                            .candidate_axis_signs
                            .get(p)
                            .ok_or_else(internal("monotone candidate sign"))?,
                    );
                    let mut values = vec![0.0_f64; 1usize << (scan.level + 1)];
                    for leaf in 0..nl {
                        let dlg = *data_l_g.get(leaf).ok_or_else(internal("dlg"))?;
                        let dlh = *data_l_h.get(leaf).ok_or_else(internal("dlh"))?;
                        let (lg, lh) = if ml {
                            (
                                dlg + *miss_g.get(leaf).ok_or_else(internal("mg"))?,
                                dlh + *miss_h.get(leaf).ok_or_else(internal("mh"))?,
                            )
                        } else {
                            (dlg, dlh)
                        };
                        let tg = *total_g.get(leaf).ok_or_else(internal("tg"))?;
                        let th = *total_h.get(leaf).ok_or_else(internal("th"))?;
                        let high = leaf;
                        let low = leaf | (1usize << scan.level);
                        *values
                            .get_mut(low)
                            .ok_or_else(internal("monotone low value"))? =
                            newton_leaf(lg, lh, lambda, scan.lr, scan.max_delta_step);
                        *values
                            .get_mut(high)
                            .ok_or_else(internal("monotone high value"))? =
                            newton_leaf(tg - lg, th - lh, lambda, scan.lr, scan.max_delta_step);
                    }
                    if !candidate_monotone_ok(&values, scan.level + 1, &signs)? {
                        continue;
                    }
                }
                // Strict `>` ⇒ the first candidate (lowest axis/bin_le, ml=false) wins
                // ties ⇒ deterministic argmax.
                let improves = match best {
                    Some(b) => gain > b.gain,
                    None => true,
                };
                if gain > min_split_gain && improves {
                    best = Some(Candidate {
                        axis,
                        bin_le: u8::try_from(v).map_err(|_| PbError::Internal {
                            what: "bin_le exceeded u8".into(),
                        })?,
                        missing_left: ml,
                        gain,
                    });
                }
            }
        }
    }
    Ok(best)
}

/// Exact Newton leaf values from FULL-PRECISION sums (spec §06.4): `w* = −G/(H+λ)`,
/// scaled by `lr`. `leaf_of_row[r] ∈ 0..2^depth` is row `r`'s leaf; the unused tail
/// `leaves[2^depth..]` stays `0.0`. Sequential f64 fold ⇒ thread-count independent.
///
/// # Errors
/// [`PbError::Internal`] if a row's leaf id is out of range or an index escapes.
pub(crate) fn leaf_values(
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    depth: usize,
    lambda: f64,
    lr: f64,
    max_delta_step: Option<f64>,
) -> Result<[f32; 8], PbError> {
    let n_leaves = 1usize << depth;
    let mut g = vec![0.0_f64; n_leaves];
    let mut h = vec![0.0_f64; n_leaves];
    for &r in rows {
        let ru = r as usize;
        let leaf = usize::from(*leaf_of_row.get(ru).ok_or_else(internal("leaf map"))?);
        if leaf >= n_leaves {
            return Err(PbError::Internal {
                what: "leaf id out of range in leaf_values".into(),
            });
        }
        *g.get_mut(leaf).ok_or_else(internal("leaf g"))? +=
            f64::from(*gh.g.get(ru).ok_or_else(internal("gh.g"))?);
        *h.get_mut(leaf).ok_or_else(internal("leaf h"))? +=
            f64::from(*gh.h.get(ru).ok_or_else(internal("gh.h"))?);
    }
    let mut leaves = [0.0_f32; 8];
    for j in 0..n_leaves {
        let gj = *g.get(j).ok_or_else(internal("g[j]"))?;
        let hj = *h.get(j).ok_or_else(internal("h[j]"))?;
        let denom = hj + lambda;
        let w = if denom > 0.0 { -gj / denom } else { 0.0 };
        // §05.6 max_delta_step: clamp |w*| ≤ δ on the FULL-PRECISION aggregated step
        // (before lr), so the cap never perturbs the future quantized histogram.
        let w = match max_delta_step {
            Some(d) => w.clamp(-d, d),
            None => w,
        };
        let value = lr * w;
        if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
            return Err(PbError::InvalidInput {
                what: "Newton leaf value is not finite/representable as f32".into(),
            });
        }
        *leaves.get_mut(j).ok_or_else(internal("leaf slot"))? = value as f32;
    }
    Ok(leaves)
}

/// Refit an already-chosen tree structure from full-precision gradients over `rows`.
///
/// MVS uses a sampled row set only to choose the split structure; the leaf values are
/// then recomputed on all training rows so the final model remains a standard exact
/// constant-leaf ensemble.
///
/// # Errors
/// Propagates typed shape/index errors from row routing and the Newton leaf solve.
pub(crate) fn refit_tree_leaves(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    tree: &mut ObliviousTree,
    cfg: &GrowConfig<'_>,
) -> Result<(), PbError> {
    let mut leaf_of_row: Vec<u8> = Hist::try_zeroed_vec(x.n_rows as usize, "refit leaf map")?;
    for &r in rows {
        let ru = r as usize;
        let mut leaf = 0u8;
        for (level, split) in tree.splits.iter().enumerate() {
            let col = x
                .data
                .get(split.axis as usize)
                .ok_or_else(internal("refit split col"))?;
            let bin = *col.get(ru).ok_or_else(internal("refit split bin"))?;
            leaf |= u8::from(low_bit(bin, split.bin_le, split.missing_left)) << level;
        }
        *leaf_of_row
            .get_mut(ru)
            .ok_or_else(internal("refit leaf slot"))? = leaf;
    }
    tree.leaves = leaf_values(
        gh,
        rows,
        &leaf_of_row,
        usize::from(tree.depth),
        cfg.lambda,
        cfg.lr,
        cfg.max_delta_step,
    )?;
    Ok(())
}

/// Grow one depth-`1..=3` oblivious tree (spec §06.2/§06.6) over `rows`, scanning the
/// candidate `axes` (already column-sampled by the caller). Returns `None` when no
/// admissible candidate clears `min_split_gain` at the first level (a degenerate
/// no-split round — the boosting loop handles it). FLAG (spec §14 P2 signature
/// refinement): returns `Option<ObliviousTree>` so the no-split case is explicit
/// rather than a forced low-gain tree.
///
/// # Errors
/// [`PbError::Internal`] on an index/shape bug; [`Invariant::FeatureBudget`] if the
/// assembled tree violates I1 (cannot happen given the guard, but checked at
/// construction).
pub(crate) fn grow_oblivious_tree(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    axes: &[u32],
    cfg: &GrowConfig<'_>,
) -> Result<Option<ObliviousTree>, PbError> {
    let n_rows = x.n_rows as usize;
    let mut leaf_of_row: Vec<u8> = Hist::try_zeroed_vec(n_rows, "leaf assignment")?;
    let mut splits: Vec<Split> = Vec::new();
    let mut split_signs: Vec<Option<MonoSign>> = Vec::new();
    let mut used_raws: smallvec::SmallVec<[u32; 3]> = smallvec::SmallVec::new();
    let order_cap = usize::from(cfg.max_order).min(3);

    for level in 0..3usize {
        if used_raws.len() >= order_cap {
            break;
        }
        // Admissible axes: their raw feature is not already used by an earlier level (I1).
        let admissible: Vec<u32> = axes
            .iter()
            .copied()
            .filter(|&a| axis_is_admissible(x, a, &used_raws, cfg.groups))
            .collect();
        if admissible.is_empty() {
            break;
        }
        let n_data_bins: Vec<usize> = admissible
            .iter()
            .map(|&a| {
                x.grids
                    .get(a as usize)
                    .map_or(0, |grid| usize::from(grid.n_bins).saturating_sub(1))
            })
            .collect();

        let n_leaves = 1usize << level;
        let hist = match cfg.hist_precision {
            HistPrecision::FullF64 => {
                build_histogram(x, gh, rows, &leaf_of_row, n_leaves, &admissible)?
            }
            HistPrecision::QuantizedI32 => build_quantized_histogram(
                x,
                gh,
                rows,
                &leaf_of_row,
                n_leaves,
                &admissible,
                QuantizeContext {
                    seed: cfg.quant_seed,
                    round: cfg.round,
                },
            )?,
        };
        let candidate_axis_signs: Vec<Option<MonoSign>> = match cfg.monotone {
            Some(signs) => admissible
                .iter()
                .map(|&axis| signs.get(axis as usize).copied().flatten())
                .collect(),
            None => Vec::new(),
        };
        let monotone_scan = cfg.monotone.map(|_| MonotoneScan {
            level,
            chosen: &split_signs,
            candidate_axis_signs: &candidate_axis_signs,
            lr: cfg.lr,
            max_delta_step: cfg.max_delta_step,
        });
        let cand = match best_level_split(
            &hist,
            &admissible,
            &n_data_bins,
            cfg.lambda,
            cfg.min_split_gain,
            monotone_scan,
        )? {
            Some(c) => c,
            None => break, // graceful early-termination (depth < 3)
        };

        splits.push(Split {
            axis: cand.axis,
            bin_le: cand.bin_le,
            missing_left: cand.missing_left,
        });
        split_signs.push(match cfg.monotone {
            Some(signs) => signs.get(cand.axis as usize).copied().flatten(),
            None => None,
        });
        let raw = x
            .provenance
            .get(cand.axis as usize)
            .ok_or_else(internal("split axis provenance"))?
            .raw
            .0;
        used_raws.push(raw);

        // Sample→leaf update: set this level's bit using the SAME canonical low_bit.
        let col = x
            .data
            .get(cand.axis as usize)
            .ok_or_else(internal("split col"))?;
        for &r in rows {
            let ru = r as usize;
            let bin = *col.get(ru).ok_or_else(internal("split bin"))?;
            let bit = u8::from(low_bit(bin, cand.bin_le, cand.missing_left)) << level;
            *leaf_of_row.get_mut(ru).ok_or_else(internal("leaf set"))? |= bit;
        }
    }

    if splits.is_empty() {
        return Ok(None);
    }
    let depth = splits.len();
    let leaves = leaf_values(
        gh,
        rows,
        &leaf_of_row,
        depth,
        cfg.lambda,
        cfg.lr,
        cfg.max_delta_step,
    )?;
    Ok(Some(ObliviousTree::try_new(splits, leaves, &x.provenance)?))
}

fn axis_is_admissible(
    x: &BinnedMatrix,
    axis: u32,
    used_raws: &[u32],
    groups: Option<&[FeatureSet]>,
) -> bool {
    let Some(prov) = x.provenance.get(axis as usize) else {
        return false;
    };
    let raw = prov.raw;
    if used_raws.contains(&raw.0) {
        return false;
    }
    match groups {
        None => true,
        Some(groups) => groups.iter().any(|group| {
            group.contains(raw)
                && used_raws
                    .iter()
                    .all(|&used| group.contains(crate::data::FeatureId(used)))
        }),
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
    use crate::data::{AxisKind, AxisProvenance, BorderGrid, FeatureId};
    use crate::engine::Hist;
    use proptest::prelude::*;

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

    /// Hand-built 1-leaf histogram (1 axis, 3 bins: missing + 2 data) with a non-zero
    /// missing mass. The Newton gain and the learned missing direction match the
    /// closed form computed by hand.
    #[test]
    fn newton_gain_and_missing_direction_match_closed_form() {
        // bin0 (missing): g=2,h=1; bin1: g=4,h=2; bin2: g=-6,h=3. λ=1.
        let mut hist = Hist::try_zeros(1, 1, 3).unwrap();
        let set = |hist: &mut Hist, bin: usize, g: f64, h: f64, c: u32| {
            let o = hist.offset(0, 0, bin).unwrap();
            hist.g[o] = g;
            hist.h[o] = h;
            hist.count[o] = c;
        };
        set(&mut hist, 0, 2.0, 1.0, 1);
        set(&mut hist, 1, 4.0, 2.0, 1);
        set(&mut hist, 2, -6.0, 3.0, 1);

        let best = best_level_split(&hist, &[0], &[2], 1.0, 0.0, None)
            .unwrap()
            .unwrap();
        // Only candidate is v=1. total_g=0,total_h=6 ⇒ parent=0²/7=0.
        //   ml=false: L=bin1 (4,2)→16/3; R=bin2+miss (-4,4)→16/5; gain=½(16/3+16/5)=4.2667.
        //   ml=true:  L=bin1+miss (6,3)→9;  R=bin2 (-6,3)→9;      gain=½(9+9)=9.
        // ml=true wins (missing routed LEFT with bin1).
        assert_eq!(best.axis, 0);
        assert_eq!(best.bin_le, 1);
        assert!(
            best.missing_left,
            "missing should be learned LEFT for higher gain"
        );
        assert!((best.gain - 9.0).abs() < 1e-9, "gain {} != 9", best.gain);
    }

    #[test]
    fn below_min_split_gain_yields_no_candidate() {
        let mut hist = Hist::try_zeros(1, 1, 3).unwrap();
        let o = hist.offset(0, 0, 1).unwrap();
        hist.g[o] = 1.0;
        hist.h[o] = 1.0;
        // Single populated bin ⇒ any split is degenerate (gain 0); a positive floor rejects it.
        assert!(best_level_split(&hist, &[0], &[2], 1.0, 1.0, None)
            .unwrap()
            .is_none());
    }

    #[test]
    fn leaf_values_match_per_leaf_newton_solve() {
        // 4 rows, 2 leaves. leaf0 rows 0,1 (g 1,2; h 1,1 ⇒ G=3,H=2);
        // leaf1 rows 2,3 (g -4,-1; h 1,1 ⇒ G=-5,H=2). λ=1, lr=0.1.
        let gh = gradhess(&[1.0, 2.0, -4.0, -1.0], &[1.0, 1.0, 1.0, 1.0]);
        let rows = [0u32, 1, 2, 3];
        let leaf_of_row = [0u8, 0, 1, 1];
        let leaves = leaf_values(&gh, &rows, &leaf_of_row, 1, 1.0, 0.1, None).unwrap();
        // w*_0 = -3/(2+1) = -1 ⇒ leaf 0.1·-1 = -0.1; w*_1 = 5/3 ⇒ leaf 0.16667.
        assert!((leaves[0] - (-0.1)).abs() < 1e-6);
        assert!((leaves[1] - (5.0 / 3.0 * 0.1) as f32).abs() < 1e-6);
        // Unused tail (depth 1 ⇒ leaves 2..8) is zero.
        assert!(leaves[2..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn max_delta_step_caps_the_leaf_newton_step() {
        // A tiny hessian makes the Newton step w* = -G/(H+λ) explode; the §05.6 clamp
        // caps |w*| ≤ δ on the full-precision aggregate BEFORE the learning rate, so the
        // stored leaf is bounded by lr·δ. (This is the Poisson stability safeguard.)
        let gh = gradhess(&[-100.0], &[0.01]); // g=-100, h=0.01 ⇒ w* = 10000 uncapped
        let rows = [0u32];
        let leaf_of_row = [0u8];
        let uncapped = leaf_values(&gh, &rows, &leaf_of_row, 0, 0.0, 0.1, None).unwrap();
        assert!(
            uncapped[0] > 100.0,
            "uncapped leaf should be huge, got {}",
            uncapped[0]
        );
        let capped = leaf_values(&gh, &rows, &leaf_of_row, 0, 0.0, 0.1, Some(0.5)).unwrap();
        // |w*| clamped to 0.5 ⇒ leaf = 0.1·0.5 = 0.05.
        assert!(
            (capped[0] - 0.05).abs() < 1e-6,
            "clamped leaf should be 0.05, got {}",
            capped[0]
        );
    }

    #[test]
    fn unrepresentable_leaf_value_errors_instead_of_storing_inf() {
        let gh = gradhess(&[f32::MAX], &[f32::MIN_POSITIVE]);
        let rows = [0u32];
        let leaf_of_row = [0u8];
        assert!(matches!(
            leaf_values(&gh, &rows, &leaf_of_row, 0, 0.0, 1.0, None),
            Err(PbError::InvalidInput { .. })
        ));
    }

    /// A clean depth-2 fixture: two informative features ⇒ the engine grows a depth-2
    /// tree on two DISTINCT raw features (I1), with no early termination.
    fn cfg(lambda: f64, lr: f64, min_split_gain: f64, max_order: u8) -> GrowConfig<'static> {
        GrowConfig {
            lambda,
            lr,
            min_split_gain,
            max_order,
            max_delta_step: None,
            hist_precision: HistPrecision::FullF64,
            quant_seed: 0,
            round: 0,
            groups: None,
            monotone: None,
        }
    }

    #[test]
    fn grow_recovers_two_feature_structure() {
        // Target depends on (x0 bin, x1 bin); gradients pull leaves apart.
        let c0: Vec<u8> = vec![1, 1, 1, 1, 2, 2, 2, 2]; // axis0 splits rows 0-3 | 4-7
        let c1: Vec<u8> = vec![1, 1, 2, 2, 1, 1, 2, 2]; // axis1 splits within
        let x = matrix(vec![c0, c1], &[3, 3]);
        // Gradients: distinct per (x0,x1) quadrant so both splits have positive gain.
        let g = [-3.0_f32, -3.0, 1.0, 1.0, 2.0, 2.0, 6.0, 6.0];
        let gh = gradhess(&g, &[1.0; 8]);
        let rows: Vec<u32> = (0..8).collect();
        let tree = grow_oblivious_tree(&x, &gh, &rows, &[0, 1], &cfg(1.0, 1.0, 0.0, 3))
            .unwrap()
            .expect("a tree should grow");
        assert!((1..=3).contains(&tree.depth));
        // Distinct raw features across splits == depth (I1).
        let mut raws: Vec<u32> = tree.splits.iter().map(|s| s.axis).collect();
        raws.sort_unstable();
        raws.dedup();
        assert_eq!(raws.len(), usize::from(tree.depth));
    }

    #[test]
    fn grow_is_deterministic_across_thread_counts() {
        let n = 400usize;
        let c0: Vec<u8> = (0..n).map(|i| u8::try_from(i % 5 + 1).unwrap()).collect();
        let c1: Vec<u8> = (0..n).map(|i| u8::try_from(i % 7 + 1).unwrap()).collect();
        let c2: Vec<u8> = (0..n).map(|i| u8::try_from(i % 3 + 1).unwrap()).collect();
        let x = matrix(vec![c0, c1, c2], &[6, 8, 4]);
        let g: Vec<f32> = (0..n).map(|i| (i as f32 % 11.0) - 5.0).collect();
        let gh = gradhess(&g, &vec![1.0; n]);
        let rows: Vec<u32> = (0..n as u32).collect();
        let run = |nt: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                grow_oblivious_tree(&x, &gh, &rows, &[0, 1, 2], &cfg(1.0, 0.1, 0.0, 3))
                    .unwrap()
                    .unwrap()
            })
        };
        let a = run(1);
        assert_eq!(a, run(2));
        assert_eq!(a, run(8));
    }

    #[test]
    fn constant_gradient_gives_no_split() {
        // All gradients equal ⇒ no split improves the objective ⇒ None (no tree).
        let x = matrix(vec![vec![1u8, 2, 1, 2]], &[3]);
        let gh = gradhess(&[1.0, 1.0, 1.0, 1.0], &[1.0; 4]);
        let rows = [0u32, 1, 2, 3];
        let tree = grow_oblivious_tree(&x, &gh, &rows, &[0], &cfg(1.0, 0.1, 1e-9, 1)).unwrap();
        assert!(tree.is_none(), "constant gradient must not split");
    }

    #[test]
    fn feature_budget_is_enforced_at_construction() {
        // depth must equal distinct raw features. 4 distinct-raw splits ⇒ depth 4 > 3.
        let prov: Vec<AxisProvenance> = (0..4u32)
            .map(|i| AxisProvenance {
                raw: FeatureId(i),
                kind: AxisKind::Numeric,
            })
            .collect();
        let splits: Vec<Split> = (0..4u32)
            .map(|a| Split {
                axis: a,
                bin_le: 1,
                missing_left: false,
            })
            .collect();
        assert!(matches!(
            ObliviousTree::try_new(splits, [0.0; 8], &prov),
            Err(PbError::InvariantViolated {
                invariant: crate::Invariant::FeatureBudget
            })
        ));

        // Two splits on the SAME raw feature ⇒ distinct (1) != depth (2).
        let prov2 = vec![
            AxisProvenance {
                raw: FeatureId(0),
                kind: AxisKind::Numeric,
            },
            AxisProvenance {
                raw: FeatureId(0),
                kind: AxisKind::Numeric,
            },
        ];
        let reused = vec![
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
        ];
        assert!(matches!(
            ObliviousTree::try_new(reused, [0.0; 8], &prov2),
            Err(PbError::InvariantViolated {
                invariant: crate::Invariant::FeatureBudget
            })
        ));
    }

    #[test]
    fn max_order_one_caps_tree_at_depth_one() {
        // max_order = 1 ⇒ at most one split even when more features are informative.
        let c0: Vec<u8> = vec![1, 1, 2, 2];
        let c1: Vec<u8> = vec![1, 2, 1, 2];
        let x = matrix(vec![c0, c1], &[3, 3]);
        let gh = gradhess(&[-2.0, -1.0, 1.0, 2.0], &[1.0; 4]);
        let rows = [0u32, 1, 2, 3];
        let tree = grow_oblivious_tree(&x, &gh, &rows, &[0, 1], &cfg(1.0, 0.1, 0.0, 1))
            .unwrap()
            .unwrap();
        assert_eq!(tree.depth, 1);
    }

    #[test]
    fn group_whitelist_gates_candidate_axes() {
        // Axis 1 is the only informative axis; a group list that contains only raw 0
        // must therefore produce no tree, while a group containing raw 1 admits it.
        let c0: Vec<u8> = vec![1, 1, 1, 1];
        let c1: Vec<u8> = vec![1, 1, 2, 2];
        let x = matrix(vec![c0, c1], &[2, 3]);
        let gh = gradhess(&[-2.0, -2.0, 2.0, 2.0], &[1.0; 4]);
        let rows = [0u32, 1, 2, 3];

        let denied_groups = vec![FeatureSet::new(&[0])];
        let mut denied = cfg(1.0, 0.1, 0.0, 2);
        denied.groups = Some(&denied_groups);
        assert!(grow_oblivious_tree(&x, &gh, &rows, &[0, 1], &denied)
            .unwrap()
            .is_none());

        let allowed_groups = vec![FeatureSet::new(&[1])];
        let mut allowed = cfg(1.0, 0.1, 0.0, 2);
        allowed.groups = Some(&allowed_groups);
        let tree = grow_oblivious_tree(&x, &gh, &rows, &[0, 1], &allowed)
            .unwrap()
            .unwrap();
        assert_eq!(tree.splits[0].axis, 1);
    }

    #[test]
    fn newton_gain_learns_missing_right() {
        // Mirror of the learned-LEFT test: with bin1/bin2 gradients swapped, routing
        // missing RIGHT (with bin2) is the higher-gain pairing, so missing_left=false.
        // bin0(missing): g=2,h=1; bin1: g=-6,h=3; bin2: g=4,h=2. λ=1, total (0,6).
        //   ml=false: L=bin1 (-6,3)→9; R=bin2+miss (6,3)→9; gain=½(18)=9.
        //   ml=true:  L=bin1+miss (-4,4)→3.2; R=bin2 (4,2)→5.333; gain=4.267.
        let mut hist = Hist::try_zeros(1, 1, 3).unwrap();
        let set = |hist: &mut Hist, bin: usize, g: f64, h: f64| {
            let o = hist.offset(0, 0, bin).unwrap();
            hist.g[o] = g;
            hist.h[o] = h;
            hist.count[o] = 1;
        };
        set(&mut hist, 0, 2.0, 1.0);
        set(&mut hist, 1, -6.0, 3.0);
        set(&mut hist, 2, 4.0, 2.0);
        let best = best_level_split(&hist, &[0], &[2], 1.0, 0.0, None)
            .unwrap()
            .unwrap();
        assert!(!best.missing_left, "missing should be learned RIGHT here");
        assert!((best.gain - 9.0).abs() < 1e-9, "gain {} != 9", best.gain);
    }

    #[test]
    fn lookup_routes_missing_and_data_bins() {
        // The ONLY ObliviousTree::lookup test: a depth-1 tree, exercising data bins AND
        // the reserved missing bin (bin 0) under both learned directions. Leaf index
        // convention: low (bin <= bin_le) → bit 1, so leaves[1] is the LOW value.
        let prov = vec![AxisProvenance {
            raw: FeatureId(0),
            kind: AxisKind::Numeric,
        }];
        let leaves = [100.0_f32, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // idx0=high, idx1=low
        let left = ObliviousTree::try_new(
            vec![Split {
                axis: 0,
                bin_le: 1,
                missing_left: true,
            }],
            leaves,
            &prov,
        )
        .unwrap();
        assert_eq!(left.lookup(&[0]).unwrap(), 200.0); // missing → low (missing_left)
        assert_eq!(left.lookup(&[1]).unwrap(), 200.0); // bin 1 ≤ 1 → low
        assert_eq!(left.lookup(&[2]).unwrap(), 100.0); // bin 2 > 1 → high

        let right = ObliviousTree::try_new(
            vec![Split {
                axis: 0,
                bin_le: 1,
                missing_left: false,
            }],
            leaves,
            &prov,
        )
        .unwrap();
        assert_eq!(right.lookup(&[0]).unwrap(), 100.0); // missing → high
    }

    #[test]
    fn grow_recovers_three_feature_structure() {
        // 3 binary features, 8 rows = all (b0,b1,b2). Additive gradient with strictly
        // decreasing coefficients (4,2,1) ⇒ a full depth-3 tree on 3 distinct features.
        let c0 = vec![1u8, 1, 1, 1, 2, 2, 2, 2];
        let c1 = vec![1u8, 1, 2, 2, 1, 1, 2, 2];
        let c2 = vec![1u8, 2, 1, 2, 1, 2, 1, 2];
        let x = matrix(vec![c0.clone(), c1.clone(), c2.clone()], &[3, 3, 3]);
        let g: Vec<f32> = (0..8)
            .map(|r| {
                let b0 = f32::from(c0[r] - 1);
                let b1 = f32::from(c1[r] - 1);
                let b2 = f32::from(c2[r] - 1);
                4.0 * b0 + 2.0 * b1 + b2 - 3.5
            })
            .collect();
        let gh = gradhess(&g, &[1.0; 8]);
        let rows: Vec<u32> = (0..8).collect();
        // λ = 0: pure (non-negative) Newton gain, so every separating split is kept.
        // (With λ > 0 the small-variance 3rd split is correctly regularized away — the
        // L2 penalty adds λ to TWO child denominators vs one parent — which is why the
        // default-λ tree may stop at depth 2; that is correct behavior, not a defect.)
        let tree = grow_oblivious_tree(&x, &gh, &rows, &[0, 1, 2], &cfg(0.0, 0.1, 0.0, 3))
            .unwrap()
            .expect("a depth-3 tree should grow");
        assert_eq!(tree.depth, 3, "all three features are informative");
        let mut raws: Vec<u32> = tree.splits.iter().map(|s| s.axis).collect();
        raws.sort_unstable();
        raws.dedup();
        assert_eq!(raws, vec![0, 1, 2]); // each level a distinct raw feature (I1)
    }

    #[test]
    fn grown_tree_passes_check_feature_budget() {
        // Run the NAMED I1 gate fn (explain::check_feature_budget) on a fitted tree.
        let c0 = vec![1u8, 1, 2, 2];
        let c1 = vec![1u8, 2, 1, 2];
        let x = matrix(vec![c0, c1], &[3, 3]);
        let gh = gradhess(&[-2.0, -1.0, 1.0, 2.0], &[1.0; 4]);
        let rows = [0u32, 1, 2, 3];
        let tree = grow_oblivious_tree(&x, &gh, &rows, &[0, 1], &cfg(1.0, 0.1, 0.0, 3))
            .unwrap()
            .unwrap();
        let model = model_from(tree, &x);
        crate::explain::check_feature_budget(&model).expect("a grown tree must satisfy I1");
    }

    /// Wrap a grown tree in a minimal `Model` so the named I1 gate can run on it.
    fn model_from(tree: ObliviousTree, x: &BinnedMatrix) -> crate::engine::Model {
        use crate::cat::CatEncoderStore;
        use crate::engine::{ExactnessMode, ModelSchema};
        use crate::loss::{Link, LossId, ObjectiveTag};
        crate::engine::Model {
            f0: 0.0,
            trees: vec![(1.0, tree)],
            grids: x.grids.clone(),
            provenance: x.provenance.clone(),
            link: Link::Identity,
            mode: ExactnessMode::Exact,
            schema: ModelSchema {
                feature_names: Vec::new(),
                feature_kinds: Vec::new(),
                cat_encoders: CatEncoderStore::new(),
                class_labels: None,
                objective: ObjectiveTag {
                    link: Link::Identity,
                    loss: LossId::SquaredError,
                    tweedie_rho: None,
                },
            },
            schema_version: crate::serialize::SCHEMA_VERSION,
        }
    }

    proptest! {
        // I1 over RANDOM fitted trees + the grow→lookup round-trip (the canonical
        // low_bit must agree between the sample→leaf grower and ObliviousTree::lookup).
        // Columns include bin 0, so missing routing is exercised end-to-end.
        #[test]
        fn grown_trees_satisfy_i1_and_lookup_roundtrip(
            data in (1usize..4, 4usize..24).prop_flat_map(|(n_feat, n_rows)| {
                (
                    prop::collection::vec(prop::collection::vec(0u8..5u8, n_rows), n_feat),
                    prop::collection::vec(-10.0f32..10.0, n_rows),
                    prop::collection::vec(0.5f32..5.0, n_rows),
                )
            })
        ) {
            let (cols, g, h) = data;
            let n_feat = cols.len();
            let x = matrix(cols, &vec![5u16; n_feat]);
            let gh = gradhess(&g, &h);
            let rows: Vec<u32> = (0..x.n_rows).collect();
            let axes: Vec<u32> = (0..n_feat as u32).collect();
            let res = grow_oblivious_tree(&x, &gh, &rows, &axes, &cfg(1.0, 0.1, 0.0, 3));
            prop_assert!(res.is_ok());
            if let Some(tree) = res.unwrap() {
                // I1: depth 1..=3, splits.len()==depth, distinct raw features == depth.
                prop_assert!((1..=3).contains(&tree.depth));
                prop_assert_eq!(tree.splits.len(), usize::from(tree.depth));
                let mut raws: Vec<u32> =
                    tree.splits.iter().map(|s| x.provenance[s.axis as usize].raw.0).collect();
                raws.sort_unstable();
                raws.dedup();
                prop_assert_eq!(raws.len(), usize::from(tree.depth));
                // grow→lookup round-trip: lookup folds to the SAME leaf low_bit assigns.
                for r in 0..x.n_rows as usize {
                    let row_bins: Vec<u8> = (0..n_feat).map(|f| x.data[f][r]).collect();
                    let mut idx = 0usize;
                    for (lvl, s) in tree.splits.iter().enumerate() {
                        let bit = usize::from(crate::engine::low_bit(
                            x.data[s.axis as usize][r],
                            s.bin_le,
                            s.missing_left,
                        ));
                        idx |= bit << lvl;
                    }
                    prop_assert_eq!(tree.lookup(&row_bins).unwrap(), tree.leaves[idx]);
                }
            }
        }
    }
}
