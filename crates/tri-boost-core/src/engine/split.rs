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

use crate::backend::{pb_seed, Stage};
use crate::constraints::{CredibilityFloor, MonoSign};
use crate::data::BinnedMatrix;
use crate::engine::hist::{
    build_histogram, build_quantized_histogram, subtract_sibling_into, QuantizeContext,
};
use crate::engine::{low_bit, Hist, HistPrecision, ObliviousTree, Split};
use crate::error::PbError;
use crate::explain::FeatureSet;
use crate::loss::GradHess;

fn internal(what: &'static str) -> impl Fn() -> PbError {
    move || PbError::Internal { what: what.into() }
}

/// The split-finder's parameters (resolved from the §06 `Config` and the per-fit
/// `FitSpec`). Monotone bounds (§07.5) and credibility floors (§07.2/§07.6) are wired
/// through `monotone` and `credibility`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GrowConfig<'a> {
    /// L2 leaf regularizer `λ` (in `w* = −G/(H+λ)` and the gain).
    pub lambda: f64,
    /// L1 leaf regularizer used to soft-threshold aggregated gradients.
    pub l1_leaf: f64,
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
    /// Decaying deterministic split-score noise (§09.6). `0.0` is exactly inert.
    pub random_strength: f64,
    /// Optional whole-tree feature-group whitelist (§07): every realized tree support
    /// must be a subset of at least one group.
    pub groups: Option<&'a [FeatureSet]>,
    /// Optional per-axis monotone signs resolved at fit entry (§07).
    pub monotone: Option<&'a [Option<MonoSign>]>,
    /// Optional soft table-size admission prior (§07.3/§07.4). It changes only the
    /// ranking score; raw Newton gain still gates and is stored.
    pub table_budget_penalty: Option<TableBudgetPenalty>,
    /// Per-leaf credibility floors + `path_smooth` (§07.2/§07.6). All-zero is exactly
    /// inert. The three hard floors veto a candidate whose level produces any
    /// under-supported cell; `path_smooth` shrinks final leaves toward their parent.
    pub credibility: CredibilityFloor,
    /// `true` iff every sample weight is exactly `1.0` (the engine sets this only when the
    /// caller supplied NO weights, so the weight vector is the materialized all-ones). It
    /// lets the histogram skip the per-row `Σw` accumulation and set `wsum = count` (which
    /// is bit-exact for unit weights: summing `1.0` `k<2^53` times is exact). Conservative:
    /// `false` whenever weights were provided, even if they happen to all be `1.0`.
    pub unit_weight: bool,
    /// Enable the level-2 FullF64 histogram-subtraction fast path (build the smaller sibling
    /// child, derive the larger from the level-1 parent). `true` in production; a kill-switch
    /// and the A/B reference for the equivalence tests (subtraction reproduces the full-build
    /// tree, with g/h differing only at ~1e-11). Inert unless FullF64 and depth reaches 2.
    pub hist_subtraction: bool,
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

/// Soft table-size admission prior (§07.3/§07.4).
///
/// This is a ranking-only multiplier:
/// `score = gain.max(0) * (budget / max(budget, projected_cells))^beta`.
/// `beta = 0` is represented as `None` by [`TableBudgetPenalty::new`], which makes
/// the split finder exactly recover the unpenalized ordering and tie behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TableBudgetPenalty {
    beta: f64,
    budget_cells: u64,
}

impl TableBudgetPenalty {
    pub(crate) fn new(beta: f64, budget_cells: u64) -> Option<Self> {
        (beta > 0.0).then_some(Self { beta, budget_cells })
    }

    fn multiplier(
        self,
        x: &BinnedMatrix,
        used_axes: &[u32],
        candidate_axis: u32,
    ) -> Result<f64, PbError> {
        let mut cells = 1u64;
        for axis in used_axes
            .iter()
            .copied()
            .chain(std::iter::once(candidate_axis))
        {
            let grid = x
                .grids
                .get(axis as usize)
                .ok_or_else(internal("budget-prior axis grid"))?;
            cells = cells
                .checked_mul(u64::from(grid.n_bins))
                .ok_or_else(|| PbError::Internal {
                    what: "budget-prior projected cell count overflowed u64".into(),
                })?;
        }
        if cells <= self.budget_cells {
            Ok(1.0)
        } else {
            Ok((self.budget_cells as f64 / cells as f64).powf(self.beta))
        }
    }
}

/// Deterministic per-candidate split-score noise.
///
/// This is deliberately a ranking-only term: the raw Newton gain still gates
/// `min_split_gain` and is what gets stored on [`Candidate`]. The seed stream is
/// position-stable in `(seed, round, level, axis, bin, missing_left)`, so thread
/// count and scan partitioning cannot affect the selected candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SplitNoise {
    seed: u64,
    round: u32,
    level: usize,
    strength: f64,
}

impl SplitNoise {
    fn new(seed: u64, round: u32, level: usize, strength: f64) -> Option<Self> {
        (strength > 0.0).then_some(Self {
            seed,
            round,
            level,
            strength,
        })
    }

    fn adjustment(self, axis: u32, bin_le: usize, missing_left: bool) -> Result<f64, PbError> {
        let level = u64::try_from(self.level).map_err(|_| PbError::Internal {
            what: "split-noise level exceeded u64".into(),
        })?;
        let bin = u64::try_from(bin_le).map_err(|_| PbError::Internal {
            what: "split-noise bin exceeded u64".into(),
        })?;
        let salt = u64::from(axis).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ bin.wrapping_mul(0xBF58_476D_1CE4_E5B9)
            ^ level.wrapping_mul(0x94D0_49BB_1331_11EB)
            ^ u64::from(missing_left);
        let bits = pb_seed(self.seed ^ salt, self.round, Stage::SplitNoise as u32, axis);
        let mantissa = bits >> 11;
        const TWO_53: f64 = 9_007_199_254_740_992.0;
        let unit = mantissa as f64 / TWO_53;
        let centered = 2.0 * unit - 1.0;
        let decay = (f64::from(self.round) + 1.0).sqrt();
        Ok(centered * self.strength / decay)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MonotoneScan<'a> {
    level: usize,
    chosen: &'a [Option<MonoSign>],
    candidate_axis_signs: &'a [Option<MonoSign>],
    lr: f64,
    l1_leaf: f64,
    max_delta_step: Option<f64>,
}

/// Ranking-only split aids. These can choose among admissible raw-gain candidates,
/// but they never alter the stored [`Candidate::gain`] and never bypass
/// `min_split_gain`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RankingContext<'a> {
    monotone: Option<MonotoneScan<'a>>,
    noise: Option<SplitNoise>,
    table_penalties: Option<&'a [f64]>,
}

/// Twice the constrained quadratic improvement for a leaf.
///
/// Without L1/clamping this is `G²/(H+λ)`, matching the usual split-gain algebra.
/// When `l1_leaf` or `max_delta_step` is active, the split scan must rank the gain
/// the emitted leaf value can actually realize rather than the unconstrained Newton
/// optimum.
fn newton_term(g: f64, h: f64, lambda: f64, l1_leaf: f64, max_delta_step: Option<f64>) -> f64 {
    let denom = h + lambda;
    if denom > 0.0 {
        let g = soft_threshold(g, l1_leaf);
        let w = match max_delta_step {
            Some(d) => (-g / denom).clamp(-d, d),
            None => -g / denom,
        };
        (-2.0 * g * w - denom * w * w).max(0.0)
    } else {
        0.0
    }
}

fn soft_threshold(g: f64, l1_leaf: f64) -> f64 {
    if l1_leaf <= 0.0 {
        return g;
    }
    g.signum() * (g.abs() - l1_leaf).max(0.0)
}

fn newton_leaf(
    g: f64,
    h: f64,
    lambda: f64,
    l1_leaf: f64,
    lr: f64,
    max_delta_step: Option<f64>,
) -> f64 {
    let denom = h + lambda;
    let g = soft_threshold(g, l1_leaf);
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

/// Project an 8-leaf vector onto the monotone cone (§07.5) so every constrained level's
/// cousin pairs satisfy the required ordering, IN PLACE. The grow-time candidate filter
/// ([`candidate_monotone_ok`]) keeps the chosen STRUCTURE feasible, but any path that
/// RECOMPUTES leaves (MVS refit on full rows, the §09 fully-corrective ridge refit, or a
/// quantized-histogram round-off) can invert a cousin pair; applying this clamp at every
/// leaf-finalization site guarantees the monotone guarantee holds on the served model
/// (not just the freshly-grown one). Iterative cousin-pair pooling (POCS over the
/// half-space constraints) converges to a feasible point — the cone is non-empty (the
/// all-equal vector is always feasible). Leaf-VALUE only on a fixed structure, so I2 is
/// untouched. No-op when no constrained level is present.
pub(crate) fn clamp_monotone(
    leaves: &mut [f32; 8],
    splits: &[Split],
    depth: usize,
    axis_signs: Option<&[Option<MonoSign>]>,
) -> Result<(), PbError> {
    let Some(axis_signs) = axis_signs else {
        return Ok(());
    };
    // Resolve the per-level sign from each level's split axis; bail if none constrained.
    let mut level_sign: [Option<MonoSign>; 3] = [None; 3];
    let mut any = false;
    for (level, split) in splits.iter().enumerate().take(depth.min(3)) {
        let s = axis_signs
            .get(split.axis as usize)
            .copied()
            .flatten()
            .filter(|s| matches!(s, MonoSign::Increasing | MonoSign::Decreasing));
        if s.is_some() {
            *level_sign
                .get_mut(level)
                .ok_or_else(internal("level sign"))? = s;
            any = true;
        }
    }
    if !any {
        return Ok(());
    }
    let n_leaves = 1usize << depth.min(3);
    for _iter in 0..64 {
        let mut changed = false;
        for (level, sign) in level_sign.iter().enumerate().take(depth.min(3)) {
            let Some(sign) = sign else {
                continue;
            };
            let bit = 1usize << level;
            for high in 0..n_leaves {
                if high & bit != 0 {
                    continue; // `high` = bit_level 0 = HIGH feature value
                }
                let low = high | bit; // `low` = bit_level 1 = LOW feature value
                let lo_v = *leaves.get(low).ok_or_else(internal("clamp low"))?;
                let hi_v = *leaves.get(high).ok_or_else(internal("clamp high"))?;
                // Increasing: low feature value ⇒ low response ⇒ lo_v <= hi_v.
                let violated = match sign {
                    MonoSign::Increasing => lo_v > hi_v,
                    MonoSign::Decreasing => lo_v < hi_v,
                    MonoSign::None => false,
                };
                if violated {
                    let avg = 0.5 * (lo_v + hi_v);
                    *leaves.get_mut(low).ok_or_else(internal("clamp low set"))? = avg;
                    *leaves
                        .get_mut(high)
                        .ok_or_else(internal("clamp high set"))? = avg;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    Ok(())
}

/// Scan one level's histogram for the best shared `(axis, bin_le, missing_left)`
/// split (spec §06.2). `axes[p]` is the global axis of histogram column `p`, and
/// `n_data_bins[p]` is that axis's data-bin count (candidate `bin_le ∈ 1..=ndb-1`).
/// Returns the best ranking-score candidate whose raw Newton gain clears
/// `min_split_gain`, or `None` (graceful early-termination). With no split noise,
/// the ranking score is the raw gain. Ties break deterministically: lowest axis,
/// then lowest `bin_le`, then `missing_left = false` (sequential first-wins,
/// strict `>`).
///
/// # Errors
/// [`PbError::Internal`] on an out-of-range histogram offset (a build/shape bug).
#[allow(clippy::too_many_arguments)]
pub(crate) fn best_level_split(
    hist: &Hist,
    axes: &[u32],
    n_data_bins: &[usize],
    lambda: f64,
    l1_leaf: f64,
    max_delta_step: Option<f64>,
    min_split_gain: f64,
    ranking: RankingContext<'_>,
    credibility: &CredibilityFloor,
) -> Result<Option<Candidate>, PbError> {
    let nl = hist.n_leaves;
    let mut best: Option<(Candidate, f64)> = None;
    // §07.3 step 2: per-cell credibility floors are a HARD reject. Only track per-cell
    // count/Σw support (alongside the always-present Σh) when a floor can actually bind,
    // so a default (inert) floor leaves the selected candidate byte-identical.
    let check_cred = !credibility.rejects_nothing();
    let min_data = u64::from(credibility.min_data_in_leaf);
    let min_hess = f64::from(credibility.min_sum_hessian_in_leaf);
    let min_wsum = f64::from(credibility.min_weight_sum_in_leaf);

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
        // Per-leaf count/Σw totals + missing mass for the credibility floor (only filled
        // when a floor can bind, so the inert path keeps its exact arithmetic).
        let mut total_c = vec![0u64; nl];
        let mut total_w = vec![0.0_f64; nl];
        let mut miss_c = vec![0u64; nl];
        let mut miss_w = vec![0.0_f64; nl];
        for leaf in 0..nl {
            let mut tg = 0.0_f64;
            let mut th = 0.0_f64;
            let mut tc = 0u64;
            let mut tw = 0.0_f64;
            for b in 0..hist.n_bins {
                let o = hist
                    .offset(leaf, p, b)
                    .ok_or_else(internal("scan offset"))?;
                tg += *hist.g.get(o).ok_or_else(internal("scan g"))?;
                th += *hist.h.get(o).ok_or_else(internal("scan h"))?;
                if check_cred {
                    tc += u64::from(*hist.count.get(o).ok_or_else(internal("scan count"))?);
                    tw += *hist.wsum.get(o).ok_or_else(internal("scan wsum"))?;
                }
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
            if check_cred {
                *total_c.get_mut(leaf).ok_or_else(internal("total_c"))? = tc;
                *total_w.get_mut(leaf).ok_or_else(internal("total_w"))? = tw;
                *miss_c.get_mut(leaf).ok_or_else(internal("miss_c"))? =
                    u64::from(*hist.count.get(o0).ok_or_else(internal("miss count"))?);
                *miss_w.get_mut(leaf).ok_or_else(internal("miss_w"))? =
                    *hist.wsum.get(o0).ok_or_else(internal("miss wsum"))?;
            }
        }
        let parent: f64 = (0..nl)
            .map(|l| {
                newton_term(
                    *total_g.get(l).unwrap_or(&0.0),
                    *total_h.get(l).unwrap_or(&0.0),
                    lambda,
                    l1_leaf,
                    max_delta_step,
                )
            })
            .sum();

        // Prefix the data bins 1..=v as v advances; evaluate both missing directions.
        let mut data_l_g = vec![0.0_f64; nl];
        let mut data_l_h = vec![0.0_f64; nl];
        let mut data_l_c = vec![0u64; nl];
        let mut data_l_w = vec![0.0_f64; nl];
        for v in 1..ndb {
            for leaf in 0..nl {
                let o = hist
                    .offset(leaf, p, v)
                    .ok_or_else(internal("prefix offset"))?;
                *data_l_g.get_mut(leaf).ok_or_else(internal("data_l_g"))? +=
                    *hist.g.get(o).ok_or_else(internal("prefix g"))?;
                *data_l_h.get_mut(leaf).ok_or_else(internal("data_l_h"))? +=
                    *hist.h.get(o).ok_or_else(internal("prefix h"))?;
                if check_cred {
                    *data_l_c.get_mut(leaf).ok_or_else(internal("data_l_c"))? +=
                        u64::from(*hist.count.get(o).ok_or_else(internal("prefix count"))?);
                    *data_l_w.get_mut(leaf).ok_or_else(internal("data_l_w"))? +=
                        *hist.wsum.get(o).ok_or_else(internal("prefix wsum"))?;
                }
            }
            // NOTE: this is the AGGREGATE dual of the canonical `low_bit` rule —
            // `ml=true` routes the missing bin into the left sum exactly as
            // `low_bit(0, _, true) = true` routes a missing row left. The two
            // encodings (this set-partition over histogram bins vs. the per-row
            // `low_bit`) must stay in agreement; the grow→lookup round-trip proptest
            // guards that. A change to the routing rule must touch both.
            for &ml in &[false, true] {
                let mut acc = 0.0_f64;
                // Credibility (§07.3 step 2): a candidate is rejected if ANY of its child
                // cells across ALL current leaves falls under a floor — the symmetric
                // whole-level credibility guarantee.
                let mut credible = true;
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
                    acc += newton_term(lg, lh, lambda, l1_leaf, max_delta_step)
                        + newton_term(tg - lg, th - lh, lambda, l1_leaf, max_delta_step);
                    if check_cred && credible {
                        let dlc = *data_l_c.get(leaf).ok_or_else(internal("dlc"))?;
                        let dlw = *data_l_w.get(leaf).ok_or_else(internal("dlw"))?;
                        let (cl, wl) = if ml {
                            (
                                dlc + *miss_c.get(leaf).ok_or_else(internal("mc"))?,
                                dlw + *miss_w.get(leaf).ok_or_else(internal("mw"))?,
                            )
                        } else {
                            (dlc, dlw)
                        };
                        let tc = *total_c.get(leaf).ok_or_else(internal("tc cred"))?;
                        let tw = *total_w.get(leaf).ok_or_else(internal("tw cred"))?;
                        // h_low = lh, h_high = th − lh (the two child cells of this leaf).
                        if cl < min_data
                            || tc.saturating_sub(cl) < min_data
                            || lh < min_hess
                            || th - lh < min_hess
                            || wl < min_wsum
                            || tw - wl < min_wsum
                        {
                            credible = false;
                        }
                    }
                }
                let gain = 0.5 * (acc - parent);
                if !gain.is_finite() || gain <= min_split_gain {
                    continue;
                }
                if check_cred && !credible {
                    continue; // hard credibility reject (§07.3 step 2)
                }
                if let Some(scan) = ranking.monotone {
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
                            newton_leaf(lg, lh, lambda, scan.l1_leaf, scan.lr, scan.max_delta_step);
                        *values
                            .get_mut(high)
                            .ok_or_else(internal("monotone high value"))? = newton_leaf(
                            tg - lg,
                            th - lh,
                            lambda,
                            scan.l1_leaf,
                            scan.lr,
                            scan.max_delta_step,
                        );
                    }
                    if !candidate_monotone_ok(&values, scan.level + 1, &signs)? {
                        continue;
                    }
                }
                // Strict `>` ⇒ the first candidate (lowest axis/bin_le, ml=false) wins
                // ties ⇒ deterministic argmax.
                let penalty = match ranking.table_penalties {
                    Some(penalties) => *penalties
                        .get(p)
                        .ok_or_else(internal("ranking penalty index"))?,
                    None => 1.0,
                };
                let score = gain.max(0.0) * penalty
                    + match ranking.noise {
                        Some(split_noise) => split_noise.adjustment(axis, v, ml)?,
                        None => 0.0,
                    };
                let improves = match best {
                    Some((_, best_score)) => score > best_score,
                    None => true,
                };
                if improves {
                    best = Some((
                        Candidate {
                            axis,
                            bin_le: u8::try_from(v).map_err(|_| PbError::Internal {
                                what: "bin_le exceeded u8".into(),
                            })?,
                            missing_left: ml,
                            gain,
                        },
                        score,
                    ));
                }
            }
        }
    }
    Ok(best.map(|(candidate, _score)| candidate))
}

/// Exact Newton leaf values from FULL-PRECISION sums (spec §06.4): `w* = −G/(H+λ)`,
/// scaled by `lr`. `leaf_of_row[r] ∈ 0..2^depth` is row `r`'s leaf; the unused tail
/// `leaves[2^depth..]` stays `0.0`. Sequential f64 fold ⇒ thread-count independent.
///
/// # Errors
/// [`PbError::Internal`] if a row's leaf id is out of range or an index escapes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn leaf_values(
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    depth: usize,
    lambda: f64,
    l1_leaf: f64,
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
        let gj = soft_threshold(*g.get(j).ok_or_else(internal("g[j]"))?, l1_leaf);
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

/// Per-leaf full-precision `(Σg, Σh, count)` aggregates over `rows` — the inputs to
/// `path_smooth`'s per-node Newton outputs. Mirrors the `leaf_values` fold but also keeps
/// the row counts. Sequential f64 fold ⇒ thread-count independent.
///
/// # Errors
/// [`PbError::Internal`] on an out-of-range leaf id or row index.
#[allow(clippy::type_complexity)]
fn leaf_aggregates(
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    depth: usize,
) -> Result<([f64; 8], [f64; 8], [u64; 8]), PbError> {
    let n_leaves = 1usize << depth.min(3);
    let mut g = [0.0_f64; 8];
    let mut h = [0.0_f64; 8];
    let mut c = [0u64; 8];
    for &r in rows {
        let ru = r as usize;
        let leaf = usize::from(*leaf_of_row.get(ru).ok_or_else(internal("ps leaf map"))?);
        if leaf >= n_leaves {
            return Err(PbError::Internal {
                what: "leaf id out of range in leaf_aggregates".into(),
            });
        }
        *g.get_mut(leaf).ok_or_else(internal("ps g"))? +=
            f64::from(*gh.g.get(ru).ok_or_else(internal("ps gh.g"))?);
        *h.get_mut(leaf).ok_or_else(internal("ps h"))? +=
            f64::from(*gh.h.get(ru).ok_or_else(internal("ps gh.h"))?);
        *c.get_mut(leaf).ok_or_else(internal("ps c"))? += 1;
    }
    Ok((g, h, c))
}

/// Shrink each leaf toward its oblivious-tree parent node (§07.6 `path_smooth`):
/// `s[node] = (v[node]·n + s[parent]·ps) / (n + ps)`, recursively from the root (whose
/// parent contributes `s = 0`). Internal-node values `v` are the raw lr-scaled Newton step
/// on that node's rows; the depth-level (leaf) `v` are the (already monotone-clamped)
/// `leaves` passed in, so smoothing composes with the clamp. A node at level `L` is
/// identified by the first `L` split bits, and its parent drops the bit added at level
/// `L-1`. No-op when `ps <= 0`. VALUE-LEVEL on a fixed structure — depth, the ≤3-feature
/// support, and I2 are untouched.
///
/// # Errors
/// [`PbError::Internal`] on an index escape; [`PbError::InvalidInput`] if a smoothed leaf
/// is not finite/representable as `f32`.
#[allow(clippy::too_many_arguments)]
fn apply_path_smooth(
    leaves: &mut [f32; 8],
    leaf_g: &[f64; 8],
    leaf_h: &[f64; 8],
    leaf_count: &[u64; 8],
    depth: usize,
    lambda: f64,
    l1_leaf: f64,
    lr: f64,
    max_delta_step: Option<f64>,
    path_smooth: f64,
) -> Result<(), PbError> {
    if !(path_smooth.is_finite() && path_smooth > 0.0) {
        return Ok(());
    }
    let depth = depth.min(3);
    let n_leaves = 1usize << depth;
    // Smoothed outputs of the previous (shallower) level, indexed by that level's node id.
    let mut s_parent: Vec<f64> = Vec::new();
    let mut parent_is_virtual = true; // level 0's parent is the virtual `s = 0` root.
    for level in 0..=depth {
        let n_nodes = 1usize << level;
        let node_mask = n_nodes - 1;
        let mut s_cur = vec![0.0_f64; n_nodes];
        for j in 0..n_nodes {
            // Aggregate this node's rows from the leaves beneath it.
            let mut g = 0.0_f64;
            let mut h = 0.0_f64;
            let mut cnt = 0u64;
            for leaf in 0..n_leaves {
                if leaf & node_mask == j {
                    g += *leaf_g.get(leaf).ok_or_else(internal("ps node g"))?;
                    h += *leaf_h.get(leaf).ok_or_else(internal("ps node h"))?;
                    cnt += *leaf_count.get(leaf).ok_or_else(internal("ps node c"))?;
                }
            }
            let v = if level == depth {
                f64::from(*leaves.get(j).ok_or_else(internal("ps leaf v"))?)
            } else {
                newton_leaf(g, h, lambda, l1_leaf, lr, max_delta_step)
            };
            let parent_s = if parent_is_virtual {
                0.0
            } else {
                *s_parent
                    .get(j & (node_mask >> 1))
                    .ok_or_else(internal("ps parent s"))?
            };
            let n = cnt as f64;
            *s_cur.get_mut(j).ok_or_else(internal("ps s_cur"))? =
                (v * n + parent_s * path_smooth) / (n + path_smooth);
        }
        s_parent = s_cur;
        parent_is_virtual = false;
    }
    // `s_parent` now holds the depth-level (leaf) smoothed values.
    for j in 0..n_leaves {
        let val = *s_parent.get(j).ok_or_else(internal("ps out"))?;
        if !val.is_finite() || val < f64::from(f32::MIN) || val > f64::from(f32::MAX) {
            return Err(PbError::InvalidInput {
                what: "path_smooth leaf value is not finite/representable as f32".into(),
            });
        }
        *leaves.get_mut(j).ok_or_else(internal("ps write"))? = val as f32;
    }
    Ok(())
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
        cfg.l1_leaf,
        cfg.lr,
        cfg.max_delta_step,
    )?;
    // Re-enforce monotonicity: the structure was chosen on the sampled subset, but these
    // leaves were just recomputed on the FULL rows and could invert a cousin pair.
    let depth = usize::from(tree.depth);
    clamp_monotone(&mut tree.leaves, &tree.splits, depth, cfg.monotone)?;
    // path_smooth applies to the FINAL (full-row) leaves too, after the clamp (§07.6).
    if cfg.credibility.path_smooth > 0.0 {
        let (lg, lh, lc) = leaf_aggregates(gh, rows, &leaf_of_row, depth)?;
        apply_path_smooth(
            &mut tree.leaves,
            &lg,
            &lh,
            &lc,
            depth,
            cfg.lambda,
            cfg.l1_leaf,
            cfg.lr,
            cfg.max_delta_step,
            f64::from(cfg.credibility.path_smooth),
        )?;
        clamp_monotone(&mut tree.leaves, &tree.splits, depth, cfg.monotone)?;
    }
    Ok(())
}

/// Build a level-`L` (`L >= 1`) FullF64 histogram via the **subtraction trick** instead of a full
/// build: accumulate only the SMALLER of each parent leaf's two children, then derive the larger by
/// subtracting from the retained level-`(L-1)` parent histogram (`prev_hist`, columns = `prev_admissible`
/// = A_{L-1}). It visits only ~half the rows. Every row's level-`L` leaf is already fixed in
/// `leaf_of_row` (bits `0..L-1`, values in `[0, 2^L)`) by the committed earlier splits, so the sibling
/// pairing — parent `p` in `[0, 2^(L-1))`, children `{p, p + 2^(L-1)}` — is known BEFORE this level's
/// split (no circular dependency). Accuracy moves only at ~1e-11 (g/h drift; count exact; wsum exact
/// under unit weights); determinism is preserved because `small_rows` is filtered in fixed row order
/// before the (chunk-deterministic) build. FullF64 only — the quantized path keeps full builds.
#[allow(clippy::too_many_arguments)]
fn build_subtracted_level(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    level: usize,
    admissible: &[u32],
    weight: &[f32],
    unit_weight: bool,
    prev_hist: &Hist,
    prev_admissible: &[u32],
) -> Result<Hist, PbError> {
    let half = 1usize << (level - 1);
    let n_leaves = 1usize << level;
    // Step A — per-(level-L)-leaf row counts via one fixed-order pass (deterministic).
    let mut child_count = vec![0u64; n_leaves];
    for &r in rows {
        let leaf = usize::from(
            *leaf_of_row
                .get(r as usize)
                .ok_or_else(internal("subtraction leaf_of_row row"))?,
        );
        let c = child_count
            .get_mut(leaf)
            .ok_or_else(internal("subtraction child_count leaf"))?;
        *c = c
            .checked_add(1)
            .ok_or_else(internal("subtraction count overflow"))?;
    }
    // Pairing: parent p split into children {p, p+half}; SMALLER = fewer rows (tie -> lower index).
    let mut pairing: Vec<(usize, usize, usize)> = Vec::with_capacity(half);
    let mut is_smaller = vec![false; n_leaves];
    for p in 0..half {
        let c0 = p;
        let c1 = p + half;
        let cnt0 = *child_count
            .get(c0)
            .ok_or_else(internal("subtraction cnt0"))?;
        let cnt1 = *child_count
            .get(c1)
            .ok_or_else(internal("subtraction cnt1"))?;
        let (sm, lg) = if cnt1 < cnt0 { (c1, c0) } else { (c0, c1) };
        pairing.push((p, sm, lg));
        *is_smaller
            .get_mut(sm)
            .ok_or_else(internal("subtraction is_smaller set"))? = true;
    }
    // Step B — build ONLY the smaller children, filtering `rows` in fixed order (chunk boundaries,
    // and therefore the fixed-order chunk reduction, stay intact ⇒ thread-count independent).
    let mut small_rows: Vec<u32> = Vec::new();
    small_rows
        .try_reserve(rows.len())
        .map_err(|_| PbError::Internal {
            what: "subtraction small_rows allocation failed".into(),
        })?;
    for &r in rows {
        let leaf = usize::from(
            *leaf_of_row
                .get(r as usize)
                .ok_or_else(internal("subtraction small leaf"))?,
        );
        if *is_smaller
            .get(leaf)
            .ok_or_else(internal("subtraction is_smaller read"))?
        {
            small_rows.push(r);
        }
    }
    let mut hist = build_histogram(
        x,
        gh,
        &small_rows,
        leaf_of_row,
        n_leaves,
        admissible,
        weight,
        unit_weight,
    )?;
    // Step C — axis-position map A_L -> A_{L-1} (total: A_L ⊆ A_{L-1}, append-only `used_raws`).
    let mut axis_map: Vec<usize> = Vec::with_capacity(admissible.len());
    for &a in admissible {
        let pos = prev_admissible
            .iter()
            .position(|&pa| pa == a)
            .ok_or_else(internal("subtraction axis absent from parent admissible"))?;
        axis_map.push(pos);
    }
    // Step D — fill the larger children by subtracting the smaller from the parent leaf.
    subtract_sibling_into(&mut hist, prev_hist, &pairing, &axis_map)?;
    Ok(hist)
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
pub(crate) fn grow_oblivious_tree_with_leaf_map(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    axes: &[u32],
    cfg: &GrowConfig<'_>,
    weight: &[f32],
) -> Result<Option<(ObliviousTree, Vec<u8>)>, PbError> {
    let n_rows = x.n_rows as usize;
    let mut leaf_of_row: Vec<u8> = Hist::try_zeroed_vec(n_rows, "leaf assignment")?;
    let mut splits: Vec<Split> = Vec::new();
    let mut split_signs: Vec<Option<MonoSign>> = Vec::new();
    let mut used_raws: smallvec::SmallVec<[u32; 3]> = smallvec::SmallVec::new();
    let mut used_axes: smallvec::SmallVec<[u32; 3]> = smallvec::SmallVec::new();
    let order_cap = usize::from(cfg.max_order).min(3);
    // Retained level-1 (FullF64) histogram + its axis ids, so level 2 can build by subtraction
    // (the histogram-subtraction trick) instead of a full pass. Captured only on the FullF64 path.
    let mut prev_hist: Option<Hist> = None;
    let mut prev_admissible: Option<Vec<u32>> = None;

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
        let ranking_penalties = match cfg.table_budget_penalty {
            Some(penalty) => Some(
                admissible
                    .iter()
                    .map(|&axis| penalty.multiplier(x, &used_axes, axis))
                    .collect::<Result<Vec<_>, PbError>>()?,
            ),
            None => None,
        };

        let n_leaves = 1usize << level;
        let hist =
            crate::engine::boost::prof::timed("grow.hist_build", || -> Result<Hist, PbError> {
                // Histogram-subtraction fast path: levels 1 and 2, FullF64, with the previous level's
                // histogram retained as the parent. Builds the larger sibling-children by subtraction
                // (visits ~half the rows). FullF64 only (quantized keeps full builds). Level 2's parent
                // is itself a subtracted hist, so g/h drift compounds to ~2e-11 — still accuracy-neutral
                // (the equivalence test grows the same tree as the full build).
                if cfg.hist_subtraction
                    && level >= 1
                    && matches!(cfg.hist_precision, HistPrecision::FullF64)
                {
                    if let (Some(ph), Some(pa)) = (prev_hist.as_ref(), prev_admissible.as_ref()) {
                        return build_subtracted_level(
                            x,
                            gh,
                            rows,
                            &leaf_of_row,
                            level,
                            &admissible,
                            weight,
                            cfg.unit_weight,
                            ph,
                            pa,
                        );
                    }
                }
                match cfg.hist_precision {
                    HistPrecision::FullF64 => build_histogram(
                        x,
                        gh,
                        rows,
                        &leaf_of_row,
                        n_leaves,
                        &admissible,
                        weight,
                        cfg.unit_weight,
                    ),
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
                        weight,
                    ),
                }
            })?;
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
            l1_leaf: cfg.l1_leaf,
            max_delta_step: cfg.max_delta_step,
        });
        let cand = match crate::engine::boost::prof::timed("grow.split_find", || {
            best_level_split(
                &hist,
                &admissible,
                &n_data_bins,
                cfg.lambda,
                cfg.l1_leaf,
                cfg.max_delta_step,
                cfg.min_split_gain,
                RankingContext {
                    monotone: monotone_scan,
                    noise: SplitNoise::new(cfg.quant_seed, cfg.round, level, cfg.random_strength),
                    table_penalties: ranking_penalties.as_deref(),
                },
                &cfg.credibility,
            )
        })? {
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
        used_axes.push(cand.axis);

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

        // Retain this level's FullF64 histogram + axis ids as the parent for the NEXT level's
        // subtraction (level 0 → parent for level 1; level 1 → parent for level 2). Level 2 is the
        // last level, so its histogram is never retained. (Quantized stays on full builds.)
        if level < 2 && matches!(cfg.hist_precision, HistPrecision::FullF64) {
            prev_admissible = Some(admissible.clone());
            prev_hist = Some(hist);
        }
    }

    if splits.is_empty() {
        return Ok(None);
    }
    let depth = splits.len();
    let mut leaves = leaf_values(
        gh,
        rows,
        &leaf_of_row,
        depth,
        cfg.lambda,
        cfg.l1_leaf,
        cfg.lr,
        cfg.max_delta_step,
    )?;
    // Project onto the monotone cone — a no-op when the structure was grown feasible, but
    // a guard against quantized-histogram round-off inverting a cousin pair (§07.5).
    clamp_monotone(&mut leaves, &splits, depth, cfg.monotone)?;
    // §07.6 path_smooth: shrink each leaf toward its oblivious-tree parent, AFTER the
    // monotone clamp, then re-clamp so smoothing cannot cross a monotone bound.
    if cfg.credibility.path_smooth > 0.0 {
        let (lg, lh, lc) = leaf_aggregates(gh, rows, &leaf_of_row, depth)?;
        apply_path_smooth(
            &mut leaves,
            &lg,
            &lh,
            &lc,
            depth,
            cfg.lambda,
            cfg.l1_leaf,
            cfg.lr,
            cfg.max_delta_step,
            f64::from(cfg.credibility.path_smooth),
        )?;
        clamp_monotone(&mut leaves, &splits, depth, cfg.monotone)?;
    }
    // `leaf_of_row[r]` (set for every r in `rows` via the SAME canonical `low_bit` used by the
    // tree walk) is the per-row leaf partition the leaf-refine line search needs — returned so it
    // can be reused instead of re-walking the tree (byte-identical, cf. `gather_memberships`).
    let tree = ObliviousTree::try_new(splits, leaves, &x.provenance)?;
    Ok(Some((tree, leaf_of_row)))
}

/// Test-only thin wrapper returning just the grown tree. Production calls
/// [`grow_oblivious_tree_with_leaf_map`] and reuses the per-row leaf map to skip the
/// leaf-refinement tree re-walk; unit tests that only assert on tree structure use this.
#[cfg(test)]
pub(crate) fn grow_oblivious_tree(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    axes: &[u32],
    cfg: &GrowConfig<'_>,
    weight: &[f32],
) -> Result<Option<ObliviousTree>, PbError> {
    Ok(grow_oblivious_tree_with_leaf_map(x, gh, rows, axes, cfg, weight)?.map(|(tree, _)| tree))
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

    #[test]
    fn l1_leaf_soft_thresholds_leaf_values_and_gain() {
        let gh = gradhess(&[-0.5, 0.5], &[1.0, 1.0]);
        let rows = [0u32, 1];
        let leaf_of_row = [0u8, 1];
        let no_l1 = leaf_values(&gh, &rows, &leaf_of_row, 1, 0.0, 0.0, 1.0, None).unwrap();
        let l1 = leaf_values(&gh, &rows, &leaf_of_row, 1, 0.0, 1.0, 1.0, None).unwrap();
        assert!(no_l1[0] > 0.0);
        assert!(no_l1[1] < 0.0);
        assert_eq!(l1[0], 0.0);
        assert_eq!(l1[1], 0.0);

        let mut hist = Hist::try_zeros(1, 1, 3).unwrap();
        let left = hist.offset(0, 0, 1).unwrap();
        hist.g[left] = -0.5;
        hist.h[left] = 1.0;
        hist.count[left] = 1;
        let right = hist.offset(0, 0, 2).unwrap();
        hist.g[right] = 0.5;
        hist.h[right] = 1.0;
        hist.count[right] = 1;
        assert!(best_level_split(
            &hist,
            &[0],
            &[2],
            0.0,
            0.0,
            None,
            0.0,
            RankingContext::default(),
            &CredibilityFloor::default(),
        )
        .unwrap()
        .is_some());
        assert!(best_level_split(
            &hist,
            &[0],
            &[2],
            0.0,
            1.0,
            None,
            0.0,
            RankingContext::default(),
            &CredibilityFloor::default(),
        )
        .unwrap()
        .is_none());
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

        let best = best_level_split(
            &hist,
            &[0],
            &[2],
            1.0,
            0.0,
            None,
            0.0,
            RankingContext::default(),
            &CredibilityFloor::default(),
        )
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
    fn random_strength_does_not_rewrite_raw_gain() {
        // Same closed-form fixture as above, but with ranking noise enabled. Either
        // missing direction may win; the stored gain must remain that candidate's raw
        // Newton gain rather than the noisy score.
        let mut hist = Hist::try_zeros(1, 1, 3).unwrap();
        let set = |hist: &mut Hist, bin: usize, g: f64, h: f64| {
            let o = hist.offset(0, 0, bin).unwrap();
            hist.g[o] = g;
            hist.h[o] = h;
            hist.count[o] = 1;
        };
        set(&mut hist, 0, 2.0, 1.0);
        set(&mut hist, 1, 4.0, 2.0);
        set(&mut hist, 2, -6.0, 3.0);

        let noise = SplitNoise::new(123, 7, 0, 100.0);
        let best = best_level_split(
            &hist,
            &[0],
            &[2],
            1.0,
            0.0,
            None,
            0.0,
            RankingContext {
                noise,
                ..RankingContext::default()
            },
            &CredibilityFloor::default(),
        )
        .unwrap()
        .unwrap();
        let expected_raw = if best.missing_left {
            9.0
        } else {
            0.5 * (16.0 / 3.0 + 16.0 / 5.0)
        };
        assert!(
            (best.gain - expected_raw).abs() < 1e-9,
            "gain {} != raw {expected_raw}",
            best.gain
        );
    }

    #[test]
    fn table_budget_penalty_is_soft_and_preserves_raw_gain() {
        // Two one-threshold axes. Axis 0 has larger raw Newton gain (~10) but is
        // assigned an expensive-support multiplier; axis 1 has smaller raw gain (~9)
        // and wins only after the ranking prior. The stored gain remains axis 1's
        // raw Newton gain.
        let mut hist = Hist::try_zeros(1, 2, 3).unwrap();
        let set_axis = |hist: &mut Hist, axis: usize, a: f64| {
            let l = hist.offset(0, axis, 1).unwrap();
            let r = hist.offset(0, axis, 2).unwrap();
            hist.g[l] = a;
            hist.h[l] = 1.0;
            hist.count[l] = 1;
            hist.g[r] = -a;
            hist.h[r] = 1.0;
            hist.count[r] = 1;
        };
        set_axis(&mut hist, 0, (20.0_f64).sqrt());
        set_axis(&mut hist, 1, (18.0_f64).sqrt());

        let raw_best = best_level_split(
            &hist,
            &[0, 1],
            &[2, 2],
            1.0,
            0.0,
            None,
            0.0,
            RankingContext::default(),
            &CredibilityFloor::default(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(raw_best.axis, 0);
        assert!((raw_best.gain - 10.0).abs() < 1.0e-9);

        let penalized = best_level_split(
            &hist,
            &[0, 1],
            &[2, 2],
            1.0,
            0.0,
            None,
            0.0,
            RankingContext {
                table_penalties: Some(&[0.5, 1.0]),
                ..RankingContext::default()
            },
            &CredibilityFloor::default(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(penalized.axis, 1);
        assert!((penalized.gain - 9.0).abs() < 1.0e-9);
    }

    #[test]
    fn max_delta_step_is_reflected_in_split_gain() {
        let mut hist = Hist::try_zeros(1, 2, 3).unwrap();
        let set_axis = |hist: &mut Hist, axis: usize, g: f64, h: f64| {
            let l = hist.offset(0, axis, 1).unwrap();
            let r = hist.offset(0, axis, 2).unwrap();
            hist.g[l] = -g;
            hist.h[l] = h;
            hist.count[l] = 1;
            hist.g[r] = g;
            hist.h[r] = h;
            hist.count[r] = 1;
        };
        set_axis(&mut hist, 0, 100.0, 1.0);
        set_axis(&mut hist, 1, 150.0, 1000.0);

        let unclamped = best_level_split(
            &hist,
            &[0, 1],
            &[2, 2],
            0.0,
            0.0,
            None,
            0.0,
            RankingContext::default(),
            &CredibilityFloor::default(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(unclamped.axis, 0);

        let clamped = best_level_split(
            &hist,
            &[0, 1],
            &[2, 2],
            0.0,
            0.0,
            Some(0.1),
            0.0,
            RankingContext::default(),
            &CredibilityFloor::default(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(clamped.axis, 1);
        assert!((clamped.gain - 20.0).abs() < 1.0e-9);
    }

    #[test]
    fn table_budget_beta_zero_is_exactly_inert() {
        assert_eq!(TableBudgetPenalty::new(0.0, 1), None);
        let x = matrix(vec![vec![1u8, 2]], &[3]);
        let penalty = TableBudgetPenalty::new(0.5, 3).unwrap();
        let no_over_budget = penalty.multiplier(&x, &[], 0).unwrap();
        assert_eq!(no_over_budget, 1.0);
    }

    #[test]
    fn below_min_split_gain_yields_no_candidate() {
        let mut hist = Hist::try_zeros(1, 1, 3).unwrap();
        let o = hist.offset(0, 0, 1).unwrap();
        hist.g[o] = 1.0;
        hist.h[o] = 1.0;
        // Single populated bin ⇒ any split is degenerate (gain 0); a positive floor rejects it.
        assert!(best_level_split(
            &hist,
            &[0],
            &[2],
            1.0,
            0.0,
            None,
            1.0,
            RankingContext::default(),
            &CredibilityFloor::default()
        )
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
        let leaves = leaf_values(&gh, &rows, &leaf_of_row, 1, 1.0, 0.0, 0.1, None).unwrap();
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
        let uncapped = leaf_values(&gh, &rows, &leaf_of_row, 0, 0.0, 0.0, 0.1, None).unwrap();
        assert!(
            uncapped[0] > 100.0,
            "uncapped leaf should be huge, got {}",
            uncapped[0]
        );
        let capped = leaf_values(&gh, &rows, &leaf_of_row, 0, 0.0, 0.0, 0.1, Some(0.5)).unwrap();
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
            leaf_values(&gh, &rows, &leaf_of_row, 0, 0.0, 0.0, 1.0, None),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn clamp_monotone_projects_inverted_leaves_onto_the_cone() {
        // depth-2, both levels Increasing. leaf idx = bit0 | bit1<<1; bit=1 ⇒ LOW feature
        // value (must have the LOWER response for Increasing). Start fully inverted.
        let splits = vec![
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
        // idx 0=(hi0,hi1) 1=(lo0,hi1) 2=(hi0,lo1) 3=(lo0,lo1): low-value leaves set HIGH.
        let mut leaves = [0.0_f32, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0];
        let signs = [Some(MonoSign::Increasing), Some(MonoSign::Increasing)];
        clamp_monotone(&mut leaves, &splits, 2, Some(&signs)).unwrap();
        // Level 0 cousins (differ in bit0): w[bit0=1] <= w[bit0=0].
        assert!(leaves[1] <= leaves[0] + 1e-6);
        assert!(leaves[3] <= leaves[2] + 1e-6);
        // Level 1 cousins (differ in bit1): w[bit1=1] <= w[bit1=0].
        assert!(leaves[2] <= leaves[0] + 1e-6);
        assert!(leaves[3] <= leaves[1] + 1e-6);
        // Decreasing flips the required direction.
        let mut dec = [0.0_f32, -10.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0];
        let dsigns = [Some(MonoSign::Decreasing), None];
        clamp_monotone(&mut dec, &splits, 2, Some(&dsigns)).unwrap();
        assert!(dec[1] >= dec[0] - 1e-6); // Decreasing: low-value leaf >= high-value leaf
        assert!(dec[3] >= dec[2] - 1e-6);
        // No signs ⇒ untouched.
        let mut none = [5.0_f32, 1.0, 9.0, 2.0, 0.0, 0.0, 0.0, 0.0];
        let before = none;
        clamp_monotone(&mut none, &splits, 2, None).unwrap();
        assert_eq!(none, before);
    }

    /// A clean depth-2 fixture: two informative features ⇒ the engine grows a depth-2
    /// tree on two DISTINCT raw features (I1), with no early termination.
    fn cfg(lambda: f64, lr: f64, min_split_gain: f64, max_order: u8) -> GrowConfig<'static> {
        GrowConfig {
            lambda,
            l1_leaf: 0.0,
            lr,
            min_split_gain,
            max_order,
            max_delta_step: None,
            hist_precision: HistPrecision::FullF64,
            quant_seed: 0,
            round: 0,
            random_strength: 0.0,
            groups: None,
            monotone: None,
            table_budget_penalty: None,
            credibility: CredibilityFloor::default(),
            // Slow Σw path in grow tests (always correct for any weights); the dedicated
            // `unit_weight_*` hist tests pin the fast path's bit-identity directly.
            unit_weight: false,
            // Subtraction ON by default in grow tests too (matches production); the
            // equivalence tests flip it off to get the full-build reference.
            hist_subtraction: true,
        }
    }

    /// Unit weights of length `n` for `grow_oblivious_tree` calls that don't exercise
    /// the `min_weight_sum_in_leaf` floor.
    fn ones(n: usize) -> Vec<f32> {
        vec![1.0_f32; n]
    }

    /// A `GrowConfig` like `cfg(1.0, 0.1, 0.0, 3)` but carrying a credibility floor.
    fn cfg_cred(floor: CredibilityFloor) -> GrowConfig<'static> {
        GrowConfig {
            credibility: floor,
            ..cfg(1.0, 0.1, 0.0, 3)
        }
    }

    #[test]
    fn credibility_floors_reject_under_supported_children() {
        // 1 leaf, 1 axis, 3 bins. The only candidate (bin_le=1) splits into bin1 (well
        // supported) and bin2 (count 2, Σh 2, Σw 2) — a thin cell each floor can veto.
        let mut hist = Hist::try_zeros(1, 1, 3).unwrap();
        let set = |hist: &mut Hist, bin: usize, g: f64, h: f64, c: u32, w: f64| {
            let o = hist.offset(0, 0, bin).unwrap();
            hist.g[o] = g;
            hist.h[o] = h;
            hist.count[o] = c;
            hist.wsum[o] = w;
        };
        set(&mut hist, 0, 0.0, 0.0, 0, 0.0); // missing: empty
        set(&mut hist, 1, -5.0, 10.0, 10, 10.0);
        set(&mut hist, 2, 5.0, 2.0, 2, 2.0);
        let split = |floor: &CredibilityFloor| {
            best_level_split(
                &hist,
                &[0],
                &[2],
                1.0,
                0.0,
                None,
                0.0,
                RankingContext::default(),
                floor,
            )
            .unwrap()
        };
        // No floor ⇒ the informative split is found.
        assert!(split(&CredibilityFloor::default()).is_some());
        // Each floor independently vetoes the under-supported bin-2 child ⇒ no candidate.
        assert!(split(&CredibilityFloor {
            min_data_in_leaf: 5,
            ..CredibilityFloor::default()
        })
        .is_none());
        assert!(split(&CredibilityFloor {
            min_sum_hessian_in_leaf: 3.0,
            ..CredibilityFloor::default()
        })
        .is_none());
        assert!(split(&CredibilityFloor {
            min_weight_sum_in_leaf: 5.0,
            ..CredibilityFloor::default()
        })
        .is_none());
        // A floor at or below the thin cell's support still admits the split.
        assert!(split(&CredibilityFloor {
            min_data_in_leaf: 2,
            min_sum_hessian_in_leaf: 2.0,
            min_weight_sum_in_leaf: 2.0,
            ..CredibilityFloor::default()
        })
        .is_some());
    }

    #[test]
    fn path_smooth_shrinks_leaves_toward_parent_node() {
        // depth 1: leaf0 raw value +1, leaf1 raw value −1, each n=10; root value 0.
        // λ=0, lr=1 ⇒ newton_leaf(g,h) = −g/h matches the leaf values below.
        let lg = [-10.0_f64, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let lh = [10.0_f64, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let lc = [10u64, 10, 0, 0, 0, 0, 0, 0];
        let base = [1.0_f32, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // ps = 0 ⇒ exact no-op.
        let mut a = base;
        apply_path_smooth(&mut a, &lg, &lh, &lc, 1, 0.0, 0.0, 1.0, None, 0.0).unwrap();
        assert_eq!(a, base);
        // ps = 2 ⇒ each leaf shrinks toward the root (0) by the factor n/(n+ps)=10/12.
        let mut s = base;
        apply_path_smooth(&mut s, &lg, &lh, &lc, 1, 0.0, 0.0, 1.0, None, 2.0).unwrap();
        assert!((s[0] - 10.0 / 12.0).abs() < 1e-6);
        assert!((s[1] + 10.0 / 12.0).abs() < 1e-6);
    }

    #[test]
    fn grow_with_large_path_smooth_collapses_leaf_spread() {
        // Distinct per-quadrant gradients ⇒ an informative depth-2 tree.
        let c0: Vec<u8> = vec![1, 1, 1, 1, 2, 2, 2, 2];
        let c1: Vec<u8> = vec![1, 1, 2, 2, 1, 1, 2, 2];
        let x = matrix(vec![c0, c1], &[3, 3]);
        let g = [-3.0_f32, -3.0, 1.0, 1.0, 2.0, 2.0, 6.0, 6.0];
        let gh = gradhess(&g, &[1.0; 8]);
        let rows: Vec<u32> = (0..8).collect();
        let w = ones(x.n_rows as usize);
        let base = grow_oblivious_tree(&x, &gh, &rows, &[0, 1], &cfg(1.0, 0.1, 0.0, 3), &w)
            .unwrap()
            .expect("a tree");
        let smooth = grow_oblivious_tree(
            &x,
            &gh,
            &rows,
            &[0, 1],
            &cfg_cred(CredibilityFloor {
                path_smooth: 1e6,
                ..CredibilityFloor::default()
            }),
            &w,
        )
        .unwrap()
        .expect("a tree");
        // path_smooth is value-level: identical structure, but the leaves collapse toward
        // the (near-zero) parent/root under heavy smoothing.
        assert_eq!(base.splits, smooth.splits);
        let spread = |t: &ObliviousTree| {
            let lo = t.leaves.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = t.leaves.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            hi - lo
        };
        assert!(spread(&base) > 1e-3, "base tree must be informative");
        assert!(
            spread(&smooth) < spread(&base),
            "heavy path_smooth must compress the leaf spread: base {} vs smooth {}",
            spread(&base),
            spread(&smooth)
        );
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
        let tree = grow_oblivious_tree(
            &x,
            &gh,
            &rows,
            &[0, 1],
            &cfg(1.0, 1.0, 0.0, 3),
            &ones(x.n_rows as usize),
        )
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
                grow_oblivious_tree(
                    &x,
                    &gh,
                    &rows,
                    &[0, 1, 2],
                    &cfg(1.0, 0.1, 0.0, 3),
                    &ones(x.n_rows as usize),
                )
                .unwrap()
                .unwrap()
            })
        };
        let a = run(1);
        assert_eq!(a, run(2));
        assert_eq!(a, run(8));
    }

    #[test]
    fn grow_with_random_strength_is_deterministic_across_thread_counts() {
        let n = 360usize;
        let c0: Vec<u8> = (0..n).map(|i| u8::try_from(i % 6 + 1).unwrap()).collect();
        let c1: Vec<u8> = (0..n).map(|i| u8::try_from(i % 5 + 1).unwrap()).collect();
        let c2: Vec<u8> = (0..n).map(|i| u8::try_from(i % 4 + 1).unwrap()).collect();
        let x = matrix(vec![c0, c1, c2], &[7, 6, 5]);
        let g: Vec<f32> = (0..n).map(|i| (i as f32 % 13.0) - 6.0).collect();
        let gh = gradhess(&g, &vec![1.0; n]);
        let rows: Vec<u32> = (0..n as u32).collect();
        let mut cfg = cfg(1.0, 0.1, 0.0, 3);
        cfg.quant_seed = 99;
        cfg.round = 5;
        cfg.random_strength = 0.75;
        let run = |nt: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                grow_oblivious_tree(&x, &gh, &rows, &[0, 1, 2], &cfg, &ones(x.n_rows as usize))
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
        let tree = grow_oblivious_tree(
            &x,
            &gh,
            &rows,
            &[0],
            &cfg(1.0, 0.1, 1e-9, 1),
            &ones(x.n_rows as usize),
        )
        .unwrap();
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
        let tree = grow_oblivious_tree(
            &x,
            &gh,
            &rows,
            &[0, 1],
            &cfg(1.0, 0.1, 0.0, 1),
            &ones(x.n_rows as usize),
        )
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
        assert!(
            grow_oblivious_tree(&x, &gh, &rows, &[0, 1], &denied, &ones(x.n_rows as usize))
                .unwrap()
                .is_none()
        );

        let allowed_groups = vec![FeatureSet::new(&[1])];
        let mut allowed = cfg(1.0, 0.1, 0.0, 2);
        allowed.groups = Some(&allowed_groups);
        let tree = grow_oblivious_tree(&x, &gh, &rows, &[0, 1], &allowed, &ones(x.n_rows as usize))
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
        let best = best_level_split(
            &hist,
            &[0],
            &[2],
            1.0,
            0.0,
            None,
            0.0,
            RankingContext::default(),
            &CredibilityFloor::default(),
        )
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
        let tree = grow_oblivious_tree(
            &x,
            &gh,
            &rows,
            &[0, 1, 2],
            &cfg(0.0, 0.1, 0.0, 3),
            &ones(x.n_rows as usize),
        )
        .unwrap()
        .expect("a depth-3 tree should grow");
        assert_eq!(tree.depth, 3, "all three features are informative");
        let mut raws: Vec<u32> = tree.splits.iter().map(|s| s.axis).collect();
        raws.sort_unstable();
        raws.dedup();
        assert_eq!(raws, vec![0, 1, 2]); // each level a distinct raw feature (I1)
    }

    /// A non-trivial depth-3 fixture for the histogram-subtraction tests: the 8-cell
    /// 3-binary-feature design replicated with deterministic per-row gradient noise (so the
    /// level-2 histograms carry real, varied mass), plus a weak 4th feature. The grower uses
    /// features 0,1,2 (coeffs 4,2,1), so at level 2 A_2 = {2,3} ⊂ A_1 = {1,2,3} with SHIFTED
    /// positions — exercising the axis-position remap.
    fn subtraction_fixture() -> (BinnedMatrix, GradHess, Vec<u32>) {
        let base0 = [1u8, 1, 1, 1, 2, 2, 2, 2];
        let base1 = [1u8, 1, 2, 2, 1, 1, 2, 2];
        let base2 = [1u8, 2, 1, 2, 1, 2, 1, 2];
        let reps = 80usize;
        let (mut c0, mut c1, mut c2, mut c3) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for r in 0..reps {
            for k in 0..8usize {
                c0.push(base0[k]);
                c1.push(base1[k]);
                c2.push(base2[k]);
                c3.push(1 + ((r * 8 + k) * 7 + 3) as u8 % 3); // weak 4-bin decoy in [1,3]
            }
        }
        let x = matrix(vec![c0.clone(), c1.clone(), c2.clone(), c3], &[3, 3, 3, 4]);
        let n = c0.len();
        let g: Vec<f32> = (0..n)
            .map(|i| {
                let b0 = f32::from(c0[i] - 1);
                let b1 = f32::from(c1[i] - 1);
                let b2 = f32::from(c2[i] - 1);
                4.0 * b0 + 2.0 * b1 + b2 - 3.5 + 0.05 * (((i * 13 + 7) % 11) as f32 - 5.0)
            })
            .collect();
        let gh = gradhess(&g, &vec![1.0_f32; n]);
        (x, gh, (0..n as u32).collect())
    }

    #[test]
    fn level2_subtraction_reproduces_full_build_tree() {
        // THE de-risking gate: grow the SAME data with the level-2 histogram-subtraction path
        // ON (default) and OFF (full build), assert the trees are byte-identical (splits, leaf
        // bits, depth). Subtraction perturbs g/h only at ~1e-11 — far below any non-tie split
        // gap — so the selected splits, and the gh-derived leaves, match exactly.
        let (x, gh, rows) = subtraction_fixture();
        let w = ones(x.n_rows as usize);
        let axes = [0u32, 1, 2, 3];
        let on = cfg(0.0, 0.1, 0.0, 3); // hist_subtraction = true
        let off = GrowConfig {
            hist_subtraction: false,
            ..cfg(0.0, 0.1, 0.0, 3)
        };
        let t_on = grow_oblivious_tree(&x, &gh, &rows, &axes, &on, &w)
            .unwrap()
            .expect("tree");
        let t_off = grow_oblivious_tree(&x, &gh, &rows, &axes, &off, &w)
            .unwrap()
            .expect("tree");
        assert!(
            t_on.depth >= 2,
            "fixture must reach depth 2 so the level-2 subtraction path engages (got {})",
            t_on.depth
        );
        assert_eq!(
            t_on, t_off,
            "level-2 subtraction must reproduce the full-build tree byte-for-byte"
        );
    }

    #[test]
    fn level2_subtraction_is_thread_count_independent() {
        // Subtraction must keep the §1 determinism GATE: same tree across 1/2/8 threads.
        let (x, gh, rows) = subtraction_fixture();
        let w = ones(x.n_rows as usize);
        let axes = [0u32, 1, 2, 3];
        let grow = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                grow_oblivious_tree(&x, &gh, &rows, &axes, &cfg(0.0, 0.1, 0.0, 3), &w)
                    .unwrap()
                    .expect("tree")
            })
        };
        let base = grow(1);
        assert_eq!(base, grow(2), "subtraction tree differs at 2 threads");
        assert_eq!(base, grow(8), "subtraction tree differs at 8 threads");
    }

    #[test]
    fn level2_subtraction_flag_off_matches_full_build_and_quantized_unaffected() {
        // The flag is inert for QuantizedI32 (always full build): a quantized grow is identical
        // whether the subtraction flag is on or off.
        let (x, gh, rows) = subtraction_fixture();
        let w = ones(x.n_rows as usize);
        let axes = [0u32, 1, 2, 3];
        let q_on = GrowConfig {
            hist_precision: HistPrecision::QuantizedI32,
            hist_subtraction: true,
            ..cfg(0.0, 0.1, 0.0, 3)
        };
        let q_off = GrowConfig {
            hist_subtraction: false,
            ..q_on.clone()
        };
        let t_on = grow_oblivious_tree(&x, &gh, &rows, &axes, &q_on, &w)
            .unwrap()
            .expect("tree");
        let t_off = grow_oblivious_tree(&x, &gh, &rows, &axes, &q_off, &w)
            .unwrap()
            .expect("tree");
        assert_eq!(
            t_on, t_off,
            "subtraction flag must be inert on the quantized path"
        );
    }

    #[test]
    fn grown_tree_passes_check_feature_budget() {
        // Run the NAMED I1 gate fn (explain::check_feature_budget) on a fitted tree.
        let c0 = vec![1u8, 1, 2, 2];
        let c1 = vec![1u8, 2, 1, 2];
        let x = matrix(vec![c0, c1], &[3, 3]);
        let gh = gradhess(&[-2.0, -1.0, 1.0, 2.0], &[1.0; 4]);
        let rows = [0u32, 1, 2, 3];
        let tree = grow_oblivious_tree(
            &x,
            &gh,
            &rows,
            &[0, 1],
            &cfg(1.0, 0.1, 0.0, 3),
            &ones(x.n_rows as usize),
        )
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
            let res = grow_oblivious_tree(&x, &gh, &rows, &axes, &cfg(1.0, 0.1, 0.0, 3), &ones(x.n_rows as usize));
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
