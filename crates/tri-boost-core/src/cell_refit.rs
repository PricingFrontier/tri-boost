//! §G1 fully-corrective cell-basis refit: a per-term adaptive ridge on the purified
//! fANOVA *cell* basis, fit to the bagged out-of-bag residual.
//!
//! tri grows the right supports but a shared greedy budget under-fits weak interaction
//! *surfaces* (the depth-2 pair-fitting gap vs EBM). This module re-fits the cell values
//! of the realized mains + pairs jointly — a totally-corrective step on the basis the
//! decomposition actually lives in (cells, not leaves) — with an adaptive ridge that gives
//! high-signal terms more freedom. The solution is returned as a [`CorrectionBank`] of raw
//! per-merged-cell deltas; attaching it to the model keeps G0 exact (see
//! [`crate::engine::Model::correction_delta`] and the decompose-side fold in
//! [`crate::explain`]).
//!
//! The design column count can be large (one column per merged cell, summed over every
//! corrected support), so the joint ridge is solved with a diagonally-preconditioned
//! conjugate gradient that never materialises `XᵀX` — the existing dense refit
//! (`fully_corrective_refit`) would be infeasible at cell width.

use crate::engine::{CorrectionBank, Model};
use crate::error::PbError;
use crate::explain::correction_scaffold;

/// Hyper-parameters for the cell-basis refit. Suite-wide (no per-dataset tuning).
#[derive(Debug, Clone, Copy)]
pub struct CellRefitSpec {
    /// Base ridge penalty `α` on the cell coefficients.
    pub base: f64,
    /// Adaptive exponent `γ`: per-term penalty is `base / (s_term^γ normalised)²`, so
    /// high-signal terms (large `s_term`) are penalised less. `γ = 0` is a flat ridge.
    pub gamma: f64,
    /// Maximum coarse cells per axis for ORDER-2 (pair) corrections; mains keep full merged
    /// resolution. The full merged pair grid (realized borders of both features) over-fits
    /// noisy high-cardinality surfaces — acute for categoricals — so pairs are solved on a
    /// coarsened grid (contiguous groups of merged cells), matching the prototype's coarse
    /// pair grids. The δ stays constant WITHIN merged cells, so the correction is still
    /// stored at merged resolution and G0 is untouched.
    pub pair_cell_cap: u32,
    /// Maximum conjugate-gradient iterations per solve.
    pub cg_iters: usize,
    /// Conjugate-gradient relative residual tolerance (‖r‖/‖b‖).
    pub cg_tol: f64,
}

impl Default for CellRefitSpec {
    fn default() -> Self {
        // base≈4000, γ≈2 — the held-out-validated prototype plateau; pair grid capped at the
        // prototype's 24 cells/axis.
        CellRefitSpec {
            base: 4000.0,
            gamma: 2.0,
            pair_cell_cap: 24,
            cg_iters: 500,
            cg_tol: 1e-7,
        }
    }
}

/// The sparse cell-indicator design, stored implicitly: every row activates exactly one
/// COARSE cell per corrected support, so we keep only the global column id per (support,
/// row). Columns are at the (possibly coarsened) solve resolution; the solved δ is expanded
/// back to merged resolution for storage, so the design need only carry the coarse layout.
struct Design {
    /// `active[u][r]` = global coarse column id activated by row `r` on support `u`.
    active: Vec<Vec<u32>>,
    /// Total coarse column count (Σ coarse cells over supports).
    n_cols: usize,
    /// `col_term[c]` = the support index owning coarse column `c` (for per-term reweighting).
    col_term: Vec<u32>,
    /// First global coarse column id of each support (prefix sum of per-support coarse cells).
    col_offset: Vec<u32>,
    /// `coarse_shape[u][k]` = coarse cell count of support `u`'s axis `k` (parallel to its axes).
    coarse_shape: Vec<Vec<u32>>,
    /// `groups[u][k][merged_cell]` = the coarse group id a merged cell maps to, for expanding
    /// the solved δ back to merged resolution when filling the [`CorrectionBank`].
    groups: Vec<Vec<Vec<u32>>>,
}

/// Map each of `merged_m` merged cells to one of `min(merged_m, cap)` contiguous coarse
/// groups (`cap` ≥ 1). Returns the per-merged-cell group ids and the coarse group count.
fn coarse_groups(merged_m: u32, cap: u32) -> (Vec<u32>, u32) {
    let k = merged_m.min(cap.max(1)).max(1);
    let groups: Vec<u32> = (0..merged_m)
        .map(|c| ((u64::from(c) * u64::from(k)) / u64::from(merged_m.max(1))) as u32)
        .collect();
    (groups, k)
}

impl Design {
    /// `y = X v` (length n_rows): sum the activated coefficients per row.
    fn matvec(&self, v: &[f64], out: &mut [f64]) {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        for active_u in &self.active {
            for (r, &c) in active_u.iter().enumerate() {
                out[r] += v[c as usize];
            }
        }
    }

    /// `z = Xᵀ y` (length n_cols): scatter each row's value to its activated columns.
    fn rmatvec(&self, y: &[f64], out: &mut [f64]) {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        for active_u in &self.active {
            for (r, &c) in active_u.iter().enumerate() {
                out[c as usize] += y[r];
            }
        }
    }
}

/// Build the implicit COARSE design for `bank`'s supports from the binned training columns.
/// `data[axis][row]` is the model bin id. For each support, each axis's merged cells are
/// grouped into ≤`pair_cell_cap` coarse cells for ORDER-2 supports (mains keep full merged
/// resolution); a row's active coarse column is the row-major fold of its per-axis
/// (model bin → merged cell → coarse group) maps, shifted by the support's coarse offset.
fn build_design(
    bank: &CorrectionBank,
    data: &[Vec<u8>],
    n_rows: usize,
    pair_cell_cap: u32,
) -> Result<Design, PbError> {
    let n_terms = bank.tables.len();
    let mut col_offset = Vec::with_capacity(n_terms);
    let mut coarse_shape: Vec<Vec<u32>> = Vec::with_capacity(n_terms);
    let mut groups: Vec<Vec<Vec<u32>>> = Vec::with_capacity(n_terms);
    let mut n_cols: usize = 0;
    for table in &bank.tables {
        let order = table.axes.len();
        let mut ks = Vec::with_capacity(order);
        let mut grps = Vec::with_capacity(order);
        let mut term_cols: usize = 1;
        for k in 0..order {
            let merged_m = table.shape[k];
            let cap = if order <= 1 { merged_m } else { pair_cell_cap };
            let (g, kk) = coarse_groups(merged_m, cap);
            ks.push(kk);
            grps.push(g);
            term_cols = term_cols
                .checked_mul(kk as usize)
                .ok_or_else(|| PbError::Internal {
                    what: "cell-refit coarse term column count overflow".into(),
                })?;
        }
        col_offset.push(u32::try_from(n_cols).map_err(|_| PbError::Internal {
            what: "cell-refit column offset exceeded u32".into(),
        })?);
        n_cols = n_cols
            .checked_add(term_cols)
            .ok_or_else(|| PbError::Internal {
                what: "cell-refit column count overflow".into(),
            })?;
        coarse_shape.push(ks);
        groups.push(grps);
    }
    let mut col_term = vec![0u32; n_cols];
    for t in 0..n_terms {
        let start = col_offset[t] as usize;
        let end = if t + 1 < n_terms {
            col_offset[t + 1] as usize
        } else {
            n_cols
        };
        for c in start..end {
            col_term[c] = t as u32;
        }
    }
    let mut active: Vec<Vec<u32>> = Vec::with_capacity(n_terms);
    for (t, table) in bank.tables.iter().enumerate() {
        let mut col = vec![0u32; n_rows];
        for (k, &axis) in table.axes.iter().enumerate() {
            let column = data.get(axis as usize).ok_or_else(|| PbError::ShapeMismatch {
                what: format!("cell-refit: data has no axis {axis}"),
            })?;
            if column.len() != n_rows {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "cell-refit: axis {axis} column len {} != n_rows {n_rows}",
                        column.len()
                    ),
                });
            }
            let map = &table.bin_to_cell[k];
            let grp = &groups[t][k];
            let kk = coarse_shape[t][k];
            for (r, slot) in col.iter_mut().enumerate() {
                let bin = column[r] as usize;
                let merged_cell = *map.get(bin).ok_or_else(|| PbError::InvalidInput {
                    what: format!("cell-refit: bin {bin} outside bin_to_cell axis {axis}"),
                })? as usize;
                let coarse = *grp.get(merged_cell).ok_or_else(|| PbError::Internal {
                    what: "cell-refit: merged cell escaped coarse groups".into(),
                })?;
                *slot = *slot * kk + coarse;
            }
        }
        let offset = col_offset[t];
        for slot in &mut col {
            *slot += offset;
        }
        active.push(col);
    }
    Ok(Design {
        active,
        n_cols,
        col_term,
        col_offset,
        coarse_shape,
        groups,
    })
}

/// Solve `(XᵀWX + diag(lambda)) δ = XᵀW r` by diagonally-preconditioned conjugate
/// gradient. `w` is the per-row IRLS weight (1.0 for squared error), `r` the working
/// residual. Deterministic: every reduction sums in a fixed (support, row) order.
fn solve_ridge_cg(
    design: &Design,
    w: &[f64],
    r: &[f64],
    lambda: &[f64],
    spec: &CellRefitSpec,
) -> Vec<f64> {
    let n_rows = r.len();
    let n_cols = design.n_cols;

    // Right-hand side b = Xᵀ (W r).
    let mut wr = vec![0.0_f64; n_rows];
    for i in 0..n_rows {
        wr[i] = w[i] * r[i];
    }
    let mut b = vec![0.0_f64; n_cols];
    design.rmatvec(&wr, &mut b);

    // Diagonal preconditioner M[c] = colW[c] + lambda[c], colW[c] = Σ_{row active in c} W.
    let mut col_w = vec![0.0_f64; n_cols];
    for active_u in &design.active {
        for (i, &c) in active_u.iter().enumerate() {
            col_w[c as usize] += w[i];
        }
    }
    let minv: Vec<f64> = (0..n_cols)
        .map(|c| {
            let d = col_w[c] + lambda[c];
            if d > 0.0 {
                1.0 / d
            } else {
                0.0
            }
        })
        .collect();

    // A·v = Xᵀ W X v + lambda ⊙ v (scratch buffers reused across iterations).
    let mut xv = vec![0.0_f64; n_rows];
    let mut wxv = vec![0.0_f64; n_rows];
    let apply_a = |v: &[f64], out: &mut [f64], xv: &mut [f64], wxv: &mut [f64]| {
        design.matvec(v, xv);
        for i in 0..n_rows {
            wxv[i] = w[i] * xv[i];
        }
        design.rmatvec(wxv, out);
        for c in 0..n_cols {
            out[c] += lambda[c] * v[c];
        }
    };

    let mut x = vec![0.0_f64; n_cols]; // solution δ, warm-started at 0
    let mut resid = b.clone(); // r0 = b - A·0 = b
    let mut z: Vec<f64> = (0..n_cols).map(|c| minv[c] * resid[c]).collect();
    let mut p = z.clone();
    let mut ap = vec![0.0_f64; n_cols];

    let b_norm = dot(&b, &b).sqrt().max(1e-30);
    let mut rz = dot(&resid, &z);
    for _ in 0..spec.cg_iters {
        if dot(&resid, &resid).sqrt() / b_norm <= spec.cg_tol {
            break;
        }
        apply_a(&p, &mut ap, &mut xv, &mut wxv);
        let pap = dot(&p, &ap);
        if pap <= 0.0 {
            break;
        }
        let alpha = rz / pap;
        for c in 0..n_cols {
            x[c] += alpha * p[c];
            resid[c] -= alpha * ap[c];
        }
        for c in 0..n_cols {
            z[c] = minv[c] * resid[c];
        }
        let rz_new = dot(&resid, &z);
        let beta = rz_new / rz;
        for c in 0..n_cols {
            p[c] = z[c] + beta * p[c];
        }
        rz = rz_new;
    }
    x
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Fit the §G1 cell-basis correction for `supports` (each a sorted list of model axis ids,
/// order 1..=2 — mains and pairs; triples are left to the trees). `data[axis][row]` is the
/// binned training matrix, `residual` the bagged OOB working residual, `sample_weight` the
/// per-row IRLS weight (1.0 for squared error). Returns a [`CorrectionBank`] of raw
/// per-merged-cell deltas ready to attach to `model`.
///
/// # Errors
/// Propagates scaffold / shape failures.
pub fn fit_cell_correction(
    model: &Model,
    data: &[Vec<u8>],
    residual: &[f64],
    sample_weight: &[f64],
    supports: &[Vec<u32>],
    spec: &CellRefitSpec,
) -> Result<CorrectionBank, PbError> {
    let n_rows = residual.len();
    if sample_weight.len() != n_rows {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "cell-refit sample_weight len {} != residual len {n_rows}",
                sample_weight.len()
            ),
        });
    }
    let mut bank = correction_scaffold(model, supports)?;
    if bank.tables.is_empty() {
        return Ok(bank);
    }
    let design = build_design(&bank, data, n_rows, spec.pair_cell_cap)?;

    // Flat ridge init (γ folded in afterwards): λ_col = base.
    let flat = vec![spec.base; design.n_cols];
    let delta0 = solve_ridge_cg(&design, sample_weight, residual, &flat, spec);

    // Per-term signal s_term = sqrt(mean δ0² over the term's columns); adaptive weight
    // w_term = s_term^γ normalised to mean 1; per-column penalty λ_col = base / w_term².
    let n_terms = bank.tables.len();
    let mut sumsq = vec![0.0_f64; n_terms];
    let mut counts = vec![0.0_f64; n_terms];
    for c in 0..design.n_cols {
        let t = design.col_term[c] as usize;
        sumsq[t] += delta0[c] * delta0[c];
        counts[t] += 1.0;
    }
    let mut w_term: Vec<f64> = (0..n_terms)
        .map(|t| {
            let s = if counts[t] > 0.0 {
                (sumsq[t] / counts[t]).sqrt()
            } else {
                0.0
            };
            s.powf(spec.gamma)
        })
        .collect();
    let mean_w = w_term.iter().sum::<f64>() / n_terms.max(1) as f64;
    if mean_w > 0.0 {
        for w in &mut w_term {
            *w /= mean_w;
        }
    } else {
        for w in &mut w_term {
            *w = 1.0;
        }
    }
    // Guard tiny weights so λ stays finite; floor relative to the mean (=1).
    const W_FLOOR: f64 = 1e-3;
    let lambda: Vec<f64> = (0..design.n_cols)
        .map(|c| {
            let wt = w_term[design.col_term[c] as usize].max(W_FLOOR);
            spec.base / (wt * wt)
        })
        .collect();

    let delta = if spec.gamma == 0.0 {
        delta0
    } else {
        solve_ridge_cg(&design, sample_weight, residual, &lambda, spec)
    };

    // Expand the solved COARSE δ to the bank's merged-resolution values: each merged cell
    // takes its coarse group's δ (constant within coarse groups ⊇ merged cells, so G0 holds).
    for (t, table) in bank.tables.iter_mut().enumerate() {
        let coarse_off = design.col_offset[t] as usize;
        let ks = &design.coarse_shape[t];
        let grps = &design.groups[t];
        let merged_shape: Vec<usize> = table.shape.iter().map(|&s| s as usize).collect();
        let order = merged_shape.len();
        let mut cells = vec![0usize; order];
        for merged_flat in 0..table.values.len() {
            // Decode merged_flat (row-major, last axis fastest) into per-axis merged cells.
            let mut rem = merged_flat;
            for k in (0..order).rev() {
                let e = merged_shape[k];
                cells[k] = rem % e;
                rem /= e;
            }
            // Map each merged cell to its coarse group, refold to the coarse column.
            let mut coarse_flat = 0usize;
            for k in 0..order {
                let g = grps[k][cells[k]] as usize;
                coarse_flat = coarse_flat * ks[k] as usize + g;
            }
            table.values[merged_flat] = delta[coarse_off + coarse_flat];
        }
    }
    Ok(bank)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]
    use super::*;

    // Build a tiny 2-feature model via the explain fixture so we have real grids/provenance.
    fn small_model() -> Model {
        crate::explain::fixture_model()
    }

    #[test]
    fn cg_recovers_a_known_main_effect() {
        // Single main on axis 0; residual is a pure step function of axis-0 bin → the ridge
        // (light penalty) should recover a delta that reduces the residual sharply.
        let model = small_model();
        let n = 200usize;
        // axis 0 cycles bins {1,2}; axis 1 bin 1 (irrelevant here).
        let a0: Vec<u8> = (0..n).map(|i| if i % 2 == 0 { 1 } else { 2 }).collect();
        let a1: Vec<u8> = vec![1u8; n];
        let data = vec![a0.clone(), a1];
        // Target residual: +3 on bin 1, -3 on bin 2.
        let resid: Vec<f64> = a0.iter().map(|&b| if b == 1 { 3.0 } else { -3.0 }).collect();
        let w = vec![1.0_f64; n];
        let spec = CellRefitSpec {
            base: 1.0,
            gamma: 0.0,
            pair_cell_cap: 24,
            cg_iters: 200,
            cg_tol: 1e-12,
        };
        let bank = fit_cell_correction(&model, &data, &resid, &w, &[vec![0]], &spec).unwrap();
        // Predict the correction at bins 1 and 2 and check it reduces the residual a lot.
        let design = build_design(&bank, &data, n, 24).unwrap();
        let delta_flat: Vec<f64> = bank.tables[0].values.clone();
        let mut pred = vec![0.0_f64; n];
        design.matvec(&delta_flat, &mut pred);
        let before: f64 = resid.iter().map(|r| r * r).sum();
        let after: f64 = resid.iter().zip(&pred).map(|(r, p)| (r - p).powi(2)).sum();
        assert!(after < 0.05 * before, "residual not reduced: before {before} after {after}");
    }

    #[test]
    fn adaptive_gamma_shrinks_a_zero_signal_term_more() {
        // Two mains: axis 0 carries signal, axis 1 carries none. With γ>0 the no-signal
        // term's coefficients should be shrunk much closer to zero than under a flat ridge.
        let model = small_model();
        let n = 400usize;
        let a0: Vec<u8> = (0..n).map(|i| if i % 2 == 0 { 1 } else { 2 }).collect();
        let a1: Vec<u8> = (0..n).map(|i| if (i / 2) % 2 == 0 { 1 } else { 2 }).collect();
        let data = vec![a0.clone(), a1.clone()];
        // axis 0 carries a STRONG signal, axis 1 a WEAK one — both nonzero so adaptive γ has
        // something to differentially shrink (a perfectly-zero term can't be "shrunk more").
        let resid: Vec<f64> = (0..n)
            .map(|i| {
                let s0 = if a0[i] == 1 { 2.0 } else { -2.0 };
                let s1 = if a1[i] == 1 { 0.4 } else { -0.4 };
                s0 + s1
            })
            .collect();
        let w = vec![1.0_f64; n];
        let supports = vec![vec![0u32], vec![1u32]];

        let flat_spec = CellRefitSpec { base: 50.0, gamma: 0.0, pair_cell_cap: 24, cg_iters: 300, cg_tol: 1e-12 };
        let adapt_spec = CellRefitSpec { base: 50.0, gamma: 2.0, pair_cell_cap: 24, cg_iters: 300, cg_tol: 1e-12 };
        let flat = fit_cell_correction(&model, &data, &resid, &w, &supports, &flat_spec).unwrap();
        let adapt = fit_cell_correction(&model, &data, &resid, &w, &supports, &adapt_spec).unwrap();

        let norm = |bank: &CorrectionBank, term: usize| -> f64 {
            bank.tables[term].values.iter().map(|v| v * v).sum::<f64>().sqrt()
        };
        // axis-1 (no signal) shrinks relative to axis-0 more under adaptive than flat.
        let flat_ratio = norm(&flat, 1) / norm(&flat, 0).max(1e-12);
        let adapt_ratio = norm(&adapt, 1) / norm(&adapt, 0).max(1e-12);
        assert!(
            adapt_ratio < flat_ratio,
            "adaptive should shrink the zero-signal term more: flat {flat_ratio} adapt {adapt_ratio}"
        );
    }
}
