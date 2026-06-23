//! The boosting loop (spec §06.6, milestone M1.5) — the Phase-2 capstone that ties
//! the objective (§05), binning (§03), histogram engine (§06.3) and split-finder
//! (§06.2) into `Booster::fit`.
//!
//! `f0 = link(weighted mean)` (the fANOVA intercept), then each round: one
//! full-precision `grad_hess` pass w.r.t. the current raw score → `grow_oblivious_tree`
//! → `update_raw`, until `n_trees` rounds or a round cannot split (graceful stop).
//! Every tree carries `alpha = 1.0` on the green path. Optional §09 boosters may
//! rewrite only leaf scalars / alphas / the intercept; the emitted model remains
//! `ExactnessMode::Exact`.
//!
//! v1 simplifications (the green spine): `subsample = 1.0` (all rows), no column
//! sampling, full-precision histograms, single Newton leaf step, early stopping off.
//! Determinism is structural: the loop is sequential round-by-round, the only
//! parallelism (the histogram build) is a fixed-order fold, so the trained `Model` is
//! byte-identical across thread counts.
//!
//! Train/serve precision: the loop accumulates `raw` in `f32` (the §05 `grad_hess`
//! contract is `raw: &[f32]`); the f64 `ensemble_f64` / §08 table path agrees with it
//! within ~`4·n_trees·f32::EPSILON·magnitude` — exactly the tolerance the §08
//! Reconstruction gate is sized for. The per-tree leaf values are bit-identical
//! between the two scorers (same `low_bit` routing); only the accumulation width differs.

use crate::backend::{pb_rng, pb_seed, Stage};
use crate::boosters::{NesterovSpec, RefitSpec};
use crate::cat::CatEncoderStore;
use crate::constraints::MonoSign;
use crate::data::{compute_offset, BinnedMatrix};
use crate::engine::split::{grow_oblivious_tree, refit_tree_leaves, GrowConfig};
use crate::engine::{
    low_bit, Config, ExactnessMode, FitSpec, Model, ModelSchema, ObliviousTree, Sampling,
};
use crate::error::PbError;
use crate::loss::{GradHess, Link};
use crate::serialize::SCHEMA_VERSION;

fn invalid_config(what: &'static str) -> PbError {
    PbError::InvalidConfig { what: what.into() }
}

fn invalid_input(what: String) -> PbError {
    PbError::InvalidInput { what }
}

const REFIT_ACCEPT_TOL: f64 = 1.0e-10;
const REFIT_MAX_BACKTRACKS: usize = 32;
const REFIT_CHOLESKY_JITTERS: [f64; 4] = [0.0, 1.0e-12, 1.0e-10, 1.0e-8];

fn validate_fit_spec(spec: &FitSpec<'_>) -> Result<(), PbError> {
    if !(1..=3).contains(&spec.interaction.max_order) {
        return Err(PbError::InvalidConfig {
            what: format!(
                "interaction.max_order must be in 1..=3, got {}",
                spec.interaction.max_order
            ),
        });
    }
    if let Some(groups) = &spec.interaction.groups {
        if groups.is_empty() {
            return Err(invalid_config(
                "interaction groups must be non-empty when set",
            ));
        }
        for group in groups {
            if group.order() == 0 || group.order() > usize::from(spec.interaction.max_order) {
                return Err(PbError::InvalidConfig {
                    what: format!(
                        "interaction group order must be in 1..={}, got {}",
                        spec.interaction.max_order,
                        group.order()
                    ),
                });
            }
        }
    }
    Ok(())
}

fn resolve_monotone(
    spec: &FitSpec<'_>,
    n_features: usize,
) -> Result<Vec<Option<MonoSign>>, PbError> {
    let mut out = vec![None; n_features];
    for (name, sign) in &spec.monotone {
        let Some(stripped) = name.strip_prefix('f') else {
            return Err(PbError::InvalidConfig {
                what: format!("unknown monotone feature `{name}`; expected default name f{{axis}}"),
            });
        };
        let axis = stripped
            .parse::<usize>()
            .map_err(|_| PbError::InvalidConfig {
                what: format!("unknown monotone feature `{name}`; expected default name f{{axis}}"),
            })?;
        if axis >= n_features {
            return Err(PbError::InvalidConfig {
                what: format!("monotone feature `{name}` is outside {n_features} model axes"),
            });
        }
        if !matches!(sign, MonoSign::None) {
            let slot = out.get_mut(axis).ok_or_else(|| PbError::Internal {
                what: "monotone axis escaped bounds".into(),
            })?;
            *slot = Some(*sign);
        }
    }
    Ok(out)
}

fn validate_binned_matrix(x: &BinnedMatrix) -> Result<(), PbError> {
    let n = x.n_rows as usize;
    let n_features = x.data.len();
    if x.grids.len() != n_features {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "BinnedMatrix grids len {} != data len {n_features}",
                x.grids.len()
            ),
        });
    }
    if x.provenance.len() != n_features {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "BinnedMatrix provenance len {} != data len {n_features}",
                x.provenance.len()
            ),
        });
    }

    for (axis, col) in x.data.iter().enumerate() {
        if col.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!("BinnedMatrix column {axis} len {} != n_rows {n}", col.len()),
            });
        }
        let grid = x.grids.get(axis).ok_or_else(|| PbError::Internal {
            what: "grid disappeared during BinnedMatrix validation".into(),
        })?;
        if grid.missing_bin != 0 {
            return Err(invalid_input(format!(
                "BinnedMatrix grid {axis} missing_bin must be 0, got {}",
                grid.missing_bin
            )));
        }
        if grid.n_bins == 0 || grid.n_bins > 255 {
            return Err(invalid_input(format!(
                "BinnedMatrix grid {axis} n_bins must be in 1..=255, got {}",
                grid.n_bins
            )));
        }
        let expected_bins = if grid.n_bins == 1 {
            if !grid.borders.is_empty() {
                return Err(invalid_input(format!(
                    "BinnedMatrix grid {axis} has n_bins=1 but {} borders",
                    grid.borders.len()
                )));
            }
            1usize
        } else {
            grid.borders
                .len()
                .checked_add(2)
                .ok_or_else(|| PbError::Internal {
                    what: "BinnedMatrix border count overflow".into(),
                })?
        };
        if expected_bins != usize::from(grid.n_bins) {
            return Err(invalid_input(format!(
                "BinnedMatrix grid {axis} n_bins {} != borders.len()+2 ({expected_bins})",
                grid.n_bins
            )));
        }
        for (i, &border) in grid.borders.iter().enumerate() {
            if !border.is_finite() {
                return Err(invalid_input(format!(
                    "BinnedMatrix grid {axis} border {i} must be finite"
                )));
            }
        }
        for pair in grid.borders.windows(2) {
            if let [a, b] = pair {
                if a >= b {
                    return Err(invalid_input(format!(
                        "BinnedMatrix grid {axis} borders must be strictly ascending"
                    )));
                }
            }
        }
        for (row, &bin) in col.iter().enumerate() {
            if u16::from(bin) >= grid.n_bins {
                return Err(invalid_input(format!(
                    "BinnedMatrix column {axis} row {row} bin {bin} outside grid n_bins {}",
                    grid.n_bins
                )));
            }
        }
    }
    Ok(())
}

/// Fit an ensemble (spec §06.6). See [`crate::engine::Booster::fit`].
///
/// # Errors
/// [`PbError::InvalidConfig`] on a bad config; [`PbError::ShapeMismatch`] on a
/// `y`/`weight` length mismatch; plus any propagated `Loss`/binning/grow error.
pub(crate) fn fit(
    config: &Config,
    x: &BinnedMatrix,
    y: &[f32],
    spec: &FitSpec,
) -> Result<Model, PbError> {
    config.validate()?;
    validate_fit_spec(spec)?;
    validate_binned_matrix(x)?;
    let n = x.n_rows as usize;
    if y.len() != n {
        return Err(PbError::ShapeMismatch {
            what: format!("y len {} != n_rows {n}", y.len()),
        });
    }

    // Weights default to all-ones; `ones_storage` backs that borrow for the fn body.
    let ones_storage: Vec<f32>;
    let weight: &[f32] = match spec.weight {
        Some(w) => {
            if w.len() != n {
                return Err(PbError::ShapeMismatch {
                    what: format!("weight len {} != n_rows {n}", w.len()),
                });
            }
            w
        }
        None => {
            ones_storage = vec![1.0_f32; n];
            &ones_storage
        }
    };

    // Exposure → per-row offset (§03.7); folded into the raw score, not into binning.
    let offset: Option<Vec<f32>> = match spec.exposure {
        Some(e) => Some(compute_offset(e, n)?),
        None => None,
    };

    // f0 = link(weighted mean) in f64 (the exact fANOVA intercept); down-cast once.
    let f0 = spec.loss.init_score(y, weight, offset.as_deref())?;
    let f0_f32 = f0 as f32;
    let mut raw = vec![f0_f32; n];
    if let Some(off) = &offset {
        for (r, o) in raw.iter_mut().zip(off) {
            *r += o;
        }
    }

    let n_features = x.data.len();
    let monotone = resolve_monotone(spec, n_features)?;
    let monotone_ref = if monotone.iter().any(Option::is_some) {
        Some(monotone.as_slice())
    } else {
        None
    };
    let axes: Vec<u32> = (0..u32::try_from(n_features).map_err(|_| PbError::Internal {
        what: "more than u32::MAX features".into(),
    })?)
        .collect();
    let rows: Vec<u32> = (0..x.n_rows).collect();
    // Resolve the leaf-stage |w*|-clamp (§05.6): an explicit Config value wins, else fall
    // back to the loss's advertised cap (Poisson ⇒ Some(0.7)).
    let max_delta_step = config
        .max_delta_step
        .or_else(|| spec.loss.max_delta_step())
        .map(f64::from);
    let grow_cfg = GrowConfig {
        lambda: f64::from(config.lambda),
        lr: f64::from(config.learning_rate),
        min_split_gain: f64::from(config.min_split_gain),
        max_order: spec.interaction.max_order,
        max_delta_step,
        hist_precision: config.hist_precision,
        quant_seed: spec.seed,
        round: 0,
        random_strength: f64::from(config.boosters.random_strength),
        groups: spec.interaction.groups.as_deref(),
        monotone: monotone_ref,
    };

    let mut trees: Vec<(f32, ObliviousTree)> = Vec::new();
    let mut gh = GradHess::default();
    let mut last_refit_tree_count = 0usize;
    let mut prev_alphas: Vec<f32> = Vec::new();
    for t in 0..config.n_trees {
        // Per-round deterministic re-seed — the seam for MVS/subsampling (M5-QHIST,
        // v1.5).
        let _round_rng = pb_rng(spec.seed, t, Stage::Sample, 0);
        let current_alphas = collect_tree_alphas(&trees)?;
        let agbm = match config.boosters.nesterov {
            NesterovSpec::Off => None,
            NesterovSpec::Agbm {
                momentum_correction,
            } => Some((agbm_beta(t), momentum_correction)),
        };
        let mut fit_raw = raw.clone();
        if let Some((beta, _)) = agbm {
            let lookahead_alphas = combine_alphas(&current_alphas, &prev_alphas, beta)?;
            set_tree_alphas(&mut trees, &lookahead_alphas)?;
            fit_raw = raw_from_tree_alphas(f0_f32, offset.as_deref(), x, &trees)?;
        }
        spec.loss.grad_hess(y, &fit_raw, weight, &mut gh)?;
        let sampled_rows = sample_rows(&config.sampling, &gh, spec.seed, t, &rows)?;
        let mut round_grow_cfg = grow_cfg.clone();
        round_grow_cfg.round = t;
        match grow_oblivious_tree(x, &gh, &sampled_rows, &axes, &round_grow_cfg)? {
            Some(mut tree) => {
                if sampled_rows.len() != rows.len() {
                    refit_tree_leaves(x, &gh, &rows, &mut tree, &round_grow_cfg)?;
                }
                prev_alphas = current_alphas;
                raw = fit_raw;
                update_raw(&mut raw, x, &tree)?;
                trees.push((1.0, tree));
                if should_refit_after_round(&config.boosters.refit_leaves, trees.len())? {
                    let problem = RefitProblem {
                        spec,
                        x,
                        y,
                        weight,
                        offset: offset.as_deref(),
                        f0: f0_f32,
                    };
                    fully_corrective_refit(
                        &config.boosters.refit_leaves,
                        &problem,
                        &mut trees,
                        &mut raw,
                    )?;
                    last_refit_tree_count = trees.len();
                }
                if matches!(
                    config.boosters.nesterov,
                    NesterovSpec::Agbm {
                        momentum_correction: true
                    }
                ) {
                    spec.loss.grad_hess(y, &raw, weight, &mut gh)?;
                    let correction_rows = sample_rows(&config.sampling, &gh, spec.seed, t, &rows)?;
                    let mut correction_cfg = grow_cfg.clone();
                    correction_cfg.round = t;
                    if let Some(mut correction) =
                        grow_oblivious_tree(x, &gh, &correction_rows, &axes, &correction_cfg)?
                    {
                        if correction_rows.len() != rows.len() {
                            refit_tree_leaves(x, &gh, &rows, &mut correction, &correction_cfg)?;
                        }
                        update_raw(&mut raw, x, &correction)?;
                        trees.push((1.0, correction));
                        if should_refit_after_round(&config.boosters.refit_leaves, trees.len())? {
                            let problem = RefitProblem {
                                spec,
                                x,
                                y,
                                weight,
                                offset: offset.as_deref(),
                                f0: f0_f32,
                            };
                            fully_corrective_refit(
                                &config.boosters.refit_leaves,
                                &problem,
                                &mut trees,
                                &mut raw,
                            )?;
                            last_refit_tree_count = trees.len();
                        }
                    }
                }
            }
            // No admissible split clears the floor (e.g. converged / constant target):
            // stop early with what we have — a valid (possibly empty) Exact model.
            None => {
                if agbm.is_some() {
                    set_tree_alphas(&mut trees, &current_alphas)?;
                }
                break;
            }
        }
    }
    if should_refit_at_end(
        &config.boosters.refit_leaves,
        trees.len(),
        last_refit_tree_count,
    ) {
        let problem = RefitProblem {
            spec,
            x,
            y,
            weight,
            offset: offset.as_deref(),
            f0: f0_f32,
        };
        fully_corrective_refit(
            &config.boosters.refit_leaves,
            &problem,
            &mut trees,
            &mut raw,
        )?;
    }

    let f0_model = if config.boosters.reanchor {
        let delta = reanchor_delta(spec.loss.link(), y, weight, &raw)?;
        let shifted = f64::from(f0_f32) + delta;
        if !shifted.is_finite() || shifted < f64::from(f32::MIN) || shifted > f64::from(f32::MAX) {
            return Err(PbError::InvalidInput {
                what: "reanchored intercept is not representable as f32".into(),
            });
        }
        shifted as f32
    } else {
        f0_f32
    };

    let schema = ModelSchema {
        feature_names: (0..n_features).map(|i| format!("f{i}")).collect(),
        feature_kinds: x.provenance.iter().map(|p| p.kind).collect(),
        cat_encoders: CatEncoderStore::new(),
        class_labels: None,
        objective: spec.loss.objective_tag(),
    };
    Ok(Model {
        f0: f0_model,
        trees,
        grids: x.grids.clone(),
        provenance: x.provenance.clone(),
        link: spec.loss.link(),
        mode: ExactnessMode::Exact,
        schema,
        schema_version: SCHEMA_VERSION,
    })
}

fn should_refit_after_round(refit: &RefitSpec, n_trees: usize) -> Result<bool, PbError> {
    match refit {
        RefitSpec::Ridge {
            every_k_trees: Some(k),
            ..
        } => {
            let k = usize::try_from(*k).map_err(|_| PbError::Internal {
                what: "refit every_k_trees exceeded usize".into(),
            })?;
            Ok(n_trees > 0 && n_trees % k == 0)
        }
        _ => Ok(false),
    }
}

fn should_refit_at_end(refit: &RefitSpec, n_trees: usize, last_refit_tree_count: usize) -> bool {
    matches!(refit, RefitSpec::Ridge { .. }) && n_trees > 0 && last_refit_tree_count != n_trees
}

fn agbm_beta(round: u32) -> f32 {
    let theta = 2.0_f32 / (round as f32 + 2.0);
    1.0 - theta
}

fn collect_tree_alphas(trees: &[(f32, ObliviousTree)]) -> Result<Vec<f32>, PbError> {
    let mut out = Vec::new();
    out.try_reserve_exact(trees.len())
        .map_err(|_| PbError::Internal {
            what: "AGBM alpha allocation failed".into(),
        })?;
    for (alpha, _) in trees {
        if !alpha.is_finite() {
            return Err(PbError::InvalidInput {
                what: "AGBM tree alpha must be finite".into(),
            });
        }
        out.push(*alpha);
    }
    Ok(out)
}

fn combine_alphas(current: &[f32], previous: &[f32], beta: f32) -> Result<Vec<f32>, PbError> {
    if previous.len() > current.len() {
        return Err(PbError::Internal {
            what: "AGBM previous alpha vector longer than current".into(),
        });
    }
    let mut out = Vec::new();
    out.try_reserve_exact(current.len())
        .map_err(|_| PbError::Internal {
            what: "AGBM combined alpha allocation failed".into(),
        })?;
    for (idx, &alpha) in current.iter().enumerate() {
        let prev = previous.get(idx).copied().unwrap_or(0.0);
        let value = (1.0 + beta) * alpha - beta * prev;
        if !value.is_finite() {
            return Err(PbError::InvalidInput {
                what: "AGBM combined alpha is not finite".into(),
            });
        }
        out.push(value);
    }
    Ok(out)
}

fn set_tree_alphas(trees: &mut [(f32, ObliviousTree)], alphas: &[f32]) -> Result<(), PbError> {
    if trees.len() != alphas.len() {
        return Err(PbError::ShapeMismatch {
            what: "AGBM alpha vector length does not match tree count".into(),
        });
    }
    for ((alpha_slot, _), &alpha) in trees.iter_mut().zip(alphas) {
        if !alpha.is_finite() {
            return Err(PbError::InvalidInput {
                what: "AGBM alpha must be finite".into(),
            });
        }
        *alpha_slot = alpha;
    }
    Ok(())
}

fn raw_from_tree_alphas(
    f0: f32,
    offset: Option<&[f32]>,
    x: &BinnedMatrix,
    trees: &[(f32, ObliviousTree)],
) -> Result<Vec<f32>, PbError> {
    let n_rows = x.n_rows as usize;
    let mut out: Vec<f32> = crate::engine::Hist::try_zeroed_vec(n_rows, "AGBM raw")?;
    for row in 0..n_rows {
        let mut score = base_raw(offset, f0, row)?;
        for (alpha, tree) in trees {
            score += f64::from(*alpha) * f64::from(tree_value_for_row(tree, x, row)?);
        }
        if !score.is_finite() || score < f64::from(f32::MIN) || score > f64::from(f32::MAX) {
            return Err(PbError::InvalidInput {
                what: "AGBM raw score is not finite/representable as f32".into(),
            });
        }
        *out.get_mut(row).ok_or_else(|| PbError::Internal {
            what: "AGBM raw write escaped".into(),
        })? = score as f32;
    }
    Ok(out)
}

struct RefitProblem<'a, 's> {
    spec: &'a FitSpec<'s>,
    x: &'a BinnedMatrix,
    y: &'a [f32],
    weight: &'a [f32],
    offset: Option<&'a [f32]>,
    f0: f32,
}

fn fully_corrective_refit(
    refit: &RefitSpec,
    problem: &RefitProblem<'_, '_>,
    trees: &mut [(f32, ObliviousTree)],
    raw: &mut [f32],
) -> Result<(), PbError> {
    let RefitSpec::Ridge { l2, max_iter, .. } = refit else {
        return Ok(());
    };
    if trees.is_empty() {
        return Ok(());
    }
    let memberships = leaf_memberships(problem.x, trees)?;
    let n_trees = trees.len();
    let n_cols = n_trees.checked_mul(8).ok_or_else(|| PbError::Internal {
        what: "refit column count overflow".into(),
    })?;
    let mut gh = GradHess::default();
    for _ in 0..*max_iter {
        problem
            .spec
            .loss
            .grad_hess(problem.y, raw, problem.weight, &mut gh)?;
        let (normal, rhs) = refit_normal_equations(
            problem,
            &gh,
            raw,
            trees,
            &memberships,
            n_cols,
            f64::from(*l2),
        )?;
        let target = solve_refit_system(&normal, &rhs, n_cols)?;
        let current = collect_leaf_theta(trees, n_cols)?;
        let current_deviance =
            f64::from(problem.spec.loss.deviance(problem.y, raw, problem.weight)?);
        let mut step = 1.0_f64;
        let mut accepted: Option<(Vec<f64>, Vec<f32>, f64)> = None;
        for _ in 0..REFIT_MAX_BACKTRACKS {
            let candidate_theta = interpolate_theta(&current, &target, step)?;
            let candidate_raw = raw_from_theta(
                problem.offset,
                problem.f0,
                trees,
                &memberships,
                &candidate_theta,
            )?;
            let deviance = f64::from(problem.spec.loss.deviance(
                problem.y,
                &candidate_raw,
                problem.weight,
            )?);
            if deviance.is_finite()
                && deviance <= current_deviance + REFIT_ACCEPT_TOL * (1.0 + current_deviance.abs())
            {
                accepted = Some((candidate_theta, candidate_raw, deviance));
                break;
            }
            step *= 0.5;
        }
        let Some((theta, candidate_raw, accepted_deviance)) = accepted else {
            break;
        };
        write_leaf_theta(trees, &theta)?;
        if raw.len() != candidate_raw.len() {
            return Err(PbError::Internal {
                what: "refit raw length changed".into(),
            });
        }
        for (dst, src) in raw.iter_mut().zip(candidate_raw) {
            *dst = src;
        }
        if (current_deviance - accepted_deviance).abs()
            <= REFIT_ACCEPT_TOL * (1.0 + current_deviance.abs())
        {
            break;
        }
    }
    Ok(())
}

fn leaf_memberships(x: &BinnedMatrix, trees: &[(f32, ObliviousTree)]) -> Result<Vec<u8>, PbError> {
    let n_rows = x.n_rows as usize;
    let n_trees = trees.len();
    let cells = n_rows
        .checked_mul(n_trees)
        .ok_or_else(|| PbError::Internal {
            what: "leaf membership shape overflow".into(),
        })?;
    let mut memberships: Vec<u8> = crate::engine::Hist::try_zeroed_vec(cells, "leaf membership")?;
    for row in 0..n_rows {
        for (tree_idx, (_, tree)) in trees.iter().enumerate() {
            let leaf =
                u8::try_from(leaf_index_for_row(tree, x, row)?).map_err(|_| PbError::Internal {
                    what: "leaf index exceeded u8".into(),
                })?;
            let slot = membership_offset(row, tree_idx, n_trees)?;
            *memberships.get_mut(slot).ok_or_else(|| PbError::Internal {
                what: "leaf membership offset escaped".into(),
            })? = leaf;
        }
    }
    Ok(memberships)
}

fn membership_offset(row: usize, tree_idx: usize, n_trees: usize) -> Result<usize, PbError> {
    row.checked_mul(n_trees)
        .and_then(|o| o.checked_add(tree_idx))
        .ok_or_else(|| PbError::Internal {
            what: "leaf membership offset overflow".into(),
        })
}

fn refit_col(tree_idx: usize, leaf: usize) -> Result<usize, PbError> {
    tree_idx
        .checked_mul(8)
        .and_then(|o| o.checked_add(leaf))
        .ok_or_else(|| PbError::Internal {
            what: "refit column offset overflow".into(),
        })
}

fn leaf_from_membership(
    memberships: &[u8],
    row: usize,
    tree_idx: usize,
    n_trees: usize,
) -> Result<usize, PbError> {
    let offset = membership_offset(row, tree_idx, n_trees)?;
    let leaf = usize::from(*memberships.get(offset).ok_or_else(|| PbError::Internal {
        what: "leaf membership lookup escaped".into(),
    })?);
    if leaf >= 8 {
        return Err(PbError::Internal {
            what: "leaf membership value escaped 0..8".into(),
        });
    }
    Ok(leaf)
}

fn base_raw(offset: Option<&[f32]>, f0: f32, row: usize) -> Result<f64, PbError> {
    let mut out = f64::from(f0);
    if let Some(off) = offset {
        out += f64::from(*off.get(row).ok_or_else(|| PbError::Internal {
            what: "refit offset row escaped".into(),
        })?);
    }
    Ok(out)
}

fn refit_normal_equations(
    problem: &RefitProblem<'_, '_>,
    gh: &GradHess,
    raw: &[f32],
    trees: &[(f32, ObliviousTree)],
    memberships: &[u8],
    n_cols: usize,
    l2: f64,
) -> Result<(Vec<f64>, Vec<f64>), PbError> {
    let n_rows = raw.len();
    if gh.g.len() != n_rows || gh.h.len() != n_rows {
        return Err(PbError::ShapeMismatch {
            what: "refit GradHess length does not match raw".into(),
        });
    }
    let cells = n_cols
        .checked_mul(n_cols)
        .ok_or_else(|| PbError::Internal {
            what: "refit normal matrix shape overflow".into(),
        })?;
    let mut normal: Vec<f64> = crate::engine::Hist::try_zeroed_vec(cells, "refit normal matrix")?;
    let mut rhs: Vec<f64> = crate::engine::Hist::try_zeroed_vec(n_cols, "refit rhs")?;
    let n_trees = trees.len();
    for row in 0..n_rows {
        let g = f64::from(*gh.g.get(row).ok_or_else(|| PbError::Internal {
            what: "refit gradient row escaped".into(),
        })?);
        let h = f64::from(*gh.h.get(row).ok_or_else(|| PbError::Internal {
            what: "refit hessian row escaped".into(),
        })?);
        if !g.is_finite() || !h.is_finite() {
            return Err(PbError::InvalidInput {
                what: "refit gradients must be finite".into(),
            });
        }
        if h <= 0.0 {
            continue;
        }
        let z_centered = f64::from(*raw.get(row).ok_or_else(|| PbError::Internal {
            what: "refit raw row escaped".into(),
        })?) - g / h
            - base_raw(problem.offset, problem.f0, row)?;
        for (a_idx, (alpha_a, _)) in trees.iter().enumerate() {
            let alpha_a = f64::from(*alpha_a);
            if !alpha_a.is_finite() {
                return Err(PbError::InvalidInput {
                    what: "refit tree alpha must be finite".into(),
                });
            }
            let col_a = refit_col(
                a_idx,
                leaf_from_membership(memberships, row, a_idx, n_trees)?,
            )?;
            add_vec(&mut rhs, col_a, h * alpha_a * z_centered)?;
            for (b_idx, (alpha_b, _)) in trees.iter().enumerate() {
                let alpha_b = f64::from(*alpha_b);
                if !alpha_b.is_finite() {
                    return Err(PbError::InvalidInput {
                        what: "refit tree alpha must be finite".into(),
                    });
                }
                let col_b = refit_col(
                    b_idx,
                    leaf_from_membership(memberships, row, b_idx, n_trees)?,
                )?;
                add_matrix(&mut normal, n_cols, col_a, col_b, h * alpha_a * alpha_b)?;
            }
        }
    }
    for col in 0..n_cols {
        add_matrix(&mut normal, n_cols, col, col, l2)?;
    }
    Ok((normal, rhs))
}

fn raw_from_theta(
    offset: Option<&[f32]>,
    f0: f32,
    trees: &[(f32, ObliviousTree)],
    memberships: &[u8],
    theta: &[f64],
) -> Result<Vec<f32>, PbError> {
    let n_trees = trees.len();
    let n_rows = if n_trees == 0 {
        0
    } else {
        memberships
            .len()
            .checked_div(n_trees)
            .ok_or_else(|| PbError::Internal {
                what: "refit membership row count overflow".into(),
            })?
    };
    let mut out: Vec<f32> = crate::engine::Hist::try_zeroed_vec(n_rows, "refit raw")?;
    for row in 0..n_rows {
        let mut score = base_raw(offset, f0, row)?;
        for (tree_idx, (alpha, _)) in trees.iter().enumerate() {
            let leaf = leaf_from_membership(memberships, row, tree_idx, n_trees)?;
            let col = refit_col(tree_idx, leaf)?;
            score += f64::from(*alpha)
                * *theta.get(col).ok_or_else(|| PbError::Internal {
                    what: "refit theta lookup escaped".into(),
                })?;
        }
        if !score.is_finite() || score < f64::from(f32::MIN) || score > f64::from(f32::MAX) {
            return Err(PbError::InvalidInput {
                what: "refit raw score is not finite/representable as f32".into(),
            });
        }
        *out.get_mut(row).ok_or_else(|| PbError::Internal {
            what: "refit raw write escaped".into(),
        })? = score as f32;
    }
    Ok(out)
}

fn collect_leaf_theta(trees: &[(f32, ObliviousTree)], n_cols: usize) -> Result<Vec<f64>, PbError> {
    let mut theta: Vec<f64> = Vec::new();
    theta
        .try_reserve_exact(n_cols)
        .map_err(|_| PbError::Internal {
            what: "refit theta allocation failed".into(),
        })?;
    for (_, tree) in trees {
        for &leaf in &tree.leaves {
            theta.push(f64::from(leaf));
        }
    }
    if theta.len() != n_cols {
        return Err(PbError::Internal {
            what: "refit theta length mismatch".into(),
        });
    }
    Ok(theta)
}

fn interpolate_theta(current: &[f64], target: &[f64], step: f64) -> Result<Vec<f64>, PbError> {
    if current.len() != target.len() {
        return Err(PbError::ShapeMismatch {
            what: "refit theta interpolation length mismatch".into(),
        });
    }
    let mut out: Vec<f64> = Vec::new();
    out.try_reserve_exact(current.len())
        .map_err(|_| PbError::Internal {
            what: "refit theta interpolation allocation failed".into(),
        })?;
    for (&a, &b) in current.iter().zip(target) {
        out.push(a + step * (b - a));
    }
    Ok(out)
}

fn write_leaf_theta(trees: &mut [(f32, ObliviousTree)], theta: &[f64]) -> Result<(), PbError> {
    for (tree_idx, (_, tree)) in trees.iter_mut().enumerate() {
        for leaf in 0..8usize {
            let col = refit_col(tree_idx, leaf)?;
            let value = *theta.get(col).ok_or_else(|| PbError::Internal {
                what: "refit theta write lookup escaped".into(),
            })?;
            if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
                return Err(PbError::InvalidInput {
                    what: "refit leaf value is not finite/representable as f32".into(),
                });
            }
            *tree.leaves.get_mut(leaf).ok_or_else(|| PbError::Internal {
                what: "refit leaf write escaped".into(),
            })? = value as f32;
        }
    }
    Ok(())
}

fn matrix_offset(n: usize, row: usize, col: usize) -> Result<usize, PbError> {
    if row >= n || col >= n {
        return Err(PbError::Internal {
            what: "matrix coordinate out of range".into(),
        });
    }
    row.checked_mul(n)
        .and_then(|o| o.checked_add(col))
        .ok_or_else(|| PbError::Internal {
            what: "matrix offset overflow".into(),
        })
}

fn matrix_get(a: &[f64], n: usize, row: usize, col: usize) -> Result<f64, PbError> {
    let offset = matrix_offset(n, row, col)?;
    a.get(offset).copied().ok_or_else(|| PbError::Internal {
        what: "matrix lookup escaped".into(),
    })
}

fn matrix_set(a: &mut [f64], n: usize, row: usize, col: usize, value: f64) -> Result<(), PbError> {
    let offset = matrix_offset(n, row, col)?;
    *a.get_mut(offset).ok_or_else(|| PbError::Internal {
        what: "matrix write escaped".into(),
    })? = value;
    Ok(())
}

fn add_matrix(a: &mut [f64], n: usize, row: usize, col: usize, delta: f64) -> Result<(), PbError> {
    let offset = matrix_offset(n, row, col)?;
    let slot = a.get_mut(offset).ok_or_else(|| PbError::Internal {
        what: "matrix add escaped".into(),
    })?;
    *slot += delta;
    Ok(())
}

fn add_vec(v: &mut [f64], idx: usize, delta: f64) -> Result<(), PbError> {
    let slot = v.get_mut(idx).ok_or_else(|| PbError::Internal {
        what: "vector add escaped".into(),
    })?;
    *slot += delta;
    Ok(())
}

fn clone_f64_slice(input: &[f64], what: &'static str) -> Result<Vec<f64>, PbError> {
    let mut out = Vec::new();
    out.try_reserve_exact(input.len())
        .map_err(|_| PbError::Internal {
            what: format!("{what} allocation failed"),
        })?;
    out.extend_from_slice(input);
    Ok(out)
}

fn solve_refit_system(normal: &[f64], rhs: &[f64], n: usize) -> Result<Vec<f64>, PbError> {
    if rhs.len() != n {
        return Err(PbError::ShapeMismatch {
            what: "refit rhs length mismatch".into(),
        });
    }
    if normal.len()
        != n.checked_mul(n).ok_or_else(|| PbError::Internal {
            what: "refit solve shape overflow".into(),
        })?
    {
        return Err(PbError::ShapeMismatch {
            what: "refit normal matrix length mismatch".into(),
        });
    }
    let mut diag_scale = 1.0_f64;
    for i in 0..n {
        diag_scale = diag_scale.max(matrix_get(normal, n, i, i)?.abs());
    }
    for jitter in REFIT_CHOLESKY_JITTERS {
        let mut a = clone_f64_slice(normal, "refit solve matrix")?;
        if jitter > 0.0 {
            for i in 0..n {
                add_matrix(&mut a, n, i, i, jitter * diag_scale)?;
            }
        }
        if let Some(solution) = cholesky_solve(&a, rhs, n)? {
            return Ok(solution);
        }
    }
    Err(PbError::InvalidInput {
        what: "refit normal equations are not positive definite".into(),
    })
}

fn cholesky_solve(a: &[f64], rhs: &[f64], n: usize) -> Result<Option<Vec<f64>>, PbError> {
    let cells = n.checked_mul(n).ok_or_else(|| PbError::Internal {
        what: "Cholesky matrix shape overflow".into(),
    })?;
    let mut l: Vec<f64> = crate::engine::Hist::try_zeroed_vec(cells, "Cholesky factor")?;
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix_get(a, n, i, j)?;
            for k in 0..j {
                sum -= matrix_get(&l, n, i, k)? * matrix_get(&l, n, j, k)?;
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return Ok(None);
                }
                matrix_set(&mut l, n, i, j, sum.sqrt())?;
            } else {
                let diag = matrix_get(&l, n, j, j)?;
                if diag <= 0.0 || !diag.is_finite() {
                    return Ok(None);
                }
                matrix_set(&mut l, n, i, j, sum / diag)?;
            }
        }
    }

    let mut y: Vec<f64> = crate::engine::Hist::try_zeroed_vec(n, "Cholesky forward solve")?;
    for i in 0..n {
        let mut sum = *rhs.get(i).ok_or_else(|| PbError::Internal {
            what: "Cholesky rhs lookup escaped".into(),
        })?;
        for k in 0..i {
            sum -= matrix_get(&l, n, i, k)?
                * *y.get(k).ok_or_else(|| PbError::Internal {
                    what: "Cholesky y lookup escaped".into(),
                })?;
        }
        let diag = matrix_get(&l, n, i, i)?;
        if diag <= 0.0 || !diag.is_finite() {
            return Ok(None);
        }
        *y.get_mut(i).ok_or_else(|| PbError::Internal {
            what: "Cholesky y write escaped".into(),
        })? = sum / diag;
    }

    let mut x: Vec<f64> = crate::engine::Hist::try_zeroed_vec(n, "Cholesky back solve")?;
    for i in (0..n).rev() {
        let mut sum = *y.get(i).ok_or_else(|| PbError::Internal {
            what: "Cholesky y back lookup escaped".into(),
        })?;
        for k in (i + 1)..n {
            sum -= matrix_get(&l, n, k, i)?
                * *x.get(k).ok_or_else(|| PbError::Internal {
                    what: "Cholesky x lookup escaped".into(),
                })?;
        }
        let diag = matrix_get(&l, n, i, i)?;
        if diag <= 0.0 || !diag.is_finite() {
            return Ok(None);
        }
        *x.get_mut(i).ok_or_else(|| PbError::Internal {
            what: "Cholesky x write escaped".into(),
        })? = sum / diag;
    }
    Ok(Some(x))
}

fn inverse_link_f64(link: Link, raw: f64) -> f64 {
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

fn weighted_response_total(link: Link, raw: &[f32], weight: &[f32]) -> f64 {
    let mut total = 0.0_f64;
    for (&v, &w) in raw.iter().zip(weight) {
        total += f64::from(w) * inverse_link_f64(link, f64::from(v));
    }
    total
}

fn weighted_observed_total(y: &[f32], weight: &[f32]) -> f64 {
    y.iter()
        .zip(weight)
        .map(|(&yi, &wi)| f64::from(wi) * f64::from(yi))
        .sum()
}

fn reanchor_delta(link: Link, y: &[f32], weight: &[f32], raw: &[f32]) -> Result<f64, PbError> {
    let sum_w: f64 = weight.iter().map(|&w| f64::from(w)).sum();
    if sum_w <= 0.0 || !sum_w.is_finite() {
        return Err(PbError::InvalidInput {
            what: "reanchor requires positive finite total weight".into(),
        });
    }
    let observed = weighted_observed_total(y, weight);
    if !observed.is_finite() {
        return Err(PbError::InvalidInput {
            what: "reanchor observed total is not finite".into(),
        });
    }
    match link {
        Link::Identity => {
            let predicted = raw
                .iter()
                .zip(weight)
                .map(|(&ri, &wi)| f64::from(wi) * f64::from(ri))
                .sum::<f64>();
            Ok((observed - predicted) / sum_w)
        }
        Link::Log => {
            if observed <= 0.0 {
                return Err(PbError::InvalidInput {
                    what: "log-link reanchor requires positive observed total".into(),
                });
            }
            let predicted = weighted_response_total(link, raw, weight);
            if predicted <= 0.0 || !predicted.is_finite() {
                return Err(PbError::InvalidInput {
                    what: "log-link reanchor predicted total must be positive and finite".into(),
                });
            }
            Ok((observed / predicted).ln())
        }
        Link::Logit => {
            if observed <= 0.0 || observed >= sum_w {
                return Err(PbError::InvalidInput {
                    what: "logit-link reanchor requires observed positives strictly inside (0, total_weight)".into(),
                });
            }
            let mut lo = -60.0_f64;
            let mut hi = 60.0_f64;
            for _ in 0..96 {
                let mid = 0.5 * (lo + hi);
                let mut predicted = 0.0_f64;
                for (&ri, &wi) in raw.iter().zip(weight) {
                    predicted += f64::from(wi) * inverse_link_f64(link, f64::from(ri) + mid);
                }
                if predicted < observed {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            Ok(0.5 * (lo + hi))
        }
    }
}

fn sample_rows(
    sampling: &Sampling,
    gh: &GradHess,
    seed: u64,
    round: u32,
    all_rows: &[u32],
) -> Result<Vec<u32>, PbError> {
    match *sampling {
        Sampling::Full => Ok(all_rows.to_vec()),
        Sampling::Mvs { rate, min_rows } => {
            let n = all_rows.len();
            if n == 0 {
                return Ok(Vec::new());
            }
            let target = ((n as f64) * f64::from(rate)).ceil() as usize;
            let min_rows = usize::try_from(min_rows).map_err(|_| PbError::Internal {
                what: "MVS min_rows exceeded usize".into(),
            })?;
            let k = target.max(min_rows).min(n).max(1);
            if k == n {
                return Ok(all_rows.to_vec());
            }
            let mut keyed: Vec<(f64, u32)> = Vec::with_capacity(n);
            for (pos, &row) in all_rows.iter().enumerate() {
                let ru = row as usize;
                let g = f64::from(*gh.g.get(ru).ok_or_else(|| PbError::Internal {
                    what: "MVS row escaped gradients".into(),
                })?);
                let h = f64::from(*gh.h.get(ru).ok_or_else(|| PbError::Internal {
                    what: "MVS row escaped hessians".into(),
                })?);
                let weight = (g * g + h * h).sqrt().max(1e-12);
                let block = u32::try_from(pos).map_err(|_| PbError::InvalidInput {
                    what: "MVS sampling supports at most u32::MAX rows".into(),
                })?;
                let bits = pb_seed(seed, round, Stage::Sample as u32, block);
                let unit = ((bits >> 11) as f64 + 1.0) / ((1_u64 << 53) as f64 + 1.0);
                // Efraimidis-Spirakis PPS-without-replacement key. Larger is better
                // (`ln(unit)` is negative; dividing by a larger gradient weight moves
                // it closer to zero).
                keyed.push((unit.ln() / weight, row));
            }
            keyed.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let mut rows: Vec<u32> = keyed.into_iter().take(k).map(|(_, row)| row).collect();
            rows.sort_unstable();
            Ok(rows)
        }
    }
}

/// Add a freshly-grown tree's contribution to every row's raw score (spec §06.6
/// sample→leaf update). Scores ALL rows (not just the round's subsample) so the next
/// round's gradients are correct everywhere. Panic-free; uses the canonical low bit.
fn update_raw(raw: &mut [f32], x: &BinnedMatrix, tree: &ObliviousTree) -> Result<(), PbError> {
    for (r, slot) in raw.iter_mut().enumerate() {
        *slot += tree_value_for_row(tree, x, r)?;
    }
    Ok(())
}

/// Score one row against one tree by column-major reads, folding the leaf index with
/// the SAME canonical `low_bit` rule as [`ObliviousTree::lookup`] and the grower.
fn tree_value_for_row(tree: &ObliviousTree, x: &BinnedMatrix, r: usize) -> Result<f32, PbError> {
    let idx = leaf_index_for_row(tree, x, r)?;
    tree.leaves
        .get(idx)
        .copied()
        .ok_or_else(|| PbError::Internal {
            what: "update_raw: leaf index escaped 0..8".into(),
        })
}

fn leaf_index_for_row(tree: &ObliviousTree, x: &BinnedMatrix, r: usize) -> Result<usize, PbError> {
    let mut idx = 0usize;
    for (level, split) in tree.splits.iter().enumerate() {
        let bin = *x
            .data
            .get(split.axis as usize)
            .ok_or_else(|| PbError::Internal {
                what: "update_raw: split axis out of range".into(),
            })?
            .get(r)
            .ok_or_else(|| PbError::Internal {
                what: "update_raw: row out of column".into(),
            })?;
        idx |= usize::from(low_bit(bin, split.bin_le, split.missing_left)) << level;
    }
    if idx >= 8 {
        return Err(PbError::Internal {
            what: "update_raw: leaf index escaped 0..8".into(),
        });
    }
    Ok(idx)
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
    use crate::boosters::{BoosterConfig, NesterovSpec, RefitSpec};
    use crate::cat::{CatEncoderStore, LeakageScheme, Smooth, TsConfig, TsEncodingId};
    use crate::constraints::MonoSign;
    use crate::data::{
        bin_columns, bin_train_columns, BinConfig, CategoricalColumn, NumericColumn,
    };
    use crate::engine::{Booster, HistPrecision};
    use crate::explain::{assert_exact_decomposition, FeatureSet, RefMeasure};
    use crate::loss::{Loss, Poisson, SquaredError};

    fn spec<'a>(loss: &'a dyn Loss) -> FitSpec<'a> {
        FitSpec {
            loss,
            weight: None,
            exposure: None,
            monotone: crate::constraints::MonotoneMap::new(),
            interaction: crate::constraints::InteractionPolicy::default(),
            seed: 0,
        }
    }

    fn binned(cols: &[Vec<f32>]) -> BinnedMatrix {
        let refs: Vec<&[f32]> = cols.iter().map(Vec::as_slice).collect();
        bin_columns(&refs, None, &BinConfig::default(), 0).unwrap()
    }

    fn predict(model: &Model, x: &BinnedMatrix, row: usize) -> f64 {
        let bins: Vec<u8> = x.data.iter().map(|c| c[row]).collect();
        model.ensemble_f64(&bins).unwrap()
    }

    /// Gate G2 (exact): an additive piecewise-constant target on 2 features is
    /// recovered to float tolerance. With λ=0, lr=1 the first depth-2 tree fits the 4
    /// regions exactly (each leaf = its region's value), so recovery is essentially bit-exact.
    #[test]
    fn g2_recovers_piecewise_constant_target_exactly() {
        let n = 60usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 3.0 { 10.0 } else { 20.0 };
                let b = if x1[i] <= 2.0 { 5.0 } else { 0.0 };
                a + b
            })
            .collect();
        let x = binned(&[x0, x1]);
        let booster = Booster::with_config(Config {
            n_trees: 20,
            learning_rate: 1.0,
            lambda: 0.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: Default::default(),
        });
        let sqe = SquaredError;
        let model = booster.fit(&x, &y, &spec(&sqe)).unwrap();
        assert_eq!(model.mode, ExactnessMode::Exact);
        for (i, &yi) in y.iter().enumerate() {
            let pred = predict(&model, &x, i);
            assert!(
                (pred - f64::from(yi)).abs() < 1e-3,
                "row {i}: pred {pred} != y {yi}"
            );
        }
    }

    /// Gate G2 (regularized convergence): with λ=1 the iterative loop converges to the
    /// target over many shrunken trees (exercises multi-round boosting, not 1-tree exactness).
    #[test]
    fn g2_converges_under_regularization() {
        let n = 80usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 3 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| if x0[i] <= 2.0 { -3.0 } else { 4.0 } + if x1[i] <= 1.0 { 1.0 } else { -1.0 })
            .collect();
        let x = binned(&[x0, x1]);
        let booster = Booster::with_config(Config {
            n_trees: 300,
            learning_rate: 0.3,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: Default::default(),
        });
        let sqe = SquaredError;
        let model = booster.fit(&x, &y, &spec(&sqe)).unwrap();
        let max_err = (0..n)
            .map(|i| (predict(&model, &x, i) - f64::from(y[i])).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 0.05, "did not converge: max_err {max_err}");
    }

    #[test]
    fn fitted_model_is_byte_identical_across_thread_counts() {
        let n = 300usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 3 + 1) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32 % 11.0) - 5.0).collect();
        let x = binned(&[x0, x1, x2]);
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let booster = Booster::with_config(Config {
                    n_trees: 40,
                    learning_rate: 0.3,
                    lambda: 1.0,
                    min_split_gain: 0.0,
                    max_delta_step: None,
                    sampling: Default::default(),
                    hist_precision: Default::default(),
                    boosters: Default::default(),
                });
                let sqe = SquaredError;
                let model = booster.fit(&x, &y, &spec(&sqe)).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert!(!b1.is_empty());
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    #[test]
    fn random_strength_fit_is_byte_identical_across_thread_counts() {
        let n = 240usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                (if x0[i] <= 3.0 { -2.0 } else { 3.0 })
                    + if x1[i] <= 2.0 { 1.5 } else { -1.0 }
                    + (i % 7) as f32 * 0.05
            })
            .collect();
        let x = binned(&[x0, x1, x2]);
        let cfg = Config {
            n_trees: 35,
            learning_rate: 0.25,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: BoosterConfig {
                random_strength: 0.35,
                ..BoosterConfig::default()
            },
        };
        let sqe = SquaredError;
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let model = Booster::with_config(cfg.clone())
                    .fit(&x, &y, &spec(&sqe))
                    .unwrap();
                let serve = crate::data::ServeBinnedMatrix(x.clone());
                let bank = model.explain(&serve, RefMeasure::default()).unwrap();
                assert_exact_decomposition(&model, &bank, &serve).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert!(!b1.is_empty());
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    #[test]
    fn ridge_refit_improves_deviance_and_preserves_exact_determinism() {
        let n = 180usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                (if x0[i] <= 3.0 { -2.0 } else { 2.5 })
                    + if x1[i] <= 2.0 { 1.25 } else { -0.75 }
                    + if x2[i] <= 2.0 { 0.5 } else { -0.25 }
                    + (i % 9) as f32 * 0.02
            })
            .collect();
        let x = binned(&[x0, x1, x2]);
        let base_cfg = Config {
            n_trees: 8,
            learning_rate: 0.12,
            lambda: 2.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: Default::default(),
        };
        let refit_cfg = Config {
            boosters: BoosterConfig {
                refit_leaves: RefitSpec::Ridge {
                    l2: 1.0e-3,
                    max_iter: 3,
                    every_k_trees: None,
                },
                ..BoosterConfig::default()
            },
            ..base_cfg.clone()
        };
        let sqe = SquaredError;
        let base = Booster::with_config(base_cfg)
            .fit(&x, &y, &spec(&sqe))
            .unwrap();
        let refit = Booster::with_config(refit_cfg.clone())
            .fit(&x, &y, &spec(&sqe))
            .unwrap();
        let w = vec![1.0_f32; y.len()];
        let base_pred = base.predict_binned(&x, None).unwrap();
        let refit_pred = refit.predict_binned(&x, None).unwrap();
        let base_dev = sqe.deviance(&y, &base_pred, &w).unwrap();
        let refit_dev = sqe.deviance(&y, &refit_pred, &w).unwrap();
        assert!(
            refit_dev < base_dev,
            "refit deviance {refit_dev} should improve base {base_dev}"
        );

        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let model = Booster::with_config(refit_cfg.clone())
                    .fit(&x, &y, &spec(&sqe))
                    .unwrap();
                let serve = crate::data::ServeBinnedMatrix(x.clone());
                let bank = model.explain(&serve, RefMeasure::default()).unwrap();
                assert_exact_decomposition(&model, &bank, &serve).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    #[test]
    fn ridge_refit_is_near_noop_on_exact_fit() {
        let n = 64usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 2 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| ((i / 2) % 2 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                (if x0[i] <= 1.0 { -2.0 } else { 3.0 }) + if x1[i] <= 1.0 { 0.5 } else { -1.5 }
            })
            .collect();
        let x = binned(&[x0, x1]);
        let base_cfg = Config {
            n_trees: 4,
            learning_rate: 1.0,
            lambda: 0.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: Default::default(),
        };
        let refit_cfg = Config {
            boosters: BoosterConfig {
                refit_leaves: RefitSpec::Ridge {
                    l2: 0.0,
                    max_iter: 2,
                    every_k_trees: Some(2),
                },
                ..BoosterConfig::default()
            },
            ..base_cfg.clone()
        };
        let sqe = SquaredError;
        let base = Booster::with_config(base_cfg)
            .fit(&x, &y, &spec(&sqe))
            .unwrap();
        let refit = Booster::with_config(refit_cfg)
            .fit(&x, &y, &spec(&sqe))
            .unwrap();
        let base_pred = base.predict_binned(&x, None).unwrap();
        let refit_pred = refit.predict_binned(&x, None).unwrap();
        for (a, b) in base_pred.iter().zip(refit_pred) {
            assert!(
                (f64::from(*a) - f64::from(b)).abs() < 1.0e-5,
                "refit moved an exact score: {a} vs {b}"
            );
        }
    }

    #[test]
    fn agbm_fit_is_alpha_folded_exact_and_thread_deterministic() {
        let n = 220usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                (if x0[i] <= 3.0 { -1.5 } else { 2.0 })
                    + if x1[i] <= 2.0 { 1.0 } else { -0.75 }
                    + (i % 11) as f32 * 0.03
            })
            .collect();
        let x = binned(&[x0, x1, x2]);
        let cfg = Config {
            n_trees: 30,
            learning_rate: 0.18,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: BoosterConfig {
                nesterov: NesterovSpec::Agbm {
                    momentum_correction: false,
                },
                ..BoosterConfig::default()
            },
        };
        let sqe = SquaredError;
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let model = Booster::with_config(cfg.clone())
                    .fit(&x, &y, &spec(&sqe))
                    .unwrap();
                assert!(
                    model
                        .trees
                        .iter()
                        .any(|(alpha, _)| (*alpha - 1.0).abs() > 1.0e-6),
                    "AGBM should fold non-unit alphas into the plain model"
                );
                let serve = crate::data::ServeBinnedMatrix(x.clone());
                let bank = model.explain(&serve, RefMeasure::default()).unwrap();
                assert_exact_decomposition(&model, &bank, &serve).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    #[test]
    fn agbm_momentum_correction_stays_exact() {
        let n = 120usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| (if x0[i] <= 3.0 { 0.0 } else { 2.0 }) + (i % 5) as f32 * 0.1)
            .collect();
        let x = binned(&[x0, x1]);
        let sqe = SquaredError;
        let model = Booster::with_config(Config {
            n_trees: 8,
            learning_rate: 0.2,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: BoosterConfig {
                nesterov: NesterovSpec::Agbm {
                    momentum_correction: true,
                },
                ..BoosterConfig::default()
            },
        })
        .fit(&x, &y, &spec(&sqe))
        .unwrap();
        assert!(
            model.trees.len() > 8,
            "momentum correction should append correction trees when splits remain"
        );
        let serve = crate::data::ServeBinnedMatrix(x);
        let bank = model.explain(&serve, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&model, &bank, &serve).unwrap();
    }

    #[test]
    fn realized_split_borders_are_a_subset_of_the_grid() {
        // Every split's bin_le indexes a real interior border of the persisted grid
        // (§03.5): bin_le ∈ 1..=borders.len(). This is the I2 precondition.
        let n = 100usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 6) as f32 * 2.0).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32 % 9.0) - 4.0).collect();
        let x = binned(&[x0, x1]);
        let booster = Booster::with_config(Config {
            n_trees: 50,
            learning_rate: 0.3,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: Default::default(),
        });
        let sqe = SquaredError;
        let model = booster.fit(&x, &y, &spec(&sqe)).unwrap();
        for (_, tree) in &model.trees {
            for split in &tree.splits {
                let grid = &model.grids[split.axis as usize];
                assert!(
                    split.bin_le >= 1 && usize::from(split.bin_le) <= grid.borders.len(),
                    "bin_le {} outside 1..={} for axis {}",
                    split.bin_le,
                    grid.borders.len(),
                    split.axis
                );
            }
        }
    }

    #[test]
    fn degenerate_inputs_give_valid_finite_exact_models() {
        let sqe = SquaredError;
        let booster = Booster::with_config(Config {
            n_trees: 10,
            learning_rate: 0.5,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: Default::default(),
        });

        // (1) Constant target ⇒ no split ⇒ 0 trees ⇒ every prediction == f0 == mean.
        let x = binned(&[vec![1.0, 2.0, 3.0, 4.0]]);
        let model = booster.fit(&x, &[5.0, 5.0, 5.0, 5.0], &spec(&sqe)).unwrap();
        assert!(model.trees.is_empty());
        assert_eq!(model.mode, ExactnessMode::Exact);
        for i in 0..4 {
            assert!((predict(&model, &x, i) - 5.0).abs() < 1e-6);
        }

        // (2) Single row.
        let x1 = binned(&[vec![1.0]]);
        let m1 = booster.fit(&x1, &[7.0], &spec(&sqe)).unwrap();
        assert!((predict(&m1, &x1, 0) - 7.0).abs() < 1e-6);

        // (3) An all-missing (all-NaN) column alongside an informative one: the
        // missing axis is never split (degenerate grid), the model stays finite/Exact.
        let informative: Vec<f32> = (0..20).map(|i| (i % 4) as f32).collect();
        let all_missing: Vec<f32> = vec![f32::NAN; 20];
        let yv: Vec<f32> = (0..20)
            .map(|i| if i % 4 < 2 { -1.0 } else { 2.0 })
            .collect();
        let x3 = binned(&[informative, all_missing]);
        let m3 = booster.fit(&x3, &yv, &spec(&sqe)).unwrap();
        assert_eq!(m3.mode, ExactnessMode::Exact);
        for i in 0..20 {
            assert!(predict(&m3, &x3, i).is_finite());
        }
        // No split ever lands on the all-missing axis (axis 1).
        for (_, tree) in &m3.trees {
            assert!(tree.splits.iter().all(|s| s.axis != 1));
        }
    }

    #[test]
    fn bad_config_and_shape_errors() {
        let sqe = SquaredError;
        let x = binned(&[vec![1.0, 2.0]]);
        // n_trees = 0.
        let bad = Booster::with_config(Config {
            n_trees: 0,
            learning_rate: 0.1,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: Default::default(),
        });
        assert!(matches!(
            bad.fit(&x, &[1.0, 2.0], &spec(&sqe)),
            Err(PbError::InvalidConfig { .. })
        ));
        let bad = Booster::with_config(Config {
            boosters: BoosterConfig {
                random_strength: -1.0,
                ..BoosterConfig::default()
            },
            ..Config::default()
        });
        assert!(matches!(
            bad.fit(&x, &[1.0, 2.0], &spec(&sqe)),
            Err(PbError::InvalidConfig { .. })
        ));
        // y length mismatch.
        let ok = Booster::new();
        assert!(matches!(
            ok.fit(&x, &[1.0], &spec(&sqe)),
            Err(PbError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn unsupported_future_fit_spec_knobs_are_rejected() {
        let sqe = SquaredError;
        let x = binned(&[vec![1.0, 2.0]]);
        let y = [1.0_f32, 2.0];
        let booster = Booster::new();

        let mut s = spec(&sqe);
        s.interaction.max_order = 0;
        assert!(matches!(
            booster.fit(&x, &y, &s),
            Err(PbError::InvalidConfig { .. })
        ));

        let mut s = spec(&sqe);
        s.interaction.max_order = 4;
        assert!(matches!(
            booster.fit(&x, &y, &s),
            Err(PbError::InvalidConfig { .. })
        ));

        let mut s = spec(&sqe);
        s.interaction.groups = Some(Vec::new());
        assert!(matches!(
            booster.fit(&x, &y, &s),
            Err(PbError::InvalidConfig { .. })
        ));

        let mut s = spec(&sqe);
        s.interaction.max_order = 1;
        s.interaction.groups = Some(vec![FeatureSet::new(&[0, 1])]);
        assert!(matches!(
            booster.fit(&x, &y, &s),
            Err(PbError::InvalidConfig { .. })
        ));

        let mut s = spec(&sqe);
        s.interaction.groups = Some(vec![FeatureSet::new(&[0])]);
        assert!(booster.fit(&x, &y, &s).is_ok());

        let mut s = spec(&sqe);
        s.monotone.insert("f0".into(), MonoSign::Increasing);
        assert!(booster.fit(&x, &y, &s).is_ok());

        let mut s = spec(&sqe);
        s.monotone
            .insert("unknown_feature".into(), MonoSign::Increasing);
        assert!(matches!(
            booster.fit(&x, &y, &s),
            Err(PbError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn monotone_constraint_admits_compatible_splits_and_rejects_opposite_direction() {
        let x = binned(&[vec![1.0, 1.0, 2.0, 2.0]]);
        let booster = Booster::with_config(Config {
            n_trees: 1,
            learning_rate: 1.0,
            lambda: 0.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: Default::default(),
        });
        let sqe = SquaredError;

        let mut inc = spec(&sqe);
        inc.monotone.insert("f0".into(), MonoSign::Increasing);
        let increasing = [0.0_f32, 0.0, 10.0, 10.0];
        let model = booster.fit(&x, &increasing, &inc).unwrap();
        assert_eq!(model.trees.len(), 1);
        assert!(predict(&model, &x, 0) <= predict(&model, &x, 2));

        let anti_monotone = [10.0_f32, 10.0, 0.0, 0.0];
        let model = booster.fit(&x, &anti_monotone, &inc).unwrap();
        assert!(
            model.trees.is_empty(),
            "anti-monotone split should terminate gracefully"
        );
        assert_eq!(predict(&model, &x, 0), predict(&model, &x, 2));
    }

    #[test]
    fn malformed_binned_matrix_errors_at_fit_boundary() {
        let sqe = SquaredError;
        let x = binned(&[vec![1.0, 2.0, 3.0]]);
        let y = [1.0_f32, 2.0, 3.0];
        let booster = Booster::new();

        let mut bad = x.clone();
        bad.data[0].push(1);
        assert!(matches!(
            booster.fit(&bad, &y, &spec(&sqe)),
            Err(PbError::ShapeMismatch { .. })
        ));

        let mut bad = x.clone();
        bad.grids.pop();
        assert!(matches!(
            booster.fit(&bad, &y, &spec(&sqe)),
            Err(PbError::ShapeMismatch { .. })
        ));

        let mut bad = x.clone();
        bad.provenance.pop();
        assert!(matches!(
            booster.fit(&bad, &y, &spec(&sqe)),
            Err(PbError::ShapeMismatch { .. })
        ));

        let mut bad = x.clone();
        bad.grids[0].n_bins = 0;
        assert!(matches!(
            booster.fit(&bad, &y, &spec(&sqe)),
            Err(PbError::InvalidInput { .. })
        ));

        let mut bad = x.clone();
        bad.grids[0].borders.clear();
        assert!(matches!(
            booster.fit(&bad, &y, &spec(&sqe)),
            Err(PbError::InvalidInput { .. })
        ));

        let mut bad = x;
        bad.data[0][0] = u8::MAX;
        assert!(matches!(
            booster.fit(&bad, &y, &spec(&sqe)),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn fit_train_persists_categorical_encoder_store() {
        let numeric = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let levels = vec!["low", "high", "low", "high", "mid", "mid"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [1.0_f32, 10.0, 2.0, 12.0, 5.0, 6.0];
        let ts = TsConfig {
            leakage: LeakageScheme::KFold { k: 3 },
            smooth: Smooth::Fixed { m: 0.0 },
            min_data_per_group: 0.0,
            ..TsConfig::default()
        };
        let fitted = bin_train_columns(
            &[NumericColumn {
                raw: crate::data::FeatureId(0),
                values: &numeric,
            }],
            &[CategoricalColumn {
                raw: crate::data::FeatureId(1),
                id: TsEncodingId(0),
                levels: &levels,
                config: &ts,
            }],
            &y,
            None,
            None,
            &BinConfig::default(),
            12,
        )
        .unwrap();
        let sqe = SquaredError;
        let booster = Booster::with_config(Config {
            n_trees: 3,
            learning_rate: 1.0,
            lambda: 0.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: Default::default(),
        });
        let model = booster
            .fit_train(&fitted.train, &y, &spec(&sqe), fitted.cat_encoders.clone())
            .unwrap();
        assert_eq!(model.schema.cat_encoders.len(), 1);
        model.validate().unwrap();

        assert!(matches!(
            booster.fit_train(&fitted.train, &y, &spec(&sqe), CatEncoderStore::new()),
            Err(PbError::InvalidInput { .. })
        ));
    }

    fn additive_2feat(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 3.0 { 10.0 } else { 20.0 };
                let b = if x1[i] <= 2.0 { 5.0 } else { 0.0 };
                a + b
            })
            .collect();
        (x0, x1, y)
    }

    #[test]
    fn per_tree_scorers_agree_bit_exactly() {
        // The column-major update scorer (tree_value_for_row) and the row-vector
        // ObliviousTree::lookup MUST produce bit-identical PER-TREE leaf values — both
        // fold the leaf index via the canonical low_bit. This is the structural
        // invariant that makes "the model scores what it trained on" hold; the f32
        // train sum and the f64 ensemble sum then differ only by accumulation WIDTH.
        let (x0, x1, y) = additive_2feat(60);
        let x = binned(&[x0, x1]);
        let sqe = SquaredError;
        let model = Booster::with_config(Config {
            n_trees: 30,
            learning_rate: 0.3,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: Default::default(),
        })
        .fit(&x, &y, &spec(&sqe))
        .unwrap();
        for r in 0..x.n_rows as usize {
            let bins: Vec<u8> = x.data.iter().map(|c| c[r]).collect();
            for (_, tree) in &model.trees {
                assert_eq!(
                    tree_value_for_row(tree, &x, r).unwrap(),
                    tree.lookup(&bins).unwrap()
                );
            }
        }
    }

    #[test]
    fn f32_train_raw_matches_f64_ensemble_within_reconstruction_tol() {
        // Training optimizes an f32-accumulated raw (the §05 `grad_hess` takes
        // `raw: &[f32]`); ensemble_f64 / the §08 tables accumulate in f64. The two
        // agree within ~4·n_trees·f32::EPSILON·magnitude — exactly the tolerance the
        // §08 Reconstruction gate is sized for, NOT a routing/structural bug.
        let (x0, x1, y) = additive_2feat(80);
        let x = binned(&[x0, x1]);
        let sqe = SquaredError;
        let model = Booster::with_config(Config {
            n_trees: 200,
            learning_rate: 0.3,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: Default::default(),
        })
        .fit(&x, &y, &spec(&sqe))
        .unwrap();
        let n_trees = model.trees.len() as f64;
        for r in 0..x.n_rows as usize {
            let mut raw_f32: f32 = model.f0;
            for (_, tree) in &model.trees {
                raw_f32 += tree_value_for_row(tree, &x, r).unwrap();
            }
            let bins: Vec<u8> = x.data.iter().map(|c| c[r]).collect();
            let ens = model.ensemble_f64(&bins).unwrap();
            let tol = 4.0 * n_trees * f64::from(f32::EPSILON) * (1.0 + ens.abs());
            assert!(
                (f64::from(raw_f32) - ens).abs() <= tol,
                "raw_f32 {raw_f32} vs ensemble_f64 {ens} exceeds recon tol {tol}"
            );
        }
    }

    #[test]
    fn weighted_fit_recovers_target() {
        // Weights scale (g,h) and the init mean; an exactly-representable target is
        // still recovered (each region's WEIGHTED mean equals its constant). λ=0,lr=1.
        let (x0, x1, y) = additive_2feat(60);
        let x = binned(&[x0, x1]);
        let w: Vec<f32> = (0..y.len()).map(|i| 0.5 + (i % 4) as f32).collect();
        let sqe = SquaredError;
        let mut s = spec(&sqe);
        s.weight = Some(&w);
        let model = Booster::with_config(Config {
            n_trees: 20,
            learning_rate: 1.0,
            lambda: 0.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: Default::default(),
        })
        .fit(&x, &y, &s)
        .unwrap();
        for (i, &yi) in y.iter().enumerate() {
            assert!((predict(&model, &x, i) - f64::from(yi)).abs() < 1e-3);
        }
    }

    #[test]
    fn exposure_fit_produces_finite_exact_model() {
        // Smoke test of the offset path: exposure → offset = ln(e) folded into raw,
        // offset-aware init_score. A valid finite Exact model results.
        let (x0, x1, y) = additive_2feat(40);
        let x = binned(&[x0, x1]);
        let e: Vec<f32> = (0..y.len()).map(|i| 1.0 + (i % 3) as f32 * 0.5).collect();
        let sqe = SquaredError;
        let mut s = spec(&sqe);
        s.exposure = Some(&e);
        let model = Booster::with_config(Config {
            n_trees: 15,
            learning_rate: 0.3,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: Default::default(),
        })
        .fit(&x, &y, &s)
        .unwrap();
        assert_eq!(model.mode, ExactnessMode::Exact);
        for i in 0..x.n_rows as usize {
            assert!(predict(&model, &x, i).is_finite());
        }
    }

    #[test]
    fn reanchor_shifts_only_intercept_and_matches_response_total() {
        let n = 96usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (1 + i % 7) as f32).collect();
        let x = binned(&[x0, x1]);
        let base_cfg = Config {
            n_trees: 8,
            learning_rate: 0.4,
            lambda: 2.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: Default::default(),
        };
        let anchored_cfg = Config {
            boosters: BoosterConfig {
                reanchor: true,
                ..BoosterConfig::default()
            },
            ..base_cfg.clone()
        };
        let base = Booster::with_config(base_cfg)
            .fit(&x, &y, &spec(&Poisson))
            .unwrap();
        let anchored = Booster::with_config(anchored_cfg)
            .fit(&x, &y, &spec(&Poisson))
            .unwrap();

        assert_eq!(base.trees, anchored.trees);
        assert_ne!(base.f0, anchored.f0);
        let observed: f64 = y.iter().map(|&yi| f64::from(yi)).sum();
        let predicted: f64 = anchored
            .predict_binned(&x, None)
            .unwrap()
            .iter()
            .map(|&yi| f64::from(yi))
            .sum();
        assert!((predicted - observed).abs() < 1.0e-3);

        let serve = crate::data::ServeBinnedMatrix(x);
        let bank = anchored.explain(&serve, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&anchored, &bank, &serve).unwrap();
    }

    #[test]
    fn g2_recovers_three_feature_target() {
        // Order-3: an additive piecewise-constant target on 3 features, recovered
        // exactly with λ=0 (a full depth-3 tree captures the 8 regions).
        let n = 64usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 2 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| ((i / 2) % 2 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| ((i / 4) % 2 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 1.0 { 0.0 } else { 4.0 };
                let b = if x1[i] <= 1.0 { 0.0 } else { 2.0 };
                let c = if x2[i] <= 1.0 { 0.0 } else { 1.0 };
                a + b + c - 3.5
            })
            .collect();
        let x = binned(&[x0, x1, x2]);
        let sqe = SquaredError;
        let model = Booster::with_config(Config {
            n_trees: 20,
            learning_rate: 1.0,
            lambda: 0.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            boosters: Default::default(),
        })
        .fit(&x, &y, &spec(&sqe))
        .unwrap();
        for (i, &yi) in y.iter().enumerate() {
            assert!(
                (predict(&model, &x, i) - f64::from(yi)).abs() < 1e-3,
                "row {i}: {} != {yi}",
                predict(&model, &x, i)
            );
        }
    }

    #[test]
    fn mvs_sampler_is_deterministic_and_gradient_weighted() {
        let rows: Vec<u32> = (0..20).collect();
        let gh = GradHess {
            g: (0..20)
                .map(|i| if i < 5 { 20.0 - i as f32 } else { 0.1 })
                .collect(),
            h: vec![1.0; 20],
        };
        let sampling = Sampling::Mvs {
            rate: 0.25,
            min_rows: 4,
        };
        let a = sample_rows(&sampling, &gh, 123, 7, &rows).unwrap();
        let b = sample_rows(&sampling, &gh, 123, 7, &rows).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 5);
        let sampled_mean_g: f32 =
            a.iter().map(|&r| gh.g[r as usize].abs()).sum::<f32>() / a.len() as f32;
        let population_mean_g: f32 = gh.g.iter().map(|g| g.abs()).sum::<f32>() / gh.g.len() as f32;
        assert!(sampled_mean_g > population_mean_g);
    }

    #[test]
    fn mvs_config_validation_is_fail_closed() {
        for sampling in [
            Sampling::Mvs {
                rate: 0.0,
                min_rows: 1,
            },
            Sampling::Mvs {
                rate: 1.1,
                min_rows: 1,
            },
            Sampling::Mvs {
                rate: 0.5,
                min_rows: 0,
            },
        ] {
            let cfg = Config {
                sampling,
                ..Config::default()
            };
            assert!(matches!(cfg.validate(), Err(PbError::InvalidConfig { .. })));
        }
    }

    #[test]
    fn mvs_fit_stays_exact_and_thread_deterministic() {
        let n = 180usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 9 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 4.0 { 2.0 } else { -1.0 };
                let b = if x1[i] <= 3.0 { 1.5 } else { -0.5 };
                let c = if x2[i] <= 2.0 { 0.75 } else { -0.25 };
                a + b + c
            })
            .collect();
        let x = binned(&[x0, x1, x2]);
        let cfg = Config {
            n_trees: 40,
            learning_rate: 0.3,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Sampling::Mvs {
                rate: 0.6,
                min_rows: 40,
            },
            hist_precision: Default::default(),
            boosters: Default::default(),
        };
        let sqe = SquaredError;
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let model = Booster::with_config(cfg.clone())
                    .fit(&x, &y, &spec(&sqe))
                    .unwrap();
                let serve = crate::data::ServeBinnedMatrix(x.clone());
                let bank = model.explain(&serve, RefMeasure::default()).unwrap();
                assert_exact_decomposition(&model, &bank, &serve).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    #[test]
    fn quantized_hist_fit_stays_exact_and_thread_deterministic() {
        let n = 160usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                (if x0[i] <= 4.0 { -2.0 } else { 3.0 }) + if x1[i] <= 3.0 { 1.0 } else { -1.0 }
            })
            .collect();
        let x = binned(&[x0, x1]);
        let cfg = Config {
            n_trees: 30,
            learning_rate: 0.3,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Sampling::Full,
            hist_precision: HistPrecision::QuantizedI32,
            boosters: Default::default(),
        };
        let sqe = SquaredError;
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let model = Booster::with_config(cfg.clone())
                    .fit(&x, &y, &spec(&sqe))
                    .unwrap();
                let serve = crate::data::ServeBinnedMatrix(x.clone());
                let bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
                assert_exact_decomposition(&model, &bank, &serve).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }
}
