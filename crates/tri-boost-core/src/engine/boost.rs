//! The boosting loop (spec §06.6, milestone M1.5) — the Phase-2 capstone that ties
//! the objective (§05), binning (§03), histogram engine (§06.3) and split-finder
//! (§06.2) into `Booster::fit`.
//!
//! `f0 = link(weighted mean)` (the fANOVA intercept), then each round: one
//! full-precision `grad_hess` pass w.r.t. the current raw score → `grow_oblivious_tree`
//! → `update_raw`, until `n_trees` rounds or a round cannot split (graceful stop).
//! Every tree carries `alpha = 1.0`. The loop emits `ExactnessMode::Exact`.
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

use crate::backend::{pb_rng, Stage};
use crate::cat::CatEncoderStore;
use crate::data::{compute_offset, BinnedMatrix};
use crate::engine::split::{grow_oblivious_tree, GrowConfig};
use crate::engine::{low_bit, Config, ExactnessMode, FitSpec, Model, ModelSchema, ObliviousTree};
use crate::error::PbError;
use crate::loss::GradHess;
use crate::serialize::SCHEMA_VERSION;

fn invalid_config(what: &'static str) -> PbError {
    PbError::InvalidConfig { what: what.into() }
}

fn invalid_input(what: String) -> PbError {
    PbError::InvalidInput { what }
}

fn validate_fit_spec(spec: &FitSpec<'_>) -> Result<(), PbError> {
    if !(1..=3).contains(&spec.interaction.max_order) {
        return Err(PbError::InvalidConfig {
            what: format!(
                "interaction.max_order must be in 1..=3, got {}",
                spec.interaction.max_order
            ),
        });
    }
    if spec.interaction.groups.is_some() {
        return Err(invalid_config(
            "interaction groups are not implemented in the Phase-2 green spine",
        ));
    }
    if !spec.monotone.is_empty() {
        return Err(invalid_config(
            "monotone constraints are not implemented in the Phase-2 green spine",
        ));
    }
    Ok(())
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
    };

    let mut trees: Vec<(f32, ObliviousTree)> = Vec::new();
    let mut gh = GradHess::default();
    for t in 0..config.n_trees {
        // Per-round deterministic re-seed — the seam for MVS/subsampling (M5-QHIST,
        // v1.5). v1 draws no randomness, so the stream is established but unused.
        let _round_rng = pb_rng(spec.seed, t, Stage::Sample, 0);
        spec.loss.grad_hess(y, &raw, weight, &mut gh)?;
        match grow_oblivious_tree(x, &gh, &rows, &axes, &grow_cfg)? {
            Some(tree) => {
                update_raw(&mut raw, x, &tree)?;
                trees.push((1.0, tree));
            }
            // No admissible split clears the floor (e.g. converged / constant target):
            // stop early with what we have — a valid (possibly empty) Exact model.
            None => break,
        }
    }

    let schema = ModelSchema {
        feature_names: (0..n_features).map(|i| format!("f{i}")).collect(),
        feature_kinds: x.provenance.iter().map(|p| p.kind).collect(),
        cat_encoders: CatEncoderStore::new(),
        class_labels: None,
        objective: spec.loss.objective_tag(),
    };
    Ok(Model {
        f0: f0_f32,
        trees,
        grids: x.grids.clone(),
        provenance: x.provenance.clone(),
        link: spec.loss.link(),
        mode: ExactnessMode::Exact,
        schema,
        schema_version: SCHEMA_VERSION,
    })
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
    tree.leaves
        .get(idx)
        .copied()
        .ok_or_else(|| PbError::Internal {
            what: "update_raw: leaf index escaped 0..8".into(),
        })
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
    use crate::constraints::MonoSign;
    use crate::data::{bin_columns, BinConfig};
    use crate::engine::Booster;
    use crate::explain::FeatureSet;
    use crate::loss::SquaredError;

    fn spec<'a>(loss: &'a SquaredError) -> FitSpec<'a> {
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
        s.interaction.groups = Some(vec![FeatureSet::new(&[0])]);
        assert!(matches!(
            booster.fit(&x, &y, &s),
            Err(PbError::InvalidConfig { .. })
        ));

        let mut s = spec(&sqe);
        s.monotone.insert("f0".into(), MonoSign::Increasing);
        assert!(matches!(
            booster.fit(&x, &y, &s),
            Err(PbError::InvalidConfig { .. })
        ));
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
        })
        .fit(&x, &y, &s)
        .unwrap();
        assert_eq!(model.mode, ExactnessMode::Exact);
        for i in 0..x.n_rows as usize {
            assert!(predict(&model, &x, i).is_finite());
        }
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
}
