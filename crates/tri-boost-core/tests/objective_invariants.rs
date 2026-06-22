//! The §13.1 invariant matrix, objective columns (plan M2 / Gate G4): every v1 loss,
//! fit end-to-end, produces an `Exact` model whose purified `TableBank` passes all five
//! I2 checks under both reference measures. Because a loss is orthogonal to tree shape
//! (§05.8 — it sees only `(y, F, w)`), the firewall stays `Exact` by construction; this
//! gate *proves* it per objective rather than asserting it. Also exercises the Poisson
//! `max_delta_step` stability path and the exposure-offset frequency fit.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use tri_boost_core::{
    assert_exact_decomposition, bin_columns, BinConfig, Booster, Config, ExactnessMode, FitSpec,
    Gamma, InteractionPolicy, Link, Logistic, Loss, LossId, MonotoneMap, Poisson, RefMeasure,
    ServeBinnedMatrix, SquaredError, Tweedie,
};

fn spec<'a>(loss: &'a dyn Loss, exposure: Option<&'a [f32]>) -> FitSpec<'a> {
    FitSpec {
        loss,
        weight: None,
        exposure,
        monotone: MonotoneMap::new(),
        interaction: InteractionPolicy::default(),
        seed: 0,
    }
}

fn cfg() -> Config {
    Config {
        n_trees: 25,
        learning_rate: 0.3,
        lambda: 1.0,
        min_split_gain: 0.0,
        max_delta_step: None,
        sampling: Default::default(),
        hist_precision: Default::default(),
    }
}

/// Two integer features over `n` rows, binned once and reused across objectives.
fn features(n: usize) -> (Vec<f32>, Vec<f32>) {
    let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
    let x1: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
    (x0, x1)
}

#[test]
fn g4_every_objective_passes_the_five_invariant_checks() {
    let n = 96usize;
    let (x0, x1) = features(n);

    // Per-objective in-domain targets (a real ≤order-2 structure on each link scale).
    let y_sqe: Vec<f32> = (0..n).map(|i| x0[i] + 2.0 * x1[i]).collect();
    let y_logit: Vec<f32> = (0..n)
        .map(|i| {
            if (x0[i] <= 3.0) ^ (x1[i] <= 2.0) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let y_pois: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect(); // counts ≥ 0 (incl. 0)
    let y_gamma: Vec<f32> = (0..n).map(|i| 1.0 + x0[i] + 0.5 * x1[i]).collect(); // > 0
    let y_tweedie: Vec<f32> = (0..n)
        .map(|i| (x0[i] + x1[i] - 4.0).max(0.0)) // ≥ 0, with genuine zeros
        .collect();

    let tweedie = Tweedie::new(1.5).unwrap();
    let cases: Vec<(&str, &dyn Loss, &[f32], Link)> = vec![
        ("squared_error", &SquaredError, &y_sqe, Link::Identity),
        ("logistic", &Logistic, &y_logit, Link::Logit),
        ("poisson", &Poisson, &y_pois, Link::Log),
        ("gamma", &Gamma, &y_gamma, Link::Log),
        ("tweedie", &tweedie, &y_tweedie, Link::Log),
    ];

    for (name, loss, y, link) in cases {
        let refs: Vec<&[f32]> = vec![&x0, &x1];
        let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
        let model = Booster::with_config(cfg())
            .fit(&x, y, &spec(loss, None))
            .unwrap_or_else(|e| panic!("{name} fit failed: {e:?}"));
        assert_eq!(model.mode, ExactnessMode::Exact, "{name} must be Exact");
        assert_eq!(model.link, link, "{name} link");
        assert_eq!(model.schema.objective.link, link);

        let serve = ServeBinnedMatrix(x);
        for w in [RefMeasure::default(), RefMeasure::Uniform] {
            let bank = model
                .explain(&serve, w.clone())
                .unwrap_or_else(|e| panic!("{name} explain({w:?}) failed: {e:?}"));
            assert_exact_decomposition(&model, &bank, &serve)
                .unwrap_or_else(|e| panic!("{name} I2 gates failed under {w:?}: {e:?}"));
        }
    }
}

#[test]
fn poisson_default_fit_is_max_delta_step_stabilized() {
    // Poisson advertises max_delta_step = Some(0.7); with no Config override the engine
    // resolves it, so every per-tree leaf step is capped at lr·0.7 — the fit stays finite
    // even on data that would otherwise drive exp(F) explosive.
    let n = 80usize;
    let (x0, x1) = features(n);
    let y: Vec<f32> = (0..n).map(|i| (i % 9) as f32).collect();
    let refs: Vec<&[f32]> = vec![&x0, &x1];
    let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
    let lr = 0.5_f32;
    let model = Booster::with_config(Config {
        n_trees: 30,
        learning_rate: lr,
        lambda: 0.1,
        min_split_gain: 0.0,
        max_delta_step: None, // ⇒ falls back to Poisson's Some(0.7)
        sampling: Default::default(),
        hist_precision: Default::default(),
    })
    .fit(&x, &y, &spec(&Poisson, None))
    .unwrap();
    assert_eq!(model.schema.objective.loss, LossId::Poisson);
    // Every leaf value is bounded by lr·δ = 0.5·0.7 = 0.35 (the clamp is per Newton step).
    let bound = f64::from(lr) * 0.7 + 1e-5;
    for (_, tree) in &model.trees {
        for &v in &tree.leaves {
            assert!(
                f64::from(v).abs() <= bound,
                "leaf {v} exceeds lr·δ bound {bound}"
            );
        }
    }
    // And predictions are finite everywhere.
    for r in 0..n {
        let bins: Vec<u8> = x.data.iter().map(|c| c[r]).collect();
        assert!(model.ensemble_f64(&bins).unwrap().is_finite());
    }
}

#[test]
fn poisson_exposure_fit_explains_exactly() {
    // The exposure-offset frequency path: offset = log(e) folded into raw, exposure-
    // weighted intercept. The fitted model still decomposes losslessly (G4 + §05.5).
    let n = 64usize;
    let (x0, x1) = features(n);
    let y: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
    let e: Vec<f32> = (0..n).map(|i| 0.5 + (i % 3) as f32).collect(); // exposure > 0
    let refs: Vec<&[f32]> = vec![&x0, &x1];
    let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
    let model = Booster::with_config(cfg())
        .fit(&x, &y, &spec(&Poisson, Some(&e)))
        .unwrap();
    assert_eq!(model.mode, ExactnessMode::Exact);
    let serve = ServeBinnedMatrix(x);
    let bank = model.explain(&serve, RefMeasure::default()).unwrap();
    assert_exact_decomposition(&model, &bank, &serve).unwrap();
}
