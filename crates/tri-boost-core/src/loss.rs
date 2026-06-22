//! Objectives & the `Loss` trait (spec §2.4 / §05). The trait + companion types are
//! frozen here; v1 ships `SquaredError` (the green-spine objective). `Logistic`,
//! `Poisson`, `Gamma`, `Tweedie` land in Phase 4.
//!
//! The trait is fully orthogonal to tree shape, so it cannot touch I1/I2. Every
//! method that can fail returns `Result<_, PbError>` (R-LOSSFALLIBLE): a fallible
//! objective maps its failure onto `PbError` rather than panicking.

use crate::error::PbError;
use itertools::izip;
use serde::{Deserialize, Serialize};

/// Per-row first/second derivatives of the loss w.r.t. the raw score `F`
/// (spec §2.3). Full precision; leaves are always refit from these exact values.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct GradHess {
    /// Per-row gradient `∂L/∂F`.
    pub g: Vec<f32>,
    /// Per-row hessian `∂²L/∂F²` (floored at `Loss::hessian_floor`).
    pub h: Vec<f32>,
}

/// The inverse-link family of an objective (spec §2.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Link {
    /// `pred = raw` (regression on the natural scale).
    Identity,
    /// `pred = exp(raw)` (Poisson / Gamma / Tweedie mean).
    Log,
    /// `pred = 1/(1+exp(-raw))` (binary probability).
    Logit,
}

/// The early-stopping / evaluation metric an objective reports (spec §2.4; this is
/// §05's canonical form). Deviance-based by default — never RMSE on
/// Poisson/Gamma/Tweedie.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Metric {
    /// Root mean squared error (squared-error objective only).
    Rmse,
    /// Binary log-loss (logistic objective).
    LogLoss,
    /// Poisson deviance.
    PoissonDeviance,
    /// Gamma deviance.
    GammaDeviance,
    /// Tweedie deviance with power parameter `rho`.
    TweedieDeviance {
        /// Tweedie variance power, `1 < rho < 2`.
        rho: f32,
    },
}

/// Stable identifier for a concrete objective, recorded in [`ObjectiveTag`] so a
/// loaded `Model` can reproduce its link + loss (spec §05).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossId {
    /// `SquaredError` (identity link).
    SquaredError,
    /// `Logistic` (logit link).
    Logistic,
    /// `Poisson` (log link).
    Poisson,
    /// `Gamma` (log link).
    Gamma,
    /// `Tweedie` (log link; power in [`ObjectiveTag::tweedie_rho`]).
    Tweedie,
}

/// The trained objective, recorded on the model so export / `predict_proba` can
/// reproduce link + loss without the caller re-supplying anything (spec §2.6, R-SCHEMA).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveTag {
    /// The inverse-link family.
    pub link: Link,
    /// Which concrete objective was trained.
    pub loss: LossId,
    /// Tweedie power, present iff `loss == LossId::Tweedie`.
    pub tweedie_rho: Option<f32>,
}

/// A loss/objective (spec §2.4). Fully orthogonal to tree shape (I1/I2 untouched).
///
/// `grad_hess` is one pass w.r.t. the raw score `F` (after the exposure offset).
/// `init_score`/`deviance` are fallible (R-LOSSFALLIBLE): invalid domains (Gamma
/// `y<=0`, all-zero weights, bad exposure, user-loss failures) return a typed
/// `PbError` rather than panicking or emitting `NaN`.
pub trait Loss: Send + Sync {
    /// One full-precision gradient/hessian pass into `out`. Never panics: a fallible
    /// objective maps its failure onto `PbError`.
    fn grad_hess(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError>;

    /// `link(weighted mean)` — the mandatory `boost_from_average` intercept `f0`.
    /// Full-width `f64` (the exact fANOVA intercept). Fallible on invalid domains.
    fn init_score(&self, y: &[f32], weight: &[f32], offset: Option<&[f32]>)
        -> Result<f64, PbError>;

    /// The inverse-link family of this objective.
    fn link(&self) -> Link;

    /// Inverse link applied to a single raw score (`exp(k·F)`, not `powf`).
    fn pred_from_raw(&self, raw: f32) -> f32;

    /// Strictly-proper deviance for early stopping (NOT RMSE on Poisson/Gamma/Tweedie).
    /// `f64` fold, reported in `f32`. Same invalid-domain typed errors as `init_score`.
    fn deviance(&self, y: &[f32], raw: &[f32], weight: &[f32]) -> Result<f32, PbError>;

    /// The objective's natural early-stopping metric (deviance by default).
    fn default_metric(&self) -> Metric;

    /// Lower clamp on the per-row hessian (numerical floor ε); default `1e-16`
    /// (NaN-guard only — stability is `λ` + `max_delta_step`'s job).
    fn hessian_floor(&self) -> f32 {
        1e-16
    }

    /// Per-objective default leaf-stage `|w*|`-clamp. `None` = uncapped; Poisson ⇒
    /// `Some(0.7)`. `Config.max_delta_step` falls back to this when unset.
    fn max_delta_step(&self) -> Option<f32> {
        None
    }
}

/// Squared-error regression (spec §05.3): Identity link, `g = w·(F − y)`, `h = w·1`,
/// `init_score = weighted mean of y` (offset-aware), half-deviance metric. The v1
/// green-spine objective — chosen first because reconstruction is testable before any
/// nonlinear link. `weight[i]` scales row `i`'s `(g, h)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SquaredError;

impl Loss for SquaredError {
    fn grad_hess(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        let n = y.len();
        if raw.len() != n || weight.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "grad_hess: y={n}, raw={}, weight={}",
                    raw.len(),
                    weight.len()
                ),
            });
        }
        out.g.clear();
        out.g.resize(n, 0.0);
        out.h.clear();
        out.h.resize(n, 0.0);
        let floor = self.hessian_floor();
        // izip! ⇒ no indexing in the hot loop (§05 hot-loop policy). g = w(F−y);
        // h = w·1, floored so the postcondition out.h[i] >= floor holds even at w=0.
        for (gi, hi, &yi, &fi, &wi) in izip!(&mut out.g, &mut out.h, y, raw, weight) {
            *gi = wi * (fi - yi);
            *hi = wi.max(floor);
        }
        Ok(())
    }

    fn init_score(
        &self,
        y: &[f32],
        weight: &[f32],
        offset: Option<&[f32]>,
    ) -> Result<f64, PbError> {
        let n = y.len();
        if weight.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!("init_score: y={n}, weight={}", weight.len()),
            });
        }
        // f64 fixed-order (sequential, index-order) fold ⇒ thread-count-independent.
        // Identity link ⇒ link(weighted mean) = weighted mean. With an offset, the
        // best initial constant is the weighted mean of (y − offset), since the
        // initial raw score is f0 + offset (offset-aware Identity form, §03.7/§05.5).
        let (mut sum_w, mut sum_wy) = (0.0_f64, 0.0_f64);
        match offset {
            Some(off) => {
                if off.len() != n {
                    return Err(PbError::ShapeMismatch {
                        what: format!("init_score: y={n}, offset={}", off.len()),
                    });
                }
                for ((&yi, &wi), &oi) in y.iter().zip(weight).zip(off) {
                    sum_w += f64::from(wi);
                    sum_wy += f64::from(wi) * (f64::from(yi) - f64::from(oi));
                }
            }
            None => {
                for (&yi, &wi) in y.iter().zip(weight) {
                    sum_w += f64::from(wi);
                    sum_wy += f64::from(wi) * f64::from(yi);
                }
            }
        }
        if sum_w <= 0.0 {
            return Err(PbError::InvalidInput {
                what: "squared-error init_score: all-zero (or non-positive) weights".into(),
            });
        }
        Ok(sum_wy / sum_w)
    }

    fn link(&self) -> Link {
        Link::Identity
    }

    fn pred_from_raw(&self, raw: f32) -> f32 {
        raw // Identity inverse link
    }

    fn deviance(&self, y: &[f32], raw: &[f32], weight: &[f32]) -> Result<f32, PbError> {
        let n = y.len();
        if raw.len() != n || weight.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "deviance: y={n}, raw={}, weight={}",
                    raw.len(),
                    weight.len()
                ),
            });
        }
        // Half-deviance = ½ Σ w (raw − y)²  (= ½ MSE·Σw). f64 fold, reported f32.
        let (mut sum_w, mut acc) = (0.0_f64, 0.0_f64);
        for ((&yi, &fi), &wi) in y.iter().zip(raw).zip(weight) {
            sum_w += f64::from(wi);
            let r = f64::from(fi) - f64::from(yi);
            acc += f64::from(wi) * r * r;
        }
        if sum_w <= 0.0 {
            return Err(PbError::InvalidInput {
                what: "squared-error deviance: all-zero (or non-positive) weights".into(),
            });
        }
        Ok((0.5 * acc) as f32)
    }

    fn default_metric(&self) -> Metric {
        Metric::Rmse
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
    use proptest::prelude::*;

    #[test]
    fn grad_hess_matches_closed_form() {
        let sqe = SquaredError;
        let y = [2.0_f32, 4.0];
        let raw = [3.0_f32, 3.0];
        let w = [1.0_f32, 2.0];
        let mut gh = GradHess::default();
        sqe.grad_hess(&y, &raw, &w, &mut gh).unwrap();
        // g = w(F−y): [1·(3−2), 2·(3−4)] = [1, −2]; h = w: [1, 2].
        assert_eq!(gh.g, vec![1.0, -2.0]);
        assert_eq!(gh.h, vec![1.0, 2.0]);
    }

    #[test]
    fn hessian_is_floored_for_zero_weight_rows() {
        let sqe = SquaredError;
        let mut gh = GradHess::default();
        sqe.grad_hess(&[1.0], &[0.0], &[0.0], &mut gh).unwrap();
        assert_eq!(gh.g, vec![0.0]); // 0·(0−1) = 0
        assert!(gh.h[0] >= sqe.hessian_floor() && gh.h[0] > 0.0);
    }

    #[test]
    fn init_score_is_the_weighted_mean() {
        let sqe = SquaredError;
        assert!(
            (sqe.init_score(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], None)
                .unwrap()
                - 2.0)
                .abs()
                < 1e-9
        );
        // weight 0 on the middle row ⇒ mean of {1, 3}.
        assert!(
            (sqe.init_score(&[1.0, 2.0, 3.0], &[1.0, 0.0, 1.0], None)
                .unwrap()
                - 2.0)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn init_score_offset_aware_identity_form() {
        let sqe = SquaredError;
        // f0 = weighted mean of (y − offset) = mean([3−1, 5−1]) = mean([2, 4]) = 3.
        let f0 = sqe
            .init_score(&[3.0, 5.0], &[1.0, 1.0], Some(&[1.0, 1.0]))
            .unwrap();
        assert!((f0 - 3.0).abs() < 1e-9);
    }

    #[test]
    fn init_score_satisfies_first_order_condition() {
        // Σ w·g(y, f0) ≈ 0 at the intercept, with and without offset.
        let sqe = SquaredError;
        let y = [1.0_f32, 5.0, 2.0, 8.0];
        let w = [1.0_f32, 2.0, 0.5, 1.0];

        let f0 = sqe.init_score(&y, &w, None).unwrap() as f32;
        let raw: Vec<f32> = vec![f0; y.len()];
        let mut gh = GradHess::default();
        sqe.grad_hess(&y, &raw, &w, &mut gh).unwrap();
        let sum_g: f64 = gh.g.iter().map(|&g| f64::from(g)).sum();
        assert!(sum_g.abs() < 1e-4, "Σ w·g should vanish at f0, got {sum_g}");

        let off = [0.5_f32, -1.0, 2.0, 0.0];
        let f0o = sqe.init_score(&y, &w, Some(&off)).unwrap() as f32;
        let raw_o: Vec<f32> = off.iter().map(|&o| f0o + o).collect();
        sqe.grad_hess(&y, &raw_o, &w, &mut gh).unwrap();
        let sum_go: f64 = gh.g.iter().map(|&g| f64::from(g)).sum();
        assert!(
            sum_go.abs() < 1e-3,
            "Σ w·g should vanish at f0+offset, got {sum_go}"
        );
    }

    #[test]
    fn deviance_is_zero_at_perfect_fit_and_nonneg() {
        let sqe = SquaredError;
        let y = [1.0_f32, 2.0, 3.0];
        assert_eq!(sqe.deviance(&y, &y, &[1.0, 1.0, 1.0]).unwrap(), 0.0);
        let d = sqe
            .deviance(&y, &[1.5, 2.0, 2.0], &[1.0, 1.0, 1.0])
            .unwrap();
        assert!(d > 0.0);
        // ½·(0.25 + 0 + 1.0) = 0.625
        assert!((d - 0.625).abs() < 1e-6);
    }

    #[test]
    fn all_zero_weights_error_on_init_and_deviance() {
        let sqe = SquaredError;
        assert!(matches!(
            sqe.init_score(&[1.0, 2.0], &[0.0, 0.0], None),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            sqe.deviance(&[1.0, 2.0], &[1.0, 1.0], &[0.0, 0.0]),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn unequal_lengths_are_shape_mismatch_not_panic() {
        let sqe = SquaredError;
        let mut gh = GradHess::default();
        assert!(matches!(
            sqe.grad_hess(&[1.0, 2.0], &[1.0], &[1.0, 1.0], &mut gh),
            Err(PbError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            sqe.init_score(&[1.0, 2.0], &[1.0], None),
            Err(PbError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            sqe.deviance(&[1.0], &[1.0], &[1.0, 1.0]),
            Err(PbError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn empty_and_single_row_do_not_panic() {
        let sqe = SquaredError;
        let mut gh = GradHess::default();
        // n = 0: grad_hess Ok with empty buffers; init_score/deviance error (Σw = 0).
        sqe.grad_hess(&[], &[], &[], &mut gh).unwrap();
        assert!(gh.g.is_empty() && gh.h.is_empty());
        assert!(sqe.init_score(&[], &[], None).is_err());
        // n = 1.
        sqe.grad_hess(&[2.0], &[5.0], &[1.0], &mut gh).unwrap();
        assert_eq!(gh.g, vec![3.0]);
        assert_eq!(gh.h, vec![1.0]);
    }

    #[test]
    fn init_score_and_deviance_are_thread_count_independent() {
        // The folds are sequential index-order f64 reductions, so thread-count cannot
        // affect the result by construction; this asserts the §05.9#7 gate holds and
        // guards against a future parallelization regressing it.
        let sqe = SquaredError;
        let y: Vec<f32> = (0..5000).map(|i| (i % 17) as f32 * 0.5).collect();
        let raw: Vec<f32> = (0..5000).map(|i| (i % 13) as f32).collect();
        let w: Vec<f32> = (0..5000).map(|i| 1.0 + (i % 3) as f32).collect();
        let run = |n: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap();
            pool.install(|| {
                (
                    sqe.init_score(&y, &w, None).unwrap().to_bits(),
                    sqe.deviance(&y, &raw, &w).unwrap().to_bits(),
                )
            })
        };
        let a = run(1);
        assert_eq!(a, run(2));
        assert_eq!(a, run(8));
    }

    proptest! {
        // Finite-difference oracle (the primary §05.9 check). For the per-row weighted
        // half-loss L(F) = ½ w (F − y)², the central difference equals g exactly and
        // the second difference equals h exactly (it's quadratic), up to f32 rounding.
        #[test]
        fn grad_hess_matches_finite_difference(
            y in -100.0f32..100.0,
            f in -100.0f32..100.0,
            w in 0.0f32..10.0,
        ) {
            let sqe = SquaredError;
            let mut gh = GradHess::default();
            sqe.grad_hess(&[y], &[f], &[w], &mut gh).unwrap();
            let l = |ff: f64| 0.5 * f64::from(w) * (ff - f64::from(y)).powi(2);
            let delta = 0.5_f64;
            let fd_g = (l(f64::from(f) + delta) - l(f64::from(f) - delta)) / (2.0 * delta);
            let fd_h = (l(f64::from(f) + delta) - 2.0 * l(f64::from(f)) + l(f64::from(f) - delta))
                / (delta * delta);
            prop_assert!((f64::from(gh.g[0]) - fd_g).abs() <= 1e-2 + 1e-3 * fd_g.abs());
            prop_assert!((f64::from(gh.h[0]) - fd_h).abs() <= 1e-3 + 1e-3 * fd_h.abs());
        }

        // The DoD-named oracle: finite-difference the SHIPPED `deviance` w.r.t. `raw`
        // and assert it matches `grad_hess`'s g (and the 2nd difference matches h).
        // This is what ties the gradient the engine boosts on to the metric that
        // actually ships — a sign/scale disagreement between them fails here. Values
        // are kept in [-10, 10] so the f32 deviance return doesn't lose the small
        // second-difference to catastrophic cancellation.
        #[test]
        fn shipped_deviance_gradient_matches_grad_hess(
            y in -10.0f32..10.0,
            f in -10.0f32..10.0,
            w in 0.01f32..10.0,
        ) {
            let sqe = SquaredError;
            let mut gh = GradHess::default();
            sqe.grad_hess(&[y], &[f], &[w], &mut gh).unwrap();
            let dev = |ff: f32| -> f64 { f64::from(sqe.deviance(&[y], &[ff], &[w]).unwrap()) };
            let d = 1.0_f32;
            let fd_g = (dev(f + d) - dev(f - d)) / (2.0 * f64::from(d));
            let fd_h = (dev(f + d) - 2.0 * dev(f) + dev(f - d)) / f64::from(d * d);
            prop_assert!(
                (f64::from(gh.g[0]) - fd_g).abs() <= 1e-2 + 1e-3 * fd_g.abs(),
                "deviance gradient {fd_g} != grad_hess g {}", gh.g[0]
            );
            prop_assert!(
                (f64::from(gh.h[0]) - fd_h).abs() <= 1e-2 + 1e-2 * fd_h.abs(),
                "deviance curvature {fd_h} != grad_hess h {}", gh.h[0]
            );
        }
    }
}
