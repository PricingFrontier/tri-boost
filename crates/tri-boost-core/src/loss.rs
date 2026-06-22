//! Objectives & the `Loss` trait (spec §2.4 / §05). Phase-0 stubs: the trait and
//! its companion types are frozen here; the concrete v1 implementors
//! (`SquaredError`, `Logistic`, `Poisson`, `Gamma`, `Tweedie`) land with §05.
//!
//! The trait is fully orthogonal to tree shape, so it cannot touch I1/I2. Every
//! method that can fail returns `Result<_, PbError>` (R-LOSSFALLIBLE): a fallible
//! objective maps its failure onto `PbError` rather than panicking.

use crate::error::PbError;
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
