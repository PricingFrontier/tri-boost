//! Objectives & the `Loss` trait (spec §2.4 / §05). The trait + companion types are
//! frozen here; the full v1 objective set ships: `SquaredError` (Identity),
//! `Logistic` (Logit), and the log-link frequency/severity objectives `Poisson`,
//! `Gamma`, `Tweedie { rho }`.
//!
//! The trait is fully orthogonal to tree shape, so it cannot touch I1/I2 — swapping
//! objectives never creates a >3-feature coupling or a non-constant leaf (§05.8). Every
//! method that can fail returns `Result<_, PbError>` (R-LOSSFALLIBLE): a fallible
//! objective maps its failure onto `PbError` rather than panicking.
//!
//! **Numerics (§05.4).** Log-link rows use `mu = exp(F)`; powers are emitted as
//! `exp(k·F)` (never `powf`), with the **exponent clamped to `[-30, 30]`**
//! ([`clamp_exp`]) so a runaway score saturates to a finite `mu` rather than `inf`/`NaN`.
//! The per-row hessian is floored at [`Loss::hessian_floor`] (`1e-16`, a NaN-guard).
//! `init_score`/`deviance` fold per-row `f32` terms into an `f64` accumulator in fixed
//! index order, so the scalar reductions are thread-count-independent (§05.9 #7).
//!
//! **Hot-loop / fail-fast (FLAG, spec §05.2 reconciliation).** §05.2 specifies a `grad_hess`
//! whose only failure is `ShapeMismatch` (saturation + the floor make the kernel total).
//! This crate makes `grad_hess` **stricter** — every loss rejects non-finite `y`/`raw`
//! and negative/non-finite `weight` per row, and rejects any non-finite computed `g/h`,
//! with a typed `InvalidInput` (the fail-fast house style established by the Phase-2
//! `SquaredError` hardening). Domain (sign) checks still live in `init_score`/`deviance`
//! per §05.3a; this adds a finiteness guard so a bug upstream surfaces as a typed error
//! rather than `NaN`/`inf` in the histogram.

use crate::error::PbError;
use itertools::izip;
use serde::{Deserialize, Serialize};

/// Link-argument floor for `init_score` (§05.3): a valid-but-extreme weighted mean is
/// clamped here so an all-zero Poisson target yields a finite very-negative `f0`, not
/// `-inf`. A genuinely out-of-domain input is a typed `Err`, not a clamp (§05.3a).
const EPS_INIT: f64 = 1e-12;

/// The §05.4 exponent clamp range: `exp(k·F)` is evaluated as `exp(clamp(k·F, ±30))`,
/// well inside `f32` `exp` range, so the result is always finite.
const EXP_CLAMP: f32 = 30.0;

fn invalid_input(what: String) -> PbError {
    PbError::InvalidInput { what }
}

/// `exp(x)` with the exponent clamped to `[-30, 30]` (§05.4): degrades a runaway score
/// to a finite saturated value rather than `inf`/`NaN`. The single power primitive for
/// every log-link loss — there is no `powf` on any objective path.
fn clamp_exp(x: f32) -> f32 {
    x.clamp(-EXP_CLAMP, EXP_CLAMP).exp()
}

/// Branch-stable logistic sigmoid `σ(F)` (§05.3): `F ≥ 0 → 1/(1+e^{−F})`, else
/// `e^F/(1+e^F)`. Never overflows; saturates to a finite `0`/`1` at large `|F|`.
fn stable_sigmoid(f: f32) -> f32 {
    if f >= 0.0 {
        let z = clamp_exp(-f);
        1.0 / (1.0 + z)
    } else {
        let z = clamp_exp(f);
        z / (1.0 + z)
    }
}

fn require_finite(obj: &str, label: &str, i: usize, v: f32) -> Result<(), PbError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(format!(
            "{obj} {label}[{i}] must be finite, got {v}"
        )))
    }
}

fn require_weight(obj: &str, i: usize, w: f32) -> Result<(), PbError> {
    if w.is_finite() && w >= 0.0 {
        Ok(())
    } else {
        Err(invalid_input(format!(
            "{obj} weight[{i}] must be finite and >= 0, got {w}"
        )))
    }
}

fn ensure_finite_grad_hess(obj: &str, i: usize, g: f32, h: f32) -> Result<(f32, f32), PbError> {
    if g.is_finite() && h.is_finite() {
        Ok((g, h))
    } else {
        Err(invalid_input(format!(
            "{obj} grad_hess row {i} produced non-finite g/h: g={g}, h={h}"
        )))
    }
}

fn finish_deviance(obj: &str, acc: f64) -> Result<f32, PbError> {
    let out = acc as f32;
    if acc.is_finite() && out.is_finite() {
        Ok(out)
    } else {
        Err(invalid_input(format!(
            "{obj} deviance is not finite/representable as f32: {acc}"
        )))
    }
}

/// Shared entry length-guard for the three slice methods (§05.2): the one in-method
/// failure for the closed-form losses, checked once before any hot loop.
fn require_equal_len(
    obj: &str,
    method: &str,
    n: usize,
    lens: &[(&str, usize)],
) -> Result<(), PbError> {
    for &(name, len) in lens {
        if len != n {
            return Err(PbError::ShapeMismatch {
                what: format!("{obj} {method}: y={n}, {name}={len}"),
            });
        }
    }
    Ok(())
}

/// The exposure-weighted log-link intercept (§05.5): `f0 = log(Σ w y / Σ w e)` with
/// `e = exp(offset)` (or `e = 1` when there is no offset), the ratio floored to
/// [`EPS_INIT`]. `validate_y` enforces the per-row domain (Poisson/Tweedie `y ≥ 0`,
/// Gamma `y > 0`). Rejects all-zero weights and non-positive exposure (`Σ w e ≤ 0`).
fn log_link_init(
    obj: &str,
    y: &[f32],
    weight: &[f32],
    offset: Option<&[f32]>,
    validate_y: impl Fn(usize, f32) -> Result<(), PbError>,
) -> Result<f64, PbError> {
    let n = y.len();
    require_equal_len(obj, "init_score", n, &[("weight", weight.len())])?;
    let (mut sum_wy, mut sum_we) = (0.0_f64, 0.0_f64);
    match offset {
        Some(off) => {
            require_equal_len(obj, "init_score", n, &[("offset", off.len())])?;
            for (i, ((&yi, &wi), &oi)) in y.iter().zip(weight).zip(off).enumerate() {
                require_finite(obj, "y", i, yi)?;
                require_weight(obj, i, wi)?;
                require_finite(obj, "offset", i, oi)?;
                validate_y(i, yi)?;
                let w = f64::from(wi);
                sum_wy += w * f64::from(yi);
                sum_we += w * f64::from(clamp_exp(oi)); // e_i = exp(offset_i) > 0
            }
        }
        None => {
            for (i, (&yi, &wi)) in y.iter().zip(weight).enumerate() {
                require_finite(obj, "y", i, yi)?;
                require_weight(obj, i, wi)?;
                validate_y(i, yi)?;
                let w = f64::from(wi);
                sum_wy += w * f64::from(yi);
                sum_we += w; // e_i = 1
            }
        }
    }
    if sum_we <= 0.0 {
        return Err(invalid_input(format!(
            "{obj} init_score: non-positive Σ w·e (all-zero weights or non-positive exposure)"
        )));
    }
    let ratio = (sum_wy / sum_we).max(EPS_INIT);
    Ok(ratio.ln())
}

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

    /// The trained-objective tag (link + loss id + Tweedie power) recorded in
    /// `ModelSchema.objective` so a loaded model can reproduce link + loss for
    /// export / `predict_proba` (R-SCHEMA). FLAG (spec §2.4/§05 trait addition): the
    /// canonical trait did not list this; the engine needs it to populate the schema.
    fn objective_tag(&self) -> ObjectiveTag;

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
        for (i, (gi, hi, &yi, &fi, &wi)) in
            izip!(&mut out.g, &mut out.h, y, raw, weight).enumerate()
        {
            require_finite("squared-error", "y", i, yi)?;
            require_finite("squared-error", "raw", i, fi)?;
            require_weight("squared-error", i, wi)?;
            let (g, h) =
                ensure_finite_grad_hess("squared-error", i, wi * (fi - yi), wi.max(floor))?;
            *gi = g;
            *hi = h;
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
                for (i, ((&yi, &wi), &oi)) in y.iter().zip(weight).zip(off).enumerate() {
                    require_finite("squared-error", "y", i, yi)?;
                    require_weight("squared-error", i, wi)?;
                    require_finite("squared-error", "offset", i, oi)?;
                    sum_w += f64::from(wi);
                    sum_wy += f64::from(wi) * (f64::from(yi) - f64::from(oi));
                }
            }
            None => {
                for (i, (&yi, &wi)) in y.iter().zip(weight).enumerate() {
                    require_finite("squared-error", "y", i, yi)?;
                    require_weight("squared-error", i, wi)?;
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
        for (i, ((&yi, &fi), &wi)) in y.iter().zip(raw).zip(weight).enumerate() {
            require_finite("squared-error", "y", i, yi)?;
            require_finite("squared-error", "raw", i, fi)?;
            require_weight("squared-error", i, wi)?;
            sum_w += f64::from(wi);
            let r = f64::from(fi) - f64::from(yi);
            acc += f64::from(wi) * r * r;
        }
        if sum_w <= 0.0 {
            return Err(PbError::InvalidInput {
                what: "squared-error deviance: all-zero (or non-positive) weights".into(),
            });
        }
        finish_deviance("squared-error", 0.5 * acc)
    }

    fn default_metric(&self) -> Metric {
        Metric::Rmse
    }

    fn objective_tag(&self) -> ObjectiveTag {
        ObjectiveTag {
            link: Link::Identity,
            loss: LossId::SquaredError,
            tweedie_rho: None,
        }
    }
}

/// Binary logistic regression (spec §05.3): Logit link, `g = w·(σ(F) − y)`,
/// `h = w·σ(F)(1−σ(F))` (floored — the corner where `σ` saturates), `init_score =
/// log(p̄/(1−p̄))`, log-loss deviance. `σ` is the branch-stable sigmoid ([`stable_sigmoid`]).
/// Accepts soft labels `y ∈ [0, 1]`, not only `{0, 1}`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Logistic;

impl Loss for Logistic {
    fn grad_hess(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        let n = y.len();
        require_equal_len(
            "logistic",
            "grad_hess",
            n,
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        out.g.clear();
        out.g.resize(n, 0.0);
        out.h.clear();
        out.h.resize(n, 0.0);
        let floor = self.hessian_floor();
        for (i, (gi, hi, &yi, &fi, &wi)) in
            izip!(&mut out.g, &mut out.h, y, raw, weight).enumerate()
        {
            require_finite("logistic", "y", i, yi)?;
            require_finite("logistic", "raw", i, fi)?;
            require_weight("logistic", i, wi)?;
            let s = stable_sigmoid(fi);
            let (g, h) = ensure_finite_grad_hess(
                "logistic",
                i,
                wi * (s - yi),
                (wi * s * (1.0 - s)).max(floor),
            )?;
            *gi = g;
            *hi = h;
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
        require_equal_len("logistic", "init_score", n, &[("weight", weight.len())])?;
        // The offset (if any) is a logit-space base margin the engine folds into `raw`;
        // it is validated for well-formedness but does NOT enter f0 (exposure is a
        // log-link concept, §05.5). f0 is the boost-from-average seed log(p̄/(1−p̄)).
        if let Some(off) = offset {
            require_equal_len("logistic", "init_score", n, &[("offset", off.len())])?;
            for (i, &oi) in off.iter().enumerate() {
                require_finite("logistic", "offset", i, oi)?;
            }
        }
        let (mut sum_w, mut sum_wy) = (0.0_f64, 0.0_f64);
        for (i, (&yi, &wi)) in y.iter().zip(weight).enumerate() {
            require_finite("logistic", "y", i, yi)?;
            require_weight("logistic", i, wi)?;
            if !(0.0..=1.0).contains(&yi) {
                return Err(invalid_input(format!(
                    "logistic: y[{i}] must be in [0, 1], got {yi}"
                )));
            }
            sum_w += f64::from(wi);
            sum_wy += f64::from(wi) * f64::from(yi);
        }
        if sum_w <= 0.0 {
            return Err(invalid_input(
                "logistic init_score: all-zero weights".into(),
            ));
        }
        if sum_wy <= 0.0 || sum_wy >= sum_w {
            return Err(invalid_input(
                "logistic: no positive (or no negative) class under w".into(),
            ));
        }
        let p = (sum_wy / sum_w).clamp(EPS_INIT, 1.0 - EPS_INIT);
        Ok((p / (1.0 - p)).ln())
    }

    fn link(&self) -> Link {
        Link::Logit
    }

    fn pred_from_raw(&self, raw: f32) -> f32 {
        stable_sigmoid(raw)
    }

    fn deviance(&self, y: &[f32], raw: &[f32], weight: &[f32]) -> Result<f32, PbError> {
        let n = y.len();
        require_equal_len(
            "logistic",
            "deviance",
            n,
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        // Binomial unit deviance: 2 Σ w [ y ln(y/p) + (1−y) ln((1−y)/(1−p)) ].
        let (mut sum_w, mut acc) = (0.0_f64, 0.0_f64);
        for (i, ((&yi, &fi), &wi)) in y.iter().zip(raw).zip(weight).enumerate() {
            require_finite("logistic", "y", i, yi)?;
            require_finite("logistic", "raw", i, fi)?;
            require_weight("logistic", i, wi)?;
            if !(0.0..=1.0).contains(&yi) {
                return Err(invalid_input(format!(
                    "logistic: y[{i}] must be in [0, 1], got {yi}"
                )));
            }
            let p = f64::from(stable_sigmoid(fi)).clamp(EPS_INIT, 1.0 - EPS_INIT);
            let yy = f64::from(yi);
            let t1 = if yy > 0.0 { yy * (yy / p).ln() } else { 0.0 };
            let omy = 1.0 - yy;
            let t2 = if omy > 0.0 {
                omy * (omy / (1.0 - p)).ln()
            } else {
                0.0
            };
            sum_w += f64::from(wi);
            acc += f64::from(wi) * 2.0 * (t1 + t2);
        }
        if sum_w <= 0.0 {
            return Err(invalid_input("logistic deviance: all-zero weights".into()));
        }
        finish_deviance("logistic", acc)
    }

    fn default_metric(&self) -> Metric {
        Metric::LogLoss
    }

    fn objective_tag(&self) -> ObjectiveTag {
        ObjectiveTag {
            link: Link::Logit,
            loss: LossId::Logistic,
            tweedie_rho: None,
        }
    }
}

/// Poisson regression for counts/frequencies (spec §05.3): Log link, `μ = exp(F)`,
/// `g = w·(μ − y)`, `h = w·μ`, `init_score = log(p̄)` (exposure-weighted form §05.5),
/// Poisson deviance. `max_delta_step = Some(0.7)` — the leaf-step cap that keeps the
/// explosive `h = exp(F)` Newton step stable (§05.6).
#[derive(Debug, Clone, Copy, Default)]
pub struct Poisson;

impl Loss for Poisson {
    fn grad_hess(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        let n = y.len();
        require_equal_len(
            "poisson",
            "grad_hess",
            n,
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        out.g.clear();
        out.g.resize(n, 0.0);
        out.h.clear();
        out.h.resize(n, 0.0);
        let floor = self.hessian_floor();
        for (i, (gi, hi, &yi, &fi, &wi)) in
            izip!(&mut out.g, &mut out.h, y, raw, weight).enumerate()
        {
            require_finite("poisson", "y", i, yi)?;
            require_finite("poisson", "raw", i, fi)?;
            require_weight("poisson", i, wi)?;
            let mu = clamp_exp(fi);
            let (g, h) =
                ensure_finite_grad_hess("poisson", i, wi * (mu - yi), (wi * mu).max(floor))?;
            *gi = g;
            *hi = h;
        }
        Ok(())
    }

    fn init_score(
        &self,
        y: &[f32],
        weight: &[f32],
        offset: Option<&[f32]>,
    ) -> Result<f64, PbError> {
        log_link_init("poisson", y, weight, offset, |i, yi| {
            if yi < 0.0 {
                Err(invalid_input(format!(
                    "poisson: y[{i}] must be >= 0, got {yi}"
                )))
            } else {
                Ok(())
            }
        })
    }

    fn link(&self) -> Link {
        Link::Log
    }

    fn pred_from_raw(&self, raw: f32) -> f32 {
        clamp_exp(raw)
    }

    fn deviance(&self, y: &[f32], raw: &[f32], weight: &[f32]) -> Result<f32, PbError> {
        let n = y.len();
        require_equal_len(
            "poisson",
            "deviance",
            n,
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        // Poisson unit deviance: 2 Σ w [ y ln(y/μ) − (y − μ) ]  (y ln(y/μ) → 0 at y=0).
        let (mut sum_w, mut acc) = (0.0_f64, 0.0_f64);
        for (i, ((&yi, &fi), &wi)) in y.iter().zip(raw).zip(weight).enumerate() {
            require_finite("poisson", "y", i, yi)?;
            require_finite("poisson", "raw", i, fi)?;
            require_weight("poisson", i, wi)?;
            if yi < 0.0 {
                return Err(invalid_input(format!(
                    "poisson: y[{i}] must be >= 0, got {yi}"
                )));
            }
            let mu = f64::from(clamp_exp(fi));
            let yy = f64::from(yi);
            let term = if yy > 0.0 { yy * (yy / mu).ln() } else { 0.0 };
            sum_w += f64::from(wi);
            acc += f64::from(wi) * 2.0 * (term - (yy - mu));
        }
        if sum_w <= 0.0 {
            return Err(invalid_input("poisson deviance: all-zero weights".into()));
        }
        finish_deviance("poisson", acc)
    }

    fn default_metric(&self) -> Metric {
        Metric::PoissonDeviance
    }

    fn objective_tag(&self) -> ObjectiveTag {
        ObjectiveTag {
            link: Link::Log,
            loss: LossId::Poisson,
            tweedie_rho: None,
        }
    }

    fn max_delta_step(&self) -> Option<f32> {
        Some(0.7)
    }
}

/// Gamma regression for positive severities (spec §05.3): Log link, `μ = exp(F)`,
/// `g = w·(1 − y·e^{−F})`, `h = w·y·e^{−F}` (floored at the `y → 0` corner),
/// `init_score = log(p̄)` (exposure form §05.5), Gamma deviance. Strict domain `y > 0`
/// (the Gamma support excludes `0`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Gamma;

impl Loss for Gamma {
    fn grad_hess(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        let n = y.len();
        require_equal_len(
            "gamma",
            "grad_hess",
            n,
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        out.g.clear();
        out.g.resize(n, 0.0);
        out.h.clear();
        out.h.resize(n, 0.0);
        let floor = self.hessian_floor();
        for (i, (gi, hi, &yi, &fi, &wi)) in
            izip!(&mut out.g, &mut out.h, y, raw, weight).enumerate()
        {
            require_finite("gamma", "y", i, yi)?;
            require_finite("gamma", "raw", i, fi)?;
            require_weight("gamma", i, wi)?;
            let em = clamp_exp(-fi); // e^{−F} = y/μ factor base
            let t = yi * em; // y·e^{−F}
            let (g, h) = ensure_finite_grad_hess("gamma", i, wi * (1.0 - t), (wi * t).max(floor))?;
            *gi = g;
            *hi = h;
        }
        Ok(())
    }

    fn init_score(
        &self,
        y: &[f32],
        weight: &[f32],
        offset: Option<&[f32]>,
    ) -> Result<f64, PbError> {
        log_link_init("gamma", y, weight, offset, |i, yi| {
            if yi <= 0.0 {
                Err(invalid_input(format!(
                    "gamma: y[{i}] must be > 0, got {yi}"
                )))
            } else {
                Ok(())
            }
        })
    }

    fn link(&self) -> Link {
        Link::Log
    }

    fn pred_from_raw(&self, raw: f32) -> f32 {
        clamp_exp(raw)
    }

    fn deviance(&self, y: &[f32], raw: &[f32], weight: &[f32]) -> Result<f32, PbError> {
        let n = y.len();
        require_equal_len(
            "gamma",
            "deviance",
            n,
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        // Gamma unit deviance: 2 Σ w [ (y−μ)/μ − ln(y/μ) ] = 2 Σ w [ r − 1 − ln r ], r=y/μ.
        let (mut sum_w, mut acc) = (0.0_f64, 0.0_f64);
        for (i, ((&yi, &fi), &wi)) in y.iter().zip(raw).zip(weight).enumerate() {
            require_finite("gamma", "y", i, yi)?;
            require_finite("gamma", "raw", i, fi)?;
            require_weight("gamma", i, wi)?;
            if yi <= 0.0 {
                return Err(invalid_input(format!(
                    "gamma: y[{i}] must be > 0, got {yi}"
                )));
            }
            let mu = f64::from(clamp_exp(fi));
            let r = f64::from(yi) / mu;
            sum_w += f64::from(wi);
            acc += f64::from(wi) * 2.0 * (r - 1.0 - r.ln());
        }
        if sum_w <= 0.0 {
            return Err(invalid_input("gamma deviance: all-zero weights".into()));
        }
        finish_deviance("gamma", acc)
    }

    fn default_metric(&self) -> Metric {
        Metric::GammaDeviance
    }

    fn objective_tag(&self) -> ObjectiveTag {
        ObjectiveTag {
            link: Link::Log,
            loss: LossId::Gamma,
            tweedie_rho: None,
        }
    }
}

/// Tweedie compound-Poisson–Gamma regression (spec §05.3): Log link, power `ρ ∈ (1, 2)`.
/// `g = w·(−y·e^{(1−ρ)F} + e^{(2−ρ)F})`, `h = w·(−y(1−ρ)e^{(1−ρ)F} + (2−ρ)e^{(2−ρ)F})`
/// (floored), `init_score = log(p̄)` (exposure form §05.5), Tweedie deviance. `ρ` is
/// validated `∈ (1, 2)` exclusive at construction ([`Tweedie::new`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tweedie {
    rho: f32,
}

impl Default for Tweedie {
    fn default() -> Self {
        Tweedie { rho: 1.5 }
    }
}

impl Tweedie {
    /// Construct a Tweedie objective with variance power `rho`.
    ///
    /// # Errors
    /// [`PbError::InvalidConfig`] if `rho` is not finite or not in `(1, 2)` exclusive
    /// (distinct from the data-domain [`PbError::InvalidInput`] of §05.3a).
    pub fn new(rho: f32) -> Result<Self, PbError> {
        if !rho.is_finite() || rho <= 1.0 || rho >= 2.0 {
            return Err(PbError::InvalidConfig {
                what: format!("Tweedie rho must be in (1, 2) exclusive, got {rho}"),
            });
        }
        Ok(Tweedie { rho })
    }

    /// The variance power `ρ`.
    #[must_use]
    pub fn rho(&self) -> f32 {
        self.rho
    }
}

impl Loss for Tweedie {
    fn grad_hess(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        let n = y.len();
        require_equal_len(
            "tweedie",
            "grad_hess",
            n,
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        out.g.clear();
        out.g.resize(n, 0.0);
        out.h.clear();
        out.h.resize(n, 0.0);
        let floor = self.hessian_floor();
        let p1 = 1.0 - self.rho; // (1−ρ) < 0
        let p2 = 2.0 - self.rho; // (2−ρ) > 0
        for (i, (gi, hi, &yi, &fi, &wi)) in
            izip!(&mut out.g, &mut out.h, y, raw, weight).enumerate()
        {
            require_finite("tweedie", "y", i, yi)?;
            require_finite("tweedie", "raw", i, fi)?;
            require_weight("tweedie", i, wi)?;
            let a = clamp_exp(p1 * fi); // e^{(1−ρ)F} = μ^{1−ρ}
            let b = clamp_exp(p2 * fi); // e^{(2−ρ)F} = μ^{2−ρ}
            let g = wi * (-yi * a + b);
            // h = w[ −y(1−ρ)e^{(1−ρ)F} + (2−ρ)e^{(2−ρ)F} ] (the §05.3 form; here
            // `-yi * p1` with p1=(1−ρ)<0 is the equivalent y(ρ−1)·a ≥ 0), ≥ 0 for y ≥ 0.
            let h = (wi * (-yi * p1 * a + p2 * b)).max(floor);
            let (g, h) = ensure_finite_grad_hess("tweedie", i, g, h)?;
            *gi = g;
            *hi = h;
        }
        Ok(())
    }

    fn init_score(
        &self,
        y: &[f32],
        weight: &[f32],
        offset: Option<&[f32]>,
    ) -> Result<f64, PbError> {
        log_link_init("tweedie", y, weight, offset, |i, yi| {
            if yi < 0.0 {
                Err(invalid_input(format!(
                    "tweedie: y[{i}] must be >= 0, got {yi}"
                )))
            } else {
                Ok(())
            }
        })
    }

    fn link(&self) -> Link {
        Link::Log
    }

    fn pred_from_raw(&self, raw: f32) -> f32 {
        clamp_exp(raw)
    }

    fn deviance(&self, y: &[f32], raw: &[f32], weight: &[f32]) -> Result<f32, PbError> {
        let n = y.len();
        require_equal_len(
            "tweedie",
            "deviance",
            n,
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        // Tweedie unit deviance:
        //   2 Σ w [ y^{2−ρ}/((1−ρ)(2−ρ)) − y·μ^{1−ρ}/(1−ρ) + μ^{2−ρ}/(2−ρ) ].
        let p1 = 1.0 - f64::from(self.rho);
        let p2 = 2.0 - f64::from(self.rho);
        let p1f = 1.0 - self.rho;
        let p2f = 2.0 - self.rho;
        let (mut sum_w, mut acc) = (0.0_f64, 0.0_f64);
        for (i, ((&yi, &fi), &wi)) in y.iter().zip(raw).zip(weight).enumerate() {
            require_finite("tweedie", "y", i, yi)?;
            require_finite("tweedie", "raw", i, fi)?;
            require_weight("tweedie", i, wi)?;
            if yi < 0.0 {
                return Err(invalid_input(format!(
                    "tweedie: y[{i}] must be >= 0, got {yi}"
                )));
            }
            let yy = f64::from(yi);
            // y^{2−ρ} via exp((2−ρ)·ln y) (no powf); 0 at y = 0.
            let y_term = if yy > 0.0 { (p2 * yy.ln()).exp() } else { 0.0 };
            let mu_p1 = f64::from(clamp_exp(p1f * fi)); // μ^{1−ρ}
            let mu_p2 = f64::from(clamp_exp(p2f * fi)); // μ^{2−ρ}
            let d = y_term / (p1 * p2) - yy * mu_p1 / p1 + mu_p2 / p2;
            sum_w += f64::from(wi);
            acc += f64::from(wi) * 2.0 * d;
        }
        if sum_w <= 0.0 {
            return Err(invalid_input("tweedie deviance: all-zero weights".into()));
        }
        finish_deviance("tweedie", acc)
    }

    fn default_metric(&self) -> Metric {
        Metric::TweedieDeviance { rho: self.rho }
    }

    fn objective_tag(&self) -> ObjectiveTag {
        ObjectiveTag {
            link: Link::Log,
            loss: LossId::Tweedie,
            tweedie_rho: Some(self.rho),
        }
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
    fn nonfinite_or_negative_inputs_are_invalid_input() {
        let sqe = SquaredError;
        let mut gh = GradHess::default();
        assert!(matches!(
            sqe.grad_hess(&[f32::NAN], &[1.0], &[1.0], &mut gh),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            sqe.grad_hess(&[1.0], &[f32::INFINITY], &[1.0], &mut gh),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            sqe.grad_hess(&[1.0], &[1.0], &[-1.0], &mut gh),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            sqe.init_score(&[1.0], &[1.0], Some(&[f32::NAN])),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            sqe.deviance(&[1.0], &[1.0], &[f32::INFINITY]),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn nonfinite_grad_hess_and_metric_outputs_are_invalid_input() {
        let sqe = SquaredError;
        let mut gh = GradHess::default();
        // Finite inputs can still overflow f32 arithmetic; never let inf enter Hist.
        assert!(matches!(
            sqe.grad_hess(&[-f32::MAX], &[f32::MAX], &[1.0], &mut gh),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            Poisson.grad_hess(&[0.0], &[30.0], &[f32::MAX], &mut gh),
            Err(PbError::InvalidInput { .. })
        ));
        // Deviance is returned as f32; unrepresentable finite f64 reductions are errors.
        assert!(matches!(
            sqe.deviance(&[-f32::MAX], &[f32::MAX], &[f32::MAX]),
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

    // ===================================================================
    // Phase 4 objectives: Logistic / Poisson / Gamma / Tweedie.
    // ===================================================================

    /// Central-difference oracle (§05.9 #2): `grad_hess` g/h equal the 1st/2nd central
    /// differences of the per-row loss `L(F)` (`L` includes the row weight). `F` is kept
    /// in a modest range by callers so `exp(F)` does not blow up the finite difference.
    fn assert_fd(loss: &dyn Loss, y: f32, f: f32, w: f32, l: impl Fn(f64) -> f64) {
        let mut gh = GradHess::default();
        loss.grad_hess(&[y], &[f], &[w], &mut gh).unwrap();
        let (g, h) = (f64::from(gh.g[0]), f64::from(gh.h[0]));
        let d = 1e-3_f64;
        let ff = f64::from(f);
        let fd_g = (l(ff + d) - l(ff - d)) / (2.0 * d);
        let fd_h = (l(ff + d) - 2.0 * l(ff) + l(ff - d)) / (d * d);
        assert!(
            (g - fd_g).abs() <= 1e-3 + 1e-3 * fd_g.abs(),
            "g {g} vs central-diff {fd_g}"
        );
        assert!(
            (h - fd_h).abs() <= 1e-2 + 1e-2 * fd_h.abs(),
            "h {h} vs central-diff {fd_h}"
        );
    }

    #[test]
    fn logistic_poisson_gamma_tweedie_closed_form() {
        let mut gh = GradHess::default();
        // Logistic y=1,F=0: σ=0.5 ⇒ g=−0.5, h=0.25.
        Logistic.grad_hess(&[1.0], &[0.0], &[1.0], &mut gh).unwrap();
        assert!((gh.g[0] + 0.5).abs() < 1e-6 && (gh.h[0] - 0.25).abs() < 1e-6);
        // Poisson y=2,F=0: μ=1 ⇒ g=−1, h=1.
        Poisson.grad_hess(&[2.0], &[0.0], &[1.0], &mut gh).unwrap();
        assert!((gh.g[0] + 1.0).abs() < 1e-6 && (gh.h[0] - 1.0).abs() < 1e-6);
        // Gamma y=2,F=0: e^{−F}=1 ⇒ g=1−2=−1, h=2.
        Gamma.grad_hess(&[2.0], &[0.0], &[1.0], &mut gh).unwrap();
        assert!((gh.g[0] + 1.0).abs() < 1e-6 && (gh.h[0] - 2.0).abs() < 1e-6);
        // Tweedie ρ=1.5, y=2,F=0: a=b=1 ⇒ g=−2+1=−1, h=2·0.5·1+0.5·1=1.5.
        let tw = Tweedie::new(1.5).unwrap();
        tw.grad_hess(&[2.0], &[0.0], &[1.0], &mut gh).unwrap();
        assert!((gh.g[0] + 1.0).abs() < 1e-6 && (gh.h[0] - 1.5).abs() < 1e-6);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(96))]

        #[test]
        fn logistic_grad_hess_matches_finite_difference(
            y in 0.0f32..=1.0, f in -6.0f32..6.0, w in 0.1f32..5.0,
        ) {
            // L = softplus(F) − y·F, weighted.
            let (y64, w64) = (f64::from(y), f64::from(w));
            assert_fd(&Logistic, y, f, w, |ff| w64 * ((1.0 + ff.exp()).ln() - y64 * ff));
        }

        #[test]
        fn poisson_grad_hess_matches_finite_difference(
            y in 0.0f32..50.0, f in -3.0f32..3.0, w in 0.1f32..5.0,
        ) {
            // L = exp(F) − y·F.
            let (y64, w64) = (f64::from(y), f64::from(w));
            assert_fd(&Poisson, y, f, w, |ff| w64 * (ff.exp() - y64 * ff));
        }

        #[test]
        fn gamma_grad_hess_matches_finite_difference(
            y in 0.01f32..50.0, f in -3.0f32..3.0, w in 0.1f32..5.0,
        ) {
            // L = y·exp(−F) + F.
            let (y64, w64) = (f64::from(y), f64::from(w));
            assert_fd(&Gamma, y, f, w, |ff| w64 * (y64 * (-ff).exp() + ff));
        }

        #[test]
        fn tweedie_grad_hess_matches_finite_difference(
            y in 0.0f32..30.0, f in -3.0f32..3.0, w in 0.1f32..5.0, rho in 1.05f32..1.95,
        ) {
            // L = −y·exp((1−ρ)F)/(1−ρ) + exp((2−ρ)F)/(2−ρ).
            let tw = Tweedie::new(rho).unwrap();
            let (y64, w64) = (f64::from(y), f64::from(w));
            let (p1, p2) = (1.0 - f64::from(rho), 2.0 - f64::from(rho));
            assert_fd(&tw, y, f, w, |ff| {
                w64 * (-y64 * (p1 * ff).exp() / p1 + (p2 * ff).exp() / p2)
            });
        }
    }

    #[test]
    fn init_score_first_order_condition_per_objective() {
        // Σ w·g(y, f0) ≈ 0 at the boost-from-average intercept, for every objective.
        let w = [1.0_f32, 2.0, 0.5, 1.5, 1.0];
        let cases: Vec<(&dyn Loss, Vec<f32>)> = vec![
            (&Logistic, vec![1.0, 0.0, 1.0, 0.0, 1.0]),
            (&Poisson, vec![0.0, 3.0, 1.0, 5.0, 2.0]),
            (&Gamma, vec![1.0, 3.0, 0.5, 5.0, 2.0]),
        ];
        let tw = Tweedie::new(1.5).unwrap();
        let mut all = cases;
        all.push((&tw, vec![0.0, 3.0, 1.0, 5.0, 2.0]));
        for (loss, y) in all {
            let f0 = loss.init_score(&y, &w, None).unwrap() as f32;
            let raw = vec![f0; y.len()];
            let mut gh = GradHess::default();
            loss.grad_hess(&y, &raw, &w, &mut gh).unwrap();
            let sum_g: f64 = gh.g.iter().map(|&g| f64::from(g)).sum();
            assert!(
                sum_g.abs() < 1e-3,
                "Σ w·g should vanish at f0 for {:?}, got {sum_g}",
                loss.objective_tag().loss
            );
        }
    }

    #[test]
    fn poisson_exposure_init_first_order_and_base_level() {
        // The exposure-weighted Poisson intercept (§05.5): f0 = log(Σwy / Σwe). With a
        // flat exposure e=1 (offset=0) the base level is e⁰ = 1.000, i.e. exp(f0) = p̄.
        let y = [0.0_f32, 2.0, 1.0, 3.0];
        let w = [1.0_f32, 1.0, 1.0, 1.0];
        let off = [0.0_f32; 4]; // e_i = 1
        let f0 = Poisson.init_score(&y, &w, Some(&off)).unwrap();
        let pbar = (0.0 + 2.0 + 1.0 + 3.0) / 4.0_f64;
        assert!(
            (f0.exp() - pbar).abs() < 1e-9,
            "exp(f0) {} != p̄ {pbar}",
            f0.exp()
        );
        // First-order condition with non-flat exposure (Poisson form is exact).
        let e = [1.0_f32, 2.0, 0.5, 1.5];
        let off2: Vec<f32> = e.iter().map(|v| v.ln()).collect();
        let f0b = Poisson.init_score(&y, &w, Some(&off2)).unwrap() as f32;
        let raw: Vec<f32> = off2.iter().map(|&o| f0b + o).collect();
        let mut gh = GradHess::default();
        Poisson.grad_hess(&y, &raw, &w, &mut gh).unwrap();
        let sum_g: f64 = gh.g.iter().map(|&g| f64::from(g)).sum();
        assert!(
            sum_g.abs() < 1e-3,
            "exposure-weighted Σ w·g should vanish, got {sum_g}"
        );
    }

    #[test]
    fn log_link_deviance_is_zero_at_perfect_fit_and_nonneg() {
        // Log-link: μ = y at raw = ln(y) (y>0). Logistic: p = y at raw = logit(y).
        let yp = [1.0_f32, 2.0, 3.0];
        let w = [1.0_f32, 1.0, 1.0];
        let lnln: Vec<f32> = yp.iter().map(|v| v.ln()).collect();
        for loss in [&Poisson as &dyn Loss, &Gamma, &Tweedie::new(1.5).unwrap()] {
            let d0 = loss.deviance(&yp, &lnln, &w).unwrap();
            assert!(d0.abs() < 1e-4, "deviance at μ=y should be ~0, got {d0}");
            let bad: Vec<f32> = yp.iter().map(|v| v.ln() + 0.5).collect();
            assert!(loss.deviance(&yp, &bad, &w).unwrap() > 0.0);
        }
        // Logistic with soft labels: p = y at raw = logit(y).
        let ys = [0.2_f32, 0.5, 0.8];
        let logit: Vec<f32> = ys.iter().map(|&v| (v / (1.0 - v)).ln()).collect();
        let d0 = Logistic.deviance(&ys, &logit, &w).unwrap();
        assert!(
            d0.abs() < 1e-4,
            "logistic deviance at p=y should be ~0, got {d0}"
        );
        assert!(Logistic.deviance(&ys, &[0.0, 0.0, 0.0], &w).unwrap() > 0.0);
    }

    #[test]
    fn domain_errors_per_objective() {
        let w = [1.0_f32, 1.0];
        // Logistic: y outside [0,1]; single-class; all-zero weights.
        assert!(matches!(
            Logistic.init_score(&[0.0, 2.0], &w, None),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            Logistic.init_score(&[0.0, 0.0], &w, None),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            Logistic.init_score(&[1.0, 1.0], &w, None),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            Logistic.init_score(&[0.0, 1.0], &[0.0, 0.0], None),
            Err(PbError::InvalidInput { .. })
        ));
        // Poisson: y<0; all-zero weights. (All-zero y is CLAMPED, not rejected.)
        assert!(matches!(
            Poisson.init_score(&[-1.0, 1.0], &w, None),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(Poisson
            .init_score(&[0.0, 0.0], &w, None)
            .unwrap()
            .is_finite());
        // Gamma: strict y>0 (a single 0 rejects); all-zero weights.
        assert!(matches!(
            Gamma.init_score(&[0.0, 1.0], &w, None),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            Gamma.deviance(&[0.0, 1.0], &[0.0, 0.0], &w),
            Err(PbError::InvalidInput { .. })
        ));
        // Tweedie: y<0; bad rho is InvalidConfig (construction), distinct from data domain.
        let tw = Tweedie::new(1.5).unwrap();
        assert!(matches!(
            tw.init_score(&[-1.0, 1.0], &w, None),
            Err(PbError::InvalidInput { .. })
        ));
        for bad in [1.0f32, 2.0, 0.5, 2.5, f32::NAN] {
            assert!(
                matches!(Tweedie::new(bad), Err(PbError::InvalidConfig { .. })),
                "rho {bad}"
            );
        }
        // Non-positive exposure via a −inf offset is rejected (not −inf into f0).
        assert!(matches!(
            Poisson.init_score(&[1.0, 2.0], &w, Some(&[f32::NEG_INFINITY, 0.0])),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn saturation_and_corners_stay_finite_and_floored() {
        let mut gh = GradHess::default();
        // Logistic F=±40: σ saturates, h floored, all finite.
        Logistic
            .grad_hess(&[1.0, 0.0], &[40.0, -40.0], &[1.0, 1.0], &mut gh)
            .unwrap();
        assert!(
            gh.g.iter().all(|g| g.is_finite())
                && gh
                    .h
                    .iter()
                    .all(|&h| h.is_finite() && h >= Logistic.hessian_floor())
        );
        // Poisson huge F: μ clamped, finite.
        Poisson
            .grad_hess(&[1.0], &[1000.0], &[1.0], &mut gh)
            .unwrap();
        assert!(gh.g[0].is_finite() && gh.h[0].is_finite());
        // Gamma y=0 corner: t=0 ⇒ g=w, h floored.
        Gamma.grad_hess(&[0.0], &[5.0], &[1.0], &mut gh).unwrap();
        assert!(
            (gh.g[0] - 1.0).abs() < 1e-6 && gh.h[0] >= Gamma.hessian_floor() && gh.h[0].is_finite()
        );
        // Tweedie y=0: h = w·(2−ρ)·e^{(2−ρ)F} ≥ 0, finite.
        let tw = Tweedie::new(1.5).unwrap();
        tw.grad_hess(&[0.0], &[3.0], &[1.0], &mut gh).unwrap();
        assert!(gh.g[0].is_finite() && gh.h[0].is_finite() && gh.h[0] >= 0.0);
    }

    #[test]
    fn no_powf_on_the_shipped_loss_path() {
        // §05.4: powers are exp(k·F), never powf/powi on the objective path. Build the
        // needles from pieces so this test's own source does not contain the literals.
        let src = include_str!("loss.rs");
        let shipped = src.split("#[cfg(test)]").next().unwrap_or(src);
        let pf = [".", "pow", "f", "("].concat();
        let pi = [".", "pow", "i", "("].concat();
        assert!(
            !shipped.contains(&pf),
            "shipped loss code must not call powf"
        );
        assert!(
            !shipped.contains(&pi),
            "shipped loss code must not call powi"
        );
    }

    #[test]
    fn log_link_init_and_deviance_are_thread_count_independent() {
        // The f64 folds are sequential index-order, so they are thread-count invariant;
        // this guards a future parallelization from regressing the §05.9 #7 gate.
        let y: Vec<f32> = (0..4000).map(|i| (i % 7) as f32).collect();
        let raw: Vec<f32> = (0..4000).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        let w: Vec<f32> = (0..4000).map(|i| 1.0 + (i % 3) as f32).collect();
        let run = |n: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap();
            pool.install(|| {
                (
                    Poisson.init_score(&y, &w, None).unwrap().to_bits(),
                    Poisson.deviance(&y, &raw, &w).unwrap().to_bits(),
                )
            })
        };
        let a = run(1);
        assert_eq!(a, run(2));
        assert_eq!(a, run(8));
    }
}
