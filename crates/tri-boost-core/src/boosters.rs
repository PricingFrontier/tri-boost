//! Predictiveness boosters (spec §09). Phase-0 stubs: only the `DistillSpec` shape
//! that `FitSpec` threads is registered here; CatBoost teacher-distillation,
//! fully-corrective refit, Nesterov mixing, and bagged ensemble selection — all
//! exactness-preserving — land with §09.

use serde::{Deserialize, Serialize};

/// Which teacher produced the per-row soft targets in a [`DistillSpec`] (spec §09).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TeacherKind {
    /// A CatBoost teacher trained inside the fit (off-by-default).
    CatBoost,
    /// Externally-supplied teacher raw scores.
    External,
}

/// Distillation data + blend for one fit (spec §2.9 / §09, R-DISTILL). Off by
/// default (`FitSpec.distill: Option<DistillSpec>`); per-row data lives here with
/// `weight`/`exposure`, not in any config struct. Exactness-preserving: the model
/// is still exact to its own tables.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistillSpec {
    /// Per-row teacher raw scores (same length as `y`).
    pub teacher_raw: Vec<f32>,
    /// True-label weight in the blended target; default `0.5`, `1.0` = no teacher.
    pub blend: f32,
    /// Which teacher produced `teacher_raw`.
    pub teacher: TeacherKind,
}
