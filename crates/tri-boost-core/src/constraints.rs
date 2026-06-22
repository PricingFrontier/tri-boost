//! Interaction selection & monotone constraints (spec §2.9 / §07). Phase-0 stubs:
//! the interaction policy and monotone map are frozen here; the `wht8` transform,
//! the online screening-variance accumulator, and the heredity/FAST/Sobol admission
//! funnel land with §07.

use crate::explain::FeatureSet;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A monotonicity direction for one feature (spec §07).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonoSign {
    /// The response must be non-decreasing in this feature.
    Increasing,
    /// The response must be non-increasing in this feature.
    Decreasing,
    /// No monotone constraint.
    None,
}

/// Monotone constraints keyed by feature NAME, never positional (spec §2.9 / §07).
/// A `BTreeMap` for deterministic iteration order (it can be serialized as part of a
/// fit record).
pub type MonotoneMap = BTreeMap<String, MonoSign>;

/// The whole-tree interaction constraint plus the optional feature-group whitelist
/// (spec §2.9 / §07). `groups` (when `Some`) restricts each tree's distinct-raw
/// support to lie within one declared group; `None` = unconstrained.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteractionPolicy {
    /// Maximum interaction order, in `{1, 2, 3}`; default `3`.
    pub max_order: u8,
    /// Allowed co-occurrence groups; `None` = unconstrained.
    pub groups: Option<Vec<FeatureSet>>,
}

impl Default for InteractionPolicy {
    fn default() -> Self {
        Self {
            max_order: 3,
            groups: None,
        }
    }
}
