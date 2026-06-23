//! Interaction selection & monotone constraints (spec §2.9 / §07). This module owns
//! the serialized interaction policy, name-keyed monotone constraints, and the
//! order-3 Walsh-Hadamard primitive used as an independent oracle for tree-local
//! interaction strength. The online screening accumulator and soft heredity/FAST/Sobol
//! admission prior build on these pieces.

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
    /// Soft table-size prior exponent (§07.3/§07.4). A value of `0.0` is exactly
    /// inert; positive values down-rank supports whose projected table cells exceed
    /// [`InteractionPolicy::table_budget_cells`], but never hard-reject them.
    #[serde(default = "default_table_budget_beta")]
    pub table_budget_beta: f32,
    /// Cell budget used by the soft table-size prior. This is an admission score
    /// prior, separate from the hard [`crate::TableBudget`] allocation firewall.
    #[serde(default = "default_table_budget_cells")]
    pub table_budget_cells: u64,
}

impl Default for InteractionPolicy {
    fn default() -> Self {
        Self {
            max_order: 3,
            groups: None,
            table_budget_beta: default_table_budget_beta(),
            table_budget_cells: default_table_budget_cells(),
        }
    }
}

fn default_table_budget_beta() -> f32 {
    0.5
}

fn default_table_budget_cells() -> u64 {
    2_000_000
}

/// Uniform 8-leaf Walsh-Hadamard / Möbius coefficients for one depth-3 oblivious
/// leaf vector (§07.4a).
///
/// Coefficients are indexed by a bitmask over split levels: `0b000` is the constant
/// term, `0b001/010/100` are main effects, `0b011/101/110` are pairs, and `0b111`
/// is the pure triple interaction. The transform is orthonormal up to the standard
/// `1/8` averaging factor, so [`inverse_wht8_uniform`] reconstructs the original leaf
/// vector exactly up to floating-point roundoff.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wht8 {
    /// Coefficients in mask order.
    pub coeffs: [f64; 8],
}

/// Compute uniform Walsh-Hadamard coefficients for an 8-leaf vector.
#[must_use]
pub fn wht8_uniform(leaves: [f64; 8]) -> Wht8 {
    let mut coeffs = [0.0_f64; 8];
    for (mask, slot) in coeffs.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (leaf, value) in leaves.iter().enumerate() {
            acc += sign(mask, leaf) * value;
        }
        *slot = acc / 8.0;
    }
    Wht8 { coeffs }
}

/// Reconstruct an 8-leaf vector from uniform Walsh-Hadamard coefficients.
#[must_use]
pub fn inverse_wht8_uniform(wht: Wht8) -> [f64; 8] {
    let mut leaves = [0.0_f64; 8];
    for (leaf, slot) in leaves.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (mask, coeff) in wht.coeffs.iter().enumerate() {
            acc += sign(mask, leaf) * coeff;
        }
        *slot = acc;
    }
    leaves
}

fn sign(mask: usize, leaf: usize) -> f64 {
    if ((mask & leaf).count_ones() & 1) == 0 {
        1.0
    } else {
        -1.0
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::indexing_slicing, clippy::float_cmp)]

    use super::*;

    #[test]
    fn wht8_round_trips_leaf_values() {
        let leaves = [1.0, -2.0, 3.5, 4.0, -1.0, 0.25, 8.0, -3.0];
        let got = inverse_wht8_uniform(wht8_uniform(leaves));
        for (a, b) in got.iter().zip(leaves) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn wht8_names_constant_main_pair_and_triple_masks() {
        let leaves = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0];
        let coeffs = wht8_uniform(leaves).coeffs;
        assert_eq!(coeffs[0], leaves.iter().sum::<f64>() / 8.0);
        // The single extra bump at leaf 0b111 is visible in the pure triple mask.
        assert!(coeffs[0b111].abs() > 0.0);
        // Main and pair masks are finite and stored in the documented mask positions.
        for coeff in coeffs.iter().take(0b110 + 1).skip(0b001) {
            assert!(coeff.is_finite());
        }
    }
}
