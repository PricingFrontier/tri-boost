//! Gate G0 invariant entrypoint (spec §13.1/§13.2, plan F7/F8). Exercises the five
//! I2 checks + the I1 feature budget through the public API over hand-built fixtures
//! — green on the positive fixture, red with the correct `Invariant` on the negative
//! ones. These are the same checks §06/§08 will point at the real `TableBank`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use tri_boost_core::explain::{
    check_reconstruction, fixture_bank, fixture_grid_corners, fixture_model,
    fixture_over_budget_model,
};
use tri_boost_core::{assert_exact_decomposition, check_feature_budget, Invariant, PbError};

#[test]
fn g0_positive_fixture_passes_all_five_checks() {
    let model = fixture_model();
    let bank = fixture_bank();
    let corners = fixture_grid_corners();
    assert_exact_decomposition(&model, &bank, &corners).expect("exact decomposition must hold");
    check_feature_budget(&model).expect("feature budget must hold");
}

#[test]
fn g0_negative_reconstruction_is_caught() {
    let model = fixture_model();
    let mut bank = fixture_bank();
    let corners = fixture_grid_corners();
    // Perturb a single main-effect cell: tables no longer reconstruct the ensemble.
    bank.tables[0].values.add(&[1], 2.0).unwrap();
    match assert_exact_decomposition(&model, &bank, &corners) {
        Err(PbError::InvariantViolated { invariant }) => {
            assert_eq!(invariant, Invariant::Reconstruction);
        }
        other => panic!("expected Reconstruction violation, got {other:?}"),
    }
    // The dedicated check names the same property.
    assert!(matches!(
        check_reconstruction(&model, &bank, &corners),
        Err(PbError::InvariantViolated {
            invariant: Invariant::Reconstruction
        })
    ));
}

#[test]
fn g0_over_budget_model_violates_feature_budget() {
    match check_feature_budget(&fixture_over_budget_model()) {
        Err(PbError::InvariantViolated { invariant }) => {
            assert_eq!(invariant, Invariant::FeatureBudget);
        }
        other => panic!("expected FeatureBudget violation, got {other:?}"),
    }
}
