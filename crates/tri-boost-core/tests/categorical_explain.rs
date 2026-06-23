//! Audit-1 regression (CRITICAL): `Model::explain` must work on a model trained with
//! categorical (TS-encoded) features. A categorical axis is a Target-Statistic-encoded
//! ORDINAL `BorderGrid` axis, so the merged-grid decomposition and the five I2 gates
//! apply to it identically — the bank is audited on a `ServeBinnedMatrix` re-encoded
//! through the frozen full-data encoders (R-CATSERVE, §08/§04).
//!
//! Before this test, NO test called `explain()` on a categorical model, and a stale
//! "numeric axes only" guard in `MergedGrids::from_model` (left over from Phase 3, when
//! categoricals did not exist) made the core decomposability guarantee structurally
//! unreachable for any categorical model: such a model is `Exact`, but `explain()` failed
//! with `InvalidConfig` before any of the five checks could run.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use tri_boost_core::{
    assert_exact_decomposition, bin_serve_columns, bin_train_columns, BinConfig, Booster,
    CategoricalColumn, Config, FeatureId, FitSpec, InteractionPolicy, LeakageScheme, MonotoneMap,
    NumericColumn, RefMeasure, ServeCategoricalColumn, Smooth, SquaredError, TsConfig,
    TsEncodingId,
};

#[test]
fn categorical_model_explains_and_passes_all_five_gates() {
    let n = 90usize;
    let numeric: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
    let cats = ["low", "mid", "high"];
    let levels: Vec<String> = (0..n).map(|i| cats[i % 3].to_owned()).collect();
    // A target additive in (numeric, category), with a DOMINANT categorical effect so
    // the categorical axis is reliably split (and thus appears in the decomposition).
    let y: Vec<f32> = (0..n)
        .map(|i| numeric[i] + [0.0_f32, 50.0, 100.0][i % 3])
        .collect();

    let ts = TsConfig {
        leakage: LeakageScheme::KFold { k: 3 },
        smooth: Smooth::Fixed { m: 0.0 },
        min_data_per_group: 0.0,
        ..TsConfig::default()
    };
    let fitted = bin_train_columns(
        &[NumericColumn {
            raw: FeatureId(0),
            values: &numeric,
        }],
        &[CategoricalColumn {
            raw: FeatureId(1),
            id: TsEncodingId(0),
            levels: &levels,
            config: &ts,
        }],
        &y,
        None,
        None,
        &BinConfig::default(),
        7,
    )
    .unwrap();

    let sqe = SquaredError;
    let spec = FitSpec {
        loss: &sqe,
        weight: None,
        exposure: None,
        monotone: MonotoneMap::new(),
        interaction: InteractionPolicy::default(),
        seed: 0,
    };
    let model = Booster::with_config(Config {
        n_trees: 25,
        learning_rate: 0.5,
        lambda: 1.0,
        min_split_gain: 0.0,
        max_delta_step: None,
        sampling: Default::default(),
        hist_precision: Default::default(),
        boosters: Default::default(),
    })
    .fit_train(&fitted.train, &y, &spec, fitted.cat_encoders.clone())
    .unwrap();

    // The categorical axis (axis 1, raw FeatureId(1)) is actually realized in the model.
    assert!(
        model
            .trees
            .iter()
            .any(|(_, t)| t.splits.iter().any(|s| s.axis == 1)),
        "the dominant categorical effect should be split on"
    );

    // R-CATSERVE: build the serve matrix by re-encoding raw labels through the FROZEN
    // full-data encoders, binned against the MODEL's own grids/provenance.
    let serve = bin_serve_columns(
        &[NumericColumn {
            raw: FeatureId(0),
            values: &numeric,
        }],
        &[ServeCategoricalColumn {
            raw: FeatureId(1),
            id: TsEncodingId(0),
            levels: &levels,
        }],
        &model.grids,
        &model.provenance,
        &model.schema.cat_encoders,
    )
    .unwrap();

    // The whole reason the library exists: a categorical model decomposes losslessly.
    for w in [RefMeasure::default(), RefMeasure::Uniform] {
        let bank = model
            .explain(&serve, w.clone())
            .unwrap_or_else(|e| panic!("categorical explain({w:?}) failed: {e:?}"));
        assert_exact_decomposition(&model, &bank, &serve)
            .unwrap_or_else(|e| panic!("categorical I2 gates failed under {w:?}: {e:?}"));
        // The categorical raw feature has a realized table in the bank.
        assert!(
            bank.tables.iter().any(|t| t.u.0.iter().any(|f| f.0 == 1)),
            "the categorical raw feature should have a realized effect table"
        );
    }
}
