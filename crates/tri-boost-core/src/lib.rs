//! # tri-boost-core
//!
//! A depth-3 **oblivious** gradient boosting machine that is *exactly* decomposable
//! into ≤3rd-order functional-ANOVA tables — the "rating tables" — without
//! sacrificing speed or accuracy. Every tree is a depth-`1..=3` symmetric tree with
//! one shared `(feature, threshold)` test per level and at most three distinct raw
//! features, so the trained ensemble truncates exactly at the 3rd interaction order.
//!
//! This crate is **pure Rust** (no Python; verified by the `NoPyo3` CI gate) and is
//! built to a high engineering standard enforced *structurally*:
//!
//! * `#![forbid(unsafe_code)]` on the whole core (SIMD arrives via safe wrappers).
//! * The no-panic gate — `unwrap`/`expect`/`panic`/`unreachable`/`indexing_slicing`
//!   are denied — so the single [`PbError`] enum is the only way to surface failure.
//! * `#![deny(missing_docs)]` — every public item is documented.
//! * The five lossless I2 invariants + the I1 feature budget are *real,
//!   build-blocking checks* ([`explain`]), live from the first commit.
//!
//! Module layout is 1:1 with the spec's section-ownership map (§4).
#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod backend;
pub mod boosters;
pub mod cat;
pub mod constraints;
pub mod data;
pub mod engine;
pub mod error;
pub mod explain;
pub mod loss;
pub mod scoring;
pub mod serialize;
pub mod simd;

// --- Canonical re-exports (spec §2 single source of truth). ---------------

pub use error::{Invariant, PbError};

pub use backend::{pb_rng, pb_seed, CpuBackend, Stage};

pub use data::{
    bin, bin_columns, bin_serve_columns, bin_train_columns, build_grid, compute_offset, AxisKind,
    AxisProvenance, BinConfig, BinnedMatrix, BorderFamily, BorderGrid, CategoricalColumn,
    FeatureId, FittedBinnedData, NumericColumn, ServeBinnedMatrix, ServeCategoricalColumn,
    TrainBinnedMatrix,
};

pub use cat::{
    exposure_weighted_base_rate, fit_cat_encoder, shrunken_encoding, CatEncoder, CatEncoderStore,
    CatFitSpec, CatLevel, LeakageScheme, Smooth, TsConfig, TsEncodingId,
};

pub use loss::{
    Gamma, GradHess, Link, Logistic, Loss, LossId, Metric, ObjectiveTag, Poisson, SquaredError,
    Tweedie,
};

pub use engine::{
    Booster, Config, ExactnessMode, FitSpec, GradScale, Hist, HistPrecision, Model, ModelSchema,
    ObliviousTree, QuantGradHess, Sampling, Split,
};

pub use constraints::{
    inverse_wht8_uniform, wht8_uniform, InteractionPolicy, MonoSign, MonotoneMap, Wht8,
};

pub use explain::{
    assert_exact_decomposition, check_feature_budget, AxisId, EffectTable, ExactTol, FeatureSet,
    OverflowPolicy, PurifyMode, RefMeasure, TableBank, TableBudget, Tensor,
};

pub use serialize::{
    decode_doc, decode_doc_json, decode_model, decode_model_json, encode_doc, encode_doc_json,
    encode_model, encode_model_json, AxisExport, ModelDoc, RatingBasis, RatingExport, RatingTable,
    FORMAT_VERSION, SCHEMA_VERSION,
};

pub use scoring::{PackedTree, ScoringBank, TableScoringBank};

pub use simd::{score_tile, CHUNK_ROWS};
