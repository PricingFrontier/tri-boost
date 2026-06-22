//! Categorical handling (spec §04). This module owns the frozen Target-Statistic
//! encoder schema that is serialized with a [`crate::Model`]. The leakage-free fit
//! algorithms land on top of these types; the store and lookup semantics are already
//! deterministic and total.

use crate::data::FeatureId;
use crate::error::PbError;
use serde::{Deserialize, Serialize};

/// Identifier for one categorical Target-Statistic encoding (spec §04). Resolves to
/// a concrete [`CatEncoder`] in the [`CatEncoderStore`].
///
/// Append-only and fixed-width (`u8`): it is serialized inside
/// [`crate::data::AxisKind::CategoricalTS`], so it must never be a platform-width int.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct TsEncodingId(pub u8);

/// Leakage-avoidance scheme used to produce training-time categorical encodings
/// (spec §04.3). Serve-time encodings always use the frozen full-data map stored in
/// [`CatEncoder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeakageScheme {
    /// Ordered target statistics over `n_perms` seeded permutations.
    Ordered {
        /// Number of deterministic permutations.
        n_perms: u32,
    },
    /// K-fold cross-fit target statistics.
    KFold {
        /// Number of folds.
        k: u32,
    },
}

impl Default for LeakageScheme {
    fn default() -> Self {
        Self::Ordered { n_perms: 1 }
    }
}

/// Smoothing rule for target-statistic shrinkage (spec §04.3).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Smooth {
    /// Fixed pseudo-count `m` in `(n_c·mean_c + m·base)/(n_c + m)`.
    Fixed {
        /// Pseudo-count strength.
        m: f32,
    },
    /// Estimate `m` from within/between-level variance.
    Auto,
}

impl Default for Smooth {
    fn default() -> Self {
        Self::Fixed { m: 20.0 }
    }
}

/// Configuration for one target-statistic encoder (spec §04.3/§04.12).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TsConfig {
    /// Leakage-avoidance scheme for training rows.
    pub leakage: LeakageScheme,
    /// Shrinkage rule.
    pub smooth: Smooth,
    /// Number of target borders used by the encoder's target pre-binning.
    pub target_borders: u32,
    /// Low-cardinality one-hot threshold.
    pub one_hot_max_size: u32,
    /// Weighted-count floor below which levels collapse into the rare bucket.
    pub min_data_per_group: f32,
}

impl Default for TsConfig {
    fn default() -> Self {
        Self {
            leakage: LeakageScheme::default(),
            smooth: Smooth::default(),
            target_borders: 16,
            one_hot_max_size: 2,
            min_data_per_group: 10.0,
        }
    }
}

/// One frozen categorical level in serve/export order (spec §04.4/§04.12).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatLevel {
    /// Human-readable level label. The reserved rare bucket is represented as its own
    /// label when rare-level collapse is active.
    pub label: String,
    /// Full-data shrunken target-statistic value used at serve time.
    pub encoding: f32,
    /// Ordinal bin id emitted by the Fisher-sorted categorical axis.
    pub bin: u8,
    /// Effective weighted count behind this level.
    pub weight: f32,
}

/// One frozen, full-data categorical encoder (spec §04). Training-time encodings are
/// leakage-free views; this stored encoder is the serve/export map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatEncoder {
    /// Raw feature this encoder belongs to.
    pub raw: FeatureId,
    /// Encoder id, unique within [`CatEncoderStore`] for a raw feature.
    pub id: TsEncodingId,
    /// Frozen levels in deterministic serve/export order.
    pub levels: Vec<CatLevel>,
    /// Base value for unseen levels at serve time.
    pub base: f32,
    /// Configuration that produced the encoder.
    pub config: TsConfig,
}

/// The frozen `TsEncodingId → CatEncoder` table backing serve/export (spec §2.6 /
/// §04, R-SCHEMA). `explain()` and `TableBank` accumulation re-encode raw
/// categoricals through THESE (never the noisy train-time encoders).
///
/// Backed by a `Vec` (not `HashMap`): serialized state must have deterministic
/// iteration order (the `check-no-hashmap-serialized` gate).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CatEncoderStore {
    encoders: Vec<CatEncoder>,
}

impl CatEncoderStore {
    /// An empty store (no categorical axes). Used by purely-numeric models.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a store from encoders already in deterministic order.
    #[must_use]
    pub fn from_encoders(encoders: Vec<CatEncoder>) -> Self {
        Self { encoders }
    }

    /// The frozen encoders in deterministic order.
    #[must_use]
    pub fn encoders(&self) -> &[CatEncoder] {
        &self.encoders
    }

    /// `true` if no categorical encoders are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.encoders.is_empty()
    }

    /// Number of registered encoders.
    #[must_use]
    pub fn len(&self) -> usize {
        self.encoders.len()
    }

    /// Look up one encoder by `(id, raw)` without panicking.
    ///
    /// # Errors
    /// [`PbError::Internal`] if no matching encoder is present. A model that names a
    /// missing encoder is internally inconsistent, not malformed user data.
    pub fn get(&self, id: TsEncodingId, raw: FeatureId) -> Result<&CatEncoder, PbError> {
        self.encoders
            .iter()
            .find(|enc| enc.id == id && enc.raw == raw)
            .ok_or_else(|| PbError::Internal {
                what: format!("categorical encoder {id:?} for raw feature {raw:?} not found"),
            })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

    use super::*;

    fn encoder(id: u8, raw: u32, label: &str) -> CatEncoder {
        CatEncoder {
            raw: FeatureId(raw),
            id: TsEncodingId(id),
            levels: vec![CatLevel {
                label: label.into(),
                encoding: 1.25,
                bin: 1,
                weight: 10.0,
            }],
            base: 0.5,
            config: TsConfig::default(),
        }
    }

    #[test]
    fn store_lookup_is_total_and_raw_aware() {
        let store = CatEncoderStore::from_encoders(vec![encoder(0, 1, "a"), encoder(0, 2, "b")]);
        assert_eq!(store.len(), 2);
        assert_eq!(
            store.get(TsEncodingId(0), FeatureId(1)).unwrap().levels[0].label,
            "a"
        );
        assert_eq!(
            store.get(TsEncodingId(0), FeatureId(2)).unwrap().levels[0].label,
            "b"
        );
        assert!(matches!(
            store.get(TsEncodingId(1), FeatureId(1)),
            Err(PbError::Internal { .. })
        ));
    }

    #[test]
    fn non_empty_store_round_trips_byte_stably() {
        let store = CatEncoderStore::from_encoders(vec![encoder(0, 1, "a"), encoder(1, 1, "rare")]);
        let cfg = bincode::config::standard();
        let a = bincode::serde::encode_to_vec(&store, cfg).unwrap();
        let b = bincode::serde::encode_to_vec(&store, cfg).unwrap();
        assert_eq!(a, b);
        let (decoded, len): (CatEncoderStore, usize) =
            bincode::serde::decode_from_slice(&a, cfg).unwrap();
        assert_eq!(len, a.len());
        assert_eq!(decoded, store);
    }
}
