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

impl TsConfig {
    /// Validate categorical encoder configuration.
    ///
    /// # Errors
    /// [`PbError::InvalidConfig`] if a count/smoothing parameter is outside its
    /// finite domain.
    pub fn validate(&self) -> Result<(), PbError> {
        match self.leakage {
            LeakageScheme::Ordered { n_perms } => {
                if n_perms == 0 {
                    return Err(PbError::InvalidConfig {
                        what: "Ordered target statistics require n_perms > 0".into(),
                    });
                }
            }
            LeakageScheme::KFold { k } => {
                if k < 2 {
                    return Err(PbError::InvalidConfig {
                        what: format!("KFold target statistics require k >= 2, got {k}"),
                    });
                }
            }
        }
        if self.target_borders == 0 {
            return Err(PbError::InvalidConfig {
                what: "target_borders must be > 0".into(),
            });
        }
        if !self.min_data_per_group.is_finite() || self.min_data_per_group < 0.0 {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "min_data_per_group must be finite and >= 0, got {}",
                    self.min_data_per_group
                ),
            });
        }
        if let Smooth::Fixed { m } = self.smooth {
            if !m.is_finite() || m < 0.0 {
                return Err(PbError::InvalidConfig {
                    what: format!("Smooth::Fixed m must be finite and >= 0, got {m}"),
                });
            }
        }
        Ok(())
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

/// Exposure-weighted base rate for categorical target statistics (§04.3/§03.7):
/// `p = Σ w_i y_i / Σ w_i e_i`, with `e_i = 1` when `exposure` is absent.
///
/// # Errors
/// [`PbError::ShapeMismatch`] on length mismatch; [`PbError::InvalidInput`] on
/// non-finite labels/weights/exposures, negative weights, non-positive exposure, or
/// a zero denominator.
pub fn exposure_weighted_base_rate(
    y: &[f32],
    weight: &[f32],
    exposure: Option<&[f32]>,
) -> Result<f32, PbError> {
    if weight.len() != y.len() {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "categorical base rate: y={}, weight={}",
                y.len(),
                weight.len()
            ),
        });
    }
    if let Some(e) = exposure {
        if e.len() != y.len() {
            return Err(PbError::ShapeMismatch {
                what: format!("categorical base rate: y={}, exposure={}", y.len(), e.len()),
            });
        }
    }
    let mut sum_wy = 0.0_f64;
    let mut sum_we = 0.0_f64;
    for (i, (&yi, &wi)) in y.iter().zip(weight).enumerate() {
        if !yi.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!("categorical y[{i}] must be finite, got {yi}"),
            });
        }
        if !wi.is_finite() || wi < 0.0 {
            return Err(PbError::InvalidInput {
                what: format!("categorical weight[{i}] must be finite and >= 0, got {wi}"),
            });
        }
        let e = match exposure {
            Some(ex) => {
                let ei = *ex.get(i).ok_or_else(|| PbError::Internal {
                    what: "validated exposure lost a row".into(),
                })?;
                if !ei.is_finite() || ei <= 0.0 {
                    return Err(PbError::InvalidInput {
                        what: format!("categorical exposure[{i}] must be finite and > 0, got {ei}"),
                    });
                }
                f64::from(ei)
            }
            None => 1.0,
        };
        let w = f64::from(wi);
        sum_wy += w * f64::from(yi);
        sum_we += w * e;
    }
    if sum_we <= 0.0 {
        return Err(PbError::InvalidInput {
            what: "categorical base rate denominator Σw·e must be > 0".into(),
        });
    }
    let out = (sum_wy / sum_we) as f32;
    if out.is_finite() {
        Ok(out)
    } else {
        Err(PbError::InvalidInput {
            what: format!(
                "categorical base rate is not representable as f32: {}",
                sum_wy / sum_we
            ),
        })
    }
}

/// Closed-form target-statistic shrinkage (§04.3):
/// `(sum_wy + m·base) / (sum_w + m)`.
///
/// # Errors
/// [`PbError::InvalidInput`] if inputs are non-finite or the denominator is zero;
/// [`PbError::InvalidConfig`] for [`Smooth::Auto`], whose variance estimator belongs
/// to the full encoder-fit path.
pub fn shrunken_encoding(
    sum_wy: f64,
    sum_w: f64,
    base: f32,
    smooth: Smooth,
) -> Result<f32, PbError> {
    if !sum_wy.is_finite() || !sum_w.is_finite() || sum_w < 0.0 || !base.is_finite() {
        return Err(PbError::InvalidInput {
            what: format!(
                "invalid categorical shrinkage inputs: sum_wy={sum_wy}, sum_w={sum_w}, base={base}"
            ),
        });
    }
    let m = match smooth {
        Smooth::Fixed { m } => {
            if !m.is_finite() || m < 0.0 {
                return Err(PbError::InvalidConfig {
                    what: format!("Smooth::Fixed m must be finite and >= 0, got {m}"),
                });
            }
            f64::from(m)
        }
        Smooth::Auto => {
            return Err(PbError::InvalidConfig {
                what: "Smooth::Auto requires the full encoder-fit variance estimator".into(),
            });
        }
    };
    let denom = sum_w + m;
    if denom <= 0.0 {
        return Err(PbError::InvalidInput {
            what: "categorical shrinkage denominator sum_w + m must be > 0".into(),
        });
    }
    let out = ((sum_wy + m * f64::from(base)) / denom) as f32;
    if out.is_finite() {
        Ok(out)
    } else {
        Err(PbError::InvalidInput {
            what: "categorical shrinkage output is not finite".into(),
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

    #[test]
    fn ts_config_validates_fail_closed() {
        assert!(TsConfig::default().validate().is_ok());
        assert!(matches!(
            TsConfig {
                leakage: LeakageScheme::Ordered { n_perms: 0 },
                ..TsConfig::default()
            }
            .validate(),
            Err(PbError::InvalidConfig { .. })
        ));
        assert!(matches!(
            TsConfig {
                leakage: LeakageScheme::KFold { k: 1 },
                ..TsConfig::default()
            }
            .validate(),
            Err(PbError::InvalidConfig { .. })
        ));
        assert!(matches!(
            TsConfig {
                smooth: Smooth::Fixed { m: f32::NAN },
                ..TsConfig::default()
            }
            .validate(),
            Err(PbError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn base_rate_matches_exposure_weighted_closed_form() {
        let y = [2.0_f32, 8.0, 10.0];
        let w = [1.0_f32, 2.0, 1.0];
        let e = [1.0_f32, 2.0, 4.0];
        let got = exposure_weighted_base_rate(&y, &w, Some(&e)).unwrap();
        let want = (1.0 * 2.0 + 2.0 * 8.0 + 1.0 * 10.0) / (1.0 * 1.0 + 2.0 * 2.0 + 1.0 * 4.0);
        assert!((got - want).abs() < 1e-6);
        assert!(matches!(
            exposure_weighted_base_rate(&y, &[0.0, 0.0, 0.0], None),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn shrinkage_matches_closed_form_and_auto_fails_closed() {
        let got = shrunken_encoding(30.0, 3.0, 5.0, Smooth::Fixed { m: 2.0 }).unwrap();
        let want = (30.0 + 2.0 * 5.0) / (3.0 + 2.0);
        assert!((got - want).abs() < 1e-6);
        assert_eq!(
            shrunken_encoding(0.0, 0.0, 7.0, Smooth::Fixed { m: 4.0 }).unwrap(),
            7.0
        );
        assert!(matches!(
            shrunken_encoding(1.0, 1.0, 0.0, Smooth::Auto),
            Err(PbError::InvalidConfig { .. })
        ));
    }
}
