//! Categorical handling (spec §04). This module owns the frozen Target-Statistic
//! encoder schema that is serialized with a [`crate::Model`]. The leakage-free fit
//! algorithms land on top of these types; the store and lookup semantics are already
//! deterministic and total.

use crate::data::FeatureId;
use crate::error::PbError;
use crate::{pb_seed, Stage};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const MAX_CAT_BINS: usize = 254;

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

impl CatEncoder {
    /// Serve-time target-statistic value for `label`; unseen levels map to
    /// [`CatEncoder::base`] (§04.8).
    #[must_use]
    pub fn encode_label(&self, label: &str) -> f32 {
        self.levels
            .iter()
            .find(|level| level.label == label)
            .map_or(self.base, |level| level.encoding)
    }
}

/// Per-fit categorical encoder specification (spec §04.3).
pub struct CatFitSpec<'a> {
    /// Raw feature this encoder belongs to.
    pub raw: FeatureId,
    /// Encoder id to stamp into [`crate::data::AxisKind::CategoricalTS`].
    pub id: TsEncodingId,
    /// Optional per-row weights; absent means all ones.
    pub weight: Option<&'a [f32]>,
    /// Optional exposure values `e_i`; absent means `e_i = 1`.
    pub exposure: Option<&'a [f32]>,
    /// Encoder configuration.
    pub config: &'a TsConfig,
    /// Deterministic base seed.
    pub seed: u64,
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

/// Fit a categorical target-statistic encoder (§04.3), returning the frozen
/// full-data serve encoder plus leakage-free per-row training encodings.
///
/// The full-data encoder is deterministic by level label order. Training encodings
/// use the configured leakage scheme: ordered prefix statistics never include the
/// current row, and K-fold encodings exclude the row's whole fold.
///
/// # Errors
/// [`PbError::ShapeMismatch`] on length mismatch; [`PbError::InvalidInput`] on bad
/// row values; [`PbError::InvalidConfig`] on unsupported/invalid config values.
pub fn fit_cat_encoder(
    levels: &[String],
    y: &[f32],
    spec: CatFitSpec<'_>,
) -> Result<(CatEncoder, Vec<f32>), PbError> {
    spec.config.validate()?;
    if levels.len() != y.len() {
        return Err(PbError::ShapeMismatch {
            what: format!("categorical fit: levels={}, y={}", levels.len(), y.len()),
        });
    }
    let weights;
    let w = match spec.weight {
        Some(w) => {
            if w.len() != y.len() {
                return Err(PbError::ShapeMismatch {
                    what: format!("categorical fit: y={}, weight={}", y.len(), w.len()),
                });
            }
            w
        }
        None => {
            weights = vec![1.0_f32; y.len()];
            &weights
        }
    };
    let base = exposure_weighted_base_rate(y, w, spec.exposure)?;
    let row_terms = categorical_row_terms(y, w, spec.exposure)?;
    let full = full_data_encoder(spec.raw, spec.id, levels, &row_terms, base, spec.config)?;
    let train = match spec.config.leakage {
        LeakageScheme::Ordered { n_perms } => ordered_training_encodings(
            levels,
            &row_terms,
            base,
            spec.config.smooth,
            spec.seed,
            n_perms,
        )?,
        LeakageScheme::KFold { k } => {
            kfold_training_encodings(levels, &row_terms, base, spec.config.smooth, spec.seed, k)?
        }
    };
    Ok((full, train))
}

#[derive(Debug, Clone, Copy, Default)]
struct CatRowTerm {
    sum_y: f64,
    denom: f64,
}

fn categorical_row_terms(
    y: &[f32],
    weight: &[f32],
    exposure: Option<&[f32]>,
) -> Result<Vec<CatRowTerm>, PbError> {
    let mut out = Vec::with_capacity(y.len());
    for (i, (&yi, &wi)) in y.iter().zip(weight).enumerate() {
        if !yi.is_finite() || !wi.is_finite() || wi < 0.0 {
            return Err(PbError::InvalidInput {
                what: format!("invalid categorical row {i}: y={yi}, weight={wi}"),
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
        out.push(CatRowTerm {
            sum_y: w * f64::from(yi),
            denom: w * e,
        });
    }
    Ok(out)
}

fn full_data_encoder(
    raw: FeatureId,
    id: TsEncodingId,
    levels: &[String],
    rows: &[CatRowTerm],
    base: f32,
    config: &TsConfig,
) -> Result<CatEncoder, PbError> {
    let mut agg: BTreeMap<String, CatRowTerm> = BTreeMap::new();
    for (label, term) in levels.iter().zip(rows) {
        let entry = agg.entry(label.clone()).or_default();
        entry.sum_y += term.sum_y;
        entry.denom += term.denom;
    }
    let mut out = Vec::with_capacity(agg.len());
    for (label, term) in agg {
        out.push(CatLevel {
            label,
            encoding: shrunken_encoding(term.sum_y, term.denom, base, config.smooth)?,
            bin: 0,
            weight: term.denom as f32,
        });
    }
    out.sort_by(|a, b| {
        a.encoding
            .total_cmp(&b.encoding)
            .then_with(|| a.label.cmp(&b.label))
    });
    let n_levels = out.len();
    let n_bins = n_levels.clamp(1, MAX_CAT_BINS);
    for (i, level) in out.iter_mut().enumerate() {
        let bin = 1 + (i * n_bins / n_levels);
        level.bin = u8::try_from(bin).map_err(|_| PbError::Internal {
            what: "categorical Fisher bin exceeded u8".into(),
        })?;
    }
    Ok(CatEncoder {
        raw,
        id,
        levels: out,
        base,
        config: config.clone(),
    })
}

fn ordered_training_encodings(
    levels: &[String],
    rows: &[CatRowTerm],
    base: f32,
    smooth: Smooth,
    seed: u64,
    n_perms: u32,
) -> Result<Vec<f32>, PbError> {
    if n_perms == 0 {
        return Err(PbError::InvalidConfig {
            what: "Ordered target statistics require n_perms > 0".into(),
        });
    }
    let n = levels.len();
    let mut out = vec![0.0_f64; n];
    for perm in 0..n_perms {
        let mut order = Vec::with_capacity(n);
        for row in 0..n {
            order.push((
                pb_seed(
                    seed,
                    perm,
                    Stage::Categorical as u32,
                    u32::try_from(row).map_err(|_| PbError::InvalidInput {
                        what: "categorical fit supports at most u32::MAX rows".into(),
                    })?,
                ),
                row,
            ));
        }
        order.sort_unstable_by_key(|(key, row)| (*key, *row));
        let mut prefix: BTreeMap<&str, CatRowTerm> = BTreeMap::new();
        for (_, row) in order {
            let label = levels.get(row).ok_or_else(|| PbError::Internal {
                what: "ordered categorical row escaped levels".into(),
            })?;
            let prev = prefix.get(label.as_str()).copied().unwrap_or_default();
            let enc = if prev.denom > 0.0 {
                shrunken_encoding(prev.sum_y, prev.denom, base, smooth)?
            } else {
                base
            };
            let slot = out.get_mut(row).ok_or_else(|| PbError::Internal {
                what: "ordered categorical row escaped output".into(),
            })?;
            *slot += f64::from(enc);
            let term = rows.get(row).ok_or_else(|| PbError::Internal {
                what: "ordered categorical row escaped terms".into(),
            })?;
            let entry = prefix.entry(label.as_str()).or_default();
            entry.sum_y += term.sum_y;
            entry.denom += term.denom;
        }
    }
    let scale = 1.0 / f64::from(n_perms);
    Ok(out.into_iter().map(|v| (v * scale) as f32).collect())
}

fn kfold_training_encodings(
    levels: &[String],
    rows: &[CatRowTerm],
    base: f32,
    smooth: Smooth,
    seed: u64,
    k: u32,
) -> Result<Vec<f32>, PbError> {
    if k < 2 {
        return Err(PbError::InvalidConfig {
            what: format!("KFold target statistics require k >= 2, got {k}"),
        });
    }
    let mut total: BTreeMap<&str, CatRowTerm> = BTreeMap::new();
    let mut by_fold: BTreeMap<(u32, &str), CatRowTerm> = BTreeMap::new();
    let mut folds = Vec::with_capacity(levels.len());
    for (row, (label, term)) in levels.iter().zip(rows).enumerate() {
        let fold = (pb_seed(
            seed,
            0,
            Stage::Categorical as u32,
            u32::try_from(row).map_err(|_| PbError::InvalidInput {
                what: "categorical fit supports at most u32::MAX rows".into(),
            })?,
        ) % u64::from(k)) as u32;
        folds.push(fold);
        let t = total.entry(label.as_str()).or_default();
        t.sum_y += term.sum_y;
        t.denom += term.denom;
        let f = by_fold.entry((fold, label.as_str())).or_default();
        f.sum_y += term.sum_y;
        f.denom += term.denom;
    }
    let mut out = Vec::with_capacity(levels.len());
    for (row, label) in levels.iter().enumerate() {
        let fold = *folds.get(row).ok_or_else(|| PbError::Internal {
            what: "kfold categorical row escaped folds".into(),
        })?;
        let all = total.get(label.as_str()).copied().unwrap_or_default();
        let held = by_fold
            .get(&(fold, label.as_str()))
            .copied()
            .unwrap_or_default();
        let term = CatRowTerm {
            sum_y: all.sum_y - held.sum_y,
            denom: all.denom - held.denom,
        };
        out.push(if term.denom > 0.0 {
            shrunken_encoding(term.sum_y, term.denom, base, smooth)?
        } else {
            base
        });
    }
    Ok(out)
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

    #[test]
    fn fit_cat_encoder_is_deterministic_and_freezes_full_data_map() {
        let levels = vec!["b", "a", "b", "c", "a", "c"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [2.0_f32, 1.0, 4.0, 8.0, 3.0, 10.0];
        let cfg = TsConfig {
            leakage: LeakageScheme::Ordered { n_perms: 3 },
            smooth: Smooth::Fixed { m: 2.0 },
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(2),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 99,
        };
        let (enc_a, train_a) = fit_cat_encoder(&levels, &y, spec).unwrap();
        let spec = CatFitSpec {
            raw: FeatureId(2),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 99,
        };
        let (enc_b, train_b) = fit_cat_encoder(&levels, &y, spec).unwrap();
        assert_eq!(enc_a, enc_b);
        assert_eq!(train_a, train_b);
        assert_eq!(enc_a.base, y.iter().sum::<f32>() / y.len() as f32);
        assert_eq!(
            enc_a
                .levels
                .iter()
                .map(|l| l.label.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        assert_eq!(
            enc_a.levels.iter().map(|l| l.bin).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(enc_a.encode_label("unseen"), enc_a.base);
        assert!(train_a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn full_data_encoder_uses_fisher_sorted_ordinal_order() {
        let levels = vec!["z", "a", "m", "z", "a", "m"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [30.0_f32, 1.0, 10.0, 40.0, 2.0, 12.0];
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 2 },
            smooth: Smooth::Fixed { m: 0.0 },
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(7),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 4,
        };
        let (enc, _) = fit_cat_encoder(&levels, &y, spec).unwrap();
        assert_eq!(
            enc.levels
                .iter()
                .map(|l| l.label.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "m", "z"]
        );
        assert_eq!(enc.encode_label("a"), 1.5);
        assert_eq!(enc.encode_label("m"), 11.0);
        assert_eq!(enc.encode_label("z"), 35.0);
        assert_eq!(
            enc.levels.iter().map(|l| l.bin).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn high_cardinality_levels_share_the_254_bin_budget() {
        let n = 300usize;
        let levels = (0..n).map(|i| format!("l{i:03}")).collect::<Vec<_>>();
        let y = (0..n).map(|i| i as f32).collect::<Vec<_>>();
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 5 },
            smooth: Smooth::Fixed { m: 0.0 },
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(3),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 10,
        };
        let (enc, _) = fit_cat_encoder(&levels, &y, spec).unwrap();
        assert_eq!(enc.levels.len(), n);
        assert_eq!(enc.levels.first().unwrap().bin, 1);
        assert_eq!(enc.levels.last().unwrap().bin, 254);
        assert!(enc.levels.windows(2).all(|w| w[0].bin <= w[1].bin));
    }

    #[test]
    fn kfold_training_encoding_does_not_consult_own_target() {
        let levels = vec!["a".to_string(); 30];
        let y = (0..30).map(|i| i as f32).collect::<Vec<_>>();
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 5 },
            smooth: Smooth::Fixed { m: 0.0 },
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(0),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 123,
        };
        let (_, train_a) = fit_cat_encoder(&levels, &y, spec).unwrap();
        let mut changed = y.clone();
        changed[0] += 10_000.0;
        let spec = CatFitSpec {
            raw: FeatureId(0),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 123,
        };
        let (_, train_b) = fit_cat_encoder(&levels, &changed, spec).unwrap();
        assert_eq!(train_a[0], train_b[0]);
    }
}
