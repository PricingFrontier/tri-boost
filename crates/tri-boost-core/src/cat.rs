//! Categorical handling (spec §04). This module owns the frozen Target-Statistic
//! encoder schema that is serialized with a [`crate::Model`]. The leakage-free fit
//! algorithms land on top of these types; the store and lookup semantics are already
//! deterministic and total.

use crate::data::{BorderGrid, FeatureId};
use crate::error::PbError;
use crate::{pb_seed, Stage};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const MAX_CAT_BINS: usize = 254;
const MAX_AUTO_SMOOTH: f64 = 1_000_000.0;
const MIN_AUTO_VARIANCE: f64 = 1.0e-12;
const RARE_LEVEL_LABEL: &str = "__tri_boost_rare__";

type RareMembers = BTreeMap<String, Vec<String>>;

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
    /// Leave-one-out target statistics: each row sees its level's full statistics with
    /// ITSELF removed. Deterministic and knob-free, but NOT the default: although it is the
    /// nominal cross-fit variance floor, it reintroduces a per-row dependence on the row's
    /// own target (`enc ≈ (Σ − yᵣ)/(n−1)`) that depth-3 trees partially invert — measured
    /// WORSE than KFold on both MTPL tasks (the classic LOO target-encoding pathology).
    LeaveOneOut,
}

impl Default for LeakageScheme {
    fn default() -> Self {
        // K-fold cross-fit: each row's encoding comes from OTHER folds, so (unlike LOO) rows
        // in the same fold share a held-out value and the per-row self-dependence is broken.
        // Empirically the lowest-variance / best-accuracy leakage-free scheme on MTPL
        // (beats the old `Ordered{1}` default and LOO on both frequency and severity).
        Self::KFold { k: 5 }
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

/// Target transform used by the categorical target-statistic encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CatTarget {
    /// Encode the exposure-weighted mean target, matching the original TS path.
    Mean,
    /// Encode the weighted mean of `log(y)`, useful for positive severity targets.
    LogMean,
}

impl Default for CatTarget {
    fn default() -> Self {
        Self::Mean
    }
}

/// Configuration for one target-statistic encoder (spec §04.3/§04.12).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TsConfig {
    /// Leakage-avoidance scheme for training rows.
    pub leakage: LeakageScheme,
    /// Shrinkage rule.
    pub smooth: Smooth,
    /// Target transform used before shrinkage.
    pub target: CatTarget,
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
            target: CatTarget::default(),
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
            LeakageScheme::LeaveOneOut => {}
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
    /// Original labels represented by this row. Non-rare rows contain exactly their
    /// own label; the rare row contains all collapsed training labels.
    pub members: Vec<String>,
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
            .find(|level| level.label == label || level.members.iter().any(|m| m == label))
            .map_or(self.base, |level| level.encoding)
    }

    /// Build a `label → encoding` lookup map (each level's own label AND its `members`) for batch
    /// serve-time encoding — O(1) per label vs [`encode_label`]'s O(#levels) linear scan. Labels and
    /// members are disjoint across levels, so `map.get(label)` returns exactly what `encode_label`'s
    /// `find` would; unseen labels still fall back to [`CatEncoder::base`] at the call site. The map
    /// is a local lookup (never iterated for output order, never serialized).
    #[must_use]
    pub fn encoding_map(&self) -> std::collections::HashMap<&str, f32> {
        let mut map = std::collections::HashMap::with_capacity(self.levels.len());
        for level in &self.levels {
            map.insert(level.label.as_str(), level.encoding);
            for member in &level.members {
                map.insert(member.as_str(), level.encoding);
            }
        }
        map
    }

    /// Reconstruct the ordinal Fisher [`BorderGrid`] for this encoder.
    ///
    /// The grid is derived from the stored level encodings and their frozen bin ids,
    /// so it round-trips with the model schema without persisting duplicate border
    /// state. Unseen levels are encoded as [`CatEncoder::base`] and then binned
    /// against this same grid.
    ///
    /// # Errors
    /// [`PbError::InvalidInput`] if the stored encoder has non-finite encodings or
    /// inconsistent bin ordering; [`PbError::Internal`] on impossible width casts.
    pub fn border_grid(&self) -> Result<BorderGrid, PbError> {
        let mut levels = self.levels.clone();
        levels.sort_by(|a, b| {
            a.encoding
                .total_cmp(&b.encoding)
                .then_with(|| a.label.cmp(&b.label))
        });

        let mut borders = Vec::new();
        let mut prev: Option<(u8, f32)> = None;
        for level in &levels {
            if !level.encoding.is_finite() {
                return Err(PbError::InvalidInput {
                    what: format!(
                        "categorical encoder {:?}/{:?} has non-finite encoding for `{}`",
                        self.raw, self.id, level.label
                    ),
                });
            }
            if level.bin == 0 {
                return Err(PbError::InvalidInput {
                    what: format!(
                        "categorical encoder {:?}/{:?} assigned reserved bin 0 to `{}`",
                        self.raw, self.id, level.label
                    ),
                });
            }
            if let Some((prev_bin, prev_encoding)) = prev {
                if level.bin < prev_bin {
                    return Err(PbError::InvalidInput {
                        what: "categorical encoder bins must be non-decreasing in encoding order"
                            .into(),
                    });
                }
                if level.bin > prev_bin {
                    if level.encoding <= prev_encoding {
                        return Err(PbError::InvalidInput {
                            what: "categorical encoder split tied encodings across bins".into(),
                        });
                    }
                    let border =
                        ((f64::from(prev_encoding) + f64::from(level.encoding)) / 2.0) as f32;
                    if !border.is_finite() {
                        return Err(PbError::InvalidInput {
                            what: "categorical Fisher border is not finite".into(),
                        });
                    }
                    if borders.last().is_some_and(|last| *last >= border) {
                        return Err(PbError::InvalidInput {
                            what: "categorical Fisher borders must be strictly ascending".into(),
                        });
                    }
                    borders.push(border);
                }
            }
            prev = Some((level.bin, level.encoding));
        }
        let n_bins =
            u16::try_from(
                borders
                    .len()
                    .checked_add(2)
                    .ok_or_else(|| PbError::Internal {
                        what: "categorical border count overflow".into(),
                    })?,
            )
            .map_err(|_| PbError::Internal {
                what: "categorical n_bins exceeded u16".into(),
            })?;
        Ok(BorderGrid {
            borders,
            n_bins,
            missing_bin: 0,
        })
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
    let row_terms = categorical_row_terms(y, w, spec.exposure, spec.config.target)?;
    let (fit_levels, members) =
        collapse_rare_levels(levels, &row_terms, spec.config.min_data_per_group)?;
    // Intern the collapsed per-row labels into integer ids ONCE, so the dominant full-data + k-fold
    // encoders index dense Vecs by id instead of string-keyed BTreeMaps per row (the high-card
    // binning cost). Byte-identical (every reduction stays in row order; Fisher re-sorts by label).
    let (fit_ids, id_labels) = intern_levels(&fit_levels)?;
    let base = categorical_base_from_terms(&row_terms, spec.config.target)?;
    let smooth = resolve_smooth(&fit_levels, &row_terms, base, spec.config.smooth)?;
    let mut resolved_config = spec.config.clone();
    resolved_config.smooth = smooth;
    let full = full_data_encoder(
        spec.raw,
        spec.id,
        &fit_ids,
        &id_labels,
        &row_terms,
        base,
        &resolved_config,
        &members,
    )?;
    let train = match spec.config.leakage {
        // Ordered / LOO are non-default and keep the per-row string path (lower priority).
        LeakageScheme::Ordered { n_perms } => {
            ordered_training_encodings(&fit_levels, &row_terms, base, smooth, spec.seed, n_perms)?
        }
        LeakageScheme::KFold { k } => kfold_training_encodings(
            &fit_ids,
            id_labels.len(),
            &row_terms,
            base,
            smooth,
            spec.seed,
            k,
        )?,
        LeakageScheme::LeaveOneOut => {
            loo_training_encodings(&fit_levels, &row_terms, base, smooth)?
        }
    };
    Ok((full, train))
}

/// Intern per-row (collapsed) labels into integer ids in FIRST-SEEN / ROW order, returning the
/// per-row id vector and the id→label table. The `HashMap` is a local lookup only (never iterated,
/// never serialized) so the result is deterministic by row order. Lets the full-data + k-fold
/// encoders index dense `Vec`s by id — byte-identical to the string-keyed path.
fn intern_levels(levels: &[String]) -> Result<(Vec<u32>, Vec<&str>), PbError> {
    let mut id_of: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    let mut id_labels: Vec<&str> = Vec::new();
    let mut ids: Vec<u32> = Vec::with_capacity(levels.len());
    for label in levels {
        let id = match id_of.get(label.as_str()) {
            Some(&id) => id,
            None => {
                let id = u32::try_from(id_labels.len()).map_err(|_| PbError::InvalidInput {
                    what: "categorical fit supports at most u32::MAX distinct levels".into(),
                })?;
                id_labels.push(label.as_str());
                id_of.insert(label.as_str(), id);
                id
            }
        };
        ids.push(id);
    }
    Ok((ids, id_labels))
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
    target: CatTarget,
) -> Result<Vec<CatRowTerm>, PbError> {
    if let Some(ex) = exposure {
        if ex.len() != y.len() {
            return Err(PbError::ShapeMismatch {
                what: format!("categorical fit: y={}, exposure={}", y.len(), ex.len()),
            });
        }
    }
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
        let (sum_y, denom) = match target {
            CatTarget::Mean => (w * f64::from(yi), w * e),
            CatTarget::LogMean => {
                if yi <= 0.0 {
                    return Err(PbError::InvalidInput {
                        what: format!("cat_target='log_mean' requires y[{i}] > 0, got {yi}"),
                    });
                }
                (w * f64::from(yi).ln(), w)
            }
        };
        out.push(CatRowTerm { sum_y, denom });
    }
    Ok(out)
}

fn categorical_base_from_terms(rows: &[CatRowTerm], target: CatTarget) -> Result<f32, PbError> {
    let mut sum_y = 0.0_f64;
    let mut denom = 0.0_f64;
    for term in rows {
        sum_y += term.sum_y;
        denom += term.denom;
    }
    if denom <= 0.0 {
        let label = match target {
            CatTarget::Mean => "categorical base rate denominator Σw·e",
            CatTarget::LogMean => "categorical log-mean denominator Σw",
        };
        return Err(PbError::InvalidInput {
            what: format!("{label} must be > 0"),
        });
    }
    let out = (sum_y / denom) as f32;
    if out.is_finite() {
        Ok(out)
    } else {
        Err(PbError::InvalidInput {
            what: format!(
                "categorical base target statistic is not representable as f32: {}",
                sum_y / denom
            ),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn full_data_encoder(
    raw: FeatureId,
    id: TsEncodingId,
    fit_ids: &[u32],
    id_labels: &[&str],
    rows: &[CatRowTerm],
    base: f32,
    config: &TsConfig,
    members: &BTreeMap<String, Vec<String>>,
) -> Result<CatEncoder, PbError> {
    // Aggregate (sum_y, denom) per level id by walking rows in order — same row order, hence the
    // same f64 sums, as the old string-keyed BTreeMap. `out` is built in id order, but
    // `assign_fisher_bins` re-sorts by (encoding, label) (labels distinct ⇒ total order), so the
    // final encoder is byte-identical regardless of the build order.
    let mut agg = vec![CatRowTerm::default(); id_labels.len()];
    for (&lid, term) in fit_ids.iter().zip(rows) {
        let entry = agg.get_mut(lid as usize).ok_or_else(|| PbError::Internal {
            what: "full-data encoder level id escaped".into(),
        })?;
        entry.sum_y += term.sum_y;
        entry.denom += term.denom;
    }
    let mut out = Vec::with_capacity(id_labels.len());
    for (lid, term) in agg.iter().enumerate() {
        let label = *id_labels.get(lid).ok_or_else(|| PbError::Internal {
            what: "full-data encoder id label escaped".into(),
        })?;
        out.push(CatLevel {
            members: members
                .get(label)
                .cloned()
                .unwrap_or_else(|| vec![label.to_owned()]),
            label: label.to_owned(),
            encoding: shrunken_encoding(term.sum_y, term.denom, base, config.smooth)?,
            bin: 0,
            weight: term.denom as f32,
        });
    }
    assign_fisher_bins(&mut out)?;
    Ok(CatEncoder {
        raw,
        id,
        levels: out,
        base,
        config: config.clone(),
    })
}

fn collapse_rare_levels(
    levels: &[String],
    rows: &[CatRowTerm],
    min_data_per_group: f32,
) -> Result<(Vec<String>, RareMembers), PbError> {
    let mut agg: BTreeMap<&str, f64> = BTreeMap::new();
    for (label, term) in levels.iter().zip(rows) {
        let entry = agg.entry(label.as_str()).or_default();
        *entry += term.denom;
    }

    if min_data_per_group <= 0.0 {
        let mut members = BTreeMap::new();
        for label in agg.keys() {
            members.insert((*label).to_owned(), vec![(*label).to_owned()]);
        }
        return Ok((levels.to_vec(), members));
    }

    if agg.contains_key(RARE_LEVEL_LABEL) {
        return Err(PbError::InvalidInput {
            what: format!("categorical label `{RARE_LEVEL_LABEL}` is reserved for rare buckets"),
        });
    }

    let min = f64::from(min_data_per_group);
    let mut rare_members = Vec::new();
    let mut members = BTreeMap::new();
    for (label, denom) in &agg {
        if *denom < min {
            rare_members.push((*label).to_owned());
        } else {
            members.insert((*label).to_owned(), vec![(*label).to_owned()]);
        }
    }
    if !rare_members.is_empty() {
        members.insert(RARE_LEVEL_LABEL.to_owned(), rare_members.clone());
    }

    // Membership test against a set (O(1)/row) instead of `rare_members.iter().any()` (O(rare)/row,
    // i.e. O(n·rare) total on high-cardinality columns). Byte-identical: same partition into rare
    // (→ RARE_LEVEL_LABEL) vs kept (→ label).
    let rare_set: std::collections::HashSet<&str> =
        rare_members.iter().map(String::as_str).collect();
    let collapsed = levels
        .iter()
        .map(|label| {
            if rare_set.contains(label.as_str()) {
                RARE_LEVEL_LABEL.to_owned()
            } else {
                label.clone()
            }
        })
        .collect();
    Ok((collapsed, members))
}

fn resolve_smooth(
    levels: &[String],
    rows: &[CatRowTerm],
    base: f32,
    smooth: Smooth,
) -> Result<Smooth, PbError> {
    match smooth {
        Smooth::Fixed { m } => Ok(Smooth::Fixed { m }),
        Smooth::Auto => {
            let m = auto_smooth_strength(levels, rows, base)?;
            Ok(Smooth::Fixed { m })
        }
    }
}

fn auto_smooth_strength(levels: &[String], rows: &[CatRowTerm], base: f32) -> Result<f32, PbError> {
    let mut agg: BTreeMap<&str, CatRowTerm> = BTreeMap::new();
    for (label, term) in levels.iter().zip(rows) {
        let entry = agg.entry(label.as_str()).or_default();
        entry.sum_y += term.sum_y;
        entry.denom += term.denom;
    }
    let mut total = 0.0_f64;
    let mut between = 0.0_f64;
    for term in agg.values() {
        if term.denom > 0.0 {
            let mean = term.sum_y / term.denom;
            let diff = mean - f64::from(base);
            between += term.denom * diff * diff;
            total += term.denom;
        }
    }
    if total <= 0.0 {
        return Err(PbError::InvalidInput {
            what: "Smooth::Auto requires positive categorical weight".into(),
        });
    }
    between /= total;

    let mut within = 0.0_f64;
    for (label, term) in levels.iter().zip(rows) {
        if term.denom > 0.0 {
            let level = agg.get(label.as_str()).ok_or_else(|| PbError::Internal {
                what: "auto-smooth level missing from aggregate".into(),
            })?;
            let level_mean = level.sum_y / level.denom;
            let row_rate = term.sum_y / term.denom;
            let diff = row_rate - level_mean;
            within += term.denom * diff * diff;
        }
    }
    within /= total;

    let m = if between <= MIN_AUTO_VARIANCE {
        if within <= MIN_AUTO_VARIANCE {
            0.0
        } else {
            MAX_AUTO_SMOOTH
        }
    } else {
        (within / between).min(MAX_AUTO_SMOOTH)
    };
    let out = m as f32;
    if out.is_finite() && out >= 0.0 {
        Ok(out)
    } else {
        Err(PbError::InvalidInput {
            what: "Smooth::Auto produced a non-finite shrinkage strength".into(),
        })
    }
}

fn assign_fisher_bins(levels: &mut [CatLevel]) -> Result<(), PbError> {
    levels.sort_by(|a, b| {
        a.encoding
            .total_cmp(&b.encoding)
            .then_with(|| a.label.cmp(&b.label))
    });
    let mut distinct = Vec::<f32>::new();
    for level in levels.iter() {
        if !level.encoding.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!(
                    "categorical level `{}` has non-finite encoding",
                    level.label
                ),
            });
        }
        if distinct.last() != Some(&level.encoding) {
            distinct.push(level.encoding);
        }
    }
    let n_distinct = distinct.len();
    if n_distinct == 0 {
        return Ok(());
    }
    let n_bins = n_distinct.clamp(1, MAX_CAT_BINS);
    let mut rank = 0usize;
    let mut prev_encoding: Option<f32> = None;
    for level in levels.iter_mut() {
        if let Some(prev) = prev_encoding {
            if level.encoding != prev {
                rank = rank.checked_add(1).ok_or_else(|| PbError::Internal {
                    what: "categorical rank overflow".into(),
                })?;
            }
        }
        let bin = 1 + (rank * n_bins / n_distinct);
        level.bin = u8::try_from(bin).map_err(|_| PbError::Internal {
            what: "categorical Fisher bin exceeded u8".into(),
        })?;
        prev_encoding = Some(level.encoding);
    }
    Ok(())
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

/// Leave-one-out training encodings: each row sees its level's full `(Σwy, Σw)` with its
/// OWN term subtracted, then shrunk. The variance floor of the cross-fit family (no fold or
/// permutation noise), deterministic and seed-free, and leakage-free (the row's own target
/// never enters its encoding). Falls back to `base` when the row is the only member of its
/// level. Summation is in fixed row order ⇒ thread-count independent.
fn loo_training_encodings(
    levels: &[String],
    rows: &[CatRowTerm],
    base: f32,
    smooth: Smooth,
) -> Result<Vec<f32>, PbError> {
    let mut total: BTreeMap<&str, CatRowTerm> = BTreeMap::new();
    for (label, term) in levels.iter().zip(rows) {
        let t = total.entry(label.as_str()).or_default();
        t.sum_y += term.sum_y;
        t.denom += term.denom;
    }
    let mut out = Vec::with_capacity(levels.len());
    for (label, term) in levels.iter().zip(rows) {
        let all = total.get(label.as_str()).copied().unwrap_or_default();
        let held = CatRowTerm {
            sum_y: all.sum_y - term.sum_y,
            denom: all.denom - term.denom,
        };
        out.push(if held.denom > 0.0 {
            shrunken_encoding(held.sum_y, held.denom, base, smooth)?
        } else {
            base
        });
    }
    Ok(out)
}

fn kfold_training_encodings(
    fit_ids: &[u32],
    n_ids: usize,
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
    // Dense `[id]` total and `[fold*n_ids + id]` per-fold accumulators (vs the old string-keyed
    // `BTreeMap<&str>` / `BTreeMap<(u32,&str)>`). Same row order ⇒ byte-identical f64 sums, and the
    // OOF held-out subtraction `total[id] − by_fold[fold,id]` reads the same values.
    let fold_cells = usize::try_from(k)
        .ok()
        .and_then(|kk| kk.checked_mul(n_ids))
        .ok_or_else(|| PbError::Internal {
            what: "kfold accumulator size overflow".into(),
        })?;
    let mut total = vec![CatRowTerm::default(); n_ids];
    let mut by_fold = vec![CatRowTerm::default(); fold_cells];
    let mut folds = Vec::with_capacity(fit_ids.len());
    for (row, (&lid, term)) in fit_ids.iter().zip(rows).enumerate() {
        let fold = (pb_seed(
            seed,
            0,
            Stage::Categorical as u32,
            u32::try_from(row).map_err(|_| PbError::InvalidInput {
                what: "categorical fit supports at most u32::MAX rows".into(),
            })?,
        ) % u64::from(k)) as u32;
        folds.push(fold);
        let t = total
            .get_mut(lid as usize)
            .ok_or_else(|| PbError::Internal {
                what: "kfold total level id escaped".into(),
            })?;
        t.sum_y += term.sum_y;
        t.denom += term.denom;
        let fc = (fold as usize)
            .checked_mul(n_ids)
            .and_then(|b| b.checked_add(lid as usize))
            .ok_or_else(|| PbError::Internal {
                what: "kfold by_fold index overflow".into(),
            })?;
        let f = by_fold.get_mut(fc).ok_or_else(|| PbError::Internal {
            what: "kfold by_fold cell escaped".into(),
        })?;
        f.sum_y += term.sum_y;
        f.denom += term.denom;
    }
    let mut out = Vec::with_capacity(fit_ids.len());
    for (row, &lid) in fit_ids.iter().enumerate() {
        let fold = *folds.get(row).ok_or_else(|| PbError::Internal {
            what: "kfold categorical row escaped folds".into(),
        })?;
        let all = *total.get(lid as usize).ok_or_else(|| PbError::Internal {
            what: "kfold total read escaped".into(),
        })?;
        let fc = (fold as usize)
            .checked_mul(n_ids)
            .and_then(|b| b.checked_add(lid as usize))
            .ok_or_else(|| PbError::Internal {
                what: "kfold by_fold read index overflow".into(),
            })?;
        let held = *by_fold.get(fc).ok_or_else(|| PbError::Internal {
            what: "kfold by_fold read escaped".into(),
        })?;
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
                members: vec![label.into()],
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
    fn smooth_auto_resolves_to_closed_form_variance_ratio() {
        let levels = vec!["a", "a", "b", "b"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [0.0_f32, 2.0, 4.0, 8.0];
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 2 },
            smooth: Smooth::Auto,
            min_data_per_group: 0.0,
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(5),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 3,
        };
        let (enc, train) = fit_cat_encoder(&levels, &y, spec).unwrap();
        assert!(train.iter().all(|v| v.is_finite()));
        let resolved_m = match enc.config.smooth {
            Smooth::Fixed { m } => m,
            Smooth::Auto => f32::NAN,
        };
        assert!((resolved_m - 0.4).abs() < 1e-6);
        assert!((enc.encode_label("a") - (2.0 + 0.4 * 3.5) / 2.4).abs() < 1e-6);
        assert!((enc.encode_label("b") - (12.0 + 0.4 * 3.5) / 2.4).abs() < 1e-6);
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
            min_data_per_group: 0.0,
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
    fn log_mean_target_encodes_positive_targets_on_log_scale() {
        let levels = vec!["low", "low", "high", "high"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [2.0_f32, 8.0, 20.0, 80.0];
        let weight = [1.0_f32, 3.0, 2.0, 2.0];
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 2 },
            smooth: Smooth::Fixed { m: 0.0 },
            target: CatTarget::LogMean,
            min_data_per_group: 0.0,
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(2),
            id: TsEncodingId(0),
            weight: Some(&weight),
            exposure: None,
            config: &cfg,
            seed: 99,
        };
        let (enc, train) = fit_cat_encoder(&levels, &y, spec).unwrap();
        let low = (2.0_f32.ln() + 3.0 * 8.0_f32.ln()) / 4.0;
        let high = (2.0 * 20.0_f32.ln() + 2.0 * 80.0_f32.ln()) / 4.0;
        assert!((enc.encode_label("low") - low).abs() < 1e-6);
        assert!((enc.encode_label("high") - high).abs() < 1e-6);
        assert_eq!(enc.config.target, CatTarget::LogMean);
        assert!(train.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn log_mean_target_rejects_non_positive_targets() {
        let levels = vec!["a".to_owned(), "b".to_owned()];
        let y = [1.0_f32, 0.0];
        let cfg = TsConfig {
            target: CatTarget::LogMean,
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
        assert!(matches!(
            fit_cat_encoder(&levels, &y, spec),
            Err(PbError::InvalidInput { .. })
        ));
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
            min_data_per_group: 0.0,
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
    fn rare_levels_collapse_before_fisher_ordering_but_unseen_uses_base() {
        let levels = vec!["rare_a", "common", "common", "rare_b", "common"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [100.0_f32, 1.0, 3.0, 200.0, 5.0];
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 2 },
            smooth: Smooth::Fixed { m: 0.0 },
            min_data_per_group: 2.0,
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(8),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 5,
        };
        let (enc, train) = fit_cat_encoder(&levels, &y, spec).unwrap();
        assert_eq!(
            enc.levels
                .iter()
                .map(|level| level.label.as_str())
                .collect::<Vec<_>>(),
            vec!["common", RARE_LEVEL_LABEL]
        );
        let rare = enc
            .levels
            .iter()
            .find(|level| level.label == RARE_LEVEL_LABEL)
            .unwrap();
        assert_eq!(
            rare.members,
            vec!["rare_a".to_string(), "rare_b".to_string()]
        );
        assert_eq!(enc.encode_label("rare_a"), rare.encoding);
        assert_eq!(enc.encode_label("rare_b"), rare.encoding);
        assert_eq!(enc.encode_label("brand_new"), enc.base);
        assert_eq!(train.len(), levels.len());
        assert!(train.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn high_cardinality_levels_share_the_254_bin_budget() {
        let n = 300usize;
        let levels = (0..n).map(|i| format!("l{i:03}")).collect::<Vec<_>>();
        let y = (0..n).map(|i| i as f32).collect::<Vec<_>>();
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 5 },
            smooth: Smooth::Fixed { m: 0.0 },
            min_data_per_group: 0.0,
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
    fn tied_encodings_share_bins_and_reconstruct_a_grid() {
        let levels = vec!["b", "a", "d", "c"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [1.0_f32, 1.0, 1.0, 1.0];
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 2 },
            smooth: Smooth::Fixed { m: 0.0 },
            min_data_per_group: 0.0,
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(9),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 1,
        };
        let (enc, _) = fit_cat_encoder(&levels, &y, spec).unwrap();
        assert!(enc.levels.iter().all(|level| level.bin == 1));
        let grid = enc.border_grid().unwrap();
        assert!(grid.borders.is_empty());
        assert_eq!(grid.n_bins, 2);
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

    #[test]
    fn leave_one_out_excludes_each_rows_own_target() {
        // Two rows in one level; m=0 ⇒ no shrinkage, so each row's LOO encoding is exactly
        // the OTHER row's target (its own is excluded) — leakage-free by construction.
        let levels = vec!["a".to_string(), "a".to_string()];
        let rows = vec![
            CatRowTerm {
                sum_y: 1.0,
                denom: 1.0,
            },
            CatRowTerm {
                sum_y: 3.0,
                denom: 1.0,
            },
        ];
        let enc = loo_training_encodings(&levels, &rows, 2.0, Smooth::Fixed { m: 0.0 }).unwrap();
        assert!(
            (enc[0] - 3.0).abs() < 1e-6,
            "row 0 sees only row 1's target"
        );
        assert!(
            (enc[1] - 1.0).abs() < 1e-6,
            "row 1 sees only row 0's target"
        );
        // A singleton level has no other rows ⇒ falls back to base.
        let solo = loo_training_encodings(
            &["b".to_string()],
            &[CatRowTerm {
                sum_y: 5.0,
                denom: 1.0,
            }],
            2.0,
            Smooth::Fixed { m: 0.0 },
        )
        .unwrap();
        assert!((solo[0] - 2.0).abs() < 1e-6, "singleton ⇒ base");
    }

    #[test]
    fn default_leakage_scheme_is_kfold() {
        assert!(matches!(
            LeakageScheme::default(),
            LeakageScheme::KFold { k: 5 }
        ));
    }
}
