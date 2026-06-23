//! Inference & serialization (spec §2.6, §02.8 / §10): the versioned model wire
//! envelope, JSON/bincode helpers, validation-on-load, and the rating-table export
//! artifact.
//!
//! The binary path uses bincode 2.x's `encode_to_vec`/`decode_from_slice` with the
//! config **frozen** to `bincode::config::standard()`. `ModelDoc` is a plain nested
//! struct (NOT `#[serde(flatten)]`, which is incompatible with the non-self-describing
//! bincode round-trip).

use crate::engine::{ExactnessMode, Model, ModelSchema};
use crate::error::PbError;
use crate::explain::{FeatureSet, RefMeasure, TableBank};
use crate::loss::{Link, ObjectiveTag};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The on-disk container format version (the envelope around `Model`). Bumped on any
/// wire-incompatible change to the *framing*; a load of a newer `format_version` is
/// rejected.
pub const FORMAT_VERSION: u32 = 1;

/// The `Model`/`schema` wire-schema version (spec §2.6). A single monotone `u32`
/// bumped on any wire-incompatible change to the model contents.
pub const SCHEMA_VERSION: u32 = 1;

/// The serialized envelope (spec §02.8): a plain nested `{ format_version,
/// schema_version, model }`. No `#[serde(flatten)]` — it does not round-trip through
/// non-self-describing bincode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelDoc {
    /// The container framing version (see [`FORMAT_VERSION`]).
    pub format_version: u32,
    /// The model/schema wire version (see [`SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// The wrapped trained model.
    pub model: Model,
}

impl ModelDoc {
    /// Wrap `model` in a current-version envelope.
    #[must_use]
    pub fn new(model: Model) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            schema_version: SCHEMA_VERSION,
            model,
        }
    }
}

fn serialization_error(e: impl ToString) -> PbError {
    PbError::Serialization(e.to_string())
}

fn validate_doc(doc: &ModelDoc) -> Result<(), PbError> {
    if doc.format_version != FORMAT_VERSION {
        return Err(PbError::Serialization(format!(
            "unsupported format_version {} (this build supports exactly {FORMAT_VERSION})",
            doc.format_version
        )));
    }
    if doc.schema_version != SCHEMA_VERSION {
        return Err(PbError::Serialization(format!(
            "unsupported schema_version {} (this build supports exactly {SCHEMA_VERSION})",
            doc.schema_version
        )));
    }
    doc.model.validate()
}

/// Encode a [`ModelDoc`] to the fast binary format (bincode 2.x, frozen
/// `config::standard()`).
///
/// # Errors
/// [`PbError::Serialization`] if encoding fails.
pub fn encode_doc(doc: &ModelDoc) -> Result<Vec<u8>, PbError> {
    bincode::serde::encode_to_vec(doc, bincode::config::standard()).map_err(serialization_error)
}

/// Decode a [`ModelDoc`] from the fast binary format, rejecting a newer
/// `format_version` (spec §02.8 version gate).
///
/// # Errors
/// [`PbError::Serialization`] if decoding fails or the framing version is unknown.
pub fn decode_doc(bytes: &[u8]) -> Result<ModelDoc, PbError> {
    let (doc, len): (ModelDoc, usize) =
        bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map_err(serialization_error)?;
    if len != bytes.len() {
        return Err(PbError::Serialization(format!(
            "trailing bytes after ModelDoc: decoded {len}, input {}",
            bytes.len()
        )));
    }
    validate_doc(&doc)?;
    Ok(doc)
}

/// Encode a [`ModelDoc`] to the canonical pretty JSON format.
///
/// # Errors
/// [`PbError::Serialization`] if encoding fails.
pub fn encode_doc_json(doc: &ModelDoc) -> Result<String, PbError> {
    serde_json::to_string_pretty(doc).map_err(serialization_error)
}

/// Decode a [`ModelDoc`] from canonical JSON and validate the contained model.
///
/// # Errors
/// [`PbError::Serialization`] if decoding fails or versions are unsupported; plus
/// model validation errors.
pub fn decode_doc_json(s: &str) -> Result<ModelDoc, PbError> {
    let doc: ModelDoc = serde_json::from_str(s).map_err(serialization_error)?;
    validate_doc(&doc)?;
    Ok(doc)
}

/// Migrate a JSON [`ModelDoc`] value from `from_format_version` to the current model.
///
/// There are no historical versions before [`FORMAT_VERSION`] yet, so the only
/// accepted migration today is the identity migration. The explicit facade is still
/// useful: newer documents fail closed before deserialization details leak, and future
/// version shims have a single public entry point.
///
/// # Errors
/// [`PbError::Serialization`] if `from_format_version` is newer than this build, has
/// no registered migration path, disagrees with the document's own version, or if the
/// migrated model fails validation.
pub fn migrate(value: serde_json::Value, from_format_version: u32) -> Result<Model, PbError> {
    if from_format_version > FORMAT_VERSION {
        return Err(PbError::Serialization(format!(
            "cannot migrate future format_version {from_format_version}; this build supports {FORMAT_VERSION}"
        )));
    }
    if from_format_version != FORMAT_VERSION {
        return Err(PbError::Serialization(format!(
            "no migration path registered from format_version {from_format_version} to {FORMAT_VERSION}"
        )));
    }
    let doc: ModelDoc = serde_json::from_value(value).map_err(serialization_error)?;
    if doc.format_version != from_format_version {
        return Err(PbError::Serialization(format!(
            "document format_version {} disagrees with requested migration source {from_format_version}",
            doc.format_version
        )));
    }
    validate_doc(&doc)?;
    Ok(doc.model)
}

/// Encode a bare [`Model`] (wrapped in a current-version [`ModelDoc`]) to binary.
///
/// # Errors
/// [`PbError::Serialization`] if encoding fails.
pub fn encode_model(model: &Model) -> Result<Vec<u8>, PbError> {
    model.validate()?;
    encode_doc(&ModelDoc::new(model.clone()))
}

/// Decode a bare [`Model`] from binary, applying the version gate.
///
/// # Errors
/// [`PbError::Serialization`] if decoding fails or a version is unsupported.
pub fn decode_model(bytes: &[u8]) -> Result<Model, PbError> {
    Ok(decode_doc(bytes)?.model)
}

/// Encode a bare [`Model`] to canonical JSON.
///
/// # Errors
/// [`PbError::Serialization`] if encoding fails; plus validation errors.
pub fn encode_model_json(model: &Model) -> Result<String, PbError> {
    model.validate()?;
    encode_doc_json(&ModelDoc::new(model.clone()))
}

/// Decode a bare [`Model`] from canonical JSON.
///
/// # Errors
/// [`PbError::Serialization`] if decoding fails or versions are unsupported; plus
/// model validation errors.
pub fn decode_model_json(s: &str) -> Result<Model, PbError> {
    Ok(decode_doc_json(s)?.model)
}

impl Model {
    /// Serialize this model to the compact same-version bincode cache format.
    ///
    /// # Errors
    /// [`PbError::Serialization`] if encoding fails; plus validation errors.
    pub fn to_bincode(&self) -> Result<Vec<u8>, PbError> {
        encode_model(self)
    }

    /// Deserialize a model from the compact same-version bincode cache format.
    ///
    /// # Errors
    /// [`PbError::Serialization`] if decoding fails or versions are unsupported; plus
    /// model validation errors.
    pub fn from_bincode(bytes: &[u8]) -> Result<Self, PbError> {
        decode_model(bytes)
    }

    /// Serialize this model to canonical pretty JSON.
    ///
    /// # Errors
    /// [`PbError::Serialization`] if encoding fails; plus validation errors.
    pub fn to_json(&self) -> Result<String, PbError> {
        encode_model_json(self)
    }

    /// Deserialize a model from canonical JSON.
    ///
    /// # Errors
    /// [`PbError::Serialization`] if decoding fails or versions are unsupported; plus
    /// model validation errors.
    pub fn from_json(s: &str) -> Result<Self, PbError> {
        decode_model_json(s)
    }
}

/// One reference-cell selector: a table support (sorted distinct raw feature ids) and
/// the merged-cell coordinate within that table that should read as neutral.
///
/// A flat struct of `Vec<u32>` fields (rather than a `FeatureSet`-keyed map) so the
/// enclosing [`RatingBasis`] round-trips through JSON — a `FeatureSet` (a sequence) is
/// not a valid JSON object key, so the former `BTreeMap<FeatureSet, _>` could not be
/// authored as `basis_json` through the public Python API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RatingReference {
    /// The table support, as sorted distinct raw feature ids.
    pub feature_set: Vec<u32>,
    /// The reference coordinate within that table (one merged-cell id per axis).
    pub coord: Vec<u32>,
}

/// Optional reference-cell selector for rating-view exports. Each entry identifies a
/// table support and the coordinate within that table that should read as neutral
/// (`0.0` in score space, `1.000` as a log-link relativity). The shifted mass is
/// folded into the exported intercept, so reconstructed scores are unchanged.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RatingBasis {
    /// Per-table reference coordinates. JSON-representable (a sequence of entries, not a
    /// non-string-keyed map).
    pub reference: Vec<RatingReference>,
}

impl RatingBasis {
    /// The reference coordinate for a table support `u`, matched by its sorted raw ids.
    #[must_use]
    pub fn coord_for(&self, u: &FeatureSet) -> Option<&[u32]> {
        self.reference.iter().find_map(|entry| {
            let matches = entry.feature_set.len() == u.0.len()
                && entry
                    .feature_set
                    .iter()
                    .zip(u.0.iter())
                    .all(|(&f, raw)| f == raw.0);
            if matches {
                Some(entry.coord.as_slice())
            } else {
                None
            }
        })
    }
}

/// One exported rating-table axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisExport {
    /// Raw feature id.
    pub raw: u32,
    /// Human-readable feature name from [`ModelSchema`].
    pub name: String,
    /// Merged-grid finite borders.
    pub borders: Vec<f32>,
    /// Number of cells including the explicit missing cell.
    pub cells: u32,
}

/// One exported purified rating table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RatingTable {
    /// Raw feature set this table represents.
    pub feature_set: FeatureSet,
    /// Feature names parallel to [`RatingTable::feature_set`].
    pub feature_names: Vec<String>,
    /// Axis metadata parallel to tensor dimensions.
    pub axes: Vec<AxisExport>,
    /// Tensor shape as fixed-width dimensions.
    pub shape: Vec<u32>,
    /// Score-space table values in dense row-major order.
    pub values: Vec<f64>,
    /// Log-link relativities (`exp(value)`) when applicable; `None` otherwise.
    pub relativities: Option<Vec<f64>>,
    /// Per-cell support counts, display-only.
    pub support: Vec<f64>,
    /// Optional per-cell standard-error band, display-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub se_band: Option<Vec<f64>>,
    /// Cached table variance.
    pub variance: f64,
    /// Sobol share under the bank's reference measure.
    pub sobol: f64,
}

/// The rating-table export artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RatingExport {
    /// Export wire format version.
    pub format_version: u32,
    /// Model/schema wire version.
    pub schema_version: u32,
    /// Exactness mode carried by the source model.
    pub mode: ExactnessMode,
    /// The trained inverse-link family.
    pub link: Link,
    /// The trained objective tag.
    pub objective: ObjectiveTag,
    /// Exported intercept, possibly shifted by [`RatingBasis`] rebasing.
    pub f0: f64,
    /// Reference measure used for purification.
    pub reference_measure: RefMeasure,
    /// Sobol-sorted purified tables.
    pub tables: Vec<RatingTable>,
}

impl TableBank {
    /// Export this exact bank as a rating-table artifact.
    ///
    /// # Errors
    /// [`PbError::ExactnessFirewall`] if `mode` is approximate; [`PbError::ShapeMismatch`]
    /// if schema/table metadata is inconsistent; [`PbError::Internal`] if a reference
    /// coordinate escapes a tensor.
    pub fn to_rating_export(
        &self,
        link: Link,
        mode: &ExactnessMode,
        schema: &ModelSchema,
        basis: Option<&RatingBasis>,
    ) -> Result<RatingExport, PbError> {
        if let ExactnessMode::Approximate { reason } = mode {
            return Err(PbError::ExactnessFirewall(reason.clone()));
        }
        let sobol: BTreeMap<FeatureSet, f64> = self.sobol().into_iter().collect();
        let mut f0 = self.f0;
        let mut tables = Vec::with_capacity(self.tables.len());
        for table in &self.tables {
            let mut values = table.values.clone();
            if let Some(coord) = basis.and_then(|b| b.coord_for(&table.u)) {
                if coord.len() != table.u.order() {
                    return Err(PbError::ShapeMismatch {
                        what: format!(
                            "rating basis for order-{} table has {} coordinates",
                            table.u.order(),
                            coord.len()
                        ),
                    });
                }
                let coord_usize: Vec<usize> = coord.iter().map(|&c| c as usize).collect();
                let shift = values.at(&coord_usize).ok_or_else(|| PbError::Internal {
                    what: "rating basis coordinate escaped table".into(),
                })?;
                values.add_scalar(-shift);
                f0 += shift;
            }
            let mut feature_names = Vec::with_capacity(table.u.order());
            for raw in &table.u.0 {
                let name = schema
                    .feature_names
                    .get(raw.0 as usize)
                    .ok_or_else(|| PbError::ShapeMismatch {
                        what: format!("schema missing feature name for raw {}", raw.0),
                    })?
                    .clone();
                feature_names.push(name);
            }
            let mut axes = Vec::with_capacity(table.axes.len());
            for axis in &table.axes {
                let name = schema
                    .feature_names
                    .get(axis.raw.0 as usize)
                    .ok_or_else(|| PbError::ShapeMismatch {
                        what: format!("schema missing feature name for raw {}", axis.raw.0),
                    })?
                    .clone();
                axes.push(AxisExport {
                    raw: axis.raw.0,
                    name,
                    borders: axis.borders.clone(),
                    cells: axis.cells,
                });
            }
            let relativities = if link == Link::Log {
                Some(
                    values
                        .values()
                        .iter()
                        .map(|v| v.clamp(-30.0, 30.0).exp())
                        .collect(),
                )
            } else {
                None
            };
            tables.push(RatingTable {
                feature_set: table.u.clone(),
                feature_names,
                axes,
                shape: values.shape_u32().to_vec(),
                values: values.values().to_vec(),
                relativities,
                support: table.support.values().to_vec(),
                se_band: table
                    .se_band
                    .as_ref()
                    .map(|band| band.per_cell.values().to_vec()),
                variance: table.variance,
                sobol: *sobol.get(&table.u).unwrap_or(&0.0),
            });
        }
        tables.sort_by(|a, b| b.sobol.total_cmp(&a.sobol));
        Ok(RatingExport {
            format_version: FORMAT_VERSION,
            schema_version: SCHEMA_VERSION,
            mode: mode.clone(),
            link,
            objective: schema.objective.clone(),
            f0,
            reference_measure: self.w.clone(),
            tables,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]
    use super::*;
    use crate::engine::ExactnessMode;
    use crate::explain::{fixture_serve, RefMeasure};

    #[test]
    fn bincode_round_trip_is_bit_identical() {
        let model = crate::explain::fixture_model();
        let bytes = encode_model(&model).unwrap();
        let back = decode_model(&bytes).unwrap();
        assert_eq!(model, back);
        // Re-encoding the decoded model reproduces the exact bytes.
        assert_eq!(bytes, encode_model(&back).unwrap());
    }

    #[test]
    fn json_round_trip_matches_and_model_methods_work() {
        let doc = ModelDoc::new(crate::explain::fixture_model());
        let json = encode_doc_json(&doc).unwrap();
        let back = decode_doc_json(&json).unwrap();
        assert_eq!(doc, back);

        let model_json = doc.model.to_json().unwrap();
        assert_eq!(Model::from_json(&model_json).unwrap(), doc.model);
        let model_bytes = doc.model.to_bincode().unwrap();
        assert_eq!(Model::from_bincode(&model_bytes).unwrap(), doc.model);
    }

    #[test]
    fn migrate_identity_loads_current_version_and_revalidates() {
        let doc = ModelDoc::new(crate::explain::fixture_model());
        let value = serde_json::to_value(&doc).unwrap();
        assert_eq!(migrate(value, FORMAT_VERSION).unwrap(), doc.model);

        let value = serde_json::to_value(&doc).unwrap();
        assert!(matches!(
            migrate(value, FORMAT_VERSION + 1),
            Err(PbError::Serialization(_))
        ));

        let mut mismatched = serde_json::to_value(&doc).unwrap();
        mismatched["format_version"] = serde_json::json!(FORMAT_VERSION + 1);
        assert!(matches!(
            migrate(mismatched, FORMAT_VERSION),
            Err(PbError::Serialization(_))
        ));
    }

    #[test]
    fn schema_metadata_round_trips_through_json_and_bincode() {
        let mut model = crate::explain::fixture_model();
        model.schema.feature_names = vec!["territory".into(), "age_band".into()];
        model.schema.class_labels = Some(vec!["low".into(), "high".into()]);

        let json = model.to_json().unwrap();
        let from_json = Model::from_json(&json).unwrap();
        assert_eq!(from_json.schema.feature_names, model.schema.feature_names);
        assert_eq!(from_json.schema.class_labels, model.schema.class_labels);

        let bytes = model.to_bincode().unwrap();
        let from_bytes = Model::from_bincode(&bytes).unwrap();
        assert_eq!(from_bytes.schema.feature_names, model.schema.feature_names);
        assert_eq!(from_bytes.schema.class_labels, model.schema.class_labels);
    }

    #[test]
    fn bumped_format_version_is_rejected() {
        let mut doc = ModelDoc::new(crate::explain::fixture_model());
        doc.format_version = FORMAT_VERSION + 1;
        let bytes = encode_doc(&doc).unwrap();
        assert!(matches!(decode_doc(&bytes), Err(PbError::Serialization(_))));
    }

    #[test]
    fn bumped_schema_version_is_rejected() {
        let mut doc = ModelDoc::new(crate::explain::fixture_model());
        doc.schema_version = SCHEMA_VERSION + 1;
        let bytes = encode_doc(&doc).unwrap();
        assert!(matches!(decode_doc(&bytes), Err(PbError::Serialization(_))));
    }

    #[test]
    fn trailing_bincode_bytes_are_rejected() {
        let mut bytes = encode_model(&crate::explain::fixture_model()).unwrap();
        bytes.push(0);
        assert!(matches!(
            decode_model(&bytes),
            Err(PbError::Serialization(_))
        ));
    }

    #[test]
    fn decode_revalidates_tree_axes() {
        let mut doc = ModelDoc::new(crate::explain::fixture_model());
        doc.model.trees[0].1.splits[0].axis = 99;
        let bytes = encode_doc(&doc).unwrap();
        assert!(decode_doc(&bytes).is_err());
    }

    #[test]
    fn decode_revalidates_schema_lengths() {
        let mut doc = ModelDoc::new(crate::explain::fixture_model());
        doc.model.schema.feature_kinds.pop();
        let json = encode_doc_json(&doc).unwrap();
        assert!(decode_doc_json(&json).is_err());
    }

    #[test]
    fn path_a_scoring_matches_ensemble_and_predicts_response() {
        let model = crate::explain::fixture_model();
        let x = fixture_serve();
        let mut out = vec![0.0_f32; x.0.n_rows as usize];
        model.score_trees(&x.0, None, &mut out).unwrap();
        for (row, &score) in out.iter().enumerate() {
            let bins: Vec<u8> = x.0.data.iter().map(|c| c[row]).collect();
            assert_eq!(
                score.to_bits(),
                (model.ensemble_f64(&bins).unwrap() as f32).to_bits()
            );
        }
        assert_eq!(model.predict_binned(&x.0, None).unwrap(), out);
        assert_eq!(model.predict(&x.0, None).unwrap(), out);
    }

    #[test]
    fn rating_export_pure_and_rebased_forms_are_exact() {
        let model = crate::explain::fixture_model();
        let x = fixture_serve();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let pure = bank
            .to_rating_export(model.link, &model.mode, &model.schema, None)
            .unwrap();
        assert_eq!(pure.mode, ExactnessMode::Exact);
        assert_eq!(pure.tables.len(), bank.tables.len());

        let first = bank.tables[0].clone();
        let coord = vec![0_u32; first.u.order()];
        let shift = first.values.at(&vec![0_usize; first.u.order()]).unwrap();
        let basis = RatingBasis {
            reference: vec![RatingReference {
                feature_set: first.u.0.iter().map(|f| f.0).collect(),
                coord,
            }],
        };
        // A NON-EMPTY RatingBasis must round-trip through JSON (the public basis_json path).
        let json = serde_json::to_string(&basis).unwrap();
        let basis: RatingBasis = serde_json::from_str(&json).unwrap();
        let rebased = bank
            .to_rating_export(model.link, &model.mode, &model.schema, Some(&basis))
            .unwrap();
        assert!((rebased.f0 - (bank.f0 + shift)).abs() < 1e-12);
        let table = rebased
            .tables
            .iter()
            .find(|t| t.feature_set == first.u)
            .unwrap();
        assert!(table.values[0].abs() < 1e-12);
    }

    #[test]
    fn rating_export_refuses_approximate_mode() {
        let model = crate::explain::fixture_model();
        let x = fixture_serve();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let mode = ExactnessMode::Approximate {
            reason: "test".into(),
        };
        assert!(matches!(
            bank.to_rating_export(model.link, &mode, &model.schema, None),
            Err(PbError::ExactnessFirewall(_))
        ));
    }
}
