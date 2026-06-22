//! Inference & serialization (spec §2.6, §02.8 / §10). Phase-0 scope: the version
//! constants, the `ModelDoc` envelope, and the frozen-config bincode round-trip with
//! a `format_version` gate. The scoring path (`ScoringBank`/`PackedTree`), the JSON
//! rating-table export, and the `usize`→`u32` wasm guard land with §10.
//!
//! The binary path uses bincode 2.x's `encode_to_vec`/`decode_from_slice` with the
//! config **frozen** to `bincode::config::standard()`. `ModelDoc` is a plain nested
//! struct (NOT `#[serde(flatten)]`, which is incompatible with the non-self-describing
//! bincode round-trip).

use crate::engine::Model;
use crate::error::PbError;
use serde::{Deserialize, Serialize};

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

/// Encode a [`ModelDoc`] to the fast binary format (bincode 2.x, frozen
/// `config::standard()`).
///
/// # Errors
/// [`PbError::Serialization`] if encoding fails.
pub fn encode_doc(doc: &ModelDoc) -> Result<Vec<u8>, PbError> {
    bincode::serde::encode_to_vec(doc, bincode::config::standard())
        .map_err(|e| PbError::Serialization(e.to_string()))
}

/// Decode a [`ModelDoc`] from the fast binary format, rejecting a newer
/// `format_version` (spec §02.8 version gate).
///
/// # Errors
/// [`PbError::Serialization`] if decoding fails or the framing version is unknown.
pub fn decode_doc(bytes: &[u8]) -> Result<ModelDoc, PbError> {
    let (doc, _len): (ModelDoc, usize) =
        bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map_err(|e| PbError::Serialization(e.to_string()))?;
    if doc.format_version > FORMAT_VERSION {
        return Err(PbError::Serialization(format!(
            "unsupported format_version {} (this build supports up to {FORMAT_VERSION})",
            doc.format_version
        )));
    }
    if doc.schema_version > SCHEMA_VERSION {
        return Err(PbError::Serialization(format!(
            "unsupported schema_version {} (this build supports up to {SCHEMA_VERSION})",
            doc.schema_version
        )));
    }
    Ok(doc)
}

/// Encode a bare [`Model`] (wrapped in a current-version [`ModelDoc`]) to binary.
///
/// # Errors
/// [`PbError::Serialization`] if encoding fails.
pub fn encode_model(model: &Model) -> Result<Vec<u8>, PbError> {
    encode_doc(&ModelDoc::new(model.clone()))
}

/// Decode a bare [`Model`] from binary, applying the version gate.
///
/// # Errors
/// [`PbError::Serialization`] if decoding fails or a version is unsupported.
pub fn decode_model(bytes: &[u8]) -> Result<Model, PbError> {
    Ok(decode_doc(bytes)?.model)
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
    fn json_round_trip_matches() {
        let doc = ModelDoc::new(crate::explain::fixture_model());
        let json = serde_json::to_vec(&doc).unwrap();
        let back: ModelDoc = serde_json::from_slice(&json).unwrap();
        assert_eq!(doc, back);
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
}
