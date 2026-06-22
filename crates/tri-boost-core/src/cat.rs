//! Categorical handling (spec §04). Phase-0 stubs: the identifiers and the frozen
//! encoder store are registered here so `AxisKind`/`ModelSchema` can name them; the
//! leakage-free Target-Statistic encoding and the audit-on-serve machinery land
//! with §04.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Identifier for one categorical Target-Statistic encoding (spec §04). Resolves to
/// a concrete [`CatEncoder`] in the [`CatEncoderStore`]. A fixed-width `u32` because
/// it is serialized inside `AxisKind` (no `usize` on the wire, §02.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TsEncodingId(pub u32);

/// One frozen, full-data categorical encoder (spec §04): a category → TS value/bin
/// map plus level labels. Phase-0 placeholder — fields land with §04.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CatEncoder {}

/// The frozen `TsEncodingId → CatEncoder` table backing serve/export (spec §2.6 /
/// §04, R-SCHEMA). `explain()` and `TableBank` accumulation re-encode raw
/// categoricals through THESE (never the noisy train-time encoders).
///
/// Backed by a `BTreeMap` (not `HashMap`): serialized state must have deterministic
/// iteration order (the `check-no-hashmap-serialized` gate).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CatEncoderStore {
    encoders: BTreeMap<u32, CatEncoder>,
}

impl CatEncoderStore {
    /// An empty store (no categorical axes). Used by purely-numeric models.
    #[must_use]
    pub fn new() -> Self {
        Self {
            encoders: BTreeMap::new(),
        }
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
}
