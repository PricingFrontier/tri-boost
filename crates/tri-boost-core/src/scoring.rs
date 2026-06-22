//! Runtime-only scoring views (spec §10.2a / §11.9).
//!
//! [`ScoringBank`] is a load-derived view of a [`crate::Model`]: it packs each tree
//! into a cache-friendly row for prediction, but it is never serialized and never owns
//! independent model semantics. Every scoring path is tested bit-equal to
//! [`crate::Model::score_trees_row`].

use crate::engine::{low_bit, Model};
use crate::error::PbError;

/// A 64-byte runtime scoring row for one oblivious tree.
///
/// The leaves are byte-exact copies of the model leaves; `alpha` is kept separate so
/// DART/Nesterov/ensemble weights do not mutate leaf values in this derived view.
/// This type is never serialized.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PackedTree {
    /// Axis id per level. Valid only for models whose split axes fit in `u8`.
    pub feat: [u8; 3],
    /// Threshold bin per level.
    pub thresh: [u8; 3],
    /// Missing-left bits packed by level.
    pub miss: u8,
    /// Tree depth, `1..=3`.
    pub depth: u8,
    /// Tree multiplier.
    pub alpha: f32,
    /// Leaf lookup table.
    pub leaf: [f32; 8],
    _pad: [u8; 20],
}

impl PackedTree {
    fn score_row(&self, row: &[u8]) -> Result<f32, PbError> {
        let mut idx = 0usize;
        for level in 0..usize::from(self.depth) {
            let axis = *self.feat.get(level).ok_or_else(|| PbError::Internal {
                what: "packed tree level escaped feat array".into(),
            })? as usize;
            let thresh = *self.thresh.get(level).ok_or_else(|| PbError::Internal {
                what: "packed tree level escaped thresh array".into(),
            })?;
            let bin = *row.get(axis).ok_or_else(|| PbError::ShapeMismatch {
                what: format!("row has no axis {axis} for packed scoring"),
            })?;
            let missing_left = ((self.miss >> level) & 1) != 0;
            idx |= usize::from(low_bit(bin, thresh, missing_left)) << level;
        }
        let leaf = self
            .leaf
            .get(idx)
            .copied()
            .ok_or_else(|| PbError::Internal {
                what: "packed leaf index escaped 0..8".into(),
            })?;
        Ok(self.alpha * leaf)
    }
}

/// Runtime scoring row for models whose split axes do not fit in `u8`.
///
/// Fields stay private because this is a derived view, not a wire or modeling API.
#[derive(Debug, Clone, PartialEq)]
pub struct WideTree {
    feat: [u32; 3],
    thresh: [u8; 3],
    miss: u8,
    depth: u8,
    alpha: f32,
    leaf: [f32; 8],
}

impl WideTree {
    fn score_row(&self, row: &[u8]) -> Result<f32, PbError> {
        let mut idx = 0usize;
        for level in 0..usize::from(self.depth) {
            let axis = *self.feat.get(level).ok_or_else(|| PbError::Internal {
                what: "wide tree level escaped feat array".into(),
            })? as usize;
            let thresh = *self.thresh.get(level).ok_or_else(|| PbError::Internal {
                what: "wide tree level escaped thresh array".into(),
            })?;
            let bin = *row.get(axis).ok_or_else(|| PbError::ShapeMismatch {
                what: format!("row has no axis {axis} for wide scoring"),
            })?;
            let missing_left = ((self.miss >> level) & 1) != 0;
            idx |= usize::from(low_bit(bin, thresh, missing_left)) << level;
        }
        let leaf = self
            .leaf
            .get(idx)
            .copied()
            .ok_or_else(|| PbError::Internal {
                what: "wide leaf index escaped 0..8".into(),
            })?;
        Ok(self.alpha * leaf)
    }
}

/// Runtime-only path-A scoring view.
#[derive(Debug, Clone, PartialEq)]
pub enum ScoringBank {
    /// All split axes fit in `u8`, so each tree can use the compact 64-byte layout.
    Packed {
        /// Packed trees in stored model order.
        trees: Vec<PackedTree>,
    },
    /// A model has at least one axis id above 255; keep `u32` axes instead of
    /// truncating. Correctness is identical, only the compact layout is forfeited.
    Wide {
        /// Wide-axis trees in stored model order.
        trees: Vec<WideTree>,
    },
}

impl ScoringBank {
    /// Build a runtime scoring view from a validated model.
    ///
    /// # Errors
    /// Propagates [`Model::validate`] failures.
    pub fn from_model(model: &Model) -> Result<Self, PbError> {
        model.validate()?;
        let packed_ok = model
            .trees
            .iter()
            .flat_map(|(_, tree)| tree.splits.iter())
            .all(|split| u8::try_from(split.axis).is_ok());
        if packed_ok {
            let mut trees = Vec::with_capacity(model.trees.len());
            for (alpha, tree) in &model.trees {
                let mut feat = [0u8; 3];
                let mut thresh = [0u8; 3];
                let mut miss = 0u8;
                for (level, split) in tree.splits.iter().enumerate() {
                    let feat_slot = feat.get_mut(level).ok_or_else(|| PbError::Internal {
                        what: "tree depth escaped packed feat array".into(),
                    })?;
                    *feat_slot = u8::try_from(split.axis).map_err(|_| PbError::Internal {
                        what: "packed_ok accepted a non-u8 axis".into(),
                    })?;
                    let thresh_slot = thresh.get_mut(level).ok_or_else(|| PbError::Internal {
                        what: "tree depth escaped packed thresh array".into(),
                    })?;
                    *thresh_slot = split.bin_le;
                    if split.missing_left {
                        miss |= 1_u8 << level;
                    }
                }
                trees.push(PackedTree {
                    feat,
                    thresh,
                    miss,
                    depth: tree.depth,
                    alpha: *alpha,
                    leaf: tree.leaves,
                    _pad: [0; 20],
                });
            }
            Ok(ScoringBank::Packed { trees })
        } else {
            let mut trees = Vec::with_capacity(model.trees.len());
            for (alpha, tree) in &model.trees {
                let mut feat = [0u32; 3];
                let mut thresh = [0u8; 3];
                let mut miss = 0u8;
                for (level, split) in tree.splits.iter().enumerate() {
                    let feat_slot = feat.get_mut(level).ok_or_else(|| PbError::Internal {
                        what: "tree depth escaped wide feat array".into(),
                    })?;
                    *feat_slot = split.axis;
                    let thresh_slot = thresh.get_mut(level).ok_or_else(|| PbError::Internal {
                        what: "tree depth escaped wide thresh array".into(),
                    })?;
                    *thresh_slot = split.bin_le;
                    if split.missing_left {
                        miss |= 1_u8 << level;
                    }
                }
                trees.push(WideTree {
                    feat,
                    thresh,
                    miss,
                    depth: tree.depth,
                    alpha: *alpha,
                    leaf: tree.leaves,
                });
            }
            Ok(ScoringBank::Wide { trees })
        }
    }

    /// Score one already-binned row in raw-score space.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if the row lacks a referenced axis.
    pub fn score_row(&self, row: &[u8], offset: f32) -> Result<f32, PbError> {
        let mut acc = offset;
        match self {
            ScoringBank::Packed { trees } => {
                for tree in trees {
                    acc += tree.score_row(row)?;
                }
            }
            ScoringBank::Wide { trees } => {
                for tree in trees {
                    acc += tree.score_row(row)?;
                }
            }
        }
        Ok(acc)
    }

    /// Number of trees in the view.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            ScoringBank::Packed { trees } => trees.len(),
            ScoringBank::Wide { trees } => trees.len(),
        }
    }

    /// `true` if the view contains no trees.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
    use crate::data::{AxisKind, AxisProvenance, BorderGrid, FeatureId};
    use crate::engine::{ModelSchema, Split};
    use crate::explain::{fixture_model, fixture_serve};

    #[test]
    fn packed_tree_is_one_cache_line() {
        assert_eq!(std::mem::size_of::<PackedTree>(), 64);
        assert_eq!(std::mem::align_of::<PackedTree>(), 64);
    }

    #[test]
    fn packed_scoring_matches_model_tree_walk() {
        let model = fixture_model();
        let bank = ScoringBank::from_model(&model).unwrap();
        assert!(matches!(bank, ScoringBank::Packed { .. }));
        let x = fixture_serve();
        for row in 0..x.0.n_rows as usize {
            let bins: Vec<u8> = x.0.data.iter().map(|c| c[row]).collect();
            let model_score = model.score_trees_row(&bins, 0.0).unwrap();
            let packed_score = bank.score_row(&bins, model.f0).unwrap();
            assert_eq!(packed_score.to_bits(), model_score.to_bits());
        }
    }

    #[test]
    fn missing_left_bit_is_preserved() {
        let mut model = fixture_model();
        model.trees[0].1.splits[0].missing_left = true;
        let bank = ScoringBank::from_model(&model).unwrap();
        let row = [0_u8, 1_u8];
        assert_eq!(
            bank.score_row(&row, model.f0).unwrap().to_bits(),
            model.score_trees_row(&row, 0.0).unwrap().to_bits()
        );
    }

    #[test]
    fn wide_axis_fallback_matches_model_tree_walk() {
        let mut model = fixture_model();
        let wide_axis = 300usize;
        while model.grids.len() <= wide_axis {
            model.grids.push(BorderGrid {
                borders: vec![1.5],
                n_bins: 3,
                missing_bin: 0,
            });
            let raw = u32::try_from(model.provenance.len()).unwrap();
            model.provenance.push(AxisProvenance {
                raw: FeatureId(raw),
                kind: AxisKind::Numeric,
            });
            model.schema.feature_names.push(format!("f{raw}"));
            model.schema.feature_kinds.push(AxisKind::Numeric);
        }
        model.trees[0].1.splits[0] = Split {
            axis: u32::try_from(wide_axis).unwrap(),
            bin_le: 1,
            missing_left: false,
        };
        model.schema = ModelSchema {
            feature_names: model.schema.feature_names.clone(),
            feature_kinds: model.schema.feature_kinds.clone(),
            cat_encoders: model.schema.cat_encoders.clone(),
            class_labels: None,
            objective: model.schema.objective.clone(),
        };
        let bank = ScoringBank::from_model(&model).unwrap();
        assert!(matches!(bank, ScoringBank::Wide { .. }));
        let mut row = vec![2_u8; model.grids.len()];
        row[wide_axis] = 1;
        row[1] = 1;
        assert_eq!(
            bank.score_row(&row, model.f0).unwrap().to_bits(),
            model.score_trees_row(&row, 0.0).unwrap().to_bits()
        );
    }
}
