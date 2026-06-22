//! Runtime-only scoring views (spec §10.2a / §11.9).
//!
//! [`ScoringBank`] is a load-derived view of a [`crate::Model`]: it packs each tree
//! into a cache-friendly row for prediction, but it is never serialized and never owns
//! independent model semantics. Every scoring path is tested bit-equal to
//! [`crate::Model::score_trees_row`].

use crate::data::{BinnedMatrix, BorderGrid};
use crate::engine::{low_bit, Model};
use crate::error::PbError;
use crate::explain::TableBank;

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

#[derive(Debug, Clone, PartialEq)]
struct FlatTable {
    axes: Vec<u32>,
    shape: Vec<u32>,
    strides: Vec<usize>,
    values: Vec<f64>,
}

impl FlatTable {
    fn offset(&self, x_cells: &[u32]) -> Result<usize, PbError> {
        let mut off = 0usize;
        for ((&raw, &dim), &stride) in self.axes.iter().zip(&self.shape).zip(&self.strides) {
            let cell = *x_cells
                .get(raw as usize)
                .ok_or_else(|| PbError::ShapeMismatch {
                    what: format!("x_cells missing raw feature {raw} for flat table scoring"),
                })?;
            if cell >= dim {
                return Err(PbError::InvalidInput {
                    what: format!("raw feature {raw} cell {cell} outside merged extent {dim}"),
                });
            }
            let cell = usize::try_from(cell).map_err(|_| PbError::Internal {
                what: "merged cell id exceeded usize".into(),
            })?;
            off = off
                .checked_add(cell.checked_mul(stride).ok_or_else(|| PbError::Internal {
                    what: "flat table offset multiplication overflowed".into(),
                })?)
                .ok_or_else(|| PbError::Internal {
                    what: "flat table offset addition overflowed".into(),
                })?;
        }
        Ok(off)
    }

    fn eval(&self, x_cells: &[u32]) -> Result<f64, PbError> {
        let off = self.offset(x_cells)?;
        self.values
            .get(off)
            .copied()
            .ok_or_else(|| PbError::Internal {
                what: "flat table offset escaped values arena".into(),
            })
    }
}

/// Runtime-only path-B table scorer.
///
/// This is the flattened single-arena view from spec §10.3: every table in a
/// [`TableBank`] is validated once, copied into row-major flat storage, and then rows
/// are scored by digitizing model bins into merged-grid cells exactly once per row.
/// Like [`ScoringBank`], it is never serialized and owns no independent model
/// semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct TableScoringBank {
    f0: f64,
    tables: Vec<FlatTable>,
    merged_grids: Vec<BorderGrid>,
}

impl TableScoringBank {
    /// Build a runtime table scorer from an exact [`TableBank`].
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if a table's axes and tensor shape disagree;
    /// [`PbError::Internal`] if stride/product arithmetic overflows.
    pub fn from_bank(bank: &TableBank) -> Result<Self, PbError> {
        let mut tables = Vec::with_capacity(bank.tables.len());
        for table in &bank.tables {
            let shape = table.values.shape_u32().to_vec();
            if shape.len() != table.axes.len() {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "table order {} has {} axes but tensor rank {}",
                        table.u.order(),
                        table.axes.len(),
                        shape.len()
                    ),
                });
            }
            for (axis, &dim) in table.axes.iter().zip(&shape) {
                if axis.cells != dim {
                    return Err(PbError::ShapeMismatch {
                        what: format!(
                            "table axis raw {} cells {} != tensor dim {dim}",
                            axis.raw.0, axis.cells
                        ),
                    });
                }
            }
            let mut expected = 1usize;
            for &dim in &shape {
                expected = expected
                    .checked_mul(usize::try_from(dim).map_err(|_| PbError::Internal {
                        what: "tensor dim exceeded usize".into(),
                    })?)
                    .ok_or_else(|| PbError::Internal {
                        what: "flat table cell count overflowed".into(),
                    })?;
            }
            if expected != table.values.values().len() {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "table values len {} != product(shape) {expected}",
                        table.values.values().len()
                    ),
                });
            }
            let mut strides = vec![1usize; shape.len()];
            let mut suffix = 1usize;
            for (slot, &dim) in strides.iter_mut().rev().zip(shape.iter().rev()) {
                *slot = suffix;
                suffix = suffix
                    .checked_mul(usize::try_from(dim).map_err(|_| PbError::Internal {
                        what: "tensor dim exceeded usize".into(),
                    })?)
                    .ok_or_else(|| PbError::Internal {
                        what: "flat table stride overflowed".into(),
                    })?;
            }
            tables.push(FlatTable {
                axes: table.axes.iter().map(|axis| axis.raw.0).collect(),
                shape,
                strides,
                values: table.values.values().to_vec(),
            });
        }
        Ok(Self {
            f0: bank.f0,
            tables,
            merged_grids: bank.merged_grids.clone(),
        })
    }

    /// Score one merged-cell row in raw-score space.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `x_cells` lacks a referenced raw feature;
    /// [`PbError::InvalidInput`] if a cell id exceeds a table extent.
    pub fn score_cells_row(&self, x_cells: &[u32]) -> Result<f64, PbError> {
        let mut acc = self.f0;
        for table in &self.tables {
            acc += table.eval(x_cells)?;
        }
        Ok(acc)
    }

    /// Score an already-binned matrix through the flat table arena in raw-score space.
    ///
    /// The input matrix must use the model's original grids; this scorer derives the
    /// merged-grid cell ids once per row and then reads the flat tables in fixed order.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] for width/length mismatches; [`PbError::InvalidInput`]
    /// if a model bin cannot be mapped to the bank's merged grid.
    pub fn score_binned(&self, x: &BinnedMatrix, out: &mut [f64]) -> Result<(), PbError> {
        let n_rows = x.n_rows as usize;
        if out.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("out len {} != n_rows {n_rows}", out.len()),
            });
        }
        let maps = build_cell_maps(&self.merged_grids, x)?;
        let mut cells = vec![0u32; self.merged_grids.len()];
        for row in 0..n_rows {
            fill_row_cells(x, &maps, row, &mut cells)?;
            let score = self.score_cells_row(&cells)?;
            let dst = out.get_mut(row).ok_or_else(|| PbError::Internal {
                what: "flat table output row escaped buffer".into(),
            })?;
            *dst = score;
        }
        Ok(())
    }
}

fn build_cell_maps(
    merged_grids: &[BorderGrid],
    x: &BinnedMatrix,
) -> Result<Vec<Vec<u32>>, PbError> {
    if x.data.len() != merged_grids.len() {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "matrix has {} columns, table bank has {} merged grids",
                x.data.len(),
                merged_grids.len()
            ),
        });
    }
    if x.grids.len() != merged_grids.len() {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "matrix has {} grids, table bank has {} merged grids",
                x.grids.len(),
                merged_grids.len()
            ),
        });
    }
    let n_rows = x.n_rows as usize;
    for (axis, col) in x.data.iter().enumerate() {
        if col.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("matrix column {axis} len {} != n_rows {n_rows}", col.len()),
            });
        }
    }

    let mut maps = Vec::with_capacity(merged_grids.len());
    for (axis, (merged, model)) in merged_grids.iter().zip(&x.grids).enumerate() {
        if merged.missing_bin != 0 || model.missing_bin != 0 {
            return Err(PbError::InvalidInput {
                what: format!("axis {axis} missing_bin must be 0 for table scoring"),
            });
        }
        if merged.n_bins == 0 || model.n_bins == 0 {
            return Err(PbError::InvalidInput {
                what: format!("axis {axis} has zero bins"),
            });
        }
        let mut merged_border_index = Vec::with_capacity(merged.borders.len());
        for &border in &merged.borders {
            let pos = model
                .borders
                .iter()
                .position(|&candidate| candidate.to_bits() == border.to_bits())
                .ok_or_else(|| PbError::InvalidInput {
                    what: format!(
                        "merged border {border} for axis {axis} is absent from model grid"
                    ),
                })?;
            merged_border_index.push(pos);
        }
        let mut map = Vec::with_capacity(usize::from(model.n_bins));
        for bin in 0..model.n_bins {
            let cell = if bin == 0 {
                0usize
            } else {
                let threshold = i64::from(bin) - 2;
                merged_border_index
                    .iter()
                    .filter(|&&idx| (idx as i64) <= threshold)
                    .count()
                    + 1
            };
            if cell >= usize::from(merged.n_bins) {
                return Err(PbError::InvalidInput {
                    what: format!(
                        "axis {axis} model bin {bin} maps to merged cell {cell}, outside {}",
                        merged.n_bins
                    ),
                });
            }
            map.push(u32::try_from(cell).map_err(|_| PbError::Internal {
                what: "merged cell id exceeded u32".into(),
            })?);
        }
        maps.push(map);
    }
    Ok(maps)
}

fn fill_row_cells(
    x: &BinnedMatrix,
    maps: &[Vec<u32>],
    row: usize,
    cells: &mut [u32],
) -> Result<(), PbError> {
    for (axis, ((col, map), cell_slot)) in x.data.iter().zip(maps).zip(cells).enumerate() {
        let bin = *col.get(row).ok_or_else(|| PbError::Internal {
            what: "validated binned row escaped column".into(),
        })?;
        let cell = *map
            .get(usize::from(bin))
            .ok_or_else(|| PbError::InvalidInput {
                what: format!("axis {axis} bin {bin} outside cell map"),
            })?;
        *cell_slot = cell;
    }
    Ok(())
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
    use crate::explain::{fixture_model, fixture_serve, RefMeasure};

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

    #[test]
    fn flat_table_cells_match_table_bank_score_bit_exactly() {
        let model = fixture_model();
        let serve = fixture_serve();
        let bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
        let flat = TableScoringBank::from_bank(&bank).unwrap();
        let mut cells = vec![0_u32; bank.merged_grids.len()];
        for c0 in 0..bank.merged_grids[0].n_bins {
            for c1 in 0..bank.merged_grids[1].n_bins {
                cells[0] = u32::from(c0);
                cells[1] = u32::from(c1);
                assert_eq!(
                    flat.score_cells_row(&cells).unwrap().to_bits(),
                    bank.score(&cells).unwrap().to_bits()
                );
            }
        }
    }

    #[test]
    fn flat_table_binned_path_digitizes_once_and_matches_bank() {
        let model = fixture_model();
        let serve = fixture_serve();
        let bank = model
            .explain(&serve, RefMeasure::ProductMarginals { laplace: 1.0 })
            .unwrap();
        let flat = TableScoringBank::from_bank(&bank).unwrap();
        let mut out = vec![0.0_f64; serve.0.n_rows as usize];
        flat.score_binned(&serve.0, &mut out).unwrap();

        let maps = build_cell_maps(&flat.merged_grids, &serve.0).unwrap();
        let mut cells = vec![0_u32; flat.merged_grids.len()];
        for (row, score) in out.iter().enumerate() {
            fill_row_cells(&serve.0, &maps, row, &mut cells).unwrap();
            assert_eq!(
                score.to_bits(),
                bank.score(&cells).unwrap().to_bits(),
                "row {row}"
            );
        }
    }

    #[test]
    fn flat_table_rejects_grid_that_lost_a_merged_border() {
        let model = fixture_model();
        let serve = fixture_serve();
        let bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
        let flat = TableScoringBank::from_bank(&bank).unwrap();
        let mut bad = serve.0.clone();
        bad.grids[0].borders.clear();
        let mut out = vec![0.0_f64; bad.n_rows as usize];
        assert!(matches!(
            flat.score_binned(&bad, &mut out),
            Err(PbError::InvalidInput { .. })
        ));
    }
}
