//! The compute seam — the `Backend` trait + `CpuBackend` (spec §02.5, OWNED HERE) —
//! and the frozen deterministic re-seeding mixer (`pb_seed`, §02.3b).
//!
//! A backend owns the four kernels that dominate train/inference time; everything
//! else (boosting loop, split argmax, purification, serde) stays
//! backend-independent. Crucially, the trait exposes NO leaf-value method: leaves
//! are always refit from full-precision `GradHess` on the host, so a backend can
//! never bend I1/I2 — that is enforced by *where code is allowed to live*.

use crate::data::BinnedMatrix;
use crate::engine::{Hist, Model, Split};
use crate::error::PbError;
use crate::loss::{GradHess, Loss};
use rand::SeedableRng;
use rand_pcg::Pcg64;

/// Per-level constraint context passed to the split-finder (spec §06/§07). Phase-0
/// placeholder registered here so the `Backend` trait compiles; the monotone /
/// interaction / credibility fields land with §06/§07.
#[derive(Debug, Clone, Default)]
pub struct LevelConstraints {}

/// The compute seam (spec §02.5). v1 ships only [`CpuBackend`]. A backend MUST be
/// bit-reproducible: identical inputs ⇒ identical outputs, independent of internal
/// thread count (the §1 determinism `[GATE]`). `pub(crate)` — an internal seam, not
/// a public API contract in v1.
///
/// Reproducibility is a property of the impl: in v1 `build_histograms` accumulates
/// full-precision `GradHess` with a FIXED-ORDER float fold (feature-parallel,
/// sequential within each axis — `engine::hist`), never a steal-order rayon
/// `reduce`. A `CpuBackend` built with different `n_threads` MUST produce
/// byte-identical `Hist`s and predictions. (The associative `i64`-quantized
/// accumulation is the M5-QHIST, v1.5, alternative.)
// The seam is deliberately defined ahead of its first consumer: the boosting loop
// (§06, phase P1) is what calls these kernels. `CpuBackend` already implements the
// whole trait, so the contract is frozen now; `dead_code` is expected until P1.
#[allow(dead_code)]
pub(crate) trait Backend: Send + Sync {
    /// Build the per-level g/h histogram: full-precision `f64` sums per
    /// `(leaf, axis, bin)` into `Hist` (the single §06-owned accumulator; the v1
    /// green spine accumulates `GradHess` directly — the quantized `QuantGradHess`
    /// input is the M5-QHIST/v1.5 variant).
    ///
    /// # Errors
    /// [`PbError`] on shape mismatch or because this legacy seam lacks the §06
    /// leaf-assignment and axis-subset context used by the implemented engine.
    fn build_histograms(
        &self,
        x: &BinnedMatrix,
        gh: &GradHess,
        rows: &[u32],
        hist: &mut Hist,
    ) -> Result<(), PbError>;

    /// Evaluate the oblivious level-wise summed Newton gain for every candidate
    /// `(axis, bin_le)` and return the single argmax split for the whole level.
    ///
    /// # Errors
    /// [`PbError`] because this legacy seam lacks the §06 axis metadata and
    /// constraint context used by the implemented split finder.
    fn best_level_split(
        &self,
        hist: &Hist,
        lambda: f32,
        constraints: &LevelConstraints,
    ) -> Result<Option<Split>, PbError>;

    /// Accumulate full-precision per-row `(g, h)` for the current raw scores.
    /// `Loss::grad_hess` is itself fallible, so this kernel propagates with `?` —
    /// never `.expect`.
    ///
    /// # Errors
    /// Propagates [`Loss::grad_hess`] failures.
    fn grad_hess(
        &self,
        loss: &dyn Loss,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError>;

    /// Branch-free 8-cell leaf lookup + table-sum scoring for a row block.
    ///
    /// # Errors
    /// [`PbError`] on shape mismatch or row-index overflow.
    fn predict_block(
        &self,
        model: &Model,
        x: &BinnedMatrix,
        rows: &[u32],
        out: &mut [f32],
    ) -> Result<(), PbError>;
}

/// The v1 backend: rayon per-thread padded `Hist`s + fixed-order reduce,
/// multiversion-dispatched dense kernels (spec §02.5). The only `Backend` in v1.
/// The row-local kernels delegate to the canonical core path; histogram/split methods
/// remain fail-closed until the backend trait is reconciled with the implemented §06
/// leaf-assignment/axis-aware engine.
#[derive(Debug, Clone)]
pub struct CpuBackend {
    /// Worker-thread count. The output MUST be byte-identical across values of this
    /// field — that is exactly the determinism contract the §13.4 gate enforces.
    pub n_threads: usize,
}

impl CpuBackend {
    /// A CPU backend with `n_threads` workers.
    #[must_use]
    pub fn new(n_threads: usize) -> Self {
        Self { n_threads }
    }
}

impl Backend for CpuBackend {
    fn build_histograms(
        &self,
        x: &BinnedMatrix,
        gh: &GradHess,
        rows: &[u32],
        hist: &mut Hist,
    ) -> Result<(), PbError> {
        let _ = (x, gh, rows, hist);
        Err(PbError::Internal {
            what: "CpuBackend::build_histograms legacy seam lacks §06 leaf assignments/axis list; use engine::hist/split".into(),
        })
    }

    fn best_level_split(
        &self,
        hist: &Hist,
        lambda: f32,
        constraints: &LevelConstraints,
    ) -> Result<Option<Split>, PbError> {
        let _ = (hist, lambda, constraints);
        Err(PbError::Internal {
            what: "CpuBackend::best_level_split legacy seam lacks §06 axis metadata/constraints; use engine::split".into(),
        })
    }

    fn grad_hess(
        &self,
        loss: &dyn Loss,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        loss.grad_hess(y, raw, weight, out)
    }

    fn predict_block(
        &self,
        model: &Model,
        x: &BinnedMatrix,
        rows: &[u32],
        out: &mut [f32],
    ) -> Result<(), PbError> {
        if out.len() != rows.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "predict_block out len {} != rows len {}",
                    out.len(),
                    rows.len()
                ),
            });
        }
        let mut full = vec![0.0_f32; x.n_rows as usize];
        model.score_trees(x, None, &mut full)?;
        for (dst, &row) in out.iter_mut().zip(rows) {
            let score = *full
                .get(row as usize)
                .ok_or_else(|| PbError::InvalidInput {
                    what: format!("predict_block row {row} outside n_rows {}", x.n_rows),
                })?;
            *dst = score;
        }
        Ok(())
    }
}

/// Frozen deterministic re-seeding (spec §02.3b). `splitmix64` is the standard
/// 64-bit mixer; this exact function is part of the determinism `[GATE]` contract
/// and MUST NOT change without a schema/repro bump.
///
/// The per-`(round, stage, block)` stream is a pure function of the base seed and
/// the work-unit coordinates, so draws are position-stable and **independent of
/// thread count**. Downstream: `Pcg64::seed_from_u64(pb_seed(base, round, stage,
/// block))`. The `wrapping_mul`/`>>`/`^` here are the documented exception to the
/// integer-overflow trap — wrapping is intentional in the mixer.
#[must_use]
pub fn pb_seed(base: u64, round: u32, stage: u32, block: u32) -> u64 {
    let mut z = base
        ^ u64::from(round).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ u64::from(stage).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ u64::from(block).wrapping_mul(0x94D0_49BB_1331_11EB);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The randomized stage a re-seed belongs to — the `stage` coordinate of
/// [`pb_seed`] (spec §1/§02.3b). The discriminants are **frozen** (part of the
/// determinism `[GATE]` contract): never renumber an existing stage, only append.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Per-feature binning subsample (§03.3).
    Binning = 1,
    /// Row/feature subsampling & MVS (§06.7).
    Sample = 2,
    /// Stochastic rounding for quantized histograms (§06/§11, v1.5).
    Quantize = 3,
    /// Bagged ensemble selection (§09.6).
    Bagging = 4,
    /// Categorical target-statistic permutations/folds (§04.3).
    Categorical = 5,
    /// Split-score random-strength tie/regularization noise (§09.6).
    SplitNoise = 6,
    /// DART tree-dropout masks (§09.6).
    Dart = 7,
}

/// Construct the per-work-unit [`Pcg64`] from the frozen [`pb_seed`] mixer
/// (spec §02.3b). The single canonical way the library obtains a stream, so every
/// randomized stage is position-stable and thread-count-independent.
#[must_use]
pub fn pb_rng(base: u64, round: u32, stage: Stage, block: u32) -> Pcg64 {
    Pcg64::seed_from_u64(pb_seed(base, round, stage as u32, block))
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

    /// Known-vector test pinning the frozen mixer so the determinism RNG can never
    /// silently drift. The all-zero input is provable by hand (`z` stays `0` through
    /// every step), so it anchors the implementation; the remaining vectors are
    /// frozen outputs of THIS `splitmix64` — regenerate ONLY with a documented
    /// schema/repro bump.
    #[test]
    fn pb_seed_is_frozen() {
        // Pure function of the coordinates: same inputs ⇒ same output, always.
        assert_eq!(pb_seed(7, 3, 2, 9), pb_seed(7, 3, 2, 9));

        // All-zero input mixes to zero (0 ⊕ 0 = 0; every multiply is of 0).
        assert_eq!(pb_seed(0, 0, 0, 0), 0);

        // Distinct coordinates yield distinct streams (no trivial collisions).
        assert_ne!(pb_seed(1, 0, 0, 0), pb_seed(0, 0, 0, 0));
        assert_ne!(pb_seed(0, 1, 0, 0), pb_seed(0, 0, 0, 0));
        assert_ne!(pb_seed(0, 0, 1, 0), pb_seed(0, 0, 0, 0));
        assert_ne!(pb_seed(0, 0, 0, 1), pb_seed(0, 0, 0, 0));

        // Frozen reference vectors (outputs of this exact mixer).
        assert_eq!(pb_seed(1, 0, 0, 0), 6_238_072_747_940_578_789);
        assert_eq!(pb_seed(0, 1, 0, 0), 16_294_208_416_658_607_535);
        assert_eq!(pb_seed(42, 1, 2, 3), 1_962_896_480_199_194_022);
    }

    #[test]
    fn cpu_backend_delegates_row_kernels_and_fail_closes_unreconciled_methods() {
        let be = CpuBackend::new(1);
        let m = crate::explain::fixture_model();
        let x = crate::explain::fixture_serve().0;
        let mut out = vec![0.0_f32; 2];
        be.predict_block(&m, &x, &[0, 3], &mut out).unwrap();
        let mut full = vec![0.0_f32; x.n_rows as usize];
        m.score_trees(&x, None, &mut full).unwrap();
        assert_eq!(out, vec![full[0], full[3]]);

        let loss = crate::loss::SquaredError;
        let mut gh = crate::loss::GradHess::default();
        be.grad_hess(&loss, &[1.0, 2.0], &[1.5, 1.0], &[1.0, 2.0], &mut gh)
            .unwrap();
        assert_eq!(gh.g, vec![0.5, -2.0]);
        assert_eq!(gh.h, vec![1.0, 2.0]);

        let mut hist = Hist::try_zeros(1, 1, 2).unwrap();
        assert!(be.build_histograms(&x, &gh, &[0, 1], &mut hist).is_err());
        assert!(be
            .best_level_split(&hist, 1.0, &LevelConstraints::default())
            .is_err());
    }
}
