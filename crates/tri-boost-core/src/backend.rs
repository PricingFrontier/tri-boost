//! The compute seam — the `Backend` trait + `CpuBackend` (spec §02.5, OWNED HERE) —
//! and the frozen deterministic re-seeding mixer (`pb_seed`, §02.3b).
//!
//! A backend owns the four kernels that dominate train/inference time; everything
//! else (boosting loop, split argmax, purification, serde) stays
//! backend-independent. Crucially, the trait exposes NO leaf-value method: leaves
//! are always refit from full-precision `GradHess` on the host, so a backend can
//! never bend I1/I2 — that is enforced by *where code is allowed to live*.

use crate::data::BinnedMatrix;
use crate::engine::{Hist, Model, QuantGradHess, Split};
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
/// Reproducibility is a property of the impl: `build_histograms` accumulates with
/// associative `i64` integer sums (or a fixed-order float fold on the cross-check
/// path), never a steal-order rayon `reduce`. A `CpuBackend` built with different
/// `n_threads` MUST produce byte-identical `Hist`s and predictions.
// The seam is deliberately defined ahead of its first consumer: the boosting loop
// (§06, phase P1) is what calls these kernels. `CpuBackend` already implements the
// whole trait, so the contract is frozen now; `dead_code` is expected until P1.
#[allow(dead_code)]
pub(crate) trait Backend: Send + Sync {
    /// Build the integer g/h histogram for one level: per-(axis, bin) `i64` sums
    /// into `Hist` (the single §06-owned accumulator; no `HistogramSet` alias).
    ///
    /// # Errors
    /// [`PbError`] on shape mismatch or an unimplemented backend (Phase 0).
    fn build_histograms(
        &self,
        x: &BinnedMatrix,
        gh: &QuantGradHess,
        rows: &[u32],
        hist: &mut Hist,
    ) -> Result<(), PbError>;

    /// Evaluate the oblivious level-wise summed Newton gain for every candidate
    /// `(axis, bin_le)` and return the single argmax split for the whole level.
    ///
    /// # Errors
    /// [`PbError`] on an unimplemented backend (Phase 0).
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
    /// Propagates [`Loss::grad_hess`] failures; [`PbError`] (Phase 0 stub).
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
    /// [`PbError`] on shape mismatch or an unimplemented backend (Phase 0).
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
/// Phase-0 stub: the kernels are not yet implemented.
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
        gh: &QuantGradHess,
        rows: &[u32],
        hist: &mut Hist,
    ) -> Result<(), PbError> {
        let _ = (x, gh, rows, hist);
        Err(PbError::Internal {
            what: "CpuBackend::build_histograms is not implemented in Phase 0 (§06)".into(),
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
            what: "CpuBackend::best_level_split is not implemented in Phase 0 (§06)".into(),
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
        let _ = (loss, y, raw, weight, out);
        Err(PbError::Internal {
            what: "CpuBackend::grad_hess is not implemented in Phase 0 (§06)".into(),
        })
    }

    fn predict_block(
        &self,
        model: &Model,
        x: &BinnedMatrix,
        rows: &[u32],
        out: &mut [f32],
    ) -> Result<(), PbError> {
        let _ = (model, x, rows, out);
        Err(PbError::Internal {
            what: "CpuBackend::predict_block is not implemented in Phase 0 (§10/§11)".into(),
        })
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
    fn cpu_backend_stub_errs_not_panics() {
        let be = CpuBackend::new(1);
        let mut out = vec![0.0_f32; 0];
        let m = crate::explain::fixture_model();
        let x = BinnedMatrix {
            data: vec![],
            n_rows: 0,
            grids: vec![],
            provenance: vec![],
        };
        // The point: a not-yet-implemented kernel returns Err, never panics.
        assert!(be.predict_block(&m, &x, &[], &mut out).is_err());
    }
}
