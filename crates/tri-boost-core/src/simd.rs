//! Performance engineering kernels (spec §11). Phase-0 placeholder: the
//! `multiversion`/`pulp` dense SIMD kernels (all safe wrappers — the core stays
//! `#![forbid(unsafe_code)]`) land with §11. Only the fixed-order-fold chunk size,
//! which is part of the determinism contract, is fixed now.

/// The fixed chunk size for deterministic, order-independent float folds (spec §11 /
/// §02.5). Reductions `fold` over `par_chunks(CHUNK_ROWS)` combined in index order —
/// never a steal-order rayon `reduce` — so results are byte-identical regardless of
/// thread count. The value is frozen as part of the reproducibility `[GATE]`.
pub const CHUNK_ROWS: usize = 4096;
