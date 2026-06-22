//! Determinism gate (spec §13.4 / §02.10(2), plan F6). The reproducibility engine,
//! pre-wired: a model "trained" at `n_threads ∈ {1, 2, 8}` must serialize to
//! byte-identical bytes. Driven now by a hand-built model whose leaves come from a
//! FIXED-ORDER parallel fold (the §11 pattern: `par_chunks` mapped, partials combined
//! in index order — never a steal-order `reduce`). When §06 lands, this same harness
//! points at the real `fit`; today it proves the *harness* works.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use rayon::prelude::*;
use tri_boost_core::{encode_model, explain::fixture_model, pb_seed, CHUNK_ROWS};

/// A per-leaf statistic computed as a fixed-order fold over a synthetic row set:
/// each chunk is summed in parallel, the partials are collected in index order, then
/// combined sequentially. The result is independent of how many threads ran it.
fn leaf_value(seed: u64, leaf: u32) -> f32 {
    const N_ROWS: u32 = 50_000;
    let rows: Vec<u32> = (0..N_ROWS).collect();
    let partials: Vec<f64> = rows
        .par_chunks(CHUNK_ROWS)
        .map(|chunk| {
            chunk
                .iter()
                .map(|&r| {
                    let s = pb_seed(seed, leaf, 0, r);
                    // Deterministic pseudo-gradient in [-1, 1).
                    (s as f64 / u64::MAX as f64) * 2.0 - 1.0
                })
                .sum::<f64>()
        })
        .collect(); // IndexedParallelIterator::collect preserves chunk order.
    let total: f64 = partials.iter().sum(); // combined in index order
    (total / f64::from(N_ROWS)) as f32
}

/// Build the fixture model with its 8 leaves replaced by fixed-order folds, inside a
/// rayon pool of exactly `n_threads`, and return its frozen-config bincode bytes.
fn model_bytes_in_pool(n_threads: usize, seed: u64) -> Vec<u8> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();
    pool.install(|| {
        let mut model = fixture_model();
        let leaves: [f32; 8] = std::array::from_fn(|k| leaf_value(seed, k as u32));
        let (_, tree) = model.trees.get_mut(0).unwrap();
        tree.leaves = leaves;
        encode_model(&model).unwrap()
    })
}

#[test]
fn model_bytes_are_thread_count_independent() {
    let seed = 0x00C0_FFEE;
    let b1 = model_bytes_in_pool(1, seed);
    let b2 = model_bytes_in_pool(2, seed);
    let b8 = model_bytes_in_pool(8, seed);

    assert!(!b1.is_empty(), "harness produced no bytes");
    assert_eq!(b1, b2, "n_threads 1 vs 2 produced different bytes");
    assert_eq!(b1, b8, "n_threads 1 vs 8 produced different bytes");
}

#[test]
fn pb_seed_drives_a_stable_fold() {
    // The fold is a pure function of the seed regardless of thread count.
    let a = model_bytes_in_pool(8, 123);
    let b = model_bytes_in_pool(1, 123);
    assert_eq!(a, b);
    // A different seed yields different bytes (the fold is not constant).
    assert_ne!(model_bytes_in_pool(1, 123), model_bytes_in_pool(1, 124));
}
