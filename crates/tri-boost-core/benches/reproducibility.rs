//! Criterion benches for the deterministic runtime kernels (spec §11 / M5-T10).

// The `criterion_group!`/`criterion_main!` macros expand to undocumented public
// items + a `main`; benches are dev-only (not shipped), so the crate-wide
// `deny(missing_docs)` and no-panic set are relaxed for this target.
#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tri_boost_core::explain::{fixture_model, fixture_serve, RefMeasure};
use tri_boost_core::{pb_seed, score_tile, ScoringBank, TableScoringBank};

fn bench_pb_seed(c: &mut Criterion) {
    c.bench_function("pb_seed", |b| {
        b.iter(|| pb_seed(black_box(1), black_box(2), black_box(3), black_box(4)));
    });
}

fn bench_scoring(c: &mut Criterion) {
    let model = fixture_model();
    let serve = fixture_serve();
    let rows: Vec<u32> = (0..serve.0.n_rows).collect();
    let packed = ScoringBank::from_model(&model).unwrap();
    let bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
    let table = TableScoringBank::from_bank(&bank).unwrap();
    let mut raw = vec![0.0_f32; serve.0.n_rows as usize];
    let mut table_raw = vec![0.0_f64; serve.0.n_rows as usize];

    c.bench_function("score_trees_batch", |b| {
        b.iter(|| {
            model
                .score_trees(black_box(&serve.0), None, black_box(&mut raw))
                .unwrap()
        });
    });
    c.bench_function("scoring_bank_tile", |b| {
        b.iter(|| {
            score_tile(
                black_box(&packed),
                black_box(&serve.0),
                black_box(&rows),
                None,
                black_box(&mut raw),
            )
            .unwrap()
        });
    });
    c.bench_function("table_scoring_bank", |b| {
        b.iter(|| {
            table
                .score_binned(black_box(&serve.0), black_box(&mut table_raw))
                .unwrap()
        });
    });
    c.bench_function("model_bincode_roundtrip", |b| {
        b.iter(|| {
            let bytes = model.to_bincode().unwrap();
            tri_boost_core::Model::from_bincode(black_box(&bytes)).unwrap()
        });
    });
}

criterion_group!(benches, bench_pb_seed, bench_scoring);
criterion_main!(benches);
