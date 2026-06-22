//! Criterion bench skeleton (spec §11, plan F4). Phase-0 placeholder: it benches the
//! frozen `pb_seed` mixer so `cargo bench --no-run` has a real target. The histogram /
//! split-finder / scoring benches (the bit-reproducibility yardstick) land with §11.

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
use tri_boost_core::pb_seed;

fn bench_pb_seed(c: &mut Criterion) {
    c.bench_function("pb_seed", |b| {
        b.iter(|| pb_seed(black_box(1), black_box(2), black_box(3), black_box(4)));
    });
}

criterion_group!(benches, bench_pb_seed);
criterion_main!(benches);
