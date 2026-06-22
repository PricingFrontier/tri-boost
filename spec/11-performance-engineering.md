## 11 — Performance engineering & benchmarking

> Owns: memory layout (column-major `u8`, SoA g/h, cache-line padded histograms); the rayon threading model (per-thread private histograms + fixed-order reduction) and GIL release contract; the SIMD strategy (stable autovectorization + cache layout + runtime-dispatched intrinsics via `multiversion`/`pulp`/`wide`; **no nightly `portable_simd` in the shipping core**); the determinism-vs-speed knobs; the bit-reproducibility test harness that realizes the §1 [GATE]; the criterion bench layout and the internal TabArena/EBM yardstick. Uses: `QuantGradHess`/histogram engine and the §06-owned `Hist` accumulator type (§06), the determinism rules (§1), the `Backend` seam (§02), inference scoring (§10). This section spends no budget on the split-finder *math* (§06), purification (§08), or serialization format (§10) — only on making them fast and bit-reproducible.

This section is the concrete speed plan. It serves the three aims directly: **fast** is its remit; **decomposable** is upheld because every optimization here is exactness-neutral (it changes *how* sums are computed, never *what* model results); **predictive** is served indirectly — cheaper trees mean more trees per wall-clock second within the same budget. The honest stance is stated up front and held throughout: **we target training parity with LightGBM/CatBoost `hist`, not a training win, and we bank a structural inference win** from the branch-free 8-cell lookup.

### 11.1 Decisions (defaults)

| Decision | Default | Rationale |
|---|---|---|
| Binned matrix layout | Column-major `Vec<Vec<u8>>` (`BinnedMatrix.data[f]` = column `f`) | Per-feature histogram build is a sequential pass; cache-friendly; rayon-shardable by feature (§2.2). |
| Grad/hess layout | Struct-of-arrays `GradHess { g: Vec<f32>, h: Vec<f32> }` | Separate contiguous slices autovectorize; no AoS padding waste (§2.3). |
| Histogram accumulator | Quantized `i64` (`Hist`, the §06-owned bin accumulator) on the hot path; `f32`/`f64` only for leaf refit | Integer adds are associative ⇒ order-independent ⇒ bit-reproducible *and* ~2× cheaper; `i64` width can never overflow the accumulation bound (§06). |
| Histogram parallelism | Per-thread private, cache-line-padded buffers; **`fold`/`reduce` in index order, never steal order** | No `Mutex` (rayon's #1 perf killer); fixed-order reduce preserves bit-identity across thread counts (§1). |
| Parallel axis | Feature-parallel split search; data-parallel histogram build under `fold` | ≤8 leaves makes the histogram tensor tiny; the dominant cost is the full-data accumulation pass. |
| SIMD on scatter-add | None hand-written; rely on quantized-`i64` autovectorization + `_mm_prefetch` | The `hist[bin[i]] += g[i]` scatter is gather/scatter-bound; XGBoost/LightGBM don't hand-SIMD it either. |
| SIMD on dense kernels | Runtime-dispatched via `multiversion` (grad/hess, prediction, binning, prefix-sum gain scan) | Safe wrappers; per-CPU variants chosen at runtime; no `unsafe` in our code. |
| Wheel ISA baseline | `-C target-cpu=x86-64-v3` (AVX2/FMA/BMI2), AVX-512 lifted at runtime | Portable floor; **never** `target-cpu=native` (non-portable, miscompiles on some toolchains). |
| Determinism mode | `Deterministic` (the only mode); bit-identity is non-optional | The five Invariant gates (§3) and regulated reproducibility both depend on it. |
| Threads | `n_threads` (Rust) / `n_jobs` (Python), `0`/`-1`/`None` ⇒ all cores | Per-call scoped pool (§11.5); never hijack the user's global rayon pool. |

There is **one genuinely-open fork**, deferred to §14: whether to offer a `FastReduce` non-deterministic mode (rayon `reduce` in steal order) for a marginal float-path speedup. **Recommended default: do not ship it.** The quantized-integer path is already order-independent at full speed, so `FastReduce` would buy nothing on the hot path while creating a second, untested code path that can silently break I2. It stays a `// NOTE` in the source, not a feature flag.

### 11.2 Memory layout — the cache contract

```rust
/// One feature's per-leaf gradient/hessian histogram, struct-of-arrays for
/// vectorizable prefix-sum scans. `g_q`/`h_q` are the quantized i64 bin
/// accumulators (the §06-owned `Hist` width; §2.3); the scatter-add target on
/// the hot path. i64 because the accumulation bound n_rows·max|g_q| is proven
/// to stay below i64::MAX (§06), so the sum never overflows under
/// overflow-checks=true.
pub(crate) struct FeatureHist {
    pub g_q: Vec<i64>,   // len = n_bins
    pub h_q: Vec<i64>,   // len = n_bins
    pub cnt: Vec<u32>,   // len = n_bins (min_data_in_leaf, display)
}

/// The level histogram tensor: [leaf][feature] -> FeatureHist. At depth d there
/// are 2^d leaves (<=4 while building level d's split; <=8 leaves total).
/// Flattened to one contiguous arena to keep the whole tensor cache-resident.
pub(crate) struct LevelHists {
    arena: Vec<i64>,             // g_q | h_q interleaved-by-block, cache-line padded per (leaf,feature)
    cnt: Vec<u32>,
    n_leaves: usize,             // 1, 2, or 4
    n_features: usize,
    n_bins: usize,               // <= 255 (bin 0 = missing; max_bin default 254)
    stride_pad: usize,           // pads each (leaf,feature) block to a 64-byte boundary
}
```

The whole level tensor is `n_leaves × n_features × n_bins × (2×i64)` — at depth 3 with `max_bin = 254` (`n_bins ≤ 255`) and 100 features that is `4 × 100 × 255 × 16 B ≈ 1.6 MiB`, L2-resident. **Cache-line padding** (`stride_pad` rounds each per-`(leaf,feature)` block up to 64 bytes) eliminates false sharing when threads accumulate adjacent feature blocks. The binned matrix stays column-major `u8` so each feature's accumulation is a single sequential scan over `data[f]`; row→leaf membership is a `Vec<u8>` of 3-bit leaf indices updated in place after each level's split is fixed (§06).

**Complexity.** Histogram build is `O(3 · n_rows · n_features)` worst case, halved in practice by the subtraction trick (build the smaller child, derive the sibling in `O(n_features · n_bins)`; §06/§02-research). The per-level split search is `O(n_features · n_bins · n_leaves)` — **independent of `n_rows`**, the core histogram win. Memory is `O(n_rows · n_features)` for the binned `u8` matrix plus the tiny level tensor.

### 11.3 Threading model — per-thread private histograms, fixed-order reduce

The histogram accumulation is the only place where parallel float/int reduction order is observable, so it is the linchpin of the §1 bit-reproducibility [GATE]. The contract:

```rust
/// Build the level histogram tensor over `rows`, deterministically and in
/// parallel. Each rayon task owns a PRIVATE zeroed tensor; partials are merged
/// in FIXED CHUNK-INDEX ORDER, never in work-steal order. Chunk boundaries are
/// a function of a FIXED `CHUNK_ROWS` constant (NOT current_num_threads()), so
/// they are identical across pool sizes and the byte-equality gate holds.
pub(crate) fn build_level_hists(
    binned: &BinnedMatrix,
    gh_q: &QuantGradHess,
    leaf_of: &[u8],              // row -> current leaf (3-bit)
    layout: &HistLayout,
) -> LevelHists {
    binned.rows_par_chunks(CHUNK_ROWS)       // fixed-size, index-stable, pool-independent chunks
        .map(|chunk| accumulate_private(chunk, gh_q, leaf_of, layout)) // i64 adds
        .fold(|| LevelHists::zeroed(layout), merge_in_place)
        .reduce(|| LevelHists::zeroed(layout), merge_in_place)
}
```

Two properties make this bit-reproducible across `n_threads ∈ {1, 2, 8}`:

1. **Integer adds are associative**, so `merge_in_place` (per-bin `i64 += i64`) gives the same total regardless of how rayon schedules `fold`/`reduce`. This is *why* the quantized path is the reproducibility mechanism, not merely a speed lever — the same decision serves two aims.
2. **Chunk boundaries are a deterministic function of `n_rows` and the fixed `CHUNK_ROWS` constant** (not of `current_num_threads()`), so they are identical across pool sizes. The *partial* sums differ by chunk, but their integer total is identical, and the final feature-parallel split argmax breaks ties by lowest `(axis, bin_le)` index — never by which thread finished first.

Any unavoidable **float** reduction (leaf refit from full-precision g/h, deviance for early stopping) uses the same pattern: `fold` over fixed-size `par_chunks` derived from the same `CHUNK_ROWS` constant, combine partials in index order with a fixed-arity tree reduction — **never** rayon `reduce`/`sum`, whose steal-order combination is non-associative on `f32`. This is enforced by a clippy-style review rule and the determinism harness (§11.7), not left to discipline.

**Determinism re-seeding.** Any randomized hot-path stage (e.g. stochastic-rounding quantization, MVS/subsample masks) draws from a per-work-unit `Pcg64` that is **re-seeded deterministically**, never from a notional "splittable" PRNG. Each work unit derives its stream by mixing `(base, round, stage, block)` through a frozen `splitmix64` into `Pcg64::seed_from_u64(..)`; because the mix is a pure function of position (not of thread or steal order), the draws are position-stable and thread-count-independent, so they do not perturb the bits (§1/§06).

### 11.4 SIMD strategy

The split is sharp and matches every production GBM:

- **Scatter-add (`hist[bin[i]].g_q += g_q[i]`) is NOT hand-SIMD'd.** It is a data-dependent scatter with same-bin write conflicts and is memory-bound. We win it the way LightGBM/XGBoost/Tangram do: quantized-`i64` accumulation (autovectorizes the *load* and keeps the order-independent reduction), cache-friendly column-major layout, and a software `_mm_prefetch` (via a safe `multiversion` wrapper) on the next row's bin slot.
- **Dense, regular kernels ARE SIMD'd**, via runtime-dispatched safe wrappers — no `unsafe` in our code, honoring `#![forbid(unsafe_code)]` (§1):
  - per-row gradient/hessian (`Loss::grad_hess`, §05, which returns `Result<(), PbError>`): a straight elementwise map;
  - the **prefix-sum split-gain scan** over bins (sequential reads, vectorizes cleanly);
  - **prediction** — the branch-free 8-cell leaf lookup and table-sum scoring (§10), our structural inference win; the leaf-select step commits to an **in-register permute** (see §11.9), not a hardware gather;
  - binning (`f32 → u8` border search) and the `exp(k·F)` inverse-link.

**Leaf-select instruction decision (path A).** The 8 leaves of a depth-3 tree fit one 256-bit register, so the per-row 3-bit leaf index is resolved by an **in-register permute over the 8-entry register-resident leaf LUT** — `vpermps` (AVX2), `vpermi2ps` (AVX-512, a 16-entry table selecting two trees' lanes at once), or `vqtbl` (NEON) — **NOT a hardware gather (`vgatherdps`)**, which is microcoded and load-port-bound. For an 8-entry register-resident table the permute strictly dominates; gather is the wrong tool. This goes through the existing `multiversion`/`pulp`/`wide` safe wrappers (**no raw `unsafe`**, honoring `#![forbid(unsafe_code)]`; §1). It is exactness-neutral (the same `f32` leaf is selected, no reduction reorder) and determinism-safe (cross-row accumulation stays fixed tree order; §11.9). Magnitude: 3–8× on the **leaf-select step only** (not end-to-end) on AVX2; neutral elsewhere. Detailed under §11.9.

```rust
use multiversion::multiversion;

/// Quantize full-precision g/h to the i64 `Hist` accumulators with stochastic
/// rounding (the `Pcg64` is deterministically re-seeded per work unit; §06).
/// Dispatched at runtime to the widest available ISA (SSE2 .. AVX-512).
#[multiversion(targets("x86_64+avx512f", "x86_64+avx2+fma", "x86_64+sse2", "aarch64+neon"))]
pub(crate) fn quantize_gh(g: &[f32], h: &[f32], scale: GradScale, rng: &mut Pcg64, out: &mut QuantGradHess) { /* ... */ }
```

**Crate choices:** `multiversion` for per-CPU function variants + runtime dispatch (lowest friction; owns the dispatch macro on the kernels above); `pulp`/`wide` available for any kernel that needs an explicit portable-vector type. **No `std::simd`/`portable_simd`** in the default build — it is nightly-only with no stabilization date, so it lives strictly behind the optional `nightly` cargo feature (§1) and is never on the path the invariant gates run against. Wheels are built with `-C target-cpu=x86-64-v3` so the baseline binary already uses AVX2/FMA, with AVX-512 selected at runtime by `multiversion`; we never compile with `target-cpu=native`.

**Determinism note:** float SIMD changes *association order within a kernel*, so a SIMD'd float reduction is only allowed where the result feeds a non-reproducibility-critical path or where the kernel reduces in a fixed lane-fold order. The reproducibility-critical accumulation is integer (associative under any order), so SIMD there is always safe. Leaf-refit float reductions use a fixed lane-fold + fixed chunk order so the dispatched ISA does not change the bits — this is asserted in the harness by running the determinism test with AVX-512 forced off vs on.

### 11.5 GIL release & scoped pool (the Python contract)

§12 owns the binding; this section owns the *performance* contract it must honor. All heavy compute runs inside `py.detach` with a **per-call scoped** rayon pool, never the global one:

```rust
// In tri-boost-py (§12), but the perf contract is specified here.
let view = x.as_array();                 // zero-copy ArrayView, GIL held, NOT Send
let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build()?;
let model = py.detach(|| pool.install(|| booster.fit(&binned, &y, &spec)))?;  // GIL released
```

`ArrayView`/`PyReadonlyArray` are derived *before* `detach` (they are not `Send`); the closure is Python-free. This prevents the two failure modes: (1) holding the GIL serializes every other Python thread and can deadlock rayon workers that touch Python; (2) using the global pool oversubscribes under `GridSearchCV(n_jobs=k)` (k outer × n inner threads on n cores). The scoped pool bounds concurrency to exactly `n`.

### 11.6 How it upholds the invariants and serves the aims

- **I1 / I2 are untouched.** Nothing here changes tree shape, feature budget, or table structure — these are arithmetic-ordering and layout decisions. The quantized histograms feed the *split search* only; **leaves are always refit from full-precision g/h** (§06), so the exported tables are exact, not quantized. The five Invariant gates (§3) run against the model these kernels produce and must pass bit-for-bit.
- **Decomposable.** Bit-reproducibility *is* a precondition of the `ThreeWayEqual` gate: tree-sum = table-sum = Shapley-sum can only be asserted bit-equal if the trees are themselves reproducible. The deterministic reduce-order rule of §11.3 is therefore load-bearing for explainability, not just for regulated reproducibility.
- **Fast / honest stance.** Training: **parity, not a win.** Depth-3 obliviousness makes each tree weak, so we expect more trees at a lower learning rate; the quantized-histogram ~2× headroom and the tiny ≤8-leaf cache-resident tensor are what buy parity back. We explicitly do **not** promise to beat LightGBM on training wall-clock, and we do not tune the benchmark to claim otherwise. Inference: a **structural win** — `raw(x) = f0 + offset + Σ alpha_t · tree_t.lookup(x)` is three comparisons → a 3-bit index → one of 8 `f32` reads per tree, fully branch-free and SIMD-friendly (§10), and the table-sum form is even cheaper (a handful of LUT reads independent of tree count).

### 11.7 Hot-loop no-panic policy

Per §1, the deny-gates `indexing_slicing` and `arithmetic_side_effects` are **scoped**, not blanket — the real policy this section's kernels honor is:

- **Integer overflow** is caught by `overflow-checks = true` in *all* cargo profiles (release included), so a width bug surfaces as a typed error path rather than a silent wrap. The `i64` `Hist` width is chosen so the proven accumulation bound `n_rows · max|g_q| < i64::MAX` (§06) means the hot adds never trip the check.
- **Float arithmetic is exempt** from `arithmetic_side_effects` (it cannot overflow-panic); the lint is enabled scoped on the integer accumulation modules, not at the crate root.
- **Indexing.** A hot loop indexes through either the panic-free form `slice.get(i).ok_or(PbError::Internal { what })?` on the cold/setup edges, or a **scoped** `#[allow(clippy::indexing_slicing)]` function carrying a `// JUSTIFIED:` bounds proof (e.g. `bin < n_bins` because bins are produced by the §03 binner against this grid) plus a boundary test that exercises the `bin == n_bins − 1` and missing-bin (`bin == 0`) edges. The scatter-add inner loop uses the second form; its proof is that `leaf_of[row] < n_leaves` and `bin[row] ≤ max_bin < n_bins`, both established at `LevelHists` construction.

### 11.8 Testing & benchmarking

**The bit-reproducibility harness (realizes the §1 [GATE], owned here).**

```rust
#[test]
fn model_is_bit_reproducible_across_thread_counts() -> Result<(), PbError> {
    let cfg = bincode::config::standard();   // FROZEN config; byte-equality depends on it
    let bytes: Vec<Vec<u8>> = [1usize, 2, 8].iter().map(|&n| {
        let m = fit_fixture(seed = 42, n_threads = n)?;
        bincode::serde::encode_to_vec(&m, cfg).map_err(|e| PbError::Serialization(e.to_string()))
    }).collect::<Result<_, _>>()?;
    assert_eq!(bytes[0], bytes[1]);   // byte-identical serialized Model
    assert_eq!(bytes[0], bytes[2]);
    Ok(())
}
```

This trains the same fixture at `n_threads ∈ {1, 2, 8}` and asserts byte-equality of the `bincode`-serialized `Model` — the build-blocking determinism gate. Byte-equality holds across platforms because the serialized index fields are fixed-width (`Split.axis: u32`, `BinnedMatrix.n_rows: u32`; §02/§10), never platform-dependent `usize`, and the `bincode::config::standard()` config is frozen (§10). A companion test forces the SIMD dispatch low (SSE2) vs high (AVX-512, where available) and asserts the same byte-equality, proving the dispatched ISA does not perturb the bits. A third runs under `proptest` over random fixtures (sizes, feature counts, seeds) so reproducibility is a property, not a single golden.

**Microbenchmarks (`criterion`, in `benches/`, dev-only — never shipped in the wheel).** One bench per hot kernel: `build_level_hists` (vs `n_rows`, `n_features`, `n_threads`), `quantize_gh`, the prefix-sum gain scan, and prediction throughput (rows/s). These guard against constant-factor regressions and verify the histogram search is `O(1)` in `n_rows` after binning. Benches live behind the workspace's dev-deps and are excluded from `tri-boost-core`'s published artifact.

**The internal accuracy/speed yardstick (NOT shipped).** Per the AIM and §01, we use **TabArena** with **EBM/GA2M as the headline rival** and unconstrained XGBoost/LightGBM/CatBoost as the accuracy ceiling, purely as a *development* measurement to prove the §01 milestone ("beat EBM, near-parity, all-exact"). It is a script in the repo's dev tooling, run by us; **no benchmarking, model-comparison, or "cost-of-the-cap" tooling ships in the library** (§06 of the brainstorm boundary). Training-speed comparisons here are the evidence base for the honest "parity, not a win" claim — measured, not asserted — and the bagged-vs-bagged comparison against EBM (which itself bags internally) is the fair framing.

### 11.9 Path-A inference micro-architecture — the packed `ScoringBank` and row-blocked streaming kernel

This subsection is the layout + kernel detail behind §10's path-A (tree-walk) scorer. **§10 owns** the `ScoringBank` / `PackedTree` type, its construction in `finalize`/`deserialize`, and the determinism guarantee that the serialized `Model` is untouched; this section specifies the cache-line layout and the streaming kernel that make path A bandwidth-bound rather than latency-bound. Everything here is a **load-derived, scoring-only view** — a byte-exact re-encoding of the same `f32` leaves and the same `u8 ≤ u8` compares, so it is exactness- and determinism-neutral by construction and the §11.8 byte-equality gate (whose subject is the serialized `Model`) is unaffected.

**Why a derived view.** The canonical `ObliviousTree` (§2.5) stores `splits: Vec<Split>` + `leaves: [f32;8]` — a heap indirection per tree, and `Split { axis: u32, bin_le: u8, missing_left: bool }` padding makes a tree straddle two cache lines with a dependent load to reach the leaves. The hot scorer should instead stream a contiguous, unit-stride, one-cache-line-per-tree array that the hardware prefetcher walks linearly.

**The packed layout (owned/registered under §10; reference §2.5 for the source `f32` leaves and compares).**

```rust
/// Scoring-only path-A view of one depth-3 oblivious tree, packed to one
/// 64-byte cache line. A byte-exact re-encoding of `ObliviousTree.leaves`
/// (same f32) and its level compares (same `bin <= thresh` tests); NOT a
/// second source of truth. Built once in finalize/deserialize; the serialized
/// `Model` (the §11.8 determinism gate's subject) is never touched. Owned by §10.
#[repr(C)]
pub(crate) struct PackedTree {
    pub feat:   [u8; 3],   // per-level axis (u8 ⇒ HOT PATH CAPS AT <=255 FEATURES)
    pub thresh: [u8; 3],   // per-level `bin_le` border
    pub miss:   u8,        // per-level missing-left bits (the §2.5 `missing_left` carrier, packed)
    pub leaf:   [f32; 8],  // index = b0 | b1<<1 | b2<<2; the register-resident LUT for the §11.4 permute
    // padded by `#[repr(C)]` alignment to a full 64-byte cache line
}

/// The contiguous path-A bank, in STORED tree order (the §10 serialization
/// order), so the scorer's reduction order is fixed and bit-reproducible.
pub(crate) struct ScoringBank(pub Vec<PackedTree>);
```

**Feature-width cap.** The `u8` `feat` axis caps the hot path at **≤255 features**; wider models fall back to a **side-table** path (a `u32` axis variant of the kernel) — the packed bank is the fast path, never the only path. This is the single caveat of the layout and is checked at `ScoringBank` construction.

**The row-blocked streaming kernel.** Scoring one row at a time re-reads the entire tree bank per row — `O(M)` tree-bank bytes per row, latency-bound when `M` spills L2. Instead hold a **tile of `W ≈ 8–32` rows' `u8` feature columns resident**, stream the **entire `ScoringBank` exactly once**, and for each `PackedTree` resolve all `W` rows' 3-bit leaf indices (the in-register permute of §11.4 over the register-resident `leaf` LUT) and accumulate `W` `f32` partial scores:

```rust
/// Score a tile of W rows against the whole packed bank in one streaming pass.
/// Per-row tree-bank bandwidth drops by a factor of W (the bank is read once for
/// the whole tile, not once per row). Accumulation is in FIXED stored tree order
/// for every lane, so it is bit-identical to the one-row-at-a-time path and to
/// the canonical `tree_lookup` scorer — determinism is preserved. Dispatched via
/// `multiversion` (no raw `unsafe`; §1). Realizes the `Backend::predict_block`
/// seam (§02).
fn score_tile(bank: &ScoringBank, tile: &RowTile<W>, out: &mut [f32; W]) { /* ... */ }
```

Two properties make this safe and worthwhile:

1. **Bandwidth.** The bank is streamed once per tile of `W` rows instead of once per row, so **per-row tree-bank bandwidth drops by a factor of `W`**. `_mm_prefetch` (safe `multiversion` wrapper) on the next cache line keeps the prefetcher ahead of the stream.
2. **Determinism.** Each lane accumulates in **fixed stored tree order**, identical to the canonical per-row `tree_lookup` (§10), so the tiled result is bit-for-bit equal to the scalar path. No cross-row or cross-tree reduction reorder is introduced. This is exercised by the §11.8 harness (tiled vs scalar byte-equality of predictions) and the §10 path-equality gate.

**Magnitude — honest and conditional.** The packed bank + tiled stream is a **2–4× path-A win only on cold-cache / large-`M` / L2-spilling ensembles**, where it converts memory-latency-bound scoring into L2 streaming. **When the ensemble is L2-resident it is roughly neutral** — the canonical scorer already hits cache. The leaf-select permute (§11.4) is a separate 3–8× on the leaf-select *step only*, not end-to-end. None of these is a training win; path B (table-sum, §10.3) remains the tree-count-independent scorer and is untouched here.
