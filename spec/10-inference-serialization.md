## 10 — Inference & serialization

> Owns (per §4): `Model` scoring (branch-free 8-cell lookup + table-sum equivalence); the serde format (`serde_json` canonical, `bincode` fast cache, never pickle); `schema_version` round-trip + migration; the `TableBank` rating-table export format. Owns and registers (skeleton §4): the export/wire types `ModelDoc`, `RatingExport`, `RatingTable`, `AxisExport`, `RatingBasis` (the optional reference-level rating-view re-basing selector); the load-derived path-A scoring view `ScoringBank` / `PackedTree` (one cache line per tree; built in finalize/deserialize; never serialized — its layout + kernel detailed in §11). Uses, does not redefine: `Model`, `ModelSchema` / `CatEncoderStore` / `ObjectiveTag` (§2.6 / §04, for export readability + round-trip), `ObliviousTree`, `Split` (incl. `missing_left`, honored at scoring), `TableBank`, `EffectTable`, `BorderGrid`, `RefMeasure`, `PbError`, `ExactnessMode` from §2/§3.

This section is the "two views, one number" boundary made concrete: a trained `Model` scores two provably-equal ways — walking its trees, or summing its purified `TableBank` — and both serialize to a stable, versioned, human-readable format. Scoring serves *fast* (branch-free, table-sum cost independent of tree count); the export format serves *decomposable* (the tables ARE the rating structure); the byte format serves *robustness* (one serde impl in the core crate, identical in Rust and Python). I1/I2 are not re-litigated here — they are *consumed*: scoring assumes ≤3-feature trees and a shared grid; export refuses to emit `Exact` tables unless §08's five checks pass.

### 10.1 Decisions (defaults locked)

- **Two scoring paths, proven equal.** `score_trees` (path A, default for `predict`) and `score_tables` (path B, default once a `TableBank` is materialized for explanation). Equality is a build-blocking property test (§10.7), not a comment.
- **f32 scoring, no `powf`.** Leaf values and the score `raw` are `f32`; the link inverse uses `exp(k·F)` (§05), never `powf`. f32 is mandatory for bit-exactness against the stored leaves, and the reconstruction tolerance is an f32 tolerance. **The purified `Tensor` cells are `f64`, not f32** (§08 owns purification and accumulates/stores the tables in f64 to keep mass-conservation tight); path B reads those f64 cells and the per-row sum lands within the derived f32 reconstruction tolerance (§10.4/§13) of the f32 tree-sum. So: leaves and scores are f32, the purified `Tensor` is f64.
- **Branch-free per-row eval.** Each tree contributes via 3 comparisons → a 3-bit index → one of 8 leaf reads. No data-dependent branch; the 8 leaves live in-register (`[f32; 8]`) and the index selects without a jump. SIMD is *across rows* (multiversion-dispatched), never across the 3 compares of one tree.
- **No-panic hot loops.** The scoring kernels are on the no-panic gate (§1): every index either goes through `slice.get(..).ok_or(PbError::Internal{..})?` or sits inside a `#[allow(clippy::indexing_slicing)]`-scoped `fn` carrying a `// JUSTIFIED:` bounds proof plus a boundary test (§10.7). Float arithmetic is exempt from `arithmetic_side_effects`; integer index arithmetic (the `bit << level` index build) is bounded `< 8` by construction. See §10.2 for the policy in code.
- **Canonical on-disk format = stable JSON** (`serde_json`, pretty by default, self-describing, the public cross-language contract à la XGBoost). **`bincode` is an opt-in same-version cache only**, never the archival format. Pickle is forbidden for the native model — the Python layer (§12) exposes `save_json`/`load_json`, not `__reduce__` of the Rust object. The `bincode` path is **bincode 2.x** via `bincode::serde::encode_to_vec` / `decode_from_slice` against a frozen `bincode::config::standard()` (the removed top-level `bincode::serialize`/`deserialize` are not used); the frozen config is what makes the binary bytes reproducible.
- **Single `format_version: u32`** is the wire contract version (distinct from `Model.schema_version`, the struct-shape version; see §10.4). Migration is forward-only, `From`-based, gated.
- **The rating-table export is a distinct artifact**, `RatingExport`, derived from a `TableBank`. In its canonical PURE form it is the exact fANOVA decomposition: the purified tables, each **zero-mean under `w`** (Purity, §08), rendered as relativities `exp(f_u)` on a log link, every table Sobol-ranked and `w`-stamped. It is *not* the inference object; it is the human/filing view, and it carries the `ExactnessMode` so an `Approximate` model cannot masquerade as a filed rating table. **The PURE export does NOT claim a 1.000 reference row per axis** and an all-baseline row does **not** predict `base` — that is a property of the *rating-view re-based* form only (below), not of the zero-mean decomposition.
- **Open fork (one):** whether `predict` auto-switches to path B when a `TableBank` is cached on the `Model`. Recommended default: **no** — `predict` always uses path A (tree walk) for determinism and to keep `Model` independent of whether explanation was requested; path B is reached only via the explicit `score_tables`/`TableBank` API. Revisit if profiling shows table-sum dominates on very deep ensembles.

### 10.2 Path A — branch-free oblivious tree evaluation

A `Split { axis, bin_le, missing_left }` tests `binned[axis][row] <= bin_le` — **except** the reserved missing bin (bin 0), which is routed by the *learned* default direction `missing_left` (§2.5/§03/§06), NOT silently sent left. The canonical per-level low/left bit (identical in §06 split evaluation/sample→leaf update, §08 accumulation, and both kernels below) is:

```rust
let low = if bin == 0 { split.missing_left } else { bin <= split.bin_le };
// the leaf-index bit at this level = low as usize
```

(`axis: u32`, fixed-width for cross-platform byte-equality — see §10.5). For a tree of `depth` levels the row's leaf index is the little-endian bit pattern of its `depth` low bits; unused high bits are 0, so early-terminated trees (depth < 3, a legal lower-order fANOVA outcome under I1) index the low `2^depth` leaves and the unused leaf tail stays `0.0`. Honoring `missing_left` here is **required for tree/table equality (I2)**: routing missing the same learned way in scoring and in §08 accumulation is what keeps `score_trees_row` and `table_bank.score_row` equal; a bare `bin <= bin_le` would always route missing left and break Reconstruction/ThreeWayEqual.

```rust
/// Score one already-binned row through one tree. Branch-free: the 3 tests
/// build a 3-bit index; the leaf is a direct array read.
///
/// Hot loop, no-panic policy (§1): the `axis` cast and the two indexed reads
/// are bounds-proven below, so this fn is the scoped-`allow` form. The
/// alternative `row.get(ax).ok_or(PbError::Internal{..})?` form is used by the
/// fallible batch entry points; here the proof is total, so the read is direct.
#[allow(clippy::indexing_slicing)]
#[inline]
fn tree_lookup(tree: &ObliviousTree, row: &[u8]) -> f32 {
    // JUSTIFIED bounds proof:
    //  * `split.axis < n_features == row.len()`: validated at ObliviousTree
    //    construction (I1, §06) AND re-checked on deserialize (§10.5); the
    //    batch entry point asserts `row.len() == self.grids.len()` up front.
    //  * `idx < 2^depth <= 8 == tree.leaves.len()`: `level` ranges 0..depth<=3,
    //    so `idx` accumulates at most bits 0,1,2 ⇒ idx in 0..=7.
    // Both reads are therefore in-bounds; a boundary test (§10.7) pins idx==7
    // and the max-axis row. Index arithmetic (`bit << level`) cannot overflow.
    let mut idx = 0usize;
    for (level, split) in tree.splits.iter().enumerate() {
        let bin = row[split.axis as usize];
        // Canonical low/left bit: missing (bin 0) follows the LEARNED default
        // direction; all other bins use the threshold test. (R-MISSING; §2.5.)
        let low = if bin == 0 { split.missing_left } else { bin <= split.bin_le };
        idx |= (low as usize) << level;
    }
    tree.leaves[idx]
}
```

The public row/batch entry points take the binned design (binning of raw f32 → u8 is owned by §03 and happens once at ingest; `predict` on raw input binds through the stored grids first). They are fallible and use the `get(..).ok_or(PbError::Internal)?` form on the per-row boundary so a malformed argument is a typed error, never a panic:

```rust
impl Model {
    /// Raw score for one binned row: f0 + offset + Σ_t alpha_t · tree_t(row).
    /// Errors `PbError::ShapeMismatch` if `row.len() != self.grids.len()`.
    pub fn score_trees_row(&self, row: &[u8], offset: f32) -> Result<f32, PbError> {
        if row.len() != self.grids.len() {
            return Err(PbError::ShapeMismatch { what: "row width != grid count".into() });
        }
        let mut acc = self.f0 + offset;
        for (alpha, tree) in &self.trees {
            acc += alpha * tree_lookup(tree, row); // row width proven == grids ⊇ axes
        }
        Ok(acc)
    }

    /// Batch raw scores over a column-major BinnedMatrix into `out` (len = n_rows).
    /// SIMD-across-rows via multiversion; deterministic (fixed tree order, f32).
    pub fn score_trees(
        &self,
        x: &BinnedMatrix,
        offset: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<(), PbError>;

    /// Response-space predictions = loss.pred_from_raw(raw) applied to score_trees.
    pub fn predict(&self, x: &BinnedMatrix, offset: Option<&[f32]>)
        -> Result<Vec<f32>, PbError>;

    /// Score an already-`BinnedMatrix` design directly — no re-binning. The
    /// raw `predict` path digitizes raw f32 → u8 per call; callers scoring the
    /// same already-binned design repeatedly (or who binned at ingest, §03)
    /// take this to amortize binning to ONCE: O(n_rows·n_features) digitize,
    /// not O(n_rows·n_trees·n_features). Numerically identical to `predict` on
    /// the same bins (it is the same kernel), so it is under the path-equality
    /// and reproducibility gates (§10.7), not a new contract.
    pub fn predict_binned(&self, x: &BinnedMatrix, offset: Option<&[f32]>)
        -> Result<Vec<f32>, PbError>;
}
```

**Note — digitize-once amortization.** Binning is hoisted out of the per-tree loop: `predict_binned(&BinnedMatrix)` makes the "bins are computed once" path explicit, and path B's flattened kernel (§10.3) digitizes each merged axis once per row rather than once per table. Across both paths the rule is the same — pay O(n_rows · n_features) to digitize, never O(n_rows · n_trees · n_features) — and because only *where* digitization happens changes (not the resulting bins), every such path stays bit-identical to the reference scorer under the §10.7 path-equality gate.

`score_trees` checks `x.n_features() == self.grids.len()` and `out.len() == x.n_rows` up front (→ `PbError::ShapeMismatch`), then iterates trees in stored order and rows in index order — both fixed, so the float sum is reduction-order-stable and bit-reproducible regardless of thread count (per §1). The `alpha` weights carry DART/Nesterov/ensemble mixing (§09) transparently; scoring never inspects how they were produced. **Complexity:** O(n_rows · n_trees · depth) compares, O(n_rows · n_trees) leaf reads — the per-tree cost is constant (≤3 compares + 1 read), so it is exactly the CatBoost-class fast path. SIMD widens the row loop: load `W` rows' `row[axis]` lanes, derive each lane's low bit as `if bin == 0 { missing_left } else { bin <= bin_le }` (a splat compare against `bin_le`, with the `bin == 0` lanes blended to the splat `missing_left` mask — R-MISSING, identical to the scalar form), build `W` 3-bit indices, and **select the leaf with an in-register permute** (next paragraph). Overflow-checks are on in all profiles (§1); the only integer arithmetic in the kernel is the bounded index build, and float accumulation is overflow-exempt.

**Leaf-select is an in-register permute, not a hardware gather (decision locked).** With one tree's 8 leaves resident in a single 256-bit register, the per-row 3-bit index selecting one of 8 lanes is an in-register permute over the 8-entry register-resident leaf LUT — `vpermps` on AVX2 (≈1 cycle), `vpermi2ps` on AVX-512 (a 16-entry table, i.e. two trees' leaf LUTs at once), `vqtbl` on NEON. This deliberately replaces any "leaf gather" reading: a hardware gather (`vgatherdps`) over the leaf array in memory is microcoded and load-port-bound (~5–20 cycles) and is the wrong tool for an 8-entry register-resident LUT, where permute strictly dominates. It is exactness- and determinism-neutral — the same f32 lane is selected, no reduction is reordered, and cross-row accumulation stays in fixed tree order — and it goes through the existing `multiversion`/`pulp`/`wide` safe wrappers (§11.4), so the `#![forbid(unsafe_code)]` policy (§1) holds. The permute targets the packed leaf LUT of the `ScoringBank` (§10.2a).

### 10.2a Path-A scoring view — the packed `ScoringBank` / `PackedTree` (owned here)

The in-memory `ObliviousTree` stores `splits: Vec<Split>` (a heap pointer per tree) + `leaves: [f32; 8]`, and `Split { axis: u32, bin_le: u8, missing_left: bool }` pads such that a tree spans two cache lines with a dependent load to reach the leaves. For the hot scoring path that is one dependent load and one extra cache line per tree — the difference between memory-latency-bound and L2-streaming on large, cold, or L2-spilling ensembles. **§10 owns and registers** a load-derived, scoring-only re-encoding of the ensemble that lowers each scored tree to one 64-byte cache line:

```rust
/// One depth-3 oblivious tree, packed into a single 64-byte cache line for the
/// path-A scorer. SoA: 3 feature axes, 3 compare thresholds, the missing-direction
/// mask, and the 8-entry leaf LUT the in-register permute (§10.2) selects from.
/// `#[repr(C)]`, padded to 64 B; a byte-exact re-encoding of the SAME f32 leaves
/// and SAME compares as the owning `ObliviousTree` — NOT a second model.
#[repr(C)]
pub struct PackedTree {
    pub feat: [u8; 3],    // axis per level; u8 ⇒ ≤255-feature hot-path cap (fallback below)
    pub thresh: [u8; 3],  // bin_le per level
    pub miss: u8,         // packed missing_left bits (one per level)
    pub leaf: [f32; 8],   // the leaf LUT; permute target of §10.2
    // remainder of the 64-byte line is padding
}

/// The contiguous path-A scoring view: one PackedTree per tree, in STORED tree
/// order. Built once in `finalize`/deserialize, never serialized. Scoring streams
/// it unit-stride so the hardware prefetcher walks it (and may `_mm_prefetch` the
/// next line). The serialized `Model` — the determinism gate's subject (§10.5) —
/// is untouched; `ScoringBank` is a runtime-derived view, not the wire artifact.
pub struct ScoringBank(pub Vec<PackedTree>);
```

- **Byte-exact, exactness- and determinism-neutral.** `ScoringBank` is a re-encoding of the same `f32` leaves and the same per-level low/left test already in the `Model` — the `miss: u8` field carries the same packed `missing_left` bits as the `Split`s, so the kernel computes the **identical** low bit `if bin == 0 { (miss >> level) & 1 != 0 } else { bin <= thresh[level] }` (R-MISSING, byte-for-byte the §10.2 scalar form). It selects the identical leaf and accumulates in the identical fixed tree order, so it is bit-for-bit equal to `score_trees` walking the `ObliviousTree`s. It is built in `finalize`/deserialize, lives only at runtime, and is **never serialized** — the wire `ModelDoc` (§10.5) and the n_threads ∈ {1,2,8} byte-equality gate (§1) are over the `Model`, which `ScoringBank` does not touch. So it is invisible to I1/I2 and to the determinism gate by construction.
- **u8-axis cap + side-table fallback.** `feat: [u8; 3]` caps the packed hot path at **≤255 features**. Models wider than that fall back to a side-table form (axis indices in a parallel `Vec<u32>` keyed by tree, leaves still packed) and the generic `tree_lookup` path of §10.2 — correctness unchanged, only the one-cache-line layout is forfeited above the cap. Selection between packed and side-table is made once at build, recorded on the bank, and is purely a layout choice (no numeric difference, so no `format_version`/`ExactnessMode` impact).
- **Magnitude (honest).** The streaming win is real but *conditional*: 2–4× on cold-cache / large-ensemble / L2-spilling models, roughly neutral when the ensemble is already L2-resident. It removes one dependent load and one cache line per tree; it is not an exactness or accuracy change.
- **Row-blocked streaming kernel (its implementation).** The bandwidth multiplier on the packed bank is to hold a tile of `W` rows' `u8` columns resident, stream `ScoringBank` once, derive each lane's per-level low bit with the missing-aware test above (the `bin == 0` lanes follow `miss`, all others compare against `thresh[level]`), and accumulate `W` f32 lanes per tree via the §10.2 in-register permute — fixed tree order, so still bit-exact. This is the `Backend::predict_block(rows)` seam anticipated in §02; the layout + kernel detail lives in §11.

### 10.3 Path B — LUT-sum scoring from the TableBank

Once §08 has accumulated and purified the ensemble into a `TableBank` on the merged grid, a row scores by summing one read per realized table:

```rust
impl TableBank {
    /// Raw score = f0 + offset + Σ_u f_u(x_u), one lookup per realized table.
    /// Cost is INDEPENDENT of tree count — it depends only on |tables|.
    /// Fallible: `lookup` returns `PbError::Internal` if a digitized offset is
    /// out of its tensor (cannot happen on a well-formed bank; gated not panicked).
    pub fn score_row(&self, x_binned_on_merged: &[u8], offset: f32) -> Result<f32, PbError> {
        let mut acc = self.f0 as f32 + offset;
        for t in &self.tables {
            acc += t.lookup(x_binned_on_merged)?; // O(1) strided tensor read
        }
        Ok(acc)
    }
    pub fn score(&self, x: &BinnedMatrix, offset: Option<&[f32]>, out: &mut [f32])
        -> Result<(), PbError>;
}
```

`EffectTable::lookup` digitizes each of the ≤3 axis values onto that axis's *merged* borders (sorted union of realized cuts) and reads the dense `Tensor` at the resulting strided offset; the strided offset is range-checked (`get(..).ok_or(PbError::Internal)?`) rather than blindly indexed, honoring the no-panic gate. **Complexity:** O(n_rows · |tables|) reads; crucially **independent of n_trees** — a 4000-tree ensemble that realized 47 tables scores in 47 reads/row. After a fully-corrective refit or ensemble average (§09) collapses tree count, path A speeds up but path B is unchanged; on dense ensembles path B is the faster scorer, which is *why* it is the default once a bank exists. This is the structural inference win the AIM promises: the explainable view is also the fast view.

**Note — flattened single-arena precomputed-offset path-B kernel.** The naive `score_row` above re-digitizes per table; since many tables share an axis, a tile-friendly kernel digitizes each axis **once per row** (the merged-cell index is shared across every table touching that axis) and then reads a flat `Vec<f32>` arena holding all tensors back-to-back at compile-time-known strides, accumulating in fixed table order. This is a constant-factor improvement on path B with no numeric change — same f32 cells, same order — and it is covered by the existing path-equality gate (§10.7), not a new contract. Like the path-A `ScoringBank`, the arena is a load-derived runtime view, never serialized.

### 10.4 Equality of the two paths (the contract)

**Property:** for every binned row `x`, `score_trees_row(x) == table_bank.score_row(x)` to f32 reconstruction tolerance. This is exactly §3's **Reconstruction** check (`Invariant::Reconstruction`) and §3's **ThreeWayEqual** (tree-sum = table-sum = Shapley-sum). It holds to a *derived* float tolerance (the `4·n_trees·EPSILON` accumulation bound of §13, not a magic floor and not literal bit-equality — bit-equality is reserved for the serialized-model determinism gate, §10.5/§10.7) because (a) the merged grid represents each tree piecewise-constant with zero error (research/03 §4.2), and (b) purification only moves mass between tensors, conserving the sum (research/03 §5). Section 10 does not re-prove it; it *invokes* the §08 checks as the gate on any `Model` that ships a bank, and adds the per-row equality property test of §10.7. If they ever disagree, `score_tables` returns `PbError::InvariantViolated { invariant: Invariant::Reconstruction }` rather than silently returning a different number than `predict`.

### 10.5 Serde model format

The serde impl lives in `tri-boost-core` (zero pyo3), so bytes written from Python deserialize identically in any pure-Rust consumer — one impl, no drift. The wire struct wraps the in-memory `Model` as a **plain nested field** — `#[serde(flatten)]` is *not* used, because flatten relies on self-describing formats and would break the required `bincode` (non-self-describing) round-trip:

```rust
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ModelDoc {
    pub format_version: u32,        // wire-contract version (THIS file owns the value)
    pub schema_version: u32,        // Model struct-shape version (§02)
    pub model: Model,               // plain nested (NO #[serde(flatten)] — bincode-safe)
}

impl Model {
    /// Stable, self-describing JSON — the canonical archival format.
    pub fn to_json(&self) -> Result<String, PbError>;        // serde_json::to_string_pretty
    pub fn from_json(s: &str) -> Result<Model, PbError>;     // migrate then validate
    /// Compact same-version binary cache. NOT archival; rejects on version mismatch.
    /// bincode 2.x: encode_to_vec / decode_from_slice over a frozen standard() config.
    pub fn to_bincode(&self) -> Result<Vec<u8>, PbError>;
    pub fn from_bincode(b: &[u8]) -> Result<Model, PbError>;
}
```

The `bincode` bodies are exactly (config frozen once, crate-wide):

```rust
const BINCODE_CFG: bincode::config::Configuration = bincode::config::standard();

fn to_bincode(&self) -> Result<Vec<u8>, PbError> {
    let doc = ModelDoc { format_version: CURRENT_FORMAT_VERSION, schema_version: self.schema_version, model: self.clone() };
    bincode::serde::encode_to_vec(&doc, BINCODE_CFG)
        .map_err(|e| PbError::Serialization(e.to_string()))
}
fn from_bincode(b: &[u8]) -> Result<Model, PbError> {
    let (doc, _len): (ModelDoc, usize) = bincode::serde::decode_from_slice(b, BINCODE_CFG)
        .map_err(|e| PbError::Serialization(e.to_string()))?;
    // bincode is a same-version cache: refuse cross-version, then re-validate.
    if doc.format_version != CURRENT_FORMAT_VERSION {
        return Err(PbError::Serialization(format!(
            "bincode cache format v{} != build v{}; use JSON", doc.format_version, CURRENT_FORMAT_VERSION)));
    }
    validate(doc.model)
}
```

**Fixed-width serialized index fields (cross-platform byte-equality).** `Split.axis` is `u32` and `BinnedMatrix.n_rows` is `u32` (not `usize`), so the serialized bytes are identical on 64-bit hosts and the wasm32 smoke build; `usize` is platform-width-dependent and would break the byte-equality gate. Axis/row counts above 4 G are out of scope for v1 (the choice is made once here; widen to `u64` only with a `format_version` bump).

Evolution rules (frozen): new fields land with `#[serde(default = "...")]`; renames use `#[serde(alias = "old")]`; **never** `#[serde(deny_unknown_fields)]` on `ModelDoc` (forward-compat). A `format_version` newer than the build's `CURRENT_FORMAT_VERSION` → `PbError::Serialization("model format vN is newer than this build (vM)")`. An older version routes through forward-only `From`-based migration:

```rust
fn migrate(doc: serde_json::Value, from: u32) -> Result<Model, PbError>;
```

`from_json`/`from_bincode` always **re-validate** on load: every `ObliviousTree` re-checks I1 (≤3 distinct raw features via `provenance`), `grids`/`provenance` lengths agree, every `Split.axis` is `< grids.len()` (the bound `tree_lookup` relies on — see §10.2), `f0`/leaves are finite, `mode` is consistent, and the `Model.schema: ModelSchema` (§2.6 / R-SCHEMA — `feature_names`, `feature_kinds`, the frozen `cat_encoders: CatEncoderStore`, optional `class_labels`, and `objective: ObjectiveTag`) round-trips intact: `schema.feature_kinds.len()` agrees with `grids.len()`/`provenance.len()`, and every `AxisKind::CategoricalTS { encoding }` resolves to a present encoder in `schema.cat_encoders`. The schema is a plain nested field of `Model`, so it is serialized with the model and covered by `schema_version` (no separate file, no drift) and by the byte-equality gate. A malformed or invariant-violating document is a typed error, never a panic and never a silently-wrong model. `bincode` carries the same `format_version` but **refuses cross-version load** (it is a cache): a mismatch → `PbError::Serialization(..)`, directing the caller to JSON. Determinism: `BorderGrid.borders`, `provenance`, and tree order serialize in stored order with no map iteration (no `HashMap` in the wire types — `FeatureSet`/axes are sorted `Vec`/`SmallVec`; any config map the determinism gate touches is a `BTreeMap`, not `HashMap`), so the byte output is reproducible and diffable, and the n_threads ∈ {1,2,8} byte-equality gate (§1) covers the serialized form (true bit-equality, tolerance 0).

### 10.6 Rating-table / lookup-table export

The filing artifact is produced from a `TableBank` (the *complete* realized support — lossless), not the pruned display view. There are **two distinct forms**, and §10 specifies both:

1. **PURE fANOVA export (the canonical default).** The purified additive tables exactly as §08 produced them — each table **zero-mean under `w`** (the Purity invariant), the intercept `f0` carrying the whole anchor. On a log link the additive tables render as multiplicative relativities `exp(f_u)`; `base = exp(f0)` (or `f0` on Identity). This form **makes no claim that any axis has a 1.000 reference row**, and an all-baseline row does **not** generally predict `base` — the per-axis mean is zero under `w`, not at any particular level. This is the exact decomposition: nothing is moved, the relativities ARE `exp` of the purified cells.

2. **Rating-view re-basing (optional, opt-in).** A filing convention often wants a chosen *reference level* per axis to read exactly `1.000` (so every other cell is a relativity *against* that reference). This is an **exact basis re-centering**, not a re-fit: pick a reference cell per table, subtract that cell's value from every cell of the table (so the reference cell becomes `0` ⇒ `exp(0) = 1.000` on a log link), and **fold the per-axis subtracted shift into `f0`**. Because the same total is added back into `f0` that was removed from the table, `F(x)` is **identically unchanged** for every row (sum conserved); only the *basis* in which the decomposition is written changes. After re-basing — and only then — `base` is the prediction of the all-reference row and each re-centered table reads `1.000` at its reference level. The PURE zero-mean and the re-based forms are the same model in two bases; re-basing is reversible and exactness-preserving.

Each exported cell carries its per-cell `support` (effective `w`-mass, §08) so a filer can see — and a regulator can question — relativities standing on thin data; `support` is display metadata only and never re-enters scoring.

**Readability is sourced from `Model.schema` (R-SCHEMA).** The export resolves raw `FeatureId`s to human-readable `feature_names`, renders each `AxisExport`'s level labels for `AxisKind::CategoricalTS` axes through the frozen `schema.cat_encoders` (`CatEncoderStore`: the per-encoding category → TS-value/bin map *plus* the original level labels), and stamps `class_labels`/`objective` (the `ObjectiveTag`'s `link`/`loss`/`tweedie_rho`) so a `RatingExport` is self-describing without the caller re-supplying names. Because `schema` is part of the serialized `Model` (covered by `schema_version`, §10.5), a round-tripped model exports byte-identical readable labels — the names and category labels are not reconstructed heuristically, they travel with the model.

```rust
#[derive(serde::Serialize, serde::Deserialize)]
pub struct RatingExport {
    pub format_version: u32,
    pub link: Link,
    pub base: f32,                 // exp(f0) on Log; f0 on Identity. In the PURE form this is
                                   // just the intercept; only after `rebased` is it the
                                   // all-reference-row prediction (the "1.000" anchor).
    pub rebased: bool,             // false = PURE zero-mean fANOVA; true = reference-level
                                   // rating-view re-basing applied (per-axis shifts folded into f0)
    pub w: RefMeasure,             // stamped — relativities are only meaningful given w
    pub mode: ExactnessMode,       // Exact tables only; else firewall (below)
    pub tables: Vec<RatingTable>,  // Sobol-ranked, descending σ²(f_u)/σ²(F)
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct RatingTable {
    pub features: Vec<FeatureId>,  // 1..=3 raw features (resolved to feature_names via Model.schema)
    pub axes: Vec<AxisExport>,     // per-axis merged borders / category labels (labels from schema.cat_encoders)
    pub relativities: Vec<f32>,    // exp(f_u) on Log, else raw f_u. PURE: zero-mean under w
                                   // (NOT 1.000-centered); after rebasing: 1.000 at `reference`
    pub reference: Option<Vec<usize>>, // per-axis reference cell index — Some(..) iff rebased
    pub support: Vec<f32>,         // effective w-mass per cell, parallel to relativities — flags thin cells (display only)
    pub sobol: f64,                // σ²(f_u)/σ²(F), the rank key
}
```

```rust
/// Optional per-axis reference selection for the rating-view re-basing. `None`
/// ⇒ the PURE zero-mean fANOVA export (no re-centering). `Some(map)` ⇒ for each
/// listed table, the chosen reference cell is re-centered to read 1.000 and the
/// removed shift is folded into f0 — an EXACT basis re-centering (sum conserved).
pub struct RatingBasis {
    /// Per table (keyed by its FeatureSet), the per-axis reference cell index.
    pub reference: BTreeMap<FeatureSet, Vec<usize>>,
}

impl TableBank {
    /// Export the COMPLETE support as relativities. Errors via the firewall if the
    /// owning Model is Approximate (cannot certify the tables ARE the model).
    ///
    /// `schema` (the owning `Model.schema`, R-SCHEMA) supplies readability: feature
    /// names, categorical level labels (via the frozen `cat_encoders`), class labels,
    /// and the objective tag stamped onto the export.
    ///
    /// `basis = None` produces the canonical PURE zero-mean fANOVA export (each table
    /// zero-mean under `w`; no 1.000 reference row claimed). `basis = Some(b)` applies
    /// the optional rating-view re-basing: each referenced cell is re-centered to 0
    /// (⇒ exp = 1.000 on Log) and the per-axis shift is folded into `f0`, leaving every
    /// row's `F(x)` identically unchanged (exactness-preserving basis change).
    pub fn to_rating_export(
        &self,
        link: Link,
        mode: &ExactnessMode,
        schema: &ModelSchema,
        basis: Option<&RatingBasis>,
    ) -> Result<RatingExport, PbError>;
}
```

The **firewall is enforced at export, not at scoring**: if `mode` is `Approximate { reason }`, `to_rating_export` returns `PbError::ExactnessFirewall(format!("cannot export Exact rating tables: {reason}"))`. An `Approximate` model can still `predict` (path A is always valid) and can still emit *tables + a residual disclaimer* via a separate, explicitly-named API — it just cannot stamp a filing artifact as exact. Tables sort by `sobol` descending so the most consequential relativities lead; the export carries the *full* set for losslessness, and §08's display pruning is a separate top-k *view* over this same structure, never a different file. The optional rating-view re-basing (`basis = Some(..)`) is applied *after* purification and is purely a basis change — it never re-runs §08 and never alters `F(x)`, so it cannot move an `Exact` model to `Approximate`. Both `RatingExport` and `ModelDoc` default to pretty JSON; their `bincode` form (when used as a cache) goes through the same frozen `bincode::serde` path as §10.5.

### 10.7 Testing approach

- **Path-equality property test** (`proptest`, build-blocking): generate random binned rows; assert `|score_trees_row − table_bank.score_row| < tol` for every row, where `tol` is the derived `4·n_trees·EPSILON` reconstruction bound (§13), not a magic floor — this is `Invariant::Reconstruction` exercised through the public scorers.
- **Branch-free correctness:** a reference scalar walk vs `tree_lookup` over all 8 leaf-index patterns per tree, **including rows whose bin is the reserved missing bin 0 on each split axis under both `missing_left` settings** (R-MISSING — the missing lane must follow the learned direction, not always route left); SIMD batch path vs scalar batch path bit-identical.
- **Packed scoring-view equivalence:** scoring through the `ScoringBank` (and its row-blocked kernel) is **bit-identical** to `score_trees` walking the `ObliviousTree`s, over the proptest model corpus, at n_threads ∈ {1,2,8} (tolerance 0) — `ScoringBank` is a byte-exact re-encoding, never a second number; the in-register permute leaf-select (`vpermps`/`vpermi2ps`/`vqtbl`) matches the scalar leaf read across all 8 index patterns and across each `multiversion` target; and a >255-feature model exercises the side-table fallback with identical results.
- **Digitize-once / `predict_binned` equivalence:** `predict_binned` and the flattened single-arena path-B kernel produce bit-identical raw scores to `predict`/`score_row` on the same bins (digitization is hoisted, not changed) — folded into the §10.7 path-equality gate.
- **Hot-loop boundary test:** exercise `tree_lookup` at `idx == 7` (all three bits set) and with a `Split.axis == grids.len() - 1` row so the scoped-`allow` bounds proof of §10.2 is pinned by an executed test; a deserialized model whose `Split.axis >= grids.len()` is rejected by `from_json`/`from_bincode` with a typed error (never reaches the kernel).
- **Reproducibility:** `score_trees` at n_threads ∈ {1,2,8} → bit-identical `out` (the §1 [GATE], scoring side); serialized `ModelDoc` bytes identical across thread counts (JSON and the frozen-config `bincode`), tolerance 0.
- **Round-trip:** `from_json(to_json(m)) == m` and `from_bincode(to_bincode(m)) == m` (structural + bit-equal predictions) over the proptest model corpus; the `bincode` round-trip uses `encode_to_vec`/`decode_from_slice` with the frozen `standard()` config; `bincode` cross-version load → typed error.
- **Migration:** a fixture corpus of every prior `format_version` JSON loads, migrates, and re-validates I1/I2.
- **Firewall:** `to_rating_export` on an `Approximate` model returns `ExactnessFirewall`; on an `Exact` model the round-tripped relativities re-exponentiate to scores equal to `predict` (in BOTH the PURE and the re-based form — re-basing conserves the sum).
- **Pure export is zero-mean:** the PURE export (`basis = None`) has each table `w`-weighted mean zero (Purity); it does **not** assert a 1.000 reference row, and an all-baseline row need not predict `base`. `base == exp(f0)` on Log (`f0` on Identity) is the intercept only.
- **Rating-view re-basing (exact basis change):** `to_rating_export(.., Some(basis))` leaves `F(x)` bit-identical to the PURE export's reconstruction for every row (the per-axis shift folded into `f0` exactly cancels the table re-centering, sum conserved); each re-centered table reads `1.000` at its declared `reference` cell, and only then does the all-reference row predict `base`.
