## 03 — Data model & binning

> Owner of: 'BinnedMatrix', 'BorderGrid', 'AxisKind::{Numeric, Missing}', 'AxisProvenance', 'BinConfig' (the binning config: 'max_bin', 'subsample_for_binning', 'min_data_per_bin', border family); the global per-feature border grid (seeded quantile/midpoint borders, 'max_bin', grid persistence); the column-major 'u8' design matrix; the merged/union-grid rule; the missing-value subsystem (reserved bin + learned default direction); rare-level collapse and unseen-level→base; exposure/offset plumbing ('offset = log(e)'). Consumes 'Loss::init_score' (§05) and 'FitSpec.exposure' (§09). Conforms verbatim to the §2 shared types.

This section defines how raw user data becomes the frozen, persisted representation every later stage reads. The load-bearing decision is the **one global per-feature border grid**: computed once, persisted in the 'Model', and reused bit-identically at fit, validate, predict, and table-export time. This is not a speed convenience — it is the precondition for I2. Because every tree shares the same axes, every tree's 8-cell tensor is exactly representable on the merged grid, so accumulation is lossless and 'purify(Σ trees) = Σ purify(trees)' (§08). Without one frozen grid, none of the exactness machinery holds.

### 03.1 Decisions (with defaults)

- **Binning is global and one-shot.** Borders are computed once at fit, before the first tree, and never re-derived. The grid is part of the 'Model' (§2.6) and round-trips through serde (§10).
- **Quantile borders on a seeded subsample.** For numerics with more than 'max_bin' distinct values, borders are sample quantiles over a deterministic subsample of 'subsample_for_binning' (default **200 000**) rows, drawn with a per-feature re-seeded 'Pcg64' (deterministic re-seeding, §03.3; *not* a "splittable" PRNG) derived from 'FitSpec.seed'. Equal-count bins (not Hessian-weighted) in v1 — see §03.11 fork.
- **Midpoint borders for low-cardinality numerics.** If a feature has at most 'max_bin' distinct finite values, borders are the midpoints of consecutive sorted distinct values. This gives exact splits and one table row per real value — ordinal rating factors (driver age, NCD years, vehicle group) read 1:1 with reality.
- **'max_bin' default 254.** Bin ids are 'u8'; bin 0 is the reserved missing bin and data bins occupy '1..=n_data_bins'. 'max_bin' caps the interior borders: 'borders.len() <= max_bin - 1', so 'n_data_bins = borders.len() + 1 <= max_bin = 254' and 'n_bins = n_data_bins + 1 <= 255' (the realized bin ids stay in '0..=254', leaving '255' free; never 256). 'max_bin' is per-fit; §07 raises it on monotone-constrained features so coarse borders do not wipe out all monotone-feasible splits.
- **Missing → reserved bin 0 + learned per-split default direction.** NaN (and Arrow nulls) map to bin 0. Direction is learned per 'Split' during gain evaluation (§06) and carried in the explicit 'Split.missing_left: bool' field (§2.5), persisted as the side bin 0 joins. No imputation; missingness is information.
- **Rare-level / sparse-bin collapse.** A numeric bin whose row count is below 'min_data_per_bin' (default 0, off) is merged into its lower neighbour at grid-build time. Categorical rare-level collapse and unseen-level→base live in §04, built on this section's numeric-axis machinery and provenance plumbing.
- **Exposure/offset is first-class.** 'FitSpec.exposure: Option<&[f32]>' sets per-row 'offset = log(e)', added to the raw score before the inverse link every iteration; the intercept initializes to 'F0 = log(Σ w·y / Σ w·e)', anchoring 'e^0 = 1.000' as the base level. Plumbing only — the offset vector is computed here and threaded; §05/§09 consume it.

### 03.2 Core types

These are the §2 signatures verbatim, with the field-level notes this section owns.

```rust
/// One feature's border grid. `borders` are the interior breakpoints between
/// adjacent finite-data bins; `borders.len()` borders define `borders.len() + 1`
/// half-open intervals, so there are `n_data_bins = borders.len() + 1` data bins.
/// A finite value v lands in data bin `1 + (count of borders strictly below v)`:
/// the *k-th* interval (0-indexed) is data bin `k + 1`. Equivalently
/// `bin(v) = 1` iff `v <= borders[0]`, and `bin(v) = n_data_bins` iff
/// `v > borders.last()`. Ascending, no duplicates.
/// `missing_bin` is always 0; data bins occupy `1..=n_data_bins`.
pub struct BorderGrid {
    pub borders: Vec<f32>, // sorted strictly ascending; len = n_data_bins - 1
    pub n_bins: u16,       // n_data_bins + 1 (the missing bin); == borders.len()+2; <= 255
    pub missing_bin: u8,   // == 0, reserved
}

/// Column-major, pre-binned design matrix. `data[f]` is feature f as bin ids
/// (`u8`), length `n_rows`. f32 -> u8 binning happens once at ingest.
/// `n_rows` is a fixed-width `u32` so the serialized matrix is byte-identical
/// across platforms (the determinism [GATE]; the core smoke-builds on wasm32).
/// This is the §2.2 base type; §03.2a defines the two *roles* it plays
/// (`TrainBinnedMatrix` vs `ServeBinnedMatrix`) — same layout, different
/// categorical encodings, only one of which is audited.
pub struct BinnedMatrix {
    pub data: Vec<Vec<u8>>,            // [n_features][n_rows], column-major
    pub n_rows: u32,                   // fixed-width; <= 4G rows in v1
    pub grids: Vec<BorderGrid>,        // one per feature, index = column
    pub provenance: Vec<AxisProvenance>,
}

pub enum AxisKind { Numeric, CategoricalTS { encoding: TsEncodingId }, Missing }
pub struct AxisProvenance { pub raw: FeatureId, pub kind: AxisKind }

/// Binning configuration (owned here; the §06 Config references it, never redefines it).
pub struct BinConfig {
    pub max_bin: u8,                   // default 254; caps borders.len() <= max_bin-1,
                                       // so n_data_bins <= max_bin and n_bins <= max_bin+1.
                                       // (bin 0 = missing; n_bins <= 255 at the default.)
    pub subsample_for_binning: u32,    // default 200_000
    pub min_data_per_bin: u32,         // default 0 (rare-bin merge off)
    pub border_family: BorderFamily,   // default EqualCount (see §03.11)
}
pub enum BorderFamily { EqualCount, HessianWeighted } // v1.5 fork, §03.11
```

'provenance[f].raw' is the **raw underlying feature** behind axis 'f'. For a plain numeric, axis index == raw 'FeatureId'. §04 may introduce extra axes (categorical TS) that point back to the same or different raw features; I1's ≤3-distinct-feature budget (§3) is always checked on 'provenance[axis].raw', never on the axis index. This section guarantees provenance is populated for every numeric axis ('AxisKind::Numeric', 'raw = FeatureId(column)').

### 03.2a Train vs Serve binned matrices (the audit-on-serve seam)

A 'BinnedMatrix' plays one of **two roles**, and conflating them silently leaks the categorical target into the audited tables. This section owns both role types; the skeleton (§4) registers them. They share the layout, the numeric grids, and the provenance above — they differ **only** in the categorical (TS) columns.

```rust
/// Fitting-time matrix. Numeric columns are binned through the frozen global
/// grids (§03.3). Categorical TS columns carry the **leakage-free** encodings
/// (out-of-fold / prefix target statistics, §04) used ONLY to compute gradients
/// and grow trees. These encodings are noisy by design (fold-dependent) and MUST
/// NEVER be accumulated into the TableBank.
pub struct TrainBinnedMatrix(pub BinnedMatrix);

/// Prediction-time matrix. Numeric columns are binned through the SAME frozen
/// grids (numeric binning is fold-independent — Train == Serve there). Categorical
/// TS columns are re-encoded through the **frozen full-data `CatEncoderStore`**
/// (the served `ModelSchema.cat_encoders`, §04/skeleton §2.6), so a given raw
/// category always maps to one fixed bin. This is the matrix the served model
/// function is evaluated on AND the matrix the audited TableBank is accumulated
/// from (§08).
pub struct ServeBinnedMatrix(pub BinnedMatrix);
```

**Audit-on-serve rule (load-bearing for I2).** I2 lossless-equivalence is asserted between the **served model function** (frozen encoders) and the **TableBank**, *both* evaluated on a 'ServeBinnedMatrix'. Therefore:
- 'explain()' / table accumulation (§08) and 'predict' (§10) MUST re-encode raw categoricals through the frozen 'CatEncoderStore' into a 'ServeBinnedMatrix'. They MUST NEVER reuse the noisy 'TrainBinnedMatrix' (its fold-dependent TS values do not match the served function and would corrupt the Reconstruction gate).
- 'fit' (§06) consumes the 'TrainBinnedMatrix' for gradient/gain; the engine never accumulates tables, so no leakage path exists at fit time.
- **Numeric binning is fold-independent.** A numeric column produces byte-identical bins under both roles (same global grid, same 'bin'); only categorical TS axes differ between Train and Serve. The §04 'CatEncoder' freeze is what makes the Serve encoding deterministic and auditable.

The two-role distinction is stated again from the consuming side in §04 (encoder freeze) and §08 (the audited bank is built on Serve). This section is the single owner of the types.

### 03.3 Border construction algorithm

Borders are built independently per feature; the loop over features is rayon-parallel and order-independent (each feature writes only its own 'BorderGrid'), so the result is bit-reproducible across thread counts (§1 [GATE]).

```text
fn build_grid(col: &[f32], weight: Option<&[f32]>, max_bin: u8, seed: u64,
              feat: FeatureId) -> Result<BorderGrid, PbError>:
    finite = collect finite (non-NaN, non-inf) values of col   // missing excluded
    if finite empty:
        // all-missing axis: no interior borders ⇒ n_data_bins = 0+1 = 1,
        // n_bins = 2 (bin 0 = missing, bin 1 = the lone degenerate data bin).
        return BorderGrid { borders: [], n_bins: 2, missing_bin: 0 }
    distinct = sorted unique finite values
    if distinct.len() <= max_bin as usize:
        // midpoint borders: exact, one bin per real value
        // borders.len() = distinct.len() - 1 <= max_bin - 1  ⇒  n_data_bins = distinct.len()
        borders = midpoints of consecutive distinct values            // len = distinct.len() - 1
    else:
        // per-feature deterministic re-seed: frozen splitmix64 mix of
        // (base=seed, round=0, stage=BINNING_STAGE, block=feat) -> Pcg64::seed_from_u64
        let rng = Pcg64::seed_from_u64(splitmix64_mix(seed, 0, BINNING_STAGE, feat.0 as u64));
        sample = deterministic_subsample(finite, weight, SUBSAMPLE_FOR_BINNING, rng)
        // at most max_bin - 1 interior quantile borders at evenly spaced ranks,
        // so borders.len() <= max_bin - 1 and n_data_bins = borders.len()+1 <= max_bin
        qs = linspace(0, 1, max_bin + 1)[1 .. max_bin]                // interior only
        borders = weighted_quantiles(sample, weight, qs)
        borders = dedup_ascending(borders)                            // collapse ties
    // n_data_bins = borders.len() + 1; n_bins = n_data_bins + 1 (the missing bin).
    BorderGrid { borders, n_bins: borders.len() as u16 + 2, missing_bin: 0 }
```

Notes that make this exact and reproducible:
- **Tie dedup is mandatory.** Quantiles on skewed columns produce duplicate borders; duplicates are collapsed so 'borders' is strictly ascending and 'n_bins' reflects realized resolution. A feature can end with fewer than 'max_bin' bins.
- **Subsample is seeded and weight-aware.** 'deterministic_subsample' draws without replacement from a 'Pcg64' stream that is *re-seeded per work unit*: 'Pcg64::seed_from_u64(splitmix64_mix(seed, round=0, stage=BINNING_STAGE, block=feat))', where 'splitmix64_mix' is the frozen mix shared by all stages (§1). This is deterministic re-seeding, **not** a "splittable" PRNG (which is unimplementable as previously named); each feature's draw is independent of feature iteration order and thread count. Below 'SUBSAMPLE_FOR_BINNING' rows the full column is used. With weights, ranks are weight-cumulative.
- **Weighted quantile** uses the 'averaged_inverted_cdf' convention (sklearn-compatible), computed in 'f64' internally, borders stored 'f32'.

### 03.4 Binning a column (f32 → u8)

Given a frozen 'BorderGrid', mapping a value to its bin is a branchless binary search. The hot-loop policy (§1) is enforced: no indexing that can go out of bounds, no arithmetic that can overflow on the fallible path. The canonical map is **bin(v) = 1 + (count of borders strictly below v)** for finite 'v': 'partition_point(|&b| b < v)' returns exactly that count in '0..=borders.len()', and 'borders.len() <= max_bin - 1 = 253' (default), so 'k + 1 <= 254' fits 'u8' with no overflow; the cast is exact and the conversion is checked.

```text
fn bin(v: f32, g: &BorderGrid) -> Result<u8, PbError>:
    if v.is_nan(): return Ok(g.missing_bin)        // == 0 — NaN/null is the ONLY missing case
    // Finite v (including ±inf, which are non-NaN): count borders strictly below v.
    // This naturally clamps out-of-range / extreme finite values to the first/last
    // finite data bin — v <= borders[0] ⇒ bin 1; v > borders.last() ⇒ bin n_data_bins.
    let k = g.borders.partition_point(|&b| b < v); // k in 0..=borders.len() <= 253
    // data bin = k + 1, in 1..=n_data_bins = 1..=borders.len()+1 <= 254.
    // u8::try_from never fails here, but we map the impossible case to
    // PbError::Internal rather than panic-on-overflow.
    u8::try_from(k + 1).map_err(|_| PbError::Internal { what: "bin index > u8".into() })
```

'partition_point' over an ascending slice is the idiomatic, panic-free binary search (no indexing, no 'unwrap'). The '+ 1' is bounded by the cardinality invariant ('// JUSTIFIED:' 'k <= borders.len() <= max_bin - 1 = 253', so 'k + 1 <= 254 = n_data_bins'); a boundary test exercises the maximum-cardinality grid (254 data bins) to prove no overflow and no panic. **Missing vs extremes (canonical, matches §13):** only 'NaN'/Arrow-null maps to the missing bin 0; every finite value — including out-of-range or extreme values and '±inf' (which are non-NaN, so they bypass the missing branch) — clamps to the first/last finite data bin via 'partition_point'. The reserved bin 0 is for NaN/null *only*. Binning the whole matrix is rayon-parallel by column, writing column-major 'Vec<u8>'. The transform is also the public prediction-time path: the same 'grids' that trained the model bin the scoring data, so train and predict share one grid by construction (numeric binning is fold-independent, so the Train and Serve matrices are byte-identical on numeric axes; §03.2a).

### 03.5 The merged / union grid (the I2 lever)

The §2.7 'TableBank.merged_grids' are derived here-style but owned by §08; this section fixes the rule that makes them exact. Each tree splits a feature only at borders drawn from that feature's 'BorderGrid', so the set of breakpoints the *ensemble* uses on feature 'i' is a subset of 'grids[i].borders'. The merged grid for feature 'i' is:

```text
merged_borders[i] = sorted_unique( ⋃ over trees t, splits s on raw i of grid_border(s) )
```

i.e. the union of *realized* split borders, not the full 'BorderGrid' and not a dense grid. Between two consecutive merged borders the ensemble is exactly constant, so:
1. Every tree's 8-cell tensor expands onto the merged grid with **zero approximation** (a tree's value over a merged-cell is constant ⇒ broadcasts losslessly).
2. 'Σ_u T_u^raw(x) = F(x)' identically (Reconstruction check, §3.I2.1).
3. The merged grid is the *minimal* exact grid — finer adds zero information and wastes table memory.

This section guarantees the property the rule depends on: realized borders are a subset of the persisted 'BorderGrid', because split-finding only ever proposes borders from it.

### 03.6 Missing-value subsystem

A designed subsystem, not an afterthought:
- **Reserved bin 0** holds every NaN/null per feature. It carries its own '(G, H)' mass in the histogram (the 'Hist' type, 'i64' bin accumulators, owned by §06), so missing rows participate in gain evaluation rather than being dropped.
- **Learned default direction** is decided per 'Split' by the engine (§06): during the gain sweep, bin 0's mass is tried on both sides and the higher-gain side is kept, recorded in the explicit 'Split.missing_left: bool' field (§2.5; not a 'bin_le' sentinel — §03/§06/§08 all cite this one carrier). Persisted with the model, so a missing/never-seen value scores deterministically — an audit requirement for filed tables.
- **All-missing axis** ('borders' empty) is legal: every row lands in bin 0, the feature contributes no usable split (gain ≡ 0), and the tree gracefully terminates lower-order (I1 early-termination, §3) rather than erroring.

Categorical unseen-level→base and rare-level grouping are §04's, built on this same reserved-bin mechanism.

### 03.7 Exposure / offset plumbing

'FitSpec.exposure: Option<&[f32]>' is validated and transformed here, then threaded to §05/§09:

```text
offset[i] = ln(exposure[i])        // requires exposure[i] > 0
F0        = link(Σ w·y / Σ w·e)    // via Loss::init_score(y, weight, Some(offset))
raw(x)    = F0 + offset + Σ alpha_t · tree_t.lookup(x)   // offset added every round
```

Validation: 'exposure' length must equal 'n_rows' ('PbError::ShapeMismatch'); any non-finite or non-positive entry is 'PbError::InvalidInput { what }'. Offset is only meaningful for 'Link::Log' / 'Link::Logit'; under 'Link::Identity' an offset is added in score space directly (still valid). The offset never enters binning or the grids — it is a per-row score shift, orthogonal to tree shape, so I1/I2 are untouched.

### 03.8 Complexity & performance

- Grid build: 'O(s log s)' per feature on the 's'-row subsample (sort for distinct/quantiles); 'O(n_features)' parallel.
- Binning: 'O(n_rows · n_features · log n_bins)', column-major sequential writes — cache-friendly and the input layout §06's histogram build wants.
- Memory: the 'u8' matrix is 4× smaller than 'f32'; one 'Vec<u8>' per feature, no row-major copy. Borders are a handful of 'f32' per feature, negligible.
- Reproducibility: per-feature independence + per-feature re-seeded subsample + 'f64' quantile accumulation in fixed rank order ⇒ identical grids regardless of thread count, satisfying the §1 byte-equality [GATE].

### 03.9 How it serves the three aims

- **Predictiveness:** midpoint borders keep ordinal factors exact; learned missing direction recovers XGBoost's sparsity-aware gain; 254 bins are ample resolution.
- **Decomposable (I2):** the single frozen grid + the realized-union merged grid make every tree losslessly representable on a shared grid — the algebraic precondition for purify-then-sum equality and the Reconstruction gate.
- **Fast:** one-shot 'u8' binning makes per-level split search independent of 'n_rows' (§06); column-major layout vectorizes the histogram build; the small grid keeps merged tensors tiny (§08).

### 03.10 Testing

- **Unit:** 'bin' round-trips border semantics (upper-inclusive; NaN→0; ±inf→extremes); midpoint vs quantile branch selection at the 'max_bin' boundary; tie dedup yields strictly-ascending borders. A **boundary test** drives a maximum-cardinality grid (254 data bins) to prove 'bin' never overflows 'u8' and never panics (the §1 hot-loop policy).
- **Property ('proptest'):** for random columns, every binned value lies in '0..g.n_bins'; monotone — 'a <= b ⇒ bin(a) <= bin(b)' for finite values; permuting rows leaves the grid unchanged (subsample is value-set-deterministic given seed).
- **Reproducibility [GATE]:** identical grids and identical 'BinnedMatrix' across 'n_threads ∈ {1, 2, 8}' (the §1 determinism gate, exercised at the binning layer); the per-feature re-seed 'splitmix64_mix(seed, 0, BINNING_STAGE, feat)' is asserted position-stable and thread-count-independent.
- **Realized-subset invariant:** a fitted model's realized split borders are a subset of the persisted 'BorderGrid' (guards the §03.5 precondition for I2; a property test over fitted models).
- **Exposure:** 'F0' matches 'link(Σwy/Σwe)' in closed form; non-positive exposure errors; offset leaves grids byte-identical.

### 03.11 Open fork

**Equal-count vs Hessian-weighted quantile borders.** XGBoost's weighted quantile sketch places borders at equal *Newton loss mass* (rank weighted by 'h_i'), which is more accurate for the second-order objective. v1 uses **equal-count** borders ('BorderFamily::EqualCount'; simpler, weight-aware, sklearn-aligned). Recommended default: ship equal-count in v1; treat Hessian-weighted borders as a benchmark-gated v1.5 knob ('BorderFamily::HessianWeighted'). Within the Hessian-weighted family the **border-selection objective** is a distinct lever from sample weighting — the candidate families are **GreedyLogSum** (CatBoost's default, maximizes Σ log of bin loss-mass) and **MinEntropy** (maximizes bin-count entropy); v1.5 would expose one of these as the realized objective, defaulting to GreedyLogSum. The whole fork is exactness-neutral (borders are still a per-feature grid frozen before training), so it cannot threaten I1/I2 — purely an accuracy/speed trade to measure, not a correctness question.
