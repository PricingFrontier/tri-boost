## 04 — Categorical handling

> Owns: `AxisKind::CategoricalTS`, `TsEncodingId`, the leakage-free Target-Statistic (TS) encoder, `CatEncoder`/`CatEncoderStore` (the frozen full-data encoders persisted in `Model.schema.cat_encoders`), empirical-Bayes auto-shrinkage, the Fisher sorted-ordinal split's *encoding* half, low-cardinality one-hot, multi-distinct-categorical-axis trees, the forbid-combination-CTR rule, and the **audit-on-serve** re-encoding rule. Uses (does not own): `AxisProvenance`/`BorderGrid` + `TrainBinnedMatrix`/`ServeBinnedMatrix` (§03), `ModelSchema`/`Model.schema` (skeleton §2.6, registered there), the I1 budget check and the split-finder (§06), the `Loss` trait + exposure offset (§05/§03), purification + `RefMeasure` (§08).

This section turns each raw categorical column into **one ordered numeric axis** that the symmetric histogram splitter consumes natively, while persisting enough provenance and label metadata that the exported tables read on the original category labels and the ≤3-raw-feature invariant is enforced on *distinct underlying features*. Everything here is **upstream of the split search**, so it composes with depth-3 oblivious trees without touching I1/I2 — the work is preserving raw-feature provenance and keeping each axis legible.

### 04.1 Decisions (with defaults)

| # | Decision | Default | Aim served |
|---|---|---|---|
| D1 | Encode each categorical to **one** numeric axis via a leakage-free target statistic. | on | accuracy + decomposable |
| D2 | **Empirical-Bayes auto-shrinkage** toward the exposure-weighted base rate; `m` estimated as `within_var / between_var` (sklearn `TargetEncoder` auto-smoothing; no tuning). | `smooth = Auto` | accuracy + readable rows |
| D3 | Leakage avoidance via **K-fold cross-fitting** (v1.5 baseline) or **ordered TS** (the differentiator); plain mean encoding is **forbidden**. | `Ordered` (fallback `KFold{k:5}`) | accuracy (no prediction shift) |
| D4 | Category stays a **distinct human-readable row** via the **Fisher sorted-ordinal split**: order categories by encoding, split on a single numeric border. | on | decomposable + readable |
| D5 | **One-hot** for genuinely low-cardinality factors; dummies reassembled onto one axis at table time. | `one_hot_max_size = 2` | exactness + readability |
| D6 | A tree may split on **up to 3 distinct categorical axes** → exact cat×cat[×cat] tables. | on | accuracy |
| D7 | **Forbid CatBoost combination CTRs** (concatenated multi-categorical crosses): they pack >3 raw features into one axis. | hard-forbidden | upholds I1/I2 |
| D8 | For regression, raise the target-quantization border count above CatBoost's default 1. | `ctr_target_border_count = 16` | accuracy |
| D9 | Persist the frozen `category → encoding` map + label list in `Model.schema.cat_encoders` (the `CatEncoderStore`); serve + TableBank re-encode through it (audit-on-serve), unseen levels → base prior. | always | reproducibility + deployability |

`Counter` (frequency) encoding and per-feature multi-prior grids are deferred (v1.5/v2) and listed as open forks in §04.11; they do not change any signature below.

### 04.2 Axis kind, encoding id, and provenance

The encoded axis is registered through the shared `AxisKind::CategoricalTS { encoding }` (§2.1). `TsEncodingId` is a stable, serializable handle identifying the encoder variant so predict-time decoding is unambiguous and bit-reproducible.

```rust
/// Stable id of the encoder that produced a CategoricalTS axis. Serialized in the
/// Model so scoring re-applies the exact transform. New variants are append-only.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum TsEncodingId { OrderedTs = 0, KFoldTs = 1, OneHot = 2, Counter = 3 }

/// Leakage-avoidance scheme. Plain mean encoding is intentionally absent.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum LeakageScheme {
    /// CatBoost-style online prefix encoding over a seeded permutation.
    Ordered { n_perms: u8 },          // default n_perms = 1 (CPU plain-boosting parity)
    /// Disjoint hold-out folds; deterministic given the seed.
    KFold { k: u8 },                  // default k = 5
}

/// Empirical-Bayes prior strength `m`.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum Smooth { Auto, Fixed(f32) } // Auto = sklearn-style m ≈ within_var / between_var
```

Crucially, a `CategoricalTS` axis carries `AxisProvenance { raw: FeatureId, kind: CategoricalTS { encoding } }` with the **single** raw `FeatureId` of the source column. The encoded numeric values are an internal column; the budget check in §2.5 (`splits → HashSet of provenance[axis].raw`) therefore counts the *one* underlying categorical, never the encoded column — this is the structural guarantee that D6/D7 cannot be violated by accident.

### 04.3 The encoder: ordered/cross-fitted TS + empirical-Bayes shrinkage

For category level `c` with within-category count `n_c`, link-appropriate mean `mean_c`, exposure-weighted global mean `p`, and prior strength `m`, the shrunken encoding is

```
enc(c) = (n_c · mean_c + m · p) / (n_c + m)
```

`p` is the **exposure-weighted / link-appropriate base rate** (`p = Σ w·y / Σ w·e` for frequency; the deviance-appropriate weighted mean for Gamma/Tweedie), never a bare `mean(y|c)` — mandatory under exposure offsets, not optional. `mean_c` and the counts are accumulated with the same fixed-order reduction discipline as §06 histograms so encodings are bit-identical across thread counts.

`Smooth::Auto` sets `m ≈ within_var / between_var` from the between- vs within-category variance of the (weighted) target, so well-separated categories shrink little and noisy singletons collapse toward `p`. Count/severity targets use a weighted/deviance-aware variance form; the estimator is documented and frozen for reproducibility.

**Leakage avoidance** is a *separate* concern from shrinkage and the two are kept orthogonal:

- **`Ordered`** — encode row `k` from only the rows preceding it in a deterministic permutation σ:
  `x̂_k = (Σ_{σ(j)<σ(k), x_j=x_k} w_j·y_j + m·p) / (Σ_{σ(j)<σ(k), x_j=x_k} w_j + m)`.
  Because `y_k` never enters its own encoding, train/test conditional distributions match (no prediction shift). Default `n_perms = 1` (CPU plain-boosting parity); >1 only reduces the early-rows-starved variance.
- **`KFold`** — encode each fold's rows from the other `k−1` folds' means, then shrink. Fully batch, trivially parallel, deterministic given the seed; a strict special case of ordered TS. Reimplemented internally — **no runtime dependency on sklearn's `TargetEncoder`** — so encoding is owned, seeded, and serialized.

Both σ (the `Ordered` permutation) and the `KFold` fold assignment are drawn from a **deterministically re-seeded** `Pcg64`: this column's stream is `Pcg64::seed_from_u64(splitmix64_mix(base, round, stage, block))` over the frozen `splitmix64` mix of `(base, round, stage, block)` (§1/§06), where `base = seed`, the categorical-encode `stage` is fixed, and `block` is the column's dense `FeatureId` — never a "splittable" PRNG. This makes σ and the folds position-stable and thread-count-independent, which the §1 byte-equality [GATE] requires.

**Train/serve consistency is the footgun — the audit-on-serve rule.** There are two distinct binned design matrices and they MUST NOT be confused (R-CATSERVE):

- **`TrainBinnedMatrix`** (§03) carries the **out-of-fold / prefix** categorical TS encodings — leakage-free, but *noisy* (they carry fold/permutation noise). It is used **only for fitting**.
- **`ServeBinnedMatrix`** (§03) re-encodes raw categoricals through the **frozen full-data `CatEncoder`s** (`Model.schema.cat_encoders`). It is used for **prediction AND for accumulating the audited `TableBank`** (§08).

Numeric binning is fold-independent, so `Train == Serve` on every numeric axis; **only categorical axes differ** between the two matrices. The I2 lossless-equivalence (§3) is therefore stated between the **served model function** (frozen encoders) and the `TableBank`, **both evaluated on a `ServeBinnedMatrix`** — never on the training matrix. Concretely: `explain()` and table accumulation MUST build a `ServeBinnedMatrix` by re-encoding the raw categoricals through the frozen `CatEncoderStore`; they MUST NEVER reuse the noisy `TrainBinnedMatrix`. The frozen `category → enc(c)` maps and the bin borders are persisted in the `Model` (in `schema.cat_encoders` and `grids`) and re-applied identically at predict time, so a row never lands in a different histogram bin at serve time than the audited table was built for.

For regression, the target is first quantized into `ctr_target_border_count = 16` borders (vs CatBoost's median-binarized default of 1) before forming `mean_c`, recovering signal that median binarization discards on continuous severity / pure-premium. Counts and means are computed over the quantized target.

```rust
/// Configuration for one categorical column's encoding.
pub struct TsConfig {
    pub scheme: LeakageScheme,        // default Ordered { n_perms: 1 }
    pub smooth: Smooth,               // default Auto
    pub target_borders: u16,          // default 16 (regression); 1 for binary classification
    pub one_hot_max_size: u16,        // default 2
    pub min_data_per_group: u32,      // default 10; levels below collapse to a shared «rare» bucket (§04.12)
}

/// The fitted, serializable, FROZEN full-data encoding for one categorical column —
/// the single source of truth at serve time and for building the audited TableBank.
/// `labels[i]` is the original level string/id; `encoding[i]` is its frozen full-data
/// value; `base` is the prior `p` used for unseen levels at score time. This is the
/// `category → TS value/bin map + level labels` the skeleton's `ModelSchema` cites.
pub struct CatEncoder {
    pub raw: FeatureId,
    pub id: TsEncodingId,             // the variant that produced this axis (schema key)
    pub labels: Vec<CatLevel>,        // original level identity, for the export
    pub encoding: Vec<f32>,           // labels[i] → encoding[i]; bin borders come from §03
    pub base: f32,                    // unseen / new level → prior p (collapse to base rate)
    pub config: TsConfig,
}

/// The frozen full-data `CatEncoder` set for a trained model, addressed by the axis's
/// `TsEncodingId`. Persisted inside `Model.schema.cat_encoders` (skeleton §2.6 /
/// `ModelSchema`, R-SCHEMA) and serialized with the `Model` (covered by `schema_version`).
/// `AxisKind::CategoricalTS { encoding: TsEncodingId }` resolves to a concrete `CatEncoder`
/// here. This store is what `ServeBinnedMatrix` construction (§03) and `explain()` /
/// TableBank accumulation (§08) re-encode raw categoricals through — NEVER the noisy
/// training encodings (the audit-on-serve rule above). §04-owned.
pub struct CatEncoderStore {
    /// One frozen encoder per (raw `FeatureId`, `TsEncodingId`) categorical axis,
    /// indexed by the axis's `TsEncodingId`; lookup is deterministic and total over
    /// the model's categorical axes. Unknown ids ⇒ `PbError::Internal { what }`.
    encoders: Vec<CatEncoder>,
}

impl CatEncoderStore {
    /// Resolve the frozen encoder for a `CategoricalTS` axis. Returns
    /// `PbError::Internal { what }` if the id is absent — never panics.
    pub fn get(&self, id: TsEncodingId, raw: FeatureId) -> Result<&CatEncoder, PbError>;
}

/// Fit the leakage-free training encodings (out-of-fold/prefix) AND the frozen
/// full-data map, given the raw column, target, exposure-weighted weights and seed.
pub fn fit_cat_encoder(
    raw: FeatureId,
    levels: &[u32],                   // dense level ids for this column
    y: &[f32],
    weight: &[f32],                   // already folds in exposure for the base rate
    cfg: &TsConfig,
    seed: u64,                        // `base`; σ/folds = Pcg64::seed_from_u64(splitmix64_mix(base, round, stage, block))
) -> Result<(CatEncoder, Vec<f32>), PbError>; // (frozen encoder, per-row TRAINING encodings)
```

The returned per-row **training** encodings feed §03's binning to produce the categorical `u8` axes of the **`TrainBinnedMatrix`** (fit-only). The frozen `CatEncoder` is stored in the model's **`CatEncoderStore` (`Model.schema.cat_encoders`)** for scoring, `ServeBinnedMatrix` construction, and table export — it is what every serve-time and audit path re-encodes through (the audit-on-serve rule). Errors surface as `PbError::InvalidInput` (e.g. a non-finite target into the variance estimate) — never a panic.

### 04.4 The Fisher sorted-ordinal split — category stays a row

A *continuous* TS axis would erode the "category → relativity" reading actuaries need. We use the leakage-free TS only to derive a **data-driven ordering**, then split on a single numeric border between consecutive categories:

1. Sort the column's distinct levels by `enc(c)` ascending (Fisher 1958: the optimal contiguous partition is reachable in this order).
2. Bin the *ordered* levels — one bin per level for low cardinality, quantile-merged for very high cardinality (§03) — so bin id is rank, not raw encoding.
3. The split is the ordinary symmetric `bin <= bin_le` test (`Split`, §2.5).

Each category stays a distinct, ordered, human-readable row ("postcode group G → 1.18×") and the split is a single `(axis, border)` test, fully compatible with the one-split-per-level rule. The fitted ordering is part of `CatEncoder`, so the exporter renders the ordered category list and the same ordering is reproduced at serve time.

**Static vs exact ordering (deliberate v1.5 simplification).** Fisher's contiguity theorem is exact for the mean of the quantity the loss averages — `Σg/Σh` recomputed per node per level — which equals the static target mean *only for squared error*; for log-link Poisson/Gamma/Tweedie the static order is slightly sub-optimal. v1.5 ships the **static** TS-sorted axis (bin-once, exact tables); the **per-level re-sort by `Σg/Σh`** is a §06 optimization (fork §04.11) — an accuracy refinement, never an exactness question.

### 04.5 One-hot for low cardinality

For factors with cardinality ≤ `one_hot_max_size` (default 2), skip the TS and one-hot encode: each level becomes a 0/1 indicator with a single border at 0.5 — exact, leakage-free, the cleanest mapping for a binary rating factor (an exact 1.000-reference row). Two hard rules:

1. **Reassemble dummies onto one categorical axis at table-accumulation time** (§08 consumes this). Independent per-dummy axes would fragment the main effect, manufacture spurious within-categorical "interactions" over the structurally-empty all-ones cell, and break product-marginal purification over zero-mass cells. All dummies of one source column share **one `AxisProvenance.raw`**, so they count as **one** raw feature for the budget check.
2. **Respect the 3-slot budget.** A `k`-level field spreads across `k` binary axes; sharing one `raw`, they consume one raw-feature slot — and one-hot is reserved for cardinality ≤2 so a single field never needs more. Higher-cardinality fields route to ordered TS (one slot, multiple borders).

### 04.6 Multi-distinct-categorical-axis trees (the safe combination edge)

A depth-3 oblivious tree may split on **up to 3 distinct categorical TS axes**, so `cat × cat [× cat]` interactions surface as **exact, purified 2D/3D tables over raw factors** — recovering the ~1.86% logloss CatBoost gets from combinations, inside the order-3/lossless invariant. No new mechanism is needed: each TS axis carries its single raw `FeatureId`, and §2.5's construction check counts distinct raw features. A tree on `city_ts`, `product_ts`, `age` touches exactly three raw features and is admitted; a fourth distinct categorical is rejected with `PbError::InvariantViolated { FeatureBudget }`.

### 04.7 Forbidding combination CTRs — the load-bearing prohibition

CatBoost's greedy **combinations** concatenate categoricals already in the current tree with each remaining one and CTR-encode the cross (e.g. `city × product`). A combination column packs a multi-feature interaction into **one axis the tree treats as one feature**: a tree splitting on a 2-categorical combination plus two other features depends on **four** raw features — silently breaking the `≤3 raw features ⇒ exact ≤3rd-order fANOVA` invariant that is pattern-boost's reason to exist.

**Decision: combination CTRs are hard-forbidden** — no config flag enables them. Structural enforcement: the encoder only ever emits an axis whose `AxisProvenance.raw` is a **single** `FeatureId`; no code path produces a multi-raw axis, so the §2.5 budget check is a redundant second line of defence, not the only one. `model_size_reg` (which only bites once combinations exist) is therefore a no-op and not implemented. The constructive replacement is §04.6 — strictly better for the explainable-table product, with no opaque crossed axis.

### 04.8 How categoricals become exact table axes; I1/I2 upkeep

All categorical processing is upstream of the split-finder and each emitted axis is a single piecewise-constant numeric column, so leaves stay constant, the lookup-table form is preserved, and order ≤3 holds — the decomposition is **exact for the model**. A target-dependent TS changes *semantics*, not exactness: the axis is a target-statistic-ordered bin. The export always ships the `category → encoding` map and ordered label list with the tables (§10) so the rating table reads on original labels, not "TS-bin 0.37".

- **I1** — upheld by provenance: every axis maps to one raw `FeatureId`, the §2.5 distinct-raw-feature count is the gate, and combination CTRs that would violate it are forbidden by construction (§04.7).
- **I2** — upheld because every emitted axis is a single deterministic piecewise-constant scalar function of the row, so the five §3 checks (Reconstruction/MassConservation/Purity/VarianceSum/ThreeWayEqual) pass unchanged. The TS encoder is target-dependent but **deterministic and frozen**, so it never flips the model out of `ExactnessMode::Exact`. (A *continuous* TS axis would — it is not implemented; the Fisher sorted-ordinal path is the only TS consumer.) Critically, I2 is asserted between the **served model function** and the `TableBank` **both on a `ServeBinnedMatrix`** (frozen full-data encoders) — the audited bank is accumulated by re-encoding raw categoricals through `Model.schema.cat_encoders`, **never** from the noisy `TrainBinnedMatrix`; otherwise the bank would not equal the function the model actually serves.

For purification, rare-level collapsing (§03) and one-hot reassembly shift the feature marginal, hence `RefMeasure` mass on that axis — harmless to exactness, flagged in export (§08).

### 04.9 Complexity, performance

- Encoder fit: `Ordered` is `O(n)` per column over the permutation with `O(cardinality)` running accumulators; `KFold` is `O(k·n)` but embarrassingly parallel over folds. Both are linear and run once at ingest, off the hot loop.
- The Fisher ordering is `O(cardinality · log cardinality)` per categorical, once.
- At split time a categorical axis is indistinguishable from a numeric one — same `u8` histogram, same summed-gain scan — so there is **zero** per-iteration categorical overhead. Inference is the standard branch-free 8-cell lookup.
- Memory: one `CatEncoder` per categorical (labels + `f32` encodings), persisted in the model; negligible versus histograms.

### 04.10 Testing

- **Leakage gate (mandatory).** Train on a shuffled target; assert the encoded-feature Sobol importance does not spike — the standard test that flags plain mean encoding. Run for both `Ordered` and `KFold`.
- **No-self-leak property** (`proptest`): for `Ordered`, the encoding of row `k` is invariant to changing `y_k`.
- **Reproducibility**: encoder fit at `n_threads ∈ {1,2,8}` produces byte-identical `CatEncoder` and encodings (feeds the §1 [GATE]).
- **Budget enforcement**: constructing a tree on 4 distinct categorical axes returns `InvariantViolated { FeatureBudget }`; one-hot dummies of one column count as one raw feature.
- **Exactness**: a fitted model with categorical axes passes all five §3 checks and stays `Exact`; the five checks are the I2 oracle.
- **Train/serve consistency**: round-trip a `CatEncoderStore` (in `Model.schema.cat_encoders`) through serde (§10) and assert the `ServeBinnedMatrix` re-encoding equals the frozen full-data map; an unseen level maps to `base`. **Audit-on-serve gate**: assert the TableBank is accumulated on a `ServeBinnedMatrix` (frozen encoders), not the `TrainBinnedMatrix` — feed a fixture whose out-of-fold and full-data encodings differ on a categorical axis and assert the five §3 checks pass against the served function (they would fail if accumulation reused the training encodings).
- **Shrinkage correctness**: singleton categories collapse to within float tolerance of `p`; `Smooth::Auto`'s `m` matches the closed-form `within_var / between_var` estimate on synthetic data.
- **Math unit tests**: `enc(c)` against the closed form; exposure-weighted `p` against hand-computed `Σwy/Σwe`.

### 04.11 Open forks (with recommended defaults)

1. **`Counter` (frequency) encoding** — leakage-free, target-independent prevalence axis; safest fANOVA semantics. *Recommend: ship at v1.5 as an opt-in axis (`TsEncodingId::Counter` already reserved); off by default since prevalence can proxy portfolio composition.*
2. **Exact per-level re-sort by `Σg/Σh`** (Fisher flavor (b)) for log-link losses — more accurate ordering at the cost of a sort in the hot loop and a per-tree category→bit map. *Recommend: default off; ship static TS-sorted (flavor (a)) at v1.5; benchmark (b) before promoting.*
3. **Multi-prior grid per categorical** (several `m` values as parallel axes) — tuning-free shrinkage sweep, but proliferates columns. *Recommend: v2, off by default.*
4. **Native bitset categorical splits** (LightGBM/XGBoost set-membership) — incompatible with the shared one-split-per-level rule and yields a model-chosen coarsened axis rather than per-category rows. *Recommend: research-only; the ordered-TS path (which also keeps the monotone machinery) is preferred.*

### 04.12 Rare-level collapse mechanics (the postcode case)

Rare-level collapse is **owned by §03** (it shifts the feature marginal and hence `RefMeasure` mass; §04.8); this subsection pins the categorical-specific knobs the encoder honors so the behavior is concrete and frozen.

- **Threshold.** A level with exposure-weighted count `n_c < min_data_per_group` (default **`min_data_per_group = 10`**) is *rare* and does not get its own encoding/row.
- **Collapse target.** Rare levels collapse to a **single shared "rare" bucket per column** (not folded silently into the base prior): the bucket accumulates `Σ n_c`, `mean_c`, and weights across all rare levels and is encoded by the same `enc(·)` formula, so it gets one credible shrunken value and one exportable row labelled `«rare»`. The base prior `p` remains reserved for **unseen** levels at score time (`CatEncoder.base`); rare-at-train and unseen-at-serve stay distinct. The bucket's member labels are retained in `CatEncoder.labels` so the export can list which raw postcodes mapped to `«rare»`.
- **Interaction with Fisher ordering.** Collapse runs **before** the §04.4 sort, so the `«rare»` bucket is a single level ordered by *its own* `enc(value)` — it slots into the rank order like any other category and the one-bin-per-rank invariant is unchanged. Quantile-merging of remaining high-cardinality levels (§04.4 step 2) then proceeds on the collapsed level set.

`min_data_per_group` is distinct from §03's `min_data_per_bin` (grid-build rare-*bin* merge) and from §06/§07's leaf credibility floors; it is a per-column encoder knob carried on `TsConfig` (default 10).
