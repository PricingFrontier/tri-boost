# SPEC-README

tri-boost is a gradient boosting machine library (pure-Rust core `tri-boost-core` + Python bindings `tri_boost`) whose defining property is that **the trained model is exactly explainable by construction**. This `spec/` directory is the binding contract: the canonical skeleton (`00-spec-skeleton.md`) plus 14 numbered sections (`01`–`14`). This README is the index, decision log, glossary, and the pre-final fix list.

## The three aims (every decision serves these, in tension-managed balance)

1. **Maximum predictiveness** — as accurate as possible; use every technique at our disposal (the gap-closing playbook plus the 229-technique inventory). Headliners: CatBoost distillation, quantized histograms, MVS sampling, fully-corrective refit, Nesterov/AGBM, the interaction-selection funnel, bagged ensemble selection.
2. **Fully decomposable** — the trained model decomposes *losslessly* (exactly) into a constant + 1D main-effect + 2D pairwise + 3D triple purified fANOVA tables on a shared grid. Mechanism: every tree is a **depth-3 oblivious/symmetric tree** (one shared `(feature, threshold)` split per level ⇒ at most 3 distinct raw features per tree ⇒ an 8-cell lookup table).
3. **Fast** — competitive with LightGBM/CatBoost on CPU training; structurally faster at inference (branch-free 8-cell lookup + table-sum scoring).

## The hard invariants

- **I1 — depth-3 oblivious / ≤3 distinct raw features per tree.** Enforced on *distinct raw features* (via `AxisProvenance`), not encoded columns. Non-symmetric growth, linear/soft leaves, and >3-raw-feature encoded axes (e.g. CatBoost combination CTRs) are forbidden on the exact path.
- **I2 — exact ≤3rd-order decomposability on a shared grid.** The ensemble equals a constant + sum of ≤3-feature purified tables, bit-for-bit within float tolerance. Backed by five build-blocking checks: Reconstruction, MassConservation, Purity, VarianceSum, ThreeWayEqual.

Anything that bends I1/I2 is flagged in its owning section and gated behind the typed **Exact/Approximate firewall** (`ExactnessMode`), which refuses to export `Exact` rating tables when an invariant-bending technique is active. Exactness-preserving operations (always stay `Exact`): reference-measure recomputation, multi-step Newton leaf refit, fully-corrective refit, Nesterov mixing, self-ensemble averaging, global mean re-anchoring.

## How the 14 sections fit together

The library is a pipeline. **§03** ingests and bins raw columns into a `BinnedMatrix` on a shared per-feature `BorderGrid`; **§04** layers leakage-free categorical target-statistic axes onto that grid. **§05** supplies the `Loss` trait (the only objective-aware code) which produces per-row `GradHess`. **§06** is the engine: it quantizes g/h, builds histograms, runs the oblivious level-wise Newton split-finder, refits leaves at full precision, and emits a `Model` of weighted `ObliviousTree`s. **§07** constrains *which* feature supports the engine may grow (monotone, interaction-order, the admission funnel). **§08** accumulates the trained ensemble and purifies it into the `TableBank` (the exact fANOVA tables) and runs the five invariant checks. **§09** layers predictiveness boosters (distillation, refit, Nesterov, ensemble selection) that touch only target/leaf-scalars/tree-weights/table-averages — never tree shape. **§10** scores and serializes (`Model` + `TableBank`, JSON canonical + bincode fast). **§11** is the performance spine (cache layout, SIMD, the determinism harness); **§12** is the Python surface; **§13** realizes the engineering gates; **§14** sequences the build.

The data flow of types: `BinnedMatrix` → `GradHess`/`QuantGradHess` → `ObliviousTree` → `Model` → `TableBank` → exports. Each is defined once in skeleton §2 and used verbatim.

## How to read the spec

Read the **skeleton first** (`00-spec-skeleton.md`): §1 (engineering standards/gates), §2 (canonical types — the single source of truth), §3 (the invariant contract + firewall), §4 (section-ownership map with the no-overlap rule). Then read sections in pipeline order. Each section *owns* a slice of decisions and types and may not redefine another's; cross-references use the §2 shared types. `[GATE]` marks a build-blocking CI check.

## Status

Draft, internally consistent. All 14 sections + the frozen skeleton are written and mapped. The **CORRECTIONS** list below (a de-duplicated merge of the consistency, engineering, and completeness critic passes) has been **APPLIED across all files (2026-06-21) and grep-verified** — the cross-cutting fixes (i64 `Hist` accumulators, `max_bin=254`, deterministic `splitmix64` re-seeding, the `bincode` 2.x API, the no-panic hot-loop policy, `Result`-returning `Loss::grad_hess`, fixed-width serialized index fields) are resolved identically everywhere. The list is retained below as an audit trail. The spine and all named headline gap-closers are present and well-specified; none of the corrections threatened I1/I2.

# DECISIONS

The load-bearing decisions the spec commits to. Forks the brainstorm already resolved are stated as decisions; the one genuinely open fork is flagged.

- **Decision:** Every tree is a depth-3 oblivious tree (≤3 distinct raw features, 8 leaves), with legal graceful early-termination to fewer levels. **Rationale:** the unique growth policy giving an exact finite fANOVA decomposition; also branch-free fast inference. **Depth is fixed at 3 deliberately** (decided 2026-06-22, knob declined): a depth-3 *ensemble* is already a universal ≤3rd-order approximator, so deeper trees add no representational capacity — only marginal convergence efficiency — and are not worth the variable-depth machinery (see spec §01 Remark; `design/03-methodology-review.md`). **Owner:** §06.
- **Decision:** Split gain is the oblivious level-wise summed **Newton** gain `½Σ[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)]`; leaves are the exact `w* = −G/(H+λ)`. **Rationale:** XGBoost-grade accuracy and exact leaf values (not CatBoost's cosine score). **Owner:** §06.
- **Decision:** Mandatory `boost_from_average` intercept `f0 = link(weighted mean)` as a scalar, never "tree 0". **Rationale:** correct base level; rating-table base rate. **Owner:** §05/§06.
- **Decision:** Histograms use **quantized integer g/h** with stochastic rounding; leaves are refit from full-precision g/h (mandatory on log-link). **Rationale:** associative integer sums make accumulation order-independent ⇒ the bit-reproducibility mechanism, plus ~2× speed. **Owner:** §06/§11.
- **Decision:** Bit-reproducibility is a first-class `[GATE]`: identical inputs+config+seed ⇒ byte-identical model independent of thread count, tested at `n_threads ∈ {1,2,8}`. **Rationale:** auditability and trust. **Owner:** §1/§11/§13.
- **Decision:** Histogram float reductions (the `FullF64` cross-check path and leaf refit) use fixed-size chunk folds in index order with a **fixed `CHUNK_ROWS` constant** (not thread-count-derived). **Rationale:** makes the float reduction tree identical across pool sizes so byte-equality holds. **Owner:** §11.
- **Decision:** A single `seed: u64` threads through all randomized stages; per-work-unit independent streams are produced by **deterministic re-seeding** `Pcg64::seed_from_u64(splitmix64_mix(base, round, stage, block))`, not by an (unimplementable) "split". **Rationale:** position-stable, thread-count-independent draws. **Owner:** §1/§06.
- **Decision:** `max_bin` default **254** everywhere (bin 0 reserved for missing; `n_bins ≤ 255`, fits `u8`). **Rationale:** one internally-consistent grid invariant. **Owner:** §03.
- **Decision:** Reserved missing bin (bin 0) with a **learned default direction** per split. **Rationale:** sparsity-aware accuracy. **Owner:** §03.
- **Decision:** Categoricals use leakage-free ordered/cross-fitted **Target Statistics → Fisher sorted-ordinal split** with empirical-Bayes auto-shrinkage; the category stays a row. Up to 3 distinct categorical axes per tree. **Combination CTRs are forbidden** (would break I1). **Rationale:** CatBoost's accuracy edge, kept exactness-clean. **Owner:** §04.
- **Decision:** Pricing objectives Poisson/Gamma/Tweedie{rho} are in scope as `Loss` implementors with exact (g,h); exposure offset `offset = log(e)`. Early stopping uses **deviance, not RMSE**. **Rationale:** correct objective and base-level=1.000 anchoring. **Owner:** §05.
- **Decision:** `max_delta_step` is a leaf-stage `|w*|`-clamp (hessian-inflation rejected — it would perturb the quantized histograms); Poisson default **0.7**. **Rationale:** stabilizes log-link without mutating histograms. **Owner:** §05/§06.
- **Decision:** `max_interaction_order ∈ {1,2,3}`, **default 3**. **Rationale:** the full exact capacity. **Owner:** §07.
- **Decision:** Interaction admission is the **heredity + FAST-RSS + Sobol/triple-detector funnel** as a *soft prior*, never a hard gate; joint boost over admitted supports, single final purification. **Rationale:** focuses capacity without sacrificing exactness. **Owner:** §07.
- **Decision:** Monotone constraints are by feature **name → sign** (never positional), realized as a per-level joint leaf-clamp that may trigger graceful early-termination. **Rationale:** pricing economics + readable tables. **Owner:** §07.
- **Decision:** Explainability path is **accumulate → purify (3→2→1→intercept, Lengerich mass-moving) → tables**. **Rationale:** exact lossless decomposition. **Owner:** §08.
- **Decision:** Default reference measure `w` is **Laplace-smoothed empirical product-of-marginals** (`RefMeasure::ProductMarginals { laplace > 0 }`); `Uniform` and `Joint` (Hooker) are alternatives. **Rationale:** clean Sobol/variance-sum behavior; sensible "pure" effects. **Owner:** §08.
- **Decision:** Local attributions are exact interventional SHAP / Faith-Shap ≤order-3 as O(1) table reads; TreeSHAP is a **test oracle only**, never shipped. **Rationale:** exact-by-construction beats approximate post-hoc. **Owner:** §08/§13.
- **Decision:** Distillation is a **two-parameter, off-by-default** mode. (1) *Whether* to distil = `FitSpec.distill: Option<DistillSpec>` (`None` default ⇒ no teacher is fit, train on true labels); the teacher (default **CatBoost**, itself an oblivious-tree ensemble) is fit **only** when distillation is on. (2) *How much* = `DistillSpec.blend` = **true-label weight**, default **0.5** (balanced); the "soft" gradient is the base `grad_hess` called with the teacher output as target (`g = blend·g_true + (1−blend)·g_soft`). `blend=1.0` is the degenerate zero-teacher case (a test oracle), not the disable switch. **Rationale:** an explicit on/off avoids overloading a blend sentinel and skips teacher training when off; 0.5 avoids over-following the teacher's tail bias. **Owner:** §09 (config) / §05 (`BlendedLoss` adaptor).
- **Decision:** Fully-corrective refit (ridge IRLS over #trees×8 leaves), Nesterov/AGBM, DART, and self-ensemble averaging are offered and are **exactness-preserving**. **Rationale:** accuracy without touching tree shape. **Owner:** §09.
- **Decision:** Ensemble selection is **bagged greedy Caruana** with `Σα=1, α≥0`, shared `w`, union grid, re-purification. **Rationale:** robust accuracy gain that folds into tree weights. **Owner:** §09.
- **Decision (open fork):** Fully-corrective refit / Nesterov are **default-off**, benchmark-gated. **Rationale:** net gain unverified under the oblivious cap. **Owner:** §09/§14. This is the one genuinely open fork.
- **Decision:** Serialization is **serde_json** (canonical, self-describing) + **bincode v2** (fast binary, `bincode::config::standard()`, frozen) — **never pickle**. `schema_version` round-trips. **Rationale:** stable documented contract; byte-equality requires a pinned bincode config. **Owner:** §10/§02.
- **Decision:** `#![forbid(unsafe_code)]` at the `tri-boost-core` crate root; SIMD via safe `multiversion`/`pulp`/`wide` wrappers; any core `unsafe` is encapsulated, `// SAFETY:`-proven, tested, Miri-run. PyO3 unsafe is quarantined in `tri-boost-py`. **Rationale:** safety-first engineering standard without making the binding crate impossible to compile. **Owner:** §1/§11.
- **Decision:** Typed `PbError` + `Result` on every fallible public fn; **no panics in library code** (`unwrap_used`/`expect_used`/`panic`/`indexing_slicing` denied). **Rationale:** a bug degrades to a typed error in a user's Python session, never a crash. **Owner:** §02 (enum) / all.
- **Decision:** Python is PyO3 + maturin, **abi3-py310**, sklearn-compatible, zero-copy numpy (`PyReadonlyArray2<f32>` in, `into_pyarray` out), GIL released around training. **Rationale:** portable wheels, familiar API. **Owner:** §12.
- **Decision:** MSRV **1.74** (`[workspace.lints]` floor, still manylinux2014-compatible), verified in CI; `cargo deny` is `[GATE]`. **Rationale:** broad portability, audited deps, inherited lint gates. **Owner:** §1/§13.
- **Decision:** Milestone = on **TabArena**, beat EBM/GA2M and reach near-parity with unconstrained XGBoost/LightGBM/CatBoost, with *every* model exactly decomposable. TabArena is an internal yardstick; **no benchmarking tooling ships**. **Rationale:** competitive predictiveness *with* exact explainability is the whole thesis. **Owner:** §01/§14.

# GLOSSARY

The canonical shared types/terms (skeleton §2) — single source of truth, one line each.

- **`FeatureId(u32)`** — index into the user's original raw feature columns.
- **`AxisKind` / `AxisProvenance`** — an axis is `Numeric`, `CategoricalTS { encoding }`, or `Missing`; provenance tracks the raw `FeatureId` behind an axis so I1 counts *distinct raw* features.
- **`BorderGrid`** — one feature's ascending bin borders; bin 0 is the reserved missing bin; `n_bins ≤ 255`.
- **`BinnedMatrix`** — column-major pre-binned design matrix (`data[f]` = column f as `u8` bin ids) plus grids and provenance.
- **`GradHess`** — full-precision per-row first/second derivatives of the loss w.r.t. raw score F.
- **`QuantGradHess` / `GradScale`** — quantized integer g/h (with scale) for associative, order-independent histogram sums; leaves refit from full precision.
- **`Loss` / `Link`** — the objective trait (`grad_hess`, `init_score`, `link`, `pred_from_raw`, `deviance`); `Link ∈ {Identity, Log, Logit}`.
- **`ObliviousTree` / `Split`** — depth-3 symmetric tree: `splits` (1..=3 levels, one `(axis, bin_le)` each), `leaves: [f32; 8]`, `depth`. Test is `bin ≤ bin_le`.
- **`Model`** — trained ensemble: scalar `f0`, weighted trees `Vec<(f32, ObliviousTree)>`, grids, provenance, `link`, `ExactnessMode`, `schema_version`.
- **`EffectTable` / `FeatureSet`** — one purified effect tensor for a feature set `u` (size 0..=3, sorted distinct raw ids) on the merged grid, plus its Sobol variance and per-cell effective `support` (display-only metadata flagging thin cells).
- **`TableBank`** — the complete lossless decomposition: intercept `f0`, all realized tables, merged grids, and the stamped reference measure `w`.
- **`RefMeasure`** — purification reference: `ProductMarginals { laplace }` (default), `Uniform`, or `Joint` (Hooker).
- **`PbError` / `Invariant`** — the typed crate error enum; `Invariant ∈ {FeatureBudget, Decomposability, MassConservation, Reconstruction, VarianceSum, ThreeWayEqual}`.
- **`ExactnessMode`** — `Exact` | `Approximate { reason }`; the firewall that gates `Exact` table export.
- **`Booster` / `FitSpec`** — the public estimator handle; `FitSpec` carries `loss`, `weight`, `exposure`, monotone/interaction config, and `seed`.
- **Raw score `F` / `raw`** — score-space (pre-link) quantity. **`pred`** — response-space (post-inverse-link). **`w`** — the reference measure. **`u`** — a `FeatureSet`. **`f0`** — the scalar intercept.

# CORRECTIONS

**STATUS: APPLIED & VERIFIED (2026-06-21).** Retained as an audit trail of what was reconciled. Prioritized, section-keyed, de-duplicated merge of the three critic passes. P0 = contradiction/violation; P1 = real gap or API mismatch; P2 = naming/registration hygiene. Items marked **(skeleton)** were amended in `00-spec-skeleton.md` by the skeleton owner.

## P0 — contradictions & violations (must fix)

- **[§06/§11/§02 — P0]** Histogram accumulator type/width contradicts itself: §06 `Hist` uses `Vec<i64>`, §11 `FeatureHist`/`LevelHists.arena` use `Vec<i32>`, §02's `Backend` names it `HistogramSet`. i32 accumulators overflow on large n (and overflow panics under `overflow-checks`, breaking no-panic). **Fix:** i64 bin accumulators everywhere (counts stay u32); prove the bound `n_rows·max|g_q| < i64::MAX`; unify on one name (`Hist`, registered §06-owned) referenced by §11/§02.
- **[§03/§11/§12 — P0]** `max_bin` default conflicts (254 vs 256 vs 255). **Fix:** **254** everywhere; correct §11's memory example and its `n_bins ≤ 256` comment to `≤ 255`; correct §12 sklearn default to 254.
- **[§05/§06 — P0]** `max_delta_step` Poisson default conflicts: §05 says 0.7 (`Option<f32>`), §06 `Config` is a bare `f32` defaulting to 0.0 (uncapped) — Poisson via §06 default is unstabilized, plus a type mismatch. **Fix:** make `Config.max_delta_step: Option<f32>` defaulting `None`; when `None` fall back to `Loss::max_delta_step()` (Poisson ⇒ 0.7).
- **[§05.6 — P0]** Self-contradiction: heading names "hessian-inflation" as default, body adopts the leaf-stage clamp and rejects hessian-inflation. **Fix:** change the decision line to "leaf-stage `|w*|`-clamp is the default (hessian-inflation rejected — perturbs `QuantGradHess`)".
- **[§04 — P0]** `Smooth::Auto` formula direction contradicts itself: D2/test say `between/within`, §04.3/type-comment say `within/between` (reciprocals). **Fix:** settle on `within/between` (sklearn TargetEncoder auto-smoothing; matches the prose rationale); fix D2 and the test.
- **[cross-cutting — P0]** "Splittable Pcg64" is unimplementable as named — the determinism `[GATE]` rests on it. **Fix:** drop "splittable"; specify deterministic per-work-unit re-seeding via a frozen `splitmix64` mix of `(base, round, stage, block)` into `Pcg64::seed_from_u64`. Update §1/§02.3/§03.3/§06.7/§09.6/§11. **(skeleton §1)**
- **[cross-cutting — P0]** `bincode` 2.x API: §10/§11/§12 call removed top-level `serialize`/`deserialize`; byte-equality depends on the (unspecified) config. **Fix:** use `bincode::serde::encode_to_vec(.., bincode::config::standard())` / `decode_from_slice`, freeze the config, update all call sites. **(skeleton §1 dep note)**
- **[§10/§06/§03 — P0]** Canonical hot-path code (`tree_lookup`'s `row[split.axis]`/`tree.leaves[idx]`, the `fit` loop's `+=`, `bin`'s `(k+1) as u8`) violates the spec's own `indexing_slicing`/`arithmetic_side_effects` deny-gates. **Fix:** show the panic-free form (`slice.get(i).ok_or(PbError::Internal)?`) or a `#[allow(clippy::indexing_slicing)]`-scoped fn with a `// JUSTIFIED:` bounds proof and a boundary test; state the hot-loop policy explicitly.
- **[§02.3/§13.8 — P0]** `arithmetic_side_effects = "deny"` crate-wide makes every accumulation a lint error and is never reconciled. **Fix:** restate the real policy — `overflow-checks = true` in all profiles for integer overflow, float arithmetic explicitly exempt, clippy `arithmetic_side_effects` scoped not crate-root-blanket. **(skeleton §1)**

## P1 — gaps & API mismatches

- **[§07/§06/§12 — P1]** Interaction `groups` whitelist is tested (§07.9) but has no entry point: `FitSpec` carries only scalar `max_interaction_order`. **Fix:** replace `FitSpec.max_interaction_order: u8` with `FitSpec.interaction: InteractionPolicy { max_order, groups }`; thread through §06 `fit` and the §12 kwarg. **(skeleton §2.9)**
- **[§07/§13 — P1]** `MonotoneMap(HashMap)` / `AdmissionPrior` HashMaps collide with the §13.4 anti-HashMap determinism gate. **Fix:** `MonotoneMap` → `BTreeMap<String, MonoSign>`; state `AdmissionPrior` lookups are order-independent pure lookups (or switch to `BTreeMap`); document these maps are config-only, never serialized.
- **[§05/§09 — P1]** Distillation blend polarity is inverted between §05 (`tau` = soft weight) and §09 (`blend` = true weight), and §09 references a nonexistent `grad_hess_soft` trait method. **Fix:** unify on §09's `blend` (true-label weight, default 0.5); §05 `BlendedLoss` uses the same name/polarity (`g = blend·g_true + (1−blend)·g_soft`); replace `grad_hess_soft(t)` with "base `grad_hess` called with `teacher_raw` as target"; fix the §05.9 test (`blend=1.0` reproduces base loss).
- **[§06/§12 — P1]** `early_stopping` default conflicts (§06 `Some(50)` on, §12 sklearn `None` off); also `n_trees`/`learning_rate` diverge (1000/0.05 vs 2000/0.03). **Fix:** pin one source of truth — sklearn forwards the core Config defaults; recommend early stopping on (`Some(50)`). Define the no-eval-set fallback (internal holdout vs disabled).
- **[§06/§07 — P1]** Leaf-credibility floors are defined in *both* §06 `Config` and §07 `CredibilityFloor` (`min_data_in_leaf`, `min_sum_hessian` vs `min_sum_hessian_in_leaf`, `path_smooth`) — a no-overlap breach plus a name mismatch. **Fix:** §07 owns the floors (per §4); §06 references them; unify the name to `min_sum_hessian_in_leaf`. Keep §03's distinct `min_data_per_bin` (grid-build rare-bin merge) separate.
- **[§08/§12 — P1]** §08's release-gate `verify: bool` is unplumbed (`FitSpec`/`explain` have no such field). **Fix (recommended):** defer the flag; drive the release/debug split by `cfg!(debug_assertions)` with **all five checks run by default** including Purity (it certifies individual tables, which the rating export reads). State the fork as resolved-default, not open.
- **[cross-cutting / §05/§12 — P1]** `Loss::grad_hess` is infallible but the §12 `PyLoss` path is fallible and the only specified resolution is `.expect("py loss")` — a no-panic-gate violation. **Fix:** make the trait `fn grad_hess(..) -> Result<(), PbError>`; the engine already returns `Result`, cost is one `?`. Delete the `/// Panics: never` line. **(skeleton §2.4)**
- **[§08/§13 — P1]** The five Invariant checks have divergent signatures: §08 `-> Result<(), PbError>`, §13 `-> Result<(), Invariant>`. **Fix:** standardize on `Result<(), PbError>` (with `InvariantViolated { invariant }` carrying the `Invariant`); reconcile §13.
- **[§08/§13 — P1]** "Bit-equal" (§08) vs `recon_tol = 1e-4 + 4·n_trees·EPSILON` (§13) is a precision contradiction; `1e-4` is an unjustified magic floor. **Fix:** drop the `1e-4` floor; derive the `4·n_trees·EPSILON` accumulation bound and call the claim "equal to a derived float tolerance" (not "bit-equal") for the reconstruction checks; reserve true bit-equality for the serialized-model determinism gate (tolerance 0), which the fixed-`CHUNK_ROWS` discipline makes achievable.

## P2 — registration & naming hygiene (no-overlap rule)

- **[§02/§10 — P2]** `Split.axis: usize` and `BinnedMatrix.n_rows: usize` are serialized but `usize` is platform-width-dependent, breaking cross-platform byte-equality (the core smoke-builds on wasm32). **Fix:** type serialized index fields fixed-width — `Split.axis: u32`, `n_rows: u32` (or `u64` if >4B rows must be supported, chosen once).
- **[§10/§02 — P2]** `#[serde(flatten)]` on `ModelDoc.model` does not work with bincode (non-self-describing), breaking the required binary round-trip. **Fix:** drop `flatten`; use a plain nested `ModelDoc { format_version, schema_version, model: Model }`.
- **[§11 — P2]** `n_chunks = rayon::current_num_threads()` makes float-reduction boundaries pool-dependent (the `FullF64` byte-equality claim fails). **Fix:** derive chunk count from a fixed `CHUNK_ROWS` constant (covered by the P0 determinism fix; flagged here as the concrete site).
- **[§02 — P2]** Public `Backend` trait references `engine`-local `HistogramSet`/`LevelConstraints` (visibility contradiction). **Fix:** make `Backend` `pub(crate)` (only `CpuBackend` ships in v1), or promote and register those types.
- **[§06/§02 — P2]** `HistPrecision::FullF64` goes through `Backend` but §02's contract mandates "quantized-integer accumulation". **Fix:** soften the Backend contract to "quantized-integer *or* fixed-order float-fold accumulation".
- **[skeleton §4 — P2]** Register all new public types under their owning sections (the no-overlap rule requires registration first): `Metric` + the three added `Loss` methods `default_metric`/`hessian_floor`/`max_delta_step` (§05, also update §2.4); `RatingExport`/`RatingTable`/`AxisExport`/`ModelDoc` (§10); `InteractionPolicy`/`MonotoneMap`/`MonoSign`/`AdmissionPrior`/`HeredityMode`/`CredibilityFloor`/`LevelDecision`/`FeatureMask` (§07); `TsConfig`/`CatEncoder`/`CatLevel`/`LeakageScheme`/`Smooth`/`TsEncodingId` (§04); `BinConfig`/`subsample_for_binning` home (§03); `Config`/`Sampling`/`HistPrecision`/`Accel`/`Hist` (§06); `DistillSpec`/`TeacherKind`/`RefitSpec`/`NesterovSpec`/`EnsembleSpec`/`OuterBag`/`DartSpec` (§09); the §11/§12 layout and Py types. **(skeleton §4)**
- **[§06/§08 — P2]** Missing-direction carrier (`bin_le` sentinel vs an explicit field) is split across three sections and pinned in none. **Fix:** add explicit `Split.missing_left: bool` (one byte, clearest) and have §03/§06/§08 cite it. **(skeleton §2.5)**

## Completeness — committed-but-omitted techniques (apply as additive, exactness-neutral knobs)

All value/sampling-level, none threatening I1/I2.

- **[§06]** Add per-level/per-node column subsampling (`colsample_bylevel`/`bynode`/`rsm`) — assumed by the §09 ensemble-diversity path. **[v1/v1.5]**
- **[§06]** Add `min_split_gain` (`gamma`) → graceful early-termination on `LevelGain < min_split_gain`. **[v1.5]**
- **[§06]** Specify the overfitting detector / `best_iteration` selection rule concretely (currently an opaque `early_stop.update(..)`); `best_iteration` is load-bearing for the §08 accumulator. **[v1]**
- **[§06]** Decide and reserve an `LrSchedule` hook (gap-closing treats LR×n_trees as CORE; currently a `// later` comment). **[v2 but CORE]**
- **[§06]** Add optional L1 leaf regularization `alpha` with soft-thresholding (`w* = −SoftThreshold(G,α)/(H+λ)`) — sparser, more readable tables. **[v2]**
- **[§06]** Add `Sampling::Bayesian { temperature }` (Bayesian bootstrap) or record an explicit skip. **[v1.5]**
- **[§05]** Add a positive-hessian robust loss (Pseudo-Huber or Log-Cosh — drops straight into the Newton engine) or record deferral; currently only Quantile/L1 are mentioned as deferred. **[v1.5]**
- **[§09 (or §05)]** Add global mean / bias **re-anchoring** (`reanchor` scalar folding `δ = log(Σwy/Σwμ̂)` into `f0`) — exact, the one calibration-adjacent nicety the brainstorm kept in scope. **[v1]**
- **[§04]** Pin the categorical rare-level collapse mechanics: `min_data_per_group` default + collapse target (base-prior bin vs shared "rare" bucket) + interaction with Fisher ordering (the postcode case). **[v1.5]**
- **[§09/§08]** Specify inner-bag smoothing and the per-cell SE-band annotation (computation, type, display-only status; not an fANOVA component) or record as a non-goal. **[v1.5]**
- **[§03]** Name the border-selection-objective family (GreedyLogSum/MinEntropy) in the §03.11 hessian-border fork (distinct lever from sample weighting). **[v1.5]**
- **[§09]** Add the DART dropout-renormalization formula (its exactness depends on folding into the alphas). **[v2]**
- **[§06/§07]** Note `feature_weights` is expressible through the existing §07 `AdmissionPrior` soft-prior seam (low priority). **[v1.5]**
