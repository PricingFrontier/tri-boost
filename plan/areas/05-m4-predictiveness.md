# Milestone 4 — Predictiveness Boosters (§09) — Implementation Plan

## How this area's structure makes quality automatic

These are the accuracy levers, and every one of them is a trap for exactness — the temptation is to bolt on a non-linear calibration, a foreign teacher margin, or a learned meta-weight, all of which silently inflate interaction order and break I2. The structural defense is that **§09 owns no tree-shape decision**: every booster touches only the *loss target*, the *leaf scalars*, the *tree alphas*, or a *convex average of banks*. So I built the sequencing around two hard rules. First, **nothing in this milestone may merge until the five Invariant gates (§13.1) and the determinism gate (§13.4) — already build-blocking from the green spine (G0–G3, §14.2) — re-pass with the booster ON.** Each task's Definition-of-Done is "those gates green with the lever enabled," not "code written." This is the §09.7 decomposition-safety property test, made the literal merge condition. Second, every booster ships **with its inert-default oracle written first**: `blend=1.0` reproduces the non-distilled fit bit-for-bit, `RefitSpec::Off` / `NesterovSpec::Off` / `EnsembleSpec::Off` are identity, so the first thing each PR proves is that the default-off path changes nothing — the firewall (§3, `ExactnessMode`) and the gate are one wall.

I deliberately front-load the two **leverage primitives** the other tasks depend on: the §08 purification-linearity proptest (`purify(Σαf) = Σα·purify(f)`, Lengerich Cor 2.2) and the `wht8` Faith-Shap cross-check oracle (§13.7a). The linearity proptest is the *license* for ensemble averaging and refit; the wht8 oracle is an independent second route to the order-3 coefficient. Both are tests, both gate everything downstream, so the math identity is verified before any code leans on it. These extend a model that is already exactly decomposable (the green spine, §14.2) — I reference G3's five checks and the §13.4 determinism gate rather than re-specifying them.

Build order follows §09.8 / §14: **v1.5 = M4-1…M4-7 (distillation + ensemble)**, then **v2 = M4-8…M4-12 (refit, Nesterov, MVS-gate, DART/knobs)**. The one genuinely-open fork (refit/AGBM default-on, §14.6.1) stays off, decided only by a benchmark task at the end.

---

## Tasks

### M4-0 — Purification-linearity proptest (the leverage primitive)
- **Spec:** §08 (Cor 2.2), §13.3 purification identities, §09.5.
- **Deliverable:** `proptest` suite in `tests/invariants/` asserting on random small banks (random order-≤3 tensors, random merged grid, random positive `w`): idempotence, mass-conservation, **linearity** `purify(αA+βB) == α·purify(A)+β·purify(B)` with `α+β=1`, and permutation-invariance. No production code — this verifies the §08 primitive the ensemble/refit/Nesterov tasks rely on.
- **Depends on:** G3 (§14.2 P3 — purification + five gates green).
- **DoD:** the four `proptest` properties green under `ProductMarginals` and `Uniform` (`ProptestConfig{cases:256}`, frozen seed, §13.3); `cargo clippy -D warnings`, `fmt --check`, doctests green.
- **Size:** S

### M4-1 — `BlendedLoss` soft-target adaptor + blend-polarity gate
- **Spec:** §05.7, §09.2.
- **Deliverable:** `BlendedLoss { base: &dyn Loss, soft_target: &[f32], blend: f32 }` in §05's module (the §09 path *uses* it): `grad_hess` calls `base.grad_hess` twice (once on true `y`, once on `teacher_raw`) and combines in fixed order `g = blend·g_true + (1−blend)·g_soft`; `deviance`/`init_score` delegate to `base` on true `y`; `?`-propagates both passes. Clamp `blend∈[0,1]`, reject NaN → `PbError::InvalidConfig`.
- **Depends on:** v1 loss set (§14.2 P4 / G4).
- **DoD:** §13.3 loss test green — **`blend=1.0` reproduces base `grad_hess` bit-for-bit**, `blend=0.0` is pure-soft; finite-difference g/h match (1e-3 rel); NaN/out-of-range blend → correct `PbError` variant (§13.8 per-fn variant test); no-panic lints green.
- **Size:** S

### M4-2 — `DistillSpec` plumbing + distilled-fit decomposition-safety gate
- **Spec:** §09.2, skeleton §2.9 (`FitSpec.distill`), §06 (engine consumes `DistillSpec`).
- **Deliverable:** `DistillSpec { teacher_raw: Vec<f32>, blend, teacher: TeacherKind }` + `enum TeacherKind`; wire `FitSpec.distill: Option<DistillSpec>` through §06 `fit` so that when `Some`, each iteration's `grad_hess` is the `BlendedLoss` over `teacher_raw`; `None` is the unchanged true-label path. Stamp `teacher` provenance on `Model.schema`. Length-check `teacher_raw.len()==n_rows` → `PbError::ShapeMismatch`.
- **Depends on:** M4-1.
- **DoD:** **all five Invariant gates (§13.1) + determinism gate `{1,2,8}` (§13.4) green on a distilled model**; model exports `Exact` tables (firewall stays `Exact`, §3); inert oracle: `distill=None` ≡ `blend=1.0` byte-identical to non-distilled fit (§09.7); synthetic order-3 teacher matched to float tolerance (≤3rd-order-projection property, §09.7); `ShapeMismatch` variant test.
- **Size:** M

### M4-3 — CatBoost teacher helper (Python, data-side only)
- **Spec:** §09.2, §12 (R-DISTILL), feature-flag `distill` (skeleton §1).
- **Deliverable:** Python-side `distill` helper in `python/`: fits CatBoost, returns `(teacher_raw in our score-space F, blend)` for `TriBoost*.fit(..., distill=)` / `teacher_raw=`. Behind the `distill` cargo/extra feature; **tri-boost-core never links CatBoost**. Caller-aligned link documented.
- **Depends on:** M4-2; v1 Python bindings (§14.2 P7 / G7).
- **DoD:** Python CI (§13.10) green with and without the `distill` extra; default build (no `distill`) compiles and passes all gates (feature-flag matrix, §1); a direct-vs-distilled **A/B** smoke (pytest) shows the distilled model is still `Exact` and round-trips; no pyo3/CatBoost dep leaks into core (grep gate).
- **Size:** M

### M4-4 — `average_banks` + outer-bag on-ramp (`EnsembleSpec::OuterBag`)
- **Spec:** §09.5 (4 exact-mechanics rules), §08.2 (union grid), §13.3 linearity.
- **Deliverable:** `EnsembleSpec` enum (`Off` default, `OuterBag{n_bags:u16}`); `fn average_banks(members: &[(f32, TableBank)], w: &RefMeasure) -> Result<TableBank, PbError>` — common merged grid (sorted union per axis), union of supports (zero tensor for absent `u`), average in **score space**, **re-purify once**; enforce `Σα=1`, `α≥0`, single shared `w` (else `PbError::InvalidConfig`). `OuterBag` trains `n_bags` at one HP setting (seeded distinct subsamples) and averages.
- **Depends on:** M4-0.
- **DoD:** **five Invariant gates + determinism gate green on the averaged bank**; `purify(Σαf)=Σα·purify(f)` proptest (M4-0) covers the math; union-grid mapping lossless (member score == its image at one interior point per cell, §09.7); `Σα≠1` / `α<0` / `w`-mismatch each raise the **correct typed `PbError`** (§13.8); table-budget stress on the bagged union grid (§14.3 release gate) — no silent inflation.
- **Size:** L

### M4-5 — SE-band annotation (display-only)
- **Spec:** §09.5 (SE bands), §08 (`SeBand`), §13.1 (negative gate).
- **Deliverable:** `SeBand { per_cell: Tensor }` on `EffectTable` (the §08 type), populated by `OuterBag` (`n_bags>1`) as across-bag stddev/√B; `None` for single-fit. Published cell value = across-bag mean (== the averaged bank). Surfaced in §10 export only.
- **Depends on:** M4-4.
- **DoD:** **negative gate (§13.1)** — `se` excluded from Reconstruction/VarianceSum/Purity/ThreeWayEqual and from inference: scoring a model with vs without the `se` annotation is **bit-identical** (§09.7); published value == across-bag mean per cell (unit test); `se=None` for single-fit.
- **Size:** S

### M4-6 — Bagged greedy ensemble selection (`EnsembleSpec::GreedySelect`)
- **Spec:** §09.5 (Caruana selection), §14.3.
- **Deliverable:** `GreedySelect{ library_size, hp_grid: HpGrid, selection_bags, seed_top_n }`; `HpGrid`; train K HP-diverse models (max_bin/λ/sub-colsample/`max_interaction_order∈{2,3}`/LR×n_trees/`random_strength`), **greedy forward selection with replacement on held-out deviance** (never RMSE), **seed from top-`seed_top_n`**, **bag the selection** over `selection_bags` bootstrap replicates, average weight vectors → convex `(α, member)` soup → `average_banks`.
- **Depends on:** M4-4.
- **DoD:** **five Invariant gates + determinism gate green** on the selected ensemble; selection is seeded/reproducible (`{1,2,8}`-thread byte-equal model); deviance (not RMSE) optimized — unit test on Poisson fixture; table-budget stress (§14.3); honesty note (bagged-vs-bagged) documented on export.
- **Size:** L

### M4-7 — `BoosterConfig` registration + pipeline-order wiring
- **Spec:** §09.1 (config surface), §09 intro (fixed pipeline order), §06 (`Config` extension).
- **Deliverable:** Register `BoosterConfig { refit_leaves, nesterov, ensemble, dart, random_strength, reanchor }` (all inert defaults) onto §06's `Config`; enforce the **fixed pipeline order** *distill→fit→refit→ensemble-average→purify→tables*; validate once in `fit` → `PbError::InvalidConfig`.
- **Depends on:** M4-2, M4-4.
- **DoD:** all-inert default config produces a model **byte-identical** to a no-booster fit (determinism/oracle); each knob's invalid value → correct `PbError` variant (§13.8); five gates green with all-off; `fmt`/`clippy -D warnings`/MSRV/`cargo deny` green.
- **Size:** S

### M4-8 — `reanchor` (v1-eligible, exactness-trivial)
- **Spec:** §09.6 (re-anchoring), §13 (re-anchor test).
- **Deliverable:** `reanchor: bool`; after fit compute `δ` (log-ratio of weighted observed/predicted; identity-link variant) and fold `f0 ← f0 + δ`. Only the scalar moves.
- **Depends on:** M4-7.
- **DoD:** after fold, `Σw·μ̂ == Σw·y` to tolerance; **all five Invariant gates still pass** (only `f0` moved, §09.7); determinism gate green.
- **Size:** S

### M4-9 — Fully-corrective leaf refit (`RefitSpec::Ridge`)
- **Spec:** §09.3, §06.4 (Armijo backtracker), §2.3 (full-precision g/h).
- **Deliverable:** `RefitSpec::{Off, Ridge{l2,max_iter,every_k_trees}}`; `LeafMembership{leaf_of,n_trees}`; `fn refit_leaves(model, gh, z, spec)` — structure frozen ⇒ ridge IRLS over `#trees×8`: solve `(ZᵀWZ+λI)θ = Zᵀ(W·z_target)` by Cholesky, Newton response `z_target=F−g/h`, iterate `max_iter` with **Armijo** (reuse §06). Refit from **full-precision** g/h. **Order: refit→purify.** Guard: never expose `Z` externally.
- **Depends on:** M4-7.
- **DoD:** **five Invariant gates + determinism gate green** post-refit (only 8 scalars/tree change → `Exact`, §3); single-tree IRLS == direct Newton leaf optimum (unit test); refitting an optimal ensemble is a near-no-op; `proptest` that `Zθ` reconstructs leaf-lookup scores (§09.7); `RefitSpec::Off` byte-identical oracle.
- **Size:** L

### M4-10 — Nesterov / accelerated boosting (`NesterovSpec::Agbm`)
- **Spec:** §09.4, §06 (`Accel`, `Model.trees` alphas).
- **Deliverable:** `NesterovSpec::{Off, Agbm{momentum_correction}}`; AGBM three-sequence loop (`f,g,h`, `θ_m=2/(m+2)`), primary tree at look-ahead `g`, optional momentum-correction tree; **alpha-fold** the mixing coefficients into `Model.trees[t].0` (closed-form) so scoring is unchanged; deviance early stopping.
- **Depends on:** M4-7.
- **DoD:** **five Invariant gates + determinism gate green**; alpha-folded `Model.trees` scores **bit-identically** to the three-sequence accumulation (§09.7); on quadratic loss matches closed-form accelerated trajectory; `NesterovSpec::Off` byte-identical oracle.
- **Size:** L

### M4-11 — MVS decomposition-safety gate + DART/`random_strength` knobs
- **Spec:** §09.6 (DART, `random_strength`), §06.5 (MVS sampler — owned by §06).
- **Deliverable:** §09-side **decomposition-safety + determinism gate** for the §06-owned MVS sampler (booster-facing test, not the sampler). `DartSpec{drop_rate,normalize}` with the `1/(k+1)` / `k·(k+1)⁻¹` renormalization folded into alphas; `random_strength: f32` noise from the deterministically re-seeded `Pcg64` (§1).
- **Depends on:** M4-7; §06 MVS (§14.3).
- **DoD:** **five gates + determinism gate green under MVS, DART, and `random_strength`**; DART non-unit-weight fold reproduces pre-fold score exactly and renormalization preserves total contribution to tolerance (§09.7); `random_strength` noise bit-reproducible across `{1,2,8}` threads (§13.4); all default-off oracles byte-identical.
- **Size:** M

### M4-12 — `wht8` ↔ purification Faith-Shap cross-check oracle
- **Spec:** §13.7a, §07 (`wht8`), §08.5 (Faith-Shap), §09 intro.
- **Deliverable:** `fn assert_wht8_triple_matches_purified(tree, w, tol)` — `proptest` over random depth-3 leaf vectors and random positive per-cut `w`-marginals asserting the `wht8` order-3 coefficient `c_123` equals the per-tree order-3 Faith-Shap from §08's mass-moving path, to a derived `wht8_tol`; all eight coefficients checked. **Single-tree only** (coefficients never summed across trees).
- **Depends on:** M4-0; §07 `wht8` (§14.2 P5).
- **DoD:** oracle **[GATE]** green under `ProductMarginals`/`Uniform`; `Joint` arm **[CHECK]** (looser tol, non-blocking) per §13.7a; negative property — no `wht8` accumulator element is read by `assert_exact_decomposition` (§13.1).
- **Size:** M

### M4-13 — Benchmark fork resolution (refit / AGBM default-on?)
- **Spec:** §09.8, §14.6.1 (the one open fork), §13.7 (accuracy harness).
- **Deliverable:** `xtask accuracy` runs (dev-only, ships nothing) comparing refit-on/off and AGBM(`momentum_correction` true/false) vs Biau-AGB on pricing/TabArena fixtures — tree-count reduction vs per-round cost. Log result in §14 fork register. **Recommended default: both off** unless the benchmark proves repayment.
- **Depends on:** M4-9, M4-10.
- **DoD:** deviance/logloss (strictly-proper, never RMSE) recorded; **all candidate configs still pass the five gates + determinism gate**; fork decision logged; defaults unchanged unless benchmark-justified (§14.6.1).
- **Size:** M

---

**Sequencing rationale:** M4-0 (linearity proptest) and the inert-oracle discipline gate everything — no booster builds on an unverified identity. v1.5 ships M4-1…M4-7 (distillation, ensemble on-ramp, full selection, config wiring) + M4-8 (`reanchor`, trivially exact). v2 ships M4-9…M4-11 (refit, Nesterov, MVS-gate/DART/knobs), M4-12 (the cross-check oracle, which can land any time after M4-0 + §07 `wht8`), and M4-13 resolves the lone open default-on fork last, behind a benchmark. Every task's "done" is gates-green-with-the-lever-on, so exactness and reproducibility are a byproduct of the merge condition, never an audit.
