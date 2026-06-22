# Milestone 4 — Predictiveness Boosters (§09) — Implementation Plan

## How this area's structure makes quality automatic

These are the accuracy levers, and every one of them is a trap for exactness — the temptation is to bolt on a non-linear calibration or a learned meta-weight, both of which can silently inflate interaction order and break I2. The structural defense is that **§09 owns no tree-shape decision**: every booster touches only the *leaf scalars*, the *tree alphas*, or a *convex average of banks*. So I built the sequencing around two hard rules. First, **nothing in this milestone may merge until the five Invariant gates (§13.1) and the determinism gate (§13.4) — already build-blocking from the green spine (G0–G3, §14.2) — re-pass with the booster ON.** Each task's Definition-of-Done is "those gates green with the lever enabled," not "code written." This is the §09.7 decomposition-safety property test, made the literal merge condition. Second, every booster ships **with its inert-default oracle written first**: `RefitSpec::Off` / `NesterovSpec::Off` / `EnsembleSpec::Off` are identity, so the first thing each PR proves is that the default-off path changes nothing — the firewall (§3, `ExactnessMode`) and the gate are one wall.

I deliberately front-load the two **leverage primitives** the other tasks depend on: the §08 purification-linearity proptest (`purify(Σαf) = Σα·purify(f)`, Lengerich Cor 2.2) and the `wht8` Faith-Shap cross-check oracle (§13.7a). The linearity proptest is the *license* for ensemble averaging and refit; the wht8 oracle is an independent second route to the order-3 coefficient. The non-trivial implementation this area leans on is owned elsewhere and only **referenced** here, never re-implemented: the `wht8` Walsh–Hadamard accumulator is owned by M2 (M2-J, §07.4a), and M4-12 *uses* M2-J's `wht8`. The math identity is verified before any code leans on it. These extend a model that is already exactly decomposable (the green spine, §14.2) — I reference G3's five checks and the §13.4 determinism gate rather than re-specifying them.

Build order follows §09.8 / §14: **v1.5 = M4-0 + M4-4…M4-7 (ensemble)**, then **v2 = M4-8…M4-10 + M4-13 (refit, Nesterov, the open default-on fork)**. The §06-owned row sampler that dominates GOSS — **Minimal Variance Sampling — is a v1.5 lever, implemented in M5 (M5-MVS)**; M4-11 is the §09-side decomposition-safety *gate* pointing at that implementation (plus the v2 DART/`random_strength` knobs). The one genuinely-open fork (refit/AGBM default-on, §14.6.1) stays off, decided only by a benchmark task at the end.

---

## Tasks

### M4-0 — Purification-linearity proptest (the leverage primitive)
- **Spec:** §08 (Cor 2.2), §13.3 purification identities, §09.5.
- **Deliverable:** `proptest` suite in `tests/invariants/` asserting on random small banks (random order-≤3 tensors, random merged grid, random positive `w`): idempotence, mass-conservation, **linearity** `purify(αA+βB) == α·purify(A)+β·purify(B)` with `α+β=1`, and permutation-invariance. No production code — this verifies the §08 primitive the ensemble/refit/Nesterov tasks rely on.
- **Depends on:** G3 (§14.2 P3 — purification + five gates green).
- **DoD:** the four `proptest` properties green under `ProductMarginals` and `Uniform` (`ProptestConfig{cases:256}`, frozen seed, §13.3); `cargo clippy -D warnings`, `fmt --check`, doctests green.
- **Size:** S

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
- **Deliverable:** Register `BoosterConfig { refit_leaves, nesterov, ensemble, dart, random_strength, reanchor }` (all inert defaults) onto §06's `Config`; enforce the **fixed pipeline order** *fit→refit→ensemble-average→purify→tables*; validate once in `fit` → `PbError::InvalidConfig`.
- **Depends on:** M4-4.
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

### M4-11 — MVS decomposition-safety gate (points at M5-MVS) + DART/`random_strength` knobs
- **Spec:** §09.6 (DART, `random_strength`), §06 (Minimal Variance Sampling — the `Sampling::Mvs` engine path **implemented in M5-MVS, v1.5**).
- **Deliverable:** the §09-side **decomposition-safety + determinism gate** for the **M5-MVS** sampler — a booster-facing test that verifies the M5 `Sampling::Mvs` path **preserves exactness and determinism** (the sampler itself is NOT implemented here, and is NOT a v2 item; it is the v1.5 M5-MVS task). Plus the v2 knobs: `DartSpec{drop_rate,normalize}` with the `1/(k+1)` / `k·(k+1)⁻¹` renormalization folded into alphas; `random_strength: f32` noise from the deterministically re-seeded `Pcg64` (§1).
- **Depends on:** M4-7; **M5-MVS** (the v1.5 `Sampling::Mvs` implementation this gate points at).
- **DoD:** **five gates + determinism gate green under MVS (M5-MVS), DART, and `random_strength`**; DART non-unit-weight fold reproduces pre-fold score exactly and renormalization preserves total contribution to tolerance (§09.7); `random_strength` noise bit-reproducible across `{1,2,8}` threads (§13.4); all default-off oracles byte-identical.
- **Size:** M

### M4-12 — `wht8` ↔ purification Faith-Shap cross-check ORACLE test (over M2's `wht8`)
- **Spec:** §13.7a, §07.4a (`wht8`, owned by M2-J), §08.5 (Faith-Shap), §09 intro.
- **Deliverable:** **the cross-check oracle TEST only — references, does not re-implement, the `wht8` Walsh–Hadamard/Möbius transform owned by M2 (M2-J, §07.4a).** `fn assert_wht8_triple_matches_purified(tree, w, tol)` — a `proptest` over random depth-3 leaf vectors and random positive per-cut `w`-marginals asserting M2-J's `wht8` order-3 coefficient `c_123` equals the per-tree order-3 Faith-Shap from §08's mass-moving path, to a derived `wht8_tol`; all eight coefficients checked. **Single-tree only** (coefficients never summed across trees). No accumulator code lives here; this is the independent second route that gates the order-3 coefficient M2-J produces.
- **Depends on:** M4-0; **M2-J** (`wht8`, §14.2 P5).
- **DoD:** oracle **[GATE]** green under `ProductMarginals`/`Uniform`; `Joint` arm **[CHECK]** (looser tol, non-blocking) per §13.7a; negative property — no `wht8` accumulator element is read by `assert_exact_decomposition` (§13.1).
- **Size:** M

### M4-13 — Benchmark fork resolution (refit / AGBM default-on?)
- **Spec:** §09.8, §14.6.1 (the one open fork), §13.7 (accuracy harness).
- **Deliverable:** `xtask accuracy` runs (dev-only, ships nothing) comparing refit-on/off and AGBM(`momentum_correction` true/false) vs Biau-AGB on pricing/TabArena fixtures — tree-count reduction vs per-round cost. Log result in §14 fork register. **Recommended default: both off** unless the benchmark proves repayment.
- **Depends on:** M4-9, M4-10.
- **DoD:** deviance/logloss (strictly-proper, never RMSE) recorded; **all candidate configs still pass the five gates + determinism gate**; fork decision logged; defaults unchanged unless benchmark-justified (§14.6.1).
- **Size:** M

---

**Sequencing rationale:** M4-0 (linearity proptest) and the inert-oracle discipline gate everything — no booster builds on an unverified identity. v1.5 ships M4-4…M4-7 (ensemble on-ramp, full selection, config wiring) + M4-8 (`reanchor`, trivially exact). **Minimal Variance Sampling is a v1.5 lever, implemented in M5 (M5-MVS); it is not pushed to v2.** v2 ships M4-9, M4-10 (refit, Nesterov), the v2 DART/`random_strength` knobs (the rest of M4-11), and M4-13 resolves the lone open default-on fork last, behind a benchmark. M4-11's MVS-side work is the §09 *decomposition-safety gate* pointing at M5-MVS (it can land once M5-MVS exists); M4-12 (the cross-check oracle over M2-J's `wht8`) can land any time after M4-0 + M2-J. Every task's "done" is gates-green-with-the-lever-on, so exactness and reproducibility are a byproduct of the merge condition, never an audit.
