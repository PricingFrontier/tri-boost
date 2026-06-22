# pattern-boost — SPEC §09: Predictiveness boosters

> 2026-06-21. Conforms to `spec/00-spec-skeleton.md` (§1 engineering standards, §2 shared types, §3 invariant contract, §4 ownership). This section OWNS: teacher-distillation training mode; fully-corrective leaf refit; Nesterov/accelerated boosting; bagged greedy ensemble selection (and its outer-bag table-average on-ramp); the residual optional knobs (DART, `random_strength`). It USES `Model.trees` alphas (§06), the `Loss` trait (§05), purification linearity (§08), and the firewall (§3). Single quotes mark inline Rust identifiers.

## 09 — Predictiveness boosters

Where the gap-closing playbook is cashed in. The v1 spine (§05–§08) reaches "beat EBM, near-parity on most data"; the boosters here chase the unconstrained GBMs — distillation leads (highest-upside *new* lever), ensembling closes (the last variance slice). **The governing constraint is uniform: every booster touches only the loss target, the leaf scalars, the tree weights, or a convex average of banks — never tree shape.** So I1 and I2 hold *by construction* and every booster stays `ExactnessMode::Exact`; the firewall (§3) enumerates these five as exactness-preserving precisely because none can inflate interaction order. Honest disclaimer, meant: **none lift the order-3 *bias* ceiling** — they recover variance and search slack, not genuine ≥4-way structure.

All boosters are **default-off**; the default is the single best-tuned model. Fixed pipeline order: *distill (target) → fit → refit leaves → ensemble-average banks → purify → tables*.

### 09.1 Configuration surface

These extend the §06 config struct (owned there; the fields below are registered here per §2/§4). All default to the inert setting.

```rust
/// Predictiveness-booster knobs. Defaults are all-off / identity.
#[derive(Clone, Debug)]
pub struct BoosterConfig {
    pub refit_leaves: RefitSpec,             // RefitSpec::Off by default
    pub nesterov: NesterovSpec,              // NesterovSpec::Off by default
    pub ensemble: EnsembleSpec,              // EnsembleSpec::Off by default
    pub dart: Option<DartSpec>,              // None = plain MART
    pub random_strength: f32,                // 0.0 = deterministic split scores
    pub reanchor: bool,                      // false = no global-mean bias fold (09.6)
}
```

**Distillation is NOT a `BoosterConfig` knob — it is a `FitSpec` field.** Whether to distil lives in `FitSpec.distill: Option<DistillSpec>` (skeleton §2.9), alongside `weight` and `exposure`, because — like those — it carries **per-row data** (the `teacher_raw` vector is one score per training row) that must be supplied at `fit` time, not baked into a reusable config. `BoosterConfig` holds only the per-row-data-free booster knobs; the per-row distillation target is threaded through `FitSpec` (09.2). The other boosters here (refit, Nesterov, ensemble, DART, `random_strength`, `reanchor`) carry no per-row data and so stay in `BoosterConfig`.

### 09.2 Teacher-distillation training mode

**Decision.** Ship a distillation training mode (v1.5, default off): the booster fits against a teacher's soft score blended with true labels. The distillation target lives in **`FitSpec.distill: Option<DistillSpec>`** (skeleton §2.9), NOT in `BoosterConfig` — it is a per-row-data field (the per-row `teacher_raw` scores), so it belongs with `weight`/`exposure` at `fit` time. **Teacher default is CatBoost** — itself an oblivious-tree ensemble, so distilling takes the ≤3rd-order projection of the *same tree family*, and it carries best-in-class ordered-TS categorical signal matching our Fisher-TS axes (§04). The teacher is consumed data-side only (pattern-boost never links CatBoost; the user, or the optional `distill` Python helper §12, supplies per-row soft scores).

**What changes — only the target.** A `Loss`-target substitution; never the split-finder, histogram engine, or tree shape. Given teacher raw scores `t` (in *our* score space `F`, pre-link — caller aligns the link) and labels `y`, each iteration fits gradients against a blended target

```text
ĝ_i, ĥ_i  =  loss.grad_hess on the BLENDED objective
          =  blend · grad_hess(y_i, F_i) + (1−blend) · grad_hess(t_i, F_i)
```

where `blend ∈ [0,1]` is the true-label weight (`soft_weight = 1−blend`). The "soft" term is the **base `grad_hess` called with `teacher_raw` as the target** (no separate trait method): squared error `(F_i − t_i)`; Logistic a cross-entropy to the soft label `σ(t_i)`; Poisson/Gamma/Tweedie the deviance gradient toward `pred_from_raw(t_i)`. This is the §05 `BlendedLoss` adaptor, same name/polarity (`g = blend·g_true + (1−blend)·g_soft`). Because the trait is fallible (`Loss::grad_hess(..) -> Result<(), PbError>`, §2.4), each of the two passes is `?`-propagated and the blend itself is infallible. Blending in *gradient* space composes with exposure offsets and weights with zero special-casing.

```rust
/// Teacher-distillation target. Teacher scores are pre-supplied (data-side).
#[derive(Clone, Debug)]
pub struct DistillSpec {
    /// Teacher raw scores in OUR score space F (pre-link), one per training row.
    pub teacher_raw: Vec<f32>,
    /// Soft+true blend: target = blend·true + (1−blend)·soft. Default 0.5
    /// (balanced). Meaningful only when this DistillSpec is present (= ON);
    /// blend=1.0 is the degenerate zero-teacher case (reproduces the
    /// non-distilled fit, kept as a test oracle), NOT the off switch —
    /// disabling is FitSpec.distill = None, which skips teacher training.
    pub blend: f32,
    pub teacher: TeacherKind,                 // provenance tag, for the model card
}
pub enum TeacherKind { CatBoost, LightGbm, XgBoost, Other(String) }
```

**Two parameters, off by default.** *Whether* to distil = `FitSpec.distill: Option<DistillSpec>` — `None` (the default) fits no teacher and trains on true labels; the teacher (the §12 Python helper that fits CatBoost) is invoked **only** when `distill.is_some()`. *How much* = `DistillSpec.blend`, the true-label weight, **default `0.5` (balanced)** — clamp to `[0,1]`, reject NaN with `PbError::InvalidConfig`. `blend == 1.0` is the degenerate zero-teacher case (it reproduces the non-distilled fit bit-for-bit, kept as an inert-default test oracle), *not* the disable switch — disabling is `distill = None`, which skips teacher training entirely.

**Exactness.** The distilled model is bit-exact to its own tables — distillation changes *what F approximates*, never *how F decomposes*; it stays `Exact` and exports tables normally. Its *fidelity to the teacher* is <1 (the student matches the teacher's ≤3rd-order projection exactly; genuine ≥4-way teacher structure is irreducible). Teacher provenance is stamped on the `Model` card; the teacher never enters inference. **Firewall note:** distillation only ever fits *toward* the teacher; *adding* a >3-order teacher as a base-margin is the banned path (§3) and is not this path.

**Cost.** One extra O(n) gradient pass per iteration (cheap), plus the one-time external teacher fit (caller-borne). **Serves:** accuracy (headline new lever); decomposable (untouched); fast (negligible overhead).

### 09.3 Fully-corrective leaf refit

**Decision.** Offer fully-corrective leaf refit (v2, default off, benchmark-gated): with all tree *structures* frozen, jointly re-optimize *all* leaf values against the regularized loss. Corrects stagewise over-shrinkage, yielding fewer trees at equal accuracy → smaller tables (a double win).

**Algorithm.** Structure frozen ⇒ the model is **linear in the leaf values**. For `T` trees, each row `x_i` activates one of 8 leaves per tree, so its design row is `T` one-hot blocks of width 8 — a sparse `z_i ∈ {0,1}^{8T}` with `T` ones (the 3-bit leaf code per tree, already computed at scoring). Stack into `Z ∈ {0,1}^{n×8T}`; refit is a ridge-regularized IRLS/Newton step:

```text
solve   (ZᵀWZ + λI) θ  =  Zᵀ(W·z_target)          // weighted ridge normal equations
        W = diag(h_i)   (loss hessian at current F)
        z_target_i = F_i − g_i / h_i               (Newton response)
θ  ∈ ℝ^{8T}  are the new leaf values; the f0 intercept stays a separate scalar (§2.6).
```

Iterate (recompute `g,h` at updated `F`, re-solve) for `max_iter` Newton rounds (default 3) with Armijo backtracking (reuse §06's leaf-step backtracker). The `8T×8T` Gram is small and structured; solve by Cholesky of `ZᵀWZ + λI`. **Leaves refit from full-precision g/h** (the quantized histogram is split-finding only; §2.3), so the refit injects no quantization bias.

```rust
#[derive(Clone, Debug)]
pub enum RefitSpec {
    Off,
    Ridge { l2: f32, max_iter: u8, every_k_trees: Option<u32> },
}
fn refit_leaves(model: &mut Model, gh: &GradHess, z: &LeafMembership, spec: &RefitSpec)
    -> Result<(), PbError>;
/// Sparse leaf-membership design: leaf_of[t][i] ∈ 0..8 (already computed at scoring).
pub struct LeafMembership { pub leaf_of: Vec<Vec<u8>>, pub n_trees: usize }
```

`every_k_trees = Some(k)` interleaves a refit every k trees (totally-corrective); `None` refits once at the end (the recommended default). **Exactness.** Only the 8 scalars per tree change; structure, the ≤3-feature property, and constant-cell form are untouched → `Exact` preserved. Compose order **refit → purify** (never the reverse). **Critical guard:** never expose the leaf one-hot `Z` to any external/black-box stage — that reintroduces an opaque model and breaks the "tables ARE the predictor" thesis; the refit is an *internal* re-weighting of our own tables. **Cost** an `8T`-dim solve per Newton round × refit points. **Serves:** accuracy + decomposable (smaller tables) + fast inference (fewer trees).

### 09.4 Nesterov / accelerated boosting

**Decision.** Offer AGBM-style accelerated boosting (v2, default off, benchmark-gated against the lighter Biau AGB first). O(1/m²) function-space convergence directly attacks the named weakness — depth-3 oblivious trees are weak per tree, so plain MART needs many of them; acceleration needs far fewer (an order of magnitude reported), and fewer trees = smaller, more readable tables.

**Algorithm (AGBM, Lu et al. AISTATS'20).** Three function sequences — model `f`, Nesterov combination `g`, momentum `h` — with step `θ_m = 2/(m+2)`. Each iteration fits a tree to the gradient at `g` (not `f`) and a *second* tree to a corrected residual so momentum error does not accumulate under inexact (oblivious) learners:

```text
g_m   = (1 − θ_m)·f_m + θ_m·h_m                       // Nesterov look-ahead point
b_m   = fit_oblivious_tree( −∇L(g_m) )                // primary tree at look-ahead
f_{m+1} = g_m + η·b_m
c_m   = fit_oblivious_tree( corrected_residual )       // momentum-correction tree
h_{m+1} = h_m + (η/θ_m)·c_m
```

Each `b_m, c_m` is a depth-3 oblivious tree; `f`, `g`, `h` are *linear mixes of oblivious trees* → a sum of ≤3-feature tensors (I1/I2 hold). The final `Model.trees` stores the flattened `(alpha, tree)` list with accumulated mixing coefficients folded into the alphas (the `Model` permits non-unit alphas for exactly this, §2.6); inference and accumulation are unchanged.

```rust
#[derive(Clone, Debug)]
pub enum NesterovSpec {
    Off,
    Agbm { momentum_correction: bool },     // true ⇒ two trees/iter (recommended)
}
```

**Exactness.** Linear momentum mixing of oblivious trees → `Exact` preserved; alpha-folding (a closed-form fold into `Model.trees[t].0`) is the only bookkeeping. **Cost** ~2× per iteration but far fewer iterations; needs deviance early stopping (overfits earlier than MART). **Open fork (recommended default OFF):** benchmark `momentum_correction=true` AGBM vs single-tree Biau AGB on pricing/TabArena before promoting either — the fork is whether the 2-trees-per-iter cost is repaid by the iteration-count drop (skeleton §14 logs this as the one open §09 fork). **Serves:** accuracy (fewer-trees parity) + decomposable (smaller tables) + fast.

### 09.5 Bagged greedy ensemble selection

**Decision.** The sanctioned ensemble path (v1.5, default off): train K hyperparameter-diverse models, **bagged greedy forward selection with replacement on held-out deviance** (Caruana 2004), and average the **table banks** in score space on a common merged grid under a single shared `w`. Variance reduction only — it **cannot lift the order-3 bias cap** (state this on every export). Inference stays free: the averaged-and-re-purified bank is *one* table set, LUT-sum scoring independent of K.

**Why it stays exact (airtight).** Purification is linear with `Σα_i = 1` (Lengerich Cor. 2.2; research/03): `purify(Σ α_i F_i) = Σ α_i purify(F_i)`, so "purify-then-average ≡ average-then-purify." Every member is order ≤3 (I1) → any convex combination is a sum of order-≤3 tensors → order ≤3 (I2) holds automatically. This is exactly why a self-ensemble is admissible where stacking a foreign/nonlinear model is not.

**The four exact-mechanics rules (enforced, not optional):**
1. **Common merged grid.** Align all members on the sorted union of every realized threshold per axis (§08's merge rule generalized from one ensemble to K). Each member's piecewise-constant tensor maps *losslessly* onto the finer union grid (a cell constant over [30,55) replicates across sub-cells). Zero approximation.
2. **Union of supports.** The averaged bank holds the union of all realized feature-sets (a member that never learned {i,j,k} contributes a zero tensor). Denser pre-prune; re-purify once and prune by Sobol `σ²(f_u)/σ²(F)` for display — the complete union support stays for lossless inference.
3. **Average in score space `F` (pre-link), never response space.** Averaging `exp(F)`/`σ(F)` is nonlinear and breaks additivity. Apply the link once at the end.
4. **Weights `Σα_i = 1, α_i ≥ 0`, single shared `w`.** Convex combination only (intercept folds cleanly into `f0`); a learned nonlinear meta-weighting is banned (reintroduces order inflation). Do *not* diversify `w` across members — the variance-sum identity branches on `w`; pick one `w` (the `ProductMarginals` default), average, optionally re-export under an alternate `w` post-hoc (exactness-preserving, §08).

**Selection algorithm.** Greedy forward selection *with replacement* on a held-out set, optimizing **held-out deviance/logloss** (strictly proper for Poisson/Gamma/Tweedie — **never** RMSE). With-replacement multiplicities give a *weighted* convex soup that dominates uniform averaging. Two anti-overfit fixes, non-negotiable at large K: (a) **initialize from the top-2 single models** (not empty); (b) **bag the selection** — greedy-select on bootstrap replicates of the held-out set and average the weight vectors. Diversity from hyperparameters first (`max_bin`/border type, `λ`, sub/colsample, `max_interaction_order ∈ {2,3}`, LR×n_trees, `random_strength`), then seeds/subsamples. **K ≈ 8–16**; the selected ensemble is usually 3–8 effective members.

```rust
#[derive(Clone, Debug)]
pub enum EnsembleSpec {
    Off,
    /// Cheapest on-ramp: outer-bag table averaging at ONE hp setting.
    OuterBag { n_bags: u16 },                              // default 8
    /// Full recipe: hp-diverse library → bagged greedy selection.
    GreedySelect {
        library_size: u16,                                // K, default 12
        hp_grid: HpGrid,
        selection_bags: u16,                              // default 25
        seed_top_n: u8,                                   // default 2
    },
}
/// Average member banks on the union grid under one shared w (Σα=1, α≥0).
fn average_banks(members: &[(f32, TableBank)], w: &RefMeasure)
    -> Result<TableBank, PbError>;     // re-purifies once; errs on w mismatch
```

`OuterBag { n_bags: 8 }` is the recommended **on-ramp** (ship first): most of the variance win + free per-cell standard-error bands (an annotation, *not* an fANOVA component — "purify each bag, take the spread"), no HP search. Full `GreedySelect` squeezes the last ~0.5–1.5% deviance for users who pay K×.

**Inner-bag smoothing + SE-band annotation (concrete, display-only).** Purify each of the `n_bags` member banks independently onto the common union grid, then for each table cell `u, cell`: the **published value** is the across-bag mean `μ̄ = (1/B) Σ_b f_u^{(b)}(cell)` (this *is* the averaged bank — inner-bag smoothing = mean over bags), and the **SE band** is `se = stddev_b(f_u^{(b)}(cell)) / √B` carried alongside as `Option<f32>` per cell. Type: `SeBands { per_cell: Vec<f32> }` parallel to each `EffectTable.values`, stamped only when `n_bags > 1`. **Status: display-only.** The SE band is *not* an fANOVA component — it never enters `Reconstruction`/`VarianceSum`/inference (LUT-sum scoring reads `values`, never `se`); it is an audit annotation surfaced in exports, so exactness (I2) is untouched.

**`average_banks` enforces** `Σα = 1`, `α ≥ 0`, one shared `w` (errs `PbError::InvalidConfig` on mismatch), and re-purifies the union bank exactly once. **Cost:** K× training (binding); union/selection/re-purify cheap and parallel; **inference free** (one bank, K-independent). **Honesty gate:** the legitimate "beat EBM" comparison is bagged-vs-bagged (EBM bags 14× internally) — documented, not code-enforced. **Serves:** accuracy (variance + credibility bands) + decomposable (one exact bank) + fast.

### 09.6 Residual optional knobs

All preserve exactness, all default off, all situational (benchmark-gated, none milestone-critical).

- **Global mean / bias re-anchoring (`reanchor`, [v1]).** A single exact calibration-adjacent scalar fold: after fitting, compute `δ = log(Σ w·y / Σ w·μ̂)` (the log-ratio of weighted total observed to weighted total predicted on the response scale; the identity-link form is `δ = Σ w·y − Σ w·μ̂` over `Σ w`) and fold it into the intercept, `f0 ← f0 + δ`. This re-anchors the model's base level to the observed weighted mean (base level = 1.000 on log-link, §05) without touching any tree, leaf, or table — purely a shift of the `f0` scalar (§2.6). Because only `f0` moves, accumulation (§08) and every Invariant check are unaffected → `Exact` preserved. `reanchor: bool` (default `false`).

- **DART (tree dropout).** Drops a random subset of built trees per round, then renormalizes (dropped + new trees rescaled to preserve total contribution). Each tree stays an independent depth-3 oblivious table → exact. The per-round renormalization bakes **non-unit scale weights** into trees; these fold cleanly into `Model.trees[t].0` (a closed-form fold) so table accumulation stays clean. Slows training (no running-sum buffer) and destabilizes early stopping. `DartSpec { drop_rate, normalize: bool }`. **Renormalization formula.** Let `D` be the set of trees dropped this round (`k = |D|`) and `b` the newly-fit tree. To preserve the total contribution after re-adding the dropped trees plus the new one, scale the new tree by `1/(k+1)` and each dropped tree by `k/(k+1)` (the standard DART normalization): `α_b ← η/(k+1)`, and `α_d ← α_d · k/(k+1)` for `d ∈ D`. Both are closed-form folds into the existing `Model.trees[t].0` alphas, so accumulation (§08) sees only re-weighted oblivious trees — exactness is untouched (the fold is the only bookkeeping).
- **`random_strength`.** Decaying noise added to split scores to break ties among near-equal splits (structural regularizer). `random_strength: f32` (0.0 = deterministic, the default); the noise is drawn from the **deterministically re-seeded** PRNG (§1) — `Pcg64::seed_from_u64(splitmix64_mix(base, round, stage, block))` per work unit (NOT a "splittable" PRNG, which is unimplementable as named) — so bit-reproducibility is preserved: the same seed gives the same noise sequence at any thread count.

### 09.7 Testing approach

Per §1/§13, every booster carries: (1) **a decomposition-safety property test** — after the booster, the five Invariant checks (§3: Reconstruction, MassConservation, Purity, VarianceSum, ThreeWayEqual) still pass, proving `Exact` survives; (2) **a bit-reproducibility test** — bit-identical training at `n_threads ∈ {1,2,8}` (the determinism [GATE]). Booster-specific oracles: **distillation** — `blend == 1.0` reproduces the non-distilled model bit-for-bit (inert-default check), and a synthetic order-3 teacher is matched to float tolerance (the ≤3rd-order-projection property). **Refit** — on a single tree the IRLS solve equals the direct Newton leaf optimum; refitting an optimal ensemble is a near-no-op; `proptest` that `Z θ` reconstructs the leaf-lookup scores. **Nesterov** — the alpha-folded `Model.trees` scores bit-identically to the three-sequence accumulation; on a quadratic loss it matches the closed-form accelerated trajectory. **Ensemble** — `average_banks` satisfies purify-then-average ≡ average-then-purify to tolerance (the linearity `proptest`); union-grid mapping is lossless (member score == its image at one interior point per cell); `Σα`/`α ≥ 0`/`w`-mismatch each raise the typed error. **DART** — the non-unit-weight fold reproduces the pre-fold score exactly, and the `1/(k+1)` / `k·(k+1)⁻¹` renormalization preserves total contribution to tolerance. **Re-anchoring** — after `f0 ← f0 + δ` the weighted total prediction equals the weighted total observed (`Σ w·μ̂ == Σ w·y` to tolerance) and all five Invariant checks still pass (only `f0` moved). **SE bands** — the published averaged value equals the across-bag mean per cell, the `se` annotation is excluded from `Reconstruction`/`VarianceSum` and from inference (scoring a model with and without the `se` annotation gives bit-identical predictions).

### 09.8 Build order & open forks

**v1.5:** distillation (09.2), ensemble on-ramp `OuterBag` + full `GreedySelect` (09.5). **v2:** fully-corrective refit (09.3), Nesterov/AGBM (09.4), DART/`random_strength` (09.6). **The one genuinely-open fork:** whether fully-corrective refit and/or AGBM should ever flip default-on — both are exact and both shrink tree count, but the solve/2×-iter cost must be measured on pricing data first. **Recommended default: both off**, promoted only behind a benchmark showing tree-count reduction repays the per-round cost (logged in skeleton §14).
