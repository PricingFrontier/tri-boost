# Milestone 2 — Objectives + Constraints (Implementation Plan)

**Area:** §05 (Loss trait + v1 objectives) and §07 (interaction selection + constraints). This extends the **green spine** delivered by the foundation areas: P0 scaffold (frozen §2 types, the `#![forbid(unsafe_code)]`/`#![deny(missing_docs)]` + clippy no-panic lint gate, `cargo deny`/MSRV/fmt/doctests CI — §13.8/§13.9/§13.10), P1 binning (§03), the squared-error oblivious Newton engine (P2/§06, `Config`/`find_oblivious_tree`/the boosting loop/the §6.2 split-finder + §6.4 leaf-estimation hook), and accumulate→purify→TableBank with the five Invariant gates wired (P3/§08). **All gate machinery already exists** — `assert_exact_decomposition` (§13.1), `check_feature_budget` (§13.2), the `{1,2,8}`-thread byte-equality determinism gate (§13.4), the proptest harness with frozen `ProptestConfig` (§13.3). My tasks do not re-specify those; each task's Definition-of-Done *names* the existing gate it must turn (or keep) green.

## How the structure makes quality automatic

Three sequencing choices make high standards a byproduct, not an add-on:

1. **The Loss trait lands behind a no-panic, fallible wall on day one.** The trait is fallible by contract (`grad_hess`/`init_score`/`deviance` all return `Result`, R-LOSSFALLIBLE), so the *first* objective implemented physically cannot emit `NaN`/`±inf`/panic into `f0` or the stop metric — the typed-error domain gate (§13.3 "Loss domain errors [GATE]") rejects any code that tries. The finite-difference proptest (§13.3) ships **with** each objective, so "the math is right" is proven the moment the math is written, never audited later.

2. **Each new objective re-runs the *whole* §13.1 invariant matrix before it can merge.** The matrix is `{losses} × {orders 1,2,3} × {ProductMarginals,Uniform} × {±monotone} × {±early-termination}`. Adding `Logistic` is not "write logistic.rs"; it is "the green spine stays green on a logit link." Because the objective is orthogonal to tree shape (§05.8 — it sees only `(y,F,w)`), the firewall stays `Exact` by construction, but the gate *proves* it per-objective rather than asserting it.

3. **§07 is structurally constrained to be soft.** The one hard gate (`InteractionPolicy`) is a *masking* function; every statistical mechanism (heredity/FAST/`wht8`/Sobol) only down-weights a non-negative gain. The "soft-not-hard" proptests (§13/§07.9) — planted XOR still recovered, policy-forbidden split never selected — make the funnel's softness a checkable property, so a future "optimization" that turns a prior into a gate breaks CI. The `wht8` accumulator is gated to be invisible to the five invariant checks (§13.1 negative-property [GATE]), foreclosing the rejected "`wht8` replaces purification" shortcut at the test level.

Ordering: objectives first (they are the simpler vertical slice and the §13.1 matrix's columns), then the §07 funnel/constraints which *consume* the loss set and ride the §06.4 hook. Nothing builds on an unverified objective.

---

## Tasks

### M2-A — `Loss` trait + `Link`/`Metric`/`ObjectiveTag`, fallible signatures, hot-loop policy
- **§-ref:** §05.1, §05.2; §2.4 canonical types (verbatim); R-LOSSFALLIBLE; §05.4 hot-loop policy.
- **Deliverable:** `crates/tri-boost-core/src/loss/mod.rs`. The `Loss: Send+Sync` trait verbatim from §2.4/§05.2 (`grad_hess -> Result<(),PbError>`, `init_score -> Result<f64,PbError>`, `deviance -> Result<f32,PbError>`, `link`, `pred_from_raw`, `hessian_floor()=1e-16`, `max_delta_step()=None`, `default_metric`); `enum Link {Identity,Log,Logit}`; `enum Metric {Rmse,LogLoss,PoissonDeviance,GammaDeviance,TweedieDeviance{rho}}`; `LossId`; `ObjectiveTag {link, loss, tweedie_rho}` (R-SCHEMA member). The shared single-entry length-guard helper (returns `ShapeMismatch`) and the `f64` fixed-order `par_chunks` fold helper over fixed `CHUNK_ROWS` (§11) used by `init_score`/`deviance`. `exp_f32` clamp helper (exponent → `[-30,30]`, §05.4). Doc comments with error conditions on every public item.
- **Dependencies:** P0 (PbError, GradHess, FitSpec stubs frozen).
- **DoD:** `cargo clippy --all-targets --all-features -- -D warnings` green (no-panic lints incl. `indexing_slicing`/`unwrap_used`); `cargo fmt --check`; doctests + `#![deny(missing_docs)]` compile-gate green; `cargo deny`/MSRV-1.74 jobs green. The fold helper byte-identical at `n_threads∈{1,2,8}` (the §13.4 mechanism, asserted on the helper). No `Box<dyn Error>` grep gate (§13.8).
- **Size:** S.

### M2-B — `SquaredError` (re-home onto the trait) + the loss test-harness scaffold
- **§-ref:** §05.3 (SquaredError row), §05.9 tests 1–8, §13.3 loss-correctness/domain-error suites.
- **Deliverable:** `SquaredError` implementor (`g=F−y`, `h≡1`, `init_score=p̄`, half-deviance, `default_metric=Rmse`, domain = lengths + `Σw>0`). The reusable per-objective test module: the closed-form unit table, the **finite-difference proptest** (`|g−(L(F+δ)−L(F−δ))/2δ|<tol`, central-difference hessian) parameterized over a `Loss` + its valid-domain `y`-generator, the `init_score` first-order-condition check (`Σw·g(y,f0)≈0`), the deviance-properness check (zero at `μ=y`), and the typed-domain-error harness (`Ok(finite)` in-domain, correct `PbError::InvalidInput`/`ShapeMismatch` variant out-of-domain). The `n∈{0,1}` boundary test for the hot kernel.
- **Dependencies:** M2-A.
- **DoD:** §13.3 **Loss correctness** proptest green for SquaredError; §13.3 **Loss domain errors [GATE]** green (correct variant, no panic/NaN/Inf); §13.5 degenerate-input `is_finite()` sweep green; lint/fmt/doctest gates green. Spine re-confirmation: the existing P2/P3 squared-error path now routes through the trait — G2 (recovers order-≤3, `{1,2,8}`-thread byte-equal) and G3 (five invariant gates) **still green** unchanged.
- **Size:** S.

### M2-C — `Logistic` (Logit link, stable sigmoid/softplus)
- **§-ref:** §05.3 (Logistic row), §05.3a (domain), §05.4 numerics, §05.9 test 6 (saturation `F=±40`).
- **Deliverable:** `Logistic` implementor: branch-stable `σ(F)` (`F≥0 → 1/(1+e^{−F})` else `e^F/(1+e^F)`), softplus deviance form `L=softplus(F)−y·F`, `g=σ(F)−y`, `h=σ(F)(1−σ(F))` floored to `1e-16`, `init_score=log(p̄/(1−p̄))` with `eps_init=1e-12` link-arg clamp, `pred_from_raw=σ`, `default_metric=LogLoss`. Domain (§05.3a): `y∈[0,1]`, `Σw>0`, `0<Σwy<Σw` (else typed `InvalidInput`). `class_labels` plumbed into `ModelSchema` left to §12; provenance here = `ObjectiveTag`.
- **Dependencies:** M2-B.
- **DoD:** §13.3 Loss-correctness + domain-errors gates green for Logistic; §05.9 test 6 (saturated `F=±40`, floored `h`, no NaN) green; **§13.1 invariant matrix green for the `Logistic` column** across orders {1,2,3} × {ProductMarginals,Uniform} (proving I2 holds on a logit link — exactness firewall stays `Exact`); `{1,2,8}`-thread determinism gate green on a Logistic fit. Lint/fmt/doctest gates green.
- **Size:** M.

### M2-D — Exposure-offset intercept path (`init_score` exposure form)
- **§-ref:** §05.5, §05.3a note (a), §03 offset plumbing (`offset=log(e)` is §03-owned; consumed here).
- **Deliverable:** the exposure-weighted intercept branch in the log-link `init_score`: `f0=log(Σwᵢyᵢ / Σwᵢeᵢ)` (accumulate `Σwy`,`Σwe` in `f64`, floor to `eps_init`, return `f64`); typed `InvalidInput` on `Σwe≤0` or an unrescuable non-positive numerator. Confirms `f0` and offset are **additive** (offset does not suppress `f0`). This task lands the *intercept* contract; the engine folds `offset` into `raw` before `grad_hess` (§06, upstream) so the §05.3 (g,h) formulas are unchanged.
- **Dependencies:** M2-A (the fold helper); used by M2-E/M2-F.
- **DoD:** unit test — Poisson exposure fixture anchors base level `e⁰=1.000` (the G4 acceptance criterion); `init_score` first-order condition holds for the exposure-weighted form (§05.9 test 4); typed-error test for `Σwe≤0` (§13.3 domain-error gate); `{1,2,8}`-thread byte-equality of the `f64` `Ok` payload (§05.9 test 7).
- **Size:** S.

### M2-E — `Poisson` + `Gamma` (Log link, `exp(k·F)`, no `powf`)
- **§-ref:** §05.3 (Poisson/Gamma rows), §05.3a, §05.4, §05.6 (Poisson `max_delta_step`), §05.9.
- **Deliverable:** `Poisson` (`g=exp(F)−y`, `h=exp(F)`, `init=log(p̄)`/exposure form, `default_metric=PoissonDeviance`, **`max_delta_step()=Some(0.7)`**) and `Gamma` (`g=1−y·exp(−F)`, `h=y·exp(−F)` floored, strict per-row `y>0` domain check, `default_metric=GammaDeviance`). Shared `mu=exp(F)` reuse; powers via the `exp(k·F)` clamp helper. Domain checks per §05.3a (Poisson `y≥0`/exposure; Gamma strict `y>0`).
- **Dependencies:** M2-C, M2-D.
- **DoD:** §13.3 Loss-correctness (finite-diff g and h vs analytic, `exp(k·F)` asserted within `f32` tol against a `powf` reference) + domain-errors gates green for both; **§13.4 no-`powf` lint test** green (grep of the module); §05.9 test 6 corners (`y=0` Gamma) finite/floored; **§13.1 invariant matrix green for the `Poisson` and `Gamma` columns**; `{1,2,8}`-thread determinism green on each. Note: `max_delta_step` *advertisement* lands here; its engine-side clamp is M2-G.
- **Size:** M.

### M2-F — `Tweedie{rho}` (construction-time `rho` validation)
- **§-ref:** §05.1 #3 (`rho∈(1,2)` exclusive, default 1.5), §05.3 (Tweedie row), §05.3a, §05.9.
- **Deliverable:** `Tweedie{rho}` implementor: `g=−y·exp((1−ρ)F)+exp((2−ρ)F)`, `h=−y(1−ρ)exp((1−ρ)F)+(2−ρ)exp((2−ρ)F)` floored, `init=log(p̄)`/exposure form, `default_metric=TweedieDeviance{rho}`. Constructor validates `rho∈(1,2)` → `PbError::InvalidConfig` on breach (distinct from data-domain `InvalidInput`, §05.3a note (d)). Two `exp` calls per row, terms reused across g/h.
- **Dependencies:** M2-E.
- **DoD:** §13.3 Loss-correctness + domain-errors gates green for Tweedie; **bad-`rho` returns `InvalidConfig`** (the §13.8 correct-variant-per-fallible-fn rule); §13.1 invariant matrix green for the `Tweedie{1.5}` column (the matrix's canonical Tweedie cell); `{1,2,8}`-thread determinism green. **G4 complete:** the full v1 loss set passes §13.1 across all five objectives — this is the M2 objectives exit.
- **Size:** M.

### M2-G — `max_delta_step` engine contract (leaf-stage |w*|-clamp) + multi-step Newton/Armijo for non-quadratic losses
- **§-ref:** §05.6, §06.4 (multi-step Newton + Armijo, the leaf-estimation seam owned by §06; this task wires the §05 contract into it).
- **Deliverable:** in the §06.4 leaf-estimation hook (consuming §05): the leaf-stage `|w*|≤δ` clamp applied on **full-precision aggregated sums** (never per-row `h` inflation — that would perturb the future quantized histogram), with `δ` resolved as `Config.max_delta_step.unwrap_or(loss.max_delta_step())`. Wire `leaf_newton_steps>1` re-deriving `(g,h)` at leaf-adjusted raw scores with Armijo backtracking (`c₁=1e-4`) — touches only the 8 leaf values of a fixed structure. (Default `leaf_newton_steps=1`; the multi-step path is exactness-neutral and tested but stays off pending the §14 benchmark.)
- **Dependencies:** M2-E (Poisson advertises `Some(0.7)`), M2-F.
- **DoD:** §13.5 numerical units — multi-step Newton converges to the analytic leaf optimum on a Gamma fixture; **Armijo never increases loss**; the leaf clamp caps `|w*|≤δ` on a high-`F` Poisson fixture. **§13.1 invariant matrix stays green** with the clamp + multi-step path active (proving leaf-only edits keep the tree constant-per-cell → I2 intact, firewall `Exact`). `{1,2,8}`-thread determinism green (clamp is on the order-independent full-precision aggregate).
- **Size:** M.

### M2-H — `BlendedLoss` distillation adaptor (the §05 seam; §09 owns policy)
- **§-ref:** §05.7, §13.3 (`blend=1.0` reproduces base loss).
- **Deliverable:** `BlendedLoss { base: &dyn Loss, soft_target: &[f32], blend: f32 }`: convex `g=blend·g_true+(1−blend)·g_soft`, likewise `h`, via two `base.grad_hess` calls combined in **fixed order**, inheriting floor/clamp/`f64` guarantees and `?`-propagating the second call's `Result`. `init_score`/`deviance` delegate to `base` on the **true** `y` and forward verbatim. `blend`=true-label weight, default 0.5.
- **Dependencies:** M2-B…M2-F (a base loss to wrap).
- **DoD:** §13.3 **BlendedLoss test [GATE]** — `blend=1.0` reproduces the base loss's `grad_hess` bit-for-bit; `blend=0.0` is the pure-soft fit; base-loss domain error propagates as a typed `PbError` through the second call's `?` (no-panic gate). Lint/fmt/doctest green. (Teacher orchestration is §09/v1.5 — out of scope here; only the seam + polarity.)
- **Size:** S.

### M2-I — §07 owned types: `MonotoneMap`, `InteractionPolicy`, `CredibilityFloor`, `AdmissionPrior`, `FeatureMask`
- **§-ref:** §07.2, §2.9 (`InteractionPolicy` verbatim), §13.4 anti-HashMap determinism gate.
- **Deliverable:** `crates/tri-boost-core/src/interaction/mod.rs`. `enum MonoSign`; `MonotoneMap(BTreeMap<String,MonoSign>)` with `resolve(&self, names) -> Result<Vec<Option<MonoSign>>, PbError>` (unknown name → `InvalidConfig`); `InteractionPolicy {max_order:u8, groups:Option<Vec<FeatureSet>>}` with `max_order∈{1,2,3}` validation (`InvalidConfig`) and `admissible(chosen, all) -> FeatureMask` (provenance-keyed, raw-feature bitset; empty ⇒ caller terminates early); `CredibilityFloor` (`min_data_in_leaf`/`min_sum_hessian_in_leaf`/`min_weight_sum_in_leaf`/`path_smooth`); `AdmissionPrior {pair, triple, heredity, table_budget_beta=0.5, budget_cells}` (config-only scratch, never serialized); `HeredityMode`; `FeatureMask` bitset alias.
- **Dependencies:** P0 (FeatureSet, AxisProvenance, PbError); M2-G not required (independent).
- **DoD:** unit tests — `resolve` returns `InvalidConfig` on an unknown name (correct-variant, §13.8); `admissible` honors both `max_order` and `groups` (whole-tree semantics). **§13.4 anti-HashMap grep [GATE]** green (`MonotoneMap` is `BTreeMap`, no `HashMap` in serialized-state paths; `AdmissionPrior` `pair`/`triple` proven order-independent point-lookups). Lint/fmt/doctest/`deny(missing_docs)` green.
- **Size:** S.

### M2-J — `wht8` transform + the online per-order screening accumulator (the §06.4 emit consumer)
- **§-ref:** §07.4a (OWNED here), §06.4 emit hook, §13.1 (`wht8`-not-a-gate negative property), §13.7a (cross-check oracle).
- **Deliverable:** the frozen O(8) `wht8` Walsh–Hadamard/Möbius transform (8 leaves → 1 const + 3 `c_i` + 3 `c_ij` + 1 `c_123` under the tree's per-cut `w`-marginals); the running per-(canonicalized sorted-distinct support, order) variance accumulator updated O(8) per finished tree via Parseval (`σ²_triple = m_{123}·c_123²` etc.); the `c_123²` triple witness feeding `prior.triple`. In-fit scratch only — **never serialized**. Invoked from the §06.4 hook on finalized leaves (off the hot path).
- **Dependencies:** M2-I (the `AdmissionPrior` it writes into); the §06.4 hook (foundation P2).
- **DoD:** **§13.7a cross-check oracle [GATE]** — `assert_wht8_triple_matches_purified` agrees with the §08 mass-moving order-3 Faith-Shap to a derived tolerance over proptest random leaf vectors × positive `w`-marginals (`ProductMarginals`/`Uniform` [GATE]; `Joint` [CHECK]), and all 8 coefficients match their purified terms. **§13.1 `wht8`-not-a-gate negative property [GATE]** — no accumulator element is ever read by `assert_exact_decomposition`. §07.9 determinism check — accumulator byte-identical at `n_threads∈{1,2,8}` and absent from the serialized `Model`/`TableBank`.
- **Size:** M.

### M2-K — `constrain_candidate` + the admission funnel (heredity/FAST-RSS/table-size prior/Sobol), the §06 split-finder seam
- **§-ref:** §07.3 (seam + fixed order of operations), §07.4 stages 1/2/4/5 (stage 3 detector is v1.5/v2, stage 5 R-TABLEBUDGET), §07.9.
- **Deliverable:** `enum LevelDecision {Reject, Admit{adjusted_gain}}`; `constrain_candidate(cand, chosen, g_newton, leaf_w, counts, hess, wsum, policy, mono, floor, prior) -> LevelDecision` applying, in the fixed order, (1) interaction-policy hard reject, (2) credibility-floor hard reject across all 8 cells (exact binned `counts`), (3) monotone clamp (M2-L), (4) **soft** prior `adjusted_gain = g_newton.max(0.0)·soft(...)` blending heredity admissibility + FAST-RSS pair prior + `cellprior` table-size penalty. Funnel builder: heredity composition (Sobol-fed "important" test, `Strong` default), FAST-RSS DP `O(b²)`/pair, `cellprior(support)=(budget_cells/max(budget_cells,projected_cells))^β`. The §06 search loop calls this per candidate; if all `Reject`, §06 terminates the tree early.
- **Dependencies:** M2-I, M2-J (prior), M2-L (the clamp it calls in step 3).
- **DoD:** §07.9/§13 **soft-not-hard [GATE]** — planted XOR (RSS≈0, large Newton gain) still recovered; a heavily down-weighted but policy-permitted split still wins on raw gain; a policy-forbidden split never selected. **Table-size penalty soft-not-hard (R-TABLEBUDGET)** — near-`max_bin` triple down-weighted yet a dominant-gain triple still admitted; `table_budget_beta=0.0` reproduces un-penalized ordering exactly. **§13.2 `check_feature_budget` [GATE]** + §07.9 group-whitelist property (no realized support spans two groups). **§13.1 five invariant checks green** on every funnel fixture (G3 holds — §07 never bends I2). `{1,2,8}`-thread determinism green.
- **Size:** L.

### M2-L — Monotone per-level joint leaf-clamp + graceful early-termination
- **§-ref:** §07.5, §07.3 step 3, §07.1 (no-valid-split fallback), §13.3/§07.9 monotonicity oracle, G5.
- **Deliverable:** the per-level joint leaf-clamp on the 8-leaf vector: per-leaf bounds `[lo,hi]` initialized `(−∞,+∞)`, `w_ℓ=clip(−G_ℓ/(H_ℓ+λ), lo_ℓ, hi_ℓ)`; for a constrained-feature shared split, midpoint `m=(w_L+w_R)/2`, propagate to all descendants (`Increasing`: left `hi←min(hi,m)`, right `lo←max(lo,m)`; reverse for `Decreasing`); feasibility = every cousin pair `w_L≤w_R` (Increasing) after clamping, else level gain `−∞` ⇒ `Reject`. `path_smooth` applied **after** the clamp then re-clamped (§07.6). If no candidate is feasible the tree terminates early at `depth<3` — a valid `ObliviousTree`, not a `PbError`.
- **Dependencies:** M2-I (`MonoSign`, `CredibilityFloor.path_smooth`); consumed by M2-K step 3.
- **DoD:** **§13.3 / §07.9 monotonicity oracle [GATE]** (= G5) — on a monotone target the reconstructed **1-D main-effect table** and **total score** are monotone to float tolerance (the property deliberately does *not* assert monotonicity on 2-D/3-D interaction slices, per the §07.5 caveat); an anti-monotone target forces `depth<3` **early-termination, not a `PbError` or I1 violation**. §13.5 degenerate-input sweep (all-clamped, single-feature) yields a valid model, never panic/NaN. **§13.1 invariant matrix `{±monotone}×{±early-termination}` arms green**; `{1,2,8}`-thread determinism green.
- **Size:** M.

### M2-M — Wire `InteractionPolicy`/`MonotoneMap` through `FitSpec`→§06 fit; M2 milestone gate
- **§-ref:** §2.9 (`FitSpec.interaction`/`.monotone`), §06 fit loop, §14.2 Phase 5 / G5, §13.1 full matrix.
- **Deliverable:** thread `FitSpec.interaction: InteractionPolicy` and `FitSpec.monotone: MonotoneMap` into the §06 `Booster::fit` validation + per-level split-finder (replacing any bare scalar `max_interaction_order`); resolve `MonotoneMap` against `ModelSchema.feature_names` once at fit entry; surface the resolved per-axis sign + `InteractionPolicy.admissible` mask into M2-K. Validate `max_order` and monotone names at fit entry (`InvalidConfig`). (The §12 Python `interaction=`/`monotone=` kwargs are §12/P7 — out of scope.)
- **Dependencies:** M2-F (full loss set), M2-K, M2-L.
- **DoD:** **the §13.1 invariant matrix green in full** — `{5 objectives}×{orders 1,2,3}×{ProductMarginals,Uniform}×{±monotone}×{±early-termination}` (the M2 capstone). `max_interaction_order=1` yields a pure-additive model (no realized 2-D/3-D tables) — the §07.9 staging/additive check. **§13.4 `{1,2,8}`-thread byte-equality + cross-run reproducibility [GATE]** green on the fully-constrained, fully-objective fit set. **G5 green** (Phase 5 exit): monotone holds on total score + constrained 1-D table, funnel verified soft, G3 invariants hold throughout. All standing gates (lint/fmt/deny/MSRV/doctest/coverage ≥95% on the math modules) green.
- **Size:** M.

---

**Critical-path order:** M2-A → M2-B → (M2-C → M2-D → M2-E → M2-F → M2-G, M2-H) | M2-I → (M2-J, M2-L) → M2-K → M2-M. Objectives (A–H) and the §07 type/clamp foundation (I, J, L) can proceed in parallel after their roots; M2-K joins them; M2-M is the milestone gate. No task merges until its named gates are green — "done" is gates-green, not code-written. **Files:** `crates/tri-boost-core/src/loss/{mod,squared,logistic,poisson,gamma,tweedie,blended}.rs` and `src/interaction/{mod,wht8,funnel,monotone,constrain}.rs`; tests in `tests/` (proptest suites) and `tests/invariants/` (the §13.1 matrix, already wired by the foundation, extended one column/arm per task).
