## 01 — Overview, goals, invariants & the milestone

> 2026-06-21. This is the constitution. It states **what** tri-boost is, the three aims it serves and how they trade off, the load-bearing exact-decomposition thesis, the scope boundary, the two hard invariants (I1/I2) restated for readers, and the single v1 success milestone that the remaining 13 sections (02–14) are built to reach. It defines no new types — it binds the reader to the shared types of §2 and the invariant contract of §3 of `00-spec-skeleton.md`. Where a fork is still open it is named with a recommended default. The glossary at the end fixes the vocabulary every later section uses.

---

### 1.1 What tri-boost is (one paragraph, decision-grade)

tri-boost is a gradient boosting machine library — a pure-Rust core (`tri-boost-core`, crates.io-publishable) plus Python bindings (`tri_boost`, PyO3/maturin) — whose defining structural decision is that **every tree is a depth-3 oblivious (symmetric) tree**: one shared `(axis, bin_le)` test per level, at most three distinct raw features per tree, eight leaf values indexed by three binary tests (§2.5 `ObliviousTree`). That single constraint is not a regularizer bolted onto a generic GBM — it *is* the data structure. Because each tree depends on ≤3 features, the whole ensemble is a sum of ≤3-feature functions, so its functional-ANOVA (fANOVA) decomposition **terminates exactly at 3rd order**: the trained `Model` (§2.6) rewrites, with provable zero loss, as a constant plus 1D main-effect, 2D pairwise, and 3D triple purified tables (§2.7 `TableBank`). tri-boost optimizes the *entire* methodology — losses, histograms, categoricals, interaction selection, refit, acceleration, and ensembling — around that property while refusing to pay for it in speed or accuracy.

The library is the deliverable. Insurance rating is the natural first *downstream application* — under a log link the tables read directly as multiplicative relativities (`f0` → base rate 1.000, `exp(f_u)` → rating cells) — but the rating workflow, calibration, fairness, and deployment tooling are explicitly out of scope (§1.6).

---

### 1.2 The three aims and how they trade off

Every decision in this spec serves three aims, held in tension-managed balance. When two pull against each other, the order below is the tie-breaker only where §1.4's invariants do not already decide it: **the invariants (I1/I2) are absolute and dominate all three aims.**

1. **MAXIMUM PREDICTIVENESS.** As accurate as the constraint allows, using every exactness-preserving gap-closer in the playbook (`design/02`): Newton summed-gain splits with exact leaf weights, the full order-3 budget with exact 3D tables, the heredity+FAST+Sobol admission funnel under *joint* boosting, Fisher sorted-ordinal target statistics, log-link deviance objectives, quantized/MVS training, refit/acceleration, and bagged table ensembling. Accuracy is chased aggressively, but **never by bending I1/I2** (§1.4); any technique that would (linear leaves, continuous-TS axes, combination CTRs, >3-order base margins) is forbidden on the exact path or quarantined behind the firewall (§3, §1.5).

2. **FULLY DECOMPOSABLE.** The trained model decomposes **losslessly and exactly** — not approximately, not truncated — into `f0 + Σ f_i + Σ f_ij + Σ f_ijk` on a shared merged grid. This is the thesis (§1.3) and the non-negotiable aim: the number that gets deployed (tree-sum scoring) and the number that gets audited (table-sum scoring) are the *same number*, enforced bit-for-bit in CI (§1.4 I2; §3's five checks).

3. **FAST.** Competitive with LightGBM/CatBoost on CPU *training* (histogram engine, subtraction trick, quantized integer histograms, per-thread padded histograms with fixed-order reduction, MVS sampling) and **structurally faster at inference**: a depth-3 oblivious tree is a branch-free 8-cell lookup (`b0 | b1<<1 | b2<<2`), and table-sum scoring over the purified bank is a handful of cache-resident reads independent of tree count.

**Where they trade off, and the resolved position** (from `design/01` §"central tradeoffs", binding here):

| Tension | Position taken | Rationale |
|---|---|---|
| Predictiveness ↔ Decomposable | Decomposable wins **absolutely** | A bent invariant means there is no product; accuracy is maximized strictly *inside* the cage. |
| Per-tree strength (depth-3 is weak) ↔ Fast (more trees) | **Compensate, don't relax** | Multi-step Newton leaves, optional Nesterov/fully-corrective refit, quantized-histogram headroom shrink tree count rather than deepening trees. |
| Predictiveness (continuous-TS, combination CTRs) ↔ Decomposable | **Fisher sorted-ordinal TS; forbid combination CTRs** | Recovers ~all categorical accuracy while each category stays a readable row and the ≤3-raw-feature budget holds. |
| Predictiveness (richer leaves/calibration) ↔ Decomposable | **Constant leaves always; calibration stays outside the tables** | Linear/soft leaves and nonlinear warps break `g(Σf_u)=Σg(f_u)` and within-cell monotonicity. |

The quantization mechanism is the cleanest example of the aims *aligning*: quantized integer histograms are simultaneously the ~2× **speed** lever and — because integer sums are associative and order-independent — the **bit-reproducibility** mechanism that the **decomposable** aim depends on (the same `seed`-and-quantize mechanism gives I2 its build gate). Speed and decomposability are bought with one stone.

---

### 1.3 The exact-decomposition thesis (the load-bearing fact)

For any square-integrable `F(X)` and a reference measure `w`, the fANOVA expansion is `F(X) = Σ_{u ⊆ [d]} f_u(x_u)` over all feature subsets `u` (research/03 §1.1). The thesis is a structural theorem, not an approximation:

> **Theorem (order-3 termination).** If `F = f0 + Σ_t α_t · tree_t` where every `tree_t` is a depth-3 oblivious tree depending on at most 3 distinct raw features, then `f_u ≡ 0` for every `u` with `|u| > 3`. Hence `F = f0 + Σ_{|u|=1} f_u + Σ_{|u|=2} f_u + Σ_{|u|=3} f_u`, a finite sum that is **exact**.
>
> **Remark — why depth is fixed at 3 (a deliberate, capacity-lossless choice).** The theorem requires only ≤3 *distinct* features per tree, not depth 3; a deeper tree on the same ≤3 features would still be exactly ≤3rd-order. We nonetheless fix `depth = #splits = #distinct features ≤ 3` *deliberately*, because greater depth adds **no representational capacity to the ensemble**: a depth-3 *ensemble* is already a universal approximator of ≤3rd-order functions (each tree is a rank-1 `step⊗step⊗step` term; summing over thresholds spans the full ≤3rd-order space, and any fine-grid cell is reachable by inclusion–exclusion). Depth >3 would buy at most a marginal, data-dependent greedy-boosting *convergence-efficiency* gain — not expressiveness — and is not worth its cost (variable-length leaf arrays, cross-level monotone bound-propagation, larger tables). Fixing depth at 3 keeps every tree a fixed `[f32; 8]` / 3-split object. Settled decision; see [`../design/03-methodology-review.md`](../design/03-methodology-review.md).

*Proof sketch.* Each tree is a function of ≤3 coordinates, so it is its own fANOVA expansion truncated at order 3 (all higher components integrate to zero). fANOVA components are linear in `F` for fixed `w`, and a finite weighted sum of order-≤3 functions is order-≤3. Therefore the ensemble's components above order 3 vanish identically (research/03 §1.1; `design/02` §1).

Two consequences are load-bearing for the whole spec:

- **Completeness, not truncation.** Unlike EBM/GA2M (which *cap* at order 2) or post-hoc fANOVA on a black box (which *truncates and approximates* an unbounded expansion), tri-boost's decomposition is provably complete: there are no order-≥4 terms to drop. This is why the tables *are* the model rather than a summary of it.
- **Linearity → streaming accumulation.** Purification is a linear operator with `Σ α_i = 1` (Lengerich Cor. 2.2): `purify(Σ α_i F_i) = Σ α_i purify(F_i)`. So "purify-then-sum ≡ sum-then-purify," which is exactly what makes per-tree streaming accumulation (§08), self-ensembling (§09), and post-hoc `w`-recomputation (§08) all **exactness-preserving** — they are admissible *because* they stay linear in the score-space `F`, where the model is additive.

The chosen reference measure for purification is, by default, `RefMeasure::ProductMarginals { laplace }` (Laplace-smoothed empirical product-of-marginals; §2.7) — auditable, positive, sums to 1, and the setting under which the variance-sum identity and exact equal-split SHAP hold. The measure is stamped on every export and is recomputable post-hoc without retraining (`design/01` §5).

---

### 1.4 The two hard invariants (I1, I2), restated for readers

These are stated normatively in §3 of the skeleton and enforced as build-blocking **[GATE]** checks. This section restates them for the reader and fixes that *they dominate the three aims*: a decision that improves any aim while violating an invariant is rejected, not traded.

**I1 — Depth-3 oblivious / ≤3 distinct raw features per tree.** For every `ObliviousTree`: `depth ∈ 1..=3`; each level is one shared `(axis, bin_le)` test (`Split`, §2.5); and the number of **distinct `provenance[axis].raw`** values across the splits equals `depth`. The distinctness is checked on *raw features via `AxisProvenance` (§2.1)*, never on encoded columns — this is what lets a tree split on three different categorical-TS axes (three distinct raw categoricals → legal) while forbidding a single combination-CTR axis that packs >3 raw features. Fewer than 3 levels is legal: a tree that finds no admissible (e.g. monotone-valid) split at a level **terminates early** at depth <3, a legitimate lower-order fANOVA outcome, not an error. Violation ⇒ `PbError::InvariantViolated { Invariant::FeatureBudget }`. Non-symmetric growth, linear/soft leaves, and >3-raw-feature axes are **forbidden on the exact path**.

**I2 — Exact ≤3rd-order decomposability on a shared grid.** The ensemble equals `f0 + Σ_u f_u` on the merged grid (sorted union of realized borders per axis) **bit-for-bit within float tolerance**. This is not asserted — it is *proved at build time* by the five `Invariant` checks (§3, owned by §08): **Reconstruction** (max over one interior point per merged-grid cell of `|F_ens − (f0 + Σ f_u)| < tol`, exhaustive because piecewise-constant), **MassConservation**, **Purity** (every axis-slice `w`-weighted mean zero), **VarianceSum** (`σ²(F) = Σ σ²(f_u)`, under product/uniform `w`), and **ThreeWayEqual** (tree-sum = table-sum = Shapley-sum, bit-equal). "If these ever disagree there is no product." Any technique that bends I1/I2 must be flagged in its owning section and gated behind the firewall.

The invariants ride on three engineering preconditions the rest of the spec must uphold, called out here so §01 sets the expectation: (a) **one shared global binning grid per feature** (§03), without which tables are not summable; (b) **bit-reproducibility** via quantized integer histograms + fixed-order float reductions + a single seeded PRNG with **deterministic per-work-unit re-seeding** — a frozen `splitmix64` mix of `(base, round, stage, block)` fed into `Pcg64::seed_from_u64` (a position-stable, thread-count-independent re-seed, **not** an unimplementable "splittable" PRNG; §1 of skeleton, §06, §11) — non-determinism would make the Reconstruction gate flaky and the deployed/audited numbers diverge; and (c) **constant leaves only** — `ObliviousTree::leaves` is `[f32; 8]`, never a function of `x` within a cell.

---

### 1.5 The Exact/Approximate firewall (restated)

The structural defense against death-by-a-thousand-cuts is the typed `ExactnessMode` (§3): `Exact` or `Approximate { reason: String }`, stamped on every `Model`. A model in `Exact` mode has passed all five checks and may export rating tables. The instant any operation cannot pass them — a nonlinear calibration warp, a continuous-TS axis, linear leaves, or a >3-order GLM base margin — the model flips to `Approximate { reason }` and **refuses** to export an `Exact` `TableBank`, returning `PbError::ExactnessFirewall(..)` (or exporting "tables + residual model"). Conversely, the operations that *stay* `Exact` are enumerated and load-bearing for the predictiveness aim: post-hoc `w`-recomputation, multi-step Newton leaf refit, fully-corrective refit, Nesterov alpha-mixing, and self-ensemble averaging. §01 names the firewall here so the reader treats `ExactnessMode` as a first-class property of every model from this point on; the implementation is §08's.

---

### 1.6 Scope boundary — what this library is and is not

**In scope (the library core).** The depth-3 oblivious boosting engine (§06); the `Loss` trait and its v1 implementors `SquaredError`, `Logistic`, `Poisson`, `Gamma`, `Tweedie { rho }` with exact (g,h), exposure offsets, and deviance early-stopping (§05); binning and the shared grid (§03); leakage-free Fisher sorted-ordinal categoricals (§04); interaction selection, monotone and `max_interaction_order ∈ {1,2,3}` constraints (§07); the purification/fANOVA explainability engine and exact interventional SHAP/Faith-Shap ≤order-3 (§08); refit, acceleration, re-anchoring, and self-ensembling (§09); serialization and inference (§10); the Python API (§12). Pricing **objectives** are in scope precisely because they are loss functions; the **table decomposition** is in scope because it is the explainability output.

**Out of scope (downstream application work).** Calibration beyond correctness (autocalibration, multicalibration, Venn-ABERS); fairness auditing and discrimination-free marginalization; uncertainty/distributional output (conformal intervals, distributional heads); deployment/compilation (SQL/ONNX/FPGA emission — we ship clean JSON/bincode, serving is MLOps); workflow orchestration (frequency-severity fit-twice, A/E suites, rate-filing); and — critically for the milestone — **any shipped benchmarking, model-comparison, or "cost-of-the-cap" tooling.** We do *not* fit an internal unconstrained reference `F_∞`, and we do *not* ship an order-4+ residual head (it would ride un-tabled structure beside the tables and break the guarantee). Judging a *deployed* model against other models is the user's job.

**Permanent non-goals** (owned in full by §14): interactions beyond 3rd order (a deliberate, disclosed capacity limit); GPU *training* (deferred — a `Backend` seam is kept, §02); ordered boosting (a later optional accuracy knob); and non-symmetric growth policies (would break I1 — out of scope by construction).

---

### 1.7 The v1 success milestone (the one north star)

There is exactly **one** v1 success criterion, and it is the development yardstick the whole spec serves. It is measured by us during development; **it is not shipped tooling** (per §1.6).

> **v1 milestone.** On the **TabArena** benchmark suite, every fitted tri-boost model **(a) beats EBM / GA2M** on held-out deviance/logloss, and **(b) comes within striking distance of unconstrained XGBoost / LightGBM / CatBoost**, **while (c) every model stays bit-exactly decomposable** into ≤3rd-order tables (all five I2 checks pass, `ExactnessMode::Exact`).

Clause (c) is what makes (a) and (b) worth anything — competitive predictiveness *with* exact decomposition is the entire thesis. Grounded expectations from `design/02` §3 (binding as the yardstick's interpretation, not as a promise):

- **Beat EBM essentially everywhere.** tri-boost has strictly more capacity (exact order-3 vs EBM's order-2 cap), full Newton summed-gain splits (vs EBM's tiny-LR stumps), and CatBoost-grade categoricals (Fisher TS). EBM trails LightGBM by ~160 Elo on TabArena's 51 datasets — a clear, beatable deficit.
- **Within striking distance of the black boxes.** Obliviousness itself costs ≈0 at the ensemble level (CatBoost ranks first untuned); the structural penalty is the *order cap*, not symmetric growth. Order-3 is estimated to recover ~⅓–⅔ of the residual EBM→GBDT gap where the gap is genuine 3-way structure, landing roughly half-to-two-thirds from EBM toward the GBDT frontier (~50–90 Elo behind the best GBDT on average) for the bulk of datasets.
- **The honest minority.** Dense ≥4-way / high-frequency irregular targets (the named Higgs / Year cases) are a **bias ceiling no ensembling crosses** — a 3rd-order model cannot represent them. This is the disclosed cost of exact ≤3rd-order decomposition; on such books tri-boost is the wrong tool, and saying so is better than smuggling in un-tabled structure.

**The metric.** Per-objective **deviance** (Poisson/Gamma/Tweedie) or **logloss** (Logistic) is the default training, early-stopping, and milestone metric, because deviance is strictly proper for the mean and **RMSE is not** on the log-link losses (`Loss::deviance`, §2.4 / §05). The honest comparison is **bagged-vs-bagged** (EBM already bags ~14× internally), so the self-ensemble path (§09) is the fair-comparison configuration, not single-model.

**Strong defaults, not a tuning product.** The milestone must be reachable with ship-quality out-of-the-box hyperparameters and early stopping — the depth-3 cap shifts the LR × n_trees surface toward more, lower-LR trees, and a user should get a good model without a search. Exhaustive HPO is the user's to run.

---

### 1.8 Open forks touching this section (with recommended defaults)

§01 owns no algorithm, so it inherits the skeleton's fork log; only the ones that shape the milestone framing are restated here.

- **`max_interaction_order` default → 3** (the differentiator; 1 = additive GAM safe-harbor, 2 = GA2M/EBM parity exposed as filing strategies). Order-3 is the identity setting; 1/2 can only restrict accuracy, so they are trust dials, not predictiveness levers.
- **Self-ensemble on by default → no; recommended on for milestone runs.** Single best-tuned model is the shipped default (§09 owns the toggle); the *milestone* should be evaluated bagged-vs-bagged for fairness. This is a measurement convention, not a code default.
- **Reference measure default → `ProductMarginals { laplace }`** (auditable, exact-SHAP, post-hoc recomputable; joint/Hooker built early for a between-measure relativity-drift diagnostic). Owned by §08; restated here because it fixes the meaning of "decomposable" in the milestone.

*Genuinely still open (deferred to §09/§14):* whether fully-corrective refit and Nesterov boosting should ever default on. **Recommended default: off, benchmark-gated** — both are exact and both shrink tree count, but cost on pricing/TabArena data must be measured before they earn a default-on.

---

### 1.9 How §01 upholds the invariants and serves the aims

§01 introduces no code, so it cannot *violate* I1/I2 — its job is to make them non-negotiable for every later section by (a) ranking decomposability above predictiveness in the tie-break order, (b) restating I1/I2 and the firewall as reader-facing properties, and (c) defining the milestone's clause (c) so that "exactly decomposable" is part of the success bar, not an afterthought. It serves **accuracy** by committing the spec to the full gap-closing playbook inside the cage; **decomposable** by making the exact-decomposition thesis the load-bearing fact and the five checks part of success; and **fast** by naming the inference structure (8-cell lookup + table-sum) and the quantization mechanism that unifies speed with reproducibility.

---

### 1.10 Testing approach for this section

§01 is normative prose, so its "tests" are the gates it points at, plus consistency checks that this constitution stays true to the spec:

- **No-orphan check (CI, doc-lint):** a build-time script asserts every aim, invariant, and scope claim in §01 has an owning section in the §4 ownership map and a backing gate (e.g. I2 ↔ the five `Invariant` checks in §08/§13). A claim in §01 with no owner is a documentation bug and fails the docs job.
- **Milestone harness (internal, not shipped):** a developer-only harness (lives under `benches/`/dev scripts, never in the published crate or wheel — §1.6) fits tri-boost on the TabArena suite, asserts `ExactnessMode::Exact` and all five I2 checks pass on **every** fitted model (the hard pass/fail of clause (c)), and reports per-dataset deviance/logloss vs EBM and the GBDTs (clauses (a)/(b)) bagged-vs-bagged. Clause (c) is a CI-style gate; (a)/(b) are reported, not gated (they are dataset-dependent yardsticks).
- **Glossary/type consistency (CI):** the identifiers used in §01 (`ObliviousTree`, `Model`, `TableBank`, `RefMeasure`, `ExactnessMode`, `PbError::InvariantViolated`, `Invariant`) must resolve to the §2 definitions verbatim; a drift between §01's prose names and §2's types fails the docs job.

---

### 1.11 Glossary (the vocabulary the rest of the spec uses)

- **Oblivious / symmetric tree** — a tree with one shared `(axis, bin_le)` test per level; here always depth ≤3 (`ObliviousTree`, §2.5).
- **Axis vs raw feature** — an *axis* is a model column (numeric bin axis, missing axis, or categorical-TS axis); a *raw feature* (`FeatureId`) is the user's original column. I1's budget is counted on **distinct raw features** via `AxisProvenance` (§2.1).
- **Score space (`raw` / `F`)** — the pre-link additive scale on which boosting, accumulation, and ensembling operate. **Response space (`pred`)** — after the inverse link (`Loss::pred_from_raw`).
- **fANOVA / purification** — the linear operator that turns the raw tensor bank into the canonical identifiable `f0 + Σ f_u` under reference measure `w` (§08).
- **Reference measure (`w`)** — the integration measure for purification; default `ProductMarginals { laplace }` (§2.7).
- **Merged / union grid** — the sorted union of realized borders per axis; the shared grid on which all tables are exactly summable (§03/§08).
- **EffectTable / TableBank** — one purified effect tensor for a feature set `u`; the full bank is the lossless decomposition (§2.7).
- **ExactnessMode / firewall** — the typed `Exact`/`Approximate` wall guarding the decomposition (§3, §08).
- **The three aims** — predictiveness, decomposability, speed (§1.2). **I1/I2** — the two hard invariants (§1.4). **The milestone** — beat EBM, approach the GBDTs, all-exact, on TabArena (§1.7).
