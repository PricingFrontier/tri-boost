## 07 — Interaction selection & constraints

> Owns (per §4 map): `MonotoneMap` + the per-level joint leaf-clamp; the `InteractionPolicy` (whole-tree interaction constraints + `max_interaction_order ∈ {1,2,3}`); the **heredity → FAST-RSS → triple-detector → Sobol** admission funnel (a soft prior, never a hard gate); the **joint-boost-then-single-purification** architecture rule; credibility floors as they shape the candidate set; the **`wht8`** frozen O(8) Walsh–Hadamard / Möbius transform and the **online interaction-screening prior** it feeds (07.4a — this is where `wht8` is OWNED/registered). Uses (does not own): the Newton summed-gain split-finder and config struct (§06), the `Loss` trait (§05), purification + Sobol variance + the five Invariant checks (§08), the merged grid + provenance (§03).

This section governs **how the scarce ≤3-feature budget is spent**. Every mechanism here must (a) leave I1/I2 untouched — it acts only on the *candidate-feature set* per level or on *leaf values within a fixed oblivious structure*, never on tree shape — and only then (b) maximize accuracy by steering the budget onto real structure while bounding the realized table set. There is **exactly one hard gate**: the user-declared `InteractionPolicy` (a structural contract). Everything statistical — heredity, FAST, the triple detector, Sobol — is a **soft prior on the split-candidate score**, because each optimizes a different objective than the booster's Newton gain and would amputate a real interaction if allowed to forbid it (the GAMI/FAST lesson, research/06-4 and research/03 §3.3).

### 07.1 Decisions (with defaults)

| Decision | Default | Rationale |
|---|---|---|
| Staging | **Joint boost over all admitted supports, one final purification** | Reject EBM/GAMI-Tree mains-then-interactions staging — it mis-converges under correlated features (research/06-4). |
| `max_interaction_order` | **3** | The differentiator; 1 = additive GAM safe-harbor, 2 = GA2M (matches EBM). |
| Interaction constraint semantics | **Whole-tree** (tree's realized raw-feature set must lie jointly within one allowed group) | Stricter than XGBoost's per-path union rule; "permitted interactions" == "tables that can appear" — a tested export invariant. |
| Admission funnel | heredity ∧ FAST-RSS prior ∧ triple-detector prior ∧ table-size admission penalty → Sobol arbiter | All soft; bounds the table set by composition, not by a `C(n,3)` scan. |
| Funnel as a hard gate? | **Never** | FAST-RSS ≠ Newton gain; statistical screens (incl. the table-size penalty) only bias the candidate score and post-hoc pruning. |
| Table-size admission penalty | **On by default; soft** | Deprioritize a support whose realized merged-grid cell count is large (e.g. near-`max_bin` axes), so the §08 table budget (R-TABLEBUDGET) is rarely hit by gain rather than by a hard reject. |
| Online triple-detection prior | **`wht8` per-tree fANOVA variance (07.4a), on by default** | O(8)/tree, exact-per-tree, off the inner-loop dense-reference fit; an exact triple *witness* (`c_123²`) sharper than FAST-RSS. Soft, never gates. |
| Monotonicity | per-level joint leaf-clamp on `w = −G/(H+λ)`; `name→sign` map | Wrong sign costs 2.5–21.2% AUC, correct costs <0.2% (Henckaerts). |
| No-valid-split fallback | **graceful early-termination at depth <3** | A legitimate lower-order fANOVA outcome (I1 allows `depth ∈ 1..=3`), never a `PbError`. |
| Triple-detector front-end | off by default, opt-in (v1.5) | Pre-training analysis, exactness-neutral; sharper than FAST for triples. Largely retired by 07.4a's online `wht8` witness (which needs no separate fit). |

### 07.2 Owned types

```rust
/// Monotone direction for a RAW feature. Keyed by NAME, never by column position.
pub enum MonoSign { Increasing, Decreasing }
/// `BTreeMap` (not `HashMap`) so iteration/resolution order is deterministic — the
/// §13.4 anti-HashMap determinism gate. Config-only: consumed at `resolve` time,
/// NEVER serialized into the Model or TableBank.
pub struct MonotoneMap(pub BTreeMap<String, MonoSign>);
impl MonotoneMap {
    /// Resolve a name-keyed map against the schema into a per-FeatureId sign.
    /// Errors on an unknown feature name → PbError::InvalidConfig.
    pub fn resolve(&self, names: &[String]) -> Result<Vec<Option<MonoSign>>, PbError>;
}

/// Whole-tree interaction policy (the ONE hard structural gate of §07).
/// This is the `FitSpec` entry point for interaction control: `FitSpec.interaction:
/// InteractionPolicy` (replacing the former scalar `FitSpec.max_interaction_order: u8`),
/// threaded through the §06 `fit` loop and surfaced as the §12 `interaction=` kwarg, so
/// the `groups` whitelist tested in 07.9 actually has a user-facing entry point.
/// Config-only: never serialized into the Model.
pub struct InteractionPolicy {
    pub max_order: u8,                       // {1,2,3}; validated, else InvalidConfig
    pub groups: Option<Vec<FeatureSet>>,     // allowed co-occurrence groups; None = unconstrained
}
impl InteractionPolicy {
    /// Live candidate raw features admissible at the next level given those already
    /// chosen, honoring BOTH max_order and (if present) the group whitelist.
    /// Returns a bitset over raw FeatureIds. Empty ⇒ caller terminates the tree early.
    pub fn admissible(&self, chosen: &FeatureSet, all: &FeatureMask) -> FeatureMask;
}

/// Soft prior over candidate (raw-feature) supports, in [0, 1] — multiplies the
/// per-level Newton gain magnitude. Built once per fit from the funnel (07.4),
/// refreshed lazily as Sobol importances accumulate. NEVER zero on the exact path
/// unless the InteractionPolicy hard-forbids the support (then the candidate is masked,
/// not scored).
/// The `pair`/`triple` maps are used ONLY for order-independent point lookups by a
/// canonicalized (sorted-distinct) raw-feature key, so iteration order never affects a
/// result — they do not touch the §13.4 determinism gate. They are an in-fit scratch
/// structure, never serialized into the Model or TableBank.
pub struct AdmissionPrior {
    pub pair: HashMap<[FeatureId; 2], f32>,
    pub triple: HashMap<[FeatureId; 3], f32>,
    pub heredity: HeredityMode,              // Weak | Strong | Off
    /// Table-size admission penalty (07.4 stage 5, R-TABLEBUDGET). `cellprior` decays a
    /// support's soft multiplier as its projected merged-grid cell count approaches the
    /// §08 `max_table_cells`; `beta = 0.0` disables it. Soft-only, never gates.
    pub table_budget_beta: f32,              // default 0.5; 0.0 = off
    pub budget_cells: u32,                   // = §08 max_table_cells (per-table cell budget)
}
pub enum HeredityMode { Off, Weak, Strong }

/// Per-feature credibility floors that veto a whole shared level (07.6).
/// §07 OWNS these leaf-credibility floors (per §4); §06's `Config` references this one
/// struct rather than redefining `min_data_in_leaf` / `min_sum_hessian_in_leaf` /
/// `path_smooth` (no-overlap rule). The field name is `min_sum_hessian_in_leaf` (not
/// §06's former `min_sum_hessian`). This is DISTINCT from §03's grid-build
/// `min_data_per_bin` (rare-bin merge at binning time), which stays separate.
pub struct CredibilityFloor {
    pub min_data_in_leaf: u32,               // exact binned count, all 8 cells
    pub min_sum_hessian_in_leaf: f32,        // Σh floor, all 8 cells
    pub min_weight_sum_in_leaf: f32,         // Σw (e.g. exposure) floor — stable under log-link
    pub path_smooth: f32,                    // 0 = off; parent-shrinkage of leaf values
}
```

`FeatureSet` is the §02 shared type (sorted distinct raw ids, size 0..=3); `FeatureMask` is a §07-local bitset alias over raw `FeatureId`s. §07 supplies the *masking* and *scoring* closures; §06 owns the search loop.

### 07.3 Integration with the split-finder (the seam)

The §06 per-level search evaluates every candidate `(axis, bin_le)` by summed Newton gain `g_newton = ½ Σ_leaves [G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)]`. §07 wraps that loop with two pure functions, applied **in this fixed order** so a soft prior can never resurrect a hard-forbidden split:

```rust
/// Outcome of constraining one candidate at one level. Owned by §07; called by §06.
pub enum LevelDecision { Reject, Admit { adjusted_gain: f32 } }

pub fn constrain_candidate(
    cand: &Split,                 // (axis, bin_le)
    chosen: &FeatureSet,          // raw features already in this tree
    g_newton: f32,                // raw summed Newton gain from §06
    leaf_w: &[f32; 8],            // candidate leaf weights w = −G/(H+λ)
    counts: &[u32; 8], hess: &[f32; 8], wsum: &[f32; 8],
    policy: &InteractionPolicy,
    mono: &[Option<MonoSign>],    // per-axis resolved sign
    floor: &CredibilityFloor,
    prior: &AdmissionPrior,
) -> LevelDecision;
```

Order of operations inside `constrain_candidate` (steps 1–3 are *hard rejects*):
1. **Interaction policy.** If adding `provenance[cand.axis].raw` to `chosen` is not admissible under `policy` (exceeds `max_order`, or no single group contains the union), `Reject`.
2. **Credibility floors.** If any of the 8 cells violates `min_data_in_leaf`, `min_sum_hessian_in_leaf`, or `min_weight_sum_in_leaf`, `Reject`. The floor binds across **all 8 cells of the shared level** — one under-supported cell vetoes the candidate (stricter than asymmetric trees; the credibility guarantee actuaries expect). `min_data_in_leaf` is checked **exactly** on the binned `counts` the histogram carries (no hessian approximation).
3. **Monotone clamp.** Apply the joint leaf-clamp (07.5); if it sets the gain to `−∞`, `Reject`.
4. **Soft prior.** `adjusted_gain = g_newton.max(0.0) * soft(cand, chosen, prior)`, where `soft ∈ (0,1]` blends heredity admissibility, the FAST/triple priors, **and the table-size admission penalty `cellprior(chosen ∪ {raw})` (07.4 stage 5, R-TABLEBUDGET)** for the support `chosen ∪ {raw}`. The prior multiplies the **non-negative** gain magnitude only (multiplying a signed near-zero gain can invert ordering — research/06-4); a disliked or storage-expensive support is *down-weighted*, never removed. `Admit { adjusted_gain }`.

If **every** candidate at a level `Reject`s, §06 terminates the tree early at the current depth — a valid `ObliviousTree` with `depth ∈ 1..=2`. A tested code path, not an error.

### 07.4 The admission funnel (building `AdmissionPrior`)

Built once before boosting (and the Sobol term + per-axis merged extents refreshed lazily during it). Five stages, each cheap; none can forbid.

**(1) Heredity / strong-hierarchy (composition, not enumeration).** Avoids the `C(n,3)` scan (n=100 ⇒ 161,700 triples) by building supports bottom-up: a pair `{i,j}` is *eligible* only once both mains are important; a triple `{i,j,k}` only once its three sub-pairs are eligible (`Strong`) or at least one is (`Weak`). "Important" = Sobol fraction above `heredity_tau` (default `1e-4`), read from the running purified bank (§08). Default `Strong`. Eligibility sets `soft = 1.0`; ineligible supports get `soft = heredity_floor` (default `0.1`) — discouraged, not banned.

**(2) FAST-RSS pre-filter (2-way, on the residual).** Reuse the binned cumulative target/weight histograms the engine maintains. For a pair `(i,j)` and cut `(c_i,c_j)` partitioning the plane into quadrants `q∈{a,b,c,d}` with residual-sums `T_q` and weight-sums `W_q`, the 4-quadrant predictor's RSS reduction is `ΔRSS(c_i,c_j) = Σ_q T_q²/W_q` (larger ⇒ stronger). Pair strength = `max over (c_i,c_j)`, computed `O(1)`/cut via the DP table `L^t(c_i,c_j)` filled row-by-row (research/03 §3.3), `O(b²)`/pair after `O(b²+N)` setup. Normalize to `(0,1]` into `prior.pair`. FAST's objective is **RSS-on-residual, not Newton gain**, so it only *scales* the prior — never gates.

**(3) Gradient-based triple detector (opt-in front-end, v1.5).** FAST is intrinsically pairwise; *triples* are where the order-3 budget beats EBM. When enabled, seed `prior.triple` from an interaction statistic computed **once** on a quickly-trained dense reference (NID/PID gradient·Hessian scores, or META-ANOVA's `I(j) = E[Var{∂_j f | X_j}]`). Pure pre-training analysis — it never enters the model, so it is exactness-neutral. The SIAN/NODE-GA3M *models* are soft/neural and do not port; only their *detector* transfers. **Largely retired by 07.4a:** the online `wht8` triple witness is an *exact per-tree* signal that needs no separate dense-reference fit, so this front-end becomes a redundant cross-check kept only for cold-start (before any tree has fit a triple).

**(4) Sobol arbiter (the final, exact word).** The purified per-table variance `S_u = σ²(f_u)/σ²(F)` (§08) is the **only exact** support-importance. It (a) feeds heredity's "important" test during training, and (b) drives display pruning at export. It is an arbiter/pruner, never an admission gate — a low-Sobol table stays in the complete inference support, merely hidden from the top-k display.

**(5) Table-size admission penalty (storage prior; R-TABLEBUDGET, on by default).** A high-order support over border-rich axes produces a large merged-grid tensor: a triple's purified table on the merged grid has `Π_{a∈u} extent_a` cells, where `extent_a = 1 (missing) + n_finite_intervals_a` is the per-axis merged extent (the explicit missing cell plus the realized finite-interval cells; §08, R-MERGEDCELL). Two near-`max_bin` axes already imply tens of thousands of cells, and a third pushes a single triple toward the §08 `max_table_cells` per-table budget and the total-bank budget. To keep that budget from being hit by a *hard* §08 `PbError::TableBudget` (or forcing the sparse-tensor fallback), this stage adds a **soft** down-weight on the candidate support proportional to its projected cell count, so the booster prefers an equally-good split on cheaper axes and only spends a large-tensor support when its Newton gain genuinely earns it.

The penalty is a multiplier `cellprior(support) = (budget_cells / max(budget_cells, projected_cells(support)))^{table_budget_beta} ∈ (0, 1]`, where `projected_cells(support) = Π_{a∈support} extent_a` is computed from the **per-axis realized merged extents so far** (the same `extent_a` §08 will use, so the prior tracks the storage cost it is protecting), `budget_cells` defaults to the §08 `max_table_cells`, and `table_budget_beta` (default `0.5`) tunes aggressiveness (`0` disables the penalty entirely). It is `1.0` for any support at or under budget — single splits and small pairs are never touched — and decays smoothly only as a support approaches the §08 ceiling, biting hardest on near-`max_bin` triples. Like every funnel stage it **only scales a non-negative gain and never gates**: a support whose realized gain dominates the penalty is still admitted (and §08 then stores it, via the sparse-tensor fallback if it crosses `max_table_cells`), so the penalty re-prioritizes which large tables get built without ever amputating a real interaction or touching I2.

This funnel **bounds the table set by construction** (heredity composition + whole-tree policy + the table-size prior steering away from gratuitously large tensors) without enumerating `C(n,3)`, and **does not bias toward fewer interactions on the exact path** — every realized co-split still emits its full table; only the *display* is pruned and only the *ordering* of equally-good large-support candidates is nudged toward the cheaper merged grid.

**`feature_weights` (additive knob, v1.5, exactness-neutral).** A user-supplied per-raw-feature weight vector `fw ∈ (0,1]^n` (default all `1.0`) needs no new mechanism: it folds into the existing soft-prior seam (07.3 step 4) as an extra multiplier `Π_{raw ∈ support} fw[raw]` on `soft(cand, chosen, prior)`. Like every other prior it only *down-weights* a non-negative gain, never gates — a `0`-weighted feature is treated as `ε`, discouraged but not forbidden (use `InteractionPolicy.groups` to exclude a feature outright). Config-only, never serialized.

### 07.4a The `wht8` online interaction-screening prior (OWNED/registered here)

A depth-3 oblivious tree is a function on `{0,1}^3`: its 8 leaf values map by **one fixed 8×8 transform — `wht8`** — to its 8 fANOVA coefficients: **1 constant + 3 main (`c_i`) + 3 pairwise (`c_ij`) + 1 triple (`c_123`)**, taken under *that tree's* per-cut `w`-marginals. `wht8` is a **frozen O(8) Walsh–Hadamard / Möbius transform**: under uniform measure on each binary cut it is exactly `H₂^{⊗3}` with `c_S = ⟨L, χ_S⟩`, `χ_S(b) = ∏_{j∈S}(−1)^{b_j}`; under the general product measure it is the same orthogonal structure with `w`-centered indicators `ψ_j(b_j) = b_j − w_j(1)`. It is **exact, O(8)**, and is **owned and registered under §07** here; it is **invoked at §06 leaf estimation** (§06.4), where the 8 leaves are already in hand.

**Online accumulator.** As each tree is fit, run `wht8` on its 8 leaves (~24 flops). By **Parseval**, that tree's contribution to per-order **variance** is closed-form — the sum of squared coefficients within each `|S|` shell: `σ²_main(tree) = Σ_{|S|=1} m_S·c_S²`, `σ²_pair(tree) = Σ_{|S|=2} m_S·c_S²`, `σ²_triple(tree) = m_{123}·c_123²` (with `m_S` the product of the `w`-cut variances over `S`). Maintain a **running per-feature-set, per-order variance accumulator**, updated **per tree in O(8)**, keyed by each tree's canonicalized (sorted-distinct) raw-feature support.

**Triple witness → heredity funnel.** The triple coefficient `c_123²` is an **exact witness** that *this tree fit a real 3-way interaction*. Use it as a **soft online triple-detection prior** feeding the heredity admission funnel (07.4 stage 1's "important" test and `prior.triple`), complementing and **retiring much of** the separate dense-reference NID/META-ANOVA triple detector (07.4 stage 3): the online witness is **sharper than FAST-RSS**, is computed **off the inner loop** with no extra fit, and reflects what the booster actually realized rather than a pre-training surrogate. It enters only through the 07.3-step-4 soft seam.

**THE CRITICAL CAVEAT (stated in full).** The `wht8`-derived per-order variance is a **SCREENING SIGNAL, NOT the audited ensemble Sobol**:
- The per-tree coefficients live on **each tree's OWN 2-point grid under that tree's `w`-marginals**; different trees cut different borders, so you **CANNOT sum coefficients across trees** — doing so drops cross-tree covariance. The accumulator aggregates per-order *variance contributions*, never the coefficients themselves, and even that aggregate is a heuristic ranking, not a measured ensemble variance.
- Under `RefMeasure::Joint` (`w`) the axes couple and the clean orthogonal product form **degrades to a heuristic** (the `m_S` factorization no longer holds exactly).
- It must **NEVER touch the §08 invariant gates** (`ThreeWayEqual` / `VarianceSum` / `Purity` / `Reconstruction`) — those stay on the **merged-grid purified bank** (§08). The seductive idea "`wht8` replaces §08 Lengerich purification" is **WRONG and explicitly rejected**: §08 accumulates raw then runs one single-pass purify on the merged grid; `wht8` cannot cross the merged-grid alignment (grids disagree across trees) and adding a "`wht8` == Lengerich" equivalence would introduce a second, untested purification path that can silently break I2.
- It is a **SOFT prior that never hard-gates** (it only down-weights a non-negative gain through 07.3 step 4), so it is **exactness- and determinism-neutral by construction**. The accumulator is in-fit scratch — never serialized into the `Model` or `TableBank`.

(For the *exact* per-tree order-3 Faith-Shap cross-check that the same coefficients enable, see §13: it is a test oracle that must agree bit-close with the §08 mass-moving result, an invariant check, not a perf or admission mechanism.)

### 07.5 Monotone constraints — the per-level joint leaf-clamp

Standard XGBoost/LightGBM bound-propagation is built for asymmetric per-node trees and does **not** port verbatim: an oblivious tree applies **one shared split per level to all `2^k` nodes at once**, so feasibility and clamping must hold *jointly across the whole symmetric layer* (the CatBoost adaptation, research/06-4).

Mechanism. Treat the 8 leaf values as one vector with per-leaf bounds `[lo_ℓ, hi_ℓ]`, initialized `(−∞, +∞)` at the root; `w_ℓ = clip(−G_ℓ/(H_ℓ+λ), lo_ℓ, hi_ℓ)`. When the level-`L` shared split is on an `Increasing`-constrained feature, for each parent cell splitting into left child `L` (lower values) and right child `R`:
- midpoint `m = (w_L + w_R) / 2`;
- propagate to **all descendants**: left-subtree cells `hi ← min(hi, m)`, right-subtree cells `lo ← max(lo, m)` (reverse for `Decreasing`); unconstrained-feature splits inherit `[lo, hi]` unchanged.

A candidate level is **monotone-feasible** iff, across every cousin pair `(ℓ_L, ℓ_R)` differing only in the constrained-feature bit, `w_L ≤ w_R` (for `Increasing`). Any violation after clamping ⇒ level gain `−∞` ⇒ `Reject`. If *no* candidate is feasible the tree terminates early (07.3); the standard mitigation is to raise `max_bin` on constrained features (§03) so coarse borders don't wipe out feasible thresholds.

Monotonicity then holds on the **total score** (and on premium via the log link) and survives purification on the **total function and the constrained 1-D main-effect table** — but **not** on each 2-D/3-D interaction cell viewed in isolation (higher-order purified slices can be non-monotone alone). Stated in the export docs (§10) so cells aren't misread. `intermediate`/`advanced` bound-relaxation is degenerate at depth 3 (each constrained feature splits at most once) — `basic`-style enforcement only in v1.

### 07.6 Credibility & smoothness on the exact path

`CredibilityFloor` shapes *which* shared levels may fire (07.3 step 2). `path_smooth` shrinks each leaf toward its oblivious-tree parent, `w_node = w·(n/path_smooth)/(n/path_smooth+1) + w_parent/(n/path_smooth+1)`, stabilizing thin/low-exposure cells — applied **after** the monotone clamp, then re-clamped so smoothing cannot cross a monotone bound. All value-level: structure, ≤3-feature property, and exactness untouched. **No in-training Whittaker-Henderson / ICEnet graduation on the audited path** — those change table values and break reconstruction; graduation is an opt-in *post-export* mode with a "not bit-exact" stamp that flips `ExactnessMode::Approximate`.

### 07.7 Upholding I1/I2 and serving the three aims

- **I1.** `InteractionPolicy` with `max_order ≤ 3` is the *only* gate that shrinks the candidate set, and it enforces the budget on **distinct raw features via provenance** (a categorical-TS axis or one-hot column never double-counts). Early-termination yields `depth < 3`, still a valid `ObliviousTree`. Nothing here grows a non-symmetric node.
- **I2.** Every mechanism is either a candidate mask (heredity/policy/FAST/detector/`wht8` prior) or a leaf-value clamp (monotone/credibility/path_smooth) on a *fixed* oblivious structure ⇒ leaves stay piecewise-constant, support stays ≤3 raw features, and joint-boost-then-single-purification keeps `purify(Σ trees) = Σ purify(trees)` (§08 linearity). No §07 technique trips the firewall; all are `Exact`-preserving. The `wht8` accumulator only *reads* fitted leaves and only *down-weights* a non-negative gain — it never touches §08 gates (07.4a caveat).
- **Accuracy.** Spends the budget on data-supported supports (FAST/triple/`wht8` priors), enforces correct monotone priors (near-free, large downside if wrong), and avoids staged-boosting mis-convergence — the gap-closing playbook's "accuracy heart" (Tier 0).
- **Decomposable.** Whole-tree semantics make "permitted interactions" == "realizable tables"; `max_interaction_order=1/2/3` is a filing dial that dominates EBM at order 3 and offers an additive safe harbor.
- **Fast.** Heredity composition replaces the `C(n,3)` scan; FAST is `O(b²)`/pair on existing histograms; the `wht8` online prior is O(8)/tree and **removes** an off-inner-loop dense-reference detector fit (and the periodic purification refresh of the heredity "important" test) on interaction-rich fits; `constrain_candidate` is `O(1)` extra per candidate (bitset test + 8-cell clamp), so the search stays row-count-independent.

### 07.8 Complexity

| Mechanism | Cost |
|---|---|
| `InteractionPolicy::admissible` | `O(g)` bitset ops over `g` groups, per level |
| Monotone clamp | `O(8)` per candidate (cousin-pair check + clip) |
| Credibility floors | `O(8)` per candidate, on existing histogram counts |
| Heredity build | `O(pairs admitted)`, composition not enumeration |
| FAST-RSS | `O(b²+N)` setup + `O(b²)` per admitted pair |
| Table-size penalty | `O(|support|)` — product of ≤3 per-axis merged extents + one `powf`, per candidate |
| `wht8` online prior | `O(8)` per fitted tree (one transform + per-order accumulator update), off the inner loop |
| Triple detector | one dense-reference fit + `O(detector)`, off the inner loop (largely retired by `wht8`) |
| Sobol arbiter | reuses purified table variances (§08), zero model calls |

### 07.9 Testing

- **Property (proptest):** every fitted `ObliviousTree` satisfies `distinct(raw(splits)) == depth ≤ max_order`; under a group whitelist, no realized support spans two groups (the 07.1 export invariant). Failures map to `Invariant::FeatureBudget`.
- **Monotonicity oracle:** on a monotone target, the reconstructed 1-D table and total score are monotone to float tolerance; an anti-monotone target forces `depth < 3` early-termination, not a violation or `PbError`.
- **Soft-not-hard:** a planted XOR interaction (RSS-on-residual ≈ 0, Newton gain large) is still recovered — proving the prior never gates.
- **Order-of-operations:** a heavily down-weighted (`soft→ε`) but policy-permitted split can still win on raw gain; a policy-forbidden split is never selected regardless of prior.
- **Table-size penalty soft-not-hard (R-TABLEBUDGET):** on a border-rich fixture, a near-`max_bin` triple is down-weighted (`cellprior < 1`) relative to an equally-gainful cheaper support, yet a border-rich triple with dominant Newton gain is **still admitted** (and §08 stores it within budget or via the sparse-tensor fallback) — proving the storage prior re-prioritizes without gating; `table_budget_beta = 0.0` reproduces the un-penalized ordering exactly.
- **`wht8` witness & caveat:** on a planted 3-way fixture, `c_123² > 0` flags the triple online (the witness fires) and raises its `prior.triple`; the same fixture proves the prior is soft (a high-`wht8`-screened-but-low-gain support is still beaten on raw Newton gain). A determinism check confirms the `wht8` accumulator is identical at `n_threads ∈ {1,2,8}` and never appears in the serialized `Model`/`TableBank`. (The exact-equivalence oracle — `wht8` per-tree order-3 Faith-Shap vs the §08 mass-moving result — lives in §13.)
- **Staging:** on a correlated-feature fixture, joint boosting recovers the true main/interaction split where simulated mains-then-interactions staging mis-attributes mass (GAMI-Tree regression).
- **§08 gates:** every fixture passes the five Invariant checks (Reconstruction, MassConservation, Purity, VarianceSum, ThreeWayEqual), proving §07 never bends I2 — and confirming the `wht8` screening signal never touches those gates (07.4a caveat).

### 07.10 Open forks (recommended defaults)

1. **`HeredityMode` default — `Strong` vs `Weak`.** *Recommend `Strong`* (no orphan high-order effects; usually spurious otherwise), benchmark-gated against `Weak` on interaction-rich data, since the 3rd-order extension of the pairwise GAMI heredity literature is a reasonable but unvalidated extrapolation (research/06-4).
2. **Triple detector on by default in v1.5?** *Recommend off* — the 07.4a online `wht8` witness already provides an exact, no-extra-fit triple signal, so the dense-reference detector is needed (if at all) only for cold-start before the first triple is fit; it is exactness-neutral either way, so promotion is low-risk.
3. **`heredity_tau` / `heredity_floor` defaults (`1e-4` / `0.1`).** Placeholder values; tune so heredity prunes table *bloat* without capping achievable accuracy (too-strict admission silently lowers the ceiling — research/06-4 warning).
