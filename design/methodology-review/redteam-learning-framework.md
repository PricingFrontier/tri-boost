# Red-Team Critique — The Learning Framework (Cluster: §06/§07/§08/§09)

**Scope:** Is gradient boosting + accumulate-then-purify + the soft-prior interaction funnel the *best* fitting framework for {max predictiveness, exact ≤3rd-order decomposability, fast}? Order-cap=3 is fixed; I attack every *design choice* made to hit the aims. Steelman first, then strike.

---

## Verdict at a glance

| # | Spec decision | Verdict | Better alternative exists? |
|---|---|---|---|
| 1 | **Tree = depth-3 = exactly 3 distinct splits, oblivious** | **SHOULD CHANGE (partially)** — biggest finding | Decouple *order cap* (≤3 features, fixed) from *depth/symmetry* (a free choice). Allow feature-repeating / deeper-but-≤3-feature trees. |
| 2 | Gradient boosting as the fitting procedure (vs backfitting / penalized fANOVA / GAMI) | **HOLDS** | No — boosting is the right base; rivals are weaker or break under correlation. |
| 3 | Accumulate-then-purify **once at the end** | **HOLDS** (with one caveat) | No — but identifiability should inform *selection*, not the optimizer. |
| 4 | Stagewise (not totally-corrective) as default | **HOLDS, under-sold** | Promote fully-corrective refit from v2→v1.5; it is a near-free accuracy+sparsity lever. |
| 5 | Soft-prior interaction funnel (heredity/FAST/Sobol, never a hard gate) | **HOLDS — genuinely well-designed** | No. Steelmanned below. |
| 6 | Greedy oblivious *level* split search misses interactions | **HOLDS** (search is fine; see #1 for the real gap) | Partially — the limiter is structure, not greed. |

---

## 1. The depth-3 / oblivious conflation — the one decision that leaves real accuracy on the table

**The spec's claim** (skeleton §2.5, §06 6.2; echoed verbatim across `research/02`, `research/03 §4`, `research/06/README`): "every tree is a depth-3 oblivious tree (≤3 features)." The two clauses are treated as *the same thing* everywhere. They are not.

**The decomposability requirement is `≤3 distinct raw features per tree`.** That, and only that, is what guarantees the ensemble is a sum of ≤3-feature tensors, hence exactly ≤3rd-order fANOVA-decomposable. The *proof* of losslessness (research/03 §4.2, spec §08.1–08.2) rests on **one fact**: each tree is piecewise-constant on the **union grid of realized thresholds per feature**, so it maps losslessly onto the merged grid. **That proof is completely indifferent to how many splits produced those thresholds, or whether the tree is symmetric.**

Concretely: a tree that splits feature *i* at level 0, feature *j* at level 1, and **feature *i* again** at level 2 (a different border) uses **2 distinct features**, is depth-3, and is **exactly 2nd-order decomposable** — it contributes a richer `f_i` main-effect (two breakpoints, three cells on *i*) and an `f_ij` table, all on the union grid. Likewise a depth-4 or depth-5 tree confined to ≤3 distinct features, or an *asymmetric* tree on ≤3 features, is **still exactly ≤3rd-order decomposable**. The spec forbids all of these ("Non-symmetric growth … and >3-raw-feature encoded axes are forbidden on the exact path", I1), conflating the product requirement with a self-imposed structural tax.

**What this costs (the accuracy aim).** The spec itself names the wound repeatedly and never connects it to the cure:
- "depth-3 is **more aggressive** than CatBoost's depth-6 … the per-tree weakness is *stronger*" (research/01, design/02 §3).
- Oblivious trees "**waste depth** … one condition forced across the whole level even where it is suboptimal" (research/01 §6).
- The entire §09 booster suite (Nesterov, fully-corrective refit, multi-step Newton) exists to **compensate for too-weak/too-many trees** — a problem largely manufactured by capping splits at 3 and forcing symmetry, *not* by the order cap.

A depth-3 oblivious tree gives each feature **at most one breakpoint** and the ensemble only resolves a main effect through *many* trees re-splitting the same feature at different borders across rounds. Allowing a single tree to spend extra depth on a feature it has already chosen (finer per-feature resolution) or to break symmetry directly buys back the resolution and local adaptivity that obliviousness throws away — the exact gap CatBoost's depth-6 enjoys over a depth-3 cap. This is **bias-reducing**, unlike everything in §09 (which is variance/convergence only and, as the spec honestly states, "none lift the order-3 bias ceiling").

**Decomposability safety:** *fully preserved.* ≤3 distinct features ⇒ ≤3rd-order ⇒ all five invariants hold unchanged. The union-grid accumulator and purification cascade need **zero** modification — they already operate per-feature-set, not per-split. Reconstruction/Purity/VarianceSum/ThreeWayEqual are agnostic to tree shape.

**Cost of the change:**
- *Inference:* a depth-*d* (>3) tree on ≤3 features loses the fixed 8-cell `[f32;8]` lookup and branch-free 3-bit gather. **But** — and this is decisive — the shipped artifact is the **LUT-sum table bank, not the trees** (research/06/1, spec §10). Inference scores from the *purified tables*, whose cost is "independent of tree count" and independent of tree *depth*. The 8-cell lookup is a *training-time* convenience, not the inference product. So the headline inference-speed aim is **untouched**.
- *Training:* the histogram engine must handle depth >3 (more levels) or asymmetric nodes (per-node split). Asymmetric growth forfeits the oblivious-search cost saving and the joint monotone clamp's simplicity. **The cheapest, highest-ROI move is the conservative one: keep oblivious + symmetric, but (a) drop the "distinct features == depth" identity, allowing a feature to repeat across levels, and (b) allow depth >3 as long as distinct-feature-count ≤3.** This keeps the entire oblivious fast-path and joint-clamp machinery, costs only a relaxation of the I1 construction check (`distinct(raw) ≤ 3` instead of `== depth`), and recovers most of the per-feature-resolution loss.

**Recommendation:** Split I1 into the *real* invariant (`distinct raw features ≤ 3` — keep, it is the product) and the *incidental* one (`depth == distinct features`, oblivious-symmetric — make it a **default, benchmark-gated config**, not a hard wall). Add `max_depth ∈ {3,4,5,6}` with `max_distinct_features = 3` as the binding constraint. **Estimated impact: the single largest available bias-reduction (plausibly comparable to one full §09 booster, and complementary to all of them), at low engineering cost and zero decomposability risk.** This is the one place the spec is leaving accuracy on the table by construction.

> **Steelman (why the spec chose as it did):** the depth-3 identity makes I1 *trivially* checkable, the 8-cell tensor maximally cache-friendly, the joint monotone clamp clean, and determinism easy. It is a defensible *v1* simplification. The error is **elevating it to a hard invariant** ("forbidden on the exact path") rather than a tunable default — it reads as a product requirement when it is an implementation convenience.

---

## 2. Is gradient boosting the right fitting procedure at all? — HOLDS

The honest adversarial question: would **backfitting (cyclic coordinate descent over feature blocks)**, a **penalized functional-ANOVA / sparse additive model (group-lasso over a ≤3rd-order spline/tensor basis, e.g. COSSO/SPAM/SS-ANOVA)**, or a **GAMI-Net / GAMI-Tree / EBM-exact** fit the ≤3rd-order target *more directly*?

- **Backfitting / EBM cyclic round-robin** fits one component at a time to convergence. The spec explicitly and correctly **rejects mains-then-interactions staging** (decision 1, §07.1) citing the GAMI-Tree result (research/06/4 line 90, research/03 §3.3): under **correlated features**, two-stage / sequential-by-order fitting **mis-converges and mis-attributes mass**. Insurance/tabular data is precisely the correlated regime. Joint boosting over all admitted supports + one final purification provably avoids this. **Boosting wins here.**
- **Penalized fANOVA / sparse-additive (group lasso on a fixed ≤3rd-order tensor basis)** is the most serious rival — it fits *all* selected components jointly and gives identifiability via the basis. But it requires (a) pre-committing the basis/knots (loses the data-adaptive thresholds that make pattern-boost's union grid minimal-and-exact), (b) solving a large coupled convex program (loses the row-count-independent histogram speed — a direct hit to the FAST aim), and (c) C(n,3) candidate blocks unless paired with a screen — i.e. it needs *the same heredity/FAST funnel anyway*. It would fit the structure "more directly" only at a real cost to the speed aim and the adaptive-grid exactness story. **Not better for these three aims jointly.**
- **GAMI-Net / NODE-GAM** are *soft/neural* and cap at order 2; they break exactness (research/06/2 lines 109–117) and don't port. Cite for motivation, not adopt.

**Verdict:** Gradient boosting is the right base learner. Its decisive advantage for *these* aims is the combination of (i) data-adaptive thresholds → minimal exact union grid, (ii) row-count-independent histogram speed, and (iii) joint fitting that survives correlation. The spec's choice is **correct and well-justified** — keep it. The improvement lever is *inside* boosting (#1, #4), not switching frameworks.

---

## 3. Purify once at the end vs. enforce identifiability during training — HOLDS

The tempting critique: "the optimizer never sees the canonical decomposition, so it optimizes an unidentifiable objective; purify *during* training so it sees pure effects." **This is wrong, and the spec is right to reject it**, for a precise reason: purification is a **mass-preserving, prediction-invariant** linear operator (Lengerich Cor. 2.2; spec §08.3). It changes *attribution*, never *predictions*. Boosting optimizes **prediction loss**, which is invariant to which order a unit of mass is attributed to. Purifying mid-training would therefore **change nothing about the fitted function** while adding cost every round — pure waste. Identifiability is an *attribution* property; the loss does not depend on it. Purify-once-at-the-end is **optimal**, and the linearity property makes it bit-identical to streaming purification.

**One genuine caveat (not a flaw, a dependency):** the *selection* machinery (§07 heredity uses running Sobol from the purified bank) **does** depend on purified variances mid-training. The spec handles this correctly with a *lazily-refreshed* running bank (§07.4). The right principle — which the spec follows — is: **purify lazily for selection, once exactly for export; never inside the gradient step.** Holds.

---

## 4. Stagewise vs totally-corrective — HOLDS but UNDER-PRIORITIZED

Default stagewise boosting (fit newest tree's leaves only) is the right v1 default. But the spec parks **fully-corrective leaf refit at v2, default-off, "benchmark-gated"** (§06.9, §09.3). I push back on the *priority*, not the mechanism:

- With structure frozen, the model is **linear in the 8·T leaf scalars** — a single ridge/IRLS solve (§09.3). It is **exactly decomposable** (only leaf scalars change), corrects stagewise over-shrinkage, and **reduces tree count → smaller tables → faster table-sum inference**. It serves *all three aims simultaneously* and is one of the very few levers that is both bias-relevant-ish (recovers fit slack) and table-shrinking.
- It is **especially synergistic with #1**: deeper/feature-repeating trees produce fewer, stronger trees whose leaves benefit most from a joint re-solve.

**Recommendation:** promote fully-corrective **final** refit (`every_k=None`) to **v1.5, default-ON** pending one benchmark, not v2 default-off. The `8T×8T` Cholesky is cheap relative to training. This is a low-risk, multi-aim win the spec currently under-weights. Totally-corrective (interleaved every-k) stays v2 — its per-round solve cost is the real question.

---

## 5. The soft-prior interaction funnel — HOLDS (steelman, no strike)

This is the **best-designed part of the cluster** and deserves explicit endorsement against the obvious attack ("statistical screens will amputate real interactions"). The spec's load-bearing insight (§07.1, §07.3 step 4, research/06/4 line 86) is exactly right: **FAST's RSS-on-residual ≠ the booster's Newton gain**, so any hard screen would discard interactions the booster would have found (the planted-XOR case, §07.9). Making the funnel a **soft multiplier on non-negative gain, never a gate**, with the **only hard gate being the user's structural `InteractionPolicy`**, is the correct architecture. The heredity-by-composition trick (avoids C(n,3)) serves the speed aim without capping accuracy. The one defensible worry — that `HeredityMode::Strong` could silently lower the ceiling by starving orphan triples — is already flagged as a benchmark-gated fork (§07.10). **Keep as-is.** Minor note: ensure the Strong-vs-Weak default is decided by the interaction-rich benchmark before v1 freeze, since the 3rd-order heredity extension is, by the spec's own admission, "a reasonable but unvalidated extrapolation."

---

## 6. Does greedy oblivious *level* search systematically miss interactions? — HOLDS (the limiter is #1, not greed)

A direct functional-ANOVA fit evaluates a 2D/3D cell jointly; the greedy level-search picks one shared split at a time. Does this miss pure interactions (XOR-like) with zero marginal signal? In principle a *fully* greedy first-level search could — but the spec defends against this two ways: (a) the **FAST/triple soft prior up-weights** supports with interaction signal so the booster *tries* the right axes early, and (b) **boosting is iterative** — a weak first-round main split on *i* leaves residual structure that a later round splits on *j*, recovering the pair across rounds. The §07.9 planted-XOR test asserts recovery. The residual greedy risk is real but second-order, and is the standard GBM tradeoff, not a pattern-boost-specific defect. **The far larger "missed structure" cost is the depth-3/oblivious cap of #1** (one breakpoint per feature per tree), not the greediness of the search. Fix #1 and this concern shrinks further. Holds.

---

## Bottom line

**The fitting *framework* is right; one structural over-constraint is wrong.**

1. **Change (highest impact, low cost, zero decomposability risk):** Decouple the order cap (`≤3 distinct features` — the real product invariant, keep hard) from depth/symmetry (an implementation default, not an invariant). Allow trees to **repeat a feature across levels** and to grow **deeper than 3 levels while keeping ≤3 distinct features** (and, as a later option, asymmetric ≤3-feature growth). This directly attacks the spec's own most-cited weakness — "depth-3 is too weak, needs many trees" — with *bias* reduction the entire §09 suite cannot provide, while the LUT-sum table export keeps inference speed and exactness fully intact. This is the headline recommendation.
2. **Re-prioritize:** Promote **final fully-corrective leaf refit** to v1.5 default-on (pending one benchmark) — a rare all-three-aims, exactness-safe, table-shrinking lever the spec parks at v2.
3. **Keep, endorsed:** gradient boosting as the base learner (beats backfitting/penalized-fANOVA/GAMI under correlation and on speed); purify-once-at-the-end (identifiability is attribution, invariant to the loss — mid-training purification is pure waste); joint-boost-then-single-purification; and the soft-prior heredity/FAST/Sobol funnel with the user policy as the sole hard gate. These are not just acceptable — they are the correct, well-reasoned choices for the three aims.

The spec is a strong design. Its one self-inflicted accuracy ceiling is mistaking a v1 implementation convenience (depth-3-as-3-splits, oblivious-symmetric) for a product requirement.
