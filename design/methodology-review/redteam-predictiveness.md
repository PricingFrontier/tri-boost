# Red-Team Critique: The Predictiveness Strategy

**Verdict in one line:** The gap-closing plan is *good but not the best*. The single biggest accuracy lever it leaves on the table is **not** distillation — it is the conflation the prompt flags: **depth-3-as-3-splits + mandatory obliviousness**, when the *product requirement* (≤3 distinct raw features → exact ≤3rd-order tables) permits strictly more expressive trees at zero decomposability cost. Distillation is a real but secondary bet. The reference-measure default holds up. The booster *set* is right; the *priority* is mildly wrong.

---

## 1. The split-finder cage is over-tightened (the biggest miss)

**Spec decision (§06.2, I1):** every tree is a depth-3 *oblivious* tree — exactly 3 levels, one shared `(axis, bin_le)` per level, ≤8 leaves, ≤3 distinct raw features.

**The conflation.** The *product requirement* is "≤3 distinct raw features per tree" (so the fANOVA truncates at order 3). The spec *implements* this as "≤3 splits, oblivious." These are not the same. A tree on ≤3 distinct features but with **more than 3 splits** (a feature reused on a deeper level) or **asymmetric structure** is *still* a sum of functions of ≤3 features → *still* exactly ≤3rd-order decomposable on the merged grid. The spec's own research confirms the governing principle: order is set by *distinct-feature support*, not split count (accuracy doc line 99 makes exactly this argument for linear leaves; it applies a fortiori to extra constant splits). The purification machinery (§08) operates on the merged union grid and does not care how many splits produced a cell — it only cares which axes the tensor depends on.

**Steelman of the spec's choice.** Obliviousness buys three real things: (a) branch-free 8-cell lookup inference (aim 3), (b) the rank-1-tensor-per-tree mental model that makes "model == tables" clean, (c) CatBoost-grade self-regularization. Research/01 confirms oblivious costs ≈0 Elo *at equal depth*. And depth-3 keeps each tree a tiny `[f32;8]` — trivially cache-resident.

**Why it's still leaving accuracy on the table.** The spec itself flags the live risk (research/01 close): *"depth-3 is more aggressive than CatBoost's depth-6 default, so the per-tree weakness is stronger."* Two structurally-exact relaxations recover per-tree strength:

- **(A) Repeated-feature depth (finer per-feature resolution).** Allow a feature to be re-split at a deeper level — e.g. a tree on {age, region} that splits age, then region, then age *again* at a finer border. Still 2 distinct features → 2D table, exact. This gives finer per-axis resolution and more interaction cells *per tree*, i.e. each weak learner lands more signal. This directly attacks the named "depth-3 weak → need many trees" failure mode — the same problem distillation, Nesterov, and fully-corrective refit are all deployed to patch *downstream*. **Aim 1: a real lift** (plausibly comparable to one of the Tier-1 levers, dataset-dependent — biggest on interaction-rich/high-cardinality-ordinal data, ~0 on flat data). **Aim 3:** the lookup is now `2^d`-cell, but `d` capped at, say, 5 is still 32 floats — L1-resident; inference cost is marginal. **Decomposability: SAFE** (distinct-feature invariant unchanged; merged-grid purification absorbs the extra cells natively). **Cost:** the gain scan becomes `O(2^d · axes · bins)` and the I1 guard must count *distinct raw features* not splits (the provenance machinery already exists — `AxisProvenance.raw` and the `HashSet` check in §2.5 are *already* written against distinct raw features, not split count). This is a small engine change.

- **(B) Asymmetric growth on ≤3 features.** Drop the shared-split-per-level rule but keep the ≤3-distinct-feature budget. This recovers CatBoost's documented oblivious weakness (research/01: "cannot adapt the split to a local region… wastes depth"). Bigger accuracy upside than (A), but it **forfeits the branch-free lookup** (aim 3) and complicates the rank-1 story — a genuine three-way tension. Still exactly decomposable.

**Recommendation:** Adopt **(A)** — make the cap `max_distinct_features=3` with a separate, raisable `max_depth` (default 3 = today's behavior, exposed up to ~5). It is the *cheapest* unexploited lift, it is the most-aligned with the product (zero decomposability cost, near-zero inference cost), and it attacks the root cause the three v2 boosters attack symptomatically. Treat **(B)** as a benchmarked research fork, not default. **This is the single biggest accuracy lever the spec is not using**, and it is structurally cleaner than distillation because it needs no external teacher.

---

## 2. CatBoost-teacher distillation — right to ship, wrong as *primary* bet

**Spec decision (§09.2):** distillation is "the highest-upside *new* lever," default blend 0.25 (favoring the teacher), CatBoost default teacher.

**The hard question the spec underweights:** does distilling beat *directly training* the order-capped student? Distillation can only transfer the teacher's **≤3rd-order projection** — and that projection is something we can fit directly. The honest mechanism by which distillation helps an order-capped student is **not** "importing structure we couldn't represent" (we can represent all of it); it is **denoising**: the teacher's soft scores are a lower-variance regression target than raw labels, so the student spends its capacity on signal rather than label noise. That is a real but *bounded* effect, and it is **largely the same variance win that bagged ensembling and fully-corrective refit already deliver** — so its marginal value over the rest of the stack is smaller than "highest-upside new lever" implies. The META-ANOVA evidence (student occasionally beats teacher) is on *neural* teachers distilled into additive models; with a CatBoost teacher that is *itself* an order-capped-ish oblivious ensemble, the headroom is thinner.

Worse, the default `blend=0.25` (75% teacher weight) is **aggressive and can bias held-out accuracy**: it down-weights the true labels in favor of a target that is only as good as the teacher's *own* ≤3-order projection, and it imports the teacher's miscalibration/bias on tails (a real concern on Tweedie/Gamma). On a well-specified deviance loss with good features, a 75%-teacher target can *underperform* direct training.

**Verdict: keep it, demote it, raise the default blend.** It is genuinely free (one O(n) gradient pass) and helps on noisy / high-cardinality data — ship it. But (i) it is not the primary accuracy bet (Lever §1 and the core funnel are), and (ii) **default blend should be ~0.5, not 0.25**, with 0.25 reserved for explicitly noisy regimes. Also: the spec should run direct-vs-distilled as a *gated A/B per dataset* (the teacher is fit anyway), and only keep the distilled student when it wins held-out deviance — otherwise distillation is a crutch that imports a slow teacher's projection we could have fit better ourselves. **Aim 1: modest lift** (smaller than the spec implies, concentrated on noisy/high-cardinality data). **Decomposability: SAFE** (target-only). **Cost:** the external teacher fit — *not* negligible operationally (it doubles the user's training pipeline and reintroduces the slow oblivious teacher the library exists to replace).

---

## 3. Reference measure default (product-of-marginals) — holds up

**Spec decision (§08/§2.7):** Laplace-smoothed empirical product-of-marginals, default.

**Verdict: correct, for the right reason.** The key insight the spec gets right is that **`w` does not affect the fit** — purification under any `w` reconstructs the *same* `F` (Reconstruction invariant holds for all `w`); `w` only re-attributes variance *between* tables. So "does product-`w` bias held-out accuracy?" is the wrong frame: held-out predictions are **identical** under product, uniform, or joint `w` because inference sums the complete bank. `w` is a *display/attribution* choice, post-hoc recomputable (§08), not a fit choice. Product-`w` is the right default because it keeps `σ²(F)=Σσ²(f_u)` and exact equal-split SHAP (both branch on `w`), stays positive, and dodges Hooker extrapolation. The one substantive caveat — under correlated features, product-`w` "pure" effects can be empty-cell artifacts — is correctly hedged (build the joint path, measure drift). **No change.** The only thing I'd tighten: state explicitly in §09 that ensemble averaging and distillation interact with `w` only through display, so the "single shared `w`" rule (§09.5 rule 4) is about *attribution coherence*, not accuracy.

---

## 4. The booster set and priority

**The set is right; nothing material is missing** *within the value-only / target-only / weight-only class* — given Lever §1 is adopted. Fully-corrective refit (linear-in-leaves ridge), Nesterov/AGBM, MVS, bagged greedy selection are all correctly exact and correctly motivated. Two priority corrections:

- **Fully-corrective leaf refit is under-prioritized.** It is Tier-2/v2/default-off, yet it is *cheap* (an `8T`-dim Cholesky), exactness-trivial, and attacks tree-count directly → **fewer, smaller tables** (helps aims 1 *and* 2). It deserves promotion to **v1.5, benchmark-gated-on**, ahead of Nesterov. It is a more reliable lift than distillation and has no external dependency.
- **The default-off Nesterov/refit fork (§06.9/§09.8) is right.** Both cost per-round compute for a tree-count win that must be *measured*. Default-off pending a pricing-data benchmark is the correct, honest call. Hold.
- **Bagged greedy selection default-off is right** — it's K× training for ~0.5–1.5% variance, and the "beat-EBM must be bagged-vs-bagged" honesty note is exactly correct (EBM bags 14× internally). Hold. The `OuterBag` on-ramp is the right first ship.

---

## Bottom line

| Decision | Verdict | Action |
|---|---|---|
| Depth-3 **=3 splits + oblivious** | **Should change** | Decouple: `max_distinct_features=3`, raisable `max_depth` (default 3). Biggest unexploited, decomposability-free lift. Asymmetric growth = research fork. |
| CatBoost distillation as **primary** bet, blend 0.25 | **Should change** | Keep, **demote** from "highest-upside," raise default blend to ~0.5, gate direct-vs-distilled per dataset. |
| Product-`w` default | **Holds up** | No change — `w` is attribution-only, doesn't touch held-out accuracy. |
| Fully-corrective refit at v2/off | **Should change** | Promote to v1.5 benchmark-gated; cheaper and more reliable than distillation. |
| Nesterov/ensemble default-off forks | **Hold up** | Correct; cost must be measured first. |

**The single biggest accuracy lever the spec is NOT using:** lifting the implementation conflation between "≤3 distinct features" (required) and "3 splits / oblivious" (a choice) — specifically allowing **repeated-feature depth** so each weak learner carries more signal. It is exactly ≤3rd-order decomposable, near-free at inference, needs no external teacher, and attacks the very "depth-3 trees are too weak" problem the entire §09 booster stack exists to patch. Expected lift is dataset-dependent (largest on interaction-rich / ordinal-heavy data, ~0 on flat) but it is the most *structurally aligned* gap-closer in the design space and should be the predictiveness strategy's first move, not distillation.

**Relevant files:** `/home/ralph/suite/tri-boost/spec/06-oblivious-boosting-engine.md` (§6.2 split-finder, §6.9 fork — where the depth decoupling lands), `/home/ralph/suite/tri-boost/spec/09-predictiveness-boosters.md` (§09.2 distillation blend default, §09.3 refit prioritization), `/home/ralph/suite/tri-boost/spec/00-spec-skeleton.md` (§2.5 `ObliviousTree`/`Split`, §3 I1 — the invariant is *already* written against distinct raw features, making the decoupling low-cost), `/home/ralph/suite/tri-boost/research/06-techniques/2-accuracy-model-class.md` (line 99 — the spec's own statement that ≤3-distinct-feature support keeps fANOVA at order 3 regardless of per-feature complexity).
