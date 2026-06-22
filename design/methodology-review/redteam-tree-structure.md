# Red-Team Critique — Tree Structure Cluster

**The depth-3-as-3-splits + oblivious constraint, against the three aims**

## The decision under attack

The spec hard-codes a single structural object — the `ObliviousTree` — and bakes three *separate* commitments into one type:

1. **Order cap = 3** (REQUIRED): ≤3 distinct raw features per tree.
2. **Depth = #splits = #distinct features** (DESIGN CHOICE): `splits: Vec<Split>` of length `1..=3`, one shared `(axis,bin_le)` per level, and a construction invariant `distinct_raw == splits.len() == depth` (skeleton §2.5, §06 line 98, I1). A feature **cannot repeat across levels**.
3. **Symmetric/oblivious** (DESIGN CHOICE): one shared split per whole level.

The product requirement is only (1). The spec's own theorem (§01 line 42) states it cleanly: if every tree depends on ≤3 distinct raw features, `f_u ≡ 0` for `|u|>3`, *period* — **the theorem says nothing about depth, leaf count, or symmetry.** Research/03 §1.1 confirms it: a function of ≤3 variables has an fANOVA that truncates at order 3 *however richly it is represented*. So (2) and (3) are pure expressiveness choices riding on the back of a required cap, and the naming of I1 ("Depth-3 oblivious / ≤3 distinct raw features") fuses a choice to a requirement so tightly that the spec never separately defends the choice. That fusion is the single highest-stakes thing to interrogate.

## Steelman first

The oblivious-as-3-splits design is genuinely well-motivated, and three of its pillars hold up:

- **Inference.** 3 compares → 3-bit index → one read from `[f32;8]` in-register, branch-free, SIMD-across-rows, L1-resident (§10.2, research/01 §2). This is a real, defensible structural win and is load-bearing for aim 3.
- **The 8-cell tree *is* a rank-1 grid tensor**, so accumulate→purify→tables is mechanically trivial and the union-grid stays tiny. Clean.
- **Obliviousness costs ≈0 at the ensemble level.** This is well-evidenced (research/01 §3; closability.md (b)): CatBoost (oblivious) and LightGBM (leaf-wise) are within a few TabArena rungs; the symmetric constraint trades per-tree bias for variance and is recovered with more trees. *Keep obliviousness.* I am **not** recommending asymmetric/local splits — they would forfeit the branch-free lookup, complicate the rank-1-tensor story, and the evidence says they buy ~nothing in final accuracy. The asymmetric branch of the question is a steelman-and-reject.

So obliviousness survives. What does **not** survive scrutiny is pillar (2): `depth == #distinct features ≤ 3`.

## The strike: depth-3-as-3-splits leaves real accuracy on the table

The spec's own gap analysis names the wound twice (closability.md §31; design/02 §85): *"a depth-3 cap is **more aggressive than CatBoost's depth-6** — per-tree weakness is stronger."* But it then mis-attributes the fix entirely to downstream compensation (more trees, Nesterov, fully-corrective refit) and **never notices that the per-tree weakness is self-inflicted by conflating depth with feature-count**, not forced by the order cap.

Here is the decoupled object the spec excludes. Take an oblivious tree over **≤3 distinct raw features but with depth up to 6**, where a feature is *allowed to repeat across levels*. Example: features `{age, vehicle, age, age, region, vehicle}` across 6 levels → still exactly 3 distinct raw features → **still exactly ≤3rd-order decomposable** (the theorem only counts distinct features). But instead of binarizing each feature once (one threshold → 2 cells per feature → an 8-cell box), the tree now resolves `age` at up to 4 thresholds, `vehicle` at 2, `region` at 1 — a far finer piecewise-constant surface inside the *same* 3-feature interaction cell.

Quantify the expressiveness gap:

- **depth-3, 3-splits (current):** each feature gets exactly **1 threshold** → the 3-way table it can express is **2×2×2 = 8 cells**, i.e. each axis carved into 2 intervals. The *entire* triple interaction is forced through a single binary cut per axis.
- **depth-6 over ≤3 features (proposed):** up to **64 leaves**, thresholds distributable across the 3 axes (e.g. 4×4×4 = 64, or 8×4×2, etc.). The same 3-way interaction cell is now resolved at up to **8× finer per-axis granularity** and **8× more interaction cells** — a *strictly* richer member of the *same* order-3 function class.

This is not a marginal knob. It is the difference between EBM's tiny-LR stumps and CatBoost's depth-6 trees, *projected into the 3-feature subspace*. The spec correctly credits "full Newton splits vs EBM's tiny-LR stumps" as worth real Elo (closability.md §31) — but a 3-splits/8-cell tree is *much closer to the EBM-stump end of that axis than to CatBoost depth-6*. CatBoost's competitive accuracy is measured **at depth 6** (research/01 §5, the verified default); the spec adopts CatBoost's growth policy but throws away half its depth, then spends three downstream gap-closers (Nesterov, fully-corrective refit, multi-Newton — design/02 Tier-2, all "cost high/medium, benchmark-gated, v2") trying to claw back per-tree strength it discarded for *no decomposability reason*. **You are paying a v2-complexity bill to compensate for a v1 self-limitation.**

Mechanism of the loss: with one threshold per feature per tree, a single tree can only place its 8 leaf values at one binary partition of each axis. To approximate a curved 3-way surface (the realistic insurance case: age×region×vehicle risk is not a step function at one age cut), the ensemble must stack *many* trees, each re-binarizing the same axes at slightly different thresholds — exactly the "weak-per-tree → need many trees" pathology the spec acknowledges. A depth-6, 3-feature tree captures that curvature **within one tree**, sharply reducing tree count for equal accuracy.

## Decomposability safety — airtight

Does depth-6-over-≤3-features preserve exact ≤3rd-order decomposability? **Yes, with zero qualification.** The order of the fANOVA is a property of the *support set* (distinct features), not the representation's depth or leaf count. The spec's own theorem (§01 line 42) and research/03 §1.1 both establish this. A deeper oblivious tree on ≤3 features is still a piecewise-constant function of ≤3 features; its breakpoints just land at more thresholds. Accumulation onto the merged grid (§08.2) is *already* "broadcast a tree cell onto every sub-cell it covers" — it does not care whether the tree contributed 8 or 64 cells. **Purification, the five invariant checks, Sobol, exact Faith-Shap — all unchanged.** The merged grid already takes the *union* of realized thresholds per axis; a deeper tree simply realizes more of them. I1's *real* content (≤3 distinct raw via provenance) is untouched; only its accidental `== depth` clause is relaxed to `<= depth`.

## The cost — honest accounting

This is where the change must earn its place, and the costs are real but bounded:

- **Inference — the one genuine regression.** A depth-`d` oblivious tree needs `d` compares → a `d`-bit index → a read from `[f32; 2^d]`. At depth 6 that is 6 compares and a 64-float (256 B) table — still branch-free, still SIMD-across-rows, still L1-resident (research/01 §2 measures depth-6 oblivious inference as CatBoost's headline speed story). So the *per-tree* lookup is ~2× the work of depth-3. **But** fewer, stronger trees cut the tree count, and **path B (table-sum scoring) is entirely unaffected** — it is `O(|tables|)`, independent of depth *and* tree count (§10 line 106). For the explainable/deployed view, depth is free. Net inference impact: mildly negative on path A, neutral on path B, plausibly *positive* overall once tree-count drops. The `leaves: [f32; 8]` and `depth: u8 ∈ 1..=3` type must widen to `SmallVec<[f32; 64]>` / `depth ∈ 1..=6` — a real but contained struct change touching §02/§06/§10.
- **Fitting cost.** Per *level* the histogram-build + gain-scan is `O(2^d · n_axes · n_bins)` (§06 line 61). Going to depth 6 makes the per-tree structure search up to 8× the per-tree cost of depth-3 — but again offset by needing fewer trees. The dominant term (histogram build, `O(n·F)`) is per-level and grows linearly in depth, not exponentially, since the subtraction trick still applies. **This is the cost that must be benchmarked**, and it argues for `max_depth` being *configurable*, not jumped to 6 unconditionally.
- **Merged-grid / table size — negligible.** The union grid is already the union of realized thresholds (§08.1). A deeper tree realizes a few more thresholds per axis; the spec already budgets "a few hundred cells per axis." 3-D tensor memory `|Ω_i||Ω_j||Ω_k|` grows only with *distinct realized thresholds*, which deeper trees increase modestly, not with `2^depth`. Display pruning (top-k Sobol) is unaffected.

## Verdict and the concretely better design

**Pillar (1) order cap = 3:** fixed, correct, not challenged.
**Pillar (3) obliviousness:** **holds up — keep it.** Steelmanned above; asymmetric/local-split trees rejected (forfeit branch-free lookup + rank-1 tensor for ~0 accuracy per research/01 §3).
**Pillar (2) depth == #distinct-features == 3:** **should change.** This is the conflation. It is an unforced expressiveness cap that the order requirement does not imply, costs real accuracy (aim 1), and inflates tree count (hurting aim 3 via more trees and pushing three compensating mechanisms into v2).

**The precise change:**

- Replace the construction invariant `distinct_raw == splits.len() == depth` with `distinct_raw(splits) <= 3` and `splits.len() <= max_depth`. **Allow a feature to repeat across levels.** This is a one-line relaxation of I1's accidental clause; I1's load-bearing content (≤3 distinct raw via provenance) is unchanged and still gated.
- Widen the types: `ObliviousTree { splits: Vec<Split> (len 1..=max_depth), leaves: SmallVec<[f32; 64]>, depth: u8 (1..=6) }`; the leaf index becomes a `depth`-bit pattern (§10 already computes exactly this for early-terminated trees — the machinery exists).
- Add `Config.max_depth: u8` (default to be **benchmarked**, not assumed — start at 4 and let the §14 benchmark choose between 4 and 6), with the hard ceiling 6 and the **distinct-raw-feature budget of 3 enforced independently** in the split-finder's admissibility guard (§06.2 — which already checks `provenance[axis].raw` membership; it just must stop forbidding *re-use* of an already-used raw feature, while still capping *distinct* count at 3).

**Aim impact:** Accuracy (aim 1) — likely the **single largest within-cage predictiveness lever not yet in the spec**, of the same character as "full Newton splits vs EBM stumps" which the spec already values at real Elo; it directly attacks the acknowledged "depth-3 more aggressive than depth-6" deficit *at the source* rather than via v2 compensation. Speed (aim 2/3) — net plausibly positive via reduced tree count; path-B scoring unaffected; per-tree fit cost up, must be benchmarked. Decomposability (the requirement) — **provably preserved, bit-for-bit, no firewall interaction.**

**Cost:** a contained struct/type widening across §02/§06/§10 and one new config knob; a benchmark to set `max_depth`'s default and confirm the tree-count/accuracy trade. Modest, and far cheaper than the v2 mechanisms it partially obviates.

## Bottom line

The spec correctly fixed obliviousness and the order cap, and correctly rejected asymmetric trees. But it **conflated a required order cap with an unforced depth cap**, and in doing so chose the *weakest* member of the order-3 oblivious function class — an 8-cell box per tree — when a strictly more expressive, equally-decomposable member (depth up to 6 over ≤3 repeating features, up to 64 cells, 8× finer per-axis resolution) was available at modest, benchmarkable cost. **Change "depth-3 = 3 splits" to "configurable `max_depth` (≤6) under an independent ≤3-distinct-raw-feature budget, features may repeat across levels."** Keep everything else in the cluster. This is a real improvement, not a marginal one: it targets the one accuracy deficit the spec's own analysis flags as material, at the source, while the product's decomposability guarantee remains exactly intact.
