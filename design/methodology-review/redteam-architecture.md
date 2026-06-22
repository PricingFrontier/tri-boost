# Red-Team Critique — First-Principles Architecture

**Verdict in one line:** The boosting-family, fANOVA-purified, constant-cell architecture is the right backbone — but the spec hardcodes **two** separable design choices into one "depth-3 oblivious" primitive, and one of them (per-tree feature *resolution*, via the "3 splits = 8 leaves" cap) is leaving real accuracy on the table for **zero** gain on the decomposability or speed aims. That is the single highest-value change available. The oblivious (one-split-per-level) choice itself holds up. The biggest *missing* architecture is a cheap additive backbone, which the spec gestures at (`smoothing_rounds`) but never commits to.

---

## The core conflation the spec never separates

The spec treats "interaction order ≤ 3" (REQUIRED) and "tree = 3 splits / 8 leaves / oblivious" (DESIGN) as one atom. They are three independent decisions:

1. **≤3 *distinct raw features* per tree** — this, and only this, is what I1/I2 and the order-3 fANOVA termination require. (Spec §1.4, §2.5 invariant: the budget is on `distinct provenance[axis].raw`.)
2. **One shared split per level (oblivious)** — a speed/regularization choice.
3. **At most one split per feature ⇒ depth = #features ⇒ 8 leaves** — a *resolution* choice, and the one that costs accuracy.

I verified against §08 (line 27): the purification engine keys each raw tensor `T_raw[u]` on the **distinct-raw-feature support** `u_t` and expands it onto the per-axis **merged union grid** — it is already grid-agnostic and makes **no** assumption of 8 cells or one-cut-per-axis. The only places "8" is hardcoded are `ObliviousTree.leaves: [f32; 8]` and the branch-free inference lookup. **So decision 3 is genuinely free to change without touching decomposability.** The spec's own machinery proves it.

### Challenge 1 — "depth-3 = 3 splits" should change to "≤3 distinct features, repeated splits allowed" (oblivious decision-table form). **SHOULD CHANGE.**

A tree on the *same* ≤3 features that may split a feature **more than once** (e.g. age at two thresholds, or a 2-feature tree with one feature cut twice) is still a function of ≤3 raw features, so its fANOVA still terminates at order 3 — **exactly, losslessly, bit-for-bit** under the existing five gates. But it carries a **finer piecewise-constant grid per feature** (more than one breakpoint per axis per tree) and therefore strictly more expressive 2-D/3-D interaction cells. Concretely: the current primitive can only ever place a *single* breakpoint on each of its 3 axes per tree, so a 2-way interaction surface is approximated one-cut-at-a-time across many trees; an oblivious **decision table of depth 4–6 on ≤3 features** captures a multi-cell interaction surface in *one* tree.

- **Which aim:** PREDICTIVENESS. This directly attacks the spec's own named weakness ("depth-3 oblivious trees are deliberately weak per tree → need many trees," §1.2, research/01 §6). The closability analysis estimates order-3 recovers ⅓–⅔ of the EBM→GBDT gap; per-feature resolution is orthogonal to *order* and recovers a *different* slice — the within-support curvature that single-cut trees waste tree-count emulating. Rough magnitude: this is the difference between EBM's tiny stumps and a real shape function; expect it to matter most exactly where the spec is weakest (interaction-rich, smooth-within-cell targets), plausibly tens of Elo and a meaningful **tree-count reduction** (→ smaller tables, a double win the spec prizes elsewhere).
- **Decomposability:** **PRESERVED exactly.** Same proof, same gates, finer merged grid. §08 already handles arbitrary realized-border resolution.
- **Speed:** mild inference cost (leaf table grows from 8 to 2^d; still L1-resident at d≤6 — research/01 §2: depth-6 = 64 floats = 256 B). Training cost is the histogram build, which the spec already scopes as row-count-independent. Net: small, bounded.
- **Cost to implement:** generalize `leaves: [f32; 8]` → `SmallVec`/`[f32; 2^d]`, drop the "axis used by an earlier level ⇒ inadmissible" guard in §6.2 (replace with "distinct-raw-feature count ≤ max_order"), and let the merged grid carry >1 border per axis (already supported). The I1 invariant *check* needs no change — it already counts distinct raw features, not splits.

This is the one change I would fight for. The spec's depth-3-as-3-splits is a **self-imposed accuracy tax with no compensating benefit** to either fixed requirement.

### Challenge 2 — keep oblivious (one shared split per level). **HOLDS UP — steelmanned.**

Even granting deeper-on-≤3-features tables, should the splits be *oblivious* (shared per level) or *asymmetric* (best split per node, still on ≤3 features)? Asymmetric-on-≤3-features is **also** exactly order-3 decomposable (it's still a function of ≤3 coordinates), so this is a live design question, not foreclosed by I1/I2. But oblivious wins here on the merits:

- The evidence (research/01 §3, closability §b) is unambiguous that **obliviousness costs ≈0 at the ensemble level** — CatBoost ranks first untuned; the structural penalty is the order cap, not symmetry.
- Oblivious buys the FAST aim concretely: branch-free 2^d lookup, the subtraction trick, SIMD, and — critically — the **bit-reproducibility / quantized-histogram mechanism** that I2's reconstruction gate leans on.
- Asymmetric-on-≤3 would *not* meaningfully beat deeper-oblivious-on-≤3 (you've already recovered the resolution that asymmetry mainly buys), while forfeiting all the speed/reproducibility wins.

So: relax decision 3 (resolution), **keep** decision 2 (symmetry). The spec conflates them; only one should move.

---

## Steelmanning three genuinely different architectures

**A. Sparse exact GA3M — penalized functional-ANOVA with selected 2-/3-way terms (spline or piecewise-constant components).** This is the "build the decomposition *directly*, don't reverse-engineer it from trees" architecture (EBM/NODE-GAM extended to order 3; GAMI-Net family).
- *Predictiveness:* competitive at low order, and a *spline* basis gives smooth main effects that constant cells cannot. But (closability §c, research/06-2) the published order-≤2 GAM engines **trail** unconstrained GBDTs and crucially **trail on correlated features and need explicit interaction screening**; the spline variant (`changes_table_form`) abandons the constant-cell rating tables that are a hard requirement of aim 2's downstream reading.
- *Decomposability:* a *piecewise-constant* GA3M is exactly decomposable. A spline GA3M is decomposable but **not into the constant-cell tables the firewall demands** — it would flip to `Approximate`. So splines are out on the fixed requirement, not on accuracy.
- *Speed:* fine.
- **Verdict:** this is essentially *what tri-boost already is* once you relax decision 3 — a boosted, jointly-fit, purified GA3M with constant cells. The spec made the right call routing it through boosting (joint fit avoids the GAMI-Tree staged-mis-convergence under correlation, §07.1, research/06-2 §139) rather than EBM-style cyclic round-robin. **No change; the spec already dominates this design's exact-cell variant.** The only thing worth importing is its explicit *backbone* discipline (see Challenge 3).

**B. Interpretable NN (GAMI-Net / NODE-GA3M) distilled into exact tables.** Train a soft order-3 net, then read/distill exact tables.
- *Predictiveness:* NODE-GA3M reports real order-3 wins (Housing +21%). The soft model can find better split *placement* than greedy local search.
- *Decomposability:* the **soft model itself is not exactly decomposable** (soft routing = fractional cell membership, DenseNet stacking explodes order — research/06-2 §111,117). You'd have to *distill* it into the hard constant-cell model, at which point you're back to tri-boost as the student and the NN is just a teacher.
- **Verdict:** the spec **already captures the only transferable piece** correctly — as the *teacher* in CatBoost/black-box **distillation** (§09, design/02 §4) and as the **gradient triple-detector front-end** (§07.4). Adopting the NN as the *model* breaks the requirement; adopting it as a *signal source* is exactly what the spec does. **Holds up.** (Minor: the spec lists CatBoost as the default teacher; for closing the *order-≥4* part of the gap, a leaf-wise/deep teacher like LightGBM/a NN is the more informative soft target since CatBoost's own ≤3 projection is closer to the student. Worth A/B-ing the teacher class — cheap, exactness-neutral.)

**C. Hybrid: additive backbone + boosted ≤3-feature residual trees.** This is the strongest *missing* idea.
- *Predictiveness:* fit all 1-D main effects to near-convergence as a cheap additive model first (or interleave with `smoothing_rounds`), then boost ≤3-feature trees on the residual. Mains are where most tabular signal lives; getting them sharp and stable up front lets the scarce 3-feature trees spend their budget on genuine interactions, not on re-learning main curvature. research/06-2 §137 flags `smoothing_rounds` as portable and exactness-preserving.
- *Decomposability:* **PRESERVED** — a sum of 1-D functions plus ≤3-feature trees is still ≤3-feature; purification folds the backbone into the same `f0 + Σf_u` bank with zero special-casing.
- *Speed:* backbone is cheap (1-D histograms); likely *reduces* total trees.
- **Verdict:** **SHOULD ADD (as a schedule, low cost).** The spec has the pieces (joint boosting, `max_order=1` trees) but commits to *pure* joint boosting from f0 with no main-effect warm-start. A depth-1 warm-start phase is a near-free predictiveness + table-stability win and a regulator-friendly "additive core + audited interaction cells" story. This is a smaller, surer bet than Nesterov/fully-corrective refit (which the spec already defers, benchmark-gated, correctly).

---

## Bottom line

The architecture is **right in its bones** and should not be replaced: boosting-family (not a from-scratch GA3M, not a soft NN), oblivious symmetric trees, constant cells, purified fANOVA, the typed exactness firewall. Every genuinely different design I steelmanned either *is* tri-boost with a relaxed constraint (A), survives only as a *teacher/detector* the spec already uses (B), or is an additive schedule the spec should bolt on (C).

But the spec's headline primitive — **"depth-3 oblivious = 3 splits = 8 leaves"** — overconstrains. Two concrete changes, in priority order:

1. **Relax per-tree resolution: "≤3 distinct raw features, splits may repeat" (depth 4–6 oblivious decision tables on ≤3 features).** Strictly more expressive, **exactly** order-3 decomposable (the §08 engine already supports it), modest bounded cost. This is the highest-value architectural change in the spec and it is currently invisible because the spec never separates "order" from "resolution." **Change it.**
2. **Add an additive (depth-1) backbone / warm-start phase before joint interaction boosting.** Exactness-preserving, cheap, sharpens mains and frees the 3-feature budget. **Add it.**

Keep oblivious symmetry, keep constant cells, keep the firewall — those hold up under adversarial pressure. The order-3 *bias* ceiling on genuine ≥4-way data (Higgs/Year) is real and correctly disclosed; neither change above touches it, and nothing in a fixed-≤3-order product can.
