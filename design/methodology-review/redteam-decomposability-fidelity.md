# Red-Team Critique — Cluster: Decomposability Fidelity

**Scope:** whether the purification machinery produces tables that are *trustworthy as explanations* under correlated features — not merely bit-equal to the ensemble. The reconstruction guarantee (the tables sum back to `F`) is airtight and I do not challenge it. My target is the *attribution of mass across orders* — what each 1D/2D/3D table actually *says* — and whether the spec's identifiability choice, gate suite, and firewall make those statements reliable.

---

## 1. The headline defect: the spec claims product-`w` "dodges Hooker's extrapolation." It does not.

**The decision (§08.4, restated in §01).** Default `RefMeasure::ProductMarginals { laplace }`, justified as: "respects per-axis data density (dodges Hooker's extrapolation in empty regions)."

**Steelman first.** Choosing product-of-marginals is defensible *as a default* for three real reasons, and the spec gets them right: (a) it factorizes per axis, so purification is single-pass double-centering — fast and order-independent (a genuine FAST aim win); (b) it preserves `σ²(F)=Σσ²(f_u)`, so Sobol importances sum to 1 and drive the §07 heredity funnel coherently; (c) it makes equal-split SHAP *exact* rather than merely interventional. For a v1 milestone whose success criterion is "beat EBM on TabArena," product-`w` is the right *engineering* default and I would keep it as the default.

**The strike.** The *stated rationale* is wrong, and the error is load-bearing because it tells the reader the fidelity problem is solved when it is only relabelled. Per-axis marginals respect each axis's *marginal* density. But the 2D slice-mean that purification subtracts integrates a row of `T_{ij}` against the **product** weight `p(x_i)·p(x_j)` — which is strictly positive on `(x_i, x_j)` combinations that *never co-occur*. That is precisely Hooker's objection (the research doc states it verbatim at line 79: "integrating against a product/uniform measure evaluates `F` at feature combinations that never occur… and is unstable"). Product-`w` does not dodge extrapolation — **product-`w` *is* the extrapolating measure Hooker wrote his paper against.** The only measure that dodges it is `Joint`, which the spec demotes to a diagnostic.

Worse: **Laplace smoothing actively increases the extrapolation reliance.** `ŵ_lap ∝ ŵ_unif + λ·ŵ_emp` injects uniform mass into empty cells *on purpose*, to keep weights strictly positive so zero-mean and convergence stay well-conditioned. That is a numerics fix that *deepens* the fidelity problem: it places non-trivial centering weight exactly on the never-observed combinations. The spec presents Laplace as a fidelity feature ("respects density… stays positive"); it is a fidelity *cost* paid for a numerics benefit.

**Verdict: rationale should change; default may stay.** The default is fine; the *claim* is false and must be corrected, because it currently licenses shipping product-`w` tables to a regulator with the words "we avoid evaluating effects where data doesn't exist" — which is the opposite of true. Under correlated features (the realistic and the *stated insurance* regime), a chunk of the variance the product-`w` main-effect table claims to "own" is mass purified down from interaction cells that are weighted by phantom probability. Two correlated rating factors (vehicle value × vehicle age, sum-insured × postcode) will have their relativities partly determined by combinations no policy occupies.

---

## 2. The gates certify *consistency*, not *faithfulness* — and the spec conflates the two.

**The decision (§3, §08.6).** Five build-blocking checks (Reconstruction, MassConservation, Purity, VarianceSum, ThreeWayEqual). Tagline: "If these ever disagree there is no product."

**The strike.** Every one of these five gates passes for *any* valid `w` — including a `w` that produces wildly misleading tables. Reconstruction and MassConservation hold because mass-moving conserves the sum *by construction*, independent of `w`. Purity holds by definition of what purify *did*. VarianceSum holds for any axis-constant `w`. ThreeWayEqual is an identity of the purified bank. **None of the five measures whether the table values are a trustworthy explanation** — they measure that the bank is *internally consistent and reconstructs the model*. A perfectly consistent, perfectly reconstructing, perfectly pure table set can still attribute a correlated interaction's mass to the wrong main effect. The spec's "if these disagree there is no product" is true but is being read as "if these *agree* there *is* a product," which is the converse error. The gates are necessary, not sufficient, for the *explanation* aim — yet §08 presents passing them as the whole guarantee.

**Verdict: holds as an exactness contract; oversold as a fidelity contract.** No change to the gates; a change to what the spec claims they prove. The missing gate is a *stability/faithfulness* diagnostic (below).

---

## 3. The one genuinely better methodology: ship the between-`w` drift metric as a *gate-adjacent fidelity score*, not an "internal diagnostic."

The spec already builds the joint-`w` path "early… to benchmark relativity drift between measures" and calls that drift "the credibility metric under regulator questioning" — then explicitly files it as "an internal diagnostic, not shipped tooling" (§08.4, brainstorm §5). **This is the single best fidelity instrument in the design, and the spec deliberately blunts it.**

**The better methodology (concrete):** compute every export under **both** product-`w` and joint-`w`, and attach to each table a **drift score** `D_u = ‖f_u^prod − f_u^joint‖_w / σ(F)`. This is the operational signal that a table's mass attribution is an artifact of extrapolation: a main/interaction effect that is *stable* across the two measures is real; one that *moves* is being shuffled by phantom-cell weighting. This directly serves **decomposability-as-faithfulness** — the actual product requirement is tables that are trustworthy *as explanations*, and `D_u` is the only number in the design that measures that.

- **Aim impact:** large win on aim 2 (faithfulness), zero cost to aim 1 (accuracy — purely post-hoc), small cost to aim 3 (one extra purify under joint-`w`, which is iterative but runs once post-fit on tiny tensors).
- **Decomposability-safety:** fully preserving. Recomputing under a different `w` is already declared exactness-preserving (§3); this just *retains both results* instead of discarding one.
- **Cost:** low. The joint-`w` purify is `O(#cells · log(1/ε))`, embarrassingly parallel, already specified. The only new work is the diff and surfacing it.

The spec's reason for keeping it internal — "we don't ship benchmarking tooling" — is a category error. A per-table self-consistency annotation is not model-comparison benchmarking; it is part of the explanation, exactly like the SE-band the spec *does* ship (§08.7). **Recommendation: promote `D_u` to a first-class, exported, per-table annotation alongside the SE-band.** If a regulator asks "are these relativities real or an artifact of how your features correlate?", `D_u` *is* the answer, and the spec currently throws it away.

---

## 4. Thin/empty merged cells: the tables are confidently wrong where data is absent, and nothing flags it.

**The decision (§03, §08).** Merged grid = union of *realized* split borders. Laplace keeps empty cells from breaking math.

**The strike.** A 3D triple table on the union grid has `|Ω_i|·|Ω_j|·|Ω_k|` cells; under correlation, a large fraction are near-empty. The model *does* assign each a value (the tree leaf that covers it), and purification *does* center against Laplace-inflated weight there — so the cell shows a crisp relativity backed by ~0 policies. The credibility floors in §07 (`min_data_in_leaf`, `min_sum_hessian_in_leaf`) prevent a *split* from firing on thin support, which helps — but they bind at *train* time on the 8-cell oblivious leaf, **not** on the *merged-grid* cells the exported table actually displays, which are finer (the union across trees). So a displayed table cell can be far thinner than any single leaf that contributed to it. The SE-band (§08.7) only exists for bagged models and is "display-only." **There is no per-cell support count on the exported table.** An actuary reading "postcode-G × young-driver × high-value → 1.42×" cannot see that it rests on 11 policies.

**Verdict: should change (cheap).** Carry a **per-cell `w`-mass / effective-count** alongside `EffectTable.values` (the WeightCache already computes per-axis mass; the joint count is one pass over the binned data). Surface it the way the SE-band is surfaced. This is the empty-cell honesty the firewall philosophy demands and currently omits. Cost: trivial; decomposability-safe (it is metadata, never summed into score).

---

## 5. Smaller calls, ruled quickly.

- **Single shared `w` across ensemble members (§09, design/02):** *Holds — steelman.* The variance-sum identity branches on `w`, so mixing measures across bagged members is genuinely incoherent. Forcing one `w` is correct. No change.
- **Joint-`w` skips VarianceSum rather than failing (§08.6):** *Holds.* Under hierarchical orthogonality the identity legitimately doesn't apply; skipping (not failing) is the right call, and Sobol→Shapley-effects (still summing to 1) is the correct substitute. Good design.
- **"interventional" labelling of SHAP under product-`w` (§08.5):** *Holds, and is more honest than most libraries.* Correctly refuses to call product-`w` attributions "observational SHAP." Keep.
- **Complete-vs-pruned view (§08.7):** *Holds.* Decoupling lossless inference support from top-k display is exactly right and does not hide fidelity problems — *provided* pruning is by Sobol under the *same* `w` that exported the tables (it is). One caveat: pruning by product-`w` Sobol will *hide* precisely the correlated interactions whose mass got purified into mains — so the drift metric of §3 must gate pruning too (don't hide a high-`D_u` table just because its product-`w` variance is small).

---

## Bottom line

The decomposition machinery is **mathematically sound and the reconstruction guarantee is real** — the tables *are* the model, bit-for-bit, and that is not in doubt. The defect is that the spec **conflates internal consistency with explanation-faithfulness** and then markets product-`w` as solving a correlation problem it does not solve. Three changes, all exactness-preserving, all low-cost, turn "decomposable" into "decomposable *and* faithful":

1. **Correct the rationale** at §08.4/§01: product-`w` does *not* dodge Hooker extrapolation — it is the extrapolating measure; Laplace deepens the reliance. Keep it as the *default* (it earns its place on speed, Sobol-sum, and exact-SHAP), but stop claiming it as a fidelity feature.
2. **Promote the between-`w` drift score `D_u` to a shipped, per-table annotation** (it is already computed and then thrown away). It is the only number in the design that measures whether a relativity is real or a correlation artifact — and it gates display pruning so faithful-but-low-variance tables aren't hidden.
3. **Carry per-cell effective support** on exported tables (the WeightCache already has the mass). Empty/thin merged cells currently show confident relativities backed by no data, and nothing flags it.

The chosen identifiability route (purification under a fixed product `w`) is the right *spine*; it should not be the *only* lens shipped. Faithfulness under correlation is cheap to add and is the difference between "the tables reconstruct the model" (proven) and "the tables explain the model" (currently assumed, not delivered).
