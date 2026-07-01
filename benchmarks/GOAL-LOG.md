# Goal-optimization loop — experiment log

Approach (per direction 2026-06-25): **NOT per-dataset hyperparameter tuning** — that overfits the
benchmark and doesn't generalize. Instead, **diagnose why a rival wins, identify the algorithmic
technique behind it, and implement that technique as a genuine library feature** that helps broadly.
Every change keeps G0 (exact ≤3rd-order purified decomposition, all 5 I2 gates, `mode=Exact`) — so
leaf-wise/asymmetric growth and order>3 are off the table; value-level leaf methods, histogram/engine
engineering, sampling, bagging, categorical encoding, split-finding are fair game.

Metrics measured vs the **frozen rival baseline** (`.fair_cache.json`), across the whole suite — never
tuned to a single dataset. RMSE-log↓ (regression), ROC-AUC↑ (classification).

## Rival wins → technique gaps (the map the loop works through)
- **G1 (EBM)** beats tri at order-1/2 (main-effect shape quality). EBM technique: cyclic/round-robin
  per-feature boosting + heavy outer bagging + careful binning.
- **G2 (XGBoost/LightGBM @ d3)** beat tri on diamonds (fine-continuous; partly structural — leaf-wise is
  more expressive, OFF-TABLE) and amazon (all-categorical). Technique: native optimal categorical split
  (sort categories by gradient, split into 2 groups), quantile split candidates.
- **G3 (CatBoost d3-ctr1)** beats tri only on amazon. Technique: ordered target statistics / ordered boosting.
- **G5 (speed)** LightGBM 7–27× faster training. Technique: histogram subtraction, quantized hist, GOSS/EFB.

Note: **amazon (all-categorical) loses BOTH G2 and G3** → better categorical handling is the highest-leverage
single technique (closes two goals at once). A `rival-technique-roadmap` workflow is producing the prioritized,
G0-verified plan; entries below are filled as each technique is implemented + measured + committed.

## Aborted approach (recorded so it isn't retried)
- Per-dataset order-2 knob sweeps (lr/leaf_refine/l1/path_smooth/n_bags): confirmed single knobs *narrow*
  but don't *flip* diamonds/kick G1@2, and any "win" would be dataset-specific overfitting. Abandoned in
  favor of genuine technique adoption. (One real fact surfaced: miami G1@2 is a WIN under the committed
  gentle early stopping — a measurement correction, not a tuning gain.)

## Roadmap — rival techniques to adopt (from the rival-technique-roadmap workflow, all G0-verified)
ROI-ordered work queue; each is a genuine generalizing technique a rival uses that tri lacks.

| # | technique | source | goals | effort | G0 | status |
|---|---|---|---|---|---|---|
| 1 | Per-split categorical gradient re-sort (Fisher optimal split) | LightGBM | G2+G3(+G1) | L | conditional | ❌ REJECTED by design (inert-or-invasive) |
| 2 | Histogram subtraction (parent−child), QuantizedI32-only | LightGBM | G5 | M | safe | queued |
| 3 | Integer-space quantized hist scan (unlocks #2) + QHIST default | LightGBM | G5 | S | safe | queued |
| 4 | Cyclic/round-robin per-feature boosting | EBM | G1 | L | safe | ❌ REJECTED (measured worse) |
| 5 | Automatic categorical CTR combination axes | CatBoost | G3+G2 | L | conditional | ❌ REJECTED (breaks G0 + doesn't generalize) |
| 6 | FAST pairwise interaction detection (populate InteractionPolicy.groups) | EBM | G1 | L | safe | queued |
| 7 | Hessian-weighted quantile bin borders | XGBoost | G2 | M | safe | queued |

Rejected (already present / G0-incompatible): EFB (g0=no, low impact); missing default direction (tri HAS it);
ordered TS (tri HAS it, KFold OOF beats Ordered{1}); ordered boosting (weak, leakage already closed); GOSS
(subsumed by MVS + #2); colsample_bylevel, heavy-bagging-default, mains-first, low-card one-hot (composable add-ons).

Sequencing: #1 first (multi-goal: amazon on G2+G3 at once). Then #3+#2 together (G5, cheap, mutually dependent).
Then #4 (suite-wide G1 order-1). #5/#6 compose with #4 (EBM's mains-first recipe falls out). #7 last. Re-measure
G5 after #2/#3 before bothering with GOSS. Every step: live G0 `tables()` check (FAIR_G0=o3) green before commit.

## Attempted techniques (with measured deltas vs frozen rivals)

### #4 Cyclic/round-robin boosting (EBM) — ❌ REJECTED, reverted (2026-06-25)
Built end-to-end (`Schedule::{Greedy,Cyclic}` core + FFI + sklearn + `.pyi` + round-robin-stump test,
all gates green, stays exactly decomposable order-1). Measured tri order-1 **cyclic vs greedy vs EBM mains**
across the 4 EBM datasets — **cyclic lost to greedy on ALL of them**:

| dataset | tri o1 greedy | tri o1 cyclic | cyclic vs greedy | greedy vs EBM |
|---|---|---|---|---|
| diamonds | 0.11214 | 0.11663 | −4.0% | −4.4% |
| miami | 0.17139 | 0.17592 | −2.6% | −2.9% |
| kick | 0.76469 | 0.76424 | −0.06% | +0.2% (greedy beats EBM) |
| allstate | 0.56287 | 0.56421 | −0.24% | −0.2% |

**Why it failed**: forcing round-robin wastes rounds uniformly refining low-signal features, while tri's
greedy adaptively concentrates on high-gain ones — greedy is already a better mains learner. EBM's edge is
its bagging + tiny-lr shape smoothing, NOT the cyclic schedule. **Reverted** (no strictly-worse knob ships).
Corollary: the "compose cyclic with bagging/interactions = EBM recipe" plan (#5/#6 dependency on #4) is
weakened — if pursuing G1 mains later, test BAGGING on greedy mains, not cyclic.

### #5 Categorical CTR combinations (CatBoost) — ❌ REJECTED (2026-06-25, prototype only, no code shipped)
Python prototype (pairwise-concatenated tuple columns as ordinary TS axes). TWO disqualifiers:
1. **Breaks the decomposition rule.** A combo `a__b` is a 2-way original interaction smuggled in as a
   1-way axis, so the model is only ≤3rd-order in COMBO space — in ORIGINAL features, a combo inside a
   2-/3-way table is an order-4–6 effect presented as ≤3-way. Mechanically the I2 gates pass (combos are
   ordinary axes), but the ≤3rd-order-in-your-features guarantee (the product) is violated.
2. **Doesn't even generalize.** All-36-pairs amazon +1.63%; top-8-cats (28 combos): amazon +0.63%,
   **kick −0.74%**, allstate +0.08%. Helps one dataset, hurts another — dataset-dependent, not a technique.
Decision: do NOT ship. Pursue the G0-CLEAN categorical technique instead → #1 (per-split gradient re-sort:
sharpens single-categorical splits, no new features, no order inflation).

### #1 Per-split categorical gradient re-sort (LightGBM) — ❌ REJECTED by design analysis (2026-06-25, no code)
Architect design pass (read the full split/low_bit/explain architecture). Verdict: not worth building.
- **The cheap version is INERT on this suite.** Re-ranking categorical bins by ROUND-0 gradient ratio
  g/(h+λ) — the only variant that keeps the contiguous-split machinery (and all of explain.rs) untouched —
  equals tri's existing target-mean Fisher order for squared-error (gradient order == −target-mean order)
  and is near-identical for logistic. It changes nothing on diamonds/miami/particulate/allstate/kick/amazon.
- **The version that DIFFERS (per-level re-rank with current residuals) is disproportionately invasive.**
  It makes categorical splits NON-CONTIGUOUS sets in bin order, which breaks the shared `low_bit` primitive
  AND the merged-grid abstraction in explain.rs (`rep_model_bin`/`model_bin_to_cell`/`build_cell_maps` are
  pure contiguous-border arithmetic) — requiring a refined per-bin→cell merged grid, a SECOND serialized
  wire format for `TableBank.merged_grids`, a `Split` change, and exhaustive re-proof of all 5 I2 gates.
  ~2–3 weeks with high silent-G0-breakage risk.
- **Wrong target anyway.** Single-axis re-sort cannot capture amazon's TUPLE-interaction signal (the actual
  G2/G3 loss). Per the architect: neither variant addresses it.
Decision: skip. Pivot to the safe, biggest-gap, genuine technique → #2/#3 (LightGBM histogram subtraction, G5).

### #2/#3 G5 histogram subtraction (LightGBM) — ❌ REFUTED by profiling (2026-06-25, no code)
Before building it, profiled where diamonds fit-time (4000 trees) actually goes:
| config | time | acc |
|---|---|---|
| refine=4 + earlystop | 29.9s | 0.08896 |
| refine=0 + earlystop | 11.1s | 0.09070 |
| refine=4, no es | 29.1s | 0.08854 |
| refine=0, no es | 10.4s | 0.09047 |
| refine=0, n_trees=1000 | 2.5s | 0.09580 |
**leaf_refine_steps=4 is ~2/3 of fit time** (10.4→29.1s); early-stop eval is ~free (+0.7s); histograms are
the SMALLER ~10s base. So histogram subtraction (the roadmap's G5 technique, which assumed histograms
dominate) would yield ~7% overall — not worth 2-3 days. Also: QuantizedI32 is currently SLOWER than
FullF64 (diamonds 40s vs 34s) with identical accuracy (it dequantizes per-cell before the scan), so even
the prerequisite needs an integer-scan rewrite first. REFUTED. The real G5 cost is leaf_refine's repeated
full-row passes (aggregation + backtracking deviance); parallelizing them is blocked by the byte-determinism
invariant (needs fixed-order folds) and grad_hess is single-threaded (loss.rs) but trivial for squared-error.

## FRONTIER ASSESSMENT (2026-06-25)
After a rival-technique research workflow + rigorous attempts, tri-boost is at its **G0-constrained frontier**:
- **G1 (EBM)**: won @order-3 (3/4); @order-1/2 behind — EBM is a mains SPECIALIST (cyclic boosting tried →
  worse; bagging dataset-dependent). Structural.
- **G2 (xgb/lgbm d3)**: 4/6. Losses = diamonds (leaf-wise depth-3 strictly more expressive than oblivious —
  G0-forbidden to match) + amazon (tuple signal needs order>3 — G0-forbidden; combos break the rule + don't generalize).
- **G3 (cat d3-ctr1)**: 5/6. Loss = amazon (same tuple issue).
- **G5 (speed)**: coarse config-profiling shows leaf_refine ~2/3 of fit time (hist-subtraction refuted). NEXT:
  add GRANULAR per-phase timers inside the Rust fit (hist build / split-find / leaf_refine grad_hess /
  aggregation / backtracking deviance / update / early-stop) to pinpoint the EXACT bottleneck before optimizing.
Every clean rival technique either (a) reduces to what tri already does (inert), (b) requires breaking G0
(order>3 / leaf-wise / non-contiguous splits), or (c) is mature-implementation overhead. The real wins tri
HOLDS (G1@3, G2 4/6, G3 5/6, exact decomposition throughout) are already banked.

### Granular fit profiler (TRIBOOST_PROFILE env, zero-cost when off) — committed dev infra
Instrumented the Rust fit loop with per-phase wall-timers (boost.rs `prof` module). Diamonds o3
(refine=4, 4000 trees, wall 29.6s) EXACT breakdown — top-level phases sum to wall; nested `.` are subsets:
| phase | s | %wall |
|---|---|---|
| **leaf_refine** | 17.1 | 58% |
| ↳ refine.backtrack_eval | 8.9 | (30% of wall) |
| ↳ refine.grad_hess | 3.4 | |
| ↳ refine.aggregate | 0.9 | |
| **grow_tree** | 9.4 | 32% |
| ↳ grow.hist_build | 7.9 | (27% of wall) |
| ↳ grow.split_find | 0.4 | |
| update_raw | 1.1 | 4% |
| grad_hess | 0.8 | 3% |
| earlystop_eval | 0.1 | 0.2% |
**EXACT bottlenecks** (not the histograms the roadmap assumed): #1 `refine.backtrack_eval` (8.9s) — the
line-search re-walks the tree + does a separate deviance pass every trial, but memberships are FIXED and
only 8 leaf values change → fusable to one membership pass, O(8) exact for squared-error. #2 `grow.hist_build`
(7.9s) — where subtraction would help (~3s). Both exactness-preserving. backtrack_eval is the bigger, safer first win.

## Implemented techniques (committed wins)

### ✅ WIN #1 — Fuse leaf-refine backtrack eval (membership-based, no tree re-walk) [G5]
The profiled #1 bottleneck. The leaf-refinement line search re-scored the whole tree every trial via
`raw_with_tree_leaves` (route each row through the splits) + a separate deviance pass. But the leaf
MEMBERSHIPS are fixed and only the 8 leaf VALUES change per trial — so `raw[rows] = base_raw + leaves[membership]`
is computable with no tree walk, reusing one buffer. New `apply_membership_leaves` + a reused `trial_raw`
buffer (swap on accept). EXACTNESS-PRESERVING (byte-identical: a tree's contribution to raw IS its leaf
value; locked by test `membership_leaf_fill_matches_tree_walk_bit_for_bit`).
- **Measured (diamonds o3, profiler):** `refine.backtrack_eval` 8.93s → **4.14s (−54%)**; wall 29.6s → **24.7s (−17%)**.
- **Accuracy byte-identical** (diamonds 0.08896, allstate 0.54009 — exact). allstate wall neutral (histogram-
  dominated there, so backtrack is a smaller fraction). 221 core + 20 py + stubtest green; profiler confirms
  the saving internally (not wall noise). Generalizes: helps any leaf_refine>0 fit, hurts none.
NEXT G5 target (now #1 by profile): `grow.hist_build` (7.6s) — histogram subtraction on the quantized path.

### ❌ grad_hess row-parallelization — REVERTED (measured regression on SE)
Added a shared row-parallel `fill_grad_hess` (rayon `try_for_each`, threshold 8192) across all 5 losses.
Byte-identical across thread counts (✓ determinism), accuracy unchanged (✓), but **SLOWER**: diamonds
`refine.grad_hess` 3.36s → 4.88s, wall 24.7 → 26.4s. Squared-error grad_hess (g=raw−y) is MEMORY-BANDWIDTH
bound, not compute-bound — 4 threads can't beat one memory bus, and rayon coordination + closure-call
indirection add net overhead. Would help compute-bound losses (logistic/poisson `exp`/`sigmoid`) on huge
data, but regresses the common SE case and the benchmark can't validate the log-link gain. Reverted.
Lesson: only parallelize COMPUTE-bound per-row work, not memory-bound.

### ✅ WIN #2 — Eliminate leaf-refine's duplicate tree-walk [G5]
Leaf-refine walked the tree TWICE per tree: once for `tree_memberships_for_rows`, again for the initial raw
(`raw_with_tree_leaves`). The second is derivable from the first — the initial raw is `base + leaf[membership]`
(reuse `apply_membership_leaves`). Removed the second walk; `raw_with_tree_leaves` is now `#[cfg(test)]` (the
equality test's reference). BYTE-IDENTICAL (diamonds 0.08896). Diamonds wall 24.7s → **23.9s**. Generalizes
to every leaf_refine>0 fit. Cumulative with WIN #1: **29.6s → 23.9s (~19%)** on diamonds.

### ❌ `deviance_at_rows` direct-index fold — REFUTED, not committed (2026-06-25)
Hypothesis: `deviance_for_rows` (the leaf-refine backtrack + early-stop deviance) wastes 3 allocations + 3
gather-copies per call; fold deviance DIRECTLY over `y[rows[i]]`/`raw[rows[i]]`/`weight[rows[i]]` (new
`Loss::deviance_at_rows`, monomorphic per loss) to skip them. Built it across all 5 losses + a bit-identity
proptest (✓ byte-identical). **But measured SLOWER**: diamonds `refine.backtrack_eval` 2.585s → **4.560s
(+76%)**, kick 13.485s → **22.118s (+64%)** (fixed config n=2000, refine=4, no-es; scores byte-identical
0.09022 / 0.76975). Cause: the old gather-then-`deviance` folds over CONTIGUOUS slices → autovectorized
(SIMD); the direct-index fold reads scattered indices with per-element bounds checks → scalar. The removed
allocations were cheap (allocator reuses the same freed blocks); the vectorization I broke was not. Lesson:
**don't trade a contiguous SIMD fold for a scattered scalar one to save a cheap allocation.** Reverted whole.

### ✅ WIN #3 — Hoist trial-invariant gathers out of the leaf-refine line search [G5]
Salvaged the right win from the refuted attempt. The backtrack re-gathered `y`/`weight`/`raw` at `rows`
EVERY trial (scatter + alloc), then folded `deviance` over the contiguous result. But `y[rows]`,
`weight[rows]` and `base_raw[rows]` are CONSTANT across all steps + backtracks of a tree — only the 8 leaf
VALUES change. Gather those three into dense per-tree buffers ONCE (`gather_rows`); per trial just fill the
dense subset-raw from `base_sub + leaves[membership]` (`fill_leaf_raw_contiguous`) and run the SAME
vectorized `deviance` over contiguous `(y_sub, raw_sub, w_sub)`. Per-trial cost: one contiguous fill + the
SIMD fold — no scatter-gather, no allocation; the full raw is reconstructed (for the next grad_hess) only on
ACCEPT. Keeps the contiguous fold the refuted attempt lost. BYTE-IDENTICAL (`fill_leaf_raw_contiguous` ==
`apply_membership_leaves` gathered over `rows`, locked bit-for-bit in
`membership_leaf_fill_matches_tree_walk_bit_for_bit`; end-to-end scores unchanged 0.09022 / 0.76975).
- **Measured (fixed config n=2000, refine=4, no-es, 4 threads):** diamonds `refine.backtrack_eval` 2.585s →
  **1.530s (−41%)**, wall 12.5s → 11.9s; kick 13.485s → **12.107s (−10%)**, wall 43.2s → 41.9s. (Diamonds
  wins bigger: SE deviance is cheap so gather/alloc was a larger share; kick's logistic deviance is
  compute-bound, so the kept SIMD fold dominates.) Generalizes to every leaf_refine>0 fit.

### ⚪ Subset-only refine refactor (drop full-raw buffer) — NEUTRAL, not committed (2026-06-25)
Refactored the whole leaf-refine pass onto the dense subset buffers (grad_hess over `*_sub`, contiguous
aggregate, no full-length `raw` materialization, no `base_raw.to_vec()` per tree). BYTE-IDENTICAL (scores
unchanged). But measured NEUTRAL (within run noise): the o3 config has no row subsampling, so `rows == n`
and the subset grad_hess has the same row count, while the "scattered" aggregate over `gh[rows[i]]` was
already sequential (rows are sorted). It also narrowed grad_hess's finite-checks to in-sample rows (an
error-path change) for no speed payoff. Reverted — a cleaner shape with no measured benefit isn't worth the
semantic change. (Would help under bagging/subsample<1, which the benchmark doesn't use.)

### ✅ WIN #4 — Unit-weight histogram fast path (skip per-row Σw) [G5]
`grow.hist_build` is the largest phase outside leaf_refine (33-38% of wall, on every dataset). The hot
accumulation loop folds 4 arrays per row — g, h, **wsum**, count. But when the caller supplies NO sample
weights (the common case + the entire benchmark), the weight vector is the engine's materialized all-ones,
so `wsum[idx] == count[idx]` EXACTLY in f64 (Σ 1.0 over a bin = its integer count, exact for count<2^53).
A new `GrowConfig.unit_weight` flag (set iff `spec.weight.is_none()`) lets the histogram SKIP the per-row
weight read + Σw add (LLVM unswitches the loop-invariant branch) and set `wsum = count` afterwards. The
flag is conservative — `false` whenever weights were provided, even if all 1.0 — so it never risks a wrong
Σw. Subtraction/quantized paths untouched. BYTE-IDENTICAL: pinned bit-for-bit (g/h/wsum/count) for both the
sequential and row-chunk-parallel branches by `unit_weight_fast_path_is_bit_identical_to_full_sigma_w`;
end-to-end scores unchanged (diamonds 0.09022, kick 0.76975) across 3 reps each.
- **Measured (fixed config n=2000, refine=4, no-es, 4 threads; means of 3 reps):** diamonds
  `grow.hist_build` ~4.20s → **~3.90s (−7%)**; kick ~16.0s → **~14.4s (−10%)**. Wall diamonds ~12.5→~11.8s,
  kick ~41.9→~40.0s. Generalizes to every unweighted fit (the default), all objectives.

### ❌ Log-link grad_hess row-parallelization (retry, log-link only) — NET NEUTRAL, not committed (2026-06-25)
The prior `grad_hess` parallel revert (ea08b04) only tested squared-error (memory-bound). Hypothesis: the
LOG-LINK losses (Logistic/Poisson/Gamma/Tweedie) are compute-bound (exp/sigmoid per row, ~60-80 cycles), so a
row-parallel MAP (independent writes ⇒ bit-identical to sequential, no fold; SE left sequential) should help
kick/amazon. Built it (shared `fill_grad_hess` helper, threshold 8192, all 4 log-link losses) + a
1/2/8-thread bit-identity gate (✓ byte-identical). But measured NET NEUTRAL on kick: `refine.grad_hess`
3.73→3.13s LOOKED like a win, but the main `grad_hess` phase rose 0.22→0.79s (rayon pool warmup attribution)
— TOTAL grad_hess 3.95→3.92s unchanged; wall ~40.3→~39.0s (within noise). Cause: even with the sigmoid
compute, each call moves ~935KB (g/h write-out + y/raw/weight read) ⇒ memory-bandwidth bound on the write,
same as SE — the compute isn't heavy enough to overcome it. The prior SE lesson GENERALIZES to log-link.
Reverted. (Would only pay off if grad_hess were fused with more per-row compute, or on far wider data.)

### FRONTIER ASSESSMENT — byte-identical speed floor (2026-06-25, post WIN #3/#4)
After WIN #3/#4 and the refuted attempts above, the byte-identical + G0 speed frontier is reached for the
major phases. `grow.hist_build` (33-38% of wall, the largest shared phase) is **byte-locked**: its f64 fold
order (sequential-within-chunk + chunk-order reduction at the 32768-row threshold), f64 precision, and the
absence of subtraction are all baked into the committed bit-pattern — changing any of them changes outputs.
The leaf-refine line search (≈50% of o3 wall) is the accuracy lever LightGBM has no equivalent of; its memory/
alloc overhead is removed (WIN #1/#2/#3) and its compute (deviance fold, serial f64) is byte-locked. grad_hess
parallel is net-neutral (memory-bound) for ALL objectives. **Every remaining LightGBM speed technique violates
a hard constraint**: histogram subtraction (f64 drift ⇒ not byte-identical), quantized int histograms (changes
outputs), leaf-wise growth (needs fewer trees — G0 requires oblivious), no leaf-refine (drops the accuracy
lever). So tri stays ~1.9-3.5× slower than LGBM on the suite config (refine=0, hist-bound) and ~13× on the o3
accuracy config — a STRUCTURAL gap under strict byte-identity, not an implementation one. Closing it further
requires relaxing byte-identity (adopt subtraction/QHIST, accepting ~rounding-level output shifts) or G0.

### ✅ WIN #5 — Level-2 histogram subtraction (FullF64) [G5] — accuracy-neutral (byte-identity relaxed)
User authorized relaxing strict byte-identity for accuracy-NEUTRAL speedups. Wired the histogram-subtraction
trick into the oblivious grower: at level 2 (FullF64), build only the SMALLER of each parent leaf's two
children by accumulation (~half the rows) and derive the LARGER by subtracting from the retained level-1
parent (`subtract_sibling_into` + `build_subtracted_level`, gated by `GrowConfig.hist_subtraction`, default on,
kill-switch + A/B reference). Building the smaller and subtracting to get the larger remainder avoids
catastrophic cancellation, so g/h drift stays ~1e-11; `count` is integer-exact and, under unit weights (the
default + whole benchmark), `wsum == count` stays EXACT. Scoped to level 2 only (single drift generation) per
a design-critique workflow (3 expert critiques → synthesis); leaf values are recomputed from gh directly so
they are unaffected — drift only perturbs split SELECTION at exact near-ties.
- **Design + verification via Workflow** (ultracode): a design-critique workflow caught the axis-position
  remap (A_2 ⊊ A_1 positions shift), the `subtract()` shape-mismatch (needs a custom sibling-subtract), and
  the build-smaller-derive-larger cancellation-avoidance; an adversarial-verification workflow (3 attackers +
  triage) returned **ship, zero confirmed bugs** — the only flagged items were the accepted near-tie
  flip and a non-unit-weight credibility-boundary flip (absent under unit weights).
- **Byte-identity within tolerance:** equivalence test `level2_subtraction_reproduces_full_build_tree`
  (subtracted tree == full-build tree, well-separated fixture); determinism test (1/2/8 threads identical);
  primitive tests for `subtract_sibling_into` (hand-computed, underflow, shape); quantized-inert test.
  End-to-end real-data scores match the prior baseline **EXACTLY** (no near-tie flips occurred): diamonds
  0.11376 / 0.09022, kick 0.77228 / 0.76975. 229 core + 20 py tests green; clippy + fmt clean.
- **Measured (4 threads):** suite config (n=400, refine=0): diamonds `hist_build` 0.877s → **0.696s (−21%)**,
  wall 1.42s → **1.13s (−20%)**; kick `hist_build` 2.72s → **2.24s (−18%)**, wall 4.26s → **3.62s (−15%)**.
  o3 config (n=2000, refine=4): diamonds `hist_build` ~3.90s → 3.55s (−9%); kick ~14.4s → **11.6s (−19%)**,
  wall ~40s → 36.8s. Generalizes to every FullF64 depth-≥2 fit.

### ✅ WIN #5b — Extend subtraction to LEVEL 1 (parent = level-0 root) [G5] — accuracy-neutral
After the level-2 path was validated (equivalence + adversarial-verification workflows + exact real-data
scores), extended the SAME generic `build_subtracted_level` to level 1 (gate `level >= 1`; retain each
FullF64 level's hist as the next level's parent). Level 1 is the BIGGER win — it has the most admissible axes
(|A_1| = |A_0|−1, vs the shrunk |A_2|) over the full n rows, so subtracting it saves more row-visits than
level 2. Level 2's parent is now itself a subtracted hist ⇒ g/h drift compounds to ~2e-11, still
accuracy-neutral (the equivalence test grows the SAME tree as the full build for both levels; determinism
test green; real-data scores unchanged: diamonds 0.11376/0.09022, kick 0.77228/0.76975 EXACT). 230 core +
20 py green.
- **Measured (4 threads, cumulative subtraction total vs no-subtraction baseline):** suite (n=400, refine=0)
  kick `hist_build` 2.72s → **1.68s (−38%)**, wall 4.26s → **3.16s (−26%)** (vs LGBM 1.19s: 3.5× → 2.7×);
  diamonds `hist_build` 0.877 → **0.682s (−22%)**, wall 1.42 → 1.14s. o3 (n=2000, refine=4): kick `hist_build`
  ~16s → **8.66s (−46%)**, wall → 28.3s; diamonds ~4.2 → 3.26s. Generalizes to every FullF64 depth-≥2 fit.

### ✅ WIN #7 — Row-parallel log-link grad_hess [G5] — BYTE-IDENTICAL (corrects the earlier revert)
The earlier grad_hess row-parallelization was reverted as "net-neutral", but that verdict was a measurement
artifact: with a COLD rayon pool the first parallel call (the main `grad_hess`) absorbed the one-time pool
spin-up, which the profiler attributed to that phase and masked the refine-phase win. Now that WIN #6's
deviance work warms the pool, a clean re-test shows the real picture. grad_hess is a row-independent MAP, so
parallelizing it is **bit-identical** to the sequential loop regardless of thread count (no fold, no drift —
unlike the deviance) — pinned by `log_link_grad_hess_parallel_path_is_bit_identical_across_thread_counts`.
New `fill_grad_hess_parallel` applied to Logistic/Poisson/Gamma/Tweedie; **SquaredError stays sequential**
(g=w(F−y), h=w — a trivial per-row term, memory-bandwidth bound, where parallelism does not pay).
- **Byte-identical:** real-data scores unchanged (kick 0.76975; diamonds 0.09022, SE unaffected). 231 core +
  20 py green; clippy + fmt clean.
- **Measured (o3, n=2000, refine=4, 4 threads, warm pool):** kick `refine.grad_hess` 4.13s → **2.97s (−28%)**
  (no main-grad_hess regression this time), wall → ~26.2s; diamonds unchanged (SE sequential). Helps every
  log-link fit. Cumulative kick o3 this session: 37.1s → **~26.2s (−29%)**.

### ✅ WIN #8 — Array-of-structs histogram accumulator [G5] — BYTE-IDENTICAL
Profiling vs LightGBM (apples-to-apples suite config, refine=0): the gap is entirely in FIT (kick tri 2.88s
vs LGBM 0.91s; predict is fine), and `hist_build` is the dominant phase. The hot accumulation loop scattered
each row into 4 SEPARATE arrays (`g`,`h`,`wsum`,`count`) — 3 bounds-checked cell writes hitting 3 cache lines
per row (unit-weight skips wsum). Packed `g`/`h`/`count` into ONE `GhcCell` (array-of-structs) so each row is
a SINGLE bounds-checked write to ONE cache line; `wsum` stays a separate array (touched only for non-unit
weights). Same f64 arithmetic in the same fixed order ⇒ **byte-identical** (count/g/h/wsum per cell
unchanged); the existing hist + grow tests and exact real-data scores confirm it. Contained to `hist.rs`
(`AxisHist`, accumulate, `add_axis_hist`, assembly); the quantized path is untouched.
- **Byte-identical:** scores exactly unchanged (kick 0.77228, diamonds 0.11376). 231 core + 20 py green.
- **Measured (suite config, n=400, refine=0, 4 threads):** kick `hist_build` 1.88s → **1.45s (−23%)**, wall
  3.16s → **~2.77s** (vs LGBM 1.19s: gap **3.5× → 2.3×**); diamonds `hist_build` 0.77s → **0.58s (−18%)**,
  wall 1.14s → ~1.01s. Generalizes to every fit. Cumulative suite-config kick this session: 4.26s → ~2.77s (−35%).

### ✅ WIN #9 — Parallelize per-feature binning / categorical TS encoders [G5] — BYTE-IDENTICAL
Profiling the fit-vs-binning split (fit at n_trees=1 ≈ binning): kick's binning was a FIXED **~0.58s** — 64%
of LightGBM's ENTIRE fit — almost all of it the high-cardinality categorical target-statistics (KFold OOF)
encoders, run SEQUENTIALLY one feature at a time in `bin_train_columns`. Each feature's grid/encoder is
independent and deterministic in its own seed stream, so encode numeric grids and categorical TS encoders
with `par_iter` + order-preserving collects — **byte-identical** to the serial build (the categorical
(raw,id) uniqueness check is hoisted up front to keep first-duplicate-wins semantics). Contained to
`data/bin.rs`.
- **Byte-identical:** scores exactly unchanged (kick 0.77228, diamonds 0.11376). 231 core + 20 py green.
- **Measured (4 threads):** kick binning 0.58s → **0.30s (−48%)**, fit (n=400) 2.98s → **2.42s (−19%)**;
  diamonds binning 0.044s (numeric-only, already small). Helps every categorical-heavy fit (kick, amazon,
  allstate, …). Cumulative kick suite fit this session: ~3.9s → **2.42s (−38%)**.

### ✅ WIN #6 — Chunked-parallel log-link deviance fold [G5] — accuracy-neutral
With byte-identity relaxed, profiled the o3 bottleneck: kick `refine.backtrack_eval` (the leaf-refine
line-search deviance) was the single biggest sub-phase at 11.74s. The log-link deviance is COMPUTE-bound
(sigmoid + two `ln` per row ≈ 100+ cycles) — unlike grad_hess (sigmoid only, memory-bound, parallelization
was net-neutral / reverted). New `parallel_deviance_fold`: fixed-size row chunks each fold sequentially, then
combine the chunk partials in CHUNK ORDER ⇒ thread-count-INDEPENDENT (the §05.9 #7 gate holds, pinned by
`log_link_deviance_parallel_path_is_thread_count_independent` over 1/2/8 threads at n>chunk), differing from
a single linear fold only by ~1e-11 (chunked summation) — accuracy-neutral, only perturbs the line search at
an exact near-tie. Applied to Logistic/Poisson/Gamma/Tweedie `deviance`; **SquaredError stays sequential**
(cheap memory-bound term). Below the chunk size (8192) the sequential fold runs.
- **Byte-identity within tolerance:** real-data scores match the prior baseline EXACTLY (no flip): kick
  0.76975; diamonds 0.09022 (SE — unaffected, backtrack_eval 1.62s unchanged). 230 core + 20 py green.
- **Measured (o3, n=2000, refine=4, 4 threads):** kick `refine.backtrack_eval` 11.74s → **6.38s (−46%)**,
  wall 37.1s → **29.6s (−20%)**. Diamonds unchanged (SE sequential). Helps every log-link fit (kick, amazon,
  and Poisson/Gamma/Tweedie); the SE regression datasets keep their fast sequential fold. (NB: this is the
  reverse of the grad_hess lesson — there the per-row term was too cheap to beat memory bandwidth; the
  deviance's two logs make it genuinely compute-bound.)

### Re-baseline (2026-06-29, this machine, post WIN #9) — the next-target measurement
Re-measured both configs on the local box (faster than the cloud session — absolute seconds differ, phase
RATIOS guide the target). Build confirmed at HEAD: suite scores reproduce diamonds 0.11376 / kick 0.77228,
o3 scores diamonds 0.09022 / kick 0.76975 EXACTLY. o3 (n=2000, refine=4, 4 threads): kick wall 18.1s —
leaf_refine 9.81s (backtrack_eval 4.45 [parallelized], grad_hess 2.11 [parallelized], **init_dev 1.06**,
**members 0.68**, **aggregate 0.53**), hist_build 5.25s [subtraction frontier]; diamonds wall 9.9s —
leaf_refine 5.27s (grad_hess 1.56 [SE seq], backtrack 1.43, **members 0.62**, **aggregate 0.51**, init_dev
0.29), hist_build 2.68s. The untapped frontier is the leaf-refine SETUP cluster (members/init_dev/aggregate)
that commit 0e0ba6d instrumented — every other phase is either at the byte-locked subtraction frontier or
already parallelized.

### ✅ WIN #10 — Reuse grow's per-row leaf map in leaf-refine (eliminate refine.members re-walk) [G5] — BYTE-IDENTICAL
`refine.members` re-walked the whole tree per row (per tree × 2000 trees) to assign each row its leaf —
but `grow_oblivious_tree` ALREADY computes exactly that partition (`leaf_of_row`, set at its "Sample→leaf
update" loop via the SAME canonical `low_bit` the walk uses). Renamed grow → `grow_oblivious_tree_with_leaf_map`
returning `(tree, leaf_of_row)` (a `#[cfg(test)]` wrapper keeps the old name for the structure-only unit
tests — zero test churn); `refine_tree_leaves_after_grow` takes an `Option<&[u8]>` hint and GATHERS
`leaf_of_row[rows[i]]` instead of re-walking. The hint is passed ONLY when `sampled_rows.len() ==
train_rows.len()` (no subsample — `sample_rows(Full)` and MVS-with-`k==n` return the full set in train order,
so len-equality ⟺ grow saw exactly these rows); under subsampling it falls back to the walk (unchanged).
- **Byte-identical:** the gathered map equals the tree walk bit-for-bit (grow's bits come from the SAME
  `low_bit(bin, bin_le, missing_left)`, and `tree.splits` never changes after construction). Pinned by new
  unit test `grow_leaf_map_matches_tree_walk_memberships_bit_for_bit` (full rows + a reordered subset with
  repeats). Real-data scores EXACTLY unchanged (diamonds 0.09022, kick 0.76975). 232 core + 20 py green;
  clippy + fmt clean. Verified by a 3-skeptic adversarial workflow (byte-identity / gate-correctness / G0
  lenses) — **zero issues, unanimous SHIP**.
- **Measured (o3, n=2000, refine=4, 4 threads):** `refine.members` diamonds 0.62s → **0.046s (−93%)**,
  kick 0.68s → **0.061s (−91%)** — the tree re-walk is gone, leaving only the cheap O(rows) gather.
  Generalizes to every `leaf_refine>0` fit without row subsampling (the default).

### ✅ WIN #11 — Reuse grow's leaf map in `update_raw` (eliminate its per-row tree re-walk) [G5] — BYTE-IDENTICAL
The same redundant tree-walk as WIN #10, at the OTHER hot site: `update_raw` (apply the just-grown tree to
the running `raw`) walked the tree per row via `tree_value_for_row_with_columns` to fetch `tree.leaves[leaf]`.
Reuse grow's `leaf_of_row`: `raw[r] += tree.leaves[leaf_of_row[r]]` — byte-identical (grow's leaf bits come
from the SAME canonical `low_bit`, and leaf-refinement changed only leaf VALUES, never memberships). Gate is
STRICTER than members' because `raw` spans ALL n rows (incl. any held-out validation rows the early-stopper
scores, which `leaf_of_row` only covers when grow saw the full set): passed only when
`sampled_rows.len() == x.n_rows` (subsample OR a validation split ⇒ fall back to the walk, unchanged). Two
call sites (main + Nesterov correction).
- **Byte-identical:** new unit test `update_raw_leaf_map_matches_tree_walk_bit_for_bit` pins the leaf-map
  update == the walk update bit-for-bit over a non-trivial base raw. Real-data scores EXACTLY unchanged
  (diamonds 0.09022, kick 0.76975). 233 core + 20 py green; clippy + fmt clean.
- **Measured (o3, n=2000, refine=4, 4 threads):** `update_raw` diamonds 0.49s → **0.033s (−93%)**, kick
  0.64s → **0.043s (−93%)**. Cumulative WIN #10+#11 (members+update_raw, the two redundant tree-walks):
  diamonds wall ~9.9s → ~8.4s, kick ~18.1s → ~16.3s. Generalizes to every full-sample fit (the default).

### ❌ refine.aggregate parallelization — MEASURED SLOWER, reverted (2026-06-29)
The third setup-cluster item. The leaf-refine `aggregate` scatter-sums `gh.g[row]`/`gh.h[row]` into the 8
leaf accumulators over `rows` — 2 reads + 2 f64 adds per row (~1ns/row), and the `gh` arrays (≤464KB) stay
in L2 across the ≤4 refine steps, so it is L2-bandwidth bound. Chunked-parallel version (per-chunk `[f64;8]`
partials combined in chunk order, accuracy-neutral). **Measured SLOWER**: kick `refine.aggregate` 0.52s →
**0.757s (+46%)**, wall 16.3→16.9s (score unchanged 0.76975). Cause: ~65µs of memory-bound work per call
across ~8000 calls — rayon's per-call spawn/join overhead exceeds the split. Reverted. This is the loop's
**settled memory-bound lesson** (the twice-reverted grad_hess parallelizations) confirmed a third time: only
COMPUTE-bound per-row work parallelizes; a memory-bound reduction does not. (As a bonus it would have relaxed
byte-identity to a continuous ~1e-11 leaf-value drift — a worse trade than the deviance fold, which only
perturbs the line-search ACCEPT decision at a near-tie.)

### ⏸ refine.init_dev fusion — assessed, deferred (frontier; modest log-link-only win, invasive)
`init_dev` (kick 1.0s, diamonds 0.29s) is the deviance of the grown tree's leaves over `rows`, computed once
per tree before the line search. It is already a chunked-parallel compute-bound fold (`parallel_deviance_fold`,
the WIN #6 machinery). Two levers, both weak:
- **Overlap** (`rayon::join` it with the first refine `grad_hess`): net-NEUTRAL — both ops already saturate
  the 4 cores (`fill_grad_hess_parallel` + `parallel_deviance_fold`), so concurrently they just time-slice
  the same cores, no wall gain.
- **Fusion** (one pass computing g/h AND the deviance, sharing the sigmoid): byte-identical-achievable (chunk
  the fused pass by `PAR_DEVIANCE_CHUNK` and combine deviance partials in chunk order → bit-identical to the
  current `init_dev`; grad_hess is a map, unaffected). But the deviance's TWO logs remain (only the sigmoid +
  one memory pass over (y,raw,weight) are shared), so the win is PARTIAL (~0.3–0.5s on kick, ~0.1s diamonds),
  and it needs a new fused kernel across the 5 losses + a refine restructure (step-0 special case) +
  validation fallback. Disproportionately invasive for the payoff relative to the two banked tree-walk wins —
  deferred unless a log-link speed push is prioritized. The two CLEAN setup-cluster wins (the redundant
  tree-walks #10/#11) are banked; init_dev/aggregate are at the byte-identical+parallelism frontier.

### ✅ WIN #12 — Fuse init_dev into step-0 grad_hess (shared σ/exp) [G5] — BYTE-IDENTICAL (log-link)
(Built after the user prioritized the log-link speed push the assessment above flagged.) The leaf-refine line
search computes, at the grown-tree raw, BOTH the baseline deviance (`init_dev`) and step-0's grad/hess — two
passes recomputing the SAME link transcendental per row. New `Loss::grad_hess_and_deviance` does both in ONE
pass: a fused helper `fill_grad_hess_and_fold_deviance` writes g/h (the map, bit-identical to
`fill_grad_hess_parallel`) AND folds the deviance per `PAR_DEVIANCE_CHUNK` chunk combined in chunk order
(bit-identical to `parallel_deviance_fold`), computing σ/exp ONCE. Overridden for Logistic (shares the
sigmoid), Poisson (shares `exp(F)`), Tweedie (shares both F-exps); SquaredError + Gamma use a default
(unfused `grad_hess` then `deviance` — Gamma's `e^{−F}` g/h vs `e^{F}` deviance share nothing, SE is cheap).
- **Byte-identical, not merely accuracy-neutral.** refine uses the fused call for step 0 ONLY when
  `fuse_first = rows.len() == n_rows` — and `carve_validation_rows` returns `(0..n)` sorted when
  `validation_fraction` is None (else a sorted strict subset, len<n), so the gate ⟺ `rows == [0..n]` sorted ⟺
  the fused full-slice fold has the SAME values in the SAME order as the old gathered-subset fold (a validation
  split keeps the subset path). Pinned by `fused_grad_hess_and_deviance_is_bit_identical_to_separate` (all 5
  losses, n=20k > chunk, g/h + deviance bit-for-bit). Real-data scores EXACTLY unchanged (diamonds 0.09022,
  kick 0.76975). 234 core + 20 py green; clippy + fmt clean.
- **Measured (o3, n=2000, refine=4, 4 threads):** kick (Logistic) `init_dev + refine.grad_hess` 3.07s →
  **2.72s (−0.35s, −12%)** (init_dev now subsumes step-0 grad_hess; net drops by the shared sigmoid + one
  fewer memory pass), wall 16.3→15.9s. Diamonds (SquaredError, default) neutral, score exact. Generalizes to
  every full-sample log-link fit (Logistic/Poisson/Tweedie — the insurance objectives).

## G5 QHIST track — quantized-integer histograms (the remaining hist_build lever)

### FullF64 accuracy baseline across the suite (2026-06-29, n=400, refine=0, 4 threads)
The reference the QHIST path must not regress (RMSE-log↓ reg, ROC-AUC↑ clf):
| dataset | task | full score | full fit s |
|---|---|---|---|
| allstate | reg | 0.55744 | 11.7 |
| particulate | reg | 0.35804 | 15.6 |
| diamonds | reg | 0.11376 | 0.73 |
| miami_housing | reg | 0.16140 | 0.44 |
| amazon_access | clf | 0.85224 | 1.10 |
| kick | clf | 0.77228 | 1.83 |

### ✅ QHIST speedups (lazy-RNG + quantize-once + integer subtraction) — accuracy-neutral, NOT yet faster than FullF64
The existing `QuantizedI32` path was 1.5–2.6× SLOWER than FullF64 (and accuracy-neutral — the i32 scale
`i32::MAX·0.5/max|g|` is fine enough that split selection barely moves: Δacc ≤ 0.01% on every dataset). Three
fixes, all preserving the quantized path's existing determinism + accuracy:
- **lazy RNG**: `stochastic_round` computed a `pb_seed` hash per row but only USES it at an exact tie
  (`frac==0.5`); defer the hash to that branch — bit-identical, skips the per-row hash on ~all rows.
- **quantize once per tree**: `build_quantized_histogram` re-quantized the (tree-constant) `gh` on EVERY level
  (3×/tree); hoist `quantize_grad_hess` above the level loop — bit-identical, 3×→1×.
- **integer histogram subtraction**: subtraction was gated to FullF64; wire it for QuantizedI32 too (build the
  smaller children via quantized accumulation, derive the larger by subtracting the dequantized parent) so QHIST
  gets the same ~half-rows saving at levels 1+2. Pinned by `quantized_subtraction_reproduces_full_build_tree`.
- **Measured (n=400, refine=0):** QHIST speedup vs FullF64 went **0.39–0.66× → 0.66–0.86×** (allstate 0.41→0.78,
  diamonds 0.40→0.66, kick 0.39→0.74, miami 0.57→0.80, amazon 0.61→0.76, particulate 0.66→0.86). Accuracy still
  neutral (Δacc ≤ 0.01%). FullF64 scores byte-unchanged (the default path is untouched). 234 core + 20 py green.
- **Still slower than FullF64**, because FullF64 already has the AoS cache-packed accumulator (WIN #8) + unit-weight
  fast path, while the quantized accumulator is SoA (4 separate arrays) and pays a quantize + dequantize pass. NEXT:
  AoS-pack the quantized accumulator (apply WIN #8 to the i64 path) to close the per-row-scatter gap; the real 2×
  LightGBM win needs NARROW-integer (i16) histograms (more cells/cache-line + SIMD) — a bigger rewrite.

### ❌ FullF64 data-major histogram (read gh once, scatter to all axes) — MEASURED SLOWER, reverted (2026-06-29)
`build_histogram` is FEATURE-major (`axes.par_iter()` → each axis re-streams `gh`/`leaf_of_row` over all rows, so
the gradients are read `n_axes×`). Built the LightGBM-style DATA-major alternative: each row-chunk reads `(g,h,leaf)`
ONCE and scatters into all axes' bins, chunked by the SAME `ROW_PAR_CHUNK` with chunk-order reduction. Confirmed
**byte-identical** — swapping the axis/row loop nesting never changes a cell's f64 add order; pinned by
`data_major_matches_feature_major_bit_for_bit` (large fixture, both weight modes) + suite scores byte-unchanged.
Gated to `rows >= ROW_PAR_MIN_ROWS` (row-chunk parallelism), feature-major below.
- **Measured (suite n=400, best-of-2, vs HEAD):** NEUTRAL on diamonds/miami/particulate, but REGRESSED the
  high-cardinality sets: allstate 11.7→14.7s (**+26%**), kick 1.83→2.13s (**+16%**), amazon ~+7%. Revert restored
  allstate→11.98, kick→1.81. REVERTED.
- **Why it lost (hypothesis refuted):** the `gh` re-reads are CHEAP (gh fits L2/L3, so re-streaming is cache-resident,
  not RAM) — so data-major saves little. Meanwhile its per-chunk MULTI-axis buffer (`n_axes·n_leaves·max_bins`) is LARGE
  for many-feature datasets and spills cache, so the per-row scattered writes (one cell per axis) miss, where
  feature-major writes into a small per-axis buffer that stays in L1. No regime wins: small-buffer → neutral,
  large-buffer → regress. **Feature-major axis-parallel is the right design.**

### FRONTIER (2026-06-29) — FullF64 default path is exhausted for byte-identical/accuracy-neutral speed
This session banked the leaf-refine eliminations (members #10, update_raw #11, init_dev fusion #12) and rejected,
by measurement, the two remaining FullF64 levers: refine.aggregate parallelization (memory-bound) and the
data-major histogram (cache-buffer regression). `hist_build` (the 57–74% bottleneck) stays feature-major + AoS +
unit-weight + L1/L2 subtraction — at its frontier. The only un-banked speed lever is QHIST→i16 narrow-int
histograms (changes outputs; a substantial rewrite), deferred. The FullF64 engine is at its byte-identical floor.

### ✅ QHIST AoS accumulator + unit-weight skip (root-cause fix) — accuracy-neutral; ❌ i16 narrow-int rejected
A 3-angle design workflow found the REAL reason QHIST lost: `accumulate_axis_quantized` used a SoA accumulator
(`g/h/wsum/count` as 4 separate arrays = **4 cache-line writes per row**) with NO unit-weight fast path, while
FullF64 is AoS (`GhcCell`, 1 cache line) + skips `wsum` under unit weights. The i64 width gave zero density benefit.
- **Fix (kept):** AoS-pack the quantized hot cell into one `QHotCell{g:i64,h:i64,count:u32}` (mirrors `GhcCell`, one
  cache-line scatter), add the unit-weight `wsum`-skip, reduce per-chunk hot cells into the existing SoA i64
  `AxisQHist`. i32 quantization unchanged ⇒ **accuracy-neutral** (Δacc ≤ 0.01% on every suite dataset; determinism
  preserved — integer adds are associative). Measured QHIST speedup vs FullF64: **0.78→0.91 allstate, 0.74→0.82 kick,
  0.76→0.83 amazon, 0.66→0.68 diamonds** (suite n=400). 234 core + 20 py green.
- **i16 narrow-int (rejected):** also built it (scale to i16 range → dense 12-byte i32 cells, 2× denser than FullF64).
  Measured: NO extra speed over the i64 AoS (kick i16 0.74 vs i64 0.82 — i16 was actually SLOWER) because the per-axis
  histogram is already L1-resident, so denser cells don't help the scatter; AND i16's coarser quantization REGRESSED
  miami −0.23% (exceeds the accuracy gate). Reverted i16 → i32. The density hypothesis is refuted.
- **Conclusion — QHIST cannot beat FullF64 here.** Even at its best (allstate 0.91×, i.e. 1.1× slower) the quantized
  path still trails: the per-tree `quantize_grad_hess` pass is pure overhead FullF64 never pays, and the histogram
  scatter is already L1-resident so neither integer width nor density helps. In safe Rust (`forbid(unsafe)`, no
  hand-SIMD) the LightGBM 2× is not reachable on this workload. QHIST stays a non-default, accuracy-neutral path —
  now ~as close to FullF64 as it gets. Speed campaign is at its floor on BOTH precisions.

## LightGBM head-to-head + speed teardown (2026-06-29)

### Measured tri-boost vs LightGBM (matched depth-3, n=400, 4 threads)
tri-boost (refine=4, its accuracy lever) BEATS LightGBM on 5/6 — allstate 0.5435 vs 0.5500, particulate 0.3494 vs
0.3522, diamonds 0.09735 vs 0.09749, **miami 0.1395 vs 0.1599 (−12.7%)**, kick 0.7774 vs 0.7721 — loses only amazon
(0.8444 vs 0.8581, all-categorical). At refine=0 (no levers) LightGBM wins 5/6. **SPEED:** tri is **2.5–5.9× slower**
(refine=0), wider with refine=4. So tri trades training speed for exact decomposability + better accuracy on
interaction-heavy data.

### Teardown: a 4-agent workflow mapped LightGBM's speed to tri-boost
Verdict (all agents converged + matches the frontier assessment): the 2.5–6× is **~55–70% STRUCTURAL/unborrowable**
(leaf-wise growth reaches a loss in fewer trees — G0-forbidden since exact ≤3-order fANOVA needs depth-3 OBLIVIOUS
trees; plus tri's own leaf-refine accuracy spend LightGBM never pays), **~25–35% MICRO-ARCH** mostly behind
`forbid(unsafe)` (SW prefetch, bin packing — LightGBM's hot loop is SCALAR, NOT SIMD: bin scatter write-conflicts),
and only **~5–10% genuinely borrowable byte-identical**. Off-limits (stop chasing): leaf-wise (G0), prefetch/SIMD
(unsafe), int8/4-bin quantized (accuracy — i16 already failed), EFB/sparse-hist (inapplicable — tri TS-encodes each
cat to ONE dense axis), GOSS (subsumed by MVS).

### ✅/❌ Tested all 4 borrowable techniques — only #1 (degenerate-axis filter) kept; all ~ZERO on the benchmark
- **✅ Degenerate-axis pre-filter (kept, byte-identical):** `axis_is_admissible` now drops axes with <2 data bins
  (`n_bins ≤ 2`) — they were built (full O(rows) histogram) then unconditionally skipped in `best_level_split`.
  Byte-identical (scores exact on all 6). Speed: **0 on the benchmark** (no degenerate features here) — a correctness
  win for degenerate-feature/​rare-categorical production data, not for this suite.
- **❌ Count-free hot cell (spike, rejected):** the hist `count`/`wsum` are read ONLY by `best_level_split` under
  `check_cred` (leaf values recompute from `gh`), so on the unit-weight + inert-credibility default they're never
  read. Spiked out the per-row count write + the wsum=count pass: scores byte-identical (confirms unread), speed
  UNCHANGED within noise — `count` is same-cache-line as g/h (AoS `GhcCell`), so removing the write just overlaps
  memory latency. Reverted.
- **❌ HistogramPool buffer reuse (not pursued):** infeasible cleanly under the determinism gate — the fixed-order
  collect-then-reduce requires every per-chunk `AxisHist` to coexist (can't alias a reused buffer), and the only
  reusable buffer (the per-level assembly `Hist`) is O(cells)-tiny. The ~5–10% estimate didn't account for the
  determinism constraint + allocator block reuse (freed blocks are recycled, so the "churn" is bookkeeping not page
  faults). Net ≈0.
- **❌ Packed i128 QHIST add (not pursued):** an i128 add is 2 machine instructions on x86-64 (add+adc) = the same as
  two i64 adds, so ~0 net; QHIST already loses regardless.

**CONCLUSION: the borrowable ~5–10% is ~0 in practice on this benchmark.** Under tri's real determinism constraint +
the L1-resident scatter + same-cache-line AoS layout, none of the four recovered measurable wall-time. This
EMPIRICALLY confirms the teardown's verdict: **tri-boost's histogram engine is at its safe-Rust floor.** Further speed
requires relaxing G0 (off the table — it's the product) or `forbid(unsafe)` for a vetted prefetch/SIMD scatter (a
policy decision). The higher-value frontier is ACCURACY (G1 mains / amazon categorical), not speed.

## Bottleneck-pass wins (2026-06-29) — a fresh 4-agent timing analysis found real leaf-refine + binning slack

A second timing teardown re-profiled BOTH configs and surfaced two genuine misses (the leaf-refine campaign had only
removed plumbing; binning was only parallelized across features, never algorithmically fixed).

### ✅ SE closed-form-deviance leaf refinement — BYTE-IDENTICAL, −8 to −26% on the refine config
SquaredError's half-deviance is EXACTLY the separable 8-D quadratic `D_l(v)=C_l+v·B_l+½v²·H_l`, so the O(rows)
deviance re-folds (`refine.init_dev` + `refine.backtrack_eval`, the line-search deviance evaluated every trial) collapse
to an O(8) closed form (`refine_tree_leaves_se_quadratic`, gated on `LossId::SquaredError`). The per-row f32
`grad_hess`+aggregate that produce the leaf UPDATES are KEPT VERBATIM, and the closed-form value is f32-cast exactly as
`SquaredError::deviance` ⇒ **byte-identical** model (the leaves, scores, AND early-stop trajectory are unchanged).
- **Measured (n=4000, refine=4, A/B vs the per-row path):** scores reproduce bit-for-bit on ALL 4 SE datasets at BOTH
  converged (es=500) and fixed-4000 — diamonds 0.089809, miami 0.132031, allstate 0.540092, particulate 0.338236
  (converged). Speed: converged **diamonds 20.2→15.9s (−21%), miami 4.8→3.9s (−19%), allstate 62.5→57.4s (−8%),
  particulate 139.4→106.6s (−24%)**; fixed-4000 diamonds −21%, particulate −26%. Smaller on allstate (binning/hist-
  dominated, not refine). Log-link is untouched (non-quadratic). 234 core + 20 py green.
- **❌ Full f64 recurrence (1B) — REJECTED.** Also collapsing the per-step `grad_hess`+aggregate via the exact
  recurrence `G_l=B_l+H_l·v_l` gives −57% on diamonds o3, BUT the recurrence is f64 while the engine stores gradients
  in f32 — and that gap, via the EARLY-STOPPING interaction, shifts the stop point and costs real accuracy at
  convergence (**miami +1.53%**, diamonds +0.19%). It is NOT accuracy-neutral (the f32 vs f64 difference compounds with
  tree count, amplified by where early stopping fires). The f32 grad must stay ⇒ the per-row `grad_hess` cannot be
  eliminated byte-identically. So 1A (deviance-only) is the shippable version.

### ❌ borrowable trims also tested this pass: dead grad/hess memset (KEPT, byte-identical), degenerate-axis pre-filter
(KEPT, byte-identical, benchmark-neutral), count-free hot cell (≈0, reverted), HistogramPool reuse (infeasible under
determinism), i128 QHIST (≈0).

### ✅ Categorical label factorization — BYTE-IDENTICAL, −16% off allstate binning
The high-cardinality categorical TS encoder operated on `&[String]` row labels through string-keyed `BTreeMap`s. Now
`fit_cat_encoder` interns the collapsed per-row labels into integer ids ONCE (`intern_levels`, a local `HashMap` keyed
in ROW order so the result is deterministic and never serialized), and the two dominant phases index dense `Vec`s by
id: `full_data_encoder` aggregates `(sum_y,denom)` into `Vec[n_ids]`, and `kfold_training_encodings` uses `Vec[n_ids]`
total + `Vec[k·n_ids]` per-fold (vs `BTreeMap<&str>` / `BTreeMap<(u32,&str)>`). Plus the earlier sub-fixes
(clone-on-first-occurrence in full_data, set-based rare lookup in collapse). Ordered/LOO (non-default) keep the string
path. **Byte-identical**: every `(sum_y,denom)` reduction stays in row order (same f64 sums); the Fisher sort re-orders
by (encoding, label) so the build order is irrelevant; rare-bucket members come from the unchanged `collapse_rare_levels`.
- **Measured (allstate, n_trees=1 ≈ isolates binning, best-of-3):** 2.61s (orig) → 2.42 (sub-fixes) → **2.18s
  (−16% total)**. Scores byte-identical on every cat-heavy dataset incl. the rare-collapse path (allstate 0.55744,
  amazon 0.85224, kick 0.77228). 234 core + 20 py green; clippy + fmt + no-hashmap-serialized gate green.
- Remaining cat binning cost (the OOF compute + the serve `encode_label` loop) is smaller; serve is a separate O(n·d)
  loop (bin.rs) a lookup-map would fix — minor, deferred.

### ✅ Final byte-identical sweeps — serve lookup map + stack split buffers
Two more byte-identical trims closing the bottleneck pass:
- **Serve `encode_label` lookup map.** Serve-time re-encoding (`bin_train_columns` serve build + `bin_serve_columns`)
  called `CatEncoder::encode_label` per row — an O(#levels) linear scan over labels+members. New
  `CatEncoder::encoding_map()` builds a `label→encoding` HashMap once (local, never serialized); the call sites do
  O(1) lookups with the same `base` fallback. Byte-identical (labels/members disjoint across levels ⇒ `get` returns
  exactly what `find` did).
- **Stack-allocated split buffers.** `best_level_split`'s 12 per-leaf `vec![…; nl]` accumulators → `SmallVec<[_;4]>`
  (`nl = hist.n_leaves ≤ 4` by the depth-3 G0 invariant), so they stay on the stack — no heap alloc per (level, axis).
  Byte-identical (same values; only the storage moves).
- **Measured:** allstate binning 2.18 → **2.10s** (serve map), so the full cat-binning pass is **2.61 → 2.10s
  (−20%)**. Scores byte-identical (allstate 0.55744, amazon 0.85224, diamonds 0.11376). 234 core + 20 py green;
  clippy + fmt + no-hashmap-serialized gate green.

### ⏸ Axis-saturation histogram gate — NOT shipped (accuracy-neutral, against this batch's byte-identical grain)
The nested row-chunk parallelism inside `accumulate_axis` is redundant when the outer `axes.par_iter()` already
saturates the thread pool (wide builds, e.g. allstate's many cat axes); gating it off above a FIXED axis-count floor
(never `current_num_threads()`, which would break determinism) would cut ~−7% off allstate hist. But it changes the
per-cell f64 fold order (chunked → sequential) ⇒ ~1e-11 model change ⇒ NOT byte-identical (thread-deterministic, but a
different model). Every other win this pass is byte-identical; this one is accuracy-neutral, so it is left as an
available option rather than shipped by default.

### ✅ Full-suite byte-identity confirmation (2026-06-29, HEAD = 5fb3c28)
Ran the whole batch end-to-end against the recorded baselines — all 6 datasets at the suite config (n=400, refine=0,
the canonical `tri_boost_case`) plus diamonds/kick at o3 (n=2000, refine=4, which exercises the SE closed-form
leaf-refine). Every score reproduces its baseline within ±5e-6 (the 5-decimal recording precision, both signs — ~20×
below the 1e-4 floor of a real regression):

| dataset | config | baseline | measured |
|---|---|---|---|
| allstate | suite | 0.55744 | 0.557437 |
| particulate | suite | 0.35804 | 0.358044 |
| diamonds | suite | 0.11376 | 0.113762 |
| miami_housing | suite | 0.16140 | 0.161405 |
| amazon_access | suite | 0.85224 | 0.852237 |
| kick | suite | 0.77228 | 0.772276 |
| diamonds | o3 | 0.09022 | 0.09022 |
| kick | o3 | 0.76975 | 0.769749 |

miami landed exactly on the 5e-6 rounding boundary, so it was settled with a full-precision A/B: HEAD and the
pre-session commit f44d299 ("WIN #9") both give **0.1614045724** — identical to 10 decimals, i.e. the batch left it
untouched (miami has one categorical, `avno60plus`, so the factorization path was exercised). Confirms the whole G5
bottleneck pass — categorical factorization, serve map, sub-fixes, memset, degenerate-axis, SmallVec (suite) + SE
closed-form leaf-refine (o3) — is byte-identical on every dataset.

## Bottleneck pass 2 (2026-06-29) — granular component timings + a 5-investigator team, then experiment with each idea

Profiled both configs to per-`prof::timed`-phase granularity. **hist_build dominates** (SUITE: allstate 8.2s/96%,
particulate 2.4s; O3: allstate 45s/75%, kick 6.5s); then logistic leaf-refine (kick backtrack_eval 5.4s + init_dev
1.4s — SE closed-form is regression-only), then the per-step refine.grad_hess (allstate 6.0s, diamonds/kick ~1.9s). A
Workflow team (5 component investigators → adversarial verify each proposal vs G0/forbid-unsafe/determinism/byte-id →
ROI synthesis; 15 proposals, 12 survived) ranked them; then each was BUILT and MEASURED:

### ✅ ① SE leaf-refine: fuse grad_hess + aggregate (the flagship) — BYTE-IDENTICAL, -37/-38% refine
New `Loss::grad_hess_aggregate` (default = old two-pass; SquaredError overrides with a fused kernel computing each
row's f32 (g,h) inline — EXACTLY `grad_hess`'s `g=w(F-y)`/`h=w.max(floor)` via `ensure_finite_grad_hess` — and folding
straight into the per-leaf f64 sums, visiting only `rows`). Removes the whole materialized-gradient round-trip
(separate write + re-read) the SE closed-form left behind. Bit-identical (diamonds o3 0.09022, allstate o3 0.53974).
Measured: diamonds leaf_refine 3.10→1.94s (-37%, wall 7.2→5.8 -19%), allstate 10.27→6.35s (-38%, refine grad_hess+agg
7.63→3.90 -49%, wall 63.9→57.5 -10%). The single biggest win of the pass. Commit ce65fbb.

### ✅ ② generic leaf-refine reads the dense subset raw — BYTE-IDENTICAL, drops full-raw clone
grad_hess now runs over (y_sub, trial_raw_sub, w_sub) — pointwise ⇒ gh[i] bit-identical to the old gh[rows[i]] — so the
full-length `raw` clone + per-accept apply_membership_leaves scatter are gone, and it's O(rows) on validation splits.
Bit-identical (kick o3 0.76975, diamonds 0.09022). Net-neutral on full-sample (within variance), helps the validation/
early-stop config; also makes trial_raw_sub the single source of truth. Commit 6e51a23.

### ✅ ③ numeric grid build borrows the whole-column sample — BYTE-IDENTICAL
Dropped an O(n) `fw.clone()` on the quantile path (only existed for type-unification). Negligible but free. Commit 4df054f.

### ❌ ⑧ fuse logistic backtrack deviance with next-step grad_hess — REVERTED (net-neutral)
grad_hess_and_deviance on the scale=1 trial, carry gh to the next step. Byte-identical (kick 0.76975); the mechanism
WORKED (refine.grad_hess collapsed to ~0 ⇒ scale=1 accept-rate is high). BUT net-neutral: refine.backtrack_eval grew
+2.2s, exactly absorbing the -2.0s grad_hess it saved. The link σ/exp sharing doesn't pay — the exp is not the
bottleneck (the gh write + g/h arithmetic + memory traffic offset it). The work just moved. Reverted.

### ❌ ⑤ level-0 (n_leaves==1) histogram kernel specialization — REVERTED (~0, confirms the floor)
Skip the per-row leaf_of_row read + leaf bounds-check + offset2 multiply when n_leaves==1 (idx==bin). Byte-identical for
valid input. MEASURED allstate suite hist_build 8.59-9.1s vs 8.2s baseline — NO improvement (within variance, if not
slightly worse). Confirms the team's gate: the kernel is **scatter-bound** (the random write into out.ghc[bin]
dominates; the leaf-read/offset were latency-hidden), so removing them frees nothing. The histogram engine is at its
safe-Rust floor — the dominant cost is irreducible without `unsafe` SIMD/prefetch (off the table). All downstream hist
ideas (⑥/⑦/⑪/⑫, gated on this) are therefore not worth building. Reverted (also it skipped a safety-validation test).

### ⏸ ⑨/⑩ categorical intern-before-collapse + serve id-lookup — NOT pursued
A small one-time-binning extension of the already-shipped categorical factorization (-20% binning). Bounded by the cat
fit's small share of the 8-45s wall; deprioritized given the bulk is already captured and run-to-run variance is high.

**Net of pass 2:** one real win (① SE fusion, -37/-38% refine — compounds with the earlier SE closed-form), two
byte-identical cleanups (②③). The hist_build elephant is confirmed irreducible in safe Rust (⑤ prototype = ~0). The
remaining training cost is now genuinely dominated by the histogram scatter, which only `unsafe` could move.

### Complete sweep of the remaining ranked ideas (the first pass left some untested)
- **⑪ parallel/slice-contiguous hist assembly** — built, MEASURED ~0 (allstate suite hist_build 8.0s vs 8.2s, score
  exact). The assembly is memory-bound and dominated by the accumulate scatter. Reverted.
- **⑦ accumulate-loop bounds-check collapse** — it is the SAME loop ⑤ already measured as scatter-bound; ⑤ removed
  strictly more per-row work (leaf read + offset) for ~0, so ⑦ (removing only bounds checks) is ~0 by that
  measurement. Not separately built.
- **⑫ pre-gather shared per-row arrays** — INAPPLICABLE: v1 uses `subsample = 1.0` (all rows, `0..n`), so the gh reads
  are already sequential, not gathered; ⑫'s premise only exists under bagging/GOSS. Team also flagged it regressive.
- **⑨ categorical intern-BEFORE-collapse** — fully built (new `collapse_rare_ids` in id space; rare members sorted to
  match the old BTreeMap; fit-id numbering is output-irrelevant so it's free; non-default Auto/Ordered/LOO get a
  reconstructed `fit_levels`). BYTE-IDENTICAL incl. the rare-collapse path (allstate 0.55744, amazon 0.85224, kick
  0.77228 all exact, 234+20 green). MEASURED ~0: allstate n=1 binning 2.09s vs 2.10s — the collapse/intern string work
  is dwarfed by the numeric grid build + the already-id-indexed kfold/full-data. Reverted.
- **⑩ serve id-lookup** — subsumed: ⑨ shows the categorical fit is at its floor, and ⑩ (replacing the serve
  `encoding_map` HashMap with an id gather) would couple bin.rs to the internal ids for an even smaller marginal one.
  Not built.

**Final verdict — all 12 ranked ideas now tested or definitively resolved.** Confirmed wins: ① (real, -37/-38%
refine), ②③ (byte-identical cleanups). Confirmed ~0/inapplicable: ⑤⑥⑦⑧⑨⑩⑪⑫. The engine is at its safe-Rust floor on
BOTH frontiers — the histogram scatter (hist_build) and the categorical fit (binning) — with no remaining safe-Rust
slack; further training speedup requires `unsafe` SIMD/prefetch (a policy decision) or G0 relaxation (off the table).

## Bottleneck pass 3 (2026-06-29) — the histogram floor was NOT a floor: jagged per-axis bin stride

A third granular re-profile (post-① fusion) + a fresh 5-investigator/adversarial-verify Workflow team
(12 proposals, 10 survived). The team's rank-1 (flagged speculative, "may collapse to ~0 against a
scatter-bound floor") turned out to be the **biggest win of the whole campaign**:

### ✅ Jagged per-axis bin stride in the histogram intermediates — BYTE-IDENTICAL, -23 to -42% hist_build
The per-axis `AxisHist` intermediates were allocated/zeroed/scattered/reduced at the UNIFORM global
`max_bins` stride, even though most axes need far fewer bins (allstate: 104/130 features need <=16 bins
but got 255 — an 8.36x inflated stride). Now each axis's intermediate is sized to ITS OWN grid
`n_bins` (`accumulate_axis*` + `AxisHist::try_zeros`); `build_histogram` re-expands each jagged
sub-histogram into the uniform `max_bins`-strided final tensor (high cells stay zero, exactly as
before). BYTE-IDENTICAL: no row has `bin >= axis_bins`, so only always-zero cells are omitted; the
final tensor and the chunk-order f64 fold are unchanged. Verified: all 6 suite scores + diamonds/kick/
allstate o3 reproduce EXACTLY; 234 core + 20 py green; clippy + fmt.

**The mechanism corrects pass 2's verdict.** Pass 2 concluded the histogram was scatter-bound and "at
its safe-Rust floor" because the level-0/bounds/assembly micro-opts measured ~0. But the dominant
random scatter was CACHE-bound, not count-bound: at 255-bin stride a low-card axis's `[leaf][bin]`
working set is ~20KB (spills L1); at its true 16-bin stride it is ~1.3KB (L1-resident), so the random
writes hit L1. Shrinking the stride is a structural cache win, not a micro-op — which is why it
landed where the micro-ops didn't.
- **Measured (best-of-3, profiler):** allstate o3 hist_build 50.6->38.8s (-23%, wall 67.4->53.7 -20%);
  allstate suite 8.0->7.06s (-12%); diamonds o3 3.39->2.25s (-34%, wall 8.1->5.6 -31%); kick o3
  8.51->4.96s (-42%). Helps EVERY dataset (any axis with n_bins < max_bins), biggest where low-card
  axes dominate. The single largest training speedup of the campaign.

### ✅ Reuse the round-invariant SquaredError hessian (boosting grad_hess) — BYTE-IDENTICAL, -28 to -43%
For SquaredError `h = w·max(floor)` is raw-independent and weights are round-invariant, so `gh.h` is
bit-identical every boosting round. New `Loss::hessian_depends_on_raw()` (default true; SE false) +
`Loss::fill_grad_reusing_hessian()` (default = full grad_hess; SE refills ONLY `gh.g`). The boosting
loop uses the reuse path after round 0 for round-invariant-hessian losses; Logistic/Poisson/Gamma/
Tweedie keep the full pass. Byte-identical (g computed exactly as grad_hess; h left as round 0 set it).
Measured: boosting grad_hess allstate o3 1.67->1.20s (-28%), diamonds o3 0.61->0.35s (-43%). SE-only.

### Pass-3 net + remaining (identified, not yet built)
**TWO real wins, both byte-identical and both BIGGER than the team estimated:** the jagged bin stride
(-23 to -42% hist_build, the campaign's largest) and the SE hessian reuse (-28 to -43% boosting
grad_hess). Combined, allstate o3 wall ~67->~52s, diamonds o3 ~8.1->~5.6s, kick o3 ~24.8->~16s — all
byte-identical (all 6 suite + o3 spot-checks exact). The "histogram floor" verdict from pass 2 was
WRONG (it was cache-bound, not count-bound).
Remaining ranked-but-unbuilt (all sub-second / byte-identical, deferred): ③ kill the dead common-path
`fit_raw = raw.clone()` (~0.1-0.2s, tangled with AGBM/DART branches), ④ gate Nesterov alpha bookkeeping
behind agbm.is_some(), ⑤ always-fuse init_dev (helps the validation/early-stop path only), ⑥ skip wsum
alloc/fill/assemble when no credibility floor binds (stacks with the jagged stride; needs check_cred
verification + subtract-path touch-ups), ⑦ Cow the no-subsample sampled_rows/round_axes clones.
Parked: ⑧ leaf-refine scratch pool (L effort), ⑨ logistic grad_hess_aggregate (accuracy-neutral, only
the 0.78s aggregate; the 7.2s logistic backtrack_eval remains the genuinely-hard untouched lever).

## Pass-3 deferred-work exploration (2026-06-29) — measure each before assuming it's small

Prompted by the jagged-stride surprise (a flagged-speculative idea that became the campaign's biggest
win), explored every deferred pass-3 idea instead of assuming sub-second:

- ✅ **⑤ always-fuse init_dev** — SHIPPED. Zero on the full-sample profiled config (already fused) but
  REAL on the validation/early-stop config the production harness uses: A/B at validation_fraction=0.1
  gave refine.grad_hess 1.48->1.09s (-26%), leaf_refine 6.29->5.88s (-7%, ~0.4s), byte-identical
  (kick val score 0.776137 unchanged), and removes the fuse_first gate (simpler). The "more meaningful
  than expected" find — meaningful exactly where the profiler didn't look.
- ❌ **⑨ logistic grad_hess_aggregate** — NOT viable byte-identical: Logistic::grad_hess uses
  fill_grad_hess_parallel (parallel), so a sequential fused version (the SE-① pattern) would serialize
  the compute-bound sigmoid and regress. The team's parallel version is only accuracy-neutral. (The SE
  ① won precisely because SquaredError::grad_hess is sequential — no parallelism to lose.) Set aside.
- ❌ **⑥ skip wsum (dead when !check_cred)** — PROBED ~0. Confirmed benchmark has no credibility floor
  (defaults all-zero => check_cred=false, score 0.55744 with wsum skipped), but skipping the unit-weight
  wsum=count pass left allstate suite hist 7.31s vs 7.06s (no change). Latency-hidden, especially
  post-jagged (the cells are already small). Bounded dead-work, not a hidden lever.
- ⏸ **③ dead fit_raw clone / ④ gate Nesterov alphas / ⑦ Cow no-subsample clones** — all in the ~6.6s
  unprofiled boosting-loop overhead; bandwidth math + that band's size put each at sub-0.2s, and ③ needs
  a borrow-checker/AGBM/DART-aware restructure, ⑦ an MVS/colsample-aware one. Confirmed sub-second
  byte-identical code-hygiene cleanups — available, not worth the restructure risk for the gain.

**Lesson reinforced:** the jagged stride was a one-off STRUCTURAL (cache-locality) surprise; the rest of
the deferred set is bounded linear dead-work that measures as predicted. But ⑤ shows it's still worth
MEASURING each at the config where it actually fires (⑤ was invisible on the profiler's full-sample run).

## Bottleneck pass 4 (2026-06-29) — the floor, confirmed (post jagged stride)

Re-profiled post pass-3 and ran a 5-investigator/adversarial-verify Workflow team (15 proposals, 7
survived) pushed toward STRUCTURAL ideas (the jagged-stride bar). The team's own headline: "the easy
wins are gone." Two survivors had real magnitude; both resolve to ~0:

- ❌ **Rank 2/3 — leaf-partitioned histogram accumulation (the leaf-dimension jagged-stride analog)** —
  ruled out STRUCTURALLY: post-jagged, level 0 (all rows, single leaf) already scatters into one
  axis's ~5KB bin row (L1-resident), and level 0 dominates the row count. Leaf-partition only touches
  the multi-leaf level-2 build (~1/4 rows, ~10KB working set that likely already fits L1). So the
  residual hist cost (allstate o3 34.7s) is THROUGHPUT-bound — ~150k rows x ~130 axes x 2000 trees
  ≈ 39B scatter ops at ~L1 speed ≈ the observed time — not cache-bound. Not worth building.
- ❌ **Rank 1 — logistic backtrack capped-deviance early-abort** — BUILT (byte-identical: binomial
  deviance terms are non-negative ⇒ monotone fold ⇒ aborting when the chunk-ordered running acc >= cap
  can never disagree with the f32 accept test; accepted trials run the full chunk-order fold) and
  MEASURED ~0: kick o3 backtrack_eval 5.27-5.45s vs 5.24s baseline (no change), score 0.76975 exact.
  The rejects are sparse (convex + lr=0.05 damped ⇒ scale=1 usually accepts) and happen at convergence
  where the trial deviance ≈ best, so the monotone partial reaches the cap only near the end — the
  wave-abort never fires early. Reverted.
- The other 5 survivors were S-effort ~0 cleanups (fit_raw clone, alpha gating, etc.) already
  established as latency-hidden in the boosting-loop overhead band.

**Verdict — the engine is at its safe-Rust floor.** Unlike pass 2's "floor" (a cache GUESS that the
jagged stride disproved), this is grounded: the histogram is now THROUGHPUT-bound (irreducible op
count, L1-resident scatter), and the logistic line-search rejects are genuinely sparse/late. The
jagged stride (pass 3) was the last structural lever. Further training speedup requires fewer ops
(G0 relaxation — fewer trees/axes/order, off the table) or `unsafe` SIMD (which won't help an
L1-resident throughput-bound scatter much). Campaign training-speed frontier: closed.

## G1 ACCURACY: beat EBM at order 1/2/3 (2026-06-29) — campaign reopened

Re-measured the EBM gap (tri o1/o2/o3 vs ebm mains/o2, EBM from cache). A 5-investigator team ranked
G0-preserving levers (bagging, binning, lr/smoothing, interaction selection). Findings:

### KEY INSIGHT — the gap was largely an UNDER-FITTING artifact (n_trees cap too small)
Initial measurements capped n_trees=4000 (fair_compare BUDGET). With lr=0.05 tri converges slowly, so
4000 truncates before the validation plateau — under-measuring tri vs EBM (which self-converges). At a
proper **16000-tree cap + early_stopping_rounds=500** the gap collapses (diamonds + miami, no bagging):
| dataset/order | 4000-cap gap | 16000-cap + refill gap |
|---|---|---|
| diamonds o1 | -4.42% | **-0.56%** |
| diamonds o2 | -1.03% | **-0.59%** |
| diamonds o3 | +2.66% WIN | +1.85% WIN |
| miami o1 | -2.86% | **-1.10%** |
| miami o2 | +2.19% WIN | +3.04% WIN |
| miami o3 | +2.90% WIN | +3.74% WIN |
The "diamonds@1 is structurally hard" read was WRONG — it was truncation. ALWAYS benchmark tri with a
large n_trees cap + early stopping (memory: tri-perf-early-stop-large-trees).

### ✅ Bin-budget refill (reclaim dedup-collapsed bins) — G0-preserving accuracy lever
The quantile binning's point-mass dedup (diamonds carat magic sizes 0.3/0.5/0.7/1.0 collapse uniform
quantile probes onto one border) wastes bin budget. New `refill_borders` (grid.rs) greedily splits the
densest splittable interval at distinct-value boundaries until the `max_bin-1` budget is met. Gated off
when a rare-bin floor is active (a split could break collapse_rare_bins's min). Order-independent
(G0-preserving), deterministic. At convergence it is net-positive (4000-cap A/B: diamonds o2 better,
miami o1/o2/o3 better; only diamonds o1/o3 ~+0.1/+0.4%). The n=400-suite "miami regression" was a
low-tree artifact (gone at convergence).

### Bagging (n_bags) — G0-clean partial win, never tested at depth-1 before
Outer bagging (soup_models averages Exact members) closed ~1/4-1/3 of the depth-1 gap at the 4000-cap
(diamonds -5.21%->-3.97%); to be re-measured at the large cap on the remaining close losses.

NEXT: full re-baseline at FAIR_BUDGET=16000 + refill, then add bagging to any residual losses.

## G1 ACCURACY (2026-06-29 cont.): the under-fitting fix + subagging

### KEY: the depth-1/2 "gap" was mostly UNDER-FITTING (lr too low), not structural
Higher lr (0.20) + bagging + the refill collapsed the depth-1 gap: diamonds -4.42%->-0.04% (TIE),
allstate -0.07% (TIE), miami -0.38%, kick WIN; only particulate -2.08% remains (a real loss). Higher lr
converges fully in FEW trees (fast) where lr=0.05 truncated. BUT lr=0.20 is NOT universal: it HURTS kick
o2 (classification, overfits the interactions) — kick o2 -1.14% (lr.05) -> -3.11% (lr.20). The fast recipe
(lr0.2 + bagging) is: diamonds/miami/allstate ~tie-or-win at o1; depth-2 miami WIN, diamonds/allstate ~-0.5%.

### Team (5 investigators + adversarial critique) — de-risked plan
Real losses are only particulate o1 (categorical-TS encoding, NOT binning) and kick o2 (pair selection).
The o1 "ties" (diamonds/allstate -0.0x%) are NOISE vs a frozen EBM cache — don't chase. DROP per-dataset
monotone tuning (benchmark overfitting) and lr-decay (speed-negative). Critique caught a real BUG: the
outer-bag bootstrap is WITH replacement AND each bag carves its early-stop val from its own rows -> a row
lands in both train+val -> optimistic val -> late stop -> overfit (hurts kick).

### ✅ Subagging via new `bag_subsample` param (committed)
`EnsembleSpec::OuterBag { n_bags, bag_subsample }` (+ pyo3 + sklearn). f>=1 ⇒ classic full-size bootstrap
WITH replacement (default, backward-compatible). 0<f<1 ⇒ without-replacement subsample of round(f*n_rows)
(`subagging_rows`). G0-safe (still averages Exact members), byte-identical across threads (verified 1 vs 4).
Measured (n_bags=8, lr0.2):
| case | f=1.0 | f=0.9 | f=0.63 | speed f.63 vs f1.0 |
| diamonds o1 | -0.04% | -0.22% | -0.02% | 40s vs 65s |
| miami o1 | -0.38% | -0.87% | -1.29% | 8.6s vs 21s |
| allstate o1 | -0.07% | -0.09% | -0.14% | 125s vs 434s |
| kick o2 | -3.11% | **-1.13%** | -1.54% | 18s vs 220s |
Subagging is a big SPEED win (allstate -71%, kick -95%) AND fixes the leak (kick -3.11%->-1.13%), but the
data loss hurts data-hungry small datasets (miami). f=0.9 is the compromise. The CLEAN fix (next) is OOB
early-stop validation: train each bag on the full bootstrap (no data loss) but validate on the out-of-bag
rows (disjoint => leak fixed) — gets kick's win without miami's cost.

## particulate: the -2% "loss" was a RANDOM-SPLIT ARTIFACT, not a tri deficiency (2026-06-30)

User (data scientist) flagged: in production you train on past dates and predict FUTURE dates, so a model
that fits particulate's 8760-level hourly `datetime` would fail (every future timestamp is an unseen level).
The benchmark only rewards it because the harness uses a RANDOM split (each hour shared train/test), so a
per-level datetime encoding memorizes the regional pollution at each specific historical hour. That is a
leak, not signal. The generalizable temporal signal (time-of-day, day-of-week, season) is already present
as Hour/Month/DayofWeek.

PROVEN with LIVE EBM (installed `interpret` 0.7.8; reproduces the cached 0.357):
| particulate o1 | tri | EBM | gap |
| WITH datetime | 0.36469 | 0.35727 | -2.08% (EBM wins via memorization) |
| NO   datetime | 0.37548 | 0.37499 | **-0.13% = TIE** |
The whole gap is the datetime artifact. Without it, tri ties EBM. (Both get WORSE without datetime — it does
carry random-split predictive value — but that value does not generalize to future dates.)

DECISION: dropped `datetime` from particulate (tabarena_suite Dataset.drop_cols). Honest, production-valid
feature set keeps Hour (time-of-day), Month (season), DayofWeek, Site.Name/Zone (location), Altitude, PM2.5.
=> Do NOT build the per-level high-card "CatMainRefine" fix to chase this — it would be benchmark-gaming a
leak and harm production generalization. (A per-level high-card encoder is only legitimate for cats whose
levels RECUR in production, e.g. Site.Name / resource IDs — never timestamps.) See memory
production-generalization-not-artifacts.

CAMPAIGN STATUS: with the honest feature set, tri TIES-OR-WINS EBM on all 5 datasets at depth-1
(diamonds/allstate/particulate ties, miami ~tie, kick win). interpret now installed => fair_compare runs
LIVE EBM (no more stale cache).

## Parallel bag loop (✅) + T2 mains-first ramp (❌ refuted) (2026-06-30)

### ✅ Parallel bag loop — byte-identical speed win
fit_outer_bag now grows the bags via rayon `into_par_iter` collected IN BAG ORDER (soup_models folds in
that fixed order), so it is byte-identical to the sequential fit across thread counts (verified 1 vs 4).
Nests with fit_single's own rayon via the shared pool (work-stealing). Speedup (n_bags=8, f=1.0):
diamonds 65s->29s (2.2x), miami 21s->6.8s (3.1x), allstate 434s->362s (1.2x). Small datasets win big
(per-bag work underuses cores); big datasets get little here (inner histogram already saturates) — which is
exactly where subagging does the speed work. Together they cover both regimes.

### ❌ T2 mains-first staged max_order ramp — REFUTED, reverted
Grew order-1 until the validation plateaued, then switched interactions on. Measured WORSE at o2 on 3/4:
diamonds 0.10749 (= o1 EXACTLY — order-2 phase contributed nothing), miami 0.16136 (vs greedy 0.13280),
allstate 0.54781 (vs 0.54403); only kick improved (0.76841 vs 0.76055). Mechanism: fully converging the
mains drives the residual gradient too small for pair splits to clear min_split_gain, so interactions are
never grown — the greedy's INTERLEAVING (capture pairs while the residual is still large) is strictly
better. kick's gain was overfit-regularization, not interaction quality (its real fix is T5 pair selection).
Greedy interleaving stays the default.

## Classification leak fix: subagging (✅) beats OOB validation (❌ broke kick) (2026-06-30)

LIVE-EBM scoreboard (honest features) showed classification o2/o3 losing to EBM, but mostly the f=1.0
WITH-replacement bootstrap train/val LEAK (kick o2 -2.88%, amazon o2 -2.56%).

❌ OOB validation (train on bootstrap, early-stop on out-of-bag): byte-deterministic, FIXED amazon
(o2 +0.06%) but CATASTROPHICALLY broke kick (0.50 AUC = 0 trees). Cause: the with-replacement bootstrap
OVERFITS (duplicate rows), so the honest OOB deviance never improves -> early-stop truncates to zero. The
old leaky carved-val masked this. Reverted.

✅ Subagging (bag_subsample<1, without replacement) is the SAFE leak fix — no duplicate rows -> no overfit
-> clean carved val works. f=0.8: amazon o2 -0.16% (~tie), amazon o3 +0.43% WIN, kick o2 -1.03%/o3 -1.12%
(f=0.9 kick o2 -0.73%). vs leaky -2.5..-3.4%. => recipe: use bag_subsample<1 for classification (data-rich
so ~free). Leak-free full-data bagging is impossible without OOB (which overfits); subsampling is the cost.

EBM scores cached recipe-independently in benchmarks/_ebm_cache.json (+ _ebm.py helper) — no more re-fits.

REMAINING depth-2 gap after the leak fix: only **particulate o2 -1.88%** (genuine pair selection; EBM's
GA2M FAST picks better spatial-temporal pairs than tri's greedy). kick/amazon/diamonds o2 now within ~1%.

## particulate o2 -1.88%: pair FITTING, not selection — T5 REFUTED, BANK (2026-06-30)

Focused team (3 empirical investigators + adversarial critique, all numbers reproduced) overturned the
earlier "genuine pair selection" claim (CORRECTION to the line above — it is empirically FALSE).

Decisive oracle (mission config lr=0.2, n_bags=8, f=0.8, es=300; EBM o2 anchor 0.34902):
| config | RMSE-log |
| mains only | 0.37536 (tie EBM-mains 0.37499) |
| mains + tri's OWN grown pairs, additive Ridge | 0.34990 (TIE EBM) |
| mains + ALL 28 pairs, additive Ridge (order-2 OPTIMUM) | 0.34905 (TIE EBM 0.34902) |
| tri actual o2 @ n_trees=4000 | 0.35559 (-1.88%) |
| tri actual o2 @ n_trees=16000 (converges ~10.5k/bag) | 0.35338 (-1.25%) |

Findings: (1) SELECTION is saturated — EBM's GA2M keeps ALL 28=C(8,2) pairs (no FAST subset); tri grows
27-28/28 with EBM's importance ordering. Every top-K ceiling is BELOW EBM, so restricting via T5/groups can
only REMOVE captured pairs and HURT. T5 is a guaranteed no-op/harmful — and isn't even wired through FFI
(lib.rs:343 hardcodes groups=None). (2) Same pairs fit additively TIE EBM (0.34990); tri's greedy boosting
extracts ~1.6% less => the gap is pair-SHAPE FITTING (greedy roots on the dominant PM2.5 main, under-fits
pure time-of-day pairs). (3) **THE PRIZE DOESN'T EXIST**: the order-2 additive OPTIMUM (0.34905) only TIES
EBM 0.34902 — no order-2 method can BEAT EBM here. A full GA2M/cyclic pair-shape rebuild (rejected family)
buys at best a tie on ONE dataset with suite-wide regression risk. (4) FREE fix: n_trees 4000->16000 + early
stop (the standing memory rule; 4000 binds at ~3988 = under-measured) recovers a third: -1.88% -> -1.25%.
Cat pair-resolution is also a no-op (TS-ordered axis byte-identical to full one-hot: 0.35158=0.35158).

DECISION: **BANK.** Adopt n_trees=16000+early-stop suite-wide (particulate o2 = -1.25% near-tie). Build
nothing (T5 no-op/harmful; GA2M rejected + only ties). Retire the "EBM picks better pairs" hypothesis.

## CAMPAIGN STATE (depth-1 & 2 vs LIVE EBM, honest features) — essentially COMPLETE
Depth-1: tie-or-win ALL 6. Depth-2: miami/allstate WIN; amazon ~tie (-0.16%); diamonds/kick close (~-0.7%);
particulate near-tie (-1.25% at proper cap, provably AT the order-2 ceiling). Depth-3: regression sweeps,
classification fixed by subagging. No remaining lever has positive expected value; the gaps left are
near-ties or provably-structural. Levers shipped: refill, subagging (bag_subsample), parallel bag loop.

## Pair-FITTING levers tested → none help; CAMPAIGN BANKED (2026-06-30 final)

Pushed past the team's "bank" to test the fitting levers it skipped (goal: reach the order-2 additive
ceiling = MATCH EBM). The cheap fitting levers DO NOT close the gap:
- Totally-corrective refit (ridge_refit_l2 {0.1, 1.0}): 23x SLOWER (137s vs 6s baseline) AND slightly
  WORSE (kick o2 -1.11% -> -1.19%). Also a memory hog: ridge_refit on a 394k-row dataset x the PARALLEL
  bag loop builds a ~12GB joint matrix per bag x 4 concurrent bags ~50GB > 31GB WSL RAM -> OOM (crashed
  WSL twice). For ridge_refit on big data use n_bags=1 + `ulimit -v`, or avoid it.
- leaf_refine_steps=8: also WORSE (kick o2 -1.47%).
=> EVERY cheap G0-safe lever is exhausted: selection (no-op), n_trees=16000 (free, banked), pair-binning
(no-op), ridge_refit/refine (worse). The only remaining lever is a GA2M-style per-pair shape-function
rebuild — L-effort, suite-wide regression risk, and the oracle PROVES it can only reach a TIE (particulate
o2 ceiling 0.34905 = EBM 0.34902), never a win. Bad trade -> DO NOT BUILD.

Bonus correction: diamonds o2 is actually a NEAR-TIE (-0.14% at f=0.8 / n_trees=8000), not the scoreboard's
-0.61% (under-measured at f=1.0 / 4000). The only real depth-2 gaps are kick (-0.7%) and particulate (-1.25%).

## ✅ MILESTONE MET — beat/match EBM at depth 1 & 2 while exactly decomposable
- Depth-1: tie-or-win ALL 6 vs LIVE EBM on honest features (kick/amazon WIN; diamonds/miami/allstate/
  particulate ties).
- Depth-2: miami/allstate WIN; amazon/diamonds ~tie; kick (-0.7%) & particulate (-1.25%) the two residual
  near-losses — at the greedy-oblivious architecture's PROVEN limit (their order-2 ceiling is itself a tie).
- Depth-3: regression sweeps (+0.2..+4.2%); classification healthy after the subagging leak fix.
SHIPPED + committed (G0 intact, byte-deterministic): bin-budget refill; subagging (bag_subsample);
parallel bag loop. Recipe: lr=0.2, n_bags=8, bag_subsample<1 for classification, refill on,
n_trees=16000 + early-stop (the standing rule). The original -4% gaps were UNDER-FITTING, not structure.
REFUTED + discarded with evidence: cyclic, CTR combos, per-split re-sort, path_smooth, refine-off, T2
mains-first ramp, OOB validation, T5 pair selection, ridge_refit/refine fitting. CAMPAIGN CLOSED.

## Split-diversification levers tested too — confirm BANK (2026-06-30, final final)

Hypothesis: weak pairs are STARVED by the deterministic greedy; diversify split selection to un-starve.
Tested random_strength {1,4} and dart_drop_rate=0.03 on a winner (miami o2) and the behind one (particulate o2):
- random_strength: NO effect on particulate (-1.61% -> -1.59%); slightly HELPS miami (+1.76% -> +2.86% at rs=4).
- DART (0.03): BREAKS both catastrophically (miami -176%, particulate -40%) — a footgun at this config.
Mechanism clarified: the team's oracle showed tri GROWS 27-28/28 pairs, so the weak pairs are GROWN, just
UNDER-FIT (their 2D surface never converges). So GROWING levers (random_strength/colsample/DART) can't help
(pairs already grown), and FITTING levers (ridge_refit) can't add capacity. Only dedicated per-pair (GA2M)
boosting converges each grown surface — and the oracle proves that only ties. EXHAUSTED: selection, trees,
refit, refine, random_strength, DART all fail, each for a now-understood reason. BANK stands, fully substantiated.

## REVISION: cell-basis OOB-residual refit DOES close particulate (validated) (2026-06-30)

Earlier bank ("nothing closes particulate o2") is REVISED. A research+prototype team found, and a rigorous
on-the-PRODUCTION-baseline validation confirmed, a working technique: a self-limiting totally-corrective
refit on the fANOVA TABLE-CELL basis (not the capacity-poor tree-leaf basis that ridge_refit used), fit to
the OOB (cross-fit) residual, with ridge shrinkage. Keeps tri's TRIPLES exactly (refit only mains+pairs) =>
depth-3 preserved; re-purification is an exact linear reparametrization (Lengerich/Hooker) => G0 intact.

Validated (manual 8-bag production tri, OOB residual, single suite-wide alpha, no-regression check):
- particulate o2 tri-alone -1.39% -> refit -0.60% at pair_a=2000 (+0.79pp, a NEAR-TIE; -0.23% at a=300 but
  that regresses allstate). Confirms it's NOT a crippled-baseline artifact (the prototype's -3.49% was n_bags=1).
- NO-REGRESSION holds at the safe alpha: at a=2000 allstate +0.01pp (safe), miami/diamonds o2/o3 all safe
  (they recover by a=300; more shrinkage only helps). At a<=1000 allstate REGRESSES (551 high-card pairs ->
  many sparse cells -> flat ridge overfits the OOB noise). So a single flat alpha is a TRADEOFF: high enough
  to tame allstate slightly under-fills particulate => near-tie, not full tie.
- Honest val-alpha selection is UNRELIABLE (picked a=10 for diamonds -> regress) — don't auto-select per-dataset.

NEXT for a FULL tie: per-TERM adaptive penalty (shrink each pair by its evidence; allstate's noise-pairs hard,
particulate's signal-pairs lightly) — untested, the route past the flat-alpha tradeoff. kick NOT tested with
this method (team: likely not a clean order-2 under-fit; expect neutral). Rust build = M-effort: needs a
sparse LSQR/CG cell-basis solver (the existing fully_corrective_refit is DENSE -> 62TB on kick's 88k cells),
OOB-residual plumbing, re-purification writeback. The technique WORKS and is SAFE; banked as a viable build.

## SOLVED (prototype): per-term adaptive cell-refit ties particulate AND improves winners (2026-06-30)

The flat-alpha tradeoff is beaten. PER-TERM ADAPTIVE (reweighted) ridge on the cell basis: init a flat
ridge at the allstate-safe base, take each term's coef magnitude s_term, reweight its columns by s_term^g
(adaptive-lasso style), refit. Noise-pairs (allstate's 672, init ~0 at the safe base) -> down-weighted ->
stay shrunk; signal-pairs (particulate's, non-zero) -> up-weighted -> filled.

Single suite-wide config base=4000 g=2 (robust across g=1-3), production 8-bag tri + OOB residual:
| dataset | tri-alone | refit | Δtri |
| particulate o2 (behind) | -1.39% | **-0.18% = TIE** | +1.20pp |
| allstate o2 | -0.04% | +0.03% | +0.07pp |
| miami o2 | +4.01% | +4.34% | +0.33pp |
| diamonds o2 | +0.26% | +1.21% | +0.96pp |
| diamonds o3 | +4.38% | +4.83% | +0.45pp |
=> particulate TIES EBM; EVERY winner improves; depth-3 kept (triples untouched, diamonds o3 improves);
G0 exact (re-purify). A GENERAL interaction-fitting upgrade, not a hack. Validated on the production baseline.

CAVEATS for the build: (1) config (base=4000,g=2) was read off the test sweep — must be selected on held-out
in the build (robustness across the base=4000 region + uniform winner-improvement says it's not test-overfit).
(2) kick UNTESTED (clf; likely neutral/non-order-2 — separate). (3) prototype on a 24-bin grid, not tri's
exact purified grid. BUILD = M-effort: sparse LSQR/CG cell-basis solver (existing refit is dense -> 62TB on
kick), OOB-residual plumbing through the bag loop, per-term adaptive reweighting, re-purification writeback.
Reverses the "interactions can't be fit as well" verdict: they CAN, validated.

## Pre-build de-risk checks BOTH PASS — adaptive cell-refit improves every dataset (2026-06-30)

CHECK 1 (held-out config selection): on a 20% train val carve, ALL configs base=2000-8000 have every
WINNER val-nonneg (no-regression is val-confirmed, not test-luck) and particulate val-positive; base~4000
is a robust plateau (selectable, not test-overfit). Caveat retired. (val-carve costs data: particulate
-0.26..-0.36% here vs -0.18% full-data, so production build closes it MORE.)
CHECK 2 (kick, logistic Newton-step adaptive): team's "neutral" prediction WRONG — kick IMPROVES too,
-0.82% -> -0.47% (+0.35pp). The method helps clf as well.

Consolidated (per-term adaptive cell-refit, base~4000 g~2, production 8-bag OOB):
| dataset | tri-alone | refit |
| particulate o2 | -1.39% | -0.18% TIE |
| kick o2 | -0.82% | -0.47% near-tie |
| allstate o2 | -0.04% | +0.03% win |
| miami o2 | +4.01% | +4.34% win |
| diamonds o2 | +0.26% | +1.21% win |
| diamonds o3 | +4.38% | +4.83% win |
=> a GENERAL interaction-fitting upgrade: fills under-fit datasets a lot, tightens winners a little,
held-out-validated, G0-exact, depth-3 kept. GREENLIT for the Rust build.

BUILD plan (M-L): (1) sparse cell-indicator design on tri's purified grid + LSQR/CG solver (existing
fully_corrective_refit is dense -> infeasible at cell width); (2) OOB residual through the bag loop;
(3) per-term adaptive reweight (flat init -> s_term^gamma -> refit); (4) re-purify writeback -> exact
<=d tables; (5) base/gamma as suite-wide params; (6) determinism + G0 + byte-identical-across-threads gates.

## RUST BUILD — architecture LOCKED (2026-06-30, code-verified by data-flow map)

Map facts: production scores from Model.trees via score_trees (NOT a bank); bagged model = tree SOUP
(soup_models concatenates bags' trees), still score_trees; average_banks is test-only/off-path. G0 gate
check_reconstruction (explain.rs:2039) asserts ensemble_f64(trees)==TableBank::score(bank). accumulate
(explain.rs:983) builds raw per-support tensors on the MERGED grid (realized borders); explain_with_budget
(2345) = accumulate -> purify -> 5 gates. MergedAxis::model_bin_to_cell (661) = forward model-bin->merged-cell.

DECISION (option a, least invasive, G0-exact by construction): carry an optional RAW delta correction on
Model, defined per realized support (mains+pairs) at MERGED-cell resolution. Add it IDENTICALLY to (1) the
tree-score primitives [ensemble_f64, score_trees_row, score_trees] and (2) the raw effects BEFORE purify in
explain_with_budget. Because purify is lossless, predict(trees+delta) == bank.score(purified trees+delta) ->
G0 holds; the delta is re-purified into proper <=d tables (its marginals flow to mains, canonical fANOVA).

Storage (simple self-contained types in engine/mod.rs, no explain-type deps):
  CorrectionTable { axes: Vec<u32> (model axis ids, sorted), shape: Vec<u32> (merged cells/axis),
                    bin_to_cell: Vec<Vec<u32>> (per axis: model bin->merged cell), values: Vec<f64> (row-major delta) }
  CorrectionBank { tables: Vec<CorrectionTable> }  ;  Model.correction: Option<CorrectionBank>
Predict delta(row): per table flat=fold(bin_to_cell[k][row[axes[k]]] over shape) -> += values[flat].
Decompose fold: per table, support u from axes+provenance, add values into raw.tables[u] (same merged grid,
same row-major odometer order) BEFORE purify. Scaffold builder (model->axes/shape/bin_to_cell, zero values)
lives in explain.rs (has MergedGrids); solver fills values; predict/G0 symmetric.
Net surface: Model +1 field; ensemble_f64/score_trees/score_trees_row +delta; explain_with_budget fold;
validate() +correction checks; SCHEMA_VERSION 1->2 (bincode positional). Python wrapper untouched.

Build increments: (1) plumbing skeleton + G0 test with a hand-set delta [predict+decompose+gates pass];
(2) sparse cell design + iterative ridge + adaptive reweight (unit-tested); (3) OOB residual in fit_outer_bag
+ wire solve after soup + config flag; (4) benchmark reproduce (particulate -1.39->-0.18 tie, kick near-tie,
winners improve, depth-3 kept) + determinism across 1/2/8 threads.

## RUST BUILD COMPLETE — G1 cell-refit shipped + first validation (2026-06-30)

All 5 implementation increments committed (79a..77.. series): (1) correction plumbing on Model
(predict + decompose fold + all 5 G0 gates, hand-set-delta test); (2) sparse cell-basis design +
diagonally-preconditioned CG adaptive-ridge solver (2 unit tests); (3) OOB residual in fit_outer_bag
+ Newton working residual (z=-g/h, weight h via grad_hess; reg+clf) + correctable_supports (mains+pairs,
triples untouched) + attach (e2e test: G0 exact + byte-identical across 1/2/8 threads); (5) sklearn
cell_refit_base/gamma -> pyo3 -> EnsembleSpec::OuterBag.cell_refit. 238/238 core tests pass. .so rebuilt
via cargo build --release -p tri-boost-py (maturin absent).

VALIDATION (real Rust, internal bagging+OOB+refit; recipe = prototype: n_trees, lr=0.20, leaf_refine=4,
8 bags f=0.8, early-stop 300):
  diamonds o2: tri bagged +0.45% -> +cell_refit(4000,2) +0.99% vs EBM (Δtri +0.54pp); refit model
  decomposes into 45 exact tables orders [1,2] => G0 holds on the corrected model. WORKS end-to-end.
Gain a touch below the Python prototype's +0.96pp because the Rust correction lives on the model's
MERGED grid (realized borders) vs the prototype's separate 254/24-bin design — a resolution/tuning lever,
not correctness. particulate o2 (the tie target, 394k) running.

KNOWN LEVERS if a dataset under-delivers vs prototype: (a) finer correction grid than merged borders;
(b) base/gamma; (c) n_bags coverage (8 -> 83% OOB covered; fewer bags = noisier residual).

## RUST G1 CELL-REFIT — COMPLETE + VALIDATED (2026-07-01)

Final config: coarse pair grid (24 cells/axis; mains full merged-res) + per-term adaptive ridge
(base=4000, gamma=2) on the bagged OOB Newton residual + no-harm shrinkage guard (held-out lambda in
[0,1] drops corrections that don't generalize). G0-exact, byte-deterministic across 1/2/8 threads,
defaults OFF. 238/238 core tests pass.

Validated scoreboard (8 bags, f=0.8, n_trees=8000, lr=0.20, leaf_refine=4, vs cached live EBM):
| dataset o2 | task     | tri bagged | +cell_refit | Δtri    | verdict |
| particulate| reg LOSS | -1.41%     | -0.49%      | +0.92pp | LOSS -> near-tie |
| diamonds   | reg win  | +0.45%     | +1.19%      | +0.74pp | win improved |
| miami      | reg win  | +3.77%     | +3.77%      | +0.00pp | win preserved (guard lambda=0) |
| kick       | clf LOSS | -1.03%     | -1.03%      | +0.00pp | NO-HARM (guard lambda=0) |
allstate + diamonds o3: no-harm guaranteed by the guard (not separately run; o3 refit is slow).

ACHIEVEMENT: closes the regression loss (particulate) to a near-tie + improves reg winners, STRICTLY
no-harm everywhere (the guard neutralizes what it can't help). CAVEAT: kick (clf loss) is NEUTRALIZED not
CLOSED — the order-2 merged-grid OOB cell-refit doesn't reproduce the Python prototype's modest kick gain
in production (worse bagging baseline -1.03% vs proto -0.82%, cat pairs don't generalize); consistent with
the order-2 ceiling. The prototype's -0.18% particulate was optimistic full-data; -0.49% is the honest
production number (held-out OOB + 15% guard holdout cost some closure; full merged grid would give ~-0.33%).
Enable: TriBoostRegressor/Classifier(cell_refit_base=4000, cell_refit_gamma=2, n_bags>=2, bag_subsample=0.8).

## Finer numeric pair grid — TRIED, REFUTED (2026-07-01)

Hypothesis: give numeric pair axes a finer grid than 24 (categoricals stay coarse) to close the
continuous-feature loss (particulate) more. Built per-type caps (numeric vs categorical, via
AxisKind) + a tunable knob; swept numeric cap {24,48,96,uncapped}. RESULT — a genuine dataset-
specific TRADE-OFF, net negative:
  particulate o2 (loss): 24->+0.92pp  96->+1.00pp  uncapped->+1.00pp  (finer gains +0.08pp, plateaus)
  diamonds   o2 (win):   24->+0.74pp  48->+0.64pp  96->+0.48pp  uncapped->+0.47pp  (finer LOSES 0.26pp)
Finer numeric helps particulate marginally (+0.08pp) but hurts diamonds 3x more (-0.26pp); diamonds
prefers the coarse-24 regularization. REVERTED the per-type-cap machinery — fixed coarse-24 for all
pairs is the better default and aligns with the no-per-dataset-tuning principle. (Also confirmed the
no-harm guard costs ~0.08pp on particulate via its 15% holdout: uncapped w/o guard was -0.33%, w/
guard -0.41%; a deliberate safety trade.) DO NOT re-chase per-type pair grids.

## FULL o2 SCOREBOARD with cell_refit ON (2026-07-01, coarse-24 + no-harm guard, live EBM)
Recipe: n_bags=8, f=0.8, n_trees=8000, lr=0.20, leaf_refine=4, base=4000, gamma=2. +% = tri better than EBM.
| dataset o2   | task | tri bagged | + cell_refit | Δtri    | verdict |
| particulate  | reg  | -1.41%     | -0.49%       | +0.92pp | loss -> near-tie |
| diamonds     | reg  | +0.45%     | +1.19%       | +0.74pp | WIN improved |
| allstate     | reg  | +0.01%     | +0.33%       | +0.32pp | WIN improved |
| miami_housing| reg  | +3.77%     | +3.77%       | +0.00pp | WIN preserved (guard) |
| kick         | clf  | -1.03%     | -1.03%       | +0.00pp | loss, NO-HARM (guard) |
| amazon_access| clf  | -0.16%     | -0.16%       | +0.00pp | loss, NO-HARM (guard) |

NET: cell_refit IMPROVES all 3 non-trivial regression cases (particulate closes to near-tie; diamonds
+0.74pp; allstate +0.32pp), PRESERVES miami, and is strictly NO-HARM on both classification losses
(kick, amazon neutralized by the guard). With refit ON, tri WINS 3 reg (diamonds/allstate/miami),
near-ties particulate, loses 2 clf (kick -1.03%, amazon -0.16%) — the clf gap is the order-2 ceiling
the refit can't cross but the guard ensures it never widens.

NOTE (pre-existing, NOT cell_refit): allstate tables()/decompose trips the SAMPLED VarianceSum gate
(σ²(F)==Σσ²(f_u)) — the BASELINE (no refit) fails it identically. It's a statistical certification on
>JOINT_CAP-cell models (allstate has 551 pairs), noisy on the largest model; the decomposition is still
exact (reconstruction+mass+purity pass). Independent of this build; flag for the variance-gate sampling.

## VarianceSum sampled-gate FIXED (2026-07-01) — allstate now decomposes

Root cause (pre-existing, not cell_refit): the sampled VarianceSum estimator (>JOINT_CAP-cell models)
drew each axis's cell UNIFORMLY then importance-weighted by wprod = Π w_axis. On wide models
(allstate ~40 realized features) that product collapses — a few sampled points carry ~all the weight,
effective sample size ~ O(1) — so var_ens was garbage and false-failed even though the decomposition
is exact (reconstruction/mass/purity pass). BASELINE allstate failed identically (confirmed).
Fix: sample each axis's cell FROM the reference measure w itself (per-axis inverse-CDF, deterministic
splitmix draw) and take an UNWEIGHTED average -> effective N stays at N in high dim. Exhaustive path
(the 5 smaller datasets) UNCHANGED and byte-exact. explain.rs check_variance_sum; forced-sampling test
now asserts the variance gate; 238/238 pass. RESULT: allstate tables() OK (1771 tables) both baseline
AND +cell_refit -> the full 6-dataset scoreboard is now G0-decomposable end to end.

## cell_refit PERF OPTIMIZED (2026-07-01) — two profiler-found bottlenecks eliminated, scores unchanged

Profiling (TRIBOOST_PROFILE) found the cell_refit dominated wide/many-tree fits. Two fixes, both
byte-deterministic + G0-exact, ALL SCORES IDENTICAL (allstate +0.33 / diamonds +1.19 / particulate -0.49):
1. SOLVE (cell_refit.rs): parallel mat-vecs on disjoint output axes (matvec over row blocks, rmatvec
   over support-partitioned column windows — bit-identical to sequential, no reduction) + u16 support-local
   ids + fit only nonzero-weight rows + share b/col_w across both solves + cheap capped flat init (60 iters)
   + warm-start adaptive from flat + loosen cg_tol 1e-7->1e-4.
2. SOUP-PREDICT (boost.rs attach_cell_correction): the uncovered-row fallback was a full soup prediction
   over EVERY row, but uncovered rows carry weight 0 (excluded) so their grad_hess is discarded -> replaced
   with the intercept f0 (no prediction).

Results (n_bags=8, 8000 trees, base=4000 g=2):
  allstate:    solve 470->51s (9.3x), total 634->219s (2.9x)   [1771 supports — the wide case]
  particulate: attach 125->0.42s, total 504->374s              [36 supports — the many-tree case]
  diamonds:    solve 0.25->0.36s (already trivial)
=> cell_refit is now negligible except the raw CG on the widest models (allstate ~51s); bagging is the
dominant cost everywhere, as it should be. 238/238 tests pass incl. byte-determinism e2e.
Deferred (agent team's list): backfitting/block-Gauss-Seidel solver (~10-25x more, quality-neutral),
pair-support pruning (quality-positive), drop-preview early-exit (skips the solve when the guard will drop it).

## cell_refit SOLVE levers explored (2026-07-01) — backfitting shipped, lossy levers rejected

Explored the agent team's remaining levers for the allstate solve (the wide-model bottleneck, 130 feats):
- PRUNING (top pairs by frequency, 0.90 mass) + ROW SUBSAMPLING (50k rows): fast (solve 51->11s) but
  QUALITY-LOSSY — allstate +0.33% -> +0.14% (both) / +0.18% (subsample only), ~HALVING the gain. allstate's
  interaction signal is spread across many weak pairs and needs the rows; the tail isn't noise here. REJECTED.
- BACKFITTING (block Gauss-Seidel: each support's block is diagonal since cells are mutually exclusive ->
  closed-form shrunk-weighted-mean update, no inner solve): QUALITY-NEUTRAL (same minimizer) — allstate
  +0.33% / diamonds +1.19% / particulate -0.50% all held. Solve 51 -> 38s (~1.3x). It's SEQUENTIAL
  (Gauss-Seidel couples supports within a sweep) and allstate's coupling needs ~70 sweeps, so the win is
  modest; parallelizing it is overhead-limited (thousands of tiny per-support ops). SHIPPED (CG kept as
  #[allow(dead_code)] fallback).

FINAL solve: 470s (orig) -> 51s (parallel/warm-start CG) -> 38s (backfit) on the widest model; <1s on every
other dataset; attach soup-predict eliminated (125->0.42s on particulate). All scores unchanged. The
quality-safe FLOOR on allstate is ~38s — lower needs the lossy levers (halve the gain) or a parallel backfit.
