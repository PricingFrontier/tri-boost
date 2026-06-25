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
  wall ~40s → 36.8s. Generalizes to every FullF64 depth-≥2 fit. Level 1 subtraction deferred (would compound
  drift a second generation; same machinery if the win justifies it later).

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
