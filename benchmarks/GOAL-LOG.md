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
