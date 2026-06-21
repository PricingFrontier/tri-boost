# Explainability ↔ Predictiveness Tension Audit

The five blueprints agree on the spine (oblivious Newton core, purification, balance-in-`f₀`, calibration-outside-the-tables) and converge — independently — on the same firewall discipline. The genuine disagreements are about *where on the curve to sit*, not whether the curve exists. Below are the real conflicts, a decisive resolution for each (insurance-rating-first), and the spots where a blueprint silently bends I1/I2.

## Tension 1 — Ordered Target Statistics: categorical accuracy vs "young drivers → 1.4×" readability

The sharpest split in the set. BP1 (predictiveness) calls ordered TS "the biggest single lever" and pays the cost. BP5 (contrarian) bans it from the default path because the axis becomes a *learned encoding*, not the raw level. Per the inventory, ordered TS is `changes_table_form`, not `breaks_exactness` — so this is a product decision, not an invariant violation.

**Resolution — sit center-left, axis-preserving by default.** The readability cost is not intrinsic to leakage-free encoding; it is intrinsic to *consuming* the TS as a continuous axis. So: compute the cross-fitted/ordered target statistic to get a leakage-free *ordering*, then apply BP1's **Fisher sorted-by-encoded-mean ordinal split** (v1.5, `preserves_exactness`) so each category stays a distinct, human-readable row whose order is data-driven. This recovers ~all the categorical accuracy while the table still reads "postcode group G → relativity." Mandatory empirical-Bayes auto-shrinkage (v1 `native`) so singletons collapse to base. BP5's blanket ban over-pays; BP1's raw continuous-TS axis under-pays on readability. The Fisher carve-out dominates both. One-hot only for genuinely low-cardinality factors.

## Tension 2 — The 3-way interaction budget: accuracy vs table-set bloat

A bigger triple budget fits more real structure (predictiveness) but every realized triple is an exported 3-D rating cell an actuary must sign (explainability), and `C(n,3)` is the bloat bomb (n=100 → 161,700 triples). All five blueprints converge on the *same* funnel — heredity admission → FAST soft prior → exact post-purification Sobol as final judge — which is correct and should be adopted verbatim.

**Resolution — decouple the two artifacts; this dissolves most of the tension.** Adopt BP2/BP4's hard split: **complete realized support for inference** (losslessness is non-negotiable) vs **pruned top-`k`-by-Sobol for display/filing**, with honest variance-budget labelling ("12 of 47 tables, 99.3% of variance"). Then the accuracy decision (admit the triple) and the audit-burden decision (show the triple) are independent — you keep the fit and hide the noise. On the screening itself: never let FAST *hard-gate* (its RSS objective ≠ Newton gain), and reject EBM-style mains-then-interactions staging (GAMI-Tree mis-convergence under correlation); **joint boost over admitted supports, single final purification.** Default `max_interaction_order=3`, but expose the `{1,2,3}` dial as BP3/BP5's *filing strategy*.

## Tension 3 — The purification reference measure `w`: it changes both the numbers and the "right" tables

This is the deepest tension because `w` simultaneously sets what "pure" *means* and how much variance each table appears to own (which drives Sobol pruning, i.e. *which* tables survive). Product-`w` evaluates effects at correlated combinations that never occur (Hooker extrapolation — territory×vehicle×age are correlated); joint/Hooker-`w` is faithful but breaks `σ²(F)=Σσ²(f_u)` and exact equal-split SHAP.

**Resolution — `w` is a versioned, signed, post-hoc-recomputable input; default Laplace-smoothed empirical product; stamp it on every table.** BP2 and BP5 are right that this is the real research risk, not the algorithm. Crucially, recomputing tables under a different `w` *without retraining* is `preserves_exactness` (leaves stay piecewise-constant, sum conserved) — **BP2 correctly flags this is mis-scored in the inventory and should be corrected.** Sit at product-`w` for v1 (auditable, sums to 1, SHAP exact, fast), build the joint path early, and benchmark how far relativities move between measures — that movement is the credibility metric to watch under regulator questioning. Under joint-`w`, switch importances to Shapley-effects (still sum to 1) and label SHAP "interventional," never just "SHAP."

## Tension 4 — Linear/piecewise-linear leaves: smooth-target accuracy vs "the cell IS the relativity"

BP1, BP4, BP5 all reach the same verdict and they are right. Linear leaves are `changes_table_form` (constant cells → piecewise-linear shape functions) and, worse, monotonicity is unsolved *within* a cell — a "monotone" model can be non-monotone inside a cell, which is unfileable.

**Resolution — sit hard at the constant-cell end. Off the default path entirely**, available only as a loudly-labelled `Approximate` "rating-function" research mode for non-regulated use. The whole pricing pitch is "the cell is the relativity"; do not erode it for smooth-extrapolation gains that credibility floors + bagging + finer borders mostly recover.

## Tension 5 — Local calibration & distributional heads: lift/coverage vs additivity

Autocalibration, multicalibration, Beta/Venn-ABERS all warp the aggregate score: `g(Σf_u) ≠ Σg(f_u)` — `breaks_exactness`. Raw multi-quantile heads are `changes_table_form` (per-quantile table sets that can cross).

**Resolution — unanimous and correct: only affine/intercept adjustments fold in.** Global mean re-anchoring into `f₀` is default-on (`preserves_exactness`, fixes the GBM balance violation that makes a model unfileable). Everything richer lives *outside* the decomposition as a declared score→price map, off by default. For intervals, prefer **point tariff tables + split-conformal/CQR wrapper** (`preserves_exactness`) over distributional heads on the audited path.

## Silent invariant violations to flag

- **BP1 §C — boost on a GLM base-margin (CANN).** Inventory scores raw `base_margin`/warm-start as `breaks_exactness`. It is only safe **if the GLM backbone is itself ≤3-order additive**; a filed 4-way GLM interaction in the offset silently violates I2 (the full audit trail `η_GLM + tables` no longer decomposes to ≤3rd order). BP1 names this caveat — but it must be a hard precondition, not a footnote.
- **BP1 §F — quantized-integer histograms** are listed in the inventory as `fANOVA: none` at v2, yet BP1/BP4 promote them to v1.5 as the *reproducibility mechanism*. That re-prioritization is defensible (associative integer sums → bit-stable tables) but it is a promotion beyond the catalogue verdict and should be flagged as such, with refit-leaves-from-full-precision mandatory on the Poisson/Gamma/Tweedie path.
- No blueprint violates **I1** (all correctly reject depth>3, leaf-wise/lossguide, gblinear, soft/neural "oblivious" false-friends). The only live I2 risks are the base-margin backbone (above) and any drift of ordered-TS-as-continuous-axis or linear leaves onto the default path — which BP5's typed `Exact`/`Approximate` firewall is the right structural defense against. **Adopt the firewall.**

**Net positioning:** center-left on categoricals (Fisher-sorted), full triple budget with display pruning, product-`w` now / joint-`w` benchmarked, constant cells always, calibration always outside the tables.
