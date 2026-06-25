# tri-boost — Competitive Accuracy & Speed Goals

> Status: **active targets** (added 2026-06-25). These sharpen [`AIM.md`](AIM.md)'s qualitative
> "within parity / within noise" into *measurable* head-to-head goals against the incumbents.
> Some may not be achievable under the core constraints — the explicit aim is to **exhaust every
> option and honestly report where each lands**, not to claim wins we haven't earned.

## Ground rules (non-negotiable)

Every result below must hold while preserving the load-bearing invariants (see `AIM.md`).
**The binding requirement, applying to EVERY goal (G1–G5) below: the model must remain FULLY
DECOMPOSABLE** — an exact, lossless fANOVA decomposition into ≤3rd-order tables (the factored
over-budget path of §08.10 counts as decomposable), passing all five I2 gates with
`ExactnessMode::Exact`. **A result that beats a rival but is NOT fully decomposable does not count
toward any goal** — decomposability is the product; accuracy and speed are the bar it must clear
*without* giving it up. (Tracked explicitly as **G0** below.)

- Trees stay **depth-3 oblivious**; models stay `ExactnessMode::Exact` and pass the five fANOVA
  decomposition gates (incl. the new factored over-budget path, §08.10).
- "Depth N" for tri-boost means **`max_interaction_order = N`** (depth-N oblivious ⇒ order-N fANOVA).
- A goal is **settled across a benchmark suite (TabArena), not a single dataset** — single-dataset
  wins/losses below are *evidence*, not verdicts.

### Fair-comparison protocol (required before any goal is called "met")

The current evidence is **not yet a fair fight** and must be re-run under one protocol:

1. **Converged budgets, not fixed.** Oblivious trees are weaker per-tree, so tri-boost needs more
   iterations; rivals self-converge or get early-stopping. The diamonds rival numbers below were at
   `n=400` (under-fit) while tri-boost was at `n=4000` — **re-run rivals to convergence** before trusting.
2. **Matched depth** where the goal specifies it (rivals `max_depth=3`; CatBoost `depth=3`).
3. **Matched categorical handling** (ordinal vs native TS vs the rival's native cats — stated per row).
4. **Same metric, split, seed**; report mean ± noise over seeds, not a single split.
5. **Speed on a fixed thread budget** (the sandbox over-subscribes OpenMP at `n_jobs=-1`; use
   `TRIBOOST_BENCH_THREADS`).

---

## G0 — Full decomposability (the binding constraint on G1–G5)

**Target:** *every* accuracy/speed result that counts toward G1–G5 is produced by a model that is
**exactly, losslessly decomposable** into purified ≤3rd-order fANOVA tables — `f0 + Σ fᵢ + Σ fᵢⱼ +
Σ fᵢⱼₖ` reconstructs the ensemble bit-for-bit, all five I2 gates pass (`ExactnessMode::Exact`), and the
over-budget order-3 escape hatch (§08.10 factored tables) keeps this true at competitive tree counts.

- **Status: ✅ held.** The factored high-order work (Stages 1–5, branch `build/phase-6`) makes the
  decomposition succeed even when a 3-way table would exceed the dense cell budget at n≈4000 — so the
  decomposability constraint no longer caps accuracy/convergence. The diamonds depth-3 win (G1) was
  measured *with the model fully decomposable* (109 dense tables + 17 factored triples, `mode=Exact`).
- **Non-negotiable for the others:** if closing G2/G3/G4/G5 ever appears to require a non-oblivious
  growth policy, interactions beyond 3rd order, or dropping/approximating tables, that path is
  **rejected** — it would beat the rival as a *different product*, not as tri-boost. Every scoreboard
  standing below is implicitly "**@ fully decomposable**".
- **Verify:** the five I2 gates run inside `explain`/`tables` on every benchmarked model; a goal row is
  only valid if the same fitted model also emits an exact decomposition.

## G1 — Beat EBM at depths 1, 2, and 3

**Target:** tri-boost ≥ EBM accuracy at each interaction order (EBM caps at order-2, so "depth 3"
means tri's order-3 vs EBM's best).

| depth | tri-boost | EBM | standing |
|---|---|---|---|
| 1 (mains) | 0.11221 | _mains-only EBM TBD_ | ❓ likely behind (main-effect quality) |
| 2 | 0.09236 | 0.09159 | ❌ **−0.8% (loses, but close)** |
| 3 | **0.08854** | 0.09159 | ✅ **+3.3% (beats)** |

_diamonds RMSE, ordinal cats, n=4000, leaf_refine=4; EBM self-converged. Decomposable at every depth
(order-1/2: 0 factored; order-3: 17 factored triples)._

- **Assessment:** depth-3 **achieved on diamonds**; depths 1–2 not yet. tri-boost wins by being a
  *richer model class* (order-3 interactions EBM can't represent), **not** by being a better order-2 GAM.
  At equal order, EBM's heavy outer-bagging + cyclic single-feature boosting gives slightly better
  main-effect shape functions.
- **Levers to close depths 1–2:** EBM-style **main-effect refinement** — cyclic single-feature boosting
  and/or main-effect (order-1) bagging to sharpen 1-D shapes. (Note: plain `n_bags` outer bagging did
  **not** help — measured, refuted. The cyclic/per-effect form is the untested lever.)
- **Verify:** depths 1/2/3 across TabArena; measure mains-only EBM for the depth-1 row.

## G2 — Beat XGBoost & LightGBM at `max_depth=3`

**Target:** tri-boost ≥ XGBoost and LightGBM accuracy when both are capped at depth 3.

- **Evidence (mixed):**
  - diamonds (n=400 rivals — **under-fit, not yet fair**): tri 0.0885 vs LGBM 0.0975, XGB 0.1013 →
    looks like a win, but **must re-run rivals to convergence** before believing it.
  - French MTPL frequency (full 678k, converged): tri 0.5767 vs **LGBM 0.5748, XGB 0.5758** →
    tri **loses by ~0.3%**. MTPL severity: tri 1.523 **beats** XGB 1.537 and LGBM 1.562.
- **Assessment:** genuinely **hard**. A depth-3 *leaf-wise/level-wise* tree can split on up to 7
  distinct features along its paths and route asymmetrically; a depth-3 *oblivious* tree uses exactly
  3 features symmetrically — strictly less expressive even at equal nominal depth. The MTPL gap is
  partly **structural** (oblivious vs leaf-wise). Expect parity-to-slightly-behind, dataset-dependent.
- **Levers:** convergence (more trees / lower lr), `leaf_refine`, ordinal vs TS encoding per dataset,
  the unimplemented §07 interaction funnel; honestly report datasets where the order-3 oblivious cap loses.
- **Verify:** TabArena with rivals at `max_depth=3`, converged.

## G3 — Beat CatBoost at `depth=3, max_ctr_complexity=1`

**Target:** tri-boost ≥ CatBoost when CatBoost is constrained to its **most apples-to-apples** form.

- **Why this is the fairest fight:** CatBoost is *also* depth-N **oblivious** — same tree structure as
  tri-boost. `max_ctr_complexity=1` removes CatBoost's categorical-combination CTRs, leaving the
  difference to leaf-value method (CatBoost cosine vs our exact Newton), ordered boosting, and TS details.
- **Evidence:** MTPL freq tri 0.5767 vs CatBoost **0.5759** (but that's CatBoost *default* CTRs); with
  `max_ctr_complexity=1` CatBoost should be weaker → **not yet measured at the constrained setting.**
- **Assessment:** **most achievable of the rival goals** — same structure, exact Newton leaves are a
  known accuracy edge over CatBoost's default cosine score. Needs the constrained-CatBoost measurement.
- **Levers:** exact Newton leaves (have it), TS quality, optional ordered boosting (deferred §09).
- **Verify:** TabArena with CatBoost `depth=3, max_ctr_complexity=1`, converged.

## G4 — Get as close as possible to **unconstrained default CatBoost**

**Target:** minimize the accuracy gap to CatBoost run with its defaults (depth 6, full CTR
combinations, ordered boosting) — the practical ceiling.

- **Assessment:** **aspirational / bounded by design.** The depth-3 + oblivious + order-3 cap is a
  deliberate ceiling (the explainability mechanism); a depth-6 full-CTR model has strictly more capacity.
  tri-boost will be behind — the goal is a **small, reported gap**, not a win. This is exactly the
  "honest risk" `AIM.md` names: *report where the order-3 cap loses.*
- **Levers:** all accuracy levers (TS, ordered boosting, convergence) + the §07 interaction selection to
  spend the order-3 budget well; document the residual gap per dataset as the cost of decomposability.
- **Verify:** TabArena; report the gap distribution, don't chase a win.

## G5 — As fast as LightGBM at `depth=3` (training)

**Target:** tri-boost training wall-clock ≈ LightGBM `hist`, depth 3, on CPU.

- **Evidence:** MTPL freq (542k rows, 4 threads): tri **~14s** vs **LGBM ~2.8s**, XGB ~3.6s, CatBoost ~46s.
  So tri is currently **~5× slower than LightGBM** (≈ CatBoost-class, far from LGBM).
- **Assessment:** the largest gap. Oblivious growth does *more* total work (no leaf-wise pruning) but is
  highly regular/vectorizable, so parity is plausible with the planned engine work — not guaranteed.
- **Levers (the Phase-3 / `speed_accuracy_work.md` track):** QHIST quantized histograms as default,
  **histogram subtraction** (parent − smaller child; currently deferred pending integer-quantized
  subtraction), **row-parallel histogram build**, fused/parallel serial passes, SIMD leaf indexing.
- **Verify:** matched thread budget (`TRIBOOST_BENCH_THREADS`), depth 3, across dataset sizes.

---

## Scoreboard (live — update as measured)

Every row below is implicitly **"@ fully decomposable" (G0)** — a standing only counts if the same
fitted model also emits an exact ≤3rd-order fANOVA decomposition.

| # | Goal | Status | Best current evidence |
|---|---|---|---|
| **G0** | **Fully decomposable (binding on all)** | ✅ **held** | factored §08.10 path keeps `mode=Exact` at n=4000 |
| G1 | Beat EBM @ depth 3 | ✅ on diamonds | tri 0.0885 vs EBM 0.0916 |
| G1 | Beat EBM @ depth 2 | ❌ close | tri 0.0924 vs EBM 0.0916 (−0.8%) |
| G1 | Beat EBM @ depth 1 | ❓ unmeasured | tri mains 0.112; EBM mains TBD |
| G2 | Beat XGB/LGBM @ depth 3 | ⚠️ mixed / not fair yet | wins diamonds(*under-fit rivals*), loses MTPL freq ~0.3% |
| G3 | Beat CatBoost @ depth 3, ctr 1 | ❓ unmeasured (constrained) | ~tie vs default-CTR CatBoost on MTPL |
| G4 | Approach default CatBoost | 🎯 aspirational | gap TBD; bounded by order-3 cap |
| G5 | LightGBM-speed @ depth 3 | ❌ ~5× slower | tri 14s vs LGBM 2.8s (MTPL 542k) |

**Honest summary:** the one *clean* win today is **G1@depth-3** (beat EBM by using order-3 while staying
exactly decomposable). Everything else is either close-but-behind (G1@2, G2), unmeasured under the fair
protocol (G2 converged, G3 constrained, G1@1), bounded by the design cap (G4), or a known engineering gap
(G5). Next step for any of these: stand up the fair-comparison harness (converged, matched depth/encoding)
across TabArena rather than trusting single-dataset numbers.
