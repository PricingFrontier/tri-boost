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

### Fair-comparison protocol — IMPLEMENTED in `benchmarks/fair_compare.py` (2026-06-25)

The protocol below is now a harness, not a wish. `fair_compare.py` runs it with a **frozen rival
baseline** (`benchmarks/.fair_cache.json`): rivals are fit ONCE and cached; tri-boost is NEVER cached
(its Rust core changes between iterations) and always re-fits + re-runs the live G0 decomposability
check. So iterating a tri-boost lever re-fits only the 3 tri rows and reads every rival from cache.

1. **Converged budgets, each model at ITS OWN convergence.** Rivals (XGBoost/LightGBM/CatBoost) get
   held-out early stopping (they converge fast and overfit at a fixed 4000); EBM self-converges;
   tri-boost gets early stopping too BUT with far more patience (`rounds=500` vs rivals' `50`) —
   oblivious trees improve in tiny increments, so an impatient stop under-fits badly (measured:
   `rounds=50` cost miami **−6.7%**, stopping at ~40 trees; `rounds=500` recovers it and still catches
   genuine overfit like kick **+1.6%**).
2. **Matched depth** (rivals `max_depth=3`; CatBoost `depth=3`).
3. **Matched native categoricals** — XGBoost/LightGBM/CatBoost all use native cats; tri uses
   ordinal-where-a-natural-order-is-known (diamonds) else native TS; EBM native.
4. **Same metric, split, seed**; the split is fixed (model-seed sweep is the cheap default; split-seed
   sweep a documented extension).
5. **Speed on a fixed thread budget** (`FAIR_THREADS`, default 4; the sandbox over-subscribes OpenMP
   at `n_jobs=-1`). EBM gets more threads (`FAIR_EBM_THREADS`) — its metric is thread-invariant.

**Measured fair results (2026-06-25, converged, decomposable, single split/seed)** — RMSE-log↓ for
regression, ROC-AUC↑ for classification; tri = order-3, early-stopped (rounds=500):

| dataset | tri o3 | EBM o2 | xgb d3 | lgbm d3 | cat d3-ctr1 | cat default | G1 | G2 | G3 |
|---|---|---|---|---|---|---|---|---|---|
| miami_housing | **0.13203** | 0.13853 | 0.13908 | 0.14039 | 0.13581 | 0.12931 | ✅+4.7% | ✅+5–6% | ✅+2.8% |
| particulate | **0.33824** | _(skipped)_ | 0.34452 | 0.34264 | 0.34797 | 0.32207 | n/a | ✅+1.3–1.8% | ✅+2.8% |
| allstate | **0.54009** | 0.54160 | 0.54695 | 0.54450 | 0.54096 | 0.53614 | ✅+0.3% | ✅+0.8–1.3% | ✅+0.2% |
| diamonds | **0.08896** | 0.09159 | 0.08805 | 0.08725 | 0.09035 | 0.08617 | ✅+2.9% | ❌−1–2% | ✅+1.5% |
| kick | **0.77614** | 0.78493 | 0.76868 | 0.77329 | 0.77510 | 0.77872 | ❌−1.1% | ✅+0.4–1.0% | ✅+0.1% |
| amazon_access | **0.84506** | _(skipped)_ | 0.85950 | 0.85727 | 0.87304 | 0.91081 | n/a | ❌ | ❌ |

EBM intentionally not run on the two giant datasets (particulate 315k / amazon high-card) — too slow to
converge in tractable time; compared on the four where it ran. **Net standing across the suite (all @
fully decomposable):** G1 (beat EBM) **3/4**, G2 (beat xgb+lgbm @ d3) **4/6**, G3 (beat constrained
CatBoost) **5/6**. tri wins where order-3 interactions pay off (miami/particulate/allstate — spatial &
insurance) and loses on amazon (all-categorical → TS-encoding weakness) and diamonds-G2 (fine-continuous
→ leaf-wise depth-3 is more expressive). Reproduce: `python benchmarks/fair_compare.py`.

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

Updated **2026-06-25** with the fair-harness measurements above (converged, matched, decomposable; single
split/seed; full per-dataset table in the protocol section).

| # | Goal | Status | Cross-suite standing (fair, @ decomposable) |
|---|---|---|---|
| **G0** | **Fully decomposable (binding on all)** | ✅ **held** | every benchmarked tri model emits exact ≤3rd-order tables (`mode=Exact`); factored §08.10 path holds at n=4000 |
| G1 | Beat EBM @ order-3 | ✅ **3/4** | WIN miami +4.7%, diamonds +2.9%, allstate +0.3%; lose kick −1.1% (EBM not run on particulate/amazon) |
| G1 | Beat EBM @ order-2 | ❌ mostly behind | loses diamonds −1.1%, miami, kick, allstate (−0.2 to −1.1%) — EBM's order-2 GAM shape is sharper |
| G1 | Beat EBM @ order-1 (mains) | ❌ behind | loses everywhere (−0.2 to −4.4%) — main-effect shape quality is EBM's strength |
| G2 | Beat XGB/LGBM @ depth 3 | ✅ **4/6** | WIN miami/particulate/allstate/kick; lose diamonds (−1–2%) + amazon |
| G3 | Beat CatBoost @ depth 3, ctr 1 | ✅ **5/6** | WIN all but amazon — the fairest fight (CatBoost is also oblivious), exact-Newton leaves pay off |
| G4 | Approach default CatBoost | 🎯 gap reported | +0.7% (allstate) to +7.2% (amazon); ceiling, not chased |
| G5 | LightGBM-speed @ depth 3 | ❌ ~15× slower | tri ~12–210s vs lgbm <1–23s; engineering gap (Phase-3 track) |

**Honest summary (fair, converged):** tri-boost — **while staying exactly decomposable** — now wins
**G3 on 5/6**, **G2 on 4/6**, and **G1 on 3/4 datasets where EBM ran**. It wins where order-3 interactions
matter (miami spatial, particulate, allstate insurance) and loses where they don't help it: **amazon**
(all-categorical → TS encoding trails native cat handling) and **diamonds-G2** (fine-continuous → leaf-wise
depth-3 is strictly more expressive than depth-3 oblivious). G1@order-1/2 confirm tri wins by being a
*richer model class* (order-3), not a better low-order GAM — EBM's heavy-bagged main-effect shapes still
edge it at equal order. Remaining frontiers: main-effect refinement (G1@1/@2), categorical encoding for
all-cat data (amazon/G2), and training speed (G5). Iterate with `python benchmarks/fair_compare.py`
(`FAIR_G0=off` for fast accuracy-only loops); rivals are frozen, only tri re-fits.
