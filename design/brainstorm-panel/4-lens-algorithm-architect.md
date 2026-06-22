# tri-boost — The Engine: Core Algorithm & Methodology (Architect's Brainstorm)

## 1. Thesis / the bet

**The depth-3 oblivious constraint is not a tax on a generic GBM — it is the data structure.** My bet: if we design the engine so the *trained object* and the *explained object* are literally the same thing — a set of small tensors on one shared global grid — then we get exactness, speed, and explainability from a single mechanism rather than three bolted-on subsystems. Concretely: the per-level summed-gain split-finder produces, at zero marginal cost, a per-tree `(feat[3], thr[3], leaf[8])` triple; every tree is already a rank-1 tensor on the global grid; accumulation is summation; explanation is one linear (purification) operator; inference is a sum of LUT reads whose cost is independent of tree count. **There is no "model" and "explanation" — there is one tensor bank in two views (raw for inference-equivalence checks, purified for human/rating use), and a proof that they're equal.** Everything else — losses, constraints, sampling, calibration — composes *because* it never touches tree shape. The hard part, and where most of the accuracy lives, is **interaction selection**: which ≤3-feature supports the booster is even allowed to spend trees on. That is the heart of both accuracy and table-set size, and I design the engine around making that choice well and cheaply.

## 2. Components

### A. The split-finder: oblivious level-wise *summed-gain* search [BOTH]
The core loop. Pre-bin every feature once into `u8` (`max_bin` 254 + 1 reserved missing bin). At each of the 3 levels, evaluate every `(feature, bin-border)` candidate by **applying it to all `2^level` current leaves simultaneously** and summing the Newton split gain `½ Σ_leaves[ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ]` over those leaves; keep the single argmax `(feature, border)` for the whole level (research/01 §1, README v1). 
- **Mechanism:** one shared split per level ⇒ ≤3 features/tree ⇒ exact 3rd-order fANOVA (I1, I2 by construction). Newton gain, *not* CatBoost's Cosine default — it gives XGBoost/LightGBM accuracy parity and exact leaf weights `w* = −G/(H+λ)` for free.
- **Tradeoff:** the per-tree learner is *weaker* than depth-6 (research/01 §3) — we pay in tree count, not final accuracy. Accept it; compensate downstream (momentum/fully-corrective, below).

### B. Histogram engine: per-level tensors + subtraction + quantized integer g/h [PREDICTIVENESS for speed; BOTH for reproducibility]
Per level we build one tiny gradient/hessian histogram tensor (`≤8 leaves × features × bins`). Use the **subtraction trick** (one child = parent − sibling) to halve work. Accumulate into **quantized int16/int32 histograms via stochastic rounding**, then refit leaf weights from full-precision g/h (research/06 §1, v1.5).
- **Mechanism:** integer sums are *associative* ⇒ order-independent ⇒ bit-reproducible across thread counts — which directly serves the "tables == ensemble to floating point" invariant *and* the regulated-pricing audit requirement. Up to ~2× training speed at ≤3 bits.
- **Tradeoff:** very low bins (2) can degrade complex models; refit-leaf must be ON for our Poisson/Gamma/Tweedie path (research/06 §1). Threading: **per-thread private, cache-line-padded histograms, `fold`/`reduce` in fixed order, no Mutex** — the ≤8-leaf tensor makes per-thread buffers L2-resident (README v1 systems).

### C. The shared global binning grid [BOTH — this is the keystone]
One global border set per feature, computed once on a seeded ~200k subsample, **persisted in the model**, reused identically at validation/predict/score time. All trees split only on these borders; the *union of realized borders per feature* is each table's axis (research/03 §4.2).
- **Mechanism:** because all trees share axes, every tree's tensor is *exactly* representable on the common grid → accumulation is lossless → purify-then-sum ≡ sum-then-purify (Lengerich linearity, research/03 §2.5) → tables align bit-for-bit. Without one global grid none of the exactness machinery works.
- **Tradeoff:** `max_bin` is a per-feature readability↔resolution knob; coarser = more stable, auditable relativities but risks over-shallow trees under monotone constraints. Add midpoint borders for low-cardinality ordinals so table rows align 1:1 with real feature values (README v1).

### D. The dual representation + incremental/streaming purification [EXPLAINABILITY]
Two synchronized views: **(1) the complete raw tensor bank** keyed by realized support `u` (`|u|≤3`), summed straight from trees — used for the inference-equivalence test and LUT-sum scoring; **(2) the purified bank** `f₀,{fᵢ},{f_ij},{f_ijk}` for humans and rating export. Purification = the mass-moving cascade (3-way→2-way→1-way→intercept). By linearity it's a **streaming reduction**: maintain a running purified bank, fold in each new tree's purified tensor (or re-purify the raw delta) incrementally.
- **Mechanism:** purification conserves total mass per cell (subtract `m0` here, add `m0` one order down) ⇒ `Σ_u f_u(x) ≡ Σ_u T_u^raw(x) ≡ F(x)`, exactly, always (research/03 §5).
- **Tradeoff:** purified tables are *not* a fixed linear function of raw leaves under the reference measure, so after `refit`/warm-start you must **re-purify** (research/06 §1). Cheap; just must not be skipped.

### E. Interaction selection — the accuracy heart [BOTH]
This decides both accuracy and table-set size, so it gets the most design attention. **Three composable levers, used as *soft admission* not hard gates** (research/06 §7):
1. **Heredity/strong-hierarchy at train time** — admit a triple only if its sub-pairs are already admitted; admit a pair only if both mains are. This avoids the cubic `C(n,3)` scan by *composition*, not enumeration.
2. **FAST-style RSS pre-filter on the residual** using the same binned cumulative histograms the engine already has (O(b²)/pair) — a cheap prior over pairs feeding the interaction-constraint allow-list.
3. **Exact post-hoc Sobol/Shapley-effect variance** `S_u = σ²(f_u)/σ²(F)` from the purified tables (zero model calls) as the *final* arbiter — prune tables explaining negligible variance.
- **Mechanism:** keeps the realized support — and thus exported table count — to a few hundred, not `C(n,3)`, while letting the booster find genuine interactions.
- **Tradeoff (critical):** FAST's RSS objective ≠ the booster's Newton summed-gain, and naive two-stage (mains-to-convergence then interactions-on-residual, EBM/GAMI-Tree style) **mis-converges under correlation** (research/06 §7). **So: joint boosting over the realized supports, single purification pass at the end — never EBM cyclic staging.** FAST/heredity are *soft priors widening/narrowing the constraint set*, with final selection by exact purified variance.

### F. The Loss trait + objectives [BOTH]
The trait from research/04: `grad_hess` (together, one pass, w.r.t. raw score F), mandatory `init_score` = link(weighted mean), `link ∈ {Identity, Log, Logit}`, `default_metric`, optional `max_delta_step`. Ship SquaredError, Logistic, Poisson, Gamma, Tweedie(ρ) in v1; compute log-link powers as `exp(k·F)` not `powf`.
- **Mechanism:** objective is fully orthogonal to tree shape (I1/I2 untouched) — it only changes the g/h fed to the histogram. Per-row **offset** (log exposure) added to F before the inverse link; `init_score` anchors the base rate / `1.000` reference.
- **Tradeoff:** none structural. Poisson hessian instability handled by `max_delta_step` (Newton-step cap), hessian floor ε for Logistic/Tweedie corners.

### G. LUT-sum inference + branch-free SIMD [PREDICTIVENESS]
Two scoring paths, both exact. **Per-tree SIMD:** 3 packed compares → OR-shift into a 3-bit index → **in-register permute over the 8-float LUT** (not a hardware gather; 32 bytes, L1-resident). **LUT-sum:** predict directly from the *complete* tensor bank — bin once, sum `O(#nonempty tables)` reads, **cost independent of tree count** (research/06 §1).
- **Mechanism:** prediction stays a sum of additive ≤3-feature terms, so it equals the ensemble bit-for-bit (fp32 leaves mandatory).
- **Tradeoff / footgun:** LUT-sum is lossless **only over the complete realized support**, never the pruned display subset — keep both views separate. fp16 leaves break exactness ⇒ opt-in approximate mode only. Multiplicative relativities need **sum-then-exp** per table (presentation, not a break).

### H. Reproducibility & data structures [EXPLAINABILITY/trust]
Canonical structure = **pre-binned columnar `u8` matrix** (the CPU half of QuantileDMatrix, no dependency). Tree = `{ feat: [u16;≤3], border: [u16;≤3], leaf: [f32;8], used_features:u8 }`, stored contiguous-by-round so `iteration_range` slicing is O(1). Seed everything (subsample RNG, stochastic-rounding RNG, tie-breaking, row order). Honest guarantee: **"reproducible for a fixed build+environment,"** not "bit-reproducible for regulators" (FMA/libm/fast-math break cross-build) — disable FMA-contraction and fast-math on the audited path.

## 3. The 3–5 highest-leverage ideas from my lens

1. **One grid, one tensor bank, two views — designed in from line one.** The single global border set is the linchpin that makes accumulation lossless, purification linear/streamable, and LUT-sum possible. Every "explanation" (SHAP, PDP, ICE, H-stat, Sobol) then becomes a closed-form table read (research/06 §7, Bordt–von Luxburg equal-split) — we *never* run TreeSHAP/PDP traversal; we keep stock SHAP only as a test oracle.

2. **Joint boosting over admitted supports + single final purification — explicitly reject EBM cyclic staging.** This is the accuracy bet. Cyclic round-robin and two-stage mains-then-interactions are the published interpretable-boosting recipes, and they *lose* under correlation. We boost jointly over the heredity-admitted ≤3-feature supports and purify once. This is the methodology designed *around* the structure, not bolted on.

3. **Quantized-integer histograms as the reproducibility mechanism, not just a speed trick.** Associative integer sums give bit-stable tables across threads — turning the "tables == ensemble" invariant from a hope into an arithmetic guarantee, while delivering ~2× training speed. The two goals are the *same* lever.

4. **Interaction selection as a soft-prior funnel (heredity → FAST → exact Sobol), bounded by composition.** Never scan `C(n,3)`; grow the support set by hierarchy, screen cheaply on the histograms we already have, and let exact purified variance be the final judge. This simultaneously controls accuracy (no spurious tables stealing trees) and table-set size (audit burden).

5. **Make losslessness a CI invariant, not a claim.** Per-cell reconstruction over grid-corner samples (exhaustive since piecewise-constant), mass-conservation, per-slice zero-mean, variance-sum identity, and TreeSHAP-vs-table agreement — all as unit tests gating every build (research/03 §5).

## 4. What I will NOT do, and why

- **No non-symmetric growth (Depthwise/Lossguide), no `gblinear`.** They break I1/I2 outright (README boundary). The constraint *is* the product.
- **No linear/piecewise-linear or soft/neural leaves (GBDT-PL, NODE, NODE-GAM, GRANDE, TEL, GrowNet).** They buy accuracy precisely by destroying constant-cell tables or obliviousness (research/06 §2). Linear leaves are at most a *flagged* "rating-function" research mode, never the default audited path.
- **No third-party inference compilers (Treelite/lleaves/QuickScorer/HummingBird).** They optimize irregular traversal we've already eliminated — *moot, not adopted*. Our LUT-sum kernel is leaner; their MTPL2 numbers are bars to beat, not dependencies (research/06 §1).
- **No EBM-style cyclic/round-robin boosting, no greedy mains-then-interactions staging.** Different machine; mis-converges under correlation (idea #2).
- **No post-training leaf/threshold value-quantization, no fp16 leaves, no post-hoc table editing (EBM `monotonize()`, Whittaker-Henderson on the audited artifact).** All break bit-for-bit exactness; allowed only as separately-named lossy export modes off the audited path (research/06 §1, §7).
- **No CatBoost Cosine split score, no ordered boosting in v1, no GPU in v1.** Cosine forfeits exact leaf weights; ordered boosting is ~1.7× cost for small-data gains and is separable; GPU is deferred behind a `Backend` trait with the layout already GPU-friendly (research/01 §4–5).
- **No mutable score-time caches or `pred_leaf` stacking.** Both destroy determinism or the "tables ARE the rating table" thesis (research/06 §1).

The discipline throughout: *if a technique touches tree shape, leaf form, or feature coupling, it's a boundary question, not an engineering one — and the default answer is no.*
