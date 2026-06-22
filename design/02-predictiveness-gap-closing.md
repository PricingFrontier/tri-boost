# tri-boost — Predictiveness & Closing the Gap to Unconstrained GBMs

> 2026-06-21. Companion to [`01-brainstorm.md`](01-brainstorm.md). Produced by a gap-closing analysis team (HP-ensemble deep-dive + ranked gap-closer synthesis + closability reality-check + 2023–26 frontier scan), then synthesized. Answers two questions: **(1)** is a hyperparameter-diverse ensemble of tri-boost models worth it, and how; **(2)** what components close the predictiveness gap to unconstrained XGBoost/LightGBM/CatBoost *while staying exactly decomposable into ≤3rd-order tables*.
>
> *Note: the formal adversarial-verify pass was rate-limited mid-run; the synthesizer performed the decomposition-safety checks inline instead (correctly forbidding >3-feature CatBoost combination CTRs and excluding linear/soft/neural-leaf approaches). Supporting deep-dives are in [`gap-closing-analysis/`](gap-closing-analysis/).*

---

# tri-boost: Closing the Predictiveness Gap While Staying Exactly Decomposable

## 1. The hyperparameter-diverse ensemble idea — verdict & recipe

**Verdict: worth doing, opt-in, as greedy ensemble selection — not a blind soup.** It is exactly decomposable, delivers a real but modest variance-driven lift, costs K× training, and **cannot touch the order-3 bias ceiling.** Ship it as an opt-in recipe, not a default.

**Why it stays exact (airtight).** Purification is a linear operator with `Σαᵢ=1` (Lengerich Cor. 2.2): `purify(ΣαᵢFᵢ) = Σαᵢ purify(Fᵢ)`, so "purify-then-average ≡ average-then-purify." Every member is order ≤3 by construction (I1), so any convex combination is a sum of order-≤3 tensors — order ≤3 (I2) holds automatically. This is exactly why a self-ensemble is admissible where stacking with a foreign/nonlinear model is not.

**Exact mechanics (the only real engineering):**
- **Common merged grid.** HP diversity deliberately puts members on different border grids. Tables are summable only on the **sorted union of all realized thresholds per feature.** Each member's piecewise-constant tensor maps *losslessly* onto the finer union grid (a cell constant over [30,55) replicates across sub-cells). **Zero approximation.**
- **Union of supports.** The averaged bank holds the *union* of triples (a member that never learned {i,j,k} contributes 0). So the pre-prune bank is denser than any single member. Mitigate with the existing machinery: re-purify the averaged bank once, prune by Sobol `σ²(f_u)/σ²(F)` for display; the complete union support stays for lossless inference.
- **Average in score space, not response space.** Members must be combined in raw pre-link `F` (where the model is additive). Averaging `exp(F)`/`σ(F)` is nonlinear and breaks additivity. Apply the link once at the end.
- **Weight constraints:** enforce `Σαᵢ=1`, `αᵢ≥0` (intercept folds cleanly into `f₀`); a learned nonlinear meta-weighting is banned (reintroduces order inflation).
- **Single shared reference measure `w`.** Do *not* diversify `w` across members — the variance-sum identity branches on `w`, so mixing is incoherent. Pick one `w` (Laplace-product default), average, optionally re-export under alternate `w` post-hoc.

**Bias vs variance — state it bluntly.** The order-3 cap is a *bias* limit; ensembling reduces *variance*. They do not touch. The average of order-≤3 members is order-≤3, and its error to a true order-4+ target is bounded below by the best order-≤3 approximation, which no averaging crosses. What it *does* harvest is the instability of a single greedy depth-3 oblivious fit (which grid, which near-tied split won, which seed/subsample) — and depth-3 oblivious trees are deliberately weak/high-variance, so there is more variance to harvest than in a deeper model.

**The right way: greedy ensemble selection (Caruana 2004), not select-best, not average-all.**
- **Greedy forward selection with replacement** on a held-out set, optimizing **held-out deviance/logloss** (strictly proper for Poisson/Gamma/Tweedie — never RMSE for these). With-replacement multiplicities produce a *weighted* convex soup, which dominates uniform averaging (independently confirmed by Wortsman's greedy-soup > uniform-soup).
- **Anti-overfit (non-negotiable at large K):** (1) initialize the ensemble with the top-2 single models (not empty), and (2) **bag the selection** — run greedy selection on bootstrap replicates of the held-out set and average the weight vectors. Without this the validation-metric gain evaporates out-of-sample.

**Where diversity comes from (ranked by payoff/$):** (1) **hyperparameters** — `max_bin`/border type, `λ`, sub/colsample, `max_interaction_order∈{2,3}`, LR×n_trees operating point, `random_strength` — highest decorrelation (Wenzel: HP diversity beats seed-only at equal size); (2) **seeds + row/col subsamples** — cheap, more correlated; (3) feature subspaces as one HP axis. **K≈8–16** is the sweet spot; the selected ensemble is usually 3–8 effective members.

**Cost/payoff.** Training is K×; purification/union/selection are cheap and parallel. **Inference is free** — the re-purified bank is one table set, LUT-sum scoring independent of K. Expected lift: **~0.5–1.5% deviance/logloss** over the single best-tuned model, larger where the single fit is unstable (small n, high-cardinality TS noise), ~0 where the gap is hard order-≥4 bias.

**Recommended default:** single best-tuned model is default; ensembling **off.** On-ramp (ship first): **outer-bag table averaging at one HP setting** (~8 bags) — most of the variance win plus free per-cell error bars, no HP search. Full recipe (`ensemble=True`, K≈12): HP-diverse library → union grid → bagged greedy selection on held-out deviance seeded from top-2 → re-purify → Sobol prune. **Honesty:** the legitimate "beat EBM" comparison is bagged-vs-bagged (EBM already bags 14× internally).

## 2. The gap-closing playbook — ranked

### Tier 0 — Highest leverage (do these or you trail by default)

1. **[interaction-budget] Full 3rd-order budget + exact 3D triple tables.** The headline thesis lever: extend GA2M/EBM from order-2 to exact order-3, decoupling inference (complete realized support) from filing (top-k Sobol). *Preserves. Cost low. CORE.* Dataset-dependent lift — meaningful on interaction-rich data, zero where data is ≤2-order.
2. **[per-tree-strength] Newton summed-gain split selection + exact leaf weights.** XGBoost/LightGBM/CatBoost-NewtonL2 parity: score each (feature, border) by per-level summed Newton gain across all 2^level leaves, fill with `w*=-G/(H+λ)`. *Preserves. Cost low. CORE — the v1 spine everything sits on.* Default it; keep CatBoost's Cosine as a research A/B only.
3. **[interaction-budget] Heredity + FAST + Sobol admission funnel (soft prior, never a hard gate).** Build triples by composition not C(n,3); FAST RSS pre-filter on the residual (reuse existing binned histograms); exact post-purification Sobol as final arbiter/display-pruner. *Preserves. Cost medium. CORE — "the accuracy heart."* CRITICAL: FAST's RSS ≠ Newton gain, so it must never hard-gate (would amputate a real interaction); use heredity for triples, not an O(b³) raw scan.
4. **[interaction-budget] Joint boosting over admitted supports (reject mains-then-interactions staging).** One joint ensemble + one final purification, not EBM-style two-stage. Avoids the documented GAMI-Tree mis-convergence under correlated features. *Preserves. Cost low. CORE — architecture decision, not a flag.*
5. **[representation] Ordered/Fisher-sorted target statistics + empirical-Bayes shrinkage.** The single biggest categorical lever — CatBoost's documented largest edge (Amazon +17%), attributable to ordered TS, *not* oblivious trees. Auto-shrinkage `(n_c·mean_c + m·global)/(n_c+m)`. *Preserves — but only via the Fisher **sorted-ordinal** split (category stays a distinct row); a continuous TS axis changes table form. Cost medium. CORE.*

### Tier 1 — Strong, reliable

6. **[representation] Loss trait + log-link objectives (Poisson/Gamma/Tweedie) with exposure offset.** Correct deviance-proper loss is a large correctness lever on skewed/count targets and the strictly-proper early-stop metric. *Preserves (just a new (g,h)). Cost low. CORE.*
7. **[per-tree-strength] `boost_from_average` → the fANOVA intercept `f₀`.** `F₀=link(weighted mean)` in score space; one Newton step for Gamma/Tweedie. IS the `f_∅` term; the deployable base rate. *Preserves. Cost low. CORE — convergence + calibration win, NOT a gap-closer.* Keep F₀ a separate scalar (never a "tree 0").
8. **[regularization] Leaf-value reg: L2 (λ), exact min-data veto, path_smooth.** `λ~1–3` workhorse; path_smooth shrinks thin cells toward ancestors. *Preserves (all value-level). Cost low. CORE — incremental but compounding.*
9. **[representation] `max_interaction_order∈{1,2,3}` structure dial.** Whole-tree constraint (stricter than XGBoost's union rule). **Adversarial verdict: adopt, but NOT as a predictiveness lever** — order-3 is the identity setting; order-1/2 are pure restrictions that can only lose accuracy. Real value: trust/EBM-domination/additive-cost benchmarking. *Preserves (strengthens). Cost medium.*
10. **[representation] Multi-distinct-categorical-axis trees.** Up to 3 distinct categorical TS axes per tree → exact cat×cat[×cat] tables — the invariant-safe replacement for CatBoost combination CTRs (which use >3 raw features → break exactness; **forbid them**). Recovers ~1.86% logloss CatBoost gets from combinations. *Preserves (needs raw-feature provenance). Cost medium. STRONG.*
11. **[per-tree-strength] Multi-step Newton leaf estimation + Armijo backtracking.** Sharpens leaves on non-quadratic losses (Gamma/Tweedie/MAE). *Preserves (8 values only). Cost medium. STRONG — modest, real on log-link; ~nil on squared-error.*
12. **[representation] Monotone & interaction constraints as accuracy levers.** Per-level joint leaf-clamp. Wrong sign costs 2.5–21.2% AUC; correct monotonicity costs <0.2% (Henckaerts) — a near-free correct prior. *Preserves (restricts values, not structure). Cost medium. STRONG.*
13. **[regularization] MVS sampling + Bernoulli + column subsampling.** MVS `p_i=min(1,√(g_i²+λh_i²)/μ)` dominates GOSS on estimator variance; speed + accuracy together. Skip GOSS. *Preserves. Cost medium. STRONG.* (Correct formula weights the **hessian** term `λh_i²`.)
14. **[optimization] Quantized integer g/h histograms.** ~2× speed AND the bit-reproducibility mechanism for "tables==ensemble" — same mechanism. *Preserves. Cost medium. STRONG.* Refit leaves from full precision (mandatory on Poisson/Gamma/Tweedie).
15. **[representation] Learned missing direction + reserved missing bin.** Missingness as an auditable table row. *Preserves (leaves stay constant). Cost low. STRONG.*
16. **[regularization] LR×n_trees schedule + deviance early stopping + strong defaults.** Depth-3 cap shifts toward more, lower-LR trees. Good defaults = biggest untuned lever. *Preserves. Cost low. CORE.* Export footgun: table accumulation must use the exact `best_iteration` prefix.

### Tier 2 — Real but high-cost / situational

17. **[per-tree-strength] Fully-corrective leaf refit.** Structure frozen → linear in leaf values → ridge/IRLS solve over #trees×8. Fewer trees at equal accuracy = smaller tables. *Preserves. Cost high. STRONG.* Never feed leaf-one-hot to a black-box stage.
18. **[ensembling] Bagging (average table banks).** Canonical variance fix; free per-cell SE bands. *Preserves. Cost high (~8–25×). OPTIONAL — opt-in, sell on credibility not accuracy.*
19. **[optimization] Accelerated/Nesterov boosting (AGBM).** O(1/m²) convergence → up to an order-of-magnitude fewer trees; directly attacks "depth-3 weak → many trees." *Preserves (linear momentum mix). Cost medium. OPTIONAL/v2 — benchmark vs Biau AGB first.*
20. **[per-tree-strength] `max_delta_step`.** Stability guard for log-link Newton steps. *Preserves. Cost low.* **Adversarial verdict: adopt as robustness, NOT a gap-closer** — defaults off, largely inert on TabArena regression/classification.

### Killed / demoted by the adversarial pass

- **Linear/piecewise-linear leaves (GBDT-PL):** `changes_table_form` (cells become x→a+bx, not constant relativities); production `linear_tree` couples >3 features → **breaks I2**; monotonicity doesn't bind leaf coefficients. **Research-only, behind an explicit "shape-function" mode, never default.**
- **STE/soft oblivious (GRANDE/TEL/NODE-GAM) as published:** instance-wise leaf weighting is non-additive, trees aren't oblivious, oblique nodes — **break exactness.** Only a novel axis-aligned+oblivious+global-leaf+fully-hard variant survives → research seed.
- **Ordered boosting, Hessian-weighted sketch, DART, extra_trees/random_strength, snapshot soup, robust losses, joint-w purification:** all *preserve* exactness but are optional/research — small or situational lift, not milestone-critical.

## 3. How much of the gap is realistically closable

**Honest synthesis: mostly closable on typical data, not closable on a high-order minority.**

- **Oblivious vs leaf-wise penalty ≈ 0.** CatBoost (oblivious) and LightGBM (leaf-wise) sit within a few rungs on TabArena; CatBoost ranks *first* untuned. Obliviousness itself costs essentially nothing at the ensemble level — tri-boost's structural penalty is the *order cap*, not symmetric growth.
- **Order-2 GAMs already trail by a small, consistent margin.** EBM pays ~160 Elo vs LightGBM on TabArena (51 datasets) — a clear deficit, not noise, but EBM is on the inference Pareto front. GA2M recovers most of the GAM→full gap (~34% RMSE reduction over plain GAM; Lou 2013), sometimes beating black boxes (pneumonia GA2M 0.857 > RF 0.846).
- **Order-2→3 recovers a real slice.** SIAN: the best model is usually a 2-or-3-D GAM; past order-3 rarely helps and often overfits. NODE-GA3M improves Housing 21.2% over order-2 SOTA. GAMI models match/beat XGBoost at ≤2-way and, **under correlated predictors, can beat XGBoost even at order 3** (XGBoost's greedy splits mishandle correlation — the typical tabular/insurance regime *favors* the structured low-order model). Estimate: **order-3 recovers ~⅓–⅔ of the residual EBM→GBDT gap** where the gap is genuine 3-way (not ≥4-way) structure — most datasets.

**Where we land:**
- **Win/tie (gap ≈ 0):** main-effect + low-order datasets (most business/tabular/insurance), correlated-feature problems, anything where SIAN's order-2/3 is the sweet spot. **Beat EBM essentially everywhere** (strictly more capacity at order 3 + full Newton splits + CatBoost-grade categoricals) — the actual milestone.
- **Lose (gap large):** dense ≥4-way or high-frequency irregular targets — **Higgs, Year** (the named "order ≤5 insufficient" cases). A 3rd-order cap *cannot represent* these — a bias ceiling no ensembling crosses.

**Expected residual on TabArena-like data:** stacking three gains over EBM — exact order-3, full Newton splits vs EBM's tiny-LR stumps, CatBoost-grade categoricals — tri-boost should land **roughly half-to-two-thirds from EBM toward the GBDT frontier (~50–90 Elo behind the best GBDT on average)**: clearly beating EBM/GA2M, within striking distance of the black boxes, *for the bulk of datasets*, with the average dragged down by the high-order minority. Two tempering caveats: depth-3 is more aggressive than CatBoost's depth-6 (need accelerated-boosting / fully-corrective-refit / multi-Newton to hit tree-count parity), and bagging fixes variance not the cap.

## 4. New from the frontier scan

**The biggest under-covered lever: distillation FROM an unconstrained teacher INTO the decomposition.** This is genuinely new relative to the brainstorm and the most promising single gap-closer.

- **META-ANOVA (arXiv:2408.00973, Aug 2024)** transforms *any* black box into a functional-ANOVA model with a consistent interaction-**screening** statistic `I(j)=E[Var{D_j f₀|X_j}]` (drop all interactions containing `j` at once, Apriori-style). Distillation costs almost nothing: Calhousing MSE 0.164→0.165, Abalone 0.432→**0.427 (beat the DNN teacher)**, German AUROC 0.787→0.778. **Recipe for tri-boost:** fit an unconstrained XGBoost/LightGBM teacher, then train the depth-3 oblivious booster against the teacher's *soft margin* (or a blend). This touches only the loss target — never tree shape — so **I1/I2 untouched, result stays exactly decomposable.** The ceiling is principled: the student matches the teacher's *≤3rd-order projection* exactly; only genuine ≥4-way teacher structure is irreducible. **Adopt as a v1.5 training mode** behind the exactness firewall (the distilled model is bit-exact to its own tables; only its fidelity to the teacher is <1). Corroborated by Maillart & Robert (Annals of Actuarial Science 2024), who distill tree ensembles into GAMs at EBM parity.
- **Gradient-based triple detector (FIS from SIAN arXiv:2209.09326; NID/PID).** Run feature-interaction detection on a quickly-trained dense net (gradient/Hessian statistics), feed the heredity allow-list — a better-than-FAST front-end for *triples* specifically (FAST is a 2-way screen). The *detector* is pre-training analysis, exactness-neutral → **adopt as a soft prior.** The SIAN/NODE-GA3M *models* are soft/neural → break exactness, don't port (but they validate that order-3 pays: +21.2%).
- **GRANDE STE trick (ICLR 2024):** the soft-train/hard-axis-aligned-infer primitive is the one transferable seed; a novel STE-trained **oblivious, global-leaf** variant would preserve exactness and could improve split *placement* past greedy local optima. **Research-grade.**
- **ANOVA-TPNN / ANOVA-NODE (2025):** confirm identifiability matters but are soft learners → **motivation only**; you already solve identifiability exactly via purification under a fixed `w`.

## 5. The recommended stack

To maximally close the gap while staying exactly decomposable, in priority order:

1. **Newton summed-gain splits + exact leaf weights** — the spine; match XGBoost/LightGBM split quality.
2. **Full order-3 budget with exact 3D triple tables** — the structural differentiator that strictly dominates EBM.
3. **Joint boosting + heredity/FAST/Sobol funnel (soft prior)** — spend the ≤3 budget on *real* interactions, not C(n,3) noise; single final purification.
4. **Ordered Fisher-sorted target statistics + empirical-Bayes shrinkage + multi-categorical-axis trees** — recover CatBoost's categorical edge exactly; forbid combination CTRs.
5. **Log-link deviance objectives + exposure offset + `boost_from_average` f₀ + leaf-value reg (λ, path_smooth) + deviance early stopping** — correct loss landscape, calibration anchor, generalization.
6. **Teacher-distillation training mode (META-ANOVA-style)** — soft-label fit to an unconstrained GBM; recovers everything but the ≥4-way residual; the single best *new* lever.
7. **Opt-in bagged greedy ensemble selection on tables** (with the outer-bag table-average on-ramp) — variance reduction + credibility bands, exact by purification linearity.

Items 1–5 are the v1 spine that gets you to "beat EBM, near-parity on most data." Item 6 is the highest-upside addition for chasing the black boxes. Item 7 squeezes the last 0.5–1.5% for users who can pay K×. Throughout, the honest disclosure stands: none of this lifts the order-3 *bias* ceiling — on genuine ≥4-way data (Higgs/Year-likes) tri-boost is the wrong tool, and that minority is the irreducible cost of exact ≤3rd-order decomposition.
