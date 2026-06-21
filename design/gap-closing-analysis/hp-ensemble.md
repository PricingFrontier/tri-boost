# Ensemble of Hyperparameter-Diverse pattern-boost Models: A Rigorous Evaluation

**Verdict up front:** Worth doing, with caveats. A **greedy ensemble-selection (Caruana 2004) over a hyperparameter-diverse library of pattern-boost models, with replacement and bagged to control selection overfit**, is the right framing — *not* a naive "average all K." It is exactly decomposable, gives a real, evidence-backed **variance-reduction** lift, and costs K× training. But it **cannot** close the part of the EBM→XGBoost gap that is genuine **order-3 bias**. Default: **off**, with a documented `n_ensemble` recipe for users who want the last ~0.5–1.5% and can pay for it. The cheapest high-value variant — **outer-bag table-averaging on a single HP setting** — should be the on-ramp, since it's already a v1.5 item (`Inner/outer bagging`, `Stacking/blending` in the catalogue).

## 1. Exact mechanics — and the confirmation it stays decomposable

The decisive fact is **purification linearity** (Lengerich Cor. 2.2, research/03 §2.5): purification is a linear operator with `Σαᵢ = 1`, so `purify(Σ αᵢ Fᵢ) = Σ αᵢ purify(Fᵢ)`. Any convex/linear combination of decomposable models is decomposable, and "purify-then-average ≡ average-then-purify." Since **every member is order ≤3 by construction (I1), any weighted average is a sum of order-≤3 tensors — order ≤3 (I2) holds automatically.** This is airtight and is the whole reason this proposal is admissible at all where stacking-with-a-foreign-model is not.

The wrinkle is *alignment*, because HP diversity deliberately makes members live on **different grids and supports**:

- **Different `max_bin` / border type → different border grids.** Member A may cut feature *i* at {30, 55}; member B at {25, 40, 60}. Tables are only summable on a **common merged grid** = the **sorted union of all realized thresholds across all members** for each feature (research/03 §4.2 generalized from one ensemble to K). Each member's piecewise-constant tensor maps *losslessly* onto the finer union grid (a cell that was constant over [30,55) is just replicated across the sub-cells [30,40),[40,55)). **Zero approximation** — this is the same losslessness argument as intra-ensemble accumulation, applied across ensembles.
- **Different realized supports → union of feature-sets.** If member A learned triple {i,j,k} and member B only {i,j}, the averaged bank contains {i,j,k} (B contributes 0 to it). So the **member union has *more* tables than any single member** — more before pruning. This is the real cost: grid union size grows (roughly additively in distinct cuts) and the triple count is the union, not the min. Mitigation is exactly the existing display/selection machinery: re-purify the averaged bank once, then prune by Sobol `σ²(f_u)/σ²(F)` (research/03 §6). Pruning is a *display* operation; the complete union support remains for lossless inference (brainstorm §5, complete-for-inference / pruned-for-display).
- **Different `max_interaction_order`.** Averaging an order-2 member with an order-3 member is fine: the order-2 member simply has empty 3-way tensors. Result is order-3. (Averaging *down* the cap is not a way to escape bias — see §2.)
- **Weights.** For a plain soup `αᵢ = 1/K`; for a weighted soup/ensemble-selection `αᵢ` = selection multiplicities / K. **Constraint that must be enforced: `Σαᵢ = 1` and (ideally) `αᵢ ≥ 0`** so the intercept folds cleanly into `f₀` and the operation is a genuine convex combination (matches the brainstorm's "only affine folds in" calibration stance, and the §2-table "stacking: linear blend only"). A non-convex or learned-nonlinear meta-weighting reintroduces order inflation and is banned.

**One subtlety on the link.** Members must be averaged in the **space where the model is additive** — i.e. **raw score space `F` (pre-link)**, the space the tables live in. Averaging in *response* space (`exp(F)` for Poisson, `σ(F)` for logistic) is a nonlinear op and breaks additivity. Average the score-space table banks; the link is applied once at the end. This matters because deviance ensembling intuition often lives in probability space — here it must not.

So: **confirmed exactly decomposable**, with the only real engineering being a K-way grid/support union + one re-purification pass + variance pruning.

## 2. Does it close the gap? Variance, not bias — be explicit

This is the crux and the most common misconception, so it must be stated bluntly. **The order-3 cap is a bias limit. Ensembling reduces variance. The two do not touch.** Every member is order ≤3; their average is order ≤3; the average's *approximation error to a true order-4+ target is bounded below by the best order-≤3 approximation*, which no amount of averaging crosses. If the residual EBM→XGBoost gap on a dataset is genuine ≥4th-order structure (or order-3 structure the greedy oblivious search can't *find*, which is a search/optimization gap, partly recoverable), HP-ensembling will not recover it.

What it *does* recover is the **variance** component: the instability of a single greedy depth-3 oblivious fit — which border grid, which feature won a near-tied split, which seed/subsample — exactly the regime where averaging pays. And depth-3 oblivious trees are *deliberately weak/high-variance per tree* (research/01 §3), so there is more variance to harvest than in a deeper model.

**Evidence on magnitude (realistic, not hype):**
- **Ensemble selection (Caruana et al. 2004, "Ensemble Selection from Libraries of Models").** Greedy forward selection with replacement on a held-out metric, over a large library of models trained with varied HPs/algorithms, was the strongest method across their benchmark and consistently beat the single best model and plain Bayesian averaging. The lift over "pick best by validation" is real but **modest — typically low single-digit % on the target metric** — and it is overwhelmingly variance reduction / model-selection robustness.
- **Model Soups (Wortsman et al. 2022).** Averaging *weights* of many fine-tunes gave ~1% top-1 on ImageNet. Two caveats for us: (a) soups work because the members are in one loss basin (shared init); HP-diverse GBM table banks are *not* weight-space-connected, so the right analogue is **output/table averaging**, not weight averaging — and table-averaging is exactly justified by purification linearity, so we're on firmer ground than CV soups. (b) **Greedy soup > uniform soup** in their own results — i.e. *selecting which members to add* beats averaging all, which directly motivates ensemble-selection over a blind soup.
- **Hyper-deep / hyperparameter ensembles (Wenzel et al. 2020; Lakshminarayanan 2017).** Adding **HP diversity on top of seed diversity** beats deep (seed-only) ensembles at equal size — diversity from *different inductive biases* (different `max_bin`, `λ`, interaction order, reference of which features get used) decorrelates members more than re-seeding alone. This is the single most useful finding for *where to get diversity* (§3).
- **Snapshot / cyclic-LR ensembles (Huang 2017).** Cheap (one training run, K checkpoints). The GBM analogue is **prediction truncation at different `ntree_end`** (a v1 capability per the catalogue) — near-free diversity but *weakly* decorrelated (members share early trees), so smaller lift than independently-trained members.

**Net:** expect **~0.5–1.5% deviance/logloss improvement** over the single best-tuned model on most TabArena tasks, larger where the single fit is unstable (small n, many near-tied splits, high-cardinality categoricals where the TS ordering is noisy), ~0 where the gap is hard order-≥4 bias. **This narrows but does not eliminate** the gap to unconstrained GBMs; against **EBM the relevant comparison is also ensembled** (EBM already bags 14× internally), so to claim "beat EBM" honestly, pattern-boost's bagging must be compared to bagged EBM, not single EBM.

## 3. Best way to do it

**Where diversity comes from (ranked by payoff/$):**
1. **Hyperparameters** — `max_bin`/border type (different grids = different breakpoints), `λ`/`l2_leaf_reg`, subsample/colsample fractions, `max_interaction_order ∈ {2,3}`, learning-rate×n_trees operating point, `path_smooth`, `random_strength`. Highest decorrelation (Wenzel). Keep all members order ≤3.
2. **Seeds + row/column subsamples** — cheap, real, but more correlated than HP diversity. The classic bagging win; pairs naturally with the existing subsampling knobs.
3. **Reference-measure diversity — NO.** Do *not* diversify `w` across members. `w` is a *post-hoc, retrain-free* re-purification choice (brainstorm §5); mixing reference measures within one averaged bank is incoherent (the variance-sum identity branches on `w`). Pick one `w` (default Laplace product), average, then optionally re-export under alternate `w`.
4. **Feature subspaces** — `rsm`/colsample as one *axis* of HP diversity; not a separate mechanism.

**Combination rule — greedy ensemble selection, not select-then-average, not blind soup:**
- **Greedy forward selection with replacement (Caruana)** on a held-out set, optimizing **held-out deviance/logloss** (strictly proper for the mean on Poisson/Gamma/Tweedie — brainstorm §8 already mandates deviance as the metric; **do not select on RMSE for these losses**). With-replacement selection naturally produces **weighted** members (multiplicities ⇒ a non-uniform convex soup), which dominates uniform averaging.
- This **dominates "train K, pick the single best, average them all"**: picking-the-single-best throws away variance reduction; averaging-all is dragged down by weak members. Greedy selection gets both — and Wortsman's greedy-soup result independently confirms select>average.

**Overfitting the selection (the real risk) — use Caruana's two fixes:** (1) **initialize the ensemble with the top-N single models** (don't start from empty), and (2) **bagged ensemble selection** — run the greedy selection on multiple bootstrap replicates of the *held-out* set and average the resulting weight vectors. Without this, greedy selection on a finite validation set overfits the validation metric and the gain evaporates out-of-sample. This is non-negotiable when K is large.

**How many members:** library of **K ≈ 8–16** HP-diverse models is the sweet spot (diminishing returns past ~16; Caruana's libraries were larger but most mass concentrates on a few). The *selected* ensemble is usually 3–8 effective members.

## 4. Cost vs payoff, and the default

Training is **K× a single fit** (the dominant cost; purification/union/selection are cheap by comparison — purify is `O(#cells·log(1/ε))` and embarrassingly parallel, research/03 §6). Inference is **free**: the averaged-and-re-purified bank is *one* table set of the same form — LUT-sum scoring is independent of K (this is a structural gift; the ensemble does **not** multiply inference cost the way a stacked black-box would).

**Payoff:** ~0.5–1.5% variance-driven deviance lift + smoother, error-barred tables (a credibility artifact actuaries want, per the §2-table inner/outer-bagging note) — for K× training and a denser pre-prune table bank.

**Recommended default:**
- **Single best-tuned model: default.** Ensembling **off** by default (consistent with brainstorm §8 "sensible defaults, not a tuning product").
- **On-ramp (cheapest, ship first):** **outer-bag table averaging at one HP setting**, ~8 bags, `n_ensemble`-style flag — already v1.5, gives most of the variance win + free per-cell error bars, no HP search needed.
- **Full recipe (opt-in `ensemble=True`, K≈12):** train a small HP-diverse library → align on union grid → **bagged greedy ensemble-selection with replacement on held-out deviance, seeded from top-2 singles** → re-purify once → prune by Sobol for display. Enforce `Σαᵢ=1, αᵢ≥0`, score-space averaging, single shared `w`.
- **Always disclose:** this is variance reduction; it does not lift the order-3 bias ceiling, and the honest EBM comparison is bagged-vs-bagged.
