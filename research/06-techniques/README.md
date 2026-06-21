# pattern-boost — GBM Technique Inventory (research round 2)

> 2026-06-21. An exhaustive sweep of every technique in **CatBoost, LightGBM, XGBoost** and the **current GBM research frontier**, each assessed for what it buys (speed / accuracy / memory / uncertainty / calibration / robustness / usability / interpretability) and — the part that decides everything for us — whether it survives pattern-boost's two hard invariants:
> **(I1)** depth-3 **oblivious** structure (one shared split per level, ≤3 features per tree); **(I2)** exact **≤3rd-order fANOVA** decomposition into lossless lookup / rating tables.
>
> **Method.** 10 parallel inventory agents (3 libraries + 7 research-frontier areas) → dedup → **174 consequential techniques adversarially re-judged** by independent skeptics against I1/I2 → 7 synthesis sections + a 2024–2026 completeness critic. 192 agents total. **The adversarial verdicts override the finders' (often optimistic) self-ratings** — that pass is what produced the "incompatible / breaks-exactness" boundary below.

## By the numbers — 229 distinct techniques

| Priority | Count | Meaning |
|---|---|---|
| **v1** | 35 | adopt first; native fit, clear payoff |
| **v1.5** | 44 | strong; adopt right after the core works |
| **v2** | 77 | valuable later / bigger lift / more work |
| **research** | 28 | promising but needs design work or risks an invariant |
| **skip** | 45 | redundant, incompatible, or not worth it for us |

**Oblivious fit:** 151 native · 59 adaptable · **19 structurally incompatible**.
**fANOVA impact:** 128 preserve exactness · 55 N/A · **19 change the table form** · **27 break exactness**.

## The three headline findings

1. **Our structure is a systems gift, not a tax.** A depth-3 oblivious ensemble is the friendliest possible shape for "regular-kernel" speed techniques (one tiny histogram tensor per level; one branch-free 3-bit gather for inference) — and it *already eliminates* the irregular tree-traversal cost that famous inference compilers (Treelite, lleaves, QuickScorer, HummingBird) exist to remove. Those are **moot for us, not incompatible**. The inference win — branch-free SIMD bit-index lookup, then **LUT-sum scoring straight from the purified tables at a cost independent of tree count** — is a v1 deliverable, and validated to the hardware level by TreeLUT (FPGA'25) and CatBoost's RVV work.

2. **The invariants cleanly partition the frontier.** Everything that makes a tree *non-oblivious* (leaf-wise/lossguide growth, `gblinear`) or makes a leaf *non-constant or >3-feature* (linear-leaf trees; soft/neural oblivious trees — NODE, GRANDE, TEL, GrowNet, NODE-GAM, DeepGBM) lands in **incompatible** or **breaks-exactness**. They define the design boundary, not the candidate set. Conversely, almost everything *orthogonal to tree shape* — losses, sampling, regularization, monotone/interaction constraints, missing-value handling, calibration, most uncertainty heads — composes **natively**.

3. **Two techniques are the genuine tension points.** **Ordered Target Statistics** (CatBoost's single biggest categorical-accuracy edge) and **linear / piecewise-linear leaves** (a real accuracy lift) both *change the table form* rather than break it. Ordered TS turns a categorical axis into a learned numeric encoding (still a table, but the axis is an encoding, not the raw level); linear leaves turn constant-cell tables into piecewise-linear **shape functions** (still fANOVA-decomposable — the "rating table" becomes a "rating function"). Both are adoptable **if** we accept the changed table semantics — a product decision, flagged explicitly, not an engineering accident.

## What to adopt for v1 — the distilled set (35 techniques)

These are `native` (or cleanly `adaptable`), `preserves_exactness`, and high-payoff. Grouped by what they improve:

**constraints**
- **Absence of native feature-interaction constraints (vs XGBoo…** (CatBoost) — n/a (gap); but interaction control is a stated pricing need.
- **Interaction constraints (interaction_constraints)** (XGBoost) — interpretability/robustness: enforce that the model only learns sanctioned interactions; direc…
- **L2 leaf regularization (lambda)** (XGBoost) — regularization, numerical stability of leaf weights and split gain; key accuracy/generalizatio…
- **Monotone constraints (monotone_constraints)** (CatBoost) — trust/usability + regulatory compliance; modest accuracy cost, large governance value for insu…
- **Monotone constraints (monotone_constraints) with bound prop…** (XGBoost) — usability/robustness/regulatory: guaranteed monotone rating factors (e.g. premium non-decreasi…
- **interaction_constraints** (LightGBM) — interpretability + accuracy/robustness: lets you forbid spurious or non-deployable interaction…
- **max_delta_step (Newton-step cap)** (XGBoost) — training stability/robustness for log-link pricing objectives and rare-event logistic; prevent…
- **min_sum_hessian_in_leaf / min_child_weight & min_data_in_le…** (XGBoost) — regularization + robustness; ensures every leaf/rating cell has enough credibility — important…
- **monotone_constraints + monotone_constraints_method (basic/i…** (LightGBM) — interpretability/usability + regulatory acceptability: monotonicity in price w.r.t. risk facto…

**accuracy**
- **Midpoint borders for low-cardinality numerics (exact splits…** (scikit-learn HistGradientB) — accuracy (exact splits on discrete/ordinal features) + interpretability (table rows align 1:1 …
- **boost_from_average** (CatBoost) — accuracy + speed of convergence (fewer trees to reach the same loss); better-calibrated starti…
- **boost_from_average (mean/offset initialization)** (LightGBM) — accuracy/convergence speed: fewer iterations to reach the same loss; better behaved with skewe…
- **boost_from_average / optimal base score init** (LightGBM) — faster convergence + correct calibration anchor; sets the base-rate / 1.000 reference for expo…
- **max_bin / sketch_eps (histogram resolution)** (XGBoost) — accuracy (finer splits) vs speed/memory tradeoff; raising it recovers split candidates lost un…

**robustness**
- **Empirical-Bayes / additive smoothing toward the global mean…** (Research: Pargent et al. 2) — accuracy + robustness — the decisive factor separating good from bad target encoding; auto-smo…
- **Missing-value strategy: dedicated reserved bin vs learned d…** (XGBoost / LightGBM / CatBo) — accuracy + robustness — both recover signal from missingness; the reserved/edge-bin is simpler…
- **max_delta_step (capped leaf output)** (LightGBM) — robustness/convergence stability for Poisson/Gamma/Tweedie log-link training (exactly pattern-…
- **max_delta_step (capped leaf update, Poisson safeguard)** (XGBoost) — robustness/stability: prevents numerical blow-up and overconfident leaves under near-zero Hess…
- **min_gain_to_split / min_sum_hessian_in_leaf / min_data_in_l…** (LightGBM) — robustness/overfitting control + minor speed. min_sum_hessian_in_leaf is the exposure/credibil…
- **nan_mode (Forbidden / Min / Max) - missing value handling** (CatBoost) — robustness + accuracy on sparse/missing data; zero imputation needed.

**sampling**
- **Bernoulli / Poisson subsampling (stochastic gradient boosti…** (XGBoost/LightGBM) — speed (fewer rows/tree) + regularization. Evidence: variance reduction improves generalization…
- **Column subsampling (per-tree / per-level / per-node)** (XGBoost) — regularization/decorrelation; per-tree variant also cuts histogram work. Standard accuracy-vs-…

**speed**
- **Bernoulli bootstrap (Stochastic Gradient Boosting) + subsam…** (CatBoost) — speed (~1/subsample) + regularization. Standard Friedman SGB; well understood.
- **Quantile estimation on a subsample (bin_construct_sample_cn…** (LightGBM + scikit-learn HG) — speed — removes a full-data sort/quantile pass from training setup with negligible accuracy co…

**inference**
- **Prediction-time controls: ntree_start/ntree_end, staged_pre…** (CatBoost) — inference speed (truncation) + usability (exponent output IS the multiplicative relativity for…

**usability**
- **Early stopping (Iter-type overfitting detector)** (XGBoost/LightGBM) — robustness + usability + speed (avoids wasted rounds); near-universal best practice. Critical …
- **Early stopping + callbacks (EarlyStopping, LearningRateSche…** (XGBoost) — usability/speed: avoids overfitting and wasted rounds; LR scheduling can improve final accurac…
- **Overfitting detector: od_type (IncToDec / Iter), od_pval, o…** (CatBoost) — usability + robustness: automatic iteration count, prevents overfitting without manual tuning …
- **base_margin / base_score (per-instance offset & estimated i…** (XGBoost) — usability/accuracy/convergence: warm-start from a prior model (transfer learning, stacking on …
- **deterministic mode + force_row_wise / force_col_wise reprod…** (LightGBM) — usability/auditability: bit-reproducible models are often required for regulated insurance pri…
- **first_metric_only + early stopping (early_stopping_round, e…** (LightGBM) — usability + speed (fewer wasted iterations) + robustness (avoids over-boosting). Standard, exp…

**calibration**
- **Mean/intercept bias re-anchoring of the base score (cheap g…** (Research: Denuit et al. 20) — calibration: guarantees portfolio A/E=100% at near-zero cost; necessary condition for a filed …
- **Reliability diagram / A-vs-E balance & calibration report (…** (Research: Niculescu-Mizil ) — usability/trust: makes the GBM-violates-balance problem visible and quantified; the deployment…

**interpretability**
- **Feature interaction constraints with singleton-group additi…** (XGBoost feature_interactio) — interpretability+trust+regulatory: lets you cap interaction order, exclude regulator-prohibite…

**systems**
- **Symmetric-tree multithreading layout: per-thread private hi…** (LightGBM/XGBoost histogram) — training speed: near-linear thread scaling without lock contention; deterministic with quantiz…

## The boundary — do NOT pursue (or only as opt-in approximate modes)

**Structurally incompatible with oblivious depth-3 (19)** — these require non-symmetric growth, linear/soft base learners, or full-data traversal that our structure removes:
- **DeepGBM / GBDT2NN — distillation of GBDT structure into a…** (Research: Ke et al., "Deep) — Verified the mechanism against the KDD'2019 paper (Ke et al.), the official implementation (github.com/motefly/DeepGBM)…
- **Distributional Random Forests (MMD-split, weighted empiri…** (Research: Cevid, Michel, N) — Verified against the JMLR 23 paper (Cevid et al. 2022, arXiv:2005.14458), its abstract, and the GRF/DRF mechanics. All …
- **Feature hashing (the hashing trick) for extreme cardinali…** (Research: Vowpal Wabbit / ) — Oblivious tree reaches only 3 of B hash columns; tables become buckets mixing categories. Ordered TS covers this.
- **GAMI-Tree / GAMI-Lin-Tree — adapting XGBoost into an orde…** (Research: Hu, Aramideh, Ch) — VERIFIED AGAINST PRIMARY SOURCES (arXiv:2309.02426 monotone GAMI-Tree; arXiv:2207.06950 model-based-tree GAMI-Tree/GAMI…
- **GRANDE — gradient-trained HARD axis-aligned trees via str…** (Research: Marton, Lüdtke, ) — Verified against primary sources (GRANDE arXiv:2309.17130 HTML v3; GradTree arXiv:2305.03515 v6, whose dense representa…
- **Greedy/cyclic boosting ratio (EBM greediness)** (EBM / InterpretML) — EBM's greedy_ratio/cyclic_progress is a knob over a DIFFERENT machine, not a portable trick. EBM (a GA2M GAM) boosts ON…
- **GrowNet — shallow nets as boosting weak learners with a f…** (Research: Badirli et al., ) — Verified against primary sources (arXiv:2002.07971 abstract; the GrowNet GitHub README; corroborating search summary). …
- **ICEnet smoothness + monotonicity penalties (pseudo-data I…** (Research: Richman & Wuthri) — Verified the mechanism verbatim against arXiv:2305.08807 (eqs 3.1/3.2): the finder's formulas are correct (squared Δ³ W…
- **Incremental / online add-delete GBDT (in-place updates) a…** (Research: "Online GBDT: In) — Oblivious fit (1): INCOMPATIBLE as published. The in-place method (arXiv:2502.01634) is built on standard greedy per-no…
- **Model compilation for deployment — Treelite** (Treelite) — Verdict upheld as SKIP; downgrading the finder's compat from "adaptable" to "incompatible". (1) OBLIVIOUS FIT — fails. …
- **NODE — Neural Oblivious Decision Ensembles (soft entmax o…** (Research: Popov, Morozov &) — Verified against the primary source (Popov/Morozov/Babenko, ICLR 2020, arXiv:1909.06312; full text at ar5iv.labs.arxiv.…
- **NODE-GAM (neural oblivious GAM/GA2M)** (Research: Chang, Caruana, ) — Verified against the full paper (arXiv:2106.01613) and the zzzace2000/nodegam repo. NODE-GAM is a SOFT, differentiable …
- **Order-limited additive boosting (EBM cyclic round-robin) …** (Research: Lou, Caruana, Ge) — VERDICT: reject. The finder's claims (oblivious=adaptable, fanova=none, v1.5) are both wrong FROM pattern-boost's persp…
- **QuickScorer / V-QuickScorer / RapidScorer interleaved bit…** (Research: Lucchese et al. ) — CONFIRMED INCOMPATIBLE — mechanism is redundant/inverted for depth-3 oblivious trees, payoff is ~zero, finder's verdict…
- **Soft decision trees (Frosst–Hinton) — sigmoid-gated hiera…** (Research: Frosst & Hinton,) — VERIFIED against the primary source (Frosst & Hinton 2017, arXiv:1711.09784; full PDF read, Eqs. 1-5). The finder's cla…
- **Tree Ensemble Layer (TEL) — smooth-step routing with EXAC…** (Research: Hazimeh, Ponomar) — Verified against the primary source (Hazimeh, Ponomareva, Mol, Tan, Mazumder, ICML 2020, arXiv:2002.07772, full text at…
- **gblinear booster (linear base learner + coordinate descen…** (XGBoost) — Verified the finder's record against the XGBoost primary docs (xgboost.readthedocs.io/en/stable/parameter.html) and cha…
- **grow_policy depthwise vs lossguide + max_leaves** (XGBoost) — The finder's claims hold up under hard scrutiny; I affirm incompatible / breaks_exactness / skip. (a) OBLIVIOUS/DEPTH-3…
- **grow_policy: Depthwise / Lossguide (the non-Symmetric alt…** (CatBoost) — Verified against CatBoost primary docs (catboost.ai common training parameters): Depthwise and Lossguide are defined by…

**Break the exact ≤3rd-order table decomposition (27)** — adopt only as explicitly-flagged *approximate* modes, never on the default audited path:
- **Accumulated Local Effects (ALE) — first and second order** (Research: Apley & Zhu 'Vis) — ALE is a strictly POST-HOC, model-agnostic visualization method (Apley & Zhu 2020) that finite-differences an already-t…
- **Balance correction / local recalibration (Denuit-Charpent…** (Research: Denuit, Charpent) — Verified against arXiv:2103.03635 (abstract, full text, freakonometrics repo): the balance-corrected predictor is pi_BC…
- **Beta calibration (Kull, Silva Filho & Flach)** (Research: Kull, Silva Filh) — VERIFIED REAL technique. Kull, Silva Filho, Flach, "Beta calibration: a well-founded and easily implemented improvement…
- **CatBoost RMSEWithUncertainty (NGBoost-on-oblivious-trees,…** (CatBoost) — INVARIANT (1) OBLIVIOUS/DEPTH-3 FIT — verdict native, CONFIRMED. RMSEWithUncertainty and the virtual-ensemble machinery…
- **Categorical feature combinations + one_hot_max_size** (CatBoost) — The finder bundled two distinct techniques with opposite verdicts; the dominant one (combinations) is fatal, so the rec…
- **DeepGBM / GBDT2NN — distillation of GBDT structure into a…** (Research: Ke et al., "Deep) — Verified the mechanism against the KDD'2019 paper (Ke et al.), the official implementation (github.com/motefly/DeepGBM)…
- **Distributional Random Forests (MMD-split, weighted empiri…** (Research: Cevid, Michel, N) — Verified against the JMLR 23 paper (Cevid et al. 2022, arXiv:2005.14458), its abstract, and the GRF/DRF mechanics. All …
- **EBM/GA2M production features: monotonicity, missing handl…** (InterpretML/EBM docs and ') — This is a heterogeneous bundle, not one technique; judged piece-by-piece against the invariants it is mostly already-co…
- **GRANDE — gradient-trained HARD axis-aligned trees via str…** (Research: Marton, Lüdtke, ) — Verified against primary sources (GRANDE arXiv:2309.17130 HTML v3; GradTree arXiv:2305.03515 v6, whose dense representa…
- **Global surrogate models (distillation to a glass-box)** (Research: Molnar IML ch. G) — This is not a training/split/objective/table technique at all — it is a post-hoc paradigm for explaining SOMEONE ELSE'S…
- **Greedy feature combinations / cross-features (ctr combina…** (CatBoost) — Verified against CatBoost primary docs (catboost.ai CTR settings + cat-to-numeric pages). Mechanism confirmed: during t…
- **GrowNet — shallow nets as boosting weak learners with a f…** (Research: Badirli et al., ) — Verified against primary sources (arXiv:2002.07971 abstract; the GrowNet GitHub README; corroborating search summary). …
- **Leaf-value + threshold quantization for tiny models (post…** (Research: "Boosted Trees o) — VERDICT: skip (downgraded from claimed v2). The proposed technique is LOSSY post-training low-bit quantization of leaf …
- **LightGBM linear_tree / linear_lambda (production linear-l…** (LightGBM) — VERIFIED against LightGBM's official Parameters docs (lightgbm.readthedocs.io/en/latest/Parameters.html) and the GBDT-P…
- **Multicalibration / multibalance for protected subgroups** (Research: Hebert-Johnson, ) — VERIFIED both primary sources. Hebert-Johnson/Kim/Reingold/Rothblum (ICML 2018, arXiv:1711.08513) and Denuit/Michaelide…
- **NODE — Neural Oblivious Decision Ensembles (soft entmax o…** (Research: Popov, Morozov &) — Verified against the primary source (Popov/Morozov/Babenko, ICLR 2020, arXiv:1909.06312; full text at ar5iv.labs.arxiv.…
- **NODE-GAM (neural oblivious GAM/GA2M)** (Research: Chang, Caruana, ) — Verified against the full paper (arXiv:2106.01613) and the zzzace2000/nodegam repo. NODE-GAM is a SOFT, differentiable …
- **Order-limited additive boosting (EBM cyclic round-robin) …** (Research: Lou, Caruana, Ge) — VERDICT: reject. The finder's claims (oblivious=adaptable, fanova=none, v1.5) are both wrong FROM pattern-boost's persp…
- **Post-hoc Whittaker-Henderson table graduation (smoothness…** (Research: Whittaker-Hender) — INVARIANT 1 (oblivious/depth-3): native, but VACUOUSLY. WH graduation runs entirely downstream on already-extracted 1-D…
- **SGLB virtual ensembles (epistemic uncertainty from one mo…** (CatBoost: Ustimenko & Prok) — VERIFIED AGAINST PRIMARY SOURCES. This is really TWO separable things, and the finder conflated them. (1) OBLIVIOUS COM…
- **Soft decision trees (Frosst–Hinton) — sigmoid-gated hiera…** (Research: Frosst & Hinton,) — VERIFIED against the primary source (Frosst & Hinton 2017, arXiv:1711.09784; full PDF read, Eqs. 1-5). The finder's cla…
- **Tree Ensemble Layer (TEL) — smooth-step routing with EXAC…** (Research: Hazimeh, Ponomar) — Verified against the primary source (Hazimeh, Ponomareva, Mol, Tan, Mazumder, ICML 2020, arXiv:2002.07772, full text at…
- **Venn-ABERS predictors (calibration with validity guarante…** (Research: Vovk & Petej, 'V) — VERIFIED against primary sources (Vovk & Petej, UAI 2014 / arXiv:1211.0025; RHUL IVAP-regression report 2018; "Generali…
- **Warm-start / base_margin offset / init_model continuation…** (XGBoost) — This record bundles THREE mechanically distinct things that must be judged separately; the finder's single "native/none…
- **gblinear booster (linear base learner + coordinate descen…** (XGBoost) — Verified the finder's record against the XGBoost primary docs (xgboost.readthedocs.io/en/stable/parameter.html) and cha…
- **grow_policy depthwise vs lossguide + max_leaves** (XGBoost) — The finder's claims hold up under hard scrutiny; I affirm incompatible / breaks_exactness / skip. (a) OBLIVIOUS/DEPTH-3…
- **grow_policy: Depthwise / Lossguide (the non-Symmetric alt…** (CatBoost) — Verified against CatBoost primary docs (catboost.ai common training parameters): Depthwise and Lossguide are defined by…

**Change the table form (adoptable with a product decision) (19)** — these keep a valid fANOVA decomposition but alter what a "table" means (encoded axes, piecewise-linear cells, distributional heads):
- **Distributional Gradient Boosting Machines (GBMLSS + NFBoo…** (Research: Maerz & Kneib 20) — This is a META-algorithm, not a tree-growth policy, so it must be split into its two halves before judging the invarian…
- **Embedding features → LDA/KNN numeric calcers** (CatBoost) — Mechanism (verified, CatBoost primary docs https://catboost.ai/docs/en/concepts/algorithm-main-stages_embedding-to-nume…
- **Feature hashing (the hashing trick) for extreme cardinali…** (Research: Vowpal Wabbit / ) — Oblivious tree reaches only 3 of B hash columns; tables become buckets mixing categories. Ordered TS covers this.
- **GAMI-Tree / GAMI-Lin-Tree — adapting XGBoost into an orde…** (Research: Hu, Aramideh, Ch) — VERIFIED AGAINST PRIMARY SOURCES (arXiv:2309.02426 monotone GAMI-Tree; arXiv:2207.06950 model-based-tree GAMI-Tree/GAMI…
- **GAMI-Tree / Model-based-tree boosting to fit low-order fA…** (Research: Hu, Chen, Nair, ) — Verified against the paper (arXiv:2207.06950) and the companion comparison (arXiv:2305.15670). Two decisive, primary-so…
- **Linear-leaf (piecewise-linear) oblivious trees — GBDT-PL …** (Research: Shi, Li & Li, "G) — VERIFIED MECHANISM (primary sources): GBDT-PL (IJCAI 2019 / arXiv:1802.05640) uses incremental feature selection — conf…
- **Multiple priors per categorical (prior grid) + per-featur…** (CatBoost) — VERIFIED MECHANISM (primary sources): simple_ctr / combinations_ctr / per_feature_ctr materialize SEVERAL numeric colum…
- **NGBoost (Natural Gradient Boosting for Probabilistic Pred…** (Research: Duan, Anand, Din) — Verified against the NGBoost source (ngboost.py: `models = [clone(self.Base).fit(X, g) for g in grads.T]`; `params -= l…
- **Native categorical optimal split (Fisher's method) + cat_…** (LightGBM) — VERIFIED MECHANICS (LightGBM docs, primary): the split partitions a categorical's k levels into TWO SUBSETS (set-member…
- **Ordered Target Statistics (Ordered TS) — full mechanism b…** (CatBoost) — Split the claim. The BASIC per-feature Ordered TS (permutation-ordered, prior-smoothed running mean -> a numeric column…
- **Quantile regression boosting (pinball loss) + multi-quant…** (LightGBM/XGBoost/CatBoost ) — The record bundles TWO structurally different things that earn different verdicts; the finder's "native / none / v1.5" …
- **RMSEWithUncertainty (mean+variance / NLL head)** (CatBoost) — Verified against CatBoost primary docs: RMSEWithUncertainty is a single multi-output objective with loss L = 0.5*log(2p…
- **Text features (tokenizers / dictionaries / feature_calcer…** (CatBoost) — Verified against CatBoost primary docs (catboost.ai/docs feature_calcers, algorithm-main-stages_text-to-numeric, featur…
- **Text features → numeric estimators (BoW / NaiveBayes / BM…** (CatBoost) — Mechanism confirmed against CatBoost primary docs (catboost.ai/docs/en/concepts/algorithm-main-stages_text-to-numeric a…
- **Tweedie-dominance / convex-order model selection for auto…** (Research: Denuit, Sznajder) — VERIFIED AGAINST PRIMARY SOURCES. This is a model-COMPARISON/EVALUATION criterion applied post-training to predictor ou…
- **Virtual ensembles + posterior_sampling for uncertainty (R…** (Research: Malinin, Prokhor) — Verified against the SGLB paper (arXiv:2006.10562) and CatBoost docs (uncertainty reference; virtual_ensembles_predict)…
- **XGBoostLSS / LightGBMLSS (GAMLSS-style distributional boo…** (Research: Alexander Maerz,) — VERIFIED ARCHITECTURE (primary sources): XGBoostLSS/LightGBMLSS is NOT a new tree-growth policy — it is a multi-paramet…
- **linear_lambda (linear/piecewise-linear leaf regularizatio…** (LightGBM) — The oblivious split structure survives (split-finding is orthogonal to leaf form, and per LightGBM the leaf linear mode…
- **linear_tree (piecewise-linear leaves)** (LightGBM) — VERIFIED against LightGBM source (src/treelearner/linear_tree_learner.cpp) and the official Parameters docs. (1) OBLIVI…

## Detailed sections

| # | Section | Covers |
|---|---------|--------|
| 1 | [Speed, systems & inference](1-speed-systems-inference.md) | sampling, quantized-gradient histograms, GPU, branch-free/LUT inference, compilation, memory, out-of-core |
| 2 | [Accuracy & model class](2-accuracy-model-class.md) | leaf-estimation refinements, linear leaves, DART, soft/neural oblivious trees, accelerated boosting, regularization |
| 3 | [Categorical & feature handling](3-categorical-feature-handling.md) | encodings, ordered TS, native splits, combinations, missing values, binning & table granularity |
| 4 | [Constraints & structure](4-constraints-structure.md) | monotone (all methods), interaction constraints, additive mode, smoothness — keeping tables clean & compliant |
| 5 | [Losses, sampling & objectives](5-losses-sampling-objectives.md) | robust/focal/ranking/quantile/compound losses, MVS vs GOSS, LR schedules, early stopping |
| 6 | [Uncertainty & distributional](6-uncertainty-distributional.md) | NGBoost, XGBoostLSS/LightGBMLSS, PGBM, SGLB, conformal, quantile heads — and pricing relevance |
| 7 | [Interpretability & calibration](7-interpretability-calibration.md) | SHAP/interaction values, H-statistic/FAST table-selection, purification advances, calibration & balance |
| — | [Gaps & 2024–2026 frontier](8-gaps-and-frontier.md) | what the catalogue under-covered: ANOVA-BART, InstaSHAP, Treeffuser, TreeLUT, multicalibration, TabArena/TabPFN, … |

## Master priority table (all 229)

<details>
<summary>Click to expand — every technique, sorted by priority then category (Oblivious / fANOVA columns show the post-adversarial verdict).</summary>

| Priority | Technique | Source | Improves | Oblivious | fANOVA | Cost |
|---|---|---|---|---|---|---|
| v1 | boost_from_average | CatBoost | accuracy + speed of convergence (fewer trees to reach the s… | native | preserves_exactness | low |
| v1 | boost_from_average (mean/offset initial… | LightGBM | accuracy/convergence speed: fewer iterations to reach the s… | native | preserves_exactness | low |
| v1 | max_bin / sketch_eps (histogram resolut… | XGBoost | accuracy (finer splits) vs speed/memory tradeoff; raising i… | native | preserves_exactness | low |
| v1 | boost_from_average / optimal base score… | LightGBM (boost_from_… | faster convergence + correct calibration anchor; sets the b… | native | preserves_exactness | low |
| v1 | Midpoint borders for low-cardinality nu… | scikit-learn HistGrad… | accuracy (exact splits on discrete/ordinal features) + inte… | native | preserves_exactness | low |
| v1 | Mean/intercept bias re-anchoring of the… | Research: Denuit et a… | calibration: guarantees portfolio A/E=100% at near-zero cos… | native | preserves_exactness | low |
| v1 | Reliability diagram / A-vs-E balance & … | Research: Niculescu-M… | usability/trust: makes the GBM-violates-balance problem vis… | native | preserves_exactness | low |
| v1 | Monotone constraints (monotone_constrai… | CatBoost | trust/usability + regulatory compliance; modest accuracy co… | adaptable | preserves_exactness | medium |
| v1 | Absence of native feature-interaction c… | CatBoost | n/a (gap); but interaction control is a stated pricing need. | native | preserves_exactness | medium |
| v1 | monotone_constraints + monotone_constra… | LightGBM | interpretability/usability + regulatory acceptability: mono… | native | preserves_exactness | high |
| v1 | interaction_constraints | LightGBM | interpretability + accuracy/robustness: lets you forbid spu… | adaptable | preserves_exactness | medium |
| v1 | Monotone constraints (monotone_constrai… | XGBoost | usability/robustness/regulatory: guaranteed monotone rating… | native | preserves_exactness | high |
| v1 | Interaction constraints (interaction_co… | XGBoost | interpretability/robustness: enforce that the model only le… | native | preserves_exactness | medium |
| v1 | L2 leaf regularization (lambda) | XGBoost (lambda) / Li… | regularization, numerical stability of leaf weights and spl… | native | preserves_exactness | low |
| v1 | min_sum_hessian_in_leaf / min_child_wei… | XGBoost (min_child_we… | regularization + robustness; ensures every leaf/rating cell… | native | preserves_exactness | low |
| v1 | max_delta_step (Newton-step cap) | XGBoost (max_delta_st… | training stability/robustness for log-link pricing objectiv… | native | preserves_exactness | low |
| v1 | Prediction-time controls: ntree_start/n… | CatBoost | inference speed (truncation) + usability (exponent output I… | native | preserves_exactness | low |
| v1 | Feature interaction constraints with si… | XGBoost feature_inter… | interpretability+trust+regulatory: lets you cap interaction… | native | preserves_exactness | medium |
| v1 | nan_mode (Forbidden / Min / Max) - miss… | CatBoost | robustness + accuracy on sparse/missing data; zero imputati… | native | preserves_exactness | low |
| v1 | max_delta_step (capped leaf output) | LightGBM | robustness/convergence stability for Poisson/Gamma/Tweedie … | native | preserves_exactness | low |
| v1 | min_gain_to_split / min_sum_hessian_in_… | LightGBM | robustness/overfitting control + minor speed. min_sum_hessi… | adaptable | preserves_exactness | low |
| v1 | max_delta_step (capped leaf update, Poi… | XGBoost | robustness/stability: prevents numerical blow-up and overco… | native | preserves_exactness | low |
| v1 | Empirical-Bayes / additive smoothing to… | Research: Pargent et … | accuracy + robustness — the decisive factor separating good… | native | preserves_exactness | low |
| v1 | Missing-value strategy: dedicated reser… | XGBoost / LightGBM / … | accuracy + robustness — both recover signal from missingnes… | native | preserves_exactness | low |
| v1 | Bernoulli / Poisson subsampling (stocha… | XGBoost/LightGBM (sub… | speed (fewer rows/tree) + regularization. Evidence: varianc… | native | preserves_exactness | low |
| v1 | Column subsampling (per-tree / per-leve… | XGBoost (colsample_by… | regularization/decorrelation; per-tree variant also cuts hi… | native | preserves_exactness | low |
| v1 | Bernoulli bootstrap (Stochastic Gradien… | CatBoost | speed (~1/subsample) + regularization. Standard Friedman SG… | native | preserves_exactness | low |
| v1 | Quantile estimation on a subsample (bin… | LightGBM + scikit-lea… | speed — removes a full-data sort/quantile pass from trainin… | native | preserves_exactness | low |
| v1 | Symmetric-tree multithreading layout: p… | LightGBM/XGBoost hist… | training speed: near-linear thread scaling without lock con… | native | preserves_exactness | medium |
| v1 | Overfitting detector: od_type (IncToDec… | CatBoost | usability + robustness: automatic iteration count, prevents… | native | preserves_exactness | low |
| v1 | deterministic mode + force_row_wise / f… | LightGBM | usability/auditability: bit-reproducible models are often r… | native | preserves_exactness | medium |
| v1 | first_metric_only + early stopping (ear… | LightGBM | usability + speed (fewer wasted iterations) + robustness (a… | native | preserves_exactness | low |
| v1 | base_margin / base_score (per-instance … | XGBoost | usability/accuracy/convergence: warm-start from a prior mod… | native | preserves_exactness | low |
| v1 | Early stopping + callbacks (EarlyStoppi… | XGBoost | usability/speed: avoids overfitting and wasted rounds; LR s… | native | preserves_exactness | low |
| v1 | Early stopping (Iter-type overfitting d… | XGBoost/LightGBM (ear… | robustness + usability + speed (avoids wasted rounds); near… | native | preserves_exactness | low |
| v1.5 | leaf_estimation_method (Newton / Gradie… | CatBoost | accuracy: multiple leaf-estimation iterations sharpen leaf … | adaptable | preserves_exactness | medium |
| v1.5 | Interaction filtering / detection (FAST… | Research: Lou, Caruan… | accuracy + speed + usability: avoids spurious high-order ta… | native | preserves_exactness | medium |
| v1.5 | Stacking / blending GBMs (and stacking … | Research: Wolpert sta… | accuracy: variance reduction / bias correction from combini… | adaptable | preserves_exactness | low |
| v1.5 | Quantization border-selection objective… | CatBoost (border type… | accuracy (better-placed borders capture structure with fewe… | native | preserves_exactness | medium |
| v1.5 | CANN-style residual boosting on a GLM (… | Research: Schelldorfe… | accuracy+usability: never worse than the incumbent GLM; let… | native | preserves_exactness | low |
| v1.5 | Frequency-severity (Poisson x Gamma) tw… | Research: Henckaerts … | accuracy+interpretability: isolates each peril component; s… | native | preserves_exactness | low |
| v1.5 | Balance correction / local recalibratio… | Research: Denuit, Cha… | calibration/usability: restores global+local financial bala… | adaptable | breaks_exactness | low |
| v1.5 | Isotonic regression calibration | Zadrozny & Elkan (200… | calibration: flexible monotone recalibration; for regressio… | native | preserves_exactness | low |
| v1.5 | Categorical CTR types (Borders / Bucket… | CatBoost | accuracy on categorical data - the documented largest sourc… | adaptable | preserves_exactness | high |
| v1.5 | Ordered Target Statistics (Ordered TS) … | CatBoost | accuracy on categorical-heavy data — the single biggest sou… | adaptable | changes_table_form | medium |
| v1.5 | Counter CTR (frequency / count encoding) | CatBoost | accuracy + robustness — adds a leakage-free prevalence sign… | native | preserves_exactness | low |
| v1.5 | One-hot for low-cardinality categorical… | CatBoost | accuracy + interpretability for small categoricals — exact … | native | preserves_exactness | low |
| v1.5 | Sorted-by-encoded-mean ordinal split (F… | Research: Fisher 1958… | accuracy — matches native-partition split quality (the opti… | native | preserves_exactness | medium |
| v1.5 | K-fold / cross-fitted target encoding | Research: scikit-lear… | accuracy + robustness — the standard external-preprocessing… | native | preserves_exactness | medium |
| v1.5 | Feature penalties: feature_weights, pen… | CatBoost | usability/interpretability + cost-aware modeling: shrink th… | native | preserves_exactness | low |
| v1.5 | min_gain_to_split / gamma (complexity p… | XGBoost (gamma) / Lig… | regularization; can yield <depth-3 trees (touching <3 featu… | native | preserves_exactness | low |
| v1.5 | Discrimination-free pricing (marginaliz… | Research: Lindholm, R… | fairness/robustness/usability: provably removes proxy discr… | native | preserves_exactness | medium |
| v1.5 | Heredity / strong-hierarchy constraint … | Research: GAMI-Net (Y… | interpretability + memory: fewer, hierarchically-coherent t… | adaptable | preserves_exactness | low |
| v1.5 | Branch-free SIMD-across-rows oblivious … | CatBoost (SSE/AVX eva… | inference latency/throughput: branch-free, vectorizes acros… | native | preserves_exactness | medium |
| v1.5 | LUT-sum inference (predict directly fro… | pattern-boost-specifi… | inference: collapses an M-tree ensemble into O(#nonempty-ta… | native | preserves_exactness | medium |
| v1.5 | Feature importances: PredictionValuesCh… | CatBoost | interpretability - and several map DIRECTLY onto pattern-bo… | native | preserves_exactness | medium |
| v1.5 | SHAP from the exact fANOVA decompositio… | Research: Bordt & von… | interpretability + speed: turns local SHAP into a handful o… | native | preserves_exactness | low |
| v1.5 | Partial Dependence Plots (PDP) — exact … | Research: Friedman (2… | interpretability: global feature-effect curves/surfaces for… | native | preserves_exactness | low |
| v1.5 | Friedman & Popescu H-statistic (interac… | Research: Friedman & … | interpretability + interaction SELECTION: ranks which pairs… | native | preserves_exactness | low |
| v1.5 | Sobol indices / Shapley effects for com… | Research: Owen 'Sobol… | interpretability: principled, additive, model-faithful glob… | adaptable | preserves_exactness | low |
| v1.5 | Hierarchical-orthogonality purification… | Research: Hooker (200… | interpretability/robustness under dependence: tables that r… | native | preserves_exactness | medium |
| v1.5 | Huber loss (regression) | LightGBM / CatBoost /… | robustness to heavy-tailed severity / mislabeled targets; l… | native | preserves_exactness | low |
| v1.5 | path_smooth (Bayesian parent-shrinkage … | LightGBM | robustness/calibration: stabilizes leaf estimates in low-ex… | adaptable | preserves_exactness | low |
| v1.5 | feature_fraction_bynode (per-node colum… | LightGBM | robustness (variance reduction / overfitting control). No s… | adaptable | preserves_exactness | low |
| v1.5 | reg:absoluteerror (L1) with adaptive-me… | XGBoost | robustness: outlier-resistant fit vs squared error; relevan… | adaptable | preserves_exactness | medium |
| v1.5 | Rare-level collapsing / min_data_per_gr… | LightGBM (min_data_pe… | robustness + interpretability + memory — fewer, more reliab… | native | preserves_exactness | low |
| v1.5 | Bayesian bootstrap (bootstrap_type=Baye… | CatBoost | robustness/accuracy via variance reduction; cheap stochasti… | native | preserves_exactness | low |
| v1.5 | Minimal Variance Sampling (MVS) + mvs_r… | CatBoost | speed + accuracy: for a given sample budget MVS preserves s… | native | preserves_exactness | medium |
| v1.5 | rsm / colsample_bylevel (random subspac… | CatBoost | speed (fewer candidate features scanned) + regularization (… | native | preserves_exactness | low |
| v1.5 | Quantized (low-bit) gradient/hessian tr… | LightGBM / Research: … | speed: up to 2x end-to-end vs SOTA GBDT (CPU/GPU/distribute… | native | preserves_exactness | medium |
| v1.5 | Quantile regression objective (reg:quan… | XGBoost | uncertainty: gives prediction intervals / VaR-style quantil… | adaptable | preserves_exactness | medium |
| v1.5 | Quantile regression boosting (pinball l… | LightGBM/XGBoost/CatB… | uncertainty/usability: direct prediction intervals and tail… | adaptable | changes_table_form | low |
| v1.5 | Conformal prediction wrappers: split/na… | Research: Barber/Cand… | uncertainty/robustness/usability: guaranteed-coverage inter… | native | preserves_exactness | low |
| v1.5 | Conformalized Quantile Regression (CQR) | Research: Romano, Pat… | uncertainty/calibration: guaranteed coverage with HETEROSCE… | native | preserves_exactness | low |
| v1.5 | Quantile / Pinball loss | XGBoost (reg:quantile… | uncertainty quantification (prediction intervals), robust c… | native | preserves_exactness | medium |
| v1.5 | Inner/outer bagging for smoothing + err… | EBM / InterpretML (in… | smoother, more stable tables + per-table-entry uncertainty/… | native | preserves_exactness | medium |
| v1.5 | Conformalized prediction intervals for … | Research: 'Conformal … | uncertainty/usability: valid coverage without distributiona… | native | preserves_exactness | low |
| v1.5 | refit / continued training (init_model,… | LightGBM | usability/maintainability: periodic recalibration of an in-… | native | preserves_exactness | medium |
| v1.5 | Warm-start / base_margin offset / init_… | XGBoost (base_margin,… | usability + accuracy: lets pattern-boost (a) blend on top o… | native | breaks_exactness | low |
| v2 | Stochastic Gradient Langevin Boosting (… | Research: Ustimenko &… | accuracy/robustness on multimodal losses (e.g. 0-1 loss); g… | native | none | medium |
| v2 | leaf_estimation_backtracking (No / AnyI… | CatBoost | robustness/accuracy on hard losses (e.g. Tweedie with small… | native | none | low |
| v2 | DART (Dropouts meet Multiple Additive R… | LightGBM | accuracy/robustness: 'significant margin' over MART in the … | native | none | medium |
| v2 | DART booster (dropout MART) | XGBoost | accuracy: regularization that can beat plain GBM on some ta… | native | none | medium |
| v2 | Per-tree/level/node column subsampling … | XGBoost | accuracy (variance reduction / decorrelation) + speed (fewe… | adaptable | preserves_exactness | low |
| v2 | scale_pos_weight (class-imbalance rewei… | XGBoost | accuracy on imbalanced classification (e.g. rare-claim/frau… | native | none | low |
| v2 | num_parallel_tree (boosted random fores… | XGBoost | accuracy/robustness: variance reduction via bagging within … | native | none | low |
| v2 | Accelerated Gradient Boosting (Nesterov… | Research: Biau, Cadre… | accuracy/speed: provably O(1/m^2) convergence; empirically … | native | none | medium |
| v2 | Fully-corrective / totally-corrective l… | Research: RGF (Johnso… | accuracy + fewer trees: corrects stagewise over-shrinkage a… | native | none | medium |
| v2 | GBDT+LR style leaf-index feature transf… | Research: He et al., … | accuracy + usability: automatic interaction feature enginee… | native | none | low |
| v2 | Learning-rate schedules / decay | XGBoost (LearningRate… | modest accuracy/convergence gains and overfit control in so… | native | none | low |
| v2 | Multiple priors per categorical (prior … | CatBoost | accuracy/robustness — a built-in shrinkage sweep without a … | native | changes_table_form | medium |
| v2 | GLMM / mixed-model (random-intercept) e… | Research: category_en… | accuracy — top performer in Pargent's benchmark for high-ca… | native | none | medium |
| v2 | Mondrian / group-conditional & localize… | Research: Vovk Mondri… | calibration/robustness/usability: per-segment coverage guar… | native | none | low |
| v2 | Venn-ABERS predictors (calibrated proba… | Research: Vovk & Pete… | calibration/uncertainty: provably-calibrated class probabil… | native | none | low |
| v2 | Multicalibration / multibalance for pro… | Research: Hebert-John… | calibration+fairness+usability: balances within sensitive g… | adaptable | breaks_exactness | medium |
| v2 | Tweedie-dominance / convex-order model … | Research: Denuit, Szn… | usability/evaluation: a principled, power-agnostic model-co… | native | changes_table_form | low |
| v2 | Platt / sigmoid (logistic) calibration … | Platt (2000) | calibration of the logistic objective's probabilities; chea… | native | preserves_exactness | low |
| v2 | Spline / I-spline (monotone smooth) cal… | Research: Lucena, 'Sp… | calibration: smoother, better-generalizing recalibration th… | native | preserves_exactness | medium |
| v2 | Temperature scaling (single-parameter l… | Research: Guo, Pleiss… | calibration: simplest possible recalibration for the logist… | native | preserves_exactness | low |
| v2 | Native categorical optimal split (Fishe… | LightGBM | accuracy + speed + memory on high-cardinality categoricals:… | adaptable | changes_table_form | high |
| v2 | Native categorical splits (enable_categ… | XGBoost | accuracy/usability: better handling of high-cardinality cat… | adaptable | preserves_exactness | medium |
| v2 | Focal loss (binary classification) | Research: Lin et al. … | imbalanced-classification accuracy / AUCPR (down-weights ma… | adaptable | preserves_exactness | low |
| v2 | scale_pos_weight / class weights | XGBoost / LightGBM (s… | recall/AUC on imbalanced classification at near-zero cost; … | native | preserves_exactness | low |
| v2 | monotone_penalty | LightGBM | accuracy under monotonicity: reduces the over-constraining … | adaptable | preserves_exactness | medium |
| v2 | L1 leaf regularization (alpha) with sof… | XGBoost (alpha) / Lig… | sparsity / simpler tables (some leaves -> 0 = base-level re… | native | preserves_exactness | low |
| v2 | Path smoothing (parent-shrinkage of lea… | LightGBM (path_smooth… | smoother, less overfit leaf values especially in low-data c… | native | preserves_exactness | low |
| v2 | Per-feature penalties / Cost-Effective … | LightGBM (cegb_tradeo… | cost-aware feature selection / sparser feature usage; for p… | native | none | medium |
| v2 | Monotone constraints on encoded categor… | XGBoost / LightGBM (m… | constraints/trust — enforces economically/actuarially requi… | adaptable | preserves_exactness | medium |
| v2 | LightGBM monotone constraint 'intermedi… | LightGBM docs | accuracy under monotonicity: recovers most of the accuracy … | adaptable | preserves_exactness | high |
| v2 | XGBoostLSS / LightGBMLSS (GAMLSS-style … | Research: Alexander M… | distributional/usability: full predictive distribution with… | adaptable | changes_table_form | high |
| v2 | Distributional Gradient Boosting Machin… | Research: Maerz & Kne… | accuracy/calibration of probabilistic forecasts: GBMLSS-Lig… | adaptable | changes_table_form | high |
| v2 | TreeSHAP exact contributions (pred_cont… | XGBoost | interpretability: instance-level attributions and interacti… | native | none | medium |
| v2 | Post-hoc Whittaker-Henderson table grad… | Research: Whittaker-H… | smoothness+usability: smooth, monotone, regulator-friendly … | native | breaks_exactness | medium |
| v2 | SHAP interaction values (Shapley intera… | Research: Lundberg et… | interpretability: localizes WHICH pairs drive a prediction'… | native | preserves_exactness | low |
| v2 | Faith-Shap / Shapley-Taylor interaction… | Research: Tsai, Yeh, … | interpretability/usability: resolves the non-uniqueness fla… | native | none | low |
| v2 | Individual Conditional Expectation (ICE… | Research: Goldstein, … | interpretability: local heterogeneity / interaction diagnos… | native | preserves_exactness | low |
| v2 | FAST interaction screening (GA2M) for s… | Research: Lou, Caruan… | interpretability + interaction SELECTION: keeps the interac… | adaptable | preserves_exactness | medium |
| v2 | Variable Interaction Network / interact… | Research: Friedman & … | interpretability/usability: a one-glance global view of whi… | native | none | low |
| v2 | Pseudo-Huber loss | XGBoost (reg:pseudohu… | robustness like Huber but with a smooth positive hessian ->… | native | preserves_exactness | low |
| v2 | Log-Cosh loss | CatBoost (LogCosh) / … | outlier robustness with a clean positive hessian and zero h… | native | preserves_exactness | low |
| v2 | Fair loss | CatBoost (FairLoss) /… | robustness to outliers similar to Pseudo-Huber/Log-Cosh; ma… | native | none | low |
| v2 | Lq loss (power loss) | CatBoost (Lq) | flexible robustness/aggressiveness tuning with one paramete… | native | none | low |
| v2 | Asymmetric / cost-sensitive losses | Research: cost-sensit… | aligns the model with asymmetric business cost (regulatory/… | native | none | low |
| v2 | Zero-inflated / two-part (hurdle) Tweed… | Research: Zhou et al.… | better fit on extremely zero-inflated auto/P&C claims than … | native | none | high |
| v2 | Zero-inflated Tweedie (ZITw) two-part o… | Research: So & Valdez… | accuracy on zero-heavy pure-premium data; better-calibrated… | native | none | medium |
| v2 | model_size_reg | CatBoost | memory: meaningfully smaller models when many categorical c… | native | none | medium |
| v2 | External-memory / out-of-core training … | XGBoost | memory/scale: train on >RAM data; vertical scaling. Through… | native | none | high |
| v2 | random_strength | CatBoost | robustness/accuracy: reduces overfitting to spurious best s… | native | preserves_exactness | low |
| v2 | extra_trees (Extremely Randomized split… | LightGBM | robustness (variance reduction / less overfitting) + speed … | adaptable | preserves_exactness | low |
| v2 | DART — tree-dropout regularization for … | Research: Vinayak & G… | robustness/accuracy: better generalization and less over-sp… | native | none | low |
| v2 | Quantile / robust target encoder (media… | Research: Quantile En… | robustness/accuracy on heavy-tailed targets — exactly the r… | native | none | medium |
| v2 | sampling_frequency (PerTree vs PerTreeL… | CatBoost | regularization granularity (PerTreeLevel) and correctness f… | native | none | low |
| v2 | MVS — Minimal Variance Sampling | CatBoost (bootstrap_t… | speed AND accuracy together: paper reports MVS degrades onl… | native | preserves_exactness | medium |
| v2 | Bayesian bootstrap (bagging_temperature) | CatBoost (bootstrap_t… | regularization / variance reduction with no row dropping (k… | native | none | low |
| v2 | Sampling frequency: per-tree vs per-lev… | CatBoost (sampling_fr… | extra regularization (per-level) at modest extra cost; per-… | native | none | low |
| v2 | Quantized / integer-gradient training (… | LightGBM | speed (up to ~2x) + memory (smaller histograms) with neglig… | native | none | high |
| v2 | gradient_based subsampling (sampling_me… | XGBoost | speed: train on a fraction of rows per round with minimal a… | native | none | medium |
| v2 | save_binary / two_round dataset loading… | LightGBM | speed (faster repeated loads in tuning loops) + memory (two… | native | none | low |
| v2 | GPU histogram build + ELLPACK compresse… | XGBoost gpu_hist / NV… | training speed: 10-50x histogram build on tens-to-hundreds … | native | none | high |
| v2 | Virtual ensembles + posterior_sampling … | Research: Malinin, Pr… | uncertainty + calibration: prediction intervals, OOD/anomal… | native | changes_table_form | high |
| v2 | boosting=rf (Random Forest mode) | LightGBM | uncertainty/robustness: an ensemble-variance baseline and a… | native | none | low |
| v2 | Expectile regression objective (reg:exp… | XGBoost | uncertainty/robustness: distributional summaries (expectile… | adaptable | preserves_exactness | low |
| v2 | NGBoost (Natural Gradient Boosting for … | Research: Duan, Anand… | uncertainty/calibration: emits a calibrated full predictive… | adaptable | changes_table_form | medium |
| v2 | CatBoost RMSEWithUncertainty (NGBoost-o… | CatBoost (Malinin, Pr… | uncertainty: aleatoric uncertainty from one model at ~point… | native | breaks_exactness | medium |
| v2 | PGBM (Probabilistic Gradient Boosting M… | Research: Sprangers, … | uncertainty/speed: full predictive distribution from one mo… | adaptable | preserves_exactness | high |
| v2 | SGLB virtual ensembles (epistemic uncer… | CatBoost: Ustimenko &… | uncertainty/robustness: epistemic uncertainty (out-of-distr… | adaptable | breaks_exactness | medium |
| v2 | IBUG (Instance-Based Uncertainty via tr… | Research: Brophy & Lo… | uncertainty: flexible, possibly non-Gaussian local posterio… | native | none | medium |
| v2 | Huberized / smooth pinball (quantile wi… | Research: Pinball boo… | makes quantile estimation a first-class Newton objective; a… | native | none | medium |
| v2 | Expectile loss | CatBoost (Expectile) … | asymmetric risk estimation (e.g. cost-asymmetric over/under… | native | preserves_exactness | low |
| v2 | select_features (Recursive Feature Elim… | CatBoost | usability + interpretability: smaller, more auditable model… | native | none | medium |
| v2 | forcedsplits_filename (forced top split… | LightGBM | usability/interpretability: lets the rating table align to … | adaptable | preserves_exactness | medium |
| v2 | pred_leaf (leaf-index output) & iterati… | XGBoost | usability: model introspection, staged/early-stopped infere… | native | preserves_exactness | low |
| v2 | process_type=update with refresh / prun… | XGBoost | usability/robustness: cheap recalibration of an existing ra… | adaptable | preserves_exactness | medium |
| v2 | ONNX export + ONNX Runtime / onnxmltool… | onnxmltools / ONNX Ru… | usability/portability: framework-independent deployment, in… | adaptable | preserves_exactness | low |
| v2 | IncToDec overfitting detector (p-value … | CatBoost (od_type=Inc… | more automatic early stopping (less tuning of patience); ro… | native | none | low |
| v2 | EBM/GA2M production features: monotonic… | InterpretML/EBM docs … | usability/robustness/accuracy: smoother, monotone-compliant… | adaptable | breaks_exactness | medium |
| research | score_function: Cosine / L2 / NewtonL2 … | CatBoost | accuracy: NewtonL2 (the XGBoost/LightGBM canonical second-o… | native | preserves_exactness | low |
| research | Multi-output / vector-leaf trees (multi… | XGBoost | accuracy on correlated multi-target problems + memory (one … | native | preserves_exactness | high |
| research | Fully-corrective / totally-corrective l… | Research: Shalev-Shwa… | accuracy/memory: fewer trees for the same loss (sparser ens… | native | none | high |
| research | Linear-leaf (piecewise-linear) obliviou… | Research: Shi, Li & L… | accuracy + faster convergence: paper reports PL-trees impro… | native | changes_table_form | high |
| research | LightGBM linear_tree / linear_lambda (p… | LightGBM | accuracy/usability: a proven, debugged contract for linear … | adaptable | breaks_exactness | medium |
| research | Hessian-weighted (2nd-order) quantile s… | XGBoost (weighted qua… | accuracy — borders concentrate where the loss curvature (an… | native | preserves_exactness | high |
| research | Beta calibration (Kull, Silva Filho & F… | Research: Kull, Silva… | calibration: better-founded and strictly more flexible than… | adaptable | breaks_exactness | low |
| research | Native optimal categorical partition sp… | LightGBM | accuracy — partition splits are provably stronger than one-… | adaptable | preserves_exactness | high |
| research | monotone_penalty (constraint depth pena… | LightGBM (monotone_pe… | smoother monotone effects / less aggressive constraint enfo… | adaptable | preserves_exactness | low |
| research | Fairness-aware in-processing regulariza… | Research: 'ML with Mu… | fairness: tunable accuracy/fairness frontier; handles conti… | adaptable | preserves_exactness | high |
| research | monotone_penalty (soft monotonicity: fo… | LightGBM monotone_pen… | accuracy under monotonicity: reduces the performance hit of… | adaptable | preserves_exactness | low |
| research | GAMI-Tree / GAMI-Lin-Tree — adapting XG… | Research: Hu, Aramide… | interpretability + accuracy: matches/beats EBM on simulatio… | incompatible | changes_table_form | medium |
| research | n-Shapley values / Shapley-GAM (order-n… | Research: Bordt & von… | interpretability: gives users a dial from 'fair single-feat… | adaptable | preserves_exactness | low |
| research | GAMI-Tree / Model-based-tree boosting t… | Research: Hu, Chen, N… | interpretability/accuracy parity at order<=2 with an explic… | adaptable | changes_table_form | medium |
| research | Neural Interaction Detection (NID) and … | Research: Tsang, Chen… | interaction SELECTION: a data-driven shortlist of candidate… | adaptable | preserves_exactness | high |
| research | survival:aft (Accelerated Failure Time)… | XGBoost | losses/usability: native handling of censored durations (la… | native | none | medium |
| research | Cauchy / Lorentzian loss | Research: robust M-es… | strongest outlier robustness (redescending) — useful for se… | native | none | medium |
| research | Out-of-core / external-memory training … | XGBoost (Chen & Guest… | memory/scale: train on terabyte data in fixed memory; up to… | native | none | high |
| research | Robust Focal Loss (RFL) / Robust-GBDT | Research: Robust-GBDT… | large robustness gains under label noise (up to 40%) and im… | native | none | medium |
| research | DART — Dropout for boosted trees | XGBoost/LightGBM (boo… | regularization / less overfitting, sometimes notable accura… | native | none | high |
| research | device API & GPU hist (device=cuda, rep… | XGBoost | speed: large GPU speedups on histogram-heavy training and i… | native | none | high |
| research | SketchBoost multioutput split-scoring s… | Research: Iosipoi & V… | speed: 23,919s -> 419s on Dionis (355 classes); 40x vs XGBo… | adaptable | preserves_exactness | medium |
| research | GPU-specific behaviors (NewtonL2/Newton… | CatBoost | speed (large data) - deferred; flags the math that pattern-… | native | none | high |
| research | CEGB (Cost-Efficient Gradient Boosting)… | Research: Peter, Dieg… | systems/usability: lower prediction-time feature-acquisitio… | adaptable | preserves_exactness | medium |
| research | LogLinQuantile loss | CatBoost (LogLinQuant… | quantile estimation that respects multiplicative (log-link)… | adaptable | preserves_exactness | medium |
| research | RMSEWithUncertainty (mean+variance / NL… | CatBoost (RMSEWithUnc… | per-prediction uncertainty (aleatoric/heteroscedastic) with… | adaptable | changes_table_form | high |
| research | Venn-ABERS predictors (calibration with… | Research: Vovk & Pete… | calibration+uncertainty: the only listed method with a form… | native | breaks_exactness | medium |
| research | Incremental / online add-delete GBDT (i… | Research: "Online GBD… | usability: fast model refresh / unlearning (GDPR 'right to … | incompatible | preserves_exactness | high |
| skip | grow_policy: Depthwise / Lossguide (the… | CatBoost | accuracy on deep-interaction/asymmetric data (per-leaf adap… | incompatible | breaks_exactness | medium |
| skip | gblinear booster (linear base learner +… | XGBoost | usability/accuracy: a fast linear baseline / GLM-like fallb… | incompatible | breaks_exactness | medium |
| skip | grow_policy depthwise vs lossguide + ma… | XGBoost | accuracy/speed: lossguide can reach lower loss with fewer l… | incompatible | breaks_exactness | low |
| skip | GRANDE — gradient-trained HARD axis-ali… | Research: Marton, Lüd… | accuracy: SOTA vs gradient boosting on most of 19 tabular d… | incompatible | breaks_exactness | high |
| skip | NODE — Neural Oblivious Decision Ensemb… | Research: Popov, Moro… | accuracy: end-to-end gradient training + representation lea… | incompatible | breaks_exactness | high |
| skip | Tree Ensemble Layer (TEL) — smooth-step… | Research: Hazimeh, Po… | accuracy + speed-of-inference: the exact-hard region gives … | incompatible | breaks_exactness | high |
| skip | GrowNet — shallow nets as boosting weak… | Research: Badirli et … | accuracy: reported to outperform XGBoost/LightGBM/CatBoost … | incompatible | breaks_exactness | high |
| skip | DeepGBM / GBDT2NN — distillation of GBD… | Research: Ke et al., … | usability (online/streaming updates) + accuracy on mixed sp… | incompatible | breaks_exactness | high |
| skip | Greedy/cyclic boosting ratio (EBM greed… | EBM / InterpretML (gr… | accuracy (greedy) vs smoother, more balanced tables (cyclic… | incompatible | preserves_exactness | high |
| skip | linear_lambda (linear/piecewise-linear … | LightGBM (linear_tree… | accuracy/efficiency on smooth targets with fewer trees; smo… | adaptable | changes_table_form | high |
| skip | Greedy feature combinations / cross-fea… | CatBoost | accuracy — directly models categorical interactions that si… | adaptable | breaks_exactness | high |
| skip | Categorical feature combinations + one_… | CatBoost | accuracy: captures categorical interactions automatically w… | adaptable | breaks_exactness | high |
| skip | Text features (tokenizers / dictionarie… | CatBoost | usability/accuracy when text is present (free text in claim… | adaptable | changes_table_form | high |
| skip | Embedding features (LDA projection / kN… | CatBoost | accuracy when embeddings (e.g. from a deep model) are avail… | native | preserves_exactness | high |
| skip | Plain mean (target) encoding and its le… | Research: category_en… | NOT a recommended technique — documented as the anti-patter… | native | none | low |
| skip | Leave-one-out (LOO) target encoding | Research: category_en… | accuracy/robustness at near-zero cost — removes the dominan… | adaptable | preserves_exactness | low |
| skip | ICEnet smoothness + monotonicity penalt… | Research: Richman & W… | smoothness+monotonicity+interpretability: gives soft, tunab… | incompatible | preserves_exactness | high |
| skip | Distributional Random Forests (MMD-spli… | Research: Cevid, Mich… | distributional: model-free full/multivariate conditional di… | incompatible | breaks_exactness | high |
| skip | Model compilation for deployment — Tree… | Treelite (Cho & Li, M… | inference throughput 2-6x and dependency-free deployment, b… | incompatible | preserves_exactness | low |
| skip | Model compilation via LLVM — lleaves | lleaves (Sebastian Bo… | inference 10-30x vs interpreted LightGBM (e.g. 9.7s -> 0.4s… | native | preserves_exactness | medium |
| skip | QuickScorer / V-QuickScorer / RapidScor… | Research: Lucchese et… | inference 2-25x vs naive traversal on DEEP forests (LtR, hu… | incompatible | preserves_exactness | high |
| skip | Tensor/matrix-compiled forest inference… | Research: Nakandala e… | inference: big batch throughput on GPU/SIMD; but introduces… | adaptable | preserves_exactness | high |
| skip | Prediction caching / memoization on the… | pattern-boost-specifi… | inference: avoids recomputation for repeated/segmented inpu… | native | preserves_exactness | low |
| skip | linear_tree (piecewise-linear leaves) | LightGBM | accuracy: small but real on smooth/extrapolating targets (n… | adaptable | changes_table_form | high |
| skip | Soft decision trees (Frosst–Hinton) — s… | Research: Frosst & Hi… | interpretability: distillation yields a soft tree that gene… | incompatible | breaks_exactness | medium |
| skip | Order-limited additive boosting (EBM cy… | Research: Lou, Caruan… | interpretability + calibration + accuracy-stability: order-… | incompatible | breaks_exactness | medium |
| skip | Category-axis purification: per-level r… | Research: Lengerich p… | interpretability — turns the raw encoded axis into a minima… | native | preserves_exactness | low |
| skip | Path-dependent TreeSHAP (Tree-path-depe… | Research: Lundberg, E… | interpretability: exact per-row local attributions that sum… | native | none | low |
| skip | Interventional (marginal) TreeSHAP | Research: Lundberg et… | interpretability/robustness: attributions that don't leak m… | native | none | low |
| skip | Accumulated Local Effects (ALE) — first… | Research: Apley & Zhu… | interpretability/robustness under correlated/dependent feat… | adaptable | breaks_exactness | medium |
| skip | NODE-GAM (neural oblivious GAM/GA2M) | Research: Chang, Caru… | interpretability/accuracy on large data vs spline GAMs; dem… | incompatible | breaks_exactness | high |
| skip | Global surrogate models (distillation t… | Research: Molnar IML … | interpretability of OTHER people's black boxes. | adaptable | breaks_exactness | low |
| skip | Ranking objectives + position-bias corr… | LightGBM | usability: enables ranking tasks and debiased click-data tr… | native | none | medium |
| skip | LambdaMART ranking objectives (rank:ndc… | XGBoost | losses: ranking quality (NDCG/MAP) for ordered-list problem… | native | none | high |
| skip | LambdaMART / LambdaRank ranking loss | XGBoost (rank:ndcg/ma… | ranking quality (NDCG/MAP) for ranking tasks; not a pricing… | native | none | high |
| skip | QuantileDMatrix (pre-quantized, compres… | XGBoost | memory: large reduction (skips raw-data copy; ELLPACK ~4x s… | native | preserves_exactness | medium |
| skip | Leaf-value + threshold quantization for… | Research: "Boosted Tr… | memory: 2.4-8x smaller model/tables; enables integer-only e… | adaptable | breaks_exactness | low |
| skip | Feature hashing (the hashing trick) for… | Research: Vowpal Wabb… | memory + usability for huge/streaming cardinality — constan… | incompatible | changes_table_form | medium |
| skip | Poisson bootstrap (GPU) | CatBoost | speed on GPU; equivalent regularization to classical bootst… | native | none | low |
| skip | GOSS — Gradient-based One-Side Sampling | LightGBM (data_sample… | speed (~30% of rows/tree) with bounded gain-estimation erro… | native | none | medium |
| skip | Fast TreeSHAP v1 / v2 | Research: Yang (Linke… | speed of SHAP on conventional ensembles by 1.5-3x. | native | preserves_exactness | medium |
| skip | GPUTreeShap (massively parallel exact T… | Research: Mitchell, F… | speed: large-scale SHAP/interaction throughput. | adaptable | preserves_exactness | high |
| skip | Distributed histogram parallelism: feat… | LightGBM (Features do… | training scale: near-constant communication cost in #machin… | native | none | high |
| skip | Text features → numeric estimators (BoW… | CatBoost (text featur… | usability — lets a model use free-text fields (claim descri… | native | changes_table_form | high |
| skip | Embedding features → LDA/KNN numeric ca… | CatBoost (embedding f… | usability — incorporate pretrained embeddings (e.g. for hig… | native | changes_table_form | high |

</details>
