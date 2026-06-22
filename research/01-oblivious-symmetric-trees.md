# Oblivious / Symmetric Decision Trees in Gradient Boosting — Research Report

Research feeding the design spec of `tri-boost` (depth-3 symmetric/oblivious trees, Rust core + Python bindings). All factual claims below were fetched from primary sources (CatBoost NeurIPS 2018 paper, official CatBoost docs, CatBoost source) and verified; a few flagged numbers come from secondary benchmarks and are marked as such.

> Two corrections folded in from source verification: CatBoost's default split objective is the **Cosine** score function (default `score_function=Cosine`), and `boosting_type` defaults to **Plain on CPU** (Ordered only for small GPU datasets).

## 1. How an oblivious / symmetric tree is built; the per-level split objective

**Structure.** An oblivious (a.k.a. symmetric, "decision table") tree of depth `d` uses **the same `(feature, threshold)` split at every node on a given level**. Verbatim from the paper (§5): *"Term oblivious means that the same splitting criterion is used across an entire level of the tree. Such trees are balanced, less prone to overfitting, and allow speeding up execution at testing time significantly."* Consequence: a depth-`d` tree has exactly **`d` distinct splits** (one per level) and **`2^d` leaves**, forming a complete balanced binary tree. CatBoost docs state it operationally: *"On each iteration, all leaves from the last tree level are split with the same condition. The resulting tree structure is always symmetric."* A numeric split is the indicator `a = 𝟙{x^k > t}` for feature `k` and quantization border `t`.

**How the single split per level is chosen.** The tree is grown greedily, **one level at a time**. At each level the builder iterates over **every candidate split** `c` (every quantized feature × border), and — this is the oblivious constraint — each candidate is **applied to all current leaves simultaneously** (doubling leaf count from `2^level` → `2^{level+1}`). Each candidate is scored over the resulting whole-tree partition, and the **single `(feature, threshold)` maximizing the score for the entire level** is kept. From Algorithm 2: `foreach candidate split c → loss(T_c) ← cos(Δ, G); T ← argmin loss(T_c)`. Because only one split is searched per level (instead of one per node), the per-level search cost is independent of the number of nodes at that level.

**The greedy objective actually optimized.** CatBoost's **default `score_function` is `Cosine`** (verified from docs). It scores a candidate structure by the cosine similarity between the per-example gradient vector `G` and the candidate's leaf-value vector `Δ` (the value each example would receive):

```
Cosine = (Σ_i w_i · a_i · g_i) / ( sqrt(Σ_i w_i a_i²) · sqrt(Σ_i w_i g_i²) )
```

CatBoost also offers `L2`, `NewtonL2`, `NewtonCosine` (Newton variants use 2nd derivatives; CPU supports only `Cosine`/`L2`). This differs from XGBoost/LightGBM, whose canonical split gain is the **Hessian-based** form (the reference formula `tri-boost` will most likely want):

```
Gain = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ] − γ
```

For an oblivious level-split this gain term is summed over **all current leaves** the candidate is applied to, and the best single `(feature, threshold)` for the whole level is selected.

**Leaf values.** Newton step (CatBoost's `NewtonL2`, the standard GBM leaf):

```
w*_j = − (Σ_{i∈I_j} g_i) / (Σ_{i∈I_j} h_i + λ)        leaf_value = learning_rate · w*_j
```

CatBoost's paper default uses gradient averaging instead: Plain mode `Δ(i) = avg(grad of examples in same leaf)`; Ordered mode `Δ(i) = avg(grad_{σ(i)−1} of same-leaf examples that precede i in permutation σ)` to avoid leakage. `leaf_estimation_method ∈ {Gradient, Newton}` selects between them.

**Candidate borders.** Numeric features are **pre-quantized into bins**; bin boundaries ("borders") are the only candidate thresholds. `border_count`/`max_bin` default **254 (CPU)**, 128 (GPU); `feature_border_type` default **`GreedyLogSum`** (options: Median, Uniform, UniformAndQuantiles, MaxLogSum, MinEntropy, GreedyLogSum).

**`grow_policy` options:** `SymmetricTree` (default, oblivious — one shared split per level); `Depthwise` (level-wise but each leaf gets its own best split — not oblivious); `Lossguide` (leaf-wise/best-first, splits the leaf with best loss improvement up to `max_leaves`, default 31 — LightGBM-style).

## 2. Why oblivious trees are fast (inference and training)

**Leaf indexing as a bit vector (primary-source confirmed).** Because each level shares one split, the leaf index is the concatenation of per-level comparison bits. CatBoost source (`model.h`): *"each leaf index is determined by binary vector with length equal to evaluated tree depth."* CatBoost GPU post: *"a tree of depth k has exactly 2^k leaves, and the index of a leaf can be calculated with simple bitwise operations."* Inference per tree = evaluate `d` comparisons, pack into a `d`-bit integer (bit `j = [x_{f_j} > t_j]`), do **one lookup** into a `2^d` leaf table. This replaces the data-dependent pointer-chasing / branchy root-to-leaf traversal of irregular XGBoost/LightGBM trees.

**Vectorization / SIMD (confirmed).** All samples evaluate the **same splits in the same order** → no branch divergence → branchless, vectorizable across samples. CatBoost's evaluator *"binarizes all used float features … and then uses only binary features … built in a data parallel manner with SSE intrinsics."* A peer-reviewed evaluation-optimization paper reports **20–40% additional speedup with AVX2** (no quality loss) and **50–70% with AVX-512 + float16 leaf values** (numbers via PDF summarizer — directionally solid, exact figures approximate).

**Cache behavior.** The leaf table is tiny — `2^d` floats (depth 6 = 64 floats = 256 B; **depth 3 = 8 floats = 32 B**) — fits trivially in L1; access is contiguous and predictable. CatBoost source notes oblivious trees are "very fast" to apply "even without … SIMD optimizations compared to asymmetric trees" (i.e., the cache/branch advantage exists before SIMD).

**Inference speed numbers.** Official CatBoost benchmark (Epsilon, 8000 trees): **~35× faster than XGBoost, ~83× faster than LightGBM** single-threaded (1.83 s vs 71 s vs 88 s). Independent production benchmark (Delivery Hero, MSLR-WEB10K, 300 trees): a more conservative **~2–3× faster**, with single-threaded CatBoost beating 16-threaded LightGBM — but at a **training-time cost** (CatBoost can train slower). Caveat: the big single-thread multipliers shrink when competitors use multi-threaded prediction.

**Training-speed implications.** One `(feature, threshold)` chosen per **level** (not per node) collapses the per-level search; after quantization, histogram cost scales with **features × bins**, not rows. CatBoost: *"If our inputs contain only 5-bit integers, we need to evaluate feature count times 32 different splitting conditions. This quantity does not depend on the number of rows."* This regular structure is highly GPU-friendly (CatBoost reports ~40× GPU-vs-CPU training speedups). Net training trade-off: weaker per-tree learner usually needs **more trees** to match accuracy.

## 3. Accuracy tradeoffs: oblivious vs leaf-wise vs level-wise

**Oblivious trees are weaker per-tree — deliberately.** The constraint (one split per level for all nodes) shrinks each tree's hypothesis space. Paper: oblivious trees are *"balanced, less prone to overfitting."* Secondary sources frame it explicitly as structural regularization: *"restricting … to have only one feature split per level … reduc[es] the complexity … and thereby regularization"*; *"Each split must be globally useful, not just locally optimal for one branch."*

**Three growth strategies:**

| Strategy | Library (default) | Growth | Tradeoff |
|---|---|---|---|
| Level-wise / depthwise | XGBoost | Split all nodes at a depth | Lower variance, systematic; slower, wastes compute on low-gain nodes |
| Leaf-wise / best-first | LightGBM | Split leaf with max Δloss → deep, asymmetric | Lowest loss per fixed #leaves; **highest overfit risk on small data** |
| Symmetric / oblivious | CatBoost | Whole level shares one split → balanced 2^k leaves | Weakest per-tree (= regularization), lowest variance, **fastest inference** |

LightGBM docs (verbatim): *"Holding #leaf fixed, leaf-wise algorithms tend to achieve lower loss than level-wise … Leaf-wise may cause over-fitting when #data is small."*

**Compensation + final accuracy.** CatBoost offsets weaker learners with **many shallow symmetric trees at low learning rate** (defaults: depth 6, lr ≈ 0.03 auto-selected, 1000 iters). On final-ensemble accuracy the evidence is **competitive, often marginally best — not universally dominant**:
- Paper Table 2: CatBoost wins on all 9 datasets, but most gaps are tiny (≤2.4% logloss on Adult/Click/Churn/Upselling/Appetency); **large wins on categorical-heavy data** — Amazon +17%, Internet ~+7% — are largely attributable to ordered TS, not oblivious trees per se.
- Independent studies: Riskified (4 fraud datasets) — CatBoost edges both on WAUC but calls differences "not as significant." Academic AUC study (arxiv 2305.17094): *"XGBoost and CatBoost perform best … LightGBM's performance seems unstable"* (CatBoost ties, not unique winner). A 2025 churn study put CatBoost slightly behind both.

**Bottom line:** weaker per-tree expressiveness does **not** translate into worse final accuracy; the constraint trades a little bias for lower variance and is competitive once enough trees are added. CatBoost's edge is strongest **untuned** (the built-in regularization makes it "less sensitive to tuning") and on **categorical-heavy** data.

## 4. Categorical handling (ordered TS / ordered boosting) and its separability from oblivious trees

**Greedy target statistics (TS)** replace a category with a numeric estimate of `E(y | x^i = x_k^i)` (Eq. 4):
```
x̂_k^i = (Σ_j 𝟙{x_j^i = x_k^i}·y_j + a·p) / (Σ_j 𝟙{x_j^i = x_k^i} + a)
```
Because the sum includes example `k` itself, this leaks the target → **prediction shift** (training/test conditional distributions differ).

**Ordered TS** fixes a random permutation σ and uses only **preceding** examples `D_k = {x_j : σ(j) < σ(k)}` (Eq. 5):
```
x̂_k^i = (Σ_{x_j∈D_k} 𝟙{x_j^i = x_k^i}·y_j + a·p) / (Σ_{x_j∈D_k} 𝟙{x_j^i = x_k^i} + a)
```
`y_k` never enters its own encoding → unbiased ("Ordering Principle": predict each example using only examples before it).

**Ordered boosting** attacks the same prediction shift in the gradient step: standard GBDT estimates gradients on the same data used to fit the model → biased base learner → worse generalization. Ordered boosting maintains supporting models `M_i` trained on the first `i` examples in σ; the gradient for example `j` uses `M_{j−1}` (a model that never saw `j`). CatBoost uses **s+1 permutations** (σ₁…σ_s for split evaluation, σ₀ for leaf values), resampled per tree to reduce variance.

**Separability (critical for `tri-boost`): YES, fully orthogonal.**
- Ordered boosting / ordered TS = *how gradients and categorical encodings are computed* (permutation-based bias correction).
- Oblivious trees = *the base-predictor structure*.
- The paper treats them as distinct components, and CatBoost proves independence operationally via `boosting_type`: **`Plain` = standard non-ordered GBDT but still uses symmetric oblivious trees** (and still uses ordered TS for categoricals). Verbatim docs default is **Plain on CPU** (Ordered only for GPU with ≤50k objects). So you can have oblivious trees without ordered boosting. Conversely, ordered boosting/TS are general schemes that could wrap any base learner.
- Once categoricals become numeric TS values, the split-finder treats them like any numeric feature: bin into borders, evaluate as oblivious level splits. (CatBoost additionally builds greedy categorical **feature combinations** on the fly, each re-encoded as a TS.)

## 5. Practical CatBoost defaults (verified from docs)

| Parameter | Default | Notes |
|---|---|---|
| `depth` | **6** | Recommended 4–10; cap **16** on CPU (8 for GPU pairwise modes); 16 if Lossguide |
| `learning_rate` | **auto** (else 0.03) | Auto-selected from dataset size & iterations for Logloss/MultiClass/RMSE |
| `iterations` / `n_estimators` | **1000** | |
| `l2_leaf_reg` | **3.0** | L2 on leaf values (the λ above) |
| `grow_policy` | **SymmetricTree** | oblivious |
| `score_function` | **Cosine** | CPU: Cosine/L2 only; GPU adds NewtonL2/NewtonCosine |
| `random_strength` | **1** | randomness added to split scores |
| `bagging_temperature` | **1** | Bayesian bootstrap |
| `border_count` / `max_bin` | **254 (CPU)**, 128 (GPU) | candidate thresholds per numeric feature |
| `feature_border_type` | **GreedyLogSum** | quantization mode |
| `boosting_type` | **Plain (CPU)** | Ordered only for GPU ≤50k objects (non-MultiClass) |
| `rsm` / `colsample_bylevel` | **1** | feature subsampling per split |
| `min_data_in_leaf` | **1** | Lossguide/Depthwise only |
| `max_leaves` | **31** | Lossguide only |
| `model_size_reg` | ~0.5 (flagged) | model-size penalty; matters only for categorical models; on by default CPU, off GPU |

## 6. Known downsides / failure modes of oblivious trees

- **Weaker per-tree expressiveness** → needs more (or deeper) trees to capture complex interactions; the ensemble compensates.
- **Cannot adapt the split to a local region**: one condition is forced across the whole level *"even in positions where it is suboptimal,"* which can **waste depth** vs. asymmetric trees that pick the best local split per node.
- **Worse on highly asymmetric / deep-interaction structure**: docs admit *"in some cases, other tree growing strategies can give better results than growing symmetric trees."*
- **Training can be slower than LightGBM**, especially at depth > 8 (a symmetric tree has `2^depth` leaves regardless of need) and from categorical-handling overhead (secondary/benchmark sources — directional).
- **Model-summation / incremental-learning limitation**: CatBoost *"Summation of symmetric and non-symmetric models is not supported"*; switching grow_policy mid-training breaks incremental fits.
- **Switching away from SymmetricTree forfeits**: (1) fast prediction (docs: symmetric *"can be applied much faster (up to 10 times faster)"*), (2) the built-in regularization, (3) tooling (Depthwise/Lossguide lose PredictionDiff importance, export only to json/cbm).

---

## Design implications for tri-boost

**The depth-3 oblivious choice is well-founded and uniquely synergistic with your fANOVA goal.** Each oblivious tree of depth 3 depends on ≤3 features, so the *whole ensemble* is a sum of ≤3-feature functions → its fANOVA decomposition truncates exactly at 3rd order. No other GBM growth policy gives you this guarantee (leaf-wise/level-wise trees touch arbitrarily many features per tree). The literature confirms the structure is fast and self-regularizing — both align with your goals.

**Split-finder (the core loop):**
- Build a **histogram/quantized split-finder**: pre-bin every numeric feature into ≤255 borders once (mirror CatBoost's `border_count=254`, `GreedyLogSum`). After binning, your split search is over `(feature, border)` pairs and is **row-count-independent** per level.
- At each of the 3 levels, evaluate each candidate `(feature, border)` by **applying it to all `2^level` current leaves at once** and summing the per-leaf gain. Use the **Hessian-based gain** `Σ_leaves (Σg)²/(Σh+λ)` (XGBoost/LightGBM canonical) rather than CatBoost's Cosine default — it's the standard, gives you exact second-order leaf values for free, and is what your benchmark targets use. (CatBoost's `NewtonL2` is this same family.) Keep `score_function` pluggable but default to Newton/L2 gain.
- **Exploit histogram subtraction**: compute the histogram for one child by subtracting from the parent (the standard LightGBM trick) — halves histogram work per level.
- With only 3 levels and shared splits, the entire structure search is `3 × (#features × #borders)` gain evaluations — trivially parallelizable across features (Rayon in Rust). This is your speed win; lean into SIMD over the binned gradient/hessian sums.
- **De-duplicate feature usage if you want strict ≤3 distinct features per tree** (depth-3 already gives ≤3 splits, hence ≤3 features — automatically satisfied; no extra constraint needed). Good news: the oblivious structure gives the ≤3-feature property for free.

**Leaf-value computation:**
- Use the **Newton step** `w_j = − G_j/(H_j + λ)` per leaf (8 leaves at depth 3), times `learning_rate`. Store leaves as a flat `[f32; 8]` (or `[f64; 8]`) array indexed by the 3-bit leaf code. Keep `l2_leaf_reg` (λ) default ≈ 3.
- This `2^3 = 8`-float table per tree is the lookup-table primitive that makes both fast inference **and** your fANOVA→LUT conversion natural: each tree is already a function of its 3 features evaluated by a 3-bit index into 8 values.

**Inference / LUT representation (your defining feature):**
- Represent each tree as `(feature_ids[3], thresholds[3], leaf_values[8])`. Prediction = 3 branchless comparisons → pack bits `b = b0 | b1<<1 | b2<<2` → `leaf_values[b]`. This is exactly CatBoost's bit-vector indexing; it vectorizes across rows (SIMD) and is L1-resident.
- For the **fANOVA → lookup-table** export: because every tree is ≤3rd-order by construction, you can sum trees into per-{feature-triple} accumulators and materialize dense or sparse 3D lookup tables on the model's quantization grid. The shared-grid quantization (same borders across the whole model) makes table alignment clean — keep a **global per-feature border set** so all trees share axes.

**What to borrow from CatBoost:**
- Oblivious/symmetric structure + bit-vector leaf indexing + SIMD evaluation (the entire speed story).
- Pre-quantization into ≤254 borders, `GreedyLogSum`-style border selection, shared global borders.
- Defaults as a starting point: depth (yours fixed at 3), lr ≈ 0.03–0.1 auto-scaled, ~1000 iterations, `l2_leaf_reg ≈ 3`, histogram-based row-independent split cost.
- Optionally **ordered TS** for categorical encoding — it's orthogonal to your tree structure and is the single biggest source of CatBoost's accuracy edge on categorical data. You can ship it independently of any ordered-boosting machinery.

**What to avoid / de-scope:**
- **Skip ordered boosting** initially. It is fully separable from oblivious trees, costs ~1.7× training time, needs lots of memory (per-permutation supporting models), and helps mainly on small datasets. CatBoost itself defaults to `Plain` (non-ordered) on CPU. Ship Plain boosting first; ordered boosting is a later, optional accuracy knob.
- **Don't adopt Cosine as the default split score** — use the Hessian/Newton gain for parity with XGBoost/LightGBM accuracy and to get exact leaf values.
- **Don't offer non-symmetric grow policies** — they'd break your ≤3-feature/fANOVA invariant. The whole point of tri-boost is the oblivious constraint; keep it mandatory.
- **Expect to need more, lower-lr trees** than a leaf-wise library for the same accuracy (the per-tree weakness), and budget benchmark tuning accordingly. The depth-3 cap is more aggressive than CatBoost's depth-6 default, so the per-tree weakness is *stronger* — compensate with more iterations and careful learning-rate/L2 tuning, and verify on interaction-heavy datasets where 3rd-order truncation could bite.
- **Failure mode to watch:** datasets with genuine >3rd-order interactions cannot be represented at all (by design). This is a hard expressiveness ceiling, not just a regularization bias — flag it clearly in your docs and consider it when benchmarking against unrestricted GBMs on synthetic high-order-interaction data.

## Sources

- Prokhorenkova, Gusev, Vorobev, Dorogush, Gulin — **"CatBoost: unbiased boosting with categorical features"**, NeurIPS 2018. https://arxiv.org/abs/1706.09516 (HTML: https://ar5iv.labs.arxiv.org/html/1706.09516) — oblivious-tree definition, Algorithm 1/2, cosine split objective, ordered TS (Eq. 4/5), ordered boosting, Table 2 benchmarks.
- CatBoost docs — Common training parameters: https://catboost.ai/docs/en/references/training-parameters/common
- CatBoost docs — Score functions (Cosine, L2, NewtonL2, NewtonCosine + formulas): https://catboost.ai/en/docs/concepts/algorithm-score-functions
- CatBoost docs — Quantization (border_count/max_bin=254, feature_border_type=GreedyLogSum): https://catboost.ai/docs/en/references/training-parameters/quantization
- CatBoost docs — Parameter tuning: https://catboost.ai/docs/en/concepts/parameter-tuning
- CatBoost docs — FAQ / boosting_type defaults: https://catboost.ai/docs/en/concepts/faq
- CatBoost docs — Model size regularization: https://catboost.ai/docs/en/references/model-size-reg
- CatBoost engineering — "CatBoost enables fast gradient boosting on decision trees using GPUs": https://catboost.ai/news/catboost-enables-fast-gradient-boosting-on-decision-trees-using-gpus
- CatBoost engineering — "Best-in-class inference and a ton of speedups": https://catboost.ai/news/best-in-class-inference-and-a-ton-of-speedups
- CatBoost source — `model.h`: https://github.com/catboost/catboost/blob/master/catboost/libs/model/model.h
- CatBoost issue #2978 — symmetric/non-symmetric model summation limitation: https://github.com/catboost/catboost/issues/2978
- "Optimization of Oblivious Decision Tree Ensembles Evaluation for CPU": https://arxiv.org/abs/2211.00391
- "Vectorization of GBDT Prediction in CatBoost for RISC-V": https://arxiv.org/abs/2405.11062
- LightGBM docs — Features: https://lightgbm.readthedocs.io/en/latest/Features.html
- Independent inference benchmark (Delivery Hero): https://deliveryhero.jobs/blog/is-catboost-faster-than-lightgbm-and-xgboost/
- Independent accuracy benchmarks: https://www.riskified.com/resources/article/boosting-comparison/ ; https://arxiv.org/pdf/2305.17094
- CatBoost benchmarks repo: https://github.com/catboost/benchmarks
- Secondary explainers: https://apxml.com/courses/mastering-gradient-boosting-algorithms/chapter-6-catboost-gradient-boosting/catboost-oblivious-trees ; https://deep-and-shallow.com/2020/02/29/the-gradient-boosters-v-catboost/ ; https://medium.com/@chimuichimu/the-secret-behind-catboosts-blazing-fast-inference-c6ee21ebc391
