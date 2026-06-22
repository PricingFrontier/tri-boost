# Objective Functions for Explainable GBM in Insurance Pricing — Research Report

**Scope note.** `tri-boost` builds depth-3 symmetric/oblivious trees (each tree = an 8-entry lookup table indexed by 3 binary splits), so the explainable table decomposition maps directly onto a deployable rating-table structure. Throughout, the **raw score** (model output before the inverse link) is denoted `F`; the **prediction on the response scale** is `μ`; for log-link objectives `μ = exp(F)`. Gradient `g = ∂L/∂F` and hessian `h = ∂²L/∂F²` are always taken **with respect to the raw score `F`** — this is the convention every boosting library (XGBoost, LightGBM, sklearn HGBT) uses, and the one the `Loss` trait must implement. All formulas below were verified against library source code (XGBoost `regression_obj.cu`/`elementwise_metric.cu`, LightGBM `regression_objective.hpp`, sklearn `_regression.py`/`_loss`) and the actuarial literature (Wüthrich & Merz; CAS Monograph 5).

---

## 1. Squared error and logistic / binary cross-entropy

### 1.1 Squared error (Gaussian, identity link)

Per-sample loss (the "half squared error" boosting libraries actually use, so the hessian is 1):

```
L(y, F) = ½ (y − F)²        μ = F        (identity link)
g = ∂L/∂F = F − y = μ − y
h = ∂²L/∂F² = 1
```

- **Deviance equivalence:** Gaussian unit deviance is `d(y,μ) = (y−μ)²`; half-deviance = `½(y−μ)²`.
- **Init / base score:** optimal constant = weighted mean of `y`: `F₀ = (Σ wᵢ yᵢ)/(Σ wᵢ)`.
- **Default metric:** RMSE.
- XGBoost `reg:squarederror`, LightGBM `regression`/`l2`, sklearn `squared_error` all agree: `grad = predt − y`, `hess = 1`.

### 1.2 Logistic / binary cross-entropy (Bernoulli, logit link)

`p = σ(F) = 1/(1+e^{−F})`, `y ∈ {0,1}`:

```
L(y, F) = −[ y·log p + (1−y)·log(1−p) ]
        = log(1 + e^{F}) − y·F            (numerically stable softplus form)
g = ∂L/∂F = p − y = σ(F) − y
h = ∂²L/∂F² = p(1 − p) = σ(F)(1 − σ(F))
```

- **Deviance equivalence:** Bernoulli unit deviance `d(y,μ) = 2[y·log(y/μ) + (1−y)·log((1−y)/(1−μ))]`; this is "binary cross-entropy" in Wüthrich & Merz Table 4.1.
- **Init / base score:** log-odds of the weighted mean: `F₀ = log( p̄ / (1 − p̄) )`, `p̄ = (Σ wᵢ yᵢ)/(Σ wᵢ)`.
- **Hessian floor:** `p(1−p) → 0` at saturated probabilities; clamp `h ← max(h, ε)` (ε ≈ 1e-16) before the Newton step.
- **Default metric:** logloss (and/or AUC).

---

## 2. Poisson (claim frequency) — log link, exposure offset

The canonical claim-**frequency** loss. `μ = exp(F)`, `y ≥ 0`:

```
Unit deviance:   d(y,μ) = 2[ y·log(y/μ) − (y − μ) ] = 2[ y·log(y/μ) + μ − y ]   (0·log0 ≡ 0)
Boosting loss (½ deviance, μ-dependent part):  L = μ − y·F = exp(F) − y·F
g = ∂L/∂F = μ − y = exp(F) − y
h = ∂²L/∂F² = μ = exp(F)
```

- **Init / base score:** `F₀ = log( ȳ )` where `ȳ` = weighted mean of `y` (exposure-handled as below).
- **Default metric:** Poisson deviance.
- LightGBM agrees: `grad = exp(F) − y`, `hess = exp(F)`.

### 2.1 Exposure / offset handling (the load-bearing detail for frequency)

Exposure `e` (policy-years / duration) enters as a **fixed offset in the log-link** with coefficient constrained to 1 (not estimated):

```
log(μ) = log(e) + F      ⟺      μ = e · exp(F)
```

This is mathematically equivalent to modelling the **rate** `y/e` with weight `e`. In the boosting machinery the offset enters **only through `μ`**, never as a separate additive term in the derivative:

```
g = μ − y = e·exp(F) − y
h = μ     = e·exp(F)
```

Two equivalent implementation routes, both standard:
- **(A) Offset route:** carry `log(e)` as a per-row raw-score offset (LightGBM `init_score`, XGBoost `base_margin`); model `y` (counts). Init becomes `F₀ = log(Σwᵢyᵢ / Σwᵢ eᵢ)`.
- **(B) Weight + rate route:** target = `y/e`, sample weight = `e`; model the frequency rate directly.

Note LightGBM inflates the Poisson hessian to `μ·exp(poisson_max_delta_step)` (default `max_delta_step = 0.7`) as a step-size safeguard against the exponential's instability.

---

## 3. Gamma (claim severity) — log link

The canonical claim-**severity** loss (variance function `V(μ) = μ²`). `μ = exp(F)`, requires `y > 0`, `μ > 0`:

```
Unit deviance:   d(y,μ) = 2[ −log(y/μ) + y/μ − 1 ] = 2[ log(μ/y) + y/μ − 1 ]
Boosting loss (½ deviance, μ-dependent part):  L = F + y·exp(−F)
g = ∂L/∂F = 1 − y/μ = 1 − y·exp(−F)
h = ∂²L/∂F² = y/μ = y·exp(−F)
```

- **Init / base score:** `F₀ = log(ȳ)`.
- **Default metric:** Gamma deviance.
- LightGBM agrees exactly: `grad = 1 − y·exp(−F)`, `hess = y·exp(−F)`.

---

## 4. Tweedie (pure premium, p ∈ (1,2))

The compound Poisson–Gamma distribution — a point mass at 0 (no-claim policies) plus a continuous positive distribution — is *the* canonical pure-premium model. Variance function `V(μ) = μ^p`; `p` interpolates **Poisson (p→1)** and **Gamma (p→2)**.

`μ = exp(F)`, `y ≥ 0`, `ρ ≡ tweedie_variance_power ∈ (1,2)`:

```
Unit deviance (general branch, p ∉ {0,1,2}):
  d(y,μ) = 2[ y^{2−p} / ((1−p)(2−p))  −  y·μ^{1−p}/(1−p)  +  μ^{2−p}/(2−p) ]

Boosting loss (μ-dependent part — intractable normalizing constant dropped, as in XGBoost & LightGBM):
  L = −y·μ^{1−ρ}/(1−ρ) + μ^{2−ρ}/(2−ρ)        with μ = exp(F)

g = ∂L/∂F = −y·μ^{1−ρ} + μ^{2−ρ}           = −y·exp((1−ρ)F) + exp((2−ρ)F)
h = ∂²L/∂F² = −y·(1−ρ)·μ^{1−ρ} + (2−ρ)·μ^{2−ρ}  = −y·(1−ρ)·exp((1−ρ)F) + (2−ρ)·exp((2−ρ)F)
```

**XGBoost (`reg:tweedie`) and LightGBM (`tweedie`) use this *identical* gradient/hessian.** Verified verbatim from `regression_obj.cu`:
```cpp
grad = -y * expf((1 - rho) * F) + expf((2 - rho) * F);
hess = -y * (1 - rho) * exp((1 - rho) * F) + (2 - rho) * expf((2 - rho) * F);
```

- **Role of `p`:** sets the mean–variance law `Var(y) ∝ μ^p`. Insurance practice uses `p ∈ [1.5, 1.8]` (common: 1.5, 1.6, 1.67); fine-tuning `p` "is unlikely to have a very material effect" (CAS). `p` is a hyperparameter, not learned.
- **Hessian positivity:** for `1<ρ<2`, `y≥0` the Tweedie hessian is **always non-negative — no clamping needed** (still floor by ε for the `y=0`, small-`μ` corner).
- **Init / base score:** `F₀ = log(ȳ)`.
- **Limiting deviances** (use exact branches at the limits): p→1 Poisson `2(y·log(y/μ)+μ−y)`; p→2 Gamma `2(log(μ/y)+y/μ−1)`.
- **Default metric:** Tweedie deviance / `tweedie-nloglik@ρ`. **Gotcha:** XGBoost's `tweedie-nloglik` *metric* takes input on the **response scale** (`μ`), whereas the *objective* takes the **raw score** `F`. Same math, different input space.
- **Numerics:** compute non-integer powers as `exp(k·F)` directly, not `powf(μ, k)`.

---

## 5. Frequency-severity vs single Tweedie — pricing tradeoffs

Pure premium = frequency × severity. Two modelling routes:

**Single Tweedie pure-premium model** — one model, `Var ∝ μ^p`. Pros: simplest pipeline; robust; preferred under data/time constraints. Cons — the structural limitation: holding `ϕ` and `p` constant while only `μ` varies bakes in that **frequency and severity move in the same direction**; a factor with opposite-signed effects on frequency vs severity shows up as **insignificant** and "go[es] completely unnoticed."

**Separate frequency (Poisson) + severity (Gamma) models** — Pros: each effect is isolated (cuts through severity noise), counteracting effects are visible, and — critically for `tri-boost`'s explainability mission — you get **two interpretable rating tables**, one per peril component, exactly how traditional rating plans are structured. Cons: two models to build/validate/combine; needs the freq/sev data split.

**Implication for `tri-boost`:** ship **all of** Poisson, Gamma, and Tweedie so users can run either workflow. The frequency-severity route aligns best with the explainable-rating-table goal; Tweedie is the pragmatic single-model fallback.

---

## 6. Offsets / exposure and base levels in boosting

**Base score / init (the optimal single-node leaf).** Every objective's boosting init is `link(weighted mean of y)`:

| Objective | init `F₀` |
|---|---|
| Squared error | `mean(y)` |
| Logistic | `logit(mean(y)) = log(p̄/(1−p̄))` |
| Poisson / Gamma / Tweedie | `log(mean(y))` (exposure-weighted for frequency) |
| L1 / quantile | weighted **median** / quantile of `y` |

**Per-row offset / base margin / init_score.** A per-row **raw-score** offset added to the score accumulator *before* the inverse link, on every iteration:

```
F_total(xᵢ) = offsetᵢ + F₀ + Σ_t η · tree_t(xᵢ)
μᵢ = link⁻¹(F_total(xᵢ))           # e.g. exp(F_total)
```

- This is **XGBoost `base_margin`** / **LightGBM `init_score`** (both raw scale; supplying one disables auto-init).
- **Exposure** for frequency: `offsetᵢ = log(eᵢ)`. Gradient/hessian forms are unchanged — exposure flows in purely through `μ`.
- **Base levels / rating context:** an offset lets you boost *on top of* an existing (e.g. GLM) rating model, and lets the table decomposition be read as relativities relative to a chosen base level (the `e⁰ = 1.000` reference).
- Optional **`max_delta_step`** (LightGBM `poisson_max_delta_step`, default 0.7): inflate the Poisson hessian to `μ·exp(δ)` to cap the Newton step and stabilize the exponential.

---

## 7. Constraints for pricing: monotonicity, smoothness, interaction control

**Monotonicity** — premiums must move the right way with risk. Henckaerts et al. (NAAJ 2021): an unconstrained tree can "assign a lower premium to policyholders with a worse claim history… In practice, an actuary would specify monotonicity constraints." Constraints "align model behavior with economic reasoning… facilitat[ing] explainability and model governance" at typically **<0.2% AUC cost on large data** (can even *improve* generalization). A *wrong-signed* constraint costs 2.5–21.2% AUC — sign correctness matters.

- **API convention (universal):** per-feature `+1 / 0 / −1` (increasing / none / decreasing), positionally indexed; also accept a name→sign dict.
- **Enforcement:** (1) reject any split whose child weights violate the required direction (gain = −∞); (2) propagate weight bounds down the tree (children bounded by parent/sibling midpoint) so per-tree monotonicity ⟹ ensemble monotonicity. LightGBM offers tighter `intermediate`/`advanced` methods.
- **Symmetric-tree bonus:** a depth-3 oblivious tree applies the *same* split per level, so a monotone constraint is enforced once per level it appears — simpler to reason about and verify in a filed rate plan.

**Smoothness** — actuaries impose smoothness "for commercial or statistical reasons" (Richman & Wüthrich, ICEnet). Levers: binning granularity (`max_bin`), shrinkage/`monotone_penalty`, post-hoc smoothing of the extracted table.

**Interaction control** — the heart of explainability and rating-table deployability. A pure additive model (depth-1 stumps) is a GAM — gold standard for intelligibility. Depth-3 oblivious trees admit up to 3-way interactions; controlling *which* features may interact is essential because (a) regulators may require certain interactions excluded "even if they perform well," and (b) low-order, named interactions keep the table decomposition human-readable. Recommend an `interaction_constraints` API plus a `max_interaction_order` / additive (`depth-1`) mode.

---

## 8. Evaluation metrics for pricing models

Two families:

**Discrimination of risk (lift / ranking):**
- **Lift / quantile (actual-vs-predicted) plots:** sort by prediction, bucket into equal-exposure quantiles, plot bucket-mean actual vs predicted; lift = spread between extreme buckets.
- **Double lift charts:** sort by the **ratio** model-A/model-B; extreme buckets = max disagreement.
- **Gini / Lorenz curve:** sort by predicted loss cost; `Gini = 2·area between Lorenz curve and equality line`; `AUROC = 0.5·normalizedGini + 0.5`.
- **Ordered / economic Gini (Frees–Meyers–Cummings):** axis is cumulative **premium**, observations ordered by **relativity** `R = score/price`; the actuarially correct model-comparison metric (has an asymptotic difference test).

**Correctness on average (calibration):**
- **Deviance** (Poisson / Gamma / Tweedie) — the **default training metric per objective**; a strictly proper scoring rule for the mean under the right variance law (MSE is not, for these). Supports nested-model tests; AIC/BIC for selection.
- **Actual-vs-Expected (A/E):** `A/E = Σ actual / Σ (exposure × predicted rate)`; closer to 100% = better.
- **Calibration / auto-calibration:** `E[Y | μ̂(X)] = μ̂(X)`. GLMs with canonical link + intercept guarantee balance `Σμ̂ = Σy`; **GBMs/NNs minimizing deviance often violate total balance** (Denuit et al. 2021) — so `tri-boost` should report a balance/calibration check and consider an auto-calibration post-step.

Recommend shipping per-objective deviance as default eval metric, plus Gini (and ordered Gini), lift/quantile plots, and an A-vs-E / balance report.

---

## 9. GLMs, GAMs, GBMs, EBMs in pricing — and the rating-table deployment

**The common backbone** is a GAM-with-link: `g(E[y]) = β₀ + Σⱼ fⱼ(xⱼ) (+ Σ f_{jk}(xⱼ,xₖ))`.

- **GLM** (industry standard) — the linear, **log-link** special case `g(μ) = β₀ + Σβⱼxⱼ`. Log link gives a **multiplicative** tariff: `μ = e^{β₀}·∏ⱼ e^{βⱼxⱼ}`. Exponentiated coefficients **are** the rating relativities.
- **GAM** — replaces linear terms with smooth shape functions; more flexible, still interpretable.
- **GBM** — flexible, accurate, but non-additive and discontinuous; needs constraints + post-hoc tools (PDP/ICE/ALE, SHAP) to be deployable. **This is `tri-boost`'s target gap.**
- **EBM / GA2M** (InterpretML) — "a tree-based, cyclic gradient-boosting GAM with automatic interaction detection." Directly applied to insurance frequency/severity (Krùpovà et al. 2025). **The closest existing analogue to `tri-boost`'s goal** — and the bar to clear on interpretability.
- **LocalGLMnet** (Richman & Wüthrich) — interpretable-by-design NN with feature-dependent GLM coefficients; the "interpretability as a model assumption" philosophy to emulate.

**Rating-table / rating-factor deployment** (Werner & Modlin):
> "The rate variation for different risk characteristics is achieved by modifying the base rate by a series of multipliers… The variations are contained in rating tables… referred to as relativities, factors, or multipliers."
```
Total Premium = Base Rate × Territory Reltvy × Vehicle Reltvy × … × Discounts (+ additive fees)
```
Each categorical variable has a **base level** with relativity `1.000 (= e⁰)`; other levels carry `e^{coef}`. **This is exactly what a depth-3 oblivious tree exports:** an 8-entry table indexed by 3 binary conditions; with the log link, exponentiating leaf values yields multiplicative relativities, and the constant init maps onto the base rate. The table output should therefore: (1) carry the log-link so leaves exponentiate to relativities; (2) expose the base level as the `1.000` reference; (3) name the per-table splits so each oblivious tree reads as a small, auditable interaction-rating cell.

**Regulatory explainability drivers:**
- **EU GDPR** — Art. 22 (no solely-automated significant decisions; right to human intervention), Arts. 13–15 ("meaningful information about the logic involved").
- **EU AI Act (2024/1689)** — Annex III(5)(c) makes **risk assessment & pricing in life/health insurance high-risk**; transparency/data-governance/documentation/human-oversight obligations from **2 Aug 2026**.
- **US** — ASOP 12 (risk classification), ASOP 56 (Modeling), NAIC Model Bulletin on AI (Dec 2023; no "unfairly discriminatory" outcomes; interpretability + bias testing), Colorado SB21-169 (proxy/disparate-impact testing). Filed rates must not be "excessive, inadequate, or unfairly discriminatory."
- **UK** — FCA Consumer Duty (price-and-value); IFoA/RSS ethics.
- **Proxy discrimination** (Prince & Schwarcz 2020) — feature-level interpretability is the principal tool to detect proxies a model constructed. A primary commercial argument for `tri-boost`'s transparent tables.

---

## Design implications for `tri-boost`

### Recommended `Loss` trait API

Gradient and hessian computed **together** (single pass, w.r.t. raw score `F`); the optimal-constant **init** is mandatory.

```rust
pub trait Loss: Send + Sync {
    /// Per-row loss on the raw score F (for reporting/metrics). μ = link.inverse(F).
    fn loss(&self, y: &[f32], f: &[f32], weight: Option<&[f32]>) -> Vec<f32>;

    /// First & second derivatives of the loss w.r.t. the RAW score F.
    /// Computed together (one pass). Hessian floored to >= eps internally where needed.
    fn grad_hess(&self, y: &[f32], f: &[f32], weight: Option<&[f32]>) -> (Vec<f32>, Vec<f32>);

    /// Optimal single-node leaf = link(weighted mean of y) — the boosting base score.
    /// (median for L1/quantile.) Exposure-weighted for frequency objectives.
    fn init_score(&self, y: &[f32], weight: Option<&[f32]>, offset: Option<&[f32]>) -> f64;

    /// Link object: raw score F  <->  response-scale μ.  (Identity / Log / Logit.)
    fn link(&self) -> Link;

    /// Default evaluation metric for this objective (per-objective deviance, etc.).
    fn default_metric(&self) -> Metric;

    /// Optional Newton-step safeguard (LightGBM max_delta_step). Default: no cap.
    fn max_delta_step(&self) -> Option<f32> { None }
}

pub enum Link { Identity, Log, Logit }   // inverse: id, exp, sigmoid
```

The booster adds the per-row **offset** to `F` before applying `link.inverse` on every iteration, and uses `init_score` for the base term (overridden when an offset is supplied).

### Objectives to ship (v1)

| Objective | Link | `g = ∂L/∂F` | `h = ∂²L/∂F²` | init `F₀` | Default metric | Notes |
|---|---|---|---|---|---|---|
| **SquaredError** | Identity | `μ − y` | `1` | `mean(y)` | RMSE | μ=F |
| **Logistic** | Logit | `σ(F) − y` | `σ(F)(1−σ(F))` | `logit(p̄)` | LogLoss | floor h≥ε |
| **Poisson** | Log | `μ − y` | `μ` | `log(ȳ)` | Poisson dev | exposure offset = `log(e)`; opt. max_delta_step |
| **Gamma** | Log | `1 − y/μ` | `y/μ` | `log(ȳ)` | Gamma dev | y>0 |
| **Tweedie(ρ)** | Log | `−y·μ^{1−ρ}+μ^{2−ρ}` | `−y(1−ρ)μ^{1−ρ}+(2−ρ)μ^{2−ρ}` | `log(ȳ)` | Tweedie dev@ρ | ρ∈(1,2), default 1.5; hess ≥0 |

(All Log-link rows: `μ = exp(F)`; compute powers as `exp(k·F)`.) Optional v1.5: Quantile/L1 (init = median; pinball metric) for robust severity.

### Offsets / exposure surfacing
- A first-class **per-row offset** (raw-score scale); added to `F` before the inverse link each iteration; supplying it **overrides** `init_score`.
- A **convenience `exposure` argument** for frequency objectives that sets `offset = log(exposure)` — or accept a target-rate + weight pairing.
- Offsets also enable **boost-on-top-of-GLM** and anchor the table decomposition to a **base level**.

### Monotonic & interaction constraints surfacing
- **`monotone_constraints`:** per-feature `+1/0/−1`, positional vector and/or name→sign map; enforce via split rejection + bound propagation; leverage the oblivious structure for cheaper, auditable enforcement.
- **`interaction_constraints`:** feature groups permitted to co-occur on a tree path, plus an **additive mode** (`max_interaction_order = 1`).
- **Table export:** with the Log link, exponentiate leaf values to **multiplicative relativities**; expose the base level as `1.000`; label each oblivious tree by its 3 split features.

---

## Sources

**Objective/loss formulas & library implementations**
- [XGBoost `regression_obj.cu`](https://github.com/dmlc/xgboost/blob/master/src/objective/regression_obj.cu) · [`elementwise_metric.cu`](https://github.com/dmlc/xgboost/blob/master/src/metric/elementwise_metric.cu) · [Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) · [Intercept tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/intercept.html) · [PR #10298](https://github.com/dmlc/xgboost/pull/10298)
- [LightGBM `regression_objective.hpp`](https://github.com/microsoft/LightGBM/blob/master/src/objective/regression_objective.hpp) · [`objective_function.h`](https://github.com/microsoft/LightGBM/blob/master/include/LightGBM/objective_function.h) · [Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [scikit-learn `_regression.py`](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_regression.py) · [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) · [TweedieRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html) · [mean_tweedie_deviance](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-tweedie-deviance)
- [Tweedie distribution (Wikipedia)](https://en.wikipedia.org/wiki/Tweedie_distribution) · [statsmodels GLM (offset/exposure)](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html)

**Actuarial pricing, explainability, evaluation**
- [Wüthrich & Merz, *Statistical Foundations of Actuarial Learning* (Springer 2023, open access)](https://link.springer.com/book/10.1007/978-3-031-12409-9)
- [Goldburd et al., *GLMs for Insurance Rating*, CAS Monograph No. 5](https://www.casact.org/sites/default/files/2021-01/05-Goldburd-Khare-Tevet.pdf)
- [Werner & Modlin, *Basic Ratemaking* (CAS)](https://dms.umontreal.ca/~langlois/ACT3284/Basic%20Ratemaking.pdf)
- [Clark, "Alternatives to the Tweedie Distribution in Pure Premium GLM" (CAS E-Forum 2022)](https://www.casact.org/sites/default/files/2022-07/RM9_AtlernativestoTweedieDistributioninGLM.pdf)
- [Yan et al., "Applications of the Offset in P&C Predictive Modeling" (CAS)](https://www.casact.org/sites/default/files/database/forum_09wforum_yan_et_al.pdf)
- [Richman & Wüthrich, LocalGLMnet (arXiv:2107.11059)](https://arxiv.org/abs/2107.11059) · [ICEnet (arXiv:2305.08807)](https://arxiv.org/abs/2305.08807)
- [Henckaerts et al., "Boosting insights in insurance tariff plans" (arXiv:1904.10890)](https://arxiv.org/pdf/1904.10890) · [Koklev, "What's the Price of Monotonicity?" (arXiv:2512.17945)](https://arxiv.org/pdf/2512.17945)
- [Lou et al. GA2M (KDD'13)](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf) · [Nori et al., InterpretML (arXiv:1909.09223)](https://arxiv.org/abs/1909.09223) · [EBM docs](https://interpret.ml/docs/ebm.html) · [Krùpovà et al., EBM for Claim Severity & Frequency (arXiv:2503.21321)](https://arxiv.org/abs/2503.21321)
- [Frees, Meyers, Cummings, "Summarizing Insurance Scores Using a Gini Index" (JASA 2011)](https://www.tandfonline.com/doi/abs/10.1198/jasa.2011.tm10506) · [Denuit et al., "Autocalibration and Tweedie-dominance" (arXiv:2103.03635)](https://arxiv.org/abs/2103.03635)

**Trees, constraints, software design**
- [CatBoost (arXiv:1810.11363)](https://ar5iv.labs.arxiv.org/html/1810.11363) · [CatBoost training parameters](https://catboost.ai/docs/en/references/training-parameters/common)
- ["A better method to enforce monotonic constraints" (arXiv:2011.00986)](https://arxiv.org/abs/2011.00986) · [XGBoost Monotonic Constraints](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html) · [XGBoost Feature Interaction Constraints](https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html)
- [forust (Rust GBM, ObjectiveFunction trait)](https://github.com/jinlow/forust) · [perpetual (Rust GBM, ~18 objectives)](https://github.com/perpetual-ml/perpetual) · [gbdt-rs](https://github.com/mesalock-linux/gbdt-rs)

**Regulatory**
- [EU GDPR (Reg. 2016/679)](https://eur-lex.europa.eu/eli/reg/2016/679/oj) · [Wachter et al., "Why a Right to Explanation Does Not Exist" (IDPL 2017)](https://academic.oup.com/idpl/article/7/2/76/3860948) · [EU AI Act Annex III](https://artificialintelligenceact.eu/annex/3/) · [NAIC Model Bulletin on AI (Dec 2023)](https://content.naic.org/sites/default/files/cmte-h-big-data-artificial-intelligence-wg-ai-model-bulletin.pdf.pdf) · [Prince & Schwarcz, "Proxy Discrimination in the Age of AI" (Iowa Law Review 2020)](https://ilr.law.uiowa.edu/print/volume-105-issue-3)
