# pattern-boost — Project Aim

> Status: **pre-spec / vision**. This document states *what* we are building and *why*. It deliberately does **not** specify APIs, algorithms, or file layouts — that is the job of the spec, which comes later. The research backing every claim here lives in [`research/`](research/).

---

## In one sentence

**pattern-boost is a gradient boosting machine that is completely explainable by construction** — it builds **depth-3 symmetric (oblivious) trees** so the trained ensemble decomposes *losslessly* into a small set of lookup tables (an *exact* functional-ANOVA decomposition, not an approximation) — **while matching the speed and accuracy of XGBoost, LightGBM, and CatBoost** by taking the best engineering from each.

---

## The core idea

- Every tree is a **depth-3 oblivious / symmetric tree**: one `(feature, threshold)` split per level, shared across all nodes on that level (CatBoost-style). So each tree touches **at most 3 features** and *is* an 8-cell lookup table indexed by 3 binary tests.
- Because each tree depends on ≤3 features, the **whole ensemble is a sum of ≤3-feature functions**. Its **functional ANOVA (fANOVA) decomposition therefore terminates *exactly* at 3rd order** — there are provably no higher-order terms to approximate or discard.
- So the model can be rewritten, **with zero loss**, as:

  ```
  F(x) = f₀ + Σᵢ fᵢ(xᵢ) + Σᵢⱼ fᵢⱼ(xᵢ,xⱼ) + Σᵢⱼₖ fᵢⱼₖ(xᵢ,xⱼ,xₖ)      ← terminates, exactly
  ```

  a constant + 1-D main-effect tables + 2-D pairwise tables + 3-D triple tables. Each table is human-readable and auditable; prediction is a sum of table lookups that equals the tree ensemble **bit-for-bit**.

This is the load-bearing fact of the whole project. The depth-3 oblivious constraint is usually treated only as a regularizer / inference-speed trick (that is how CatBoost uses it). **pattern-boost treats it as the mechanism for exact explainability** — it is the unique growth policy under which a boosted ensemble has a *finite, exact* decomposition into tables — and optimizes the entire methodology around that property *without paying for it in speed or accuracy*.

---

## Why this is worth building

Explainable boosting is, today, a trade-off:

- **EBM / GA2M** (InterpretML) are intelligible but cap interactions at **pairwise (2nd order)**.
- **XGBoost / LightGBM / CatBoost** are accurate but are black boxes that need post-hoc tools (SHAP, PDP/ICE) to interpret — tools that are themselves approximate and can disagree.

pattern-boost aims for **GBM accuracy with GAM-grade interpretability, extended to exact 3rd-order interactions** — and it gets there by *construction*, not by post-hoc explanation.

For **insurance / actuarial pricing** in particular, the table decomposition *is* a deployable **rating-table structure**: with a log link, leaf values exponentiate to multiplicative **relativities**, the base score maps to the base rate, and each oblivious tree reads as a small, auditable interaction-rating cell. It directly supports the **Poisson / Gamma / Tweedie** objectives, **exposure offsets**, and **monotonicity / interaction constraints** that pricing demands — and meets the explainability expectations of GDPR, the EU AI Act, NAIC and the actuarial standards.

---

## Best of three libraries — what we take, and why

The differentiator is the explainable structure. The *bar* is that it must not cost us speed or accuracy. We hit that bar by composing the proven engineering of the three incumbents rather than reinventing it.

### From CatBoost — the structural foundation (speed + the explainability backbone)
- **Oblivious / symmetric trees.** The foundation. Simultaneously (a) gives the ≤3-feature → exact-fANOVA property, and (b) gives fast, branch-free, SIMD- and cache-friendly inference via 3-bit leaf indexing into an 8-value table. *No other growth policy gives us the explainability property.*
- **Ordered target statistics** for categoricals — the single biggest source of CatBoost's accuracy edge on categorical data, and **orthogonal** to the tree structure (so we can adopt it independently). *Accuracy.*
- *(We deliberately skip CatBoost's ordered **boosting** at first — it is separable, ~1.7× the training cost, and mainly helps small data. CatBoost itself defaults to plain boosting on CPU.)*

### From XGBoost — the accuracy-defining math and trust controls
- **Second-order (Newton) split gain** and **exact leaf weights** `w* = −G/(H+λ)` — the standard that delivers accuracy parity and exact leaf values (rather than CatBoost's default cosine score). *Accuracy.*
- **Monotonic constraints** (split rejection + bound propagation) and **interaction constraints** — essential for pricing economics and for keeping the exported tables readable. *Accuracy + trust.*
- **Stable, documented model serialization** (a JSON-style contract, never pickle). *Robustness.*
- **Sparsity-aware missing-value handling** (learned default direction). *Accuracy + robustness.*

### From LightGBM — the speed engine
- **Histogram-based split finding** with up-front pre-binning — the core speed win; after binning, the per-level split search is *independent of row count*. *Speed.*
- **Histogram subtraction trick** (parent − smaller child). *Speed.*
- **Quantized integer gradient/hessian histograms** — less memory traffic, autovectorizable, *and* reproducible. *Speed.*
- *(GOSS / EFB are candidates for a later performance pass.)*

### Plus the engineering spine
Pure-**Rust core** (crates.io-publishable) + **Python bindings** via PyO3/maturin, **rayon** parallelism, a cache-friendly **columnar binned layout**, and an **sklearn-compatible** API — following the polars / tokenizers architecture for the dual Rust-crate + Python-wheel shape.

---

## Goals

- **Speed:** within parity of LightGBM / CatBoost / XGBoost `hist` on CPU, for both training and prediction.
- **Accuracy:** within noise of the best of the three on standard tabular benchmarks — accepting the order-3 interaction cap, which is the right bet for tabular and pricing data.
- **Explainability:** an *exact, lossless* decomposition into purified fANOVA tables; per-table variance (Sobol-style) importances; exact local additive attributions.
- **Pricing objectives supported (insurance is the *downstream* application, not the library's scope):** Poisson / Gamma / Tweedie are in scope as *losses*, with exposure offsets and monotone + interaction constraints, and the table decomposition doubles as a rating table. The broader insurance application layer — calibration, fairness, uncertainty quantification, deployment/compilation, rate-filing — is **downstream, out of the library's scope.**
- **Engineering:** a pure-Rust core that is usable standalone, plus portable Python wheels; sklearn-compatible.

## Non-goals (initially)

- **Interactions beyond 3rd order** — excluded by design. This is a deliberate, honest capacity limit, not an oversight.
- **GPU training** — deferred; we keep a backend abstraction and a GPU-friendly data layout so it *can* be added later.
- **Ordered boosting** — an optional later accuracy knob, not a v1 requirement.
- **Non-symmetric growth policies** — would break the core invariant; explicitly out of scope.

## Honest risks (named up front)

- The **order-3 cap is a real ceiling** on functions with genuine higher-order interactions. We will benchmark against unrestricted GBMs and report where it loses.
- Oblivious trees are **individually weaker** per tree → we may need more, lower-learning-rate trees → net speed/accuracy must be verified, not assumed.
- fANOVA purification requires a **reference-measure choice** (we expect to default to the empirical product-of-marginals with Laplace smoothing, and expose alternatives), and the chosen measure changes the "pure" effects.

## How we will know it worked

1. The **lossless-equivalence property** holds as a *tested invariant*: reconstructed tables == tree ensemble, to floating-point.
2. **Predictive parity with exact decomposition — the first milestone:** on the **TabArena** benchmark suite, **beat EBM / GA2M** and come within striking distance of unconstrained XGBoost / LightGBM / CatBoost, while *every* model stays exactly decomposable into ≤3rd-order tables. Competitive predictiveness *with* exact explainability is the whole thesis — this is the core to prove before anything downstream. (We use TabArena as our own yardstick; we do not ship benchmarking tooling.)
3. A trained model **exports a small, human-readable** set of tables / rating relativities that an actuary or analyst can read directly.

---

## Where the research lives

All of the supporting research — primary-source-verified, with formulas and citations — is in [`research/`](research/):

1. [`01-oblivious-symmetric-trees.md`](research/01-oblivious-symmetric-trees.md) — CatBoost-style oblivious trees: construction, the per-level split objective, why they are fast, accuracy trade-offs, separability of ordered statistics.
2. [`02-histogram-gbm-internals.md`](research/02-histogram-gbm-internals.md) — LightGBM/XGBoost histogram method: binning, gradient/hessian histograms, subtraction trick, the exact split-gain and leaf-weight math, parallelism, constraints — adapted to the oblivious case.
3. [`03-fanova-purification-explainability.md`](research/03-fanova-purification-explainability.md) — functional ANOVA, the purification algorithm, EBM/GA2M, and how to accumulate an oblivious ensemble into exact, lossless tables.
4. [`04-pricing-objectives-actuarial.md`](research/04-pricing-objectives-actuarial.md) — loss functions with exact gradients/hessians (squared error, logistic, Poisson, Gamma, Tweedie), exposure/offsets, pricing constraints, evaluation metrics, rating-table deployment, regulation.
5. [`05-rust-python-engineering.md`](research/05-rust-python-engineering.md) — Rust core + Python binding engineering: workspace layout, PyO3/maturin, NumPy/Arrow interop, rayon, SIMD, build/CI, serialization, sklearn API, GPU, and the existing Rust GBM landscape.
