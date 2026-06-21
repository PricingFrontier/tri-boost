# pattern-boost — Research

This folder is the research record behind [`../AIM.md`](../AIM.md). It comes in two rounds, both produced by parallel research agents (2026-06-21) instructed to verify claims against **primary sources** (the CatBoost / LightGBM / XGBoost papers, docs and source code, the fANOVA / purification / EBM literature, the actuarial-pricing literature, and the PyO3 / maturin / rust-numpy / rayon docs and reference repos). Formulas and citations are preserved as returned.

- **Round 1 — the pillars (`01`–`05`):** five deep dives into the core technical foundations.
- **Round 2 — the technique inventory ([`06-techniques/`](06-techniques/)):** an exhaustive sweep of **229** techniques across all three libraries *and* the current GBM research frontier, each adversarially assessed against pattern-boost's two hard invariants (depth-3 oblivious structure + exact ≤3rd-order fANOVA tables) and prioritized v1 / v1.5 / v2 / research / skip.

These documents are **research, not decisions**. Each ends with a "Design implications for pattern-boost" section (round 1) or a per-technique verdict + priority (round 2) that proposes a direction; those proposals feed the spec but are not yet committed.

## Contents

| # | File | Pillar |
|---|------|--------|
| 1 | [`01-oblivious-symmetric-trees.md`](01-oblivious-symmetric-trees.md) | Oblivious / symmetric (CatBoost-style) trees: how they are built, the per-level split objective, why they are fast at train and inference, accuracy trade-offs vs leaf-wise/level-wise, and why ordered target statistics are separable from the tree structure. |
| 2 | [`02-histogram-gbm-internals.md`](02-histogram-gbm-internals.md) | Histogram-based GBM internals (LightGBM/XGBoost/sklearn HGBT): feature binning, gradient/hessian histograms, the histogram-subtraction trick, the exact split-gain and optimal-leaf-weight derivation, parallelism, GOSS/EFB, missing values, monotonic constraints — and how to adapt all of it to the oblivious (one-split-per-level) constraint. |
| 3 | [`03-fanova-purification-explainability.md`](03-fanova-purification-explainability.md) | Functional ANOVA decomposition, the reference-measure / identifiability question, the purification ("mass-moving") algorithm, EBM/GA2M and FAST interaction detection, and the exact pipeline for accumulating a depth-3 oblivious ensemble into lossless purified 1-D/2-D/3-D tables. |
| 4 | [`04-pricing-objectives-actuarial.md`](04-pricing-objectives-actuarial.md) | Objective functions with exact gradient/hessian formulas (squared error, logistic, Poisson, Gamma, Tweedie), exposure/offset handling, frequency-severity vs single-Tweedie, pricing constraints (monotonicity, smoothness, interaction control), evaluation metrics, the rating-table deployment, and the regulatory drivers for explainability. |
| 5 | [`05-rust-python-engineering.md`](05-rust-python-engineering.md) | Engineering a Rust core + Python-binding numerical library: workspace layout, PyO3/maturin, NumPy & Arrow interop, rayon parallelism, the realistic state of SIMD in Rust, build/CI/wheels, model serialization, sklearn-compatible API conventions, a GPU recommendation, and a survey of the existing Rust ML / GBM ecosystem. |
| 6 | [`06-techniques/`](06-techniques/) | **Technique inventory (round 2).** 229 techniques from CatBoost / LightGBM / XGBoost + the research frontier, adversarially judged against the two invariants and prioritized. Hub README has the distilled v1 adopt-set, the incompatible / breaks-exactness boundary, and the master priority table; seven detail sections (speed, accuracy, categorical, constraints, losses, uncertainty, interpretability) + a 2024–2026 gaps survey. |

## How the research maps to the aim

- **The differentiator** (exact, lossless table decomposition) rests on #1 (oblivious trees ⇒ ≤3 features/tree) and #3 (fANOVA truncates at 3rd order ⇒ purify into tables).
- **"No sacrifice in speed/accuracy"** rests on #2 (the histogram engine + the oblivious split-finder) and #5 (the Rust/parallelism/SIMD engineering).
- **"Best of CatBoost / XGBoost / LightGBM"** is sourced across #1 (CatBoost oblivious trees, ordered TS), #2 (LightGBM histograms + subtraction + quantized gradients; XGBoost Newton gain + constraints + missing values).
- **The pricing application** (and the rating-table framing of the tables) rests on #4.
