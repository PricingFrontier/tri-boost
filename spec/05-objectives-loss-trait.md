## 05 — Objectives & the Loss trait

> Owner: §05 — the `Loss` trait + `Link`, the v1 implementors `SquaredError`, `Logistic`, `Poisson`, `Gamma`, `Tweedie { rho }`, `GradHess` production, `max_delta_step`, the hessian floor ε, the per-objective deviance metric, the `exp(k·F)` power rule, and the `Metric` enum plus the `default_metric`/`hessian_floor`/`max_delta_step` trait methods (registered §05 in skeleton §4). Consumes `FitSpec`/`BinnedMatrix`/exposure offset (§03). It owns **no** tree shape: the objective is a function of `(y, F, w)` only — it never sees a split — which is exactly why every loss composes natively with the depth-3 oblivious cage (I1/I2 untouched).

### 05.1 Decisions (with defaults)

1. **One-pass `grad_hess` w.r.t. the raw score `F`.** `g = ∂L/∂F` and `h = ∂²L/∂F²` computed **together** in one pass into a caller-owned `GradHess` (§2.3). `F` is the *total* raw score with the exposure offset already folded in (`F = offset + f0 + Σ alpha_t tree_t`); the loss never sees the offset separately. Matches the XGBoost/LightGBM/sklearn-HGBT convention every formula below assumes. The method returns `Result<(), PbError>` (§2.4): the math is total, but a fallible `PyLoss` (§12) needs to surface a typed error without panicking, so the one engine call site already absorbs a `?`.
   **[R-LOSSFALLIBLE] All three of `grad_hess`, `init_score`, and `deviance` are fallible** — `grad_hess -> Result<(), PbError>`, `init_score -> Result<f64, PbError>`, `deviance -> Result<f32, PbError>`. The intercept and the metric are NOT total: a target that violates an objective's domain (Gamma `y≤0`, all-zero weights, an all-zero-positive logistic column, bad/zero exposure) or a user-supplied `PyLoss` failure must surface a typed `PbError` rather than panic or silently produce `NaN`/`±inf` (05.3a enumerates the per-objective domains). Callers propagate with `?`: §06 at `init` (the `f0` seed) and §08/§13 at the metric reduction.
2. **Mandatory `boost_from_average` intercept.** `init_score = link(weighted mean of y)`, computed **in link space** (never `log(mean(y))`, never a separate "tree 0"). This is `Model.f0` and, downstream, the fANOVA intercept / base rate. A per-row offset does **not** suppress it — offset and `f0` are additive on the raw scale (05.5).
3. **Five v1 objectives, three links.** `SquaredError`/Identity, `Logistic`/Logit, `Poisson`/`Gamma`/`Tweedie{rho}`/Log. `Tweedie.rho` default `1.5`, validated `∈ (1.0, 2.0)` exclusive. Quantile/L1 is a flagged v1.5 candidate (05.10).
4. **Powers via `exp(k·F)`, never `powf(mu, k)`.** Log-link derivatives are sums of `exp(k·F)` terms — faster, no `mu` round-trip per power, better-behaved.
5. **Hessian floor ε.** Logistic/Tweedie/Gamma can drive `h → 0` at corners; every loss applies `h ← max(h, eps)` before returning, `eps = 1e-16_f32`. SquaredError (`h≡1`) and Poisson (`h=mu>0`) never bind it but share the one code path.
6. **`max_delta_step` cap (Poisson default `0.7`, others `None`).** `Option<f32>` from the trait; the engine clamps the leaf Newton step. **The leaf-stage `|w*|`-clamp is the default** (05.6); hessian-inflation is rejected because it would perturb the quantized `QuantGradHess` histograms. The §06 `Config.max_delta_step: Option<f32>` defaults `None` and, when `None`, falls back to `Loss::max_delta_step()` (Poisson ⇒ `Some(0.7)`).
7. **Per-objective deviance as the early-stop metric — never RMSE on log-link.** `deviance` returns the **strictly-proper** unit-deviance sum wrapped fallibly (`Result<f32, PbError>` — 05.1 #1); SquaredError reports half-deviance (`= ½ MSE·Σw`); RMSE is a presentation transform only.
8. **`f32` core, `f64` reductions, fallible scalars.** Per-row `g`/`h` are `f32`; the two scalar reductions accumulate in `f64` with a fixed-order fold (§1). `init_score` returns `Result<f64, PbError>` — the *exact* fANOVA intercept stays full-width (it is stored into the `f64` `TableBank.f0` of §08, and §06 down-casts to the `f32` `Model.f0` at the one assignment) — protecting the intercept on large `n` without widening the hot path. `deviance` folds its `f64` accumulator and returns `Result<f32, PbError>` (the metric is reported/compared in `f32`; the fold is `f64` for order-independence). Both surface domain errors as `PbError` (05.1 #1, 05.3a) instead of `NaN`/panic.

### 05.2 The trait (verbatim from §2.4, with the section's contract)

```rust
/// A loss/objective. Fully orthogonal to tree shape (I1/I2 untouched).
/// `grad_hess` is one pass w.r.t. the raw score F (after the exposure offset).
pub trait Loss: Send + Sync {
    /// Write per-row g = dL/dF and h = d2L/dF2 into `out` (one pass).
    /// `raw` is the TOTAL raw score F (offset already folded in by the engine).
    /// `weight[i]` scales row i's (g, h); pass &[1.0; n] for unweighted.
    /// Postcondition: out.h[i] >= self.hessian_floor() for all i.
    /// Errors: PbError::ShapeMismatch on unequal lengths (the only failure for
    /// the v1 closed-form losses); a fallible PyLoss (§12) may surface others.
    fn grad_hess(&self, y: &[f32], raw: &[f32], weight: &[f32], out: &mut GradHess)
        -> Result<(), PbError>;

    /// link(weighted mean of y) — the mandatory boost_from_average intercept f0,
    /// computed in LINK space with an f64 fixed-order reduction; returned f64
    /// (the exact fANOVA intercept; §06 down-casts to the f32 `Model.f0`).
    /// `offset = Some(log e)` switches frequency objectives to the exposure form.
    /// Errors: PbError::InvalidInput on a domain violation that makes f0 undefined
    /// (Gamma/log-link weighted mean of y <= 0, all-zero weights, all-zero or
    /// negative exposure); PbError::ShapeMismatch on unequal lengths (05.3a).
    fn init_score(&self, y: &[f32], weight: &[f32], offset: Option<&[f32]>)
        -> Result<f64, PbError>;

    fn link(&self) -> Link;                       // Identity | Log | Logit
    fn pred_from_raw(&self, raw: f32) -> f32;     // inverse link, exp(k*F) not powf

    /// Strictly-proper unit-deviance SUM for early stopping & reporting (returned
    /// f32; the internal reduction folds f64 for order-independence — 05.1 #8).
    /// NOT RMSE on Poisson/Gamma/Tweedie.
    /// Errors: PbError::InvalidInput on the same domain violations as init_score
    /// (a y outside the objective's support cannot have a finite unit deviance);
    /// PbError::ShapeMismatch on unequal lengths (05.3a).
    fn deviance(&self, y: &[f32], raw: &[f32], weight: &[f32])
        -> Result<f32, PbError>;

    /// Hessian floor applied inside grad_hess. CANONICAL default 1e-16 (§05 owns):
    /// a NaN-guard only — numerical stability is the job of lambda + max_delta_step,
    /// NOT the floor (so it sits well below any rate that would perturb the
    /// quantized histogram). The skeleton §2.4 inherits THIS value.
    fn hessian_floor(&self) -> f32 { 1e-16 }

    /// Newton-step cap (LightGBM max_delta_step). None = uncapped.
    /// Poisson overrides to Some(0.7); see 05.6 for the engine contract.
    fn max_delta_step(&self) -> Option<f32> { None }

    /// Default early-stop / report metric for this objective.
    fn default_metric(&self) -> Metric;
}

pub enum Link { Identity, Log, Logit }            // inverse: id, exp, sigmoid

/// The early-stop / report metric an objective reports. CANONICAL (§05 owns —
/// owner-wins over any divergent skeleton listing; the skeleton §2.4 `Metric`
/// MUST match THIS verbatim). Deviance-based by default; never RMSE on a log link.
pub enum Metric { Rmse, LogLoss, PoissonDeviance, GammaDeviance, TweedieDeviance { rho: f32 } }
```

The engine guarantees equal-length slices (it builds them); `grad_hess` iterates with `izip!` — no indexing, no `unwrap` (§1) — so the body never panics. A length mismatch is the one in-method failure and is returned as `Err(PbError::ShapeMismatch { .. })` (checked once at entry, before the hot loop), satisfying the no-panic gate without widening the per-row kernel. `init_score`/`deviance` add their own one-shot domain check at entry (05.3a) and `?`-propagate; the v1 closed-form `grad_hess` never errors after the entry guard. `default_metric` lets `FitSpec` inherit the metric when unset.

**[R-TYPEDRIFT — ownership.** `Metric` and the `hessian_floor` default are **§05-canonical**; under the skeleton's owner-wins rule the skeleton §2.4 `Metric` enum and `hessian_floor` default are reconciled to match this section verbatim: `Metric = { Rmse, LogLoss, PoissonDeviance, GammaDeviance, TweedieDeviance { rho } }` and `hessian_floor() -> f32 { 1e-16 }`. The `1e-16` floor is a NaN-guard, not a stability lever (λ and `max_delta_step` carry stability); a larger floor (e.g. `1e-6`) is **rejected** because it would bias `h` enough to perturb the quantized `QuantGradHess` histograms and break bit-reproducibility (§1 [GATE]).]**

**Hot-loop policy (§1 no-panic gate).** The per-row kernel runs under a scoped `#[allow(clippy::indexing_slicing)]` on a private `fn` carrying a `// JUSTIFIED:` proof that `y.len() == raw.len() == weight.len() == out.g.len() == out.h.len()` (established by the single entry guard above), plus a boundary test at `n ∈ {0, 1}`; equivalently the kernel is written with `izip!` over the five slices so no indexing appears at all (the preferred form). Float arithmetic in the kernel is exempt from `arithmetic_side_effects` (floats saturate to ±inf, they do not panic); the only integer arithmetic on this path is the loop counter, which `overflow-checks = true` (all profiles, §13/skeleton §1) covers. No bin accumulation happens in this section — `Hist` (the `i64` bin accumulator owned by §06) is downstream of `grad_hess`.

### 05.3 Exact gradient / hessian / init per objective

All log-link rows use `mu = exp(F)`; powers are emitted as `exp(k·F)`. `p̄ = (Σ w y)/(Σ w)` is the weighted mean; `wbar = Σ w`.

| Objective | Link | `g = ∂L/∂F` | `h = ∂²L/∂F²` | `init_score f0` | Floor binds? | `max_delta_step` |
|---|---|---|---|---|---|---|
| **SquaredError** | Identity | `F − y` | `1` | `p̄` | no | None |
| **Logistic** | Logit | `σ(F) − y` | `σ(F)(1−σ(F))` | `log(p̄/(1−p̄))` | yes (saturation) | None |
| **Poisson** | Log | `exp(F) − y` | `exp(F)` | `log(p̄)` (exposure form below) | no | **Some(0.7)** |
| **Gamma** | Log | `1 − y·exp(−F)` | `y·exp(−F)` | `log(p̄)` | yes (`y→0` corner) | None |
| **Tweedie(ρ)** | Log | `−y·exp((1−ρ)F) + exp((2−ρ)F)` | `−y(1−ρ)exp((1−ρ)F) + (2−ρ)exp((2−ρ)F)` | `log(p̄)` | yes (`y=0`, small μ) | None |

These match XGBoost `regression_obj.cu` / LightGBM `regression_objective.hpp` bit-for-bit (research/04 §§1–4). Notes:

- **SquaredError** half-squared-error gives `h ≡ 1`; `g = μ − y = F − y`.
- **Logistic** uses the stable softplus form `L = softplus(F) − y·F`; `σ(F)` is the branch-stable sigmoid (`F ≥ 0 → 1/(1+e^{−F})`, else `e^{F}/(1+e^{F})`).
- **Tweedie** `h ≥ 0` for `ρ∈(1,2), y≥0` — no sign clamp; the ε floor only guards the `y=0`, `μ→0` underflow.
- **`init_score` clamps the link argument**: a *valid-but-extreme* `p̄` is floored to `eps_init = 1e-12` for Log/Logit, so e.g. an all-zero Poisson target yields a finite very-negative `f0`, not `−inf` (returned, never panicked). The clamp handles *finite-but-tiny* means; a genuinely **out-of-domain** input (a *negative* weighted mean, an *empty* reference measure, see 05.3a) is a typed `Err`, not a clamp.

### 05.3a Per-objective domain errors (`[R-LOSSFALLIBLE]`)

`init_score` and `deviance` validate the objective's domain at entry and return a typed `PbError` (never panic, never `NaN`/`±inf` into `f0` or the stop metric). The single entry check precedes any reduction; the v1 `grad_hess` itself stays infallible after its `ShapeMismatch` guard (saturation + the ε floor already make the per-row kernel total). The domains, by objective:

| Objective | Domain check (`init_score` & `deviance`) | Error on violation |
|---|---|---|
| **SquaredError** | none beyond equal lengths and `Σw > 0` | `ShapeMismatch` / `InvalidInput { "all-zero weights" }` |
| **Logistic** | `y ∈ [0,1]`; `Σw > 0`; **not all-zero positives** (`Σ w y > 0`) and not all-one (`Σ w y < Σ w`) — else `log(p̄/(1−p̄))` is `±inf` | `InvalidInput { "logistic: no positive (or no negative) class under w" }` |
| **Poisson** | `y ≥ 0`; `Σw > 0`; with exposure, `e_i > 0` and `Σ w e > 0` | `InvalidInput { "poisson: y<0 / zero weights / zero exposure" }` |
| **Gamma** | **`y > 0`** strictly (the Gamma support excludes `0`); `Σw > 0`; valid exposure | `InvalidInput { "gamma: y<=0" }` (also zero-weight / bad-exposure variants) |
| **Tweedie(ρ)** | `y ≥ 0`; `ρ ∈ (1,2)` exclusive (validated at construction); `Σw > 0`; valid exposure | `InvalidInput { "tweedie: y<0 / zero weights / bad rho or exposure" }` |
| **PyLoss (§12)** | user callback may fail for any reason | the callback's error mapped onto `PbError` (§12) |

Notes: (a) "valid exposure" means every used `e_i > 0` and `Σ w e > 0` — a zero/negative exposure makes `log(Σ w y / Σ w e)` undefined (05.5) and is rejected, not clamped. (b) The check is *weighted-aggregate*, not per-row: a single Gamma `y=0` under positive weight already makes the weighted mean valid-or-not at the aggregate, but the strict-`y>0` Gamma check is per-row at entry (a `0` is outside the support and its unit deviance is `−∞`), so Gamma rejects any in-weight `y ≤ 0`. (c) `deviance` shares the identical domain predicate (a `y` with no finite unit deviance is exactly a `y` for which `init_score` is undefined), so a fixture that fits also reports — keeping early stopping and the base rate consistent. (d) These are `PbError::InvalidInput` (a user-data problem), distinct from `PbError::InvalidConfig` (a bad `Config`/`rho`, raised at construction).

### 05.4 Numerics

- **`exp_f32` reused within a row.** At most two exponentials per log-link row (Tweedie's `exp((1−ρ)F)`, `exp((2−ρ)F)`; one shared `mu = exp(F)` for Poisson/Gamma). Terms are reused across `g` and `h` (Gamma: `t = y·exp(−F)` once → `g = 1−t`, `h = t`).
- **Overflow guard.** The loss clamps the **exponent** `k·F` to `[-30, 30]` (well inside `f32` `exp` range) so a runaway score degrades to a finite saturated `μ`, never `inf`/`NaN` into the histogram. Loss-internal, documented, boundary-tested.
- **No `powf`.** Enforced by a unit test grepping the module for `powf`/`powi` on `mu`.
- **`f64` for the two scalars** (05.1 #8): `init_score`/`deviance` fold `f32` row terms into an `f64` accumulator over fixed-size `par_chunks` combined in index order — order- and thread-count-independent (§1 [GATE]). `init_score` returns that `f64` directly (`Result<f64, PbError>`); `deviance` rounds the `f64` sum down to its returned `f32` (`Result<f32, PbError>`) — the fold width, not the return width, is what makes the reduction deterministic.

### 05.5 Exposure / offset handling

Exposure `e` enters as a **fixed unit-coefficient offset in the log link** (research/04 §2.1, §6), owned at the plumbing level by §03 (`FitSpec.exposure → offset = log(e)`), consumed here:

```
log(μ) = log(e) + F   ⟺   μ = e·exp(F)
```

The engine folds `offset` into `raw` **before** `grad_hess`, so the 05.3 formulas are *unchanged* — exposure flows in purely through `μ = exp(raw)`. Only `init_score` differs: with `offset = Some(log e)` the frequency intercept becomes the **exposure-weighted** form

```
f0 = log( Σ w_i y_i / Σ w_i e_i )
```

(accumulate `Σ w y`, `Σ w e` in `f64`, `log` the ratio, floor to `eps_init`, return the `f64`). A non-positive denominator (`Σ w e ≤ 0`, i.e. all-zero/negative exposure) or a non-positive numerator that the `eps_init` floor cannot rescue makes `f0` undefined and is returned as `Err(PbError::InvalidInput)` (05.3a), never `−inf`. This anchors the base level `e⁰ = 1.000` and makes the tables read as multiplicative relativities. Unlike XGBoost's `base_margin`, supplying an offset **does not** disable `init_score`: `f0` and the offset are both present and additive, because `f0` is the fANOVA intercept the decomposition needs, not merely a convergence seed. (Boost-on-top-of-a-GLM base-margin is a separate firewall-gated path owned by §09, exact only if the GLM backbone is itself ≤3-order additive — a hard fit-time precondition.)

### 05.6 The `max_delta_step` contract (Poisson)

Poisson's `h = exp(F)` makes the leaf step `w* = −G/(H+λ)` explosive when `F` drifts high. We adopt LightGBM's safeguard, default `δ = 0.7`, exposed as `Option<f32>`. **Default realization — the leaf-stage `|w*|`-clamp (hessian-inflation rejected, because it would perturb the quantized `QuantGradHess` accumulation and so break bit-reproducibility):** the engine clamps `|w*| ≤ δ` when computing leaf weights — applied **on full-precision aggregated sums** (§06's mandatory leaf refit), so it leaves the quantized histogram accumulation, the bit-reproducibility mechanism, untouched. The loss only *advertises* `δ` via `Loss::max_delta_step()`; it never mutates per-row `h`. The override surface is the §06 `Config.max_delta_step: Option<f32>` (default `None`); when `None` the engine falls back to `Loss::max_delta_step()`, so a default Poisson fit is stabilized at `0.7` and a non-`None` `Config` value wins.

### 05.8 How this upholds I1/I2 and serves the three aims

- **Decomposable (I1/I2):** the loss touches only `(y, F, w)`; it never references an axis, split, or tree, so swapping objectives cannot create a >3-feature coupling or a non-constant leaf — this section never approaches the firewall (§3). The `f64` `init_score` becomes the exact fANOVA intercept `f0`.
- **Predictive:** strictly-proper deviance (not RMSE) as the stop metric, exact Newton `(g,h)` feeding `w* = −G/(H+λ)`, and `max_delta_step` stabilizing Poisson all land here. Multi-step Newton leaf refit (§06) re-calls `grad_hess` at updated leaves — sharpening the non-quadratic Gamma/Tweedie leaves that are the pricing point.
- **Fast:** one-pass fused `(g,h)` with shared `exp` terms, `f32` hot path, no `powf`, autovectorizable straight-line kernels (the only branch is the stable-sigmoid sign test, handled by `multiversion`), offset folded once by the engine.

### 05.9 Testing approach

1. **Closed-form units:** `g`/`h` at hand-picked `(y, F, w)` vs the 05.3 table, `f32` tolerance.
2. **Finite-difference check (`proptest`), the primary oracle:** `|g − (L(F+δ)−L(F−δ))/2δ| < tol` and the central-difference hessian analogue, over randomized `(y, F)` in each valid domain (`y>0` Gamma, `y≥0` Poisson/Tweedie, `y∈{0,1}` Logistic).
3. **Cross-library parity:** `(g,h)` equal to ULP tolerance against reference values lifted from XGBoost/LightGBM source for fixed inputs.
4. **`init_score` optimality:** `init_score(..)?` unwrapped on a valid fixture satisfies `Σ w·g(y, f0) ≈ 0` (first-order condition) for every objective, including the exposure-weighted Poisson form.
5. **Deviance properness:** `deviance(..)?` is minimized (zero) at `μ = y` per row, non-negative; and `≠` Poisson MSE on a divergent fixture (guards against an RMSE stop-metric regression).
6. **Numerics:** saturated Logistic (`F = ±40`), `y=0` Gamma/Tweedie corners, the `±30` exponent clamp — assert finite, floored `h`, no `NaN`. The `1e-16` floor is asserted as the canonical default (§05-owned) — a regression to `1e-6` fails this test.
7. **Determinism [GATE]:** `init_score`/`deviance` (their `Ok` payloads) byte-identical at `n_threads ∈ {1,2,8}`. (Any randomized loss path would re-seed per work unit via the frozen `splitmix64`-mix of `(base, round, stage, block)` into `Pcg64::seed_from_u64` — the v1 closed-form losses draw no randomness, but the seam follows the one cross-cutting scheme.)
8. **Error/no-panic:** unequal-length slices return `Err(PbError::ShapeMismatch)` (never panic); the hot-loop boundary test at `n ∈ {0,1}` (05.2) asserts no out-of-bounds.
9a. **Domain errors `[R-LOSSFALLIBLE]`:** per 05.3a, `init_score`/`deviance` return `Err(PbError::InvalidInput)` (never `NaN`/`±inf`/panic) for: Gamma with any in-weight `y ≤ 0`; all-zero weights on every objective; an all-zero-positive (or all-positive) Logistic column; non-positive exposure (`Σ w e ≤ 0`) on Poisson/Gamma/Tweedie. A `proptest` asserts the result is `Ok(finite)` on every in-domain draw and `Err` on every out-of-domain draw.
9. **No-`powf` lint test** (05.4).

### 05.10 Open fork (recommended default) & deferred robust loss

**Quantile / L1 (pinball) for robust severity** — a deferred objective. Its hessian is `0` (needs a quantile-aware leaf / line-search, not a Newton step), so it does **not** drop into the shared `w* = −G/(H+λ)` engine. **Recommended default: defer to v1.5**, shipped once §06 exposes a quantile leaf path; `init_score` = weighted median/quantile of `y`, `default_metric` = pinball. It is exactness-neutral (still a constant leaf per cell) so it stays `Exact` when added — but it is out of the five-objective v1 set above.

**Positive-hessian robust loss (Pseudo-Huber / Log-Cosh) — additive, exactness-neutral [v1.5].** Unlike pinball, these have a strictly positive smooth hessian and drop **straight into the shared Newton engine** with no leaf-path change. Pseudo-Huber `L_δ = δ²(√(1+(r/δ)²) − 1)` (`r = F − y`, Identity link) gives `g = r/√(1+(r/δ)²)`, `h = (1+(r/δ)²)^{−3/2}` (always `>0`, floor never binds); Log-Cosh gives `g = tanh(r)`, `h = sech²(r)`. Both are exactness-neutral (constant leaf per cell, `(y,F,w)`-only) and would register as additional `Loss` implementors with `default_metric = Rmse`. **Recommended default: defer to v1.5** behind the same robustness milestone as pinball; recorded here so the deferral is explicit, not an omission.

**Re-anchoring note (owned by §09).** Global-mean / bias re-anchoring — folding `δ = log(Σwy / Σwμ̂)` into `f0` after fitting — is an exact, calibration-adjacent fold that composes with this section's `init_score`/`f0` contract (it only adjusts the scalar intercept, never `(g,h)` or tree shape). It is **owned by §09**; §05 simply guarantees `f0` is the single mutable anchor it acts on.
