# Benchmarks

Accuracy benchmarks for tri-boost against the mainstream gradient-boosting libraries.
These are **dev-only** — they are not part of the published wheel. Install the tooling
via the `bench` extra:

```bash
pip install -e ".[bench]"          # xgboost, lightgbm, catboost, pandas, pyarrow, scikit-learn
# (the tri_boost extension must be built first, e.g. `maturin develop --release`)
```

## French MTPL — frequency & severity

`python benchmarks/french_mtpl.py`

Dataset: `freMTPL2freq` (OpenML 41214) + `freMTPL2sev` (OpenML 41215), the standard French
motor third-party-liability set (~678k policies), fetched and cached via `fetch_openml`.

Two tasks, the canonical actuarial setup (mirrors scikit-learn's *"Poisson regression and
non-normal loss"* example):

| Task | Target | Weight | Loss | Metric |
|------|--------|--------|------|--------|
| Frequency | `ClaimNb / Exposure` | `Exposure` | Poisson | Exposure-weighted mean Poisson deviance + D² |
| Severity  | `ClaimAmount / ClaimNb` (claims only) | `ClaimNb` | Gamma | ClaimNb-weighted mean Gamma deviance + D² |

### Fairness rules

- **Identical features for every model:** the 5 numeric columns plus the 4 categoricals
  *ordinal-encoded* — so the comparison is of the *boosting*, not categorical handling.
  (Native-categorical handling and tri-boost's target-statistic encoding are a follow-up.)
- **Capacity-matched, untuned:** depth 3 everywhere (tri-boost is depth-3 oblivious by
  construction; CatBoost is also oblivious; LightGBM `num_leaves=8`; XGBoost `max_depth=3`),
  identical `learning_rate`, `n_estimators`, L2, `max_bin`. No per-library tuning or early
  stopping — a level, reproducible first pass, not a tuned bake-off.
- **Equal thread budget** (`TRIBOOST_BENCH_THREADS`, default 4) for all four, set before any
  OpenMP-backed import. `n_jobs=-1` over-subscribes on CPU-limited hosts and makes fit-times
  meaningless; the fixed budget keeps timings comparable.
- **CatBoost severity** uses `Tweedie(variance_power=1.9)` — it has no native Gamma loss — and
  is labelled as a proxy in the output.

tri-boost additionally reports that the fitted model stays **exactly decomposable** into
≤3rd-order fANOVA tables (`[Exact=True, N tables]`) — the property the rivals cannot offer.

### Knobs (env vars)

- `TRIBOOST_BENCH_SAMPLE=N` — subsample to `N` policies for a quick pass (`0` = full dataset).
- `TRIBOOST_BENCH_THREADS=K` — thread budget per library (default 4).

Hyperparameters (`N_ESTIMATORS`, `LEARNING_RATE`, `MAX_DEPTH`, `L2`, `MAX_BIN`) are constants at
the top of `french_mtpl.py`.
