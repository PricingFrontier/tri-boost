# Benchmarks

Accuracy benchmarks for tri-boost against the mainstream gradient-boosting libraries.
These are **dev-only** — they are not part of the published wheel. Install the data
tooling via the `bench` extra, then install rival adapters explicitly in your local
benchmark environment:

```bash
pip install -e ".[bench]"          # pandas, pyarrow, scikit-learn
pip install "xgboost>=2.0" "lightgbm>=4.0" "catboost>=1.2"
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

- **Comparable feature set, native categoricals where stable:** every model sees the same
  5 numeric columns plus the same 4 categoricals. tri-boost uses its native
  target-statistic encoding, LightGBM and CatBoost use native categoricals, and XGBoost stays
  on the shared ordinal-encoded matrix for version/API stability.
- **Capacity-matched, untuned:** depth 3 everywhere (tri-boost is depth-3 oblivious by
  construction; CatBoost is also oblivious; LightGBM `num_leaves=8`; XGBoost `max_depth=3`),
  identical `learning_rate`, `n_estimators`, L2, `max_bin`. No per-library tuning or early
  stopping — a level, reproducible first pass, not a tuned bake-off.
- **Optional tuned exact tri-boost row:** enabled by default as `tri-boost tuned`. It keeps the
  same exact depth-3 decomposable structure while enabling conservative tri-boost-only knobs
  (`reanchor`, K-fold TS for frequency, log-mean TS for severity). Disable with
  `TRIBOOST_BENCH_TUNED=0` when you want only the untuned capacity-matched row.
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
- `TRIBOOST_BENCH_TUNED=0` — hide the extra exact-safe tuned tri-boost row.

Hyperparameters (`N_ESTIMATORS`, `LEARNING_RATE`, `MAX_DEPTH`, `L2`, `MAX_BIN`) are constants at
the top of `french_mtpl.py`.
