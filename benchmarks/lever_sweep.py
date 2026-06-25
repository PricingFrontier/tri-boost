"""Library-wide accuracy levers, measured tri-boost-vs-tri-boost across datasets.

The diamonds probe showed the suite's fixed n=400 budget UNDER-FITS tri-boost (oblivious
trees are weaker per-tree → need more iterations). EBM already self-converges via internal
early-stopping, so the honest comparison gives tri-boost early-stopping too. This sweep asks
the real library question:

  Under early-stopping (converged regime), does each EXACTNESS-PRESERVING lever improve
  final accuracy across datasets — or only paper over the fixed-budget under-fit?

Levers (all keep ExactnessMode::Exact — leaf-value refinement / momentum on fixed structure):
  - more trees           (is it just under-fit?)
  - leaf_refine_steps     (multi-step Newton leaves: accuracy-per-tree)
  - nesterov / AGBM       (momentum acceleration — suspected divergence bug)

Reuses cached datasets from tabarena_suite (no rivals needed — this is tri vs tri).
Run:  python benchmarks/lever_sweep.py            (all cached datasets)
      LEVER_ONLY=diamonds,allstate python benchmarks/lever_sweep.py
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from tabarena_suite import DATASETS, _DATA_CACHE, load_xy, prepare, score  # noqa: E402

THREADS = 4
_ONLY = {s.strip() for s in os.environ.get("LEVER_ONLY", "").split(",") if s.strip()}

# (label, kwargs) — kwargs merged over the per-task common base.
# Internal early-stopping (validation_fraction) is BLOCKED with native cats (TS leakage),
# so we trace the convergence curve at fixed budgets instead. The question: does leaf-refine
# help the CONVERGED model (n4000) or only accelerate an under-fit one (n400)?
CONFIGS = [
    ("n400 base",      dict(n_trees=400)),
    ("n400 refine4",   dict(n_trees=400, leaf_refine_steps=4)),
    ("n1500 base",     dict(n_trees=1500)),
    ("n1500 refine4",  dict(n_trees=1500, leaf_refine_steps=4)),
    ("n4000 base",     dict(n_trees=4000)),
    ("n4000 refine4",  dict(n_trees=4000, leaf_refine_steps=4)),
]


def fit_one(prep, cfg):
    from tri_boost import TriBoostClassifier, TriBoostRegressor
    common = dict(learning_rate=0.05, lambda_=1.0, max_bin=254, seed=0, n_jobs=THREADS,
                  categorical_features=prep.cats or None)
    common.update(cfg)
    if prep.task == "regression":
        m = TriBoostRegressor(objective="squared_error", **common)
        m.fit(prep.X_tr, prep.y_tr)
        pred = np.asarray(m.predict(prep.X_te), dtype=np.float64)
    else:
        m = TriBoostClassifier(objective="logistic", **common)
        m.fit(prep.X_tr, prep.y_tr)
        pred = np.asarray(m.predict_proba(prep.X_te), dtype=np.float64)[:, 1]
    n_used = getattr(m, "n_trees_", None) or getattr(m, "_n_trees_used", None)
    return score(prep, pred), n_used


def run_dataset(spec):
    path = os.path.join(_DATA_CACHE, f"{spec.data_id}.joblib")
    if not os.path.exists(path):
        print(f"\n=== {spec.name} === (not cached yet — skipping)")
        return
    X, y = load_xy(spec)
    prep = prepare(spec, X, y)
    better = "lower" if spec.task == "regression" else "higher"
    print(f"\n=== {spec.name} [{spec.task}] · {len(prep.X_tr):,}tr/{len(prep.X_te):,}te · "
          f"{len(prep.nums)}num+{len(prep.cats)}cat · {prep.metric_name} ({better}=better) ===")
    results = []
    for label, cfg in CONFIGS:
        try:
            t = time.perf_counter()
            val, n_used = fit_one(prep, cfg)
            dt = time.perf_counter() - t
            results.append((label, val, dt, n_used))
            ntxt = f" stopped@{n_used}" if n_used else ""
            print(f"  {label:<22} {val:.5f}   {dt:6.1f}s{ntxt}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:<22} FAILED {type(exc).__name__}: {exc}")
    # converged comparison: does refine4 help or hurt at the largest budget (n4000)?
    by = {r[0]: r[1] for r in results}
    if "n4000 base" in by and "n4000 refine4" in by:
        b, r = by["n4000 base"], by["n4000 refine4"]
        sign = "+" if (r < b) == (spec.task == "regression") else "-"
        print(f"  -> converged (n4000): base {b:.5f} vs refine4 {r:.5f}  "
              f"[refine {'HELPS' if sign == '+' else 'HURTS'} converged]")


def main():
    print("Lever sweep — tri-boost vs tri-boost (exactness-preserving accuracy levers)")
    print("lr=0.05 lambda=1.0 max_bin=254 seed=0 · ES=early-stop(50,val=0.1)")
    for spec in DATASETS:
        if _ONLY and spec.name not in _ONLY:
            continue
        try:
            run_dataset(spec)
        except Exception as exc:  # noqa: BLE001
            print(f"\n=== {spec.name} === FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
