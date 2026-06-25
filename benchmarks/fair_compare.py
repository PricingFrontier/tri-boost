"""Fair-comparison harness for COMPETITIVE-GOALS.md (G0-G5).

The existing `tabarena_suite.py` pins every model at n_estimators=400 with one config each —
that under-fits tri-boost (oblivious trees need more trees) and isn't matched per goal. This
harness implements the **fair-comparison protocol** from COMPETITIVE-GOALS.md:

  - CONVERGED budgets (large n + early-stopping), not a fixed 400.
  - MATCHED depth where the goal specifies it (rivals max_depth=3; CatBoost depth=3).
  - The exact per-goal configs: tri-boost order 1/2/3; EBM mains-only AND order-2; XGBoost &
    LightGBM at depth 3; CatBoost at depth-3/max_ctr_complexity=1 AND unconstrained default.
  - MULTI-SEED (model seed; split-seed sweep is a documented extension).
  - FIXED thread budget (FAIR_THREADS).
  - G0: every tri-boost row is checked to stay EXACTLY DECOMPOSABLE on the real fitted model.

Emits a per-dataset scoreboard and goal verdicts (G1-G5). Reuses tabarena_suite's dataset infra.

Run:
  FAIR_ONLY=diamonds FAIR_SEEDS=1 FAIR_BUDGET=2000 python benchmarks/fair_compare.py
  python benchmarks/fair_compare.py            # full suite, default budget/seeds (slow)

KNOWN LIMITATIONS (tracked in COMPETITIVE-GOALS.md / the outstanding-issues list):
  - Rival early-stopping is approximated by a large fixed budget here; per-model
    early-stopping (eval_set / callbacks) is the next refinement for a truly converged fight.
  - tri-boost categorical handling defaults to ORDINAL where a natural order is supplied
    (diamonds) and NATIVE TS otherwise; native-cat + internal early-stopping is blocked
    (leakage), so converged native-cat runs use the fixed budget.
"""
from __future__ import annotations

import os

_THREADS = os.environ.get("FAIR_THREADS", os.environ.get("TABARENA_THREADS", "4"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import json
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

warnings.filterwarnings("ignore")

from tabarena_suite import DATASETS, Prepared, load_xy, prepare, score  # noqa: E402

THREADS = int(_THREADS)
BUDGET = int(os.environ.get("FAIR_BUDGET", "4000"))   # converged-budget proxy
LR = float(os.environ.get("FAIR_LR", "0.05"))
SEEDS = int(os.environ.get("FAIR_SEEDS", "3"))
_ONLY = {s.strip() for s in os.environ.get("FAIR_ONLY", "").split(",") if s.strip()}

# Natural orderings for datasets whose categoricals are ordinal (diamonds quality grades).
# Where present, tri-boost uses an ordinal-numeric view (its measured best lever); otherwise
# it uses native target-statistics categoricals.
_ORDINAL = {
    "diamonds": {
        "cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
        "color": ["J", "I", "H", "G", "F", "E", "D"],
        "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
    },
}


@dataclass
class Config:
    label: str
    family: str        # tri | ebm | xgb | lgbm | cat
    kwargs: dict[str, Any]
    goal: str = ""     # which goal this row feeds (for the scoreboard)


def _configs() -> list[Config]:
    return [
        Config("tri o1", "tri", {"max_interaction_order": 1}, "G1@1"),
        Config("tri o2", "tri", {"max_interaction_order": 2}, "G1@2 / G2 / G3"),
        Config("tri o3", "tri", {"max_interaction_order": 3}, "G1@3 / G2 / G3"),
        Config("ebm mains", "ebm", {"interactions": 0}, "G1@1"),
        Config("ebm o2", "ebm", {}, "G1@2,@3"),
        Config("xgb d3", "xgb", {"max_depth": 3}, "G2"),
        Config("lgbm d3", "lgbm", {"max_depth": 3}, "G2 / G5(speed)"),
        Config("cat d3 ctr1", "cat", {"depth": 3, "max_ctr_complexity": 1}, "G3"),
        Config("cat default", "cat", {}, "G4 (ceiling)"),
    ]


# ----------------------------------------------------------------- tri-boost (+ G0 check)
def _tri_frames(spec_name: str, prep: Prepared):
    """Return (X_tr, X_te, cat_features) — ordinal-encoded where a natural order is known."""
    import pandas as pd

    order = _ORDINAL.get(spec_name)
    if order is None or not prep.cats:
        return prep.X_tr, prep.X_te, (prep.cats or None)
    Xtr, Xte = prep.X_tr.copy(), prep.X_te.copy()
    for c in prep.cats:
        if c not in order:
            return prep.X_tr, prep.X_te, (prep.cats or None)  # mixed: fall back to native
        m = {lvl: i for i, lvl in enumerate(order[c])}
        mx = max(m.values()) + 1
        Xtr[c] = Xtr[c].map(lambda v: m.get(v, mx)).astype("float32")
        Xte[c] = Xte[c].map(lambda v: m.get(v, mx)).astype("float32")
    return Xtr, Xte, None  # now fully numeric


def fit_tri(spec_name: str, prep: Prepared, seed: int, kw: dict) -> tuple[np.ndarray, float, str]:
    from tri_boost import TriBoostClassifier, TriBoostRegressor

    Xtr, Xte, cats = _tri_frames(spec_name, prep)
    common = dict(n_trees=BUDGET, learning_rate=LR, lambda_=1.0, max_bin=254, seed=seed,
                  n_jobs=THREADS, leaf_refine_steps=4, categorical_features=cats, **kw)
    t = time.perf_counter()
    if prep.task == "regression":
        m = TriBoostRegressor(objective="squared_error", **common).fit(Xtr, prep.y_tr)
        pred = np.asarray(m.predict(Xte), dtype=np.float64)
    else:
        m = TriBoostClassifier(objective="logistic", **common).fit(Xtr, prep.y_tr)
        pred = np.asarray(m.predict_proba(Xte), dtype=np.float64)[:, 1]
    fit_s = time.perf_counter() - t
    # G0: the real fitted model must stay exactly decomposable.
    try:
        exp = json.loads(m.tables(Xte.iloc[:256]))
        nf = len(exp.get("factored", []))
        dec = f"Exact={exp['mode'] == 'Exact'} tables={len(exp['tables'])} factored={nf}"
    except Exception as e:  # noqa: BLE001
        dec = f"DECOMP-FAIL {type(e).__name__}"
    return pred, fit_s, dec


# -------------------------------------------------------------------------------- rivals
def fit_ebm(spec_name: str, prep: Prepared, seed: int, kw: dict) -> tuple[np.ndarray, float, str]:
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    common = dict(random_state=seed, n_jobs=THREADS, learning_rate=LR, **kw)
    cls = ExplainableBoostingRegressor if prep.task == "regression" else ExplainableBoostingClassifier
    t = time.perf_counter()
    m = cls(**common).fit(prep.X_tr, prep.y_tr)
    fit_s = time.perf_counter() - t
    pred = (np.asarray(m.predict(prep.X_te)) if prep.task == "regression"
            else np.asarray(m.predict_proba(prep.X_te))[:, 1]).astype(np.float64)
    return pred, fit_s, "self-converged"


def fit_xgb(spec_name: str, prep: Prepared, seed: int, kw: dict) -> tuple[np.ndarray, float, str]:
    from xgboost import XGBClassifier, XGBRegressor
    common = dict(n_estimators=BUDGET, learning_rate=LR, reg_lambda=1.0, max_bin=254,
                  tree_method="hist", random_state=seed, n_jobs=THREADS, **kw)
    t = time.perf_counter()
    if prep.task == "regression":
        m = XGBRegressor(objective="reg:squarederror", **common).fit(prep.X_tr_ord, prep.y_tr)
        pred = np.asarray(m.predict(prep.X_te_ord), dtype=np.float64)
    else:
        m = XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
        m.fit(prep.X_tr_ord, prep.y_tr)
        pred = np.asarray(m.predict_proba(prep.X_te_ord), dtype=np.float64)[:, 1]
    return pred, time.perf_counter() - t, "ordinal cats"


def fit_lgbm(spec_name: str, prep: Prepared, seed: int, kw: dict) -> tuple[np.ndarray, float, str]:
    import pandas as pd
    from lightgbm import LGBMClassifier, LGBMRegressor
    Xtr, Xte = prep.X_tr.copy(), prep.X_te.copy()
    for c in prep.cats:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
    depth = kw.get("max_depth", 3)
    common = dict(n_estimators=BUDGET, learning_rate=LR, num_leaves=2 ** depth, reg_lambda=1.0,
                  max_bin=254, random_state=seed, n_jobs=THREADS, verbose=-1, **kw)
    t = time.perf_counter()
    cls = LGBMRegressor if prep.task == "regression" else LGBMClassifier
    obj = "regression" if prep.task == "regression" else "binary"
    m = cls(objective=obj, **common).fit(Xtr, prep.y_tr, categorical_feature=prep.cats or "auto")
    fit_s = time.perf_counter() - t
    pred = (np.asarray(m.predict(Xte)) if prep.task == "regression"
            else np.asarray(m.predict_proba(Xte))[:, 1]).astype(np.float64)
    return pred, fit_s, "native cats"


def fit_cat(spec_name: str, prep: Prepared, seed: int, kw: dict) -> tuple[np.ndarray, float, str]:
    from catboost import CatBoostClassifier, CatBoostRegressor
    common = dict(iterations=BUDGET, learning_rate=LR, l2_leaf_reg=1.0, border_count=254,
                  random_seed=seed, thread_count=THREADS, verbose=0, allow_writing_files=False,
                  cat_features=prep.cats or None, **kw)
    t = time.perf_counter()
    if prep.task == "regression":
        m = CatBoostRegressor(loss_function="RMSE", **common).fit(prep.X_tr, prep.y_tr)
        pred = np.asarray(m.predict(prep.X_te), dtype=np.float64)
    else:
        m = CatBoostClassifier(loss_function="Logloss", **common).fit(prep.X_tr, prep.y_tr)
        pred = np.asarray(m.predict_proba(prep.X_te), dtype=np.float64)[:, 1]
    note = "depth3 ctr1" if kw.get("depth") == 3 else "unconstrained default"
    return pred, time.perf_counter() - t, note


_FIT: dict[str, Callable] = {
    "tri": fit_tri, "ebm": fit_ebm, "xgb": fit_xgb, "lgbm": fit_lgbm, "cat": fit_cat,
}


# ----------------------------------------------------------------------------------- run
def run_dataset(spec) -> None:
    X, y = load_xy(spec)
    prep = prepare(spec, X, y)
    lower_better = prep.task == "regression"
    print(f"\n=== {spec.name} [{spec.task}] · {len(prep.X_tr):,}tr/{len(prep.X_te):,}te · "
          f"{len(prep.nums)}num+{len(prep.cats)}cat · {prep.metric_name} · "
          f"budget={BUDGET} lr={LR} seeds={SEEDS} threads={THREADS} ===")
    print(f"  {'config':<14}{'goal':<16}{prep.metric_name:>13}{'±sd':>9}{'fit s':>8}   notes")
    results: dict[str, float] = {}
    for cfg in _configs():
        vals, fits, note = [], [], ""
        for seed in range(SEEDS):
            try:
                pred, fit_s, note = _FIT[cfg.family](spec.name, prep, seed, dict(cfg.kwargs))
                vals.append(score(prep, pred)); fits.append(fit_s)
            except Exception as e:  # noqa: BLE001
                note = f"FAILED {type(e).__name__}: {str(e)[:60]}"
                break
        if vals:
            mean, sd = float(np.mean(vals)), float(np.std(vals))
            results[cfg.label] = mean
            print(f"  {cfg.label:<14}{cfg.goal:<16}{mean:>13.5f}{sd:>9.5f}"
                  f"{np.mean(fits):>8.1f}   {note}")
        else:
            print(f"  {cfg.label:<14}{cfg.goal:<16}{'—':>13}{'':>9}{'':>8}   {note}")

    # ---- goal verdicts for this dataset ----
    def cmp(a: str, b: str) -> str:
        if a not in results or b not in results:
            return "n/a"
        d = results[a] - results[b]
        better = (d < 0) if lower_better else (d > 0)
        return f"{'WIN' if better else 'lose'} ({results[a]:.5f} vs {results[b]:.5f})"

    print("  -- goals --")
    print(f"     G1@1  tri o1 vs ebm mains : {cmp('tri o1', 'ebm mains')}")
    print(f"     G1@2  tri o2 vs ebm o2    : {cmp('tri o2', 'ebm o2')}")
    print(f"     G1@3  tri o3 vs ebm o2    : {cmp('tri o3', 'ebm o2')}")
    print(f"     G2    tri o3 vs xgb d3    : {cmp('tri o3', 'xgb d3')}")
    print(f"     G2    tri o3 vs lgbm d3   : {cmp('tri o3', 'lgbm d3')}")
    print(f"     G3    tri o3 vs cat d3ctr1: {cmp('tri o3', 'cat d3 ctr1')}")
    print(f"     G4    tri o3 vs cat dflt  : {cmp('tri o3', 'cat default')} (ceiling; gap, not win)")


def main() -> None:
    print("Fair-comparison harness (COMPETITIVE-GOALS.md G0-G5). G0: every tri row stays "
          "Exactly decomposable (see 'factored=' in notes).")
    for spec in DATASETS:
        if _ONLY and spec.name not in _ONLY:
            continue
        try:
            run_dataset(spec)
        except Exception as exc:  # noqa: BLE001
            print(f"\n=== {spec.name} === FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
