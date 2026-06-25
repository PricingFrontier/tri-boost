"""Multi-dataset benchmark: tri-boost vs EBM / XGBoost / LightGBM / CatBoost.

A library-wide accuracy benchmark on real datasets that exercise the properties French
MTPL does NOT — genuine high-cardinality categoricals, feature interactions, and heavy
tails — across regression and binary classification. The headline comparison is against
**EBM** (interpret-ml): its interactions are order-2 only, which is exactly the gap
tri-boost's order-3 exact fANOVA tables target.

Datasets (sourced from TabArena + adjacent, all OpenML, verified):
  - Allstate_Claims_Severity (42571)  reg   heavy-tail insurance severity; cat≤326 levels
  - particulate-matter-ukair  (42207)  reg   high-card location + heavy tail + spatial
  - diamonds                  (46923)  reg   TabArena; canonical order-3 (carat×cut×color×clarity)
  - miami_housing             (46942)  reg   TabArena; heavy-tail + spatial interactions
  - Amazon_employee_access    (4135)   clf   extreme high-card (RESOURCE=7518 levels)
  - kick                      (41162)  clf   high-card + interactions; imbalanced

Caching (so nothing is re-run needlessly):
  - datasets:      benchmarks/.data_cache/<id>.joblib  (fetched + typed once)
  - rival results: benchmarks/.suite_cache.json keyed by (dataset, learner, config)
  tri-boost is NEVER result-cached — the Rust core changes between iterations, so a
  config-keyed cache would serve stale numbers; rivals/EBM are fixed pip versions and safe
  to cache. Net: iterating a tri-boost lever re-fits only tri-boost.

Run:  python benchmarks/tabarena_suite.py
      TABARENA_ONLY=diamonds,kick python benchmarks/tabarena_suite.py   (subset)
"""

from __future__ import annotations

import os

_THREADS = os.environ.get("TABARENA_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- config
N_ESTIMATORS = 400
LEARNING_RATE = 0.05
MAX_DEPTH = 3            # tri-boost is depth-3 oblivious; GBM rivals matched. EBM is order-2.
L2 = 1.0
MAX_BIN = 254
THREADS = int(_THREADS)
TEST_SIZE = 0.2
SEED = 0

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_CACHE = os.path.join(_HERE, ".data_cache")
_RESULT_CACHE = os.path.join(_HERE, ".suite_cache.json")
CACHEABLE = {"ebm", "xgboost", "lightgbm", "catboost"}  # tri-boost always re-fits

_ONLY = {s.strip() for s in os.environ.get("TABARENA_ONLY", "").split(",") if s.strip()}


@dataclass
class Dataset:
    name: str
    data_id: int
    task: str          # "regression" | "binary"
    log_target: bool = True   # regression: fit/score on log1p(y) (heavy-tail-appropriate)
    note: str = ""


DATASETS = [
    Dataset("allstate", 42571, "regression", note="insurance severity; cat≤326; heavy tail"),
    Dataset("particulate", 42207, "regression", note="high-card location + heavy tail + spatial"),
    Dataset("diamonds", 46923, "regression", note="TabArena; order-3 carat×cut×color×clarity"),
    Dataset("miami_housing", 46942, "regression", note="TabArena; heavy tail + spatial"),
    Dataset("amazon_access", 4135, "binary", log_target=False, note="extreme high-card (7518 levels)"),
    Dataset("kick", 41162, "binary", log_target=False, note="high-card + interactions; imbalanced"),
]


# ------------------------------------------------------------------------------- data
def load_xy(spec: Dataset) -> tuple[pd.DataFrame, pd.Series]:
    os.makedirs(_DATA_CACHE, exist_ok=True)
    path = os.path.join(_DATA_CACHE, f"{spec.data_id}.joblib")
    if os.path.exists(path):
        X, y = joblib.load(path)
        return X, y
    bunch = fetch_openml(data_id=spec.data_id, as_frame=True)
    X, y = bunch.data, bunch.target
    joblib.dump((X, y), path)
    return X, y


def cat_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if str(X[c].dtype) in ("category", "object")]


@dataclass
class Prepared:
    task: str
    cats: list[str]
    nums: list[str]
    X_tr: pd.DataFrame
    X_te: pd.DataFrame
    X_tr_ord: np.ndarray   # ordinal-encoded numeric matrix (for XGBoost)
    X_te_ord: np.ndarray
    y_tr: np.ndarray       # transformed target used for FITTING
    y_te_eval: np.ndarray  # target used for SCORING (log space for reg, 0/1 for clf)
    metric_name: str = ""


def prepare(spec: Dataset, X: pd.DataFrame, y: pd.Series) -> Prepared:
    cats = cat_columns(X)
    nums = [c for c in X.columns if c not in cats]
    # Native frame: categoricals as strings (missing → explicit token); numerics as float.
    Xn = X.copy()
    for c in cats:
        Xn[c] = Xn[c].astype("object").where(~Xn[c].isna(), "__nan__").astype(str)
    for c in nums:
        Xn[c] = pd.to_numeric(Xn[c], errors="coerce").astype("float32")

    if spec.task == "regression":
        yv = pd.to_numeric(y, errors="coerce").to_numpy(np.float64)
        y_fit_full = np.log1p(np.clip(yv, 0.0, None)) if spec.log_target else yv
        strat = None
        metric = "RMSE(log1p)" if spec.log_target else "RMSE"
    else:
        classes = pd.unique(y)
        # Map to 0/1 with the minority/first class as positive deterministically.
        order = sorted(map(str, classes))
        pos = order[-1]
        y01 = (y.astype(str).to_numpy() == pos).astype(np.float64)
        y_fit_full = y01
        strat = y01
        metric = "ROC-AUC"

    idx = np.arange(len(Xn))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED, stratify=strat)
    Xtr, Xte = Xn.iloc[tr].reset_index(drop=True), Xn.iloc[te].reset_index(drop=True)

    # Ordinal encoding for XGBoost (no native categorical path here).
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    ord_tr_cat = enc.fit_transform(Xtr[cats]) if cats else np.empty((len(Xtr), 0))
    ord_te_cat = enc.transform(Xte[cats]) if cats else np.empty((len(Xte), 0))
    num_tr = Xtr[nums].to_numpy(np.float32) if nums else np.empty((len(Xtr), 0), np.float32)
    num_te = Xte[nums].to_numpy(np.float32) if nums else np.empty((len(Xte), 0), np.float32)
    X_tr_ord = np.ascontiguousarray(np.hstack([num_tr, ord_tr_cat]), dtype=np.float32)
    X_te_ord = np.ascontiguousarray(np.hstack([num_te, ord_te_cat]), dtype=np.float32)

    return Prepared(
        task=spec.task, cats=cats, nums=nums, X_tr=Xtr, X_te=Xte,
        X_tr_ord=X_tr_ord, X_te_ord=X_te_ord,
        y_tr=y_fit_full[tr], y_te_eval=y_fit_full[te], metric_name=metric,
    )


# ----------------------------------------------------------------------------- scoring
def score(prep: Prepared, pred: np.ndarray) -> float:
    if prep.task == "regression":
        return float(np.sqrt(mean_squared_error(prep.y_te_eval, pred)))
    return float(roc_auc_score(prep.y_te_eval, pred))  # pred = P(positive)


# ----------------------------------------------------------------------------- learners
@dataclass
class Learner:
    name: str
    fit_predict: Callable[[Prepared], np.ndarray]


def tri_boost_case(prep: Prepared) -> np.ndarray:
    from tri_boost import TriBoostClassifier, TriBoostRegressor
    common = dict(n_trees=N_ESTIMATORS, learning_rate=LEARNING_RATE, lambda_=L2,
                  max_bin=MAX_BIN, seed=SEED, n_jobs=THREADS,
                  categorical_features=prep.cats or None)
    if prep.task == "regression":
        m = TriBoostRegressor(objective="squared_error", **common)
        m.fit(prep.X_tr, prep.y_tr)
        return np.asarray(m.predict(prep.X_te), dtype=np.float64)
    m = TriBoostClassifier(objective="logistic", **common)
    m.fit(prep.X_tr, prep.y_tr)
    return np.asarray(m.predict_proba(prep.X_te), dtype=np.float64)[:, 1]


def ebm_case(prep: Prepared) -> np.ndarray:
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    common = dict(random_state=SEED, n_jobs=THREADS, learning_rate=LEARNING_RATE)
    if prep.task == "regression":
        m = ExplainableBoostingRegressor(**common)
        m.fit(prep.X_tr, prep.y_tr)
        return np.asarray(m.predict(prep.X_te), dtype=np.float64)
    m = ExplainableBoostingClassifier(**common)
    m.fit(prep.X_tr, prep.y_tr)
    return np.asarray(m.predict_proba(prep.X_te), dtype=np.float64)[:, 1]


def xgboost_case(prep: Prepared) -> np.ndarray:
    from xgboost import XGBClassifier, XGBRegressor
    common = dict(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH,
                  reg_lambda=L2, max_bin=MAX_BIN, tree_method="hist", random_state=SEED,
                  n_jobs=THREADS)
    if prep.task == "regression":
        m = XGBRegressor(objective="reg:squarederror", **common)
        m.fit(prep.X_tr_ord, prep.y_tr)
        return np.asarray(m.predict(prep.X_te_ord), dtype=np.float64)
    m = XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    m.fit(prep.X_tr_ord, prep.y_tr)
    return np.asarray(m.predict_proba(prep.X_te_ord), dtype=np.float64)[:, 1]


def lightgbm_case(prep: Prepared) -> np.ndarray:
    from lightgbm import LGBMClassifier, LGBMRegressor
    # LightGBM native categoricals via 'category' dtype.
    Xtr = prep.X_tr.copy()
    Xte = prep.X_te.copy()
    for c in prep.cats:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
    common = dict(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH,
                  num_leaves=2 ** MAX_DEPTH, reg_lambda=L2, max_bin=MAX_BIN, random_state=SEED,
                  n_jobs=THREADS, verbose=-1)
    if prep.task == "regression":
        m = LGBMRegressor(objective="regression", **common)
        m.fit(Xtr, prep.y_tr, categorical_feature=prep.cats or "auto")
        return np.asarray(m.predict(Xte), dtype=np.float64)
    m = LGBMClassifier(objective="binary", **common)
    m.fit(Xtr, prep.y_tr, categorical_feature=prep.cats or "auto")
    return np.asarray(m.predict_proba(Xte), dtype=np.float64)[:, 1]


def catboost_case(prep: Prepared) -> np.ndarray:
    from catboost import CatBoostClassifier, CatBoostRegressor
    common = dict(iterations=N_ESTIMATORS, learning_rate=LEARNING_RATE, depth=MAX_DEPTH,
                  l2_leaf_reg=L2, border_count=MAX_BIN, random_seed=SEED, thread_count=THREADS,
                  verbose=0, allow_writing_files=False, cat_features=prep.cats or None)
    if prep.task == "regression":
        m = CatBoostRegressor(loss_function="RMSE", **common)
        m.fit(prep.X_tr, prep.y_tr)
        return np.asarray(m.predict(prep.X_te), dtype=np.float64)
    m = CatBoostClassifier(loss_function="Logloss", **common)
    m.fit(prep.X_tr, prep.y_tr)
    return np.asarray(m.predict_proba(prep.X_te), dtype=np.float64)[:, 1]


LEARNERS = [
    Learner("tri-boost", tri_boost_case),
    Learner("ebm", ebm_case),
    Learner("xgboost", xgboost_case),
    Learner("lightgbm", lightgbm_case),
    Learner("catboost", catboost_case),
]


# ----------------------------------------------------------------------------- caching
def _load_results() -> dict:
    if os.path.exists(_RESULT_CACHE):
        try:
            with open(_RESULT_CACHE, encoding="utf-8") as fh:
                return json.load(fh)
        except (ValueError, OSError):
            return {}
    return {}


def _save_results(cache: dict) -> None:
    try:
        with open(_RESULT_CACHE, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=1)
    except OSError:
        pass


def _result_key(dataset: str, learner: str) -> str:
    return (f"{dataset}|{learner}|n={N_ESTIMATORS}|lr={LEARNING_RATE}|d={MAX_DEPTH}"
            f"|l2={L2}|bin={MAX_BIN}|seed={SEED}|test={TEST_SIZE}")


def tri_boost_exact(prep: Prepared) -> str:
    """tri-boost's differentiator: confirm the fitted model is exactly decomposable."""
    from tri_boost import TriBoostClassifier, TriBoostRegressor
    cls = TriBoostRegressor if prep.task == "regression" else TriBoostClassifier
    obj = "squared_error" if prep.task == "regression" else "logistic"
    m = cls(objective=obj, n_trees=30, max_bin=64, seed=SEED, n_jobs=THREADS,
            categorical_features=prep.cats or None)
    m.fit(prep.X_tr.iloc[:5000], prep.y_tr[:5000])
    try:
        exp = json.loads(m.tables(prep.X_te.iloc[:256], ref_measure="uniform"))
        return f"Exact={exp['mode'] == 'Exact'}, {len(exp['tables'])} tables"
    except Exception as exc:  # noqa: BLE001
        return f"tables() failed: {type(exc).__name__}"


# -------------------------------------------------------------------------------- run
def run_dataset(spec: Dataset) -> None:
    t0 = time.perf_counter()
    X, y = load_xy(spec)
    prep = prepare(spec, X, y)
    load_s = time.perf_counter() - t0
    print(f"\n=== {spec.name}  [{spec.task}]  ({spec.note}) ===")
    print(f"    {len(prep.X_tr):,} train / {len(prep.X_te):,} test · {len(prep.nums)} num + "
          f"{len(prep.cats)} cat · loaded {load_s:.1f}s · metric {prep.metric_name}")
    lower_better = prep.task == "regression"
    cache = _load_results()
    rows: list[tuple[str, float, float, str]] = []
    for learner in LEARNERS:
        key = _result_key(spec.name, learner.name)
        if learner.name in CACHEABLE and key in cache:
            e = cache[key]
            rows.append((learner.name, e["metric"], e["fit_s"], e.get("note", "") + "  (cached)"))
            continue
        try:
            t = time.perf_counter()
            pred = learner.fit_predict(prep)
            fit_s = time.perf_counter() - t
            val = score(prep, pred)
            note = ""
            if learner.name == "tri-boost":
                note = f"[{tri_boost_exact(prep)}]"
            elif learner.name == "ebm":
                note = "order-2 only"
            rows.append((learner.name, val, fit_s, note))
            if learner.name in CACHEABLE:
                cache[key] = {"metric": val, "fit_s": fit_s, "note": note}
                _save_results(cache)
        except Exception as exc:  # noqa: BLE001
            rows.append((learner.name, float("nan"), float("nan"), f"FAILED {type(exc).__name__}: {exc}"))

    best = min if lower_better else max
    finite = [r for r in rows if np.isfinite(r[1])]
    best_val = best(r[1] for r in finite) if finite else None
    print(f"    {'model':<12}{prep.metric_name:>14}{'fit (s)':>10}   notes")
    for name, val, fit_s, note in rows:
        star = " *" if best_val is not None and val == best_val else "  "
        vs = f"{val:.5f}" if np.isfinite(val) else "FAILED"
        fs = f"{fit_s:.1f}" if np.isfinite(fit_s) else "—"
        print(f"  {star}{name:<12}{vs:>14}{fs:>10}   {note}")


def main() -> None:
    print("Multi-dataset benchmark — tri-boost vs EBM / XGBoost / LightGBM / CatBoost")
    print(f"config: n_estimators={N_ESTIMATORS} lr={LEARNING_RATE} depth={MAX_DEPTH} "
          f"max_bin={MAX_BIN} seed={SEED} threads={THREADS}")
    print("(reg: lower RMSE-on-log better; clf: higher ROC-AUC better. '*' = best. "
          "EBM is order-2; tri-boost is depth-3 oblivious + EXACTLY decomposable.)")
    for spec in DATASETS:
        if _ONLY and spec.name not in _ONLY:
            continue
        try:
            run_dataset(spec)
        except Exception as exc:  # noqa: BLE001
            print(f"\n=== {spec.name} === FAILED to load/prepare: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
