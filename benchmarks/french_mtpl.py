"""French MTPL frequency/severity benchmark: tri-boost vs XGBoost / LightGBM / CatBoost.

Dataset: `freMTPL2freq` (OpenML 41214) + `freMTPL2sev` (OpenML 41215) — the standard
French motor third-party-liability set (~678k policies).

Tasks (the canonical actuarial setup, mirroring scikit-learn's "Poisson regression and
non-normal loss" example):

  * Frequency — target = ClaimNb / Exposure, sample_weight = Exposure, Poisson loss.
    Scored by the (Exposure-weighted) mean Poisson deviance + D² (deviance explained).
  * Severity  — policies with claims only, target = ClaimAmount / ClaimNb,
    sample_weight = ClaimNb, Gamma loss. Scored by ClaimNb-weighted mean Gamma deviance + D².

Fairness notes:
  * tri-boost uses its native target-statistic categorical encoding via `categorical_features`.
    LightGBM and CatBoost use their native categorical paths. XGBoost is kept on the shared
    ordinal-encoded matrix for version/API stability.
  * Capacity-matched, UNTUNED hyperparameters: depth 3 everywhere (tri-boost is a depth-3
    oblivious GBM by construction; CatBoost is also oblivious; LightGBM gets num_leaves=8;
    XGBoost gets max_depth=3), same learning_rate / n_estimators / L2 / max_bin. No per-library
    tuning or early stopping — a level, reproducible first pass, not a tuned bake-off.
  * The optional "tri-boost tuned" row keeps the same exact depth-3 decomposable structure while
    enabling conservative exact-safe accuracy knobs (reanchor and categorical TS controls).
  * CatBoost has no native Gamma loss; severity uses Tweedie(variance_power=1.9) as the closest
    proxy and is labelled as such.

Run:  python benchmarks/french_mtpl.py            (full dataset)
      TRIBOOST_BENCH_SAMPLE=50000 python benchmarks/french_mtpl.py   (quick subsample)
"""

from __future__ import annotations

import os

# Equal, fixed thread budget for every library BEFORE importing any OpenMP-backed lib —
# n_jobs=-1 over-subscribes on CPU-limited/sandboxed hosts (reads host cores, not the
# cgroup quota), making timings meaningless. Set before numpy/xgboost/lightgbm import.
_THREADS = os.environ.get("TRIBOOST_BENCH_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import json
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    d2_tweedie_score,
    mean_gamma_deviance,
    mean_poisson_deviance,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# ----------------------------------------------------------------------------- config
N_ESTIMATORS = 500
LEARNING_RATE = 0.05
MAX_DEPTH = 3          # tri-boost is fixed depth-3 oblivious; rivals matched to it.
L2 = 1.0
MAX_BIN = 254          # tri-boost default; matched across rivals where supported.
THREADS = int(_THREADS)  # equal thread budget for every library (see top-of-file note).
TEST_SIZE = 0.2
SEED = 0
SAMPLE = int(os.environ.get("TRIBOOST_BENCH_SAMPLE", "0"))  # 0 = full dataset
INCLUDE_TUNED = os.environ.get("TRIBOOST_BENCH_TUNED", "1") != "0"

NUMERIC = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CATEGORICAL = ["VehBrand", "VehGas", "Area", "Region"]
FEATURES = NUMERIC + CATEGORICAL

# Rival GBM results are deterministic for a fixed config, so cache them to disk and only
# re-fit tri-boost when iterating. Delete benchmarks/.bench_cache.json to force a re-run.
_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bench_cache.json")
CACHEABLE = {"xgboost", "lightgbm", "catboost"}  # tri-boost rows always re-fit fresh


def _cache_key(name: str, task_name: str) -> str:
    return (f"{name}|{task_name}|sample={SAMPLE}|n={N_ESTIMATORS}|lr={LEARNING_RATE}"
            f"|depth={MAX_DEPTH}|l2={L2}|bin={MAX_BIN}|seed={SEED}|test={TEST_SIZE}")


def _load_cache() -> dict:
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, encoding="utf-8") as fh:
                return json.load(fh)
        except (ValueError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=1)
    except OSError:
        pass


# ------------------------------------------------------------------------------- data
def load_frames() -> pd.DataFrame:
    """Fetch + join freq/sev into one policy-level frame (cached by fetch_openml)."""
    freq = fetch_openml(data_id=41214, as_frame=True).frame.copy()
    sev = fetch_openml(data_id=41215, as_frame=True).frame.copy()
    # IDpol is the join key; normalise dtype.
    freq["IDpol"] = freq["IDpol"].astype("int64")
    sev["IDpol"] = sev["IDpol"].astype("int64")
    # Total claim amount per policy.
    sev_by_pol = sev.groupby("IDpol")["ClaimAmount"].sum().rename("ClaimAmount")
    df = freq.join(sev_by_pol, on="IDpol")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0.0)
    # Canonical cleaning (sklearn example): cap data-entry outliers.
    df["ClaimNb"] = pd.to_numeric(df["ClaimNb"]).clip(upper=4).astype("float64")
    df["Exposure"] = pd.to_numeric(df["Exposure"]).clip(upper=1.0).astype("float64")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col]).astype("float64")
    for col in CATEGORICAL:
        df[col] = df[col].astype("str")
    if SAMPLE:
        df = df.sample(n=min(SAMPLE, len(df)), random_state=SEED).reset_index(drop=True)
    return df


def encode_ordinal(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Shared numeric matrix for models run with ordinal-encoded categoricals."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train[CATEGORICAL])

    def to_matrix(frame: pd.DataFrame) -> np.ndarray:
        num = frame[NUMERIC].to_numpy(dtype=np.float32)
        cat = enc.transform(frame[CATEGORICAL]).astype(np.float32)
        return np.ascontiguousarray(np.hstack([num, cat]), dtype=np.float32)

    return to_matrix(train), to_matrix(test)


def native_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Pandas frame with numeric columns and native categorical dtypes."""
    out = frame[FEATURES].copy()
    for col in NUMERIC:
        out[col] = out[col].astype("float32")
    for col in CATEGORICAL:
        out[col] = out[col].astype("category")
    return out.reset_index(drop=True)


@dataclass
class Task:
    name: str
    power: float            # Tweedie power: 1 = Poisson, 2 = Gamma
    X_tr_ord: np.ndarray
    X_te_ord: np.ndarray
    X_tr_native: pd.DataFrame
    X_te_native: pd.DataFrame
    y_tr: np.ndarray
    y_te: np.ndarray
    w_tr: np.ndarray
    w_te: np.ndarray


def make_tasks(df: pd.DataFrame) -> list[Task]:
    tr, te = train_test_split(df, test_size=TEST_SIZE, random_state=SEED)
    Xtr_ord, Xte_ord = encode_ordinal(tr, te)
    Xtr_native, Xte_native = native_features(tr), native_features(te)

    # Frequency: target = ClaimNb / Exposure, weight = Exposure.
    freq = Task(
        name="frequency (Poisson)",
        power=1.0,
        X_tr_ord=Xtr_ord, X_te_ord=Xte_ord,
        X_tr_native=Xtr_native, X_te_native=Xte_native,
        y_tr=(tr["ClaimNb"] / tr["Exposure"]).to_numpy(np.float64),
        y_te=(te["ClaimNb"] / te["Exposure"]).to_numpy(np.float64),
        w_tr=tr["Exposure"].to_numpy(np.float64),
        w_te=te["Exposure"].to_numpy(np.float64),
    )

    # Severity: policies with claims, target = ClaimAmount / ClaimNb, weight = ClaimNb.
    def sev_task(
        frame: pd.DataFrame,
        x_ord: np.ndarray,
        x_native: pd.DataFrame,
    ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
        mask = (frame["ClaimNb"].to_numpy() > 0) & (frame["ClaimAmount"].to_numpy() > 0)
        y = (frame["ClaimAmount"].to_numpy()[mask] / frame["ClaimNb"].to_numpy()[mask])
        w = frame["ClaimNb"].to_numpy(np.float64)[mask]
        native = x_native.iloc[np.flatnonzero(mask)].reset_index(drop=True)
        return x_ord[mask], native, y.astype(np.float64), w

    Xtr_s_ord, Xtr_s_native, ytr_s, wtr_s = sev_task(tr, Xtr_ord, Xtr_native)
    Xte_s_ord, Xte_s_native, yte_s, wte_s = sev_task(te, Xte_ord, Xte_native)
    sev = Task(
        name="severity (Gamma)",
        power=2.0,
        X_tr_ord=Xtr_s_ord, X_te_ord=Xte_s_ord,
        X_tr_native=Xtr_s_native, X_te_native=Xte_s_native,
        y_tr=ytr_s, y_te=yte_s, w_tr=wtr_s, w_te=wte_s,
    )
    return [freq, sev]


# ----------------------------------------------------------------------------- models
@dataclass
class ModelCase:
    estimator: Any
    X_tr: Any
    X_te: Any
    fit_kwargs: dict[str, Any]
    note: str = ""


def model_cases(task: Task) -> dict[str, Callable[[], ModelCase]]:
    """name -> () -> estimator plus representation-specific fit inputs."""
    is_poisson = task.power == 1.0

    def tri_boost() -> ModelCase:
        from tri_boost import TriBoostRegressor
        return ModelCase(TriBoostRegressor(
            objective="poisson" if is_poisson else "gamma",
            n_trees=N_ESTIMATORS, learning_rate=LEARNING_RATE,
            lambda_=L2, max_bin=MAX_BIN, seed=SEED, n_jobs=THREADS,
            categorical_features=CATEGORICAL,
        ), task.X_tr_native, task.X_te_native, {}, "native TS categoricals")

    def tri_boost_tuned() -> ModelCase:
        from tri_boost import TriBoostRegressor
        common = dict(
            n_trees=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            max_bin=MAX_BIN,
            seed=SEED,
            n_jobs=THREADS,
            categorical_features=CATEGORICAL,
            reanchor=True,
            n_bags=4,              # OuterBag variance reduction (top accuracy lever)
            colsample_bytree=0.7,  # per-tree axis decorrelation (restricts axis SET only → Exact)
            leaf_refine_steps=2,   # multi-step Newton leaf refinement (Armijo-backtracked)
        )
        if is_poisson:
            params = dict(
                common,
                objective="poisson",
                lambda_=L2,
                cat_target="mean",
                cat_leakage="kfold",
                cat_k=5,
            )
        else:
            params = dict(
                common,
                objective="gamma",
                lambda_=L2,
                cat_target="log_mean",
            )
        return ModelCase(
            TriBoostRegressor(**params),
            task.X_tr_native,
            task.X_te_native,
            {},
            "native TS cats; OuterBag×4 + colsample + leaf-refine + reanchor",
        )

    def xgboost() -> ModelCase:
        from xgboost import XGBRegressor
        return ModelCase(XGBRegressor(
            objective="count:poisson" if is_poisson else "reg:gamma",
            n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH, reg_lambda=L2, max_bin=MAX_BIN,
            tree_method="hist", random_state=SEED, n_jobs=THREADS,
        ), task.X_tr_ord, task.X_te_ord, {}, "ordinal categoricals")

    def lightgbm() -> ModelCase:
        from lightgbm import LGBMRegressor
        return ModelCase(LGBMRegressor(
            objective="poisson" if is_poisson else "gamma",
            n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH, num_leaves=2 ** MAX_DEPTH, reg_lambda=L2,
            max_bin=MAX_BIN, random_state=SEED, n_jobs=THREADS, verbose=-1,
        ), task.X_tr_native, task.X_te_native, {"categorical_feature": CATEGORICAL},
            "native categoricals")

    def catboost() -> ModelCase:
        from catboost import CatBoostRegressor
        # CatBoost has no native Gamma; Tweedie(p=1.9) is the closest proxy.
        loss = "Poisson" if is_poisson else "Tweedie:variance_power=1.9"
        note = "native categoricals"
        if not is_poisson:
            note += "; Tweedie(p=1.9) proxy (no native Gamma)"
        return ModelCase(CatBoostRegressor(
            loss_function=loss, iterations=N_ESTIMATORS, learning_rate=LEARNING_RATE,
            depth=MAX_DEPTH, l2_leaf_reg=L2, border_count=MAX_BIN,
            random_seed=SEED, verbose=0, allow_writing_files=False, thread_count=THREADS,
        ), task.X_tr_native, task.X_te_native, {"cat_features": CATEGORICAL}, note)

    cases = {
        "tri-boost": tri_boost,
        "xgboost": xgboost,
        "lightgbm": lightgbm,
        "catboost": catboost,
    }
    if INCLUDE_TUNED:
        cases = {"tri-boost": tri_boost, "tri-boost tuned": tri_boost_tuned, **{
            "xgboost": xgboost,
            "lightgbm": lightgbm,
            "catboost": catboost,
        }}
    return cases


# ----------------------------------------------------------------------------- scoring
def deviance(power: float, y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    pred = np.clip(pred.astype(np.float64), 1e-8, None)
    if power == 1.0:
        return float(mean_poisson_deviance(y, pred, sample_weight=w))
    return float(mean_gamma_deviance(y, pred, sample_weight=w))


def run_task(task: Task) -> None:
    print(f"\n=== {task.name} ===")
    print(f"    train rows = {len(task.y_tr):,}   test rows = {len(task.y_te):,}")
    # Null baseline: predict the weighted-mean target.
    base_pred = np.full_like(task.y_te, np.average(task.y_tr, weights=task.w_tr))
    base_dev = deviance(task.power, task.y_te, base_pred, task.w_te)
    print(f"    {'model':<16}{'test deviance':>16}{'D² (dev. expl.)':>18}{'fit (s)':>10}   notes")
    print(f"    {'baseline':<16}{base_dev:>16.5f}{0.0:>18.4f}{0.0:>10.2f}   weighted-mean predictor")

    cache = _load_cache()
    for name, build in model_cases(task).items():
        key = _cache_key(name, task.name)
        if name in CACHEABLE and key in cache:
            e = cache[key]
            print(f"    {name:<16}{e['deviance']:>16.5f}{e['d2']:>18.4f}"
                  f"{e['fit_s']:>10.2f}   {e['note']}  (cached)")
            continue
        try:
            case = build()
            t0 = time.perf_counter()
            case.estimator.fit(
                case.X_tr,
                task.y_tr,
                sample_weight=task.w_tr,
                **case.fit_kwargs,
            )
            fit_s = time.perf_counter() - t0
            pred = np.asarray(case.estimator.predict(case.X_te), dtype=np.float64).ravel()
            dev = deviance(task.power, task.y_te, pred, task.w_te)
            d2 = float(d2_tweedie_score(task.y_te, np.clip(pred, 1e-8, None),
                                        sample_weight=task.w_te, power=task.power))
            note = case.note
            if name.startswith("tri-boost"):
                note = (note + "  " if note else "") + tri_boost_exactness(
                    case.estimator,
                    case.X_te.iloc[:256],
                )
            print(f"    {name:<16}{dev:>16.5f}{d2:>18.4f}{fit_s:>10.2f}   {note}")
            if name in CACHEABLE:
                cache[key] = {"deviance": dev, "d2": d2, "fit_s": fit_s, "note": case.note}
                _save_cache(cache)
        except Exception as exc:  # noqa: BLE001 - benchmark robustness: never abort the table
            print(f"    {name:<16}{'FAILED':>16}{'':>18}{'':>10}   {type(exc).__name__}: {exc}")


def tri_boost_exactness(est: Any, x_sample: np.ndarray) -> str:
    """tri-boost's differentiator: the fitted model is EXACTLY decomposable into tables."""
    try:
        export = json.loads(est.tables(x_sample, ref_measure="uniform"))
        return f"[Exact={export['mode'] == 'Exact'}, {len(export['tables'])} fANOVA tables]"
    except Exception as exc:  # noqa: BLE001
        return f"[tables() failed: {type(exc).__name__}]"


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    print("French MTPL benchmark — tri-boost vs XGBoost / LightGBM / CatBoost")
    print(f"config: n_estimators={N_ESTIMATORS} lr={LEARNING_RATE} depth={MAX_DEPTH} "
          f"L2={L2} max_bin={MAX_BIN} seed={SEED}" + (f"  SAMPLE={SAMPLE}" if SAMPLE else ""))
    t0 = time.perf_counter()
    df = load_frames()
    print(f"loaded {len(df):,} policies ({time.perf_counter() - t0:.1f}s); "
          f"{int((df['ClaimNb'] > 0).sum()):,} with claims")
    for task in make_tasks(df):
        run_task(task)
    print("\n(lower deviance = better; higher D² = better. tri-boost is constrained to "
          "depth-3 exact-decomposable trees — the rivals are not.)")


if __name__ == "__main__":
    main()
