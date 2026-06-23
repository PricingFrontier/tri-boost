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
  * Identical preprocessed features for every model (numeric + ORDINAL-encoded categoricals),
    so this compares the boosting, not the categorical handling. (Native-categorical and
    tri-boost's target-statistic encoding are a documented follow-up.)
  * Capacity-matched, UNTUNED hyperparameters: depth 3 everywhere (tri-boost is a depth-3
    oblivious GBM by construction; CatBoost is also oblivious; LightGBM gets num_leaves=8;
    XGBoost gets max_depth=3), same learning_rate / n_estimators / L2 / max_bin. No per-library
    tuning or early stopping — a level, reproducible first pass, not a tuned bake-off.
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

NUMERIC = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CATEGORICAL = ["VehBrand", "VehGas", "Area", "Region"]
FEATURES = NUMERIC + CATEGORICAL


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


def encode(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Identical feature matrix for every model: numeric + ordinal-encoded categoricals."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train[CATEGORICAL])

    def to_matrix(frame: pd.DataFrame) -> np.ndarray:
        num = frame[NUMERIC].to_numpy(dtype=np.float32)
        cat = enc.transform(frame[CATEGORICAL]).astype(np.float32)
        return np.ascontiguousarray(np.hstack([num, cat]), dtype=np.float32)

    return to_matrix(train), to_matrix(test)


@dataclass
class Task:
    name: str
    power: float            # Tweedie power: 1 = Poisson, 2 = Gamma
    X_tr: np.ndarray
    X_te: np.ndarray
    y_tr: np.ndarray
    y_te: np.ndarray
    w_tr: np.ndarray
    w_te: np.ndarray


def make_tasks(df: pd.DataFrame) -> list[Task]:
    tr, te = train_test_split(df, test_size=TEST_SIZE, random_state=SEED)
    Xtr, Xte = encode(tr, te)

    # Frequency: target = ClaimNb / Exposure, weight = Exposure.
    freq = Task(
        name="frequency (Poisson)",
        power=1.0,
        X_tr=Xtr, X_te=Xte,
        y_tr=(tr["ClaimNb"] / tr["Exposure"]).to_numpy(np.float64),
        y_te=(te["ClaimNb"] / te["Exposure"]).to_numpy(np.float64),
        w_tr=tr["Exposure"].to_numpy(np.float64),
        w_te=te["Exposure"].to_numpy(np.float64),
    )

    # Severity: policies with claims, target = ClaimAmount / ClaimNb, weight = ClaimNb.
    def sev_task(frame: pd.DataFrame, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = (frame["ClaimNb"].to_numpy() > 0) & (frame["ClaimAmount"].to_numpy() > 0)
        y = (frame["ClaimAmount"].to_numpy()[mask] / frame["ClaimNb"].to_numpy()[mask])
        w = frame["ClaimNb"].to_numpy(np.float64)[mask]
        return X[mask], y.astype(np.float64), w

    Xtr_s, ytr_s, wtr_s = sev_task(tr, Xtr)
    Xte_s, yte_s, wte_s = sev_task(te, Xte)
    sev = Task(
        name="severity (Gamma)",
        power=2.0,
        X_tr=Xtr_s, X_te=Xte_s,
        y_tr=ytr_s, y_te=yte_s, w_tr=wtr_s, w_te=wte_s,
    )
    return [freq, sev]


# ----------------------------------------------------------------------------- models
def model_builders(power: float) -> dict[str, Callable[[], Any]]:
    """name -> () -> fitted-able estimator, for the given task power (1=Poisson, 2=Gamma)."""
    is_poisson = power == 1.0

    def tri_boost() -> Any:
        from tri_boost import TriBoostRegressor
        return TriBoostRegressor(
            objective="poisson" if is_poisson else "gamma",
            n_trees=N_ESTIMATORS, learning_rate=LEARNING_RATE,
            lambda_=L2, max_bin=MAX_BIN, seed=SEED, n_jobs=THREADS,
        )

    def xgboost() -> Any:
        from xgboost import XGBRegressor
        return XGBRegressor(
            objective="count:poisson" if is_poisson else "reg:gamma",
            n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH, reg_lambda=L2, max_bin=MAX_BIN,
            tree_method="hist", random_state=SEED, n_jobs=THREADS,
        )

    def lightgbm() -> Any:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            objective="poisson" if is_poisson else "gamma",
            n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH, num_leaves=2 ** MAX_DEPTH, reg_lambda=L2,
            max_bin=MAX_BIN, random_state=SEED, n_jobs=THREADS, verbose=-1,
        )

    def catboost() -> Any:
        from catboost import CatBoostRegressor
        # CatBoost has no native Gamma; Tweedie(p=1.9) is the closest proxy.
        loss = "Poisson" if is_poisson else "Tweedie:variance_power=1.9"
        return CatBoostRegressor(
            loss_function=loss, iterations=N_ESTIMATORS, learning_rate=LEARNING_RATE,
            depth=MAX_DEPTH, l2_leaf_reg=L2, border_count=MAX_BIN,
            random_seed=SEED, verbose=0, allow_writing_files=False, thread_count=THREADS,
        )

    return {"tri-boost": tri_boost, "xgboost": xgboost, "lightgbm": lightgbm, "catboost": catboost}


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
    print(f"    {'model':<12}{'test deviance':>16}{'D² (dev. expl.)':>18}{'fit (s)':>10}   notes")
    print(f"    {'baseline':<12}{base_dev:>16.5f}{0.0:>18.4f}{0.0:>10.2f}   weighted-mean predictor")

    for name, build in model_builders(task.power).items():
        note = ""
        if name == "catboost" and task.power == 2.0:
            note = "Tweedie(p=1.9) proxy (no native Gamma)"
        try:
            est = build()
            t0 = time.perf_counter()
            est.fit(task.X_tr, task.y_tr, sample_weight=task.w_tr)
            fit_s = time.perf_counter() - t0
            pred = np.asarray(est.predict(task.X_te), dtype=np.float64).ravel()
            dev = deviance(task.power, task.y_te, pred, task.w_te)
            d2 = float(d2_tweedie_score(task.y_te, np.clip(pred, 1e-8, None),
                                        sample_weight=task.w_te, power=task.power))
            if name == "tri-boost":
                note = (note + "  " if note else "") + tri_boost_exactness(est, task.X_te[:256])
            print(f"    {name:<12}{dev:>16.5f}{d2:>18.4f}{fit_s:>10.2f}   {note}")
        except Exception as exc:  # noqa: BLE001 - benchmark robustness: never abort the table
            print(f"    {name:<12}{'FAILED':>16}{'':>18}{'':>10}   {type(exc).__name__}: {exc}")


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
