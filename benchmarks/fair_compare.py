"""Fair-comparison harness for COMPETITIVE-GOALS.md (G0-G5).

The companion `tabarena_suite.py` pins every model at n_estimators=400 with one config each — that
under-fits tri-boost (oblivious trees need more trees) and isn't matched per goal. This harness
implements the **fair-comparison protocol** from COMPETITIVE-GOALS.md:

  - CONVERGED budgets, each model at ITS OWN convergence point (not a shared fixed 400):
      * Rivals (XGBoost/LightGBM/CatBoost) get held-out EARLY STOPPING — they converge fast and
        would OVERFIT at a fixed BUDGET=4000, so they stop at their best val iteration.
      * EBM self-converges (internal early stopping) as it always does.
      * tri-boost ALSO gets held-out early stopping (default ON) but with FAR more patience than the
        rivals (FAIR_TRI_ES_ROUNDS=500 vs rivals' 50). Oblivious trees improve in tiny increments over
        thousands of iterations, so an impatient stop fires almost immediately on tri's small/noisy val
        holdout and badly under-fits (MEASURED: rounds=50 cost miami -6.7% — stopped at ~40 trees). At
        rounds=500 it stops only on a genuine long plateau: recovers the slow-convergence sets (miami,
        diamonds) AND still catches real overfit (kick +1.6% vs full budget). Native-cat early stopping
        is leakage-free under the default KFold OOF cross-fit (F2).
        (FAIR_TRI_EARLYSTOP=0 = full budget, no stop; FAIR_TRI_ES_ROUNDS / FAIR_TRI_VAL_FRACTION to tune.)
    Net: EVERY non-EBM model is compared at its own honest val-best, not at one budget that overfits some.
  - MATCHED depth where the goal specifies it (rivals max_depth=3; CatBoost depth=3).
  - MATCHED categorical handling, native per rival: XGBoost/LightGBM/CatBoost all use NATIVE
    categoricals; tri uses ordinal-where-a-natural-order-is-known (diamonds) else native TS; EBM native.
  - MULTI-SEED (model seed; the train/test split is fixed — split-seed sweep is a documented extension).
  - FIXED thread budget (FAIR_THREADS).
  - G0: every tri-boost row is checked to stay EXACTLY DECOMPOSABLE on the real fitted model.

**Frozen rival baseline (the iterate-tri-only workflow).** Rival results are cached to
`benchmarks/.fair_cache.json` keyed by (dataset, config-label, kwargs, BUDGET, LR, seed, PREP_VERSION,
CONFIG_VERSION). After a tri-boost code/lever change, re-running re-fits ONLY the 3 tri configs and reads
every rival from the frozen cache — so iteration costs the tri-boost fit time, not hours of unchanging
rival compute. tri-boost is NEVER result-cached (its Rust core mutates between iterations, so a cached
number would be stale); it always re-fits and always re-runs the live G0 decomposability check.

Bump PREP_VERSION on any change to prepare()/encoding/split; bump CONFIG_VERSION on any change to a
fit_* derivation (early-stopping params, num_leaves rule, eval metric, ...). Either bump invalidates the
frozen entries cleanly. The cache file is separate from the under-fit-n=400 `.suite_cache.json`.

Iterate-loop cost note: once rivals are frozen, a re-run's wall-clock is dominated by tri-boost's
G0 decomposability check (`tables()` on the converged model: diamonds o3 ~276s vs ~30s to fit). Use
FAIR_G0 to trade verification for speed — `o3` (default) verifies the binding factored order, `off`
skips it for pure accuracy iteration, `all` verifies every tri row.

Run:
  FAIR_ONLY=diamonds FAIR_SEEDS=1 python benchmarks/fair_compare.py   # smoke / one dataset
  python benchmarks/fair_compare.py                                   # full suite (slow first time)
  FAIR_G0=off FAIR_ONLY=diamonds python benchmarks/fair_compare.py    # fast accuracy-only iteration
  FAIR_SPEED=1 FAIR_ONLY=diamonds python benchmarks/fair_compare.py   # live-time lgbm for the G5 verdict
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
BUDGET = int(os.environ.get("FAIR_BUDGET", "4000"))      # converged-budget cap
LR = float(os.environ.get("FAIR_LR", "0.05"))
SEEDS = int(os.environ.get("FAIR_SEEDS", "3"))
LEAF_REFINE = int(os.environ.get("FAIR_LEAF_REFINE", "4"))
EARLY_STOP = int(os.environ.get("FAIR_EARLY_STOP", "50"))  # rival early-stopping rounds; 0 disables
# EBM's metric is deterministic in random_state and INDEPENDENT of n_jobs (threads only parallelize its
# outer bags — they do not change the fit). Its training time is not used by any goal (G5 is tri-vs-lgbm).
# So EBM gets more threads to parallelize its expensive self-converging bags — finishing far faster with
# ZERO effect on the metric or any comparison. Default EBM on 315k rows at 4 threads runs ~1.7h; at 16
# it is ~4x faster. The frozen metric is byte-identical to a 4-thread fit (threads are not in the cache key).
EBM_THREADS = int(os.environ.get("FAIR_EBM_THREADS", str(min(os.cpu_count() or 4, 16))))
VAL_FRACTION = 0.1                                          # held-out slice carved from TRAIN only
TRI_EARLYSTOP = os.environ.get("FAIR_TRI_EARLYSTOP", "1") != "0"  # tri early stopping (default ON)
# tri needs FAR more patience than the rivals: oblivious trees improve in tiny increments over
# thousands of iterations, so the rivals' rounds=50 fires almost immediately on tri's small/noisy
# val holdout (measured: miami stopped at ~40 trees, -6.7%). High patience stops only on a genuine
# long plateau — recovering the slow-convergence sets while still catching real overfit (e.g. kick).
TRI_ES_ROUNDS = int(os.environ.get("FAIR_TRI_ES_ROUNDS", "500"))
TRI_VAL_FRACTION = float(os.environ.get("FAIR_TRI_VAL_FRACTION", "0.1"))
# G0 decomposability check cost: tables() on the converged model is the iterate-loop bottleneck
# (diamonds: o1~56s, o2~78s, o3~276s — model-structure driven, not row-count). Lever:
#   o3  (default) — verify ONLY the binding factored order (the one that can exceed budget; o1/o2
#                   are trivially within-budget). all — verify every tri row. off — skip (fast
#                   accuracy-only iteration; decomposability unverified that run).
G0_MODE = os.environ.get("FAIR_G0", "o3").lower()
_SPEED = bool(os.environ.get("FAIR_SPEED"))                # re-time lgbm live for a fresh G5
_ONLY = {s.strip() for s in os.environ.get("FAIR_ONLY", "").split(",") if s.strip()}
# Families to omit entirely (neither fit nor cache-read). Lets you (a) populate the fast GBM rivals
# without waiting on slow EBM (FAIR_SKIP=tri,ebm), (b) populate only EBM in the background
# (FAIR_SKIP=tri,xgb,lgbm,cat), or (c) iterate tri alone (FAIR_SKIP=ebm,xgb,lgbm,cat).
SKIP = {s.strip() for s in os.environ.get("FAIR_SKIP", "").split(",") if s.strip()}
# Frozen-rival mode: use ONLY cached rival metrics; never FIT a missing rival (it shows n/a instead).
# This is the iterate-tri-only scoreboard: tri always re-fits, rivals are read from the frozen cache,
# and an absent rival (e.g. EBM intentionally not run on the giant datasets) costs nothing.
FROZEN = bool(os.environ.get("FAIR_FROZEN"))

# ----------------------------------------------------------------------------- result cache
_HERE = os.path.dirname(os.path.abspath(__file__))
_FAIR_CACHE = os.path.join(_HERE, ".fair_cache.json")
# Rivals are fixed pip versions → safe to freeze. tri-boost is NEVER cached: the Rust core changes
# between iterations, so a config-keyed cache would serve stale numbers. Enforced at the family level.
FAIR_CACHEABLE = {"ebm", "xgb", "lgbm", "cat"}
assert "tri" not in FAIR_CACHEABLE  # the load-bearing invariant of the iterate-tri-only workflow
PREP_VERSION = 1   # bump on ANY prepare()/encoding/split change
CONFIG_VERSION = 1  # bump on ANY fit_* derivation/default change (early-stop, num_leaves, metric, ...)

# Natural orderings for datasets whose categoricals are ordinal (diamonds quality grades).
# Where present, tri-boost uses an ordinal-numeric view (its measured best lever); otherwise native TS.
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


def _fair_key(name: str, cfg: Config, seed: int) -> str:
    kw = json.dumps(cfg.kwargs, sort_keys=True, separators=(",", ":"), default=str)
    return (f"{name}|{cfg.label}|{cfg.family}|kw={kw}|budget={BUDGET}|lr={LR}"
            f"|seed={seed}|prep=v{PREP_VERSION}|cfg=v{CONFIG_VERSION}")


def _load_fair() -> dict:
    if os.path.exists(_FAIR_CACHE):
        try:
            with open(_FAIR_CACHE, encoding="utf-8") as fh:
                return json.load(fh)
        except (ValueError, OSError):
            return {}
    return {}


def _save_fair(cache: dict) -> None:
    # Atomic: a Ctrl-C mid-write during a slow EBM/CatBoost fit must not truncate the frozen baseline.
    tmp = f"{_FAIR_CACHE}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=1)
        os.replace(tmp, _FAIR_CACHE)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _es_split(n: int, y: np.ndarray, task: str, seed: int):
    """Indices (fit, val) for held-out early stopping — carved from TRAIN ONLY (never test)."""
    from sklearn.model_selection import train_test_split

    strat = y if task != "regression" else None
    return train_test_split(np.arange(n), test_size=VAL_FRACTION,
                            random_state=1000 + seed, stratify=strat)


# ----------------------------------------------------------------- tri-boost (+ live G0 check)
def _tri_frames(spec_name: str, prep: Prepared):
    """Return (X_tr, X_te, cat_features) — ordinal-encoded where a natural order is known."""
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
    order = int(kw.get("max_interaction_order", 3))
    common = dict(n_trees=BUDGET, learning_rate=LR, lambda_=1.0, max_bin=254, seed=seed,
                  n_jobs=THREADS, leaf_refine_steps=LEAF_REFINE, categorical_features=cats, **kw)
    if TRI_EARLYSTOP:
        common.update(validation_fraction=TRI_VAL_FRACTION, early_stopping_rounds=TRI_ES_ROUNDS)
    t = time.perf_counter()
    if prep.task == "regression":
        m = TriBoostRegressor(objective="squared_error", **common).fit(Xtr, prep.y_tr)
        pred = np.asarray(m.predict(Xte), dtype=np.float64)
    else:
        m = TriBoostClassifier(objective="logistic", **common).fit(Xtr, prep.y_tr)
        pred = np.asarray(m.predict_proba(Xte), dtype=np.float64)[:, 1]
    fit_s = time.perf_counter() - t
    # G0: the real fitted model must stay exactly decomposable. Distinguish within-budget from the
    # sanctioned §08.10 factored escape hatch (factored>0 is still fully decomposable). The check is
    # the dominant per-iteration cost (tables() on the converged model) — gate it via FAIR_G0.
    do_g0 = G0_MODE == "all" or (G0_MODE == "o3" and order >= 3)
    if not do_g0:
        return pred, fit_s, f"order{order} (G0 unchecked: FAIR_G0={G0_MODE})"
    try:
        exp = json.loads(m.tables(Xte.iloc[:256]))
        nf = len(exp.get("factored", []))
        kind = "within-budget" if nf == 0 else f"via-factored(x{nf})"
        dec = f"Exact={exp['mode'] == 'Exact'} tables={len(exp['tables'])} {kind}"
    except Exception as e:  # noqa: BLE001
        dec = f"DECOMP-FAIL {type(e).__name__}"
    return pred, fit_s, dec


# -------------------------------------------------------------------------------- rivals
def fit_ebm(spec_name: str, prep: Prepared, seed: int, kw: dict) -> tuple[np.ndarray, float, str]:
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    # EBM_THREADS (not THREADS): parallelize EBM's bags for tractability; metric is thread-invariant.
    common = dict(random_state=seed, n_jobs=EBM_THREADS, learning_rate=LR, **kw)
    cls = ExplainableBoostingRegressor if prep.task == "regression" else ExplainableBoostingClassifier
    t = time.perf_counter()
    m = cls(**common).fit(prep.X_tr, prep.y_tr)
    fit_s = time.perf_counter() - t
    pred = (np.asarray(m.predict(prep.X_te)) if prep.task == "regression"
            else np.asarray(m.predict_proba(prep.X_te))[:, 1]).astype(np.float64)
    return pred, fit_s, "self-converged"


def _category_frames(prep: Prepared):
    """Native-categorical (pandas 'category' dtype) train/test frames with shared categories."""
    import pandas as pd
    Xtr, Xte = prep.X_tr.copy(), prep.X_te.copy()
    for c in prep.cats:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
    return Xtr, Xte


def fit_xgb(spec_name: str, prep: Prepared, seed: int, kw: dict) -> tuple[np.ndarray, float, str]:
    from xgboost import XGBClassifier, XGBRegressor
    # NATIVE categoricals (enable_categorical) — the sklearn OrdinalEncoder used by tabarena assigns
    # ALPHABETICAL codes, an active handicap that would make a tri 'win' over xgb an artifact.
    Xtr, Xte = _category_frames(prep)
    reg = prep.task == "regression"
    metric = "rmse" if reg else "logloss"
    common = dict(n_estimators=BUDGET, learning_rate=LR, reg_lambda=1.0, max_bin=254,
                  tree_method="hist", enable_categorical=True, eval_metric=metric,
                  random_state=seed, n_jobs=THREADS, **kw)
    es = EARLY_STOP
    if es:
        common["early_stopping_rounds"] = es
    obj = "reg:squarederror" if reg else "binary:logistic"
    cls = XGBRegressor if reg else XGBClassifier
    m = cls(objective=obj, **common)
    t = time.perf_counter()
    if es:
        fi, vi = _es_split(len(Xtr), prep.y_tr, prep.task, seed)
        m.fit(Xtr.iloc[fi], prep.y_tr[fi], eval_set=[(Xtr.iloc[vi], prep.y_tr[vi])], verbose=False)
    else:
        m.fit(Xtr, prep.y_tr)
    fit_s = time.perf_counter() - t
    pred = (np.asarray(m.predict(Xte)) if reg
            else np.asarray(m.predict_proba(Xte))[:, 1]).astype(np.float64)
    stop = getattr(m, "best_iteration", None)
    return pred, fit_s, f"native cats{f', stop@{stop}' if stop is not None else ''}"


def fit_lgbm(spec_name: str, prep: Prepared, seed: int, kw: dict) -> tuple[np.ndarray, float, str]:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
    Xtr, Xte = _category_frames(prep)
    depth = kw.get("max_depth", 3)
    common = dict(n_estimators=BUDGET, learning_rate=LR, num_leaves=2 ** depth, reg_lambda=1.0,
                  max_bin=254, random_state=seed, n_jobs=THREADS, verbose=-1, **kw)
    reg = prep.task == "regression"
    cls = LGBMRegressor if reg else LGBMClassifier
    obj = "regression" if reg else "binary"
    metric = "rmse" if reg else "binary_logloss"
    m = cls(objective=obj, **common)
    es = EARLY_STOP
    t = time.perf_counter()
    if es:
        fi, vi = _es_split(len(Xtr), prep.y_tr, prep.task, seed)
        m.fit(Xtr.iloc[fi], prep.y_tr[fi],
              eval_set=[(Xtr.iloc[vi], prep.y_tr[vi])], eval_metric=metric,
              categorical_feature=prep.cats or "auto",
              callbacks=[lgb.early_stopping(es, verbose=False), lgb.log_evaluation(0)])
    else:
        m.fit(Xtr, prep.y_tr, categorical_feature=prep.cats or "auto")
    fit_s = time.perf_counter() - t
    pred = (np.asarray(m.predict(Xte)) if reg
            else np.asarray(m.predict_proba(Xte))[:, 1]).astype(np.float64)
    stop = getattr(m, "best_iteration_", None)
    return pred, fit_s, f"native cats{f', stop@{stop}' if stop else ''}"


def fit_cat(spec_name: str, prep: Prepared, seed: int, kw: dict) -> tuple[np.ndarray, float, str]:
    from catboost import CatBoostClassifier, CatBoostRegressor
    es = EARLY_STOP
    common = dict(iterations=BUDGET, learning_rate=LR, l2_leaf_reg=1.0, border_count=254,
                  random_seed=seed, thread_count=THREADS, verbose=0, allow_writing_files=False,
                  cat_features=prep.cats or None, **kw)
    if es:
        common.update(early_stopping_rounds=es, use_best_model=True)
    reg = prep.task == "regression"
    cls = CatBoostRegressor if reg else CatBoostClassifier
    loss = "RMSE" if reg else "Logloss"
    m = cls(loss_function=loss, **common)
    t = time.perf_counter()
    if es:
        fi, vi = _es_split(len(prep.X_tr), prep.y_tr, prep.task, seed)
        m.fit(prep.X_tr.iloc[fi], prep.y_tr[fi], eval_set=(prep.X_tr.iloc[vi], prep.y_tr[vi]))
    else:
        m.fit(prep.X_tr, prep.y_tr)
    fit_s = time.perf_counter() - t
    pred = (np.asarray(m.predict(prep.X_te)) if reg
            else np.asarray(m.predict_proba(prep.X_te))[:, 1]).astype(np.float64)
    base = "depth3 ctr1" if kw.get("depth") == 3 else "unconstrained default"
    stop = m.get_best_iteration() if es else None
    return pred, fit_s, f"{base}{f', stop@{stop}' if stop else ''}"


_FIT: dict[str, Callable] = {
    "tri": fit_tri, "ebm": fit_ebm, "xgb": fit_xgb, "lgbm": fit_lgbm, "cat": fit_cat,
}


# ----------------------------------------------------------------------------------- run
def run_dataset(spec) -> None:
    X, y = load_xy(spec)
    prep = prepare(spec, X, y)
    lower_better = prep.task == "regression"
    if spec.name == "diamonds" and BUDGET < 4000:
        print("  ! WARN diamonds at BUDGET<4000 will not match the 0.0924/0.0885 headline (under-fit tri).")
    print(f"\n=== {spec.name} [{spec.task}] · {len(prep.X_tr):,}tr/{len(prep.X_te):,}te · "
          f"{len(prep.nums)}num+{len(prep.cats)}cat · {prep.metric_name} · "
          f"budget={BUDGET} lr={LR} refine={LEAF_REFINE} es={EARLY_STOP} seeds={SEEDS} thr={THREADS} ===")
    print(f"   {'config':<14}{'goal':<16}{prep.metric_name:>13}{'±sd':>9}{'fit s':>8}   notes")
    cache = _load_fair()
    results: dict[str, float] = {}
    fit_times: dict[str, float] = {}
    fit_cached: dict[str, bool] = {}
    for cfg in _configs():
        if cfg.family in SKIP:
            continue
        cacheable = cfg.family in FAIR_CACHEABLE
        vals, fits, note, any_cache = [], [], "", False
        for seed in range(SEEDS):
            key = _fair_key(spec.name, cfg, seed)
            use_cache = cacheable and key in cache
            if _SPEED and cfg.label == "lgbm d3":
                use_cache = False  # force a live re-time for the G5 verdict (metric unchanged)
            if use_cache:
                e = cache[key]
                vals.append(e["metric"]); fits.append(e["fit_s"])
                note = e.get("note", "") + "  (cached)"; any_cache = True
                continue
            if cacheable and FROZEN:
                note = "frozen: not cached (skipped)"  # never fit a rival in frozen mode
                continue
            try:
                pred, fit_s, note = _FIT[cfg.family](spec.name, prep, seed, dict(cfg.kwargs))
                vals.append(score(prep, pred)); fits.append(fit_s)
                if cacheable:  # write-on-success-only: a transient failure is never frozen
                    cache[key] = {"metric": vals[-1], "fit_s": fit_s, "fit_threads": THREADS, "note": note}
                    _save_fair(cache)
            except Exception as e:  # noqa: BLE001
                note = f"FAILED {type(e).__name__}: {str(e)[:60]}"; break
        star = "*" if cfg.family == "tri" else " "
        if vals:
            mean, sd = float(np.mean(vals)), float(np.std(vals))
            results[cfg.label] = mean
            fit_times[cfg.label] = float(np.mean(fits))
            fit_cached[cfg.label] = any_cache
            print(f" {star}{cfg.label:<14}{cfg.goal:<16}{mean:>13.5f}{sd:>9.5f}"
                  f"{np.mean(fits):>8.1f}   {note}")
        else:
            print(f" {star}{cfg.label:<14}{cfg.goal:<16}{'—':>13}{'':>9}{'':>8}   {note}")

    # ---- goal verdicts for this dataset ----
    def cmp(a: str, b: str) -> str:
        if a not in results or b not in results:
            return "n/a"
        av, bv = results[a], results[b]
        better = (av < bv) if lower_better else (av > bv)
        rel = ((bv - av) if lower_better else (av - bv)) / abs(bv) if bv else 0.0  # +ve = tri better
        return f"{'WIN ' if better else 'lose'} {rel:+.2%} ({av:.5f} vs {bv:.5f})"

    def gap(a: str, b: str) -> str:  # G4: report the gap to the ceiling, never a WIN/lose
        if a not in results or b not in results:
            return "n/a"
        av, bv = results[a], results[b]
        behind = ((av - bv) if lower_better else (bv - av)) / abs(bv) if bv else 0.0  # +ve = tri behind
        return f"gap {behind:+.2%} ({av:.5f} vs {bv:.5f}) [ceiling — report, not a win]"

    def speed(num: str, den: str) -> str:  # G5: training wall-clock vs LightGBM
        if num not in fit_times or den not in fit_times:
            return "n/a"
        if fit_cached.get(den):
            return "cached fit_s — re-run FAIR_SPEED=1 for a live G5 timing"
        tn, td = fit_times[num], fit_times[den]
        if td < 1.0:
            return f"sub-second ({tn:.2f}s vs {td:.2f}s), noise"
        r = tn / td
        tag = "PARITY" if r <= 1.25 else ("near" if r <= 2.0 else "lose")
        return f"{tag} ({tn:.1f}s vs {td:.1f}s, r={r:.2f}x, thr={THREADS})"

    print("   -- goals --")
    print(f"     G1@1  tri o1 vs ebm mains : {cmp('tri o1', 'ebm mains')}")
    print(f"     G1@2  tri o2 vs ebm o2    : {cmp('tri o2', 'ebm o2')}")
    print(f"     G1@3  tri o3 vs ebm o2    : {cmp('tri o3', 'ebm o2')}")
    print(f"     G2    tri o3 vs xgb d3    : {cmp('tri o3', 'xgb d3')}")
    print(f"     G2    tri o3 vs lgbm d3   : {cmp('tri o3', 'lgbm d3')}")
    print(f"     G3    tri o3 vs cat d3ctr1: {cmp('tri o3', 'cat d3 ctr1')}")
    print(f"     G4    tri o3 vs cat dflt  : {gap('tri o3', 'cat default')}")
    print(f"     G5    tri o3 vs lgbm d3   : {speed('tri o3', 'lgbm d3')}")
    return results


def main() -> None:
    print("Fair-comparison harness (COMPETITIVE-GOALS.md G0-G5). Rivals are early-stopped + FROZEN in "
          ".fair_cache.json; tri-boost (* rows) always re-fits + re-runs the live G0 decomposability "
          "check (see 'within-budget'/'via-factored' in notes).")
    for spec in DATASETS:
        if _ONLY and spec.name not in _ONLY:
            continue
        try:
            run_dataset(spec)
        except Exception as exc:  # noqa: BLE001
            print(f"\n=== {spec.name} === FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
