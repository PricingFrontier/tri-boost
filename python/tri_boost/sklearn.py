"""sklearn-compatible estimators for tri-boost."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._tri_boost import _Booster, _Model

__all__ = ["PrecisionWarning", "TriBoostRegressor", "TriBoostClassifier"]


class PrecisionWarning(UserWarning):
    """Input was copied to float32 before entering the Rust core."""


def _feature_names_from_x(x: Any) -> list[str] | None:
    columns = getattr(x, "columns", None)
    if columns is None:
        return None
    return [str(c) for c in columns]


def _as_float32_2d(x: Any, *, warn: bool) -> np.ndarray:
    arr0 = np.asarray(x)
    if arr0.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got ndim={arr0.ndim}")
    if warn and arr0.dtype != np.float32:
        warnings.warn(
            "tri-boost converts input features to float32 before fitting/scoring",
            PrecisionWarning,
            stacklevel=3,
        )
    return np.asarray(arr0, dtype=np.float32, order="C")


def _as_float32_1d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32, order="C")
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def _n_columns(x: Any) -> int:
    shape = getattr(x, "shape", None)
    if shape is None:
        shape = np.asarray(x).shape
    if len(shape) != 2:
        raise ValueError(f"X must be 2-dimensional, got ndim={len(shape)}")
    return int(shape[1])


def _column_as_str_list(x: Any, j: int) -> list[str]:
    if hasattr(x, "iloc"):
        return [str(v) for v in x.iloc[:, j].to_numpy()]
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got ndim={arr.ndim}")
    return [str(v) for v in arr[:, j].tolist()]


def _numeric_subset(x: Any, idx: list[int]) -> Any:
    if hasattr(x, "iloc"):
        return x.iloc[:, idx]
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got ndim={arr.ndim}")
    return arr[:, idx]


class _BaseTriBoost(BaseEstimator):  # type: ignore[misc]  # sklearn is untyped (no py.typed)
    def __init__(
        self,
        n_trees: int = 1000,
        learning_rate: float = 0.05,
        lambda_: float = 1.0,
        l1_leaf: float = 0.0,
        min_split_gain: float = 0.0,
        max_delta_step: float | None = None,
        max_bin: int = 254,
        objective: str = "squared_error",
        tweedie_rho: float = 1.5,
        min_data_in_leaf: int = 0,
        min_sum_hessian_in_leaf: float = 0.0,
        min_weight_sum_in_leaf: float = 0.0,
        path_smooth: float = 0.0,
        subsample: float | None = None,
        colsample_bytree: float = 1.0,
        learning_rate_decay: float = 0.0,
        validation_fraction: float | None = None,
        early_stopping_rounds: int = 50,
        leaf_refine_steps: int = 0,
        leaf_refine_backtracks: int = 4,
        mvs_min_rows: int = 1,
        hist_precision: str | None = None,
        n_bags: int = 0,
        ridge_refit_l2: float | None = None,
        ridge_refit_max_iter: int = 5,
        nesterov: bool = False,
        dart_drop_rate: float | None = None,
        random_strength: float = 0.0,
        reanchor: bool | None = None,
        max_interaction_order: int = 3,
        seed: int = 0,
        n_jobs: int | None = None,
        monotone_constraints: Any = None,
        categorical_features: Any = None,
        cat_smooth: float | None = None,
        cat_target: str | None = None,
        cat_leakage: str | None = None,
        cat_n_perms: int = 1,
        cat_k: int = 5,
        cat_min_data_per_group: float = 10.0,
    ) -> None:
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.l1_leaf = l1_leaf
        self.min_split_gain = min_split_gain
        self.max_delta_step = max_delta_step
        self.max_bin = max_bin
        self.objective = objective
        self.tweedie_rho = tweedie_rho
        self.min_data_in_leaf = min_data_in_leaf
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.min_weight_sum_in_leaf = min_weight_sum_in_leaf
        self.path_smooth = path_smooth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.learning_rate_decay = learning_rate_decay
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.leaf_refine_steps = leaf_refine_steps
        self.leaf_refine_backtracks = leaf_refine_backtracks
        self.mvs_min_rows = mvs_min_rows
        self.hist_precision = hist_precision
        self.n_bags = n_bags
        self.ridge_refit_l2 = ridge_refit_l2
        self.ridge_refit_max_iter = ridge_refit_max_iter
        self.nesterov = nesterov
        self.dart_drop_rate = dart_drop_rate
        self.random_strength = random_strength
        self.reanchor = reanchor
        self.max_interaction_order = max_interaction_order
        self.seed = seed
        self.n_jobs = n_jobs
        self.monotone_constraints = monotone_constraints
        self.categorical_features = categorical_features
        self.cat_smooth = cat_smooth
        self.cat_target = cat_target
        self.cat_leakage = cat_leakage
        self.cat_n_perms = cat_n_perms
        self.cat_k = cat_k
        self.cat_min_data_per_group = cat_min_data_per_group

    def set_params(self, **params: Any) -> "_BaseTriBoost":
        result: _BaseTriBoost = super().set_params(**params)
        for name in (
            "_model",
            "_precision_warning_emitted_",
            "_cat_indices_",
            "n_features_in_",
            "feature_names_in_",
            "classes_",
        ):
            if hasattr(self, name):
                delattr(self, name)
        return result

    def _resolve_n_jobs(self) -> int | None:
        """Resolve n_jobs per the sklearn convention into a positive thread count.

        ``None`` keeps the native default (all cores); ``-1`` means all cores, and a
        negative ``n`` means ``cpu_count + 1 + n`` (so ``-1`` is all, ``-2`` all-but-one).
        Negative values must be translated here because the native layer takes an
        unsigned count and would otherwise raise ``OverflowError``.
        """
        n = self.n_jobs
        if n is None:
            return None
        n = int(n)
        if n < 0:
            import os

            cores = os.cpu_count() or 1
            n = max(1, cores + 1 + n)
        return n

    def _resolve_monotone(
        self, n_features: int, feature_names: list[str] | None
    ) -> list[int] | None:
        """Resolve ``monotone_constraints`` into a positional sign vector (-1/0/+1).

        Accepts a length-``n_features`` sequence (positional) or a dict keyed by feature
        name (matched against ``feature_names`` or the canonical ``f{i}``) or integer
        index. Returns ``None`` when no constraint is active.
        """
        mc = self.monotone_constraints
        if mc is None:
            return None
        signs = [0] * n_features
        if isinstance(mc, dict):
            name_to_idx = (
                {name: i for i, name in enumerate(feature_names)}
                if feature_names is not None
                else {}
            )
            for key, val in mc.items():
                if isinstance(key, str):
                    if key in name_to_idx:
                        idx = name_to_idx[key]
                    elif key.startswith("f") and key[1:].isdigit():
                        idx = int(key[1:])
                    else:
                        raise ValueError(f"unknown monotone feature {key!r}")
                else:
                    idx = int(key)
                if not 0 <= idx < n_features:
                    raise ValueError(f"monotone feature index {idx} out of range")
                signs[idx] = int(val)
        else:
            seq = list(mc)
            if len(seq) != n_features:
                raise ValueError(
                    f"monotone_constraints length {len(seq)} != n_features {n_features}"
                )
            signs = [int(v) for v in seq]
        for s in signs:
            if s not in (-1, 0, 1):
                raise ValueError(f"monotone sign {s} must be -1, 0, or 1")
        return signs if any(signs) else None

    def _resolve_categorical(
        self, n_features: int, feature_names: list[str] | None
    ) -> list[int]:
        cats = self.categorical_features
        if cats is None:
            return []
        if isinstance(cats, (str, int, np.integer)):
            seq: list[Any] = [cats]
        else:
            seq = list(cats)
        if all(isinstance(v, (bool, np.bool_)) for v in seq):
            if len(seq) != n_features:
                raise ValueError(
                    f"categorical_features mask length {len(seq)} != n_features {n_features}"
                )
            return [i for i, flag in enumerate(seq) if bool(flag)]

        name_to_idx = (
            {name: i for i, name in enumerate(feature_names)}
            if feature_names is not None
            else {}
        )
        idxs: list[int] = []
        for key in seq:
            if isinstance(key, str):
                if key in name_to_idx:
                    idx = name_to_idx[key]
                elif key.startswith("f") and key[1:].isdigit():
                    idx = int(key[1:])
                else:
                    raise ValueError(f"unknown categorical feature {key!r}")
            else:
                idx = int(key)
            if not 0 <= idx < n_features:
                raise ValueError(f"categorical feature index {idx} out of range")
            idxs.append(idx)
        return sorted(set(idxs))

    def _split_columns(
        self, X: Any, cat_idx: list[int], feature_names: list[str] | None
    ) -> tuple[np.ndarray, list[list[str]] | None, list[str] | None]:
        if not cat_idx:
            return self._as_float32_2d_once(X), None, feature_names

        n_features = _n_columns(X)
        cat_set = set(cat_idx)
        numeric_idx = [i for i in range(n_features) if i not in cat_set]
        if not numeric_idx:
            raise ValueError("tri-boost native categoricals require at least one numeric feature")

        numeric_x = self._as_float32_2d_once(_numeric_subset(X, numeric_idx))
        cat_x = [_column_as_str_list(X, j) for j in cat_idx]
        for pos, col in zip(cat_idx, cat_x):
            if len(col) != numeric_x.shape[0]:
                raise ValueError(
                    f"categorical feature {pos} has {len(col)} rows but X has {numeric_x.shape[0]}"
                )
        axis_names = None
        if feature_names is not None:
            axis_names = [feature_names[i] for i in numeric_idx] + [
                feature_names[j] for j in cat_idx
            ]
        return numeric_x, cat_x, axis_names

    def _serve_design(self, X: Any) -> tuple[np.ndarray, list[list[str]] | None]:
        n_features = _n_columns(X)
        expected = int(getattr(self, "n_features_in_"))
        if n_features != expected:
            raise ValueError(f"X has {n_features} features but model expects {expected}")

        cat_idx = getattr(self, "_cat_indices_", None)
        if cat_idx is None:
            cat_idx = self._resolve_categorical(n_features, _feature_names_from_x(X))
            self._cat_indices_ = cat_idx
        numeric_x, cat_x, _ = self._split_columns(X, list(cat_idx), _feature_names_from_x(X))
        return numeric_x, cat_x

    def _new_booster(self) -> _Booster:
        # Reanchor (exact 1-D intercept re-solve) defaults ON for log-link objectives
        # (Gamma/Tweedie gain; Poisson is a no-op), removing post-shrinkage aggregate bias.
        # `None` = link-aware default; an explicit bool always wins.
        reanchor = self.reanchor
        if reanchor is None:
            obj = str(self.objective).replace("-", "_").lower()
            reanchor = obj in {"poisson", "gamma", "tweedie"}
        return _Booster(
            n_trees=int(self.n_trees),
            learning_rate=float(self.learning_rate),
            lambda_=float(self.lambda_),
            l1_leaf=float(self.l1_leaf),
            min_split_gain=float(self.min_split_gain),
            max_delta_step=self.max_delta_step,
            max_bin=int(self.max_bin),
            objective=self.objective,
            tweedie_rho=float(self.tweedie_rho),
            min_data_in_leaf=int(self.min_data_in_leaf),
            min_sum_hessian_in_leaf=float(self.min_sum_hessian_in_leaf),
            min_weight_sum_in_leaf=float(self.min_weight_sum_in_leaf),
            path_smooth=float(self.path_smooth),
            subsample=None if self.subsample is None else float(self.subsample),
            colsample_bytree=float(self.colsample_bytree),
            learning_rate_decay=float(self.learning_rate_decay),
            validation_fraction=(
                None if self.validation_fraction is None else float(self.validation_fraction)
            ),
            early_stopping_rounds=int(self.early_stopping_rounds),
            leaf_refine_steps=int(self.leaf_refine_steps),
            leaf_refine_backtracks=int(self.leaf_refine_backtracks),
            mvs_min_rows=int(self.mvs_min_rows),
            hist_precision=self.hist_precision,
            n_bags=int(self.n_bags),
            ridge_refit_l2=None if self.ridge_refit_l2 is None else float(self.ridge_refit_l2),
            ridge_refit_max_iter=int(self.ridge_refit_max_iter),
            nesterov=bool(self.nesterov),
            dart_drop_rate=None if self.dart_drop_rate is None else float(self.dart_drop_rate),
            random_strength=float(self.random_strength),
            reanchor=bool(reanchor),
            max_interaction_order=int(self.max_interaction_order),
            cat_smooth=None if self.cat_smooth is None else float(self.cat_smooth),
            cat_target=self.cat_target,
            cat_leakage=self.cat_leakage,
            cat_n_perms=int(self.cat_n_perms),
            cat_k=int(self.cat_k),
            cat_min_data_per_group=float(self.cat_min_data_per_group),
            seed=int(self.seed),
            n_jobs=self._resolve_n_jobs(),
        )

    def _fit_model(
        self,
        x: Any,
        y: Any,
        *,
        sample_weight: Any | None,
        exposure: Any | None,
        class_labels: list[str] | None,
    ) -> _Model:
        feature_names = _feature_names_from_x(x)
        n_features = _n_columns(x)
        if feature_names is not None and len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length {len(feature_names)} != n_features {n_features}"
            )
        cat_idx = self._resolve_categorical(n_features, feature_names)
        if cat_idx and self.monotone_constraints is not None:
            raise ValueError(
                "monotone_constraints with native categorical features are not yet supported"
            )
        x32, cat_x, axis_names = self._split_columns(x, cat_idx, feature_names)
        y32 = _as_float32_1d(y, "y")
        if x32.shape[0] != y32.shape[0]:
            raise ValueError(f"X has {x32.shape[0]} rows but y has {y32.shape[0]}")
        weight32 = None
        if sample_weight is not None:
            weight32 = _as_float32_1d(sample_weight, "sample_weight")
            if weight32.shape[0] != y32.shape[0]:
                raise ValueError(
                    f"sample_weight has {weight32.shape[0]} rows but y has {y32.shape[0]}"
                )
        exposure32 = None
        if exposure is not None:
            exposure32 = _as_float32_1d(exposure, "exposure")
            if exposure32.shape[0] != y32.shape[0]:
                raise ValueError(
                    f"exposure has {exposure32.shape[0]} rows but y has {y32.shape[0]}"
                )
        monotone = None if cat_idx else self._resolve_monotone(n_features, feature_names)
        model = self._new_booster().fit(
            x32,
            y32,
            weight=weight32,
            exposure=exposure32,
            feature_names=axis_names,
            class_labels=class_labels,
            monotone=monotone,
            cat_x=cat_x,
        )
        self._model = model
        self._cat_indices_ = cat_idx
        self.n_features_in_ = n_features
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        return model

    def _as_float32_2d_once(self, x: Any) -> np.ndarray:
        arr0 = np.asarray(x)
        warn = arr0.dtype != np.float32 and not getattr(
            self, "_precision_warning_emitted_", False
        )
        out = _as_float32_2d(x, warn=warn)
        if warn:
            self._precision_warning_emitted_ = True
        return out

    def _attach_model(self, model: _Model) -> None:
        self._model = model
        self.n_features_in_ = model.n_features
        names = model.feature_names
        if names:
            self.feature_names_in_ = np.asarray(names, dtype=object)


class TriBoostRegressor(RegressorMixin, _BaseTriBoost):  # type: ignore[misc]
    """Exact depth-3 oblivious boosting regressor."""

    @classmethod
    def from_bytes(cls, data: bytes) -> "TriBoostRegressor":
        est = cls()
        est._attach_model(_Model.from_bytes(data))
        return est

    @classmethod
    def from_json(cls, data: str) -> "TriBoostRegressor":
        est = cls()
        est._attach_model(_Model.from_json(data))
        return est

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any | None = None,
        exposure: Any | None = None,
    ) -> "TriBoostRegressor":
        self._fit_model(
            X,
            y,
            sample_weight=sample_weight,
            exposure=exposure,
            class_labels=None,
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "_model")
        x32, cat_x = self._serve_design(X)
        return self._model.predict(x32, cat_x=cat_x)

    def predict_raw(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "_model")
        x32, cat_x = self._serve_design(X)
        return self._model.predict_raw(x32, cat_x=cat_x)

    def to_bytes(self) -> bytes:
        check_is_fitted(self, "_model")
        return self._model.to_bytes()

    def to_json(self) -> str:
        check_is_fitted(self, "_model")
        return self._model.to_json()

    def tables(
        self,
        X: Any,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        basis_json: str | None = None,
        overflow: str | None = None,
    ) -> str:
        check_is_fitted(self, "_model")
        x32, cat_x = self._serve_design(X)
        return self._model.tables(
            x32,
            ref_measure=ref_measure,
            laplace=float(laplace),
            basis_json=basis_json,
            cat_x=cat_x,
            overflow=overflow,
        )


class TriBoostClassifier(ClassifierMixin, _BaseTriBoost):  # type: ignore[misc]
    """Binary exact depth-3 oblivious boosting classifier."""

    @classmethod
    def from_bytes(cls, data: bytes) -> "TriBoostClassifier":
        est = cls()
        est._attach_classifier_model(_Model.from_bytes(data))
        return est

    @classmethod
    def from_json(cls, data: str) -> "TriBoostClassifier":
        est = cls()
        est._attach_classifier_model(_Model.from_json(data))
        return est

    def __init__(
        self,
        n_trees: int = 1000,
        learning_rate: float = 0.05,
        lambda_: float = 1.0,
        l1_leaf: float = 0.0,
        min_split_gain: float = 0.0,
        max_delta_step: float | None = None,
        max_bin: int = 254,
        objective: str = "logistic",
        tweedie_rho: float = 1.5,
        min_data_in_leaf: int = 0,
        min_sum_hessian_in_leaf: float = 0.0,
        min_weight_sum_in_leaf: float = 0.0,
        path_smooth: float = 0.0,
        subsample: float | None = None,
        colsample_bytree: float = 1.0,
        learning_rate_decay: float = 0.0,
        validation_fraction: float | None = None,
        early_stopping_rounds: int = 50,
        leaf_refine_steps: int = 0,
        leaf_refine_backtracks: int = 4,
        mvs_min_rows: int = 1,
        hist_precision: str | None = None,
        n_bags: int = 0,
        ridge_refit_l2: float | None = None,
        ridge_refit_max_iter: int = 5,
        nesterov: bool = False,
        dart_drop_rate: float | None = None,
        random_strength: float = 0.0,
        reanchor: bool | None = None,
        max_interaction_order: int = 3,
        seed: int = 0,
        n_jobs: int | None = None,
        monotone_constraints: Any = None,
        categorical_features: Any = None,
        cat_smooth: float | None = None,
        cat_target: str | None = None,
        cat_leakage: str | None = None,
        cat_n_perms: int = 1,
        cat_k: int = 5,
        cat_min_data_per_group: float = 10.0,
    ) -> None:
        super().__init__(
            n_trees=n_trees,
            learning_rate=learning_rate,
            lambda_=lambda_,
            l1_leaf=l1_leaf,
            min_split_gain=min_split_gain,
            max_delta_step=max_delta_step,
            max_bin=max_bin,
            objective=objective,
            tweedie_rho=tweedie_rho,
            min_data_in_leaf=min_data_in_leaf,
            min_sum_hessian_in_leaf=min_sum_hessian_in_leaf,
            min_weight_sum_in_leaf=min_weight_sum_in_leaf,
            path_smooth=path_smooth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            learning_rate_decay=learning_rate_decay,
            validation_fraction=validation_fraction,
            early_stopping_rounds=early_stopping_rounds,
            leaf_refine_steps=leaf_refine_steps,
            leaf_refine_backtracks=leaf_refine_backtracks,
            mvs_min_rows=mvs_min_rows,
            hist_precision=hist_precision,
            n_bags=n_bags,
            ridge_refit_l2=ridge_refit_l2,
            ridge_refit_max_iter=ridge_refit_max_iter,
            nesterov=nesterov,
            dart_drop_rate=dart_drop_rate,
            random_strength=random_strength,
            reanchor=reanchor,
            max_interaction_order=max_interaction_order,
            seed=seed,
            n_jobs=n_jobs,
            monotone_constraints=monotone_constraints,
            categorical_features=categorical_features,
            cat_smooth=cat_smooth,
            cat_target=cat_target,
            cat_leakage=cat_leakage,
            cat_n_perms=cat_n_perms,
            cat_k=cat_k,
            cat_min_data_per_group=cat_min_data_per_group,
        )

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any | None = None,
        exposure: Any | None = None,
    ) -> "TriBoostClassifier":
        if str(self.objective).replace("-", "_").lower() != "logistic":
            raise ValueError("TriBoostClassifier currently requires objective='logistic'")
        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        if classes.shape[0] != 2:
            raise ValueError("TriBoostClassifier supports exactly two classes")
        y01 = (y_arr == classes[1]).astype(np.float32, copy=False)
        self.classes_ = classes
        self._fit_model(
            X,
            y01,
            sample_weight=sample_weight,
            exposure=exposure,
            class_labels=[str(c) for c in classes],
        )
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "_model")
        x32, cat_x = self._serve_design(X)
        return self._model.predict_proba(x32, cat_x=cat_x)

    def decision_function(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "_model")
        x32, cat_x = self._serve_design(X)
        return self._model.predict_raw(x32, cat_x=cat_x)

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] >= 0.5).astype(np.intp)]

    def to_bytes(self) -> bytes:
        check_is_fitted(self, "_model")
        return self._model.to_bytes()

    def to_json(self) -> str:
        check_is_fitted(self, "_model")
        return self._model.to_json()

    def tables(
        self,
        X: Any,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        basis_json: str | None = None,
        overflow: str | None = None,
    ) -> str:
        check_is_fitted(self, "_model")
        x32, cat_x = self._serve_design(X)
        return self._model.tables(
            x32,
            ref_measure=ref_measure,
            laplace=float(laplace),
            basis_json=basis_json,
            cat_x=cat_x,
            overflow=overflow,
        )

    def _attach_classifier_model(self, model: _Model) -> None:
        labels = model.class_labels
        if labels is None or len(labels) != 2:
            raise ValueError("serialized classifier model must carry exactly two class labels")
        self.classes_ = np.asarray(labels, dtype=object)
        self._attach_model(model)
