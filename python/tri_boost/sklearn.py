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


class _BaseTriBoost(BaseEstimator):
    def __init__(
        self,
        n_trees: int = 1000,
        learning_rate: float = 0.05,
        lambda_: float = 1.0,
        min_split_gain: float = 0.0,
        max_delta_step: float | None = None,
        max_bin: int = 254,
        objective: str = "squared_error",
        tweedie_rho: float = 1.5,
        seed: int = 0,
        n_jobs: int | None = None,
    ) -> None:
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.min_split_gain = min_split_gain
        self.max_delta_step = max_delta_step
        self.max_bin = max_bin
        self.objective = objective
        self.tweedie_rho = tweedie_rho
        self.seed = seed
        self.n_jobs = n_jobs

    def set_params(self, **params: Any):
        result = super().set_params(**params)
        for name in (
            "_model",
            "_precision_warning_emitted_",
            "n_features_in_",
            "feature_names_in_",
            "classes_",
        ):
            if hasattr(self, name):
                delattr(self, name)
        return result

    def _new_booster(self) -> _Booster:
        return _Booster(
            n_trees=int(self.n_trees),
            learning_rate=float(self.learning_rate),
            lambda_=float(self.lambda_),
            min_split_gain=float(self.min_split_gain),
            max_delta_step=self.max_delta_step,
            max_bin=int(self.max_bin),
            objective=self.objective,
            tweedie_rho=float(self.tweedie_rho),
            seed=int(self.seed),
            n_jobs=self.n_jobs,
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
        x32 = self._as_float32_2d_once(x)
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
        model = self._new_booster().fit(
            x32,
            y32,
            weight=weight32,
            exposure=exposure32,
            feature_names=feature_names,
            class_labels=class_labels,
        )
        self._model = model
        self.n_features_in_ = x32.shape[1]
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


class TriBoostRegressor(RegressorMixin, _BaseTriBoost):
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
    ):
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
        x32 = self._as_float32_2d_once(X)
        return self._model.predict(x32)

    def predict_raw(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "_model")
        x32 = self._as_float32_2d_once(X)
        return self._model.predict_raw(x32)

    def to_bytes(self) -> bytes:
        check_is_fitted(self, "_model")
        return self._model.to_bytes()

    def to_json(self) -> str:
        check_is_fitted(self, "_model")
        return self._model.to_json()


class TriBoostClassifier(ClassifierMixin, _BaseTriBoost):
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
        min_split_gain: float = 0.0,
        max_delta_step: float | None = None,
        max_bin: int = 254,
        objective: str = "logistic",
        tweedie_rho: float = 1.5,
        seed: int = 0,
        n_jobs: int | None = None,
    ) -> None:
        super().__init__(
            n_trees=n_trees,
            learning_rate=learning_rate,
            lambda_=lambda_,
            min_split_gain=min_split_gain,
            max_delta_step=max_delta_step,
            max_bin=max_bin,
            objective=objective,
            tweedie_rho=tweedie_rho,
            seed=seed,
            n_jobs=n_jobs,
        )

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any | None = None,
        exposure: Any | None = None,
    ):
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
        x32 = self._as_float32_2d_once(X)
        return self._model.predict_proba(x32)

    def decision_function(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "_model")
        x32 = self._as_float32_2d_once(X)
        return self._model.predict_raw(x32)

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] >= 0.5).astype(np.intp)]

    def to_bytes(self) -> bytes:
        check_is_fitted(self, "_model")
        return self._model.to_bytes()

    def to_json(self) -> str:
        check_is_fitted(self, "_model")
        return self._model.to_json()

    def _attach_classifier_model(self, model: _Model) -> None:
        labels = model.class_labels
        if labels is None or len(labels) != 2:
            raise ValueError("serialized classifier model must carry exactly two class labels")
        self.classes_ = np.asarray(labels, dtype=object)
        self._attach_model(model)
