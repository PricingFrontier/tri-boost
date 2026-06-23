from __future__ import annotations

from typing import Sequence, final

import numpy as np

__all__ = [
    "TriBoostError",
    "InvariantError",
    "ExactnessError",
    "SerializationError",
    "InternalError",
    "_Booster",
    "_Model",
    "_TableBank",
]


class TriBoostError(Exception): ...
class InvariantError(TriBoostError): ...
class ExactnessError(TriBoostError): ...
class SerializationError(TriBoostError): ...
class InternalError(TriBoostError): ...


@final
class _Booster:
    def __new__(
        cls,
        n_trees: int = 1000,
        learning_rate: float = 0.05,
        lambda_: float = 1.0,
        l1_leaf: float = 0.0,
        min_split_gain: float = 0.0,
        max_delta_step: float | None = None,
        max_bin: int = 254,
        objective: str | None = None,
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
        reanchor: bool = False,
        max_interaction_order: int = 3,
        cat_smooth: float | None = None,
        cat_target: str | None = None,
        cat_leakage: str | None = None,
        cat_n_perms: int = 1,
        cat_k: int = 5,
        cat_min_data_per_group: float = 10.0,
        seed: int = 0,
        n_jobs: int | None = None,
    ) -> _Booster: ...

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        class_labels: Sequence[str] | None = None,
        monotone: Sequence[int] | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
    ) -> _Model: ...


@final
class _Model:
    @staticmethod
    def from_json(s: str) -> _Model: ...

    @staticmethod
    def from_bytes(bytes: bytes) -> _Model: ...

    @property
    def n_features(self) -> int: ...

    @property
    def feature_names(self) -> list[str]: ...

    @property
    def class_labels(self) -> list[str] | None: ...

    def predict(
        self,
        x: np.ndarray,
        out: np.ndarray | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
    ) -> np.ndarray: ...

    def predict_raw(
        self,
        x: np.ndarray,
        out: np.ndarray | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
    ) -> np.ndarray: ...

    def predict_proba(
        self,
        x: np.ndarray,
        cat_x: Sequence[Sequence[str]] | None = None,
    ) -> np.ndarray: ...

    def explain(
        self,
        x: np.ndarray,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        cat_x: Sequence[Sequence[str]] | None = None,
    ) -> _TableBank: ...

    def tables(
        self,
        x: np.ndarray,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        basis_json: str | None = None,
        cat_x: Sequence[Sequence[str]] | None = None,
    ) -> str: ...

    def to_json(self) -> str: ...

    def to_bytes(self) -> bytes: ...


@final
class _TableBank:
    @property
    def f0(self) -> float: ...

    @property
    def n_tables(self) -> int: ...

    def score_cells(self, cells: Sequence[int]) -> float: ...

    def sobol(self) -> list[tuple[list[int], float]]: ...
