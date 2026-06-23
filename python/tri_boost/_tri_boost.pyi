from __future__ import annotations

from typing import Sequence

import numpy as np


class TriBoostError(Exception): ...
class InvariantError(TriBoostError): ...
class ExactnessError(TriBoostError): ...
class SerializationError(TriBoostError): ...
class InternalError(TriBoostError): ...


class _Booster:
    def __init__(
        self,
        n_trees: int = 1000,
        learning_rate: float = 0.05,
        lambda_: float = 1.0,
        min_split_gain: float = 0.0,
        max_delta_step: float | None = None,
        max_bin: int = 254,
        objective: str | None = None,
        tweedie_rho: float = 1.5,
        seed: int = 0,
        n_jobs: int | None = None,
    ) -> None: ...

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray | None = None,
        exposure: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
        class_labels: Sequence[str] | None = None,
    ) -> _Model: ...


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

    def predict(self, x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray: ...

    def predict_raw(self, x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray: ...

    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...

    def explain(
        self,
        x: np.ndarray,
        ref_measure: str | None = None,
        laplace: float = 1.0,
    ) -> _TableBank: ...

    def tables(
        self,
        x: np.ndarray,
        ref_measure: str | None = None,
        laplace: float = 1.0,
        basis_json: str | None = None,
    ) -> str: ...

    def to_json(self) -> str: ...

    def to_bytes(self) -> bytes: ...


class _TableBank:
    @property
    def f0(self) -> float: ...

    @property
    def n_tables(self) -> int: ...

    def score_cells(self, cells: Sequence[int]) -> float: ...

    def sobol(self) -> list[tuple[list[int], float]]: ...
