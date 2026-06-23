"""tri-boost: exact depth-3 oblivious boosting with fANOVA tables."""

from __future__ import annotations

from ._tri_boost import (
    _Booster,
    _Model,
    _TableBank,
    ExactnessError,
    InternalError,
    InvariantError,
    SerializationError,
    TriBoostError,
)

__all__ = [
    "TriBoostClassifier",
    "TriBoostRegressor",
    "PrecisionWarning",
    "_Booster",
    "_Model",
    "_TableBank",
    "TriBoostError",
    "InvariantError",
    "ExactnessError",
    "SerializationError",
    "InternalError",
]
__version__ = "0.1.0"


def __getattr__(name: str) -> type:
    if name in {"TriBoostClassifier", "TriBoostRegressor", "PrecisionWarning"}:
        from .sklearn import PrecisionWarning, TriBoostClassifier, TriBoostRegressor

        return {
            "TriBoostClassifier": TriBoostClassifier,
            "TriBoostRegressor": TriBoostRegressor,
            "PrecisionWarning": PrecisionWarning,
        }[name]
    raise AttributeError(name)
