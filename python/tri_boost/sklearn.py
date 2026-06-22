"""sklearn-compatible estimators for tri-boost (spec §12).

Phase-0 placeholder. ``TriBoostRegressor`` and ``TriBoostClassifier`` — thin
``BaseEstimator`` wrappers over the Rust ``Booster`` with feature-name / class-label
plumbing and ``predict_proba`` — land with the §12 binding. Kept as a module so the
import path and the public surface are fixed from the start.
"""

from __future__ import annotations

__all__: list[str] = []
