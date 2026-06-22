"""tri-boost — a depth-3 oblivious GBM, exactly decomposable into fANOVA tables.

Phase-0 skeleton (spec §12). The sklearn-compatible estimators
(``TriBoostRegressor`` / ``TriBoostClassifier``) and the zero-copy numpy/Arrow
interop are wired here once the §12 binding lands; for now this package only
re-exports the compiled extension module so the import path is stable.
"""

from __future__ import annotations

# The compiled Rust extension (built by maturin from crates/tri-boost-py). It is an
# empty module in Phase 0; the estimator classes attach to it in §12.
from . import _tri_boost  # noqa: F401

__all__: list[str] = []
__version__ = "0.1.0"
