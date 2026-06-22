//! PyO3 bindings for tri-boost (spec §12). Thin shell in Phase 0; the
//! sklearn-compatible estimators and zero-copy numpy interop land with §12.
//!
//! `#![allow(unsafe_code)]` is required here (and is the single justified,
//! encapsulated exception to the core's `forbid`): the pyo3 procedural macros
//! expand to `unsafe`. The core crate carries `#![forbid(unsafe_code)]`; this
//! binding does not, which is *why* `forbid` lives as a core crate attribute and
//! not in the workspace lint table (spec §02.3 reconciliation).
#![allow(unsafe_code)]

use pyo3::prelude::*;

/// The compiled extension module `tri_boost._tri_boost` (§02.7). Empty in Phase 0.
#[pymodule]
fn _tri_boost(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m;
    Ok(())
}
