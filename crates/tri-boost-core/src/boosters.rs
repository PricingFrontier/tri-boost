//! Predictiveness boosters (spec §09).
//!
//! This module owns exactness-preserving booster levers such as ensemble averaging,
//! refit, Nesterov/AGBM, DART, and re-anchoring. Phase 5 starts from an empty public
//! surface here; new booster types are added only with their invariant gates and
//! inert-default oracles.
