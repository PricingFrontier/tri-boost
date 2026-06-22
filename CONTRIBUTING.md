# Contributing to tri-boost

tri-boost is a depth-3 oblivious GBM that is **exactly decomposable** into ≤3rd-order
fANOVA tables. Two properties are non-negotiable and are enforced as build-blocking
gates, never as review etiquette: **predictiveness** and **lossless explainability**
(the five I2 invariants + the I1 feature budget). This document encodes how work is
done so that quality is a byproduct of the process.

## Definition of Done

> **A task is done when its gates are green — not when the code is written.**

Every PR description states (a) the spec **§-reference** it implements and (b) the
named gates it leaves green. The gates (run by `.github/workflows/ci.yml`):

| Gate | Command |
|------|---------|
| **Fmt** | `cargo fmt --all --check` |
| **Clippy** | `cargo clippy --all-targets --all-features -- -D warnings` (incl. the no-panic deny set) |
| **Test** | `cargo test -p tri-boost-core --all-features` **and** `--no-default-features` |
| **Doctests** | `cargo test -p tri-boost-core --doc` (+ `deny(missing_docs)` as a compile gate) |
| **Deny** | `cargo deny check` (licenses, advisories, bans, sources) |
| **MSRV** | build + test on pinned **1.74** |
| **NoPyo3** | `pyo3`/`numpy` absent from the `tri-boost-core` dep graph |
| **Wasm** | `cargo build -p tri-boost-core --target wasm32-unknown-unknown` |
| **Determinism** | byte-equal model across `n_threads ∈ {1,2,8}` (`tests/determinism.rs`) |
| **Invariants** | the five I2 checks + I1 budget over fixtures (`tests/invariants_gate.rs`) |
| **Grep-gates** | `cargo run -p xtask -- check-all` |

Run the whole set locally before pushing with `cargo run -p xtask -- check-all`
plus the cargo commands above (or just open the PR and let CI report).

## Workflow

- **Small, reviewable PRs on branches.** Trunk (`main`) always stays green. A PR that
  reds a gate does not merge — the gate is the reviewer of last resort.
- **Gates before features.** A new capability lands behind its gate: the invariant
  check, the determinism assertion, or the type contract comes first, so the feature
  physically cannot regress what is already proven.
- **Vertical slices.** Prefer a thin end-to-end *runnable-and-gated* slice over a pile
  of untested modules.

## The no-panic policy

Library code never panics. The clippy deny set forbids `unwrap`, `expect`, `panic!`,
`unreachable!`, and `indexing_slicing` everywhere except the **unwrap-allowed set**:

> **`{ tests, benches, xtask }`** — test harnesses, Criterion benches, and the
> dev-only `xtask/` crate. None of these ship in the wheel or in
> `tri-boost-core`/`tri-boost-py`.

Everywhere else, surface failure through the single [`PbError`] enum
(`Result<T, PbError>`), never `Box<dyn Error>`. Integer overflow **traps** in every
profile (`overflow-checks = true`), so the only correct code is code that provably
cannot overflow.

### The `// JUSTIFIED:` convention

A perf-critical inner loop may use a proven-unchecked index only in a small scoped
function carrying both:

1. `#[allow(clippy::indexing_slicing)]` (or scoped `clippy::arithmetic_side_effects`), **and**
2. a `// JUSTIFIED:` comment proving the index in-bounds (e.g.
   "`idx ∈ 0..8` because it is built from three `bool` bits"), **plus** a boundary
   unit test exercising the extreme indices.

An `#[allow]` of these lints without a `// JUSTIFIED:` proof fails the
`xtask check-justified` grep-gate. Prefer `.get(..).ok_or(PbError::..)?` over a
justified `#[allow]` wherever the branch is not hot.

### Serialized state

Anything that is serialized must use **fixed-width** index/count fields (`u32`, never
`usize` — the wasm32 guard) and **deterministic-order** containers (`BTreeMap`, never
`HashMap`). The `xtask check-no-usize-serialized` / `check-no-hashmap-serialized`
gates enforce both.

## The exactness firewall

Any operation that cannot preserve exact decomposability (a nonlinear calibration
warp, a continuous-TS axis, linear leaves, a >3-order base margin) must flip the model
to `ExactnessMode::Approximate { reason }` and refuse an `Exact` table export. If a
change could bend I1/I2, it is gated behind the firewall and flagged in its owning
spec section — never slipped in silently. "If the five checks ever disagree, there is
no product."

[`PbError`]: crates/tri-boost-core/src/error.rs
