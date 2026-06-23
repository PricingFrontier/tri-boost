# Release Gate Verdict

The v1.5 release gate binds exactness and predictive performance into one
publication decision. A release candidate is not publishable unless both hard
clauses pass:

- Exactness hard gate: every fitted tri-boost model in the suite has
  `ExactnessMode::Exact` and passes the five invariant checks plus the feature
  budget check.
- Beat-EBM hard gate: tri-boost beats EBM/GA2M on median held-out deviance for
  each objective represented in the benchmark suite.

The unconstrained GBDT comparison is reported, not release-blocking. A large gap
triggers review because depth-3 exact decomposition is an intentional model-class
ceiling, not a promise of parity on every dense higher-order target.

## Workflow

The release workflow is manual and requires a benchmark artifact path:

```text
cargo run -p xtask -- release-verdict --input path/to/benchmark.json --output target/release-verdict.json
```

The workflow also runs the full Rust matrix, wheel/sdist builds with clean-env
smoke, `cargo publish -p tri-boost-core --dry-run`, `cargo-semver-checks`, and the
internal M6 preflight. It never publishes by itself.

## Artifact Shape

The external benchmark artifact contains a `datasets` array and an `external`
proof section:

```json
{
  "datasets": [
    {
      "fixture": "pricing-small",
      "objective": "poisson",
      "tri_boost": { "deviance": 0.91, "exact": true },
      "ebm": { "deviance": 0.97 },
      "gbdt_best": { "deviance": 0.89 }
    }
  ],
  "external": {
    "rival_adapters": {
      "results": [
        { "name": "ebm", "deviance": 0.97, "lift": 1.12, "ordered_gini": 0.31 },
        { "name": "ga2m", "deviance": 0.96, "lift": 1.12, "ordered_gini": 0.31 },
        { "name": "xgboost", "deviance": 0.89, "lift": 1.15, "ordered_gini": 0.35 },
        { "name": "lightgbm", "deviance": 0.90, "lift": 1.15, "ordered_gini": 0.34 },
        { "name": "catboost", "deviance": 0.90, "lift": 1.14, "ordered_gini": 0.34 }
      ]
    },
    "treeshap_oracle": {
      "passed": true,
      "max_abs_error": 0.0000001,
      "tolerance": 0.000001,
      "fixtures": 5
    }
  }
}
```

By default, the rival proof must include finite metrics for `ebm`, `ga2m`,
`xgboost`, `lightgbm`, and `catboost`. The TreeSHAP oracle proof must state that
it passed on at least one fixture and that `max_abs_error <= tolerance`.

## Output

`release-verdict` writes a structured JSON decision:

- `status`: `pass` only when all hard gates pass.
- `hard_gates.exactness`: release-blocking exactness verdict.
- `hard_gates.beat_ebm_median_deviance`: release-blocking median deviance verdict.
- `hard_gates.rival_adapters`: release-blocking external adapter smoke proof.
- `hard_gates.treeshap_oracle`: release-blocking external attribution-oracle proof.
- `reported_checks.gbdt_gap`: non-blocking gap to the best reported unconstrained
  GBDT.

The local `release-preflight` artifact is intentionally not a substitute for this
verdict. It proves the internal levers and exactness smoke locally, then marks
external rival and TreeSHAP checks as not run.
