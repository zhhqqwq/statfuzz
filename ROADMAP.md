# StatFuzz Roadmap

## v0.1 — Minimal statistical stress-test engine

Goal: establish a small, auditable statistical core.

- [x] `DataGenerator` protocol
- [x] Normal DGP
- [x] shifted LogNormal DGP
- [x] Student-t DGP
- [x] two-component Normal mixture DGP
- [x] explicit Welch t-test implementation
- [x] empirical Type-I error metric
- [x] Monte Carlo standard error
- [x] deterministic seeds
- [x] readable `StressTestResult`
- [x] tests against SciPy and known null behavior
- [x] GitHub Actions CI

## v0.2 — Search spaces

Goal: move from one manually chosen DGP to a reproducible failure-region search.

- [x] parameter-space abstraction
- [x] grid search
- [x] random search
- [x] tabular search results
- [ ] multiple-comparison-aware reporting of search findings

## v0.3 — Counterexample discovery

Goal: expose a high-level API such as `find_counterexample(...)`.

- [x] high-level `find_counterexample(...)` workflow

- [x] objective functions for deviation from nominal properties
- [x] search and validation simulation budgets
- [x] reproducible best-found counterexample
- [x] hold-out re-simulation to reduce search overfitting

## v0.4 — Counterexample shrinking

Goal: simplify a discovered failure case while preserving the failure criterion.

- [x] parameter simplification strategy
- [x] sample-size shrinking
- [ ] distribution-family simplification
- [x] reproducible shrink trace

## v0.5 — Failure maps and reports

- [ ] 2D parameter maps
- [ ] uncertainty overlays
- [ ] HTML report export
- [ ] machine-readable JSON report

## v0.6 — StatCI

- [ ] pytest-style statistical assertions
- [ ] GitHub Actions summary output
- [ ] badges / machine-readable status
